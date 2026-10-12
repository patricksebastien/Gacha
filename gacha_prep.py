#!/usr/bin/env python3
"""
gacha_prep.py — carve one-shot samples out of a recording, named by note.

A take of single plucked notes (a guitar, a bell, a voice, anything with a
pitch) becomes a sample set: every note is found, cropped to its own
attack and decay, brought to one loudness and written as
samples/<set>/<set>_<note>_<source>_<start>s.wav, so a G2 is a G2 in the
Samples tab. Sources are the audio files and the videos the app knows
(a video's sound is decoded with PyAV), or any file picked by hand.

Detection is plain numpy: an RMS envelope gated relative to the file's
loudest moment; a note starts when the envelope rises over the gate and
ends after it has stayed under for a while, a second attack inside a
ringing note (a sudden rise over the decaying level) splits it. Pitch
comes from YIN on the signal right after each note's peak; the window
below the table lets you correct a note the detector got wrong, drag the
edges of a cut, untick the ones you do not want and hear each cut as it
will be written (cropped, faded, normalized). Loudness is gacha_bank's:
integrated LUFS with clipped peaks like the default set, or peak.

    python3 gacha_prep.py                 # the window on its own
    python3 gacha_prep.py samples/guitaropeng/ZOOM0139_MS90.wav

The Samples tab's "Prepare samples…" button opens the same window inside the app;
an export refreshes the tab, tick the new set and it is in the pool.
"""

import sys
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from PySide6.QtCore import Qt, QSettings, QTimer, Signal, QRectF
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget,
    QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from gacha_bank import (NORMALIZE_MODES as BANK_MODES, SAMPLES_DIR, TARGET_LUFS,
                        TARGET_PEAK_DB, integrated_lufs, normalize as bank_normalize,
                        safe_stem)

BASE = Path(__file__).parent
VIDEOS_DIR = BASE / "videos"
AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".mp3", ".w64", ".ogg")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
HOP = 256                       # envelope resolution: 5.3 ms at 48 kHz
BLOCK = 64                      # waveform overview resolution
NOTES = "C C# D D# E F F# G G# A A# B".split()
# gacha_bank's lufs / peak / none plus "lufs-noclip": the loudness target
# unless reaching it would push a peak over the peak target, then that
# ceiling wins. A plucked note is all attack: -14 LUFS over a short decay
# asks for +15..+25 dB and plain lufs clips the pick
NORMALIZE_MODES = BANK_MODES + ("lufs-noclip",)
NOTE_CHOICES = ["?"] + [f"{n}{o}" for o in range(0, 8) for n in NOTES]

PREP_STYLE = """
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: rgb(36, 15, 12); }
QTableWidget {
    background: rgba(28, 11, 9, 200); border: 1px solid rgba(255,255,255,35);
    border-radius: 8px; gridline-color: rgba(255,255,255,25);
}
QTableWidget::item:selected { background: rgba(212, 48, 30, 150); color: rgb(242, 233, 229); }
QHeaderView::section {
    background: rgba(28, 11, 9, 230); color: rgba(242, 233, 229, 167);
    border: none; padding: 2px 6px;
}
QTableCornerButton::section { background: rgba(28, 11, 9, 230); border: none; }
QGroupBox { margin-top: 12px; padding-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; }
"""

try:
    import sounddevice as sd
except Exception:                                    # pragma: no cover
    sd = None


# ---------------------------------------------------------------- analysis

@dataclass
class Params:
    gate_db: float = 45.0       # a note is louder than the file's peak minus this
    quiet_s: float = 0.2        # ...and over once the envelope stays under for this long
    min_len: float = 0.25       # shorter cuts are noise (clicks, breaths)
    max_len: float = 10.0       # longer cuts are capped here; 0 = never
    min_peak_db: float = -35.0  # quieter cuts are noise (dBFS)
    retrigger_db: float = 15.0  # a rise this steep inside a note starts a new one; 0 = off
    pre_roll_ms: float = 5.0    # kept before the detected attack
    release_ms: float = 60.0    # kept after the envelope went under the gate
    fade_in_ms: float = 2.0
    fade_out_ms: float = 20.0
    fmin: float = 55.0          # pitch search range (Hz)
    fmax: float = 1500.0
    normalize: str = "lufs"
    lufs: float = TARGET_LUFS
    peak_db: float = TARGET_PEAK_DB


@dataclass
class Segment:
    start: float                # seconds in the source, pre-roll included
    end: float
    f0: float = float("nan")
    conf: float = 0.0           # 1 - YIN aperiodicity, 0 when no pitch
    peak_db: float = -120.0
    note: str = "?"             # detected or set by hand
    keep: bool = True

    @property
    def length(self):
        return self.end - self.start


def load_audio(path):
    """(frames, channels) float32 and the sample rate; a video gives its
    audio stream."""
    p = Path(path)
    if p.suffix.lower() in VIDEO_EXTS:
        return _decode_video_audio(p)
    data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return data, sr


def _decode_video_audio(p):
    import av
    chunks = []
    with av.open(str(p)) as ctr:
        if not ctr.streams.audio:
            raise ValueError("no audio stream")
        stream = ctr.streams.audio[0]
        rate = stream.rate
        rs = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        for frame in ctr.decode(stream):
            for out in rs.resample(frame):
                chunks.append(out.to_ndarray().T)
        for out in rs.resample(None):
            chunks.append(out.to_ndarray().T)
    if not chunks:
        raise ValueError("empty audio stream")
    return np.ascontiguousarray(np.concatenate(chunks).astype("float32")), rate


def envelope_db(mono, sr, hop=HOP):
    n = len(mono) // hop
    if n == 0:
        return np.full(1, -120.0)
    rms = np.sqrt(np.mean(np.square(mono[:n * hop]).reshape(n, hop), axis=1))
    return 20 * np.log10(rms + 1e-7)


def pitch_yin(y, sr, fmin, fmax, threshold=0.15):
    """f0 (Hz) and aperiodicity (0 = perfectly periodic) at the start of
    `y`: YIN (de Cheveigné & Kawahara 2002), the difference function from
    the autocorrelation, cumulative-mean normalisation, first dip under
    the threshold, parabolic refinement. (nan, 1) when too short."""
    tmin, tmax = max(2, int(sr / fmax)), int(sr / fmin)
    n = min(len(y), int(0.1 * sr) + 2 * tmax)
    if n < 2 * tmax:
        return float("nan"), 1.0
    y = y[:n].astype("float64")
    y = y - y.mean()
    ac = fftconvolve(y, y[::-1])[n - 1:n + tmax + 1]            # lags 0..tmax+1
    e = np.concatenate([[0.0], np.cumsum(y * y)])              # e[k] = sum y[:k]^2
    t = np.arange(tmax + 2)
    d = e[n - t] + (e[n] - e[t]) - 2 * ac                        # difference function
    cmnd = np.ones_like(d)
    cs = np.cumsum(d[1:])
    cmnd[1:] = d[1:] * np.arange(1, len(d)) / np.maximum(cs, 1e-12)
    seg = cmnd[tmin:tmax]
    below = np.where(seg < threshold)[0]
    if len(below):
        k = below[0]
        while k + 1 < len(seg) and seg[k + 1] < seg[k]:        # down to the dip's bottom
            k += 1
    else:
        k = int(np.argmin(seg))
    t0 = tmin + k
    a, b, c = cmnd[t0 - 1], cmnd[t0], cmnd[t0 + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom > 1e-12 else 0.0
    return sr / (t0 + shift), float(min(1.0, max(0.0, b)))


def hz_to_note(f):
    """Note name and the offset from it in cents."""
    if not f == f or f <= 0:
        return "?", 0.0
    m = 69 + 12 * np.log2(f / 440.0)
    n = int(round(m))
    if not 12 <= n < 108:
        return "?", 0.0
    return f"{NOTES[n % 12]}{n // 12 - 1}", (m - n) * 100


def segment_pitch(mono, sr, start, end, p):
    """Pitch of one cut: YIN right after the peak, then a little later
    (the attack's noise is over), the cleaner reading wins."""
    y = mono[int(start * sr):int(end * sr)]
    if len(y) == 0:
        return float("nan"), 0.0
    pk = int(np.argmax(np.abs(y)))
    res = []
    for off in (0.02, 0.08, 0.2):
        o = pk + int(off * sr)
        if o >= len(y):
            break
        f, ap = pitch_yin(y[o:], sr, p.fmin, p.fmax)
        if f == f:
            res.append((ap, f))
    if not res:
        return float("nan"), 0.0
    ap, f = min(res)
    return f, 1.0 - ap


def detect(data, sr, p):
    """Every note in the recording as Segments, pitched."""
    mono = data.mean(axis=1) if data.ndim == 2 else data
    db = envelope_db(mono, sr)
    fps = sr / HOP
    thr = max(float(db.max()) - p.gate_db, -80.0)
    on = db > thr
    quiet = max(1, int(p.quiet_s * fps))
    hold = max(1, int(0.1 * fps))                # no retrigger inside the attack
    look = max(1, int(0.05 * fps))               # a rise is measured over this
    total = len(mono) / sr
    segs, i, n = [], 0, len(db)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j:j + quiet].any():
            j += 1
        end, released = j, True
        if p.retrigger_db > 0:
            k = i + hold
            while k < j:
                lo_at = max(i, k - look) + int(np.argmin(db[max(i, k - look):k]))
                if db[k] - db[lo_at] > p.retrigger_db and db[k] > thr + 6:
                    end, released = lo_at, False
                    break
                k += 1
        start_s = max(0.0, i / fps - p.pre_roll_ms / 1000)
        end_s = end / fps + (p.release_ms / 1000 if released else 0.0)
        end_s = min(total, end_s)
        if p.max_len > 0:
            end_s = min(end_s, start_s + p.max_len)
        i = max(end, i + 1)
        if end_s - start_s < p.min_len:
            continue
        clip = mono[int(start_s * sr):int(end_s * sr)]
        peak = float(np.abs(clip).max()) if len(clip) else 0.0
        peak_db = 20 * np.log10(peak) if peak > 0 else -120.0
        if peak_db < p.min_peak_db:
            continue
        f0, conf = segment_pitch(mono, sr, start_s, end_s, p)
        note, _ = hz_to_note(f0)
        segs.append(Segment(start_s, end_s, f0, conf, peak_db, note))
    return segs


def normalize(data, sr, mode, lufs, peak_db):
    if mode != "lufs-noclip":
        return bank_normalize(data, sr, mode, lufs, peak_db)
    gain_db = lufs - integrated_lufs(data, sr)
    peak = float(np.abs(data).max())
    ceiling = peak_db - 20 * np.log10(peak) if peak > 0 else gain_db
    note = f"{min(gain_db, ceiling):+5.1f} dB"
    if ceiling < gain_db:
        note += f" (peak-limited, {ceiling - gain_db:+.1f} LU under target)"
    return data * (10 ** (min(gain_db, ceiling) / 20)), note


def render(data, sr, seg, p):
    """The cut as it is written: cropped, faded, normalized. Returns
    (audio, note for the log)."""
    a = max(0, int(seg.start * sr))
    b = min(len(data), int(seg.end * sr))
    clip = np.array(data[a:b], dtype="float32")
    if len(clip) == 0:
        return clip, "empty"
    fi = min(int(p.fade_in_ms / 1000 * sr), len(clip) // 2)
    fo = min(int(p.fade_out_ms / 1000 * sr), len(clip) // 2)
    if fi > 0:
        clip[:fi] *= np.linspace(0, 1, fi, dtype="float32")[:, None]
    if fo > 0:
        clip[-fo:] *= np.linspace(1, 0, fo, dtype="float32")[:, None]
    return normalize(clip, sr, p.normalize, p.lufs, p.peak_db)


def cut_name(prefix, seg, source):
    note = seg.note if seg.note != "?" else "x"
    return f"{prefix}_{note}_{safe_stem(Path(source))}_{seg.start:06.2f}s.wav"


def export(data, sr, segs, p, source, set_name, prefix=None, out_root=SAMPLES_DIR):
    """Write every kept cut into out_root/set_name/. Returns the paths."""
    out_dir = Path(out_root) / set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or set_name
    written = []
    for seg in segs:
        if not seg.keep:
            continue
        clip, note = render(data, sr, seg, p)
        if len(clip) == 0:
            continue
        out = out_dir / cut_name(prefix, seg, source)
        sf.write(str(out), clip, sr, subtype="PCM_24")
        written.append(out)
        print(f"  {out.name:<48} {len(clip) / sr:5.2f}s  {note}")
    return written


# ---------------------------------------------------------------- waveform

class WaveView(QWidget):
    """The source's waveform with the cuts drawn over it. Click a cut to
    select it, drag its edge to move it, double-click to hear it; wheel
    zooms around the pointer, dragging empty space pans; a click on empty
    space places the cursor."""
    selected = Signal(int)
    edited = Signal(int)
    play_requested = Signal(int)
    cursor_moved = Signal(float)
    EDGE_PX = 6

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.sr = 1
        self.total = 0.0
        self.bmax = self.bmin = np.zeros(0, dtype="float32")
        self.segs = []
        self.current = -1
        self.v0, self.vlen = 0.0, 1.0
        self.playhead = None
        self.cursor = 0.0
        self._drag = None               # ("pan", x, v0) | ("start"|"end", index)

    def set_audio(self, data, sr):
        mono = data.mean(axis=1) if data.ndim == 2 else data
        n = len(mono) // BLOCK
        blocks = mono[:n * BLOCK].reshape(n, BLOCK) if n else np.zeros((1, BLOCK), "float32")
        self.bmax, self.bmin = blocks.max(axis=1), blocks.min(axis=1)
        self.sr, self.total = sr, len(mono) / sr
        self.v0, self.vlen = 0.0, max(self.total, 0.01)
        self.playhead, self.cursor, self.current = None, 0.0, -1
        self.update()

    def set_segments(self, segs, current=-1):
        self.segs, self.current = segs, current
        self.update()

    def set_current(self, i):
        self.current = i
        if 0 <= i < len(self.segs):
            s = self.segs[i]
            if s.start < self.v0 or s.end > self.v0 + self.vlen:     # bring into view
                self.v0 = max(0.0, min(s.start - 0.1 * self.vlen, self.total - self.vlen))
        self.update()

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    # -- geometry
    def _x(self, t):
        return (t - self.v0) / self.vlen * self.width()

    def _t(self, x):
        return self.v0 + x / max(1, self.width()) * self.vlen

    def _hit(self, x):
        """("start"|"end"|"in", index) or None for the pointer at x."""
        best = None
        for i, s in enumerate(self.segs):
            for which, t in (("start", s.start), ("end", s.end)):
                d = abs(self._x(t) - x)
                if d <= self.EDGE_PX and (best is None or d < best[0]):
                    best = (d, which, i)
        if best:
            return best[1], best[2]
        for i, s in enumerate(self.segs):
            if self._x(s.start) <= x <= self._x(s.end):
                return "in", i
        return None

    # -- painting
    def paintEvent(self, _event):
        w, h = self.width(), self.height()
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor(28, 11, 9, 200))
        if self.total <= 0 or w < 2:
            qp.end()
            return
        mid = h / 2
        # one min/max per pixel column from the block overview
        bps = self.sr / BLOCK
        edges = np.clip(((self.v0 + np.arange(w + 1) / w * self.vlen) * bps).astype(int),
                        0, len(self.bmax))
        qp.setPen(QPen(QColor(120, 160, 230, 200), 1))
        for x in range(w):
            a, b = edges[x], max(edges[x] + 1, edges[x + 1])
            if a >= len(self.bmax):
                break
            top, bot = float(self.bmax[a:b].max()), float(self.bmin[a:b].min())
            qp.drawLine(x, int(mid - top * mid * 0.95), x, int(mid - bot * mid * 0.95))
        # the cuts
        for i, s in enumerate(self.segs):
            x0, x1 = self._x(s.start), self._x(s.end)
            if x1 < 0 or x0 > w:
                continue
            if s.keep:
                fill = QColor(255, 120, 60, 70 if i == self.current else 40)
                line = QColor(255, 150, 90, 255 if i == self.current else 160)
            else:
                fill = QColor(160, 160, 160, 40 if i == self.current else 20)
                line = QColor(200, 200, 200, 200 if i == self.current else 90)
            qp.fillRect(QRectF(x0, 0, max(1.0, x1 - x0), h), fill)
            qp.setPen(QPen(line, 2 if i == self.current else 1))
            qp.drawLine(int(x0), 0, int(x0), h)
            qp.drawLine(int(x1), 0, int(x1), h)
            qp.setPen(QPen(line, 1))
            qp.drawText(int(x0) + 3, 14, f"{i + 1} {s.note}")
        # cursor and playhead
        qp.setPen(QPen(QColor(242, 233, 229, 120), 1, Qt.DashLine))
        xc = int(self._x(self.cursor))
        qp.drawLine(xc, 0, xc, h)
        if self.playhead is not None:
            qp.setPen(QPen(QColor(255, 60, 40, 230), 2))
            xp = int(self._x(self.playhead))
            qp.drawLine(xp, 0, xp, h)
        qp.setPen(QPen(QColor(242, 233, 229, 110), 1))
        qp.drawText(4, h - 6, f"{self.v0:.2f}s")
        txt = f"{self.v0 + self.vlen:.2f}s"
        qp.drawText(w - 8 * len(txt) - 4, h - 6, txt)
        qp.end()

    # -- mouse
    def mousePressEvent(self, ev):
        x = ev.position().x()
        hit = self._hit(x)
        if ev.button() != Qt.LeftButton:
            return
        if hit and hit[0] in ("start", "end"):
            self._drag = (hit[0], hit[1])
            self.current = hit[1]
            self.selected.emit(hit[1])
        elif hit:
            self.current = hit[1]
            self.selected.emit(hit[1])
            self._drag = ("pan", x, self.v0)
        else:
            self.cursor = self._t(x)
            self.cursor_moved.emit(self.cursor)
            self._drag = ("pan", x, self.v0)
        self.update()

    def mouseMoveEvent(self, ev):
        x = ev.position().x()
        if self._drag is None:
            hit = self._hit(x)
            self.setCursor(Qt.SizeHorCursor if hit and hit[0] != "in" else Qt.ArrowCursor)
            return
        kind = self._drag[0]
        if kind == "pan":
            _, x_from, v0 = self._drag
            self.v0 = max(0.0, min(self.total - self.vlen,
                                   v0 - (x - x_from) / self.width() * self.vlen))
        else:
            s = self.segs[self._drag[1]]
            t = max(0.0, min(self.total, self._t(x)))
            if kind == "start":
                s.start = min(t, s.end - 0.01)
            else:
                s.end = max(t, s.start + 0.01)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag and self._drag[0] != "pan":
            self.edited.emit(self._drag[1])
        self._drag = None

    def mouseDoubleClickEvent(self, ev):
        hit = self._hit(ev.position().x())
        if hit:
            self.play_requested.emit(hit[1])

    def wheelEvent(self, ev):
        if self.total <= 0:
            return
        steps = ev.angleDelta().y() / 120.0
        if ev.modifiers() & Qt.ShiftModifier:                    # pan
            self.v0 = max(0.0, min(self.total - self.vlen, self.v0 - steps * 0.2 * self.vlen))
        else:                                                    # zoom around the pointer
            t = self._t(ev.position().x())
            factor = 0.8 ** steps
            self.vlen = max(0.05, min(self.total, self.vlen * factor))
            self.v0 = max(0.0, min(self.total - self.vlen,
                                   t - ev.position().x() / self.width() * self.vlen))
        self.update()


# ---------------------------------------------------------------- window

class PrepWindow(QMainWindow):
    """The sample prep tool: sources on the left, waveform and the cuts in
    the middle, the knobs on the right. `exported` fires after a set is
    written, with its folder."""
    exported = Signal(str)
    COLS = ("keep", "start", "length", "note", "cents", "conf", "peak")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("gacha prep — cut notes into samples")
        self.resize(1400, 820)
        self.settings = QSettings("gacha", "prep")
        self.data, self.sr, self.source = None, 0, None
        self.segs = []
        self.current = -1
        self._filling = False
        self._play_t0 = self._play_end = None
        self._play_clip = None
        self._build()
        self._load_settings()
        # the app's stylesheet assumes a dark backdrop and never styles tables
        self.setStyleSheet(PREP_STYLE)

    # -- layout
    def _build(self):
        self.files = QTreeWidget()
        self.files.setHeaderHidden(True)
        self.files.itemClicked.connect(self._file_clicked)
        self._fill_files()
        open_btn = QPushButton("Open file…")
        open_btn.setObjectName("small")
        open_btn.clicked.connect(self._open_dialog)
        rescan_btn = QPushButton("Rescan")
        rescan_btn.setObjectName("small")
        rescan_btn.clicked.connect(self._fill_files)
        left = QVBoxLayout()
        left.addWidget(self.files, stretch=1)
        row = QHBoxLayout()
        row.addWidget(open_btn)
        row.addWidget(rescan_btn)
        left.addLayout(row)
        left_w = QWidget()
        left_w.setLayout(left)

        self.wave = WaveView()
        self.wave.selected.connect(self._select)
        self.wave.edited.connect(self._seg_dragged)
        self.wave.play_requested.connect(self._play_seg)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.table.itemChanged.connect(self._table_changed)
        self.table.itemDoubleClicked.connect(lambda it: self._play_seg(it.row()))
        self.title = QLabel("open a recording")
        self.title.setStyleSheet("font-weight: bold;")
        self.status = QLabel("")
        self.status.setStyleSheet("color: rgb(170, 120, 110);")
        mid = QVBoxLayout()
        mid.addWidget(self.title)
        mid.addWidget(self.wave, stretch=3)
        mid.addWidget(self.table, stretch=2)
        mid.addWidget(self.status)
        mid_w = QWidget()
        mid_w.setLayout(mid)

        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setWidget(self._build_params())
        right.setMinimumWidth(300)

        split = QSplitter()
        split.addWidget(left_w)
        split.addWidget(mid_w)
        split.addWidget(right)
        split.setSizes([260, 820, 320])
        self.setCentralWidget(split)

        self.redetect = QTimer(self)
        self.redetect.setSingleShot(True)
        self.redetect.setInterval(150)
        self.redetect.timeout.connect(self._detect)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(30)
        self.play_timer.timeout.connect(self._tick)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._play_current)
        QShortcut(QKeySequence("P"), self, self._play_from_cursor)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._stop)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self._toggle_keep)
        QShortcut(QKeySequence("K"), self, self._toggle_keep)

    def _spin(self, lo, hi, step, dec=1, suffix=""):
        s = QDoubleSpinBox(minimum=lo, maximum=hi, singleStep=step, decimals=dec)
        s.setSuffix(suffix)
        s.valueChanged.connect(lambda _v: self.redetect.start())
        return s

    def _build_params(self):
        self.w = {}
        box = QVBoxLayout()

        det = QGroupBox("Detect")
        f = QFormLayout(det)
        self.w["gate_db"] = self._spin(10, 90, 1, 0, " dB")
        self.w["gate_db"].setToolTip("A note is anything louder than the file's loudest "
                                     "moment minus this. Raise it to catch quiet notes, "
                                     "lower it when room noise becomes cuts")
        self.w["quiet_s"] = self._spin(0.02, 2, 0.05, 2, " s")
        self.w["quiet_s"].setToolTip("A note ends when the level stays under the gate this long")
        self.w["min_len"] = self._spin(0.02, 10, 0.05, 2, " s")
        self.w["max_len"] = self._spin(0, 120, 0.5, 1, " s")
        self.w["max_len"].setToolTip("Cuts are capped at this length; 0 = whole decay")
        self.w["min_peak_db"] = self._spin(-90, 0, 1, 0, " dBFS")
        self.w["min_peak_db"].setToolTip("Cuts whose peak sits under this are dropped (clicks, breaths)")
        self.w["retrigger_db"] = self._spin(0, 60, 1, 0, " dB")
        self.w["retrigger_db"].setToolTip("A rise this steep (over 50 ms) inside a ringing note "
                                          "starts a new cut: notes played before the last has "
                                          "died. 0 = off")
        self.w["pre_roll_ms"] = self._spin(0, 100, 1, 0, " ms")
        self.w["release_ms"] = self._spin(0, 2000, 10, 0, " ms")
        self.w["release_ms"].setToolTip("Kept after the level went under the gate, then the fade out")
        for key, label in (("gate_db", "gate"), ("quiet_s", "quiet for"), ("min_len", "min length"),
                           ("max_len", "max length"), ("min_peak_db", "min peak"),
                           ("retrigger_db", "retrigger"), ("pre_roll_ms", "pre-roll"),
                           ("release_ms", "release")):
            f.addRow(label, self.w[key])
        box.addWidget(det)

        pit = QGroupBox("Pitch")
        f = QFormLayout(pit)
        self.w["fmin"] = self._spin(20, 2000, 5, 0, " Hz")
        self.w["fmax"] = self._spin(100, 8000, 50, 0, " Hz")
        f.addRow("lowest", self.w["fmin"])
        f.addRow("highest", self.w["fmax"])
        box.addWidget(pit)

        cut = QGroupBox("Cut")
        f = QFormLayout(cut)
        self.w["fade_in_ms"] = self._spin(0, 500, 1, 0, " ms")
        self.w["fade_out_ms"] = self._spin(0, 5000, 5, 0, " ms")
        self.w["normalize"] = QComboBox()
        self.w["normalize"].addItems(NORMALIZE_MODES)
        self.w["normalize"].setToolTip("lufs: integrated loudness, peaks clipped, like the default "
                                       "set; lufs-noclip: same, but a cut whose attack would clip "
                                       "stays quieter instead; peak: loudest sample to the "
                                       "target; none: as recorded")
        self.w["normalize"].currentTextChanged.connect(lambda _t: self._refresh_table())
        self.w["lufs"] = self._spin(-40, 0, 0.5, 1, " LUFS")
        self.w["peak_db"] = self._spin(-20, 0, 0.1, 1, " dBFS")
        for key in ("fade_in_ms", "fade_out_ms", "lufs", "peak_db"):
            self.w[key].valueChanged.disconnect()
            self.w[key].valueChanged.connect(lambda _v: self._refresh_table())
        f.addRow("fade in", self.w["fade_in_ms"])
        f.addRow("fade out", self.w["fade_out_ms"])
        f.addRow("normalize", self.w["normalize"])
        f.addRow("target", self.w["lufs"])
        f.addRow("peak target", self.w["peak_db"])
        box.addWidget(cut)

        exp = QGroupBox("Export")
        f = QFormLayout(exp)
        self.set_name = QLineEdit("cuts")
        self.set_name.setToolTip("Becomes samples/<set>/ and the file prefix")
        self.set_name.textChanged.connect(self._update_export_label)
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._export)
        self.export_label = QLabel("")
        self.export_label.setProperty("role", "sub")
        self.export_label.setWordWrap(True)
        f.addRow("set", self.set_name)
        f.addRow(self.export_btn)
        f.addRow(self.export_label)
        box.addWidget(exp)

        keys = QLabel("space: hear the cut · double-click: same · P: play the source "
                      "from the cursor · esc: stop · K / delete: keep or drop · "
                      "wheel: zoom · shift+wheel: pan · drag a cut's edge to move it")
        keys.setProperty("role", "sub")
        keys.setWordWrap(True)
        box.addWidget(keys)
        box.addStretch(1)
        w = QWidget()
        w.setLayout(box)
        return w

    # -- settings
    def _params(self):
        p = Params()
        for fld in fields(Params):
            wdg = self.w[fld.name]
            setattr(p, fld.name, wdg.currentText() if isinstance(wdg, QComboBox) else wdg.value())
        return p

    def _load_settings(self):
        p = Params()
        for fld in fields(Params):
            v = self.settings.value(fld.name, getattr(p, fld.name))
            wdg = self.w[fld.name]
            wdg.blockSignals(True)
            if isinstance(wdg, QComboBox):
                wdg.setCurrentText(str(v))
            else:
                wdg.setValue(float(v))
            wdg.blockSignals(False)
        self.set_name.setText(self.settings.value("set_name", "cuts"))

    def closeEvent(self, ev):
        self._stop()
        for k, v in asdict(self._params()).items():
            self.settings.setValue(k, v)
        self.settings.setValue("set_name", self.set_name.text())
        super().closeEvent(ev)

    # -- sources
    def _fill_files(self):
        self.files.clear()
        for root, exts in ((SAMPLES_DIR, AUDIO_EXTS), (VIDEOS_DIR, VIDEO_EXTS)):
            top = QTreeWidgetItem([root.name])
            top.setFlags(top.flags() & ~Qt.ItemIsSelectable)
            self.files.addTopLevelItem(top)
            if not root.exists():
                continue
            folders = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
            for folder in folders:
                files = sorted(p for p in folder.iterdir()
                               if p.is_file() and p.suffix.lower() in exts)
                if not files:
                    continue
                node = top if folder == root else QTreeWidgetItem(
                    top, [folder.relative_to(root).as_posix()])
                for f in files:
                    it = QTreeWidgetItem(node, [f.name])
                    it.setData(0, Qt.UserRole, str(f))
            top.setExpanded(True)

    def _file_clicked(self, item, _col):
        path = item.data(0, Qt.UserRole)
        if path:
            self.load(path)

    def _open_dialog(self):
        exts = " ".join(f"*{e}" for e in AUDIO_EXTS + VIDEO_EXTS)
        path, _ = QFileDialog.getOpenFileName(self, "Open a recording", str(BASE),
                                              f"Audio and video ({exts})")
        if path:
            self.load(path)

    def load(self, path):
        self._stop()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data, sr = load_audio(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self.status.setText(f"cannot read {Path(path).name}: {e}")
            return
        QApplication.restoreOverrideCursor()
        self.data, self.sr, self.source = data, sr, Path(path)
        self.title.setText(f"{self.source.name}  ·  {sr} Hz  ·  {data.shape[1]} ch  ·  "
                           f"{len(data) / sr:.1f} s")
        self.wave.set_audio(data, sr)
        if self.set_name.text() in ("", "cuts"):
            self.set_name.setText(safe_stem(self.source.parent) + "_cuts"
                                  if self.source.parent != SAMPLES_DIR else "cuts")
        self._detect()

    # -- detection and the table
    def _detect(self):
        if self.data is None:
            return
        t0 = time.time()
        self.segs = detect(self.data, self.sr, self._params())
        self.current = 0 if self.segs else -1
        self.wave.set_segments(self.segs, self.current)
        self._refresh_table()
        self.status.setText(f"{len(self.segs)} cuts found in {time.time() - t0:.2f} s")
        self._update_export_label()

    def _refresh_table(self):
        self._filling = True
        self.table.setRowCount(len(self.segs))
        for r, s in enumerate(self.segs):
            self._fill_row(r, s)
        self._filling = False
        if 0 <= self.current < len(self.segs):
            self.table.selectRow(self.current)
        self._update_export_label()

    def _fill_row(self, r, s):
        _, cents = hz_to_note(s.f0)
        keep = QTableWidgetItem()
        keep.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        keep.setCheckState(Qt.Checked if s.keep else Qt.Unchecked)
        self.table.setItem(r, 0, keep)
        vals = (f"{s.start:.2f}", f"{s.length:.2f}", None, f"{cents:+.0f}" if s.f0 == s.f0 else "",
                f"{s.conf:.2f}", f"{s.peak_db:.1f}")
        for c, v in enumerate(vals, start=1):
            if v is None:
                continue
            it = QTableWidgetItem(v)
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, c, it)
        combo = self.table.cellWidget(r, 3)
        if combo is None:
            combo = QComboBox()
            combo.addItems(NOTE_CHOICES)
            combo.currentTextChanged.connect(lambda t, row=r: self._note_changed(row, t))
            self.table.setCellWidget(r, 3, combo)
        combo.blockSignals(True)
        combo.setCurrentText(s.note if s.note in NOTE_CHOICES else "?")
        combo.blockSignals(False)
        if s.conf < 0.7:
            combo.setStyleSheet("color: rgb(255, 160, 90);")   # the detector is unsure
        else:
            combo.setStyleSheet("")

    def _note_changed(self, row, text):
        if 0 <= row < len(self.segs):
            self.segs[row].note = text
            self.wave.update()

    def _table_changed(self, item):
        if self._filling or item.column() != 0:
            return
        r = item.row()
        if 0 <= r < len(self.segs):
            self.segs[r].keep = item.checkState() == Qt.Checked
            self.wave.update()
            self._update_export_label()

    def _table_selected(self):
        if self._filling:
            return
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.current = rows[0].row()
            self.wave.set_current(self.current)

    def _select(self, i):
        self.current = i
        self._filling = True
        self.table.selectRow(i)
        self._filling = False

    def _seg_dragged(self, i):
        s = self.segs[i]
        s.f0, s.conf = segment_pitch(self.data.mean(axis=1), self.sr, s.start, s.end, self._params())
        note, _ = hz_to_note(s.f0)
        s.note = note
        clip = self.data[int(s.start * self.sr):int(s.end * self.sr)]
        peak = float(np.abs(clip).max()) if len(clip) else 0.0
        s.peak_db = 20 * np.log10(peak) if peak > 0 else -120.0
        self._filling = True
        self._fill_row(i, s)
        self._filling = False

    def _toggle_keep(self):
        if 0 <= self.current < len(self.segs):
            s = self.segs[self.current]
            s.keep = not s.keep
            self.table.item(self.current, 0).setCheckState(Qt.Checked if s.keep else Qt.Unchecked)

    def _update_export_label(self, *_):
        kept = [s for s in self.segs if s.keep]
        name = self.set_name.text().strip() or "cuts"
        if not kept:
            self.export_label.setText(f"nothing to write into samples/{name}/")
            self.export_btn.setEnabled(False)
            return
        self.export_btn.setEnabled(True)
        example = cut_name(name, kept[0], self.source) if self.source else ""
        self.export_label.setText(f"{len(kept)} cuts into samples/{name}/\ne.g. {example}")

    # -- playback
    def _play(self, clip, start, end):
        if sd is None:
            self.status.setText("sounddevice is not installed: no playback (pip install sounddevice)")
            return
        self._stop()
        try:
            sd.play(clip, self.sr)
        except Exception as e:
            self.status.setText(f"playback failed: {e}")
            return
        self._play_t0, self._play_start, self._play_end = time.monotonic(), start, end
        self.play_timer.start()

    def _play_seg(self, i):
        if self.data is None or not 0 <= i < len(self.segs):
            return
        self.current = i
        self.wave.set_current(i)
        self._select(i)
        s = self.segs[i]
        clip, note = render(self.data, self.sr, s, self._params())
        self.status.setText(f"cut {i + 1}: {s.note}  {s.f0:.1f} Hz  {s.length:.2f} s  {note}")
        self._play(clip, s.start, s.end)

    def _play_current(self):
        self._play_seg(self.current)

    def _play_from_cursor(self):
        if self.data is None:
            return
        t = self.wave.cursor
        clip = self.data[int(t * self.sr):]
        self._play(clip, t, len(self.data) / self.sr)

    def _stop(self):
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        self.play_timer.stop()
        self.wave.set_playhead(None)

    def _tick(self):
        t = self._play_start + (time.monotonic() - self._play_t0)
        if t >= self._play_end:
            self._stop()
            return
        self.wave.set_playhead(t)

    # -- export
    def _export(self):
        if self.data is None:
            return
        name = self.set_name.text().strip() or "cuts"
        out_dir = SAMPLES_DIR / name
        print(f"writing into {out_dir}")
        written = export(self.data, self.sr, self.segs, self._params(), self.source, name)
        self.status.setText(f"wrote {len(written)} cuts into samples/{name}/ "
                            f"(refresh the Samples tab and tick '{name}')")
        self.exported.emit(str(out_dir))


def main():
    app = QApplication(sys.argv)
    try:
        from gacha import STYLE                 # the app's look, when importable
        app.setStyleSheet(STYLE)
    except Exception:
        pass
    win = PrepWindow()
    win.show()
    if len(sys.argv) > 1:
        win.load(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
