#!/usr/bin/env python3
"""
gacha.py — Qt front-end for the gacha_engine.py render engine.

Left side: all generation parameters in tabs (General / Drums / Sections /
Layers & FX) + live render log.
Right side: the sample library (samples/, one subfolder per set, with
checkboxes picking what a render may use), the video library (videos/,
same tree, picking the backdrop clips), past renders
(output/) and the section mixer — double-click anything to listen. New
renders start playing automatically when generation finishes.

Usage:  python3 gacha.py
"""

import bisect
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from PySide6.QtCore import (QEvent, QPoint, Qt, QRectF, QSettings, QThread,
                            QTimer, QUrl, Signal)
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QImage, QKeySequence,
                           QPainter, QRegularExpressionValidator, QShortcut)
from PySide6.QtCore import QRegularExpression
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QPlainTextEdit, QPushButton, QSlider, QSpinBox, QStyle,
    QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from gacha_gl import (BLEND_MODES, FrameHistory, GLBackdrop, gl_available,
                      shader_files)
from gacha_engine import (AUDIO_EXTS, RANDOM_START_MODES, SYNTH_STYLES, WORDS,
                          sample_name)

BASE = Path(__file__).parent
GACHA = BASE / "gacha_engine.py"
SAMPLES_DIR = BASE / "samples"
OUT_DIR = BASE / "output"
VIDEOS_DIR = BASE / "videos"
FONTS_DIR = BASE / "fonts"


def load_fonts():
    """Register every .ttf/.otf in fonts/; returns the family names."""
    families = []
    if FONTS_DIR.is_dir():
        for f in sorted(FONTS_DIR.iterdir()):
            if f.suffix.lower() in (".ttf", ".otf"):
                fid = QFontDatabase.addApplicationFont(str(f))
                if fid >= 0:
                    families += QFontDatabase.applicationFontFamilies(fid)
    return sorted(set(families)) or ["Sans Serif"]


def frame_palette(img, rng, count=12):
    """A few colours picked from the video frame: the brighter, more
    saturated ones among random samples, so words can borrow the clip."""
    if img is None or img.isNull():
        return []
    picks = []
    for _ in range(count * 4):
        c = img.pixelColor(rng.randrange(img.width()), rng.randrange(img.height()))
        picks.append((c.saturationF() * 0.6 + c.valueF() * 0.4, c))
    picks.sort(key=lambda x: -x[0])
    return [c for _, c in picks[:count] if c.valueF() > 0.25]


def make_word_cloud(size, words, families, rng, hue=None, n=None, palette=()):
    """A transparent image filled with random words in random fonts, sizes
    and opacities: the 'word cloud' the backdrop shows now and then. Some
    words take a colour from `palette` (sampled from the video), the rest
    are white."""
    img = QImage(size, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.TextAntialiasing)
    W, H = size.width(), size.height()
    if n is None:
        n = rng.randint(110, 180)                      # dense: fills the frame
    for i in range(n):
        big = i < 5                                    # a few headline words
        px = rng.randint(int(H * 0.09), int(H * 0.22)) if big \
            else rng.randint(12, int(H * 0.07))
        font = QFont(rng.choice(families))
        font.setPixelSize(px)
        p.setFont(font)
        word = rng.choice(words)
        if rng.random() < 0.5:
            word = word.upper()
        alpha = int(255 * (0.10 if big else rng.uniform(0.02, 0.075)))  # very faint
        if palette and rng.random() < 0.45:            # the clip's own colours
            col = QColor(rng.choice(palette))
            col.setAlpha(min(255, int(alpha * 1.6)))   # colour needs a bit more to read
        elif hue is None:
            col = QColor(255, 255, 255, alpha)
        else:                                          # faint tint of the root note
            col = QColor.fromHsvF(hue, rng.uniform(0.0, 0.35), 1.0, alpha / 255)
        p.setPen(col)
        tw = p.fontMetrics().horizontalAdvance(word)
        x = rng.randint(-tw // 4, max(0, W - (3 * tw) // 4))
        y = rng.randint(px, H)
        p.drawText(x, y, word)
    p.end()
    return img

DEFAULT_VIDEO = VIDEOS_DIR / "gacha.mp4"     # shipped with the repo, shown at start
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")


def video_files():
    """Backdrop candidates: every video in videos/ (any extension case, so
    camera files like .MP4 count), plus any gacha*.mp4 left next to the
    scripts for backwards compatibility."""
    files = [p for p in VIDEOS_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in VIDEO_EXTS] \
        if VIDEOS_DIR.is_dir() else []
    files += list(BASE.glob("gacha*.mp4"))
    return sorted(set(files))

STYLE = """
QWidget { color: rgba(242, 233, 229, 167); font-size: 15px;
          font-family: "Noto Sans Mono", "Ubuntu Mono", monospace; }
QGroupBox {
    background: rgba(28, 11, 9, 117); border: 1px solid rgba(255,255,255,40);
    border-radius: 10px; margin-top: 0px; padding-top: 0px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QListWidget, QTreeWidget, QPlainTextEdit {
    background: rgba(28, 11, 9, 117); border: 1px solid rgba(255,255,255,35);
    border-radius: 8px;
}
QTabWidget::pane {
    background: rgba(28, 11, 9, 90); border: 1px solid rgba(255,255,255,35);
    border-radius: 8px;
}
QTabBar::tab {
    background: rgba(28, 11, 9, 99); padding: 6px 16px;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}
QTabBar::tab:selected { background: rgba(188, 38, 26, 171); }
QPushButton {
    background: rgba(134, 28, 20, 144); border: 1px solid rgba(255,255,255,50);
    border-radius: 8px; padding: 6px 14px;
}
QPushButton:hover { background: rgba(212, 48, 30, 189); }
QPushButton:disabled { color: rgba(242,233,229,72); }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background: rgba(28, 11, 9, 117); border: 1px solid rgba(212, 48, 30, 170);
    border-radius: 6px; padding: 2px 6px;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border: 1px solid rgba(255, 96, 70, 240);
}
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled,
QLineEdit:disabled { border-color: rgba(212, 48, 30, 70); }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid rgba(212, 48, 30, 190); background: rgba(28, 11, 9, 144);
}
QCheckBox::indicator:hover { border-color: rgba(255, 96, 70, 240); }
QCheckBox::indicator:checked {
    background: rgba(212, 48, 30, 198); border-color: rgba(255, 120, 90, 240);
}
QLabel[role="sub"] { color: rgba(242,233,229,99); font-size: 13px; }
QComboBox QAbstractItemView { background: rgb(42, 18, 14); }
QLabel, QCheckBox { background: transparent; }
QSplitter::handle { background: transparent; }
"""


FX_WIDTH = 480      # frames are shrunk to this width before pixel effects
SHOW_AFTER_PX = 30      # hidden interface comes back after this much mouse travel
FX_INTERVAL = 0.025     # process at most one video frame per 25 ms (~40 fps)


def hue_matrix(turns):
    """3x3 RGB hue-rotation matrix, `turns` in 0..1 (SVG hueRotate)."""
    a = 2 * np.pi * turns
    c, s_ = np.cos(a), np.sin(a)
    return np.array([
        [0.213 + 0.787 * c - 0.213 * s_, 0.715 - 0.715 * c - 0.715 * s_,
         0.072 - 0.072 * c + 0.928 * s_],
        [0.213 - 0.213 * c + 0.143 * s_, 0.715 + 0.285 * c + 0.140 * s_,
         0.072 - 0.072 * c - 0.283 * s_],
        [0.213 - 0.213 * c - 0.787 * s_, 0.715 - 0.715 * c + 0.715 * s_,
         0.072 + 0.928 * c + 0.072 * s_],
    ], dtype="float32")


BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                   [3, 11, 1, 9], [15, 7, 13, 5]], dtype="float32") / 16.0

# every effect the Video tab exposes: key, label, default strength, tooltip
VIDEO_EFFECTS = [
    ("color", "color", 0.6, "Saturation and hue per section; intros bloom in, "
                            "breaks wash out"),
    ("tint", "tint", 0.5, "Tint toward the colour of the section's root note"),
    ("pump", "pump", 0.6, "Brightness follows the song's loudness"),
    ("flash", "flash", 0.7, "Drop FX in sync with the audio: the frame freezes "
                            "and stutters with a beat-repeat, builds to white "
                            "under a cymbal swell, holds still in a gap and "
                            "inverts on the drop"),
    ("pixel", "pixel", 0.5, "Pixelate during breaks"),
    ("bits", "bits", 0.5, "Colour depth: 1.0 = 1 bit, 0.75 = 2 bits, "
                          "0.5 = 3 bits, 0.25 = 4 bits"),
    ("dither", "dither", 0.4, "1-bit ordered (Bayer) dithering, blended in"),
    ("mono", "mono", 0.4, "Grayscale. Each section rolls its tone count: "
                          "smooth gray, or 2 (black & white at a random "
                          "threshold), 3, 4, 6 or 8 tones. Strength = how "
                          "often the hard tones win"),
    ("solar", "solarize", 0.3, "Solarize: bright areas flip negative"),
    ("edges", "edges", 0.3, "Edge detection, blended in"),
    ("lines", "scanlines", 0.4, "CRT scanlines"),
    ("grain", "grain", 0.3, "Film grain"),
    ("zoom", "zoom", 0.5, "Zoom punches in with the loudness"),
    ("trails", "trails", 0.4, "Motion trails: bright parts linger"),
    ("glitch", "glitch", 0.4, "Horizontal glitch tears, more when loud. Per "
     "section: thin sharp lines (0.4-1% of the frame), medium bands (4-14%) "
     "or anything from 4% up to 38% slabs"),
    ("rgb", "rgb shift", 0.4, "Chromatic aberration pulsing with loudness"),
    ("reverse", "reverse", 0.3, "Rewind: now and then a phrase runs a bar or "
                                "two backwards, and a cymbal swell rewinds "
                                "faster and faster into the drop. Strength = "
                                "how often a phrase gets one"),
    ("vign", "vignette", 0.5, "Dark corners"),
    ("words", "words", 0.25, "Word cloud: now and then a 4-bar phrase fills the "
                             "screen with random words in the fonts from fonts/. "
                             "Strength = how often"),
    ("kaleido", "kaleido", 0.15, "Kaleidoscope with 4, 6 or 8 mirrored "
                                 "segments, upright. Strength = how often a "
                                 "section gets it"),
]
NOTE_HUES = {n: i / 12 for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
# these follow the music directly and are not part of the random per-section look
# glitch band height ranges as fractions of the frame, one per section mode:
# thin sharp lines, medium bands, anything up to big slabs
GLITCH_MODES = [(0.004, 0.01), (0.04, 0.14), (0.04, 0.38)]

LOOK_EXEMPT = {"color", "pump", "flash", "pixel", "kaleido", "words", "reverse"}


class VideoFX:
    """Pixel effects on a shrunken RGB copy of each frame. Keeps the previous
    frame for trails and a cached vignette mask."""

    def __init__(self):
        self.prev = None
        self.rng = np.random.default_rng()
        self._vig = None
        self._kmaps = {}            # (h, w, n, angle) -> (rows, cols) lookup

    def kaleido_map(self, h, w, n, angle):
        """Nearest-neighbour lookup that folds the frame into n mirrored
        wedges around the centre, rotated by `angle`. Cached, a few kept."""
        key = (h, w, n, round(angle, 3))
        if key not in self._kmaps:
            if len(self._kmaps) > 6:
                self._kmaps.pop(next(iter(self._kmaps)))
            cy, cx = (h - 1) / 2, (w - 1) / 2
            yy, xx = np.mgrid[0:h, 0:w]
            dy, dx = yy - cy, xx - cx
            r = np.hypot(dx, dy)
            th = np.arctan2(dy, dx) + angle
            wedge = 2 * np.pi / n
            k = np.floor(th / wedge)
            t = th - k * wedge
            t = np.where(k.astype(int) % 2 == 1, wedge - t, t)
            # source wedge points along the frame's long axis for coverage
            sx = np.clip(np.rint(cx + r * np.cos(t)), 0, w - 1).astype(np.intp)
            sy = np.clip(np.rint(cy + r * np.sin(t)), 0, h - 1).astype(np.intp)
            self._kmaps[key] = (sy, sx)
        return self._kmaps[key]

    def vignette(self, h, w):
        if self._vig is None or self._vig.shape != (h, w):
            y = np.linspace(-1, 1, h)[:, None]
            x = np.linspace(-1, 1, w)[None, :]
            self._vig = np.clip((x * x + y * y * 1.3) * 0.7, 0, 1) \
                .astype("float32")
        return self._vig

    def process(self, img, st):
        """st: dict of effect parameters; returns (QImage, backing array)."""
        small = img.scaledToWidth(FX_WIDTH, Qt.FastTransformation) \
            .convertToFormat(QImage.Format_RGB888)
        w, h, bpl = small.width(), small.height(), small.bytesPerLine()
        px = np.frombuffer(small.constBits(), np.uint8, count=h * bpl) \
            .reshape(h, bpl)[:, : w * 3].reshape(h, w, 3).astype("float32")
        g = st.get
        rng = self.rng

        rpt = g("repeat")                     # (k, frac): k-th retrigger,
        if rpt:                               # frac = progress inside it
            k, frac = rpt
            z = 0.18 * frac                   # zoom in, snap back each hit
            y0, x0 = int(h * z / 2), int(w * z / 2)
            crop = px[y0:h - y0, x0:w - x0]
            rows = np.linspace(0, crop.shape[0] - 1, h).astype(int)
            cols = np.linspace(0, crop.shape[1] - 1, w).astype(int)
            px = crop[rows][:, cols]
            if k % 2:
                px = 255.0 - px
        kal = g("kaleido")                    # (segments, angle)
        if kal:
            sy, sx = self.kaleido_map(h, w, kal[0], kal[1])
            px = px[sy, sx]
        p = int(g("pixelate", 0))
        if p > 1:
            blocks = px[::p, ::p]
            px = np.repeat(np.repeat(blocks, p, axis=0), p, axis=1)[:h, :w]
        gl = g("glitch", 0.0)
        if gl > 0 and rng.random() < gl:
            hmin, hmax = GLITCH_MODES[int(g("glitch_mode", 2))]
            for _ in range(1 + int(3 * gl)):
                # band height log-uniform between hmin and hmax of the frame
                bh = max(1, int(h * np.exp(rng.uniform(np.log(hmin), np.log(hmax)))))
                y0 = int(rng.integers(0, max(1, h - bh)))
                shift = int(rng.integers(-w // 4, w // 4 + 1) * gl)
                px[y0:y0 + bh] = np.roll(px[y0:y0 + bh], shift, axis=1)
        sh = int(g("rgbshift", 0))
        if sh:
            px[..., 0] = np.roll(px[..., 0], -sh, axis=1)
            px[..., 2] = np.roll(px[..., 2], sh, axis=1)
        luma = px @ np.array([0.299, 0.587, 0.114], dtype="float32")
        e = g("edges", 0.0)
        if e > 0:
            gx = np.abs(np.diff(luma, axis=1, append=luma[:, -1:]))
            gy = np.abs(np.diff(luma, axis=0, append=luma[-1:, :]))
            edges = np.clip((gx + gy) * 3.0, 0, 255)
            px = px * (1 - e) + edges[..., None] * e
        hue = g("hue", 0.0)
        if hue:
            px = px @ hue_matrix(hue).T
        sat = g("saturation", 1.0)
        if sat != 1.0:
            luma = px @ np.array([0.299, 0.587, 0.114], dtype="float32")
            px = luma[..., None] + sat * (px - luma[..., None])
        tint = g("tint")                      # (hue 0..1, amount)
        if tint and tint[1] > 0:
            col = 255 * np.clip(np.abs(((tint[0] + np.array([0, 2 / 3, 1 / 3]))
                                        % 1.0) * 6 - 3) - 1, 0, 1)
            luma = px @ np.array([0.299, 0.587, 0.114], dtype="float32")
            px = px * (1 - tint[1]) + (luma[..., None] / 255 * col) * tint[1]
        px = px * g("brightness", 1.0)
        so = g("solarize", 0.0)
        if so > 0:
            thr = 255 * (1 - 0.6 * so)
            mask = px > thr
            px[mask] = 255 - px[mask]
        mono = g("mono")                      # (tones, threshold 0..1)
        if mono:
            tones, thr = mono
            gray = px @ np.array([0.299, 0.587, 0.114], dtype="float32")
            if tones == 2:
                gray = (gray > thr * 255) * 255.0
            elif tones > 2:
                gray = np.floor(np.clip(gray, 0, 255) / 256 * tones) \
                    * (255 / (tones - 1))
            px = np.repeat(gray[..., None], 3, axis=2)
        bits = g("bits", 0)
        if bits:
            levels = 2 ** int(bits)
            px = np.floor(px / 256 * levels) * (255 / (levels - 1))
        d = g("dither", 0.0)
        if d > 0:
            luma = px @ np.array([0.299, 0.587, 0.114], dtype="float32")
            thr = np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w] * 255
            bw = (luma > thr).astype("float32") * 255
            px = px * (1 - d) + bw[..., None] * d
        sl = g("scanlines", 0.0)
        if sl > 0:
            px[1::2] *= 1 - 0.7 * sl
        gr = g("grain", 0.0)
        if gr > 0:
            px = px + rng.normal(0, 45 * gr, size=(h, w, 1)).astype("float32")
        v = g("vignette", 0.0)
        if v > 0:
            px = px * (1 - v * self.vignette(h, w))[..., None]
        sw = g("swell", 0.0)                  # cymbal swell: build to white
        if sw > 0:
            px = px + (255.0 - px) * 0.75 * sw * sw
        if g("invert"):
            px = 255.0 - px
        tr = g("trails", 0.0)
        if tr > 0 and self.prev is not None and self.prev.shape == px.shape:
            px = np.maximum(px, self.prev * (0.55 + 0.42 * tr))
        self.prev = px if tr > 0 else None
        arr = np.ascontiguousarray(np.clip(px, 0, 255).astype("uint8"))
        return QImage(arr.data, w, h, w * 3, QImage.Format_RGB888), arr


class VideoBackdrop(QWidget):
    """Paints a muted, looping video behind its child widget, under a scrim.
    `state_fn`, when set, returns the effect state for the current frame."""

    def __init__(self, child, videos_fn=video_files):
        super().__init__()
        self.videos_fn = videos_fn
        self._image = None
        self._arr = None            # keeps the processed frame's memory alive
        self.state_fn = None
        self.fx = VideoFX()
        self._last_fx = 0.0
        self._zoom = 0.0
        self._frozen = None         # captured frame while a beat-repeat runs
        self._history = FrameHistory()   # for the reverse effect
        self._start_random = False  # seek somewhere random once media loads
        self.child = child
        self.current = None         # path of the video now looping
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(child)
        if video_files():
            self._sink = QVideoSink()
            self._sink.videoFrameChanged.connect(self._on_frame)
            self._vplayer = QMediaPlayer()      # no audio output = silent
            self._vplayer.setVideoSink(self._sink)
            self._vplayer.setLoops(QMediaPlayer.Loops.Infinite)
            if DEFAULT_VIDEO.exists():
                self.set_video(DEFAULT_VIDEO)   # always the stock clip at start
            else:
                self.pick_random()

    def set_video(self, path, random_start=False):
        player = getattr(self, "_vplayer", None)
        if player is None:
            return None
        self.current = Path(path)
        self._start_random = random_start
        player.stop()
        player.setSource(QUrl.fromLocalFile(str(self.current)))
        player.play()
        return self.current

    def pick_random(self):
        """Loop a random video from videos/ for a song: never the stock clip
        (unless it is the only one), and a different one than now if
        possible. Rescans the folder, so new files are picked up without a
        restart."""
        files = list(self.videos_fn())
        if not hasattr(self, "_vplayer") or not files:
            return None
        files = [f for f in files if f != DEFAULT_VIDEO] or files
        pool = [f for f in files if f != self.current] or files
        return self.set_video(random.choice(pool), random_start=True)

    def _on_frame(self, frame):
        if not frame.isValid():
            return
        if self._start_random:
            # first decoded frame of a new clip: the backend is ready to
            # seek now, so do not always open on the first frame
            self._start_random = False
            self.jump()
            return
        st = self.state_fn() if self.state_fn else None
        live = frame.toImage()
        self._history.push(live, time.monotonic())
        if not (st and st.get("reverse")):
            self._history.stop()
        if st and any(k != "music" for k in st):
            now = time.monotonic()
            if now - self._last_fx < FX_INTERVAL:
                return              # throttle: keep showing the last frame
            self._last_fx = now
            self._zoom = st.get("zoom", 0.0)
            if st.get("reverse"):
                src = self._history.rewind(now, float(st["reverse"])) or live
            elif st.get("repeat") or st.get("freeze"):
                # beat-repeat: hold the frame the roll started on, like the
                # audio holds its slice; a pre-drop gap holds it still
                if self._frozen is None:
                    self._frozen = frame.toImage()
                src = self._frozen
            else:
                self._frozen = None
                src = live
            try:
                img, self._arr = self.fx.process(src, st)
            except Exception:
                img = src               # never let an effect kill the backdrop
        else:
            self._zoom = 0.0
            self._frozen = None
            self.fx.prev = None
            img = live
        self._image = img
        self.update()

    def set_shader(self, path):          # generative layer: GL backdrop only
        self.shader_path = Path(path) if path else None    # remembered for the log

    def jump(self):
        """Seek the backdrop video to a random position (no-op without video)."""
        player = getattr(self, "_vplayer", None)
        if player is not None and player.duration() > 0:
            player.setPosition(random.randrange(player.duration()))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(16, 9, 7))
        img = self._image
        if img is not None and not img.isNull() and self.height() > 0:
            # crop the source so the video covers the window (center-crop)
            wr = self.width() / self.height()
            ir = img.width() / img.height()
            if ir > wr:
                sw = img.height() * wr
                src = QRectF((img.width() - sw) / 2, 0, sw, img.height())
            else:
                sh = img.width() / wr
                src = QRectF(0, (img.height() - sh) / 2, img.width(), sh)
            if self._zoom > 0:      # punch in: shrink the source rect
                z = 1 - min(0.45, self._zoom)
                cx, cy = src.center().x(), src.center().y()
                src = QRectF(cx - src.width() * z / 2, cy - src.height() * z / 2,
                             src.width() * z, src.height() * z)
            p.drawImage(QRectF(self.rect()), img, src)
        if self.child.isVisible():
            p.fillRect(self.rect(), QColor(12, 7, 5, 55))   # legibility scrim


class SeekSlider(QSlider):
    """A slider that jumps to wherever you click, instead of stepping toward
    it. Dragging still works; both paths emit sliderMoved."""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt_w = self.width()
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(),
                int(event.position().x()), opt_w)
            self.setValue(val)
            self.sliderMoved.emit(val)
        super().mousePressEvent(event)


class SetTree(QTreeWidget):
    """A folder of sets as a tree with tri-state checkboxes: every subfolder
    of `root` is a node, its files are leaves, loose files in the root form
    a node of their own. Ticking a folder ticks every file in it
    (Qt.ItemIsAutoTristate), a mixed folder shows the partial state.

    Ticks survive rescans and launches (QSettings under `key`). A file seen
    for the first time follows its folder when that folder was fully
    ticked; a brand-new folder starts unticked. On the very first launch
    the folder named `first_run_folder` is ticked, or everything when that
    is None or no such folder exists.

    Signals: `changed` after any tick, `activated(path)` on a double-click
    on a file."""
    changed = Signal()
    activated = Signal(str)

    def __init__(self, root, exts, settings, key, first_run_folder=None):
        super().__init__()
        self.root, self.exts = Path(root), tuple(exts)
        self.settings, self.key = settings, key
        self.first_run_folder = first_run_folder
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self._loading = False
        self._seen = None
        self.total = 0
        self.itemChanged.connect(self._on_changed)
        self.itemDoubleClicked.connect(self._on_double_click)

    # -- names are paths relative to the root, so two sets may share a filename
    def name(self, path):
        try:
            return Path(path).relative_to(self.root).as_posix()
        except ValueError:
            return Path(path).name

    def _files(self, folder):
        return sorted((p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in self.exts),
                      key=lambda p: p.name.lower())

    def _leaves(self):
        out = []

        def walk(item):
            for i in range(item.childCount()):
                c = item.child(i)
                if c.childCount():
                    walk(c)
                else:
                    out.append(c)
        walk(self.invisibleRootItem())
        return out

    def checked(self):
        """Absolute paths of the ticked files, in tree order."""
        return [c.data(0, Qt.UserRole) for c in self._leaves()
                if c.checkState(0) == Qt.Checked]

    def checked_names(self):
        return [self.name(p) for p in self.checked()]

    def _setting(self, name, default):
        v = self.settings.value(f"{self.key}/{name}", default)
        if isinstance(v, str):
            v = [v]
        return v

    def refresh(self):
        if self._loading:
            return
        if self._seen is not None:                    # a rescan: keep our own state
            checked, seen, first_run = set(self.checked_names()), self._seen, False
        else:                                         # first refresh: from settings
            saved = self._setting("checked", None)
            seen = set(self._setting("seen", []) or [])
            first_run = saved is None
            checked = set() if first_run else set(saved)

        self._loading = True
        self.clear()
        self.root.mkdir(exist_ok=True)
        folders = sorted((p for p in self.root.iterdir()
                          if p.is_dir() and not p.name.startswith(".")),
                         key=lambda p: p.name.lower())
        groups = [(f.name, self._files(f)) for f in folders]
        loose = self._files(self.root)
        if loose:
            groups.append(("(loose files)", loose))
        groups = [(l, f) for l, f in groups if f]
        has_first = any(l == self.first_run_folder for l, _ in groups)

        for label, files in groups:
            node = QTreeWidgetItem(self, [f"{label}  ({len(files)})"])
            node.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable
                          | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            node.setData(0, Qt.UserRole, "")
            names = [self.name(f) for f in files]
            old = [n for n in names if n in seen]
            if first_run:
                new_on = (label == self.first_run_folder) or not has_first
            else:
                new_on = bool(old) and all(n in checked for n in old)
            for f, n in zip(files, names):
                it = QTreeWidgetItem(node, [f.name])
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable
                            | Qt.ItemIsUserCheckable)
                it.setData(0, Qt.UserRole, str(f))
                it.setToolTip(0, str(f))
                on = (n in checked) if n in seen else new_on
                it.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            node.setExpanded(False)
        self._seen = {self.name(f) for _, fs in groups for f in fs}
        self.total = len(self._seen)
        self.settings.setValue(f"{self.key}/seen", sorted(self._seen))
        self._loading = False
        self._on_changed()

    def _on_changed(self, *_):
        if self._loading:
            return
        self.settings.setValue(f"{self.key}/checked", self.checked_names())
        self.changed.emit()

    def _on_double_click(self, item, _col=0):
        if item.data(0, Qt.UserRole):              # files only, not folders
            self.activated.emit(item.data(0, Qt.UserRole))


class RenderWorker(QThread):
    """Writes the job as JSON, runs gacha_engine.py on it in a subprocess and
    streams its log lines."""
    line = Signal(str)
    finished_ok = Signal(bool)

    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self):
        job_file = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json",
                                             prefix="gacha_job_",
                                             delete=False) as fh:
                json.dump(self.job, fh)
                job_file = fh.name
            proc = subprocess.Popen(
                [sys.executable, "-u", str(GACHA), job_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for ln in proc.stdout:
                self.line.emit(ln.rstrip())
            self.finished_ok.emit(proc.wait() == 0)
        except Exception as e:
            self.line.emit(f"error: {e}")
            self.finished_ok.emit(False)
        finally:
            if job_file:
                Path(job_file).unlink(missing_ok=True)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gacha")
        self.resize(1050, 640)
        self.worker = None

        # ---------- audio player ----------
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(0.9)
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_out)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self._segment = None        # (start_ms, end_ms) while auditioning a slice
        self._sec_bounds = []       # section start times (ms) of the playing file
        self._sec_idx = -1          # which section the playhead is in
        self._last_switch_ms = 0.0  # song position of the last clip switch
        self._sections = []         # full section dicts of the playing file
        self._beat_ms = 500.0
        self._env = None            # loudness envelope, one value per 50 ms
        self._outro = None          # (start_ms, end_ms) of the ending effect
        self._travel = 0            # mouse distance since the interface hid
        self._last_mouse = None
        self._cursor_hidden = False
        self._look = {}             # per-section random multipliers per effect
        self._look_idx = None       # (section, 4-bar phrase) of the current look
        self._words_phrase = None   # (section, phrase) currently showing words
        self._reverse_plan = None   # (phrase key, first bar, bars) to run backwards
        self._words_slot = None     # half-beat slot of the cloud on screen
        self._song_word = None      # the word in the playing render's file name
        self.font_families = load_fonts()
        self._kaleido = 0
        self._mono = (0, 0.5)
        self._pix_seed = 0.0
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self._hide_ui)

        # ---------- parameters panel ----------
        params_box = self._build_params()

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setMinimumHeight(36)
        self.generate_btn.clicked.connect(self.generate)

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(2000)

        left = QVBoxLayout()
        left.addWidget(params_box)
        left.addWidget(self.generate_btn)
        left.addWidget(QLabel("Render log"))
        left.addWidget(self.log, stretch=1)
        left_w = QWidget()
        left_w.setLayout(left)

        # ---------- browser tabs ----------
        self.outputs_list = QListWidget()
        self.outputs_list.itemDoubleClicked.connect(self._play_item)
        # libraries: one node per set folder, tri-state checkboxes
        self.settings = QSettings("gacha", "gacha")
        self.samples_tree = SetTree(SAMPLES_DIR, AUDIO_EXTS, self.settings,
                                    "samples", first_run_folder="default")
        self.samples_tree.activated.connect(self._play_path)
        self.samples_tree.changed.connect(self._on_samples_changed)
        self.videos_tree = SetTree(VIDEOS_DIR, VIDEO_EXTS, self.settings,
                                   "videos", first_run_folder="(loose files)")
        self.videos_tree.activated.connect(self._show_video)
        self.videos_tree.changed.connect(self._on_videos_changed)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_lists)

        tabs = QTabWidget()
        self.samples_status = QLabel("")
        self.videos_status = QLabel("")
        for tree, status, title in ((self.samples_tree, self.samples_status, "Samples"),
                                    (self.videos_tree, self.videos_status, "Videos")):
            status.setStyleSheet("color: rgb(170, 120, 110); padding: 2px 4px;")
            box = QVBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(0)
            box.addWidget(tree, stretch=1)
            box.addWidget(status)
            w = QWidget()
            w.setLayout(box)
            tabs.addTab(w, title)
        tabs.addTab(self.outputs_list, "Outputs")
        tabs.addTab(self._build_mixer(), "Mixer")

        # ---------- player bar ----------
        self.now_playing = QLabel("—")
        self.now_playing.setStyleSheet("font-weight: bold;")
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.player.stop)
        self.pos_slider = SeekSlider(Qt.Horizontal)
        self.pos_slider.sliderMoved.connect(self.player.setPosition)
        self.time_lbl = QLabel("0:00 / 0:00")
        vol = QSlider(Qt.Horizontal, maximumWidth=100)
        vol.setRange(0, 100)
        vol.setValue(90)
        vol.valueChanged.connect(lambda v: self.audio_out.setVolume(v / 100))

        bar = QHBoxLayout()
        bar.addWidget(self.play_btn)
        bar.addWidget(stop_btn)
        bar.addWidget(self.pos_slider, stretch=1)
        bar.addWidget(self.time_lbl)
        bar.addWidget(QLabel("Vol"))
        bar.addWidget(vol)

        right = QVBoxLayout()
        right.addWidget(refresh_btn)
        right.addWidget(tabs, stretch=1)
        right.addWidget(self.now_playing)
        right.addLayout(bar)
        right_w = QWidget()
        right_w.setLayout(right)

        split = QSplitter()
        split.addWidget(left_w)
        split.addWidget(right_w)
        left_w.setMinimumWidth(560)      # four-spinbox rows need the room
        split.setSizes([560, 490])
        self.setStyleSheet(STYLE)
        self.gl_mode = gl_available()
        if self.gl_mode:
            self.backdrop = GLBackdrop(split, self._song_videos, DEFAULT_VIDEO)
        else:
            self.backdrop = VideoBackdrop(split, self._song_videos)
        self.backdrop.state_fn = self._video_state
        self.backdrop.shader_mix = self.shader_mix.value()
        self.backdrop.shader_blend = self.shader_blend.currentIndex()
        self.setCentralWidget(self.backdrop)
        self._apply_shader_choice()
        if hasattr(self.backdrop, "prewarm"):
            self.backdrop.prewarm(shader_files())   # compile all shaders early
        # shader compile errors from the GL backdrop land in the render log
        self._gl_log_timer = QTimer(self)
        self._gl_log_timer.timeout.connect(self._drain_gl_log)
        self._gl_log_timer.start(500)
        self.ui = split
        QApplication.instance().installEventFilter(self)
        # F11 as a real application shortcut: fires exactly once per press,
        # whatever has focus, even with the interface hidden
        self.fs_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
        self.fs_shortcut.setContext(Qt.ApplicationShortcut)
        self.fs_shortcut.activated.connect(self.toggle_fullscreen)
        # Space: next video FX (next shader + new random look); S: shader on/off
        self._prev_shader_choice = "random"
        self._shader_user_off = False     # off by your own hand: songs leave it off
        self.shader.activated.connect(
            lambda _i: setattr(self, "_shader_user_off",
                               self.shader.currentText() == "off"))
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.next_shortcut.setContext(Qt.ApplicationShortcut)
        self.next_shortcut.activated.connect(self.next_video_fx)
        self.shader_shortcut = QShortcut(QKeySequence(Qt.Key_S), self)
        self.shader_shortcut.setContext(Qt.ApplicationShortcut)
        self.shader_shortcut.activated.connect(self.toggle_shader)
        # 1..9: pick that shader directly, 0: shader layer off
        self.digit_shortcuts = []
        for n in range(10):
            sc = QShortcut(QKeySequence(getattr(Qt, f"Key_{n}")), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(lambda n=n: self.select_shader_key(n))
            self.digit_shortcuts.append(sc)

        self.refresh_lists()

    def _drain_gl_log(self):
        """GL backdrop messages (shader compile errors) -> render log."""
        pending = getattr(self.backdrop, "log", None)
        if pending:
            for line in pending:
                self.log.appendPlainText(f"video: {line}")
            pending.clear()

    # ---------- parameter tabs ----------
    def _build_params(self):
        tabs = QTabWidget()
        tabs.addTab(self._tab_general(), "General")
        tabs.addTab(self._tab_drums(), "Drums")
        tabs.addTab(self._tab_sections(), "Sections")
        tabs.addTab(self._tab_layers(), "Layers && FX")
        tabs.addTab(self._tab_video(), "Video")
        return tabs

    def _tab_video(self):
        form = QFormLayout()
        self.video_jump = QCheckBox("jump to a random spot on every section")
        self.video_jump.setChecked(True)
        form.addRow("Playback", self.video_jump)
        self.video_switch = QCheckBox("switch between the ticked videos on "
                                      "drops, now and then on sections")
        self.video_switch.setChecked(True)
        self.video_switch.setToolTip(
            "Only with two or more videos ticked in the Videos tab. A drop "
            "may swap the clip once 4 bars have passed since the last swap; "
            "an ordinary section boundary swaps it about one time in three "
            "after 16 quiet bars.")
        form.addRow("", self.video_switch)

        self.auto_hide = QCheckBox("hide the interface when a song starts, "
                                   "and again after")
        self.auto_hide.setChecked(True)
        self.auto_hide.toggled.connect(self._arm_hide)
        self.hide_after = QDoubleSpinBox(minimum=0.5, maximum=10.0,
                                         singleStep=0.5, decimals=1,
                                         suffix=" s")
        self.hide_after.setValue(1.5)
        self.hide_after.setToolTip("The interface hides the moment playback "
                                   "starts. Move the mouse, click or press a "
                                   "key to bring it back; it hides again after "
                                   "this much idle time")
        form.addRow("", self._pair(self.auto_hide, self.hide_after))

        self.shuffle_look = QCheckBox("new random look on every section")
        self.shuffle_look.setChecked(True)
        self.shuffle_look.setToolTip("Each section re-rolls which of the "
                                     "effects below are active and how hard")
        form.addRow("", self.shuffle_look)
        fs_btn = QPushButton("Fullscreen (F11)")
        fs_btn.setMinimumHeight(30)
        fs_btn.clicked.connect(self.toggle_fullscreen)
        form.addRow("", fs_btn)

        self.shader = QComboBox()
        self.shader.addItems(["off", "random"] + [p.stem for p in shader_files()])
        self.shader.setCurrentText("off")          # on once a song is generated
        self.shader.setToolTip("Generative shader layer drawn over the video "
                               "(OpenGL). random = a new one every section. "
                               "Keys: S toggles the layer, Space = next FX, "
                               "1-9 pick a shader, 0 = off. "
                               "Drop Shadertoy-style .frag files in shaders/")
        self.shader.currentIndexChanged.connect(
            lambda _i: self._apply_shader_choice())
        form.addRow("Shader", self.shader)
        self.shader_mix = self._prob_spin(0.5, "Opacity of the shader layer "
                                          "over the video")
        self.shader_mix.valueChanged.connect(
            lambda v: setattr(self.backdrop, "shader_mix", v))
        self.shader_blend = QComboBox()
        self.shader_blend.addItems(BLEND_MODES)
        self.shader_blend.currentIndexChanged.connect(
            lambda i: setattr(self.backdrop, "shader_blend", i))
        form.addRow("Shader layer", self._row([("opacity", self.shader_mix),
                                               ("blend", self.shader_blend)]))

        self.vfx = {}
        for key, label, default, tip in VIDEO_EFFECTS:
            self.vfx[key] = self._prob_spin(default, tip)
        rows = [("Color", ("color", "tint", "pump", "flash")),
                ("Pixels", ("pixel", "bits", "dither")),
                ("Tone", ("mono", "solar", "edges", "lines")),
                ("Grain", ("grain",)),
                ("Motion", ("zoom", "trails", "glitch", "rgb", "reverse")),
                ("Frame", ("vign", "kaleido", "words"))]
        labels = {k: lbl for k, lbl, _, _ in VIDEO_EFFECTS}
        for title, keys in rows:
            form.addRow(title, self._row([(labels[k], self.vfx[k])
                                          for k in keys]))
        hint = QLabel("strength 0 = off; driven by the song's sections and "
                      "loudness")
        hint.setProperty("role", "sub")
        form.addRow("", hint)
        return self._wrap(form, self._randomize_video)

    def _randomize_video(self):
        self._shuffle(*self.vfx.values(), self.shader_blend)
        self.shader_mix.setValue(random.choice([0.3, 0.5, 0.7, 1.0]))
        self.shader.setCurrentText("random")
        self._look_idx = None                  # re-roll the look right away

    def _apply_shader_choice(self):
        """Combo -> backdrop. 'random' picks per section in _roll_look."""
        if not hasattr(self, "backdrop"):
            return
        choice = self.shader.currentText()
        if choice == "off":
            self.backdrop.set_shader(None)
        elif choice != "random":
            for p in shader_files():
                if p.stem == choice:
                    self.backdrop.set_shader(p)
        else:
            self._pick_section_shader()

    def toggle_shader(self):
        """S: shader layer off <-> back to the previous choice."""
        if self.shader.currentText() == "off":
            self.shader.setCurrentText(self._prev_shader_choice)
            self._shader_user_off = False
        else:
            self._prev_shader_choice = self.shader.currentText()
            self.shader.setCurrentText("off")
            self._shader_user_off = True
        self.log.appendPlainText(f"video: shader {self.shader.currentText()}")

    def select_shader_key(self, n):
        """Digit keys: 0 = shader layer off, 1..9 = the n-th shader file."""
        if n == 0:
            if self.shader.currentText() != "off":
                self._prev_shader_choice = self.shader.currentText()
            self.shader.setCurrentText("off")
            self._shader_user_off = True
        else:
            files = shader_files()
            if n > len(files):
                return
            self.shader.setCurrentText(files[n - 1].stem)
            self._shader_user_off = False
        self.log.appendPlainText(f"video: key {n} -> shader "
                                 f"{self.shader.currentText()}")

    def next_video_fx(self):
        """Space: a new random effect look, plus the next shader (random when
        the combo says random). With the shader layer off (S) only the video
        effects cycle. Never brings the interface back."""
        files = shader_files()
        choice = self.shader.currentText()
        if choice == "off":
            pass                               # shaders stay off
        elif choice == "random":
            self._pick_section_shader()
        elif files:
            names = [p.stem for p in files]
            nxt = names[(names.index(choice) + 1) % len(names)] \
                if choice in names else names[0]
            self.shader.setCurrentText(nxt)
        self._look_idx = None                  # re-roll the effect look
        cur = getattr(self.backdrop, "shader_path", None)
        shown = "off" if choice == "off" or cur is None else cur.stem
        self.log.appendPlainText(f"video: next fx -> new look, shader {shown}")

    def _pick_section_shader(self):
        files = shader_files()
        if not files:
            return
        cur = getattr(self.backdrop, "shader_path", None)
        pool = [p for p in files if p != cur] or files
        self.backdrop.set_shader(random.choice(pool))

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self._show_ui()             # back to the desktop: interface back
            self._arm_hide()
        else:
            self.showFullScreen()
            self._hide_ui(force=True)   # straight to the picture

    def _tab_general(self):
        form = QFormLayout()
        self.count = QSpinBox(minimum=1, maximum=50, value=1)
        form.addRow("Number of songs", self.count)

        self.duration = QSpinBox(minimum=10, maximum=600, value=180,
                                 suffix=" s")
        form.addRow("Duration", self.duration)

        self.bpm_random = QCheckBox("random (84–128)")
        self.bpm_random.setChecked(True)
        self.bpm = QSpinBox(minimum=40, maximum=220, value=110)
        self.bpm.setEnabled(False)
        self.bpm_random.toggled.connect(lambda on: self.bpm.setEnabled(not on))
        form.addRow("BPM", self._pair(self.bpm, self.bpm_random))

        self.seed_random = QCheckBox("random")
        self.seed_random.setChecked(True)
        self.seed = QSpinBox(minimum=0, maximum=999999, value=42)
        self.seed.setEnabled(False)
        self.seed_random.toggled.connect(lambda on: self.seed.setEnabled(not on))
        form.addRow("Seed", self._pair(self.seed, self.seed_random))

        self.all_samples = QCheckBox("use all")
        self.all_samples.setChecked(True)
        self.num_samples = QSpinBox(minimum=1, maximum=9999, value=10)
        self.num_samples.setEnabled(False)
        self.all_samples.toggled.connect(
            lambda on: self.num_samples.setEnabled(not on))
        form.addRow("Samples", self._pair(self.num_samples, self.all_samples))

        return self._wrap(form)

    def _tab_drums(self):
        form = QFormLayout()
        self.drum_style = QComboBox()
        self.drum_style.addItems(["random", "four-floor", "breakbeat",
                                  "boom-bap", "halftime", "dnb", "minimal",
                                  "ukg", "dembow", "one-drop", "footwork",
                                  "clave", "idm"])
        form.addRow("Drum style", self.drum_style)

        self.swing_random = QCheckBox("random (0–0.06)")
        self.swing_random.setChecked(True)
        self.swing = QDoubleSpinBox(minimum=0.0, maximum=0.5, singleStep=0.01,
                                    decimals=2, value=0.1)
        self.swing.setToolTip("Delay of every offbeat 16th, as a fraction "
                              "of a step. 0.33 = triplet feel.")
        self.swing.setEnabled(False)
        self.swing_random.toggled.connect(
            lambda on: self.swing.setEnabled(not on))
        form.addRow("Swing", self._pair(self.swing, self.swing_random))
        self.humanize = self._prob_spin(0.5, "How much like a person the drums "
                                        "are played. Velocities wobble, "
                                        "downbeats lean in, offbeat 16ths sit "
                                        "back, timing drifts a few ms; soft "
                                        "hits come out duller and shorter, "
                                        "hard hits bite. 0 = machine")
        form.addRow("Humanize", self.humanize)

        self.mutation_chance = self._prob_spin(0.3, "Per bar, chance a drum "
                                               "pattern is mutated")
        self.mutation_amount = self._prob_spin(0.1, "Fraction of steps a "
                                               "mutation touches")
        form.addRow("Mutation", self._row([("chance", self.mutation_chance),
                                           ("amount", self.mutation_amount)]))

        self.glitch_chance = self._prob_spin(0.15, "Per bar, chance of a "
                                             "hihat stutter burst")
        self.break_chance = self._prob_spin(0.25, "Chance a middle section "
                                            "drops to a break")
        form.addRow("Glitch / break", self._row([("hihat", self.glitch_chance),
                                                 ("break", self.break_chance)]))

        self.gain = {}
        for role, val in (("kick", 0.95), ("snare", 0.8),
                          ("hihat", 0.45), ("ohat", 0.4), ("ride", 0.3)):
            self.gain[role] = QDoubleSpinBox(minimum=0.0, maximum=2.0,
                                             singleStep=0.05, decimals=2)
            self.gain[role].setValue(val)
        self.gain["ohat"].setToolTip("Open hihat, carved from the same sample "
                                     "as the closed one. 0 = no open hats")
        form.addRow("Samples", self._row([(r, w) for r, w in self.gain.items()]))

        # synthesized 909/808-style kit, layered on the same patterns
        self.synth = {}
        for role in ("kick", "snare", "hihat", "ohat", "ride"):
            self.synth[role] = QDoubleSpinBox(minimum=0.0, maximum=2.0,
                                              singleStep=0.05, decimals=2)
            self.synth[role].setValue(0.0)
            self.synth[role].setToolTip(
                f"Synthesized {role}: no sample involved. Plays the same "
                "pattern as the sample voice, so the two layer; set the "
                "sample volume to 0 for a pure drum machine")
        self.synth_style = QComboBox()
        self.synth_style.addItems(SYNTH_STYLES)
        self.synth_style.setToolTip("Flavour of the synth kit. auto follows "
                                    "the drum style: house for four-floor, "
                                    "ukg, minimal, one-drop, boom-bap; techno "
                                    "for halftime, idm, dembow, clave; dnb for "
                                    "dnb, breakbeat, footwork. Picking a "
                                    "flavour sets the synth volumes to a mix "
                                    "that suits it; auto leaves them alone")
        self.synth_style.currentTextChanged.connect(self._apply_synth_preset)
        form.addRow("Synth", self._row([(r, w) for r, w in self.synth.items()]
                                       + [("style", self.synth_style)]))

        self.sub_level = self._prob_spin(0.6, "Synth sub under the kick, "
                                         "level relative to the mix. 0 = off")
        self.sub_phaser = self._prob_spin(0.5, "Phaser depth on the left "
                                          "oscillator. 0 = two dry sines")
        self.sub_pan = self._prob_spin(0.3, "Dry oscillator panned right, "
                                       "phased one panned left by this much. "
                                       "Keep low for a mono-safe sub")
        form.addRow("Sub bass", self._row([("level", self.sub_level),
                                           ("phaser", self.sub_phaser),
                                           ("pan", self.sub_pan)]))
        self.sub_chance = self._prob_spin(0.6, "Per bar, chance of a sub note "
                                          "on the bar's first kick")
        self.sub_release = QDoubleSpinBox(minimum=0.1, maximum=6.0,
                                          singleStep=0.1, decimals=1,
                                          suffix=" s")
        self.sub_release.setValue(1.5)
        self.sub_release.setToolTip("Release after a one-beat hold")
        form.addRow("Sub notes", self._row([("chance", self.sub_chance),
                                            ("release", self.sub_release)]))
        hint = QLabel("pitched to each section's root note, on the first kick "
                      "of a bar")
        hint.setProperty("role", "sub")
        form.addRow("", hint)
        return self._wrap(form, self._randomize_drums)

    def _randomize_drums(self):
        # a concrete style, never "random"
        self.drum_style.setCurrentIndex(
            random.randrange(1, self.drum_style.count()))
        self._shuffle(self.swing_random, self.swing, self.humanize,
                      self.mutation_chance, self.mutation_amount,
                      self.glitch_chance, self.break_chance,
                      self.sub_level, self.sub_phaser, self.sub_pan,
                      self.sub_chance, *self.gain.values(),
                      *self.synth.values(), self.synth_style)
        self.sub_release.setValue(random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0]))

    def _tab_sections(self):
        form = QFormLayout()
        self.intro_style = QComboBox()
        self.intro_style.addItems(["ambient", "sparse", "none", "build",
                                   "drums-first", "reverse-swell", "collage",
                                   "filtered"])
        form.addRow("Intro style", self.intro_style)

        self.intro_cap = QCheckBox("cap at")
        self.intro_bars = QSpinBox(minimum=0, maximum=8, value=4,
                                   suffix=" bars")
        self.intro_bars.setToolTip("0 = no intro section at all")
        self.intro_bars.setEnabled(False)
        self.intro_cap.toggled.connect(
            lambda on: self.intro_bars.setEnabled(on))
        form.addRow("Intro length", self._pair(self.intro_bars, self.intro_cap))

        self.outro_style = QComboBox()
        self.outro_style.addItems(["random", "none", "hall_wash", "dub_echo",
                                   "tape_stop", "filter_close",
                                   "shimmer_freeze", "bitcrush_collapse",
                                   "codec_rot", "glitch_stutter",
                                   "overdrive_bloom", "smear"])
        self.outro_style.setToolTip("How the song ends: an effect blends in "
                                    "over the last bars and rings out past "
                                    "the end. none = hard stop, loopable")
        form.addRow("Outro style", self.outro_style)

        self.outro_tail_auto = QCheckBox("auto")
        self.outro_tail_auto.setChecked(True)
        self.outro_tail = QDoubleSpinBox(minimum=0.5, maximum=12.0,
                                         singleStep=0.5, decimals=1,
                                         suffix=" s")
        self.outro_tail.setValue(4.0)
        self.outro_tail.setEnabled(False)
        self.outro_tail_auto.toggled.connect(
            lambda on: self.outro_tail.setEnabled(not on))
        self.outro_tail.setToolTip("Ring-out appended past the last bar. "
                                   "auto = per style, 2.5 to 7 s")
        form.addRow("Outro tail", self._pair(self.outro_tail,
                                             self.outro_tail_auto))

        self.outro_bars = QDoubleSpinBox(minimum=0.25, maximum=16.0,
                                         singleStep=0.5, decimals=2)
        self.outro_bars.setValue(2.0)
        self.outro_bars.setToolTip("The effect blends in over the last N bars")
        self.outro_wet = self._prob_spin(1.0, "Maximum wetness of the ending "
                                         "effect")
        self.outro_amount = self._prob_spin(0.5, "Per-style primary knob: "
                                            "decay, feedback or crush depth")
        form.addRow("Outro mix", self._row([("bars", self.outro_bars),
                                            ("wet", self.outro_wet),
                                            ("amount", self.outro_amount)]))

        self.section_bars = QLineEdit("4, 4, 8, 8, 16")
        self.section_bars.setValidator(QRegularExpressionValidator(
            QRegularExpression(r"^\s*\d+(\s*,\s*\d+)*\s*,?\s*$")))
        self.section_bars.setToolTip("Bar lengths a section can have, picked "
                                     "at random. Repeat a value to make it "
                                     "more likely.")
        form.addRow("Section lengths", self.section_bars)
        hint = QLabel("comma-separated bars, repeats = weight")
        hint.setProperty("role", "sub")
        form.addRow("", hint)

        self.fade_in = QDoubleSpinBox(minimum=0.0, maximum=10.0,
                                      singleStep=0.1, decimals=2, suffix=" s")
        self.fade_in.setValue(0.3)
        self.fade_out = QDoubleSpinBox(minimum=0.0, maximum=20.0,
                                       singleStep=0.5, decimals=2, suffix=" s")
        self.fade_out.setValue(0.03)
        self.fade_out.setToolTip("Default 0.03 s is only a click guard so "
                                 "the outro stays loopable in the Mixer")
        form.addRow("Fades", self._row([("in", self.fade_in),
                                        ("out", self.fade_out)]))

        self.cymbal_chance = self._prob_spin(0.3, "Per drop: reverse cymbal "
                                             "swell with an opening filter")
        self.repeat_chance = self._prob_spin(0.25, "Per drop: DJ beat-repeat "
                                             "roll on the last bar before it")
        self.gap_chance = self._prob_spin(0.0, "Per drop: half a beat to two "
                                          "beats of silence right before it")
        form.addRow("Drop FX", self._row([("cymbal", self.cymbal_chance),
                                          ("repeat", self.repeat_chance),
                                          ("gap", self.gap_chance)]))
        hint2 = QLabel("chance per drop, 0 = off")
        hint2.setProperty("role", "sub")
        form.addRow("", hint2)
        return self._wrap(form, self._randomize_sections)

    def _randomize_sections(self):
        self._shuffle(self.intro_style, self.intro_bars, self.cymbal_chance,
                      self.repeat_chance, self.gap_chance, self.outro_amount)
        # outro: a concrete ending (never "random"/"none"), musical values
        self.outro_style.setCurrentIndex(
            random.randrange(2, self.outro_style.count()))
        self.outro_tail_auto.setChecked(True)
        self.outro_bars.setValue(random.choice([1.0, 2.0, 2.0, 4.0]))
        self.outro_wet.setValue(random.choice([0.7, 0.85, 1.0, 1.0]))
        # fades: round, musical values rather than 3.32 s
        self.fade_in.setValue(random.choice([0.0, 0.3, 0.3, 0.5, 1.0, 2.0]))
        self.fade_out.setValue(random.choice([0.03, 0.03, 2.0, 4.0, 6.0, 8.0]))
        self.intro_cap.setChecked(True)        # keep intros short: 0-8 bars
        pool = [1, 2, 2, 3, 4, 4, 4, 6, 8, 8, 8, 12, 16, 16, 32]
        bars = sorted(random.choice(pool) for _ in range(random.randint(1, 5)))
        self.section_bars.setText(", ".join(map(str, bars)))

    def _tab_layers(self):
        form = QFormLayout()
        self.layers_min = QSpinBox(minimum=0, maximum=8, value=1)
        self.layers_max = QSpinBox(minimum=0, maximum=8, value=3)
        form.addRow("Textures / section",
                    self._row([("min", self.layers_min),
                               ("max", self.layers_max)]))

        self.level_min = self._prob_spin(0.18, "Quietest texture gain")
        self.level_max = self._prob_spin(0.4, "Loudest texture gain")
        form.addRow("Texture level", self._row([("min", self.level_min),
                                                ("max", self.level_max)]))

        self.chop_chance = self._prob_spin(0.6, "Per section, chance of a "
                                           "rhythmic chop layer")
        self.chop_gain = QDoubleSpinBox(minimum=0.0, maximum=2.0,
                                        singleStep=0.05, decimals=2)
        self.chop_gain.setValue(0.4)
        form.addRow("Chops", self._row([("chance", self.chop_chance),
                                        ("gain", self.chop_gain)]))

        self.reverse_chance = self._prob_spin(0.35, "Chance a texture plays "
                                              "backwards")
        self.stretch_chance = self._prob_spin(0.3, "Chance a texture is "
                                              "time-stretched to whole bars")
        form.addRow("Reverse / stretch",
                    self._row([("rev", self.reverse_chance),
                               ("stretch", self.stretch_chance)]))

        self.saturation = self._prob_spin(0.5, "Drive of the gain-compensated "
                                          "texture saturator. It never gets "
                                          "louder, only denser. 0 removes it "
                                          "from the effect pool.")
        form.addRow("Saturation", self.saturation)

        self.random_start = QComboBox()
        self.random_start.addItems(RANDOM_START_MODES)
        self.random_start.setToolTip(
            "Where cuts are taken from a sample. off: from its beginning. "
            "one-shots: the leftover one-shot events start at a random point. "
            "all: background textures and collage hits too, so long recordings "
            "with a quiet opening contribute their whole length.")
        form.addRow("Random start", self.random_start)

        self.fx_min = QSpinBox(minimum=1, maximum=9, value=1)
        self.fx_max = QSpinBox(minimum=1, maximum=9, value=3)
        form.addRow("Effects / chain", self._row([("min", self.fx_min),
                                                  ("max", self.fx_max)]))

        self.pan_drums = self._pan_spin(0.3)
        self.pan_layers = self._pan_spin(0.6)
        self.pan_events = self._pan_spin(0.5)
        form.addRow("Pan", self._row([("drums", self.pan_drums),
                                      ("layers", self.pan_layers),
                                      ("events", self.pan_events)]))
        return self._wrap(form, self._randomize_layers)

    def _randomize_layers(self):
        self._shuffle(self.layers_min, self.layers_max,
                      self.level_min, self.level_max,
                      self.chop_chance, self.chop_gain,
                      self.reverse_chance, self.stretch_chance,
                      self.random_start,
                      self.saturation, self.fx_min, self.fx_max,
                      self.pan_drums, self.pan_layers, self.pan_events)
        self._order(self.layers_min, self.layers_max)
        self._order(self.level_min, self.level_max)
        self._order(self.fx_min, self.fx_max)

    # ---------- mixer tab ----------
    def _build_mixer(self):
        self.mixer_renders = QListWidget()
        self.mixer_renders.currentItemChanged.connect(self._mixer_load_sections)
        self.mixer_sections = QListWidget()
        self.mixer_sections.itemDoubleClicked.connect(self._audition_section)
        self.arrangement = QListWidget()
        self.arrangement.itemDoubleClicked.connect(self._audition_section)

        add_btn = QPushButton("Add to final →")
        add_btn.clicked.connect(self._arr_add)
        up_btn = QPushButton("↑")
        up_btn.clicked.connect(lambda: self._arr_move(-1))
        down_btn = QPushButton("↓")
        down_btn.clicked.connect(lambda: self._arr_move(1))
        del_btn = QPushButton("Remove")
        del_btn.clicked.connect(self._arr_remove)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.arrangement.clear)
        render_btn = QPushButton("Render final")
        render_btn.setMinimumHeight(32)
        render_btn.clicked.connect(self._render_final)

        col1 = QVBoxLayout()
        col1.addWidget(QLabel("Renders (with section data)"))
        col1.addWidget(self.mixer_renders)

        col2 = QVBoxLayout()
        col2.addWidget(QLabel("Sections — double-click to listen"))
        col2.addWidget(self.mixer_sections)
        col2.addWidget(add_btn)

        arr_btns = QHBoxLayout()
        for b in (up_btn, down_btn, del_btn, clear_btn):
            arr_btns.addWidget(b)
        col3 = QVBoxLayout()
        col3.addWidget(QLabel("Final arrangement"))
        col3.addWidget(self.arrangement)
        col3.addLayout(arr_btns)
        col3.addWidget(render_btn)

        lay = QHBoxLayout()
        for col in (col1, col2, col3):
            lay.addLayout(col, stretch=1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _mixer_load_sections(self, item, _prev=None):
        self.mixer_sections.clear()
        if not item:
            return
        wav = Path(item.data(Qt.UserRole))
        try:
            meta = json.loads(wav.with_suffix(".json").read_text())
        except Exception:
            return
        for s in meta["sections"]:
            label = (f"sec{s['i']} · {s['kind']} · {s['bars']} bars · "
                     f"{self._fmt(s['start_sec'] * 1000)}–"
                     f"{self._fmt(s['end_sec'] * 1000)}")
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, {"wav": str(wav), "label": label,
                                     "name": wav.stem, "bpm": meta["bpm"],
                                     "start": s["start_sec"],
                                     "end": s["end_sec"]})
            self.mixer_sections.addItem(it)

    def _audition_section(self, item):
        d = item.data(Qt.UserRole)
        self._segment = (int(d["start"] * 1000), int(d["end"] * 1000))
        self._load_sections(d["wav"])
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(d["wav"]))
        self.player.play()          # seek happens once media is loaded
        self.now_playing.setText(f"{Path(d['wav']).name} [{d['label']}]")

    def _arr_add(self):
        item = self.mixer_sections.currentItem()
        if not item:
            return
        d = item.data(Qt.UserRole)
        it = QListWidgetItem(f"{d['name']} ({d['bpm']} BPM) · {d['label']}")
        it.setData(Qt.UserRole, d)
        self.arrangement.addItem(it)

    def _arr_remove(self):
        row = self.arrangement.currentRow()
        if row >= 0:
            self.arrangement.takeItem(row)

    def _arr_move(self, delta):
        row = self.arrangement.currentRow()
        new = row + delta
        if row < 0 or not (0 <= new < self.arrangement.count()):
            return
        item = self.arrangement.takeItem(row)
        self.arrangement.insertItem(new, item)
        self.arrangement.setCurrentRow(new)

    def _render_final(self):
        n = self.arrangement.count()
        if n == 0:
            self.log.appendPlainText("mixer: arrangement is empty")
            return
        sr = 44100
        fade = int(0.03 * sr)       # 30 ms equal-power crossfade at each seam
        parts = []
        for i in range(n):
            d = self.arrangement.item(i).data(Qt.UserRole)
            audio, _ = sf.read(d["wav"], dtype="float32", always_2d=True,
                               start=int(d["start"] * sr),
                               stop=int(d["end"] * sr))
            parts.append(audio)
        out = parts[0]
        f_in = np.sqrt(np.linspace(0, 1, fade))[:, None]
        for p in parts[1:]:
            k = min(fade, len(out), len(p))
            out[-k:] *= f_in[fade - k:][::-1]
            p = p.copy()
            p[:k] *= f_in[:k]
            merged = np.vstack([out[:-k], out[-k:] + p[:k], p[k:]])
            out = merged
        peak = np.abs(out).max()
        if peak > 0:
            out = out / peak * 0.95
        OUT_DIR.mkdir(exist_ok=True)
        path = OUT_DIR / f"final_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(path, out, sr, subtype="PCM_16")
        self.log.appendPlainText(
            f"✔ final mix: {path.name} ({len(out)/sr:.1f}s, {n} sections)")
        self.refresh_lists()
        self._segment = None
        self._load_sections(path)
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()
        self.now_playing.setText(path.name)

    # ---------- small ui helpers ----------
    @staticmethod
    def _pair(widget, check):
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(widget)
        box.addWidget(check)
        w = QWidget()
        w.setLayout(box)
        return w

    @staticmethod
    def _row(pairs):
        """Several labelled widgets side by side: [("kick", spin), ...]."""
        box = QHBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        for i, (name, widget) in enumerate(pairs):
            lbl = QLabel(name)
            lbl.setProperty("role", "sub")
            if i:
                box.addSpacing(6)
            if len(pairs) >= 3 and isinstance(widget, QAbstractSpinBox):
                # crowded rows: drop the arrows, keep typing / wheel / keys
                widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
            box.addWidget(lbl)
            box.addWidget(widget, stretch=1)
        w = QWidget()
        w.setLayout(box)
        return w

    @staticmethod
    def _wrap(form, randomize=None):
        """Form in a tab page; optional randomize callback gets a button
        pinned to the bottom-right corner."""
        form.setContentsMargins(8, 8, 8, 8)
        form.setVerticalSpacing(6)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 8, 8)
        outer.addLayout(form)
        outer.addStretch(1)
        if randomize is not None:
            btn = QPushButton("Randomize")
            btn.setToolTip("Random value for every control on this tab, "
                           "within its range")
            btn.clicked.connect(randomize)
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(btn)
            outer.addLayout(row)
        w = QWidget()
        w.setLayout(outer)
        return w

    @staticmethod
    def _shuffle(*widgets):
        """Random value for each widget, respecting its own range."""
        for w in widgets:
            if isinstance(w, QDoubleSpinBox):
                w.setValue(round(random.uniform(w.minimum(), w.maximum()),
                                 w.decimals()))
            elif isinstance(w, QSpinBox):
                w.setValue(random.randint(w.minimum(), w.maximum()))
            elif isinstance(w, QComboBox):
                w.setCurrentIndex(random.randrange(w.count()))
            elif isinstance(w, QCheckBox):
                w.setChecked(random.random() < 0.5)

    @staticmethod
    def _order(lo, hi):
        """Keep a min/max spinbox pair sorted."""
        if lo.value() > hi.value():
            a, b = lo.value(), hi.value()
            lo.setValue(b)
            hi.setValue(a)

    @staticmethod
    def _pan_spin(val):
        s = QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.1)
        s.setValue(val)
        return s

    @staticmethod
    def _prob_spin(val, tip=""):
        s = QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.05,
                           decimals=2)
        s.setValue(val)
        if tip:
            s.setToolTip(tip)
        return s

    # synth volumes that suit each flavour: house rides the open hat, techno
    # is kick first, drum and bass is a snare music
    SYNTH_PRESETS = {
        "house": {"kick": 0.9, "snare": 0.55, "hihat": 0.35, "ohat": 0.45, "ride": 0.15},
        "techno": {"kick": 1.0, "snare": 0.4, "hihat": 0.45, "ohat": 0.3, "ride": 0.1},
        "dnb": {"kick": 0.8, "snare": 0.85, "hihat": 0.4, "ohat": 0.25, "ride": 0.2},
    }

    def _apply_synth_preset(self, style):
        for role, val in self.SYNTH_PRESETS.get(style, {}).items():
            self.synth[role].setValue(val)

    # ---------- generation ----------
    def generate(self):
        samples = self.checked_samples()
        if not samples:
            self.log.appendPlainText(
                "✗ no samples selected — tick some in the Samples tab")
            return
        job = {"count": self.count.value(),
               "duration": self.duration.value(),
               "samples": samples,
               "intro_style": self.intro_style.currentText(),
               "outro_style": self.outro_style.currentText(),
               "drum_style": self.drum_style.currentText(),
               "pan_drums": self.pan_drums.value(),
               "pan_layers": self.pan_layers.value(),
               "pan_events": self.pan_events.value(),
               "seed": None if self.seed_random.isChecked() else self.seed.value(),
               "bpm": None if self.bpm_random.isChecked() else self.bpm.value()}
        if not self.all_samples.isChecked():
            job["num_samples"] = self.num_samples.value()
        if self.intro_cap.isChecked():
            job["intro_bars"] = self.intro_bars.value()
        knobs = {"humanize": self.humanize.value(),
                 "outro_bars": self.outro_bars.value(),
                 "outro_wet": self.outro_wet.value(),
                 "outro_amount": self.outro_amount.value(),
                 "mutation_chance": self.mutation_chance.value(),
                 "mutation_amount": self.mutation_amount.value(),
                 "glitch_chance": self.glitch_chance.value(),
                 "break_chance": self.break_chance.value(),
                 "fade_in": self.fade_in.value(),
                 "fade_out": self.fade_out.value(),
                 "cymbal_chance": self.cymbal_chance.value(),
                 "repeat_chance": self.repeat_chance.value(),
                 "gap_chance": self.gap_chance.value(),
                 "layers_min": self.layers_min.value(),
                 "layers_max": self.layers_max.value(),
                 "layer_level_min": self.level_min.value(),
                 "layer_level_max": self.level_max.value(),
                 "chop_chance": self.chop_chance.value(),
                 "chop_gain": self.chop_gain.value(),
                 "fx_min": self.fx_min.value(),
                 "fx_max": self.fx_max.value(),
                 "reverse_chance": self.reverse_chance.value(),
                 "stretch_chance": self.stretch_chance.value(),
                 "saturation": self.saturation.value(),
                 "random_start": self.random_start.currentText(),
                 "sub_level": self.sub_level.value(),
                 "sub_phaser": self.sub_phaser.value(),
                 "sub_pan": self.sub_pan.value(),
                 "sub_chance": self.sub_chance.value(),
                 "sub_release": self.sub_release.value()}
        for role, w in self.gain.items():
            knobs[f"gain_{role}"] = w.value()
        for role, w in self.synth.items():
            knobs[f"synth_{role}"] = w.value()
        knobs["synth_style"] = self.synth_style.currentText()
        if not self.swing_random.isChecked():
            knobs["swing"] = self.swing.value()
        if not self.outro_tail_auto.isChecked():
            knobs["outro_tail"] = self.outro_tail.value()
        bars = [int(b) for b in
                self.section_bars.text().replace(" ", "").split(",")
                if b.isdigit() and int(b) >= 1]
        if bars:
            knobs["section_bars"] = bars
        job["knobs"] = knobs

        sets = sorted({Path(sample_name(f)).parent.as_posix() for f in samples})
        summary = {k: v for k, v in job.items() if k not in ("samples", "knobs")}
        self.log.appendPlainText(
            f"$ gacha_engine.py  {len(samples)} samples from {', '.join(sets)}  "
            + " ".join(f"{k}={v}" for k, v in summary.items() if v is not None))

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Rendering…")
        self.worker = RenderWorker(job)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.finished_ok.connect(self._render_done)
        self.worker.start()

    def _render_done(self, ok):
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate")
        self.refresh_lists()
        if ok and self.outputs_list.count():
            self._shader_user_off = False              # a fresh song: the show is on
            newest = self.outputs_list.item(0)
            self.outputs_list.setCurrentItem(newest)
            self._play_item(newest)          # listen right away
        elif not ok:
            self.log.appendPlainText("✗ render failed")

    # ---------- sample / video libraries ----------
    def checked_samples(self):
        return self.samples_tree.checked()

    def _on_samples_changed(self):
        n = len(self.samples_tree.checked())
        self.samples_status.setText(
            f"{n} / {self.samples_tree.total} samples selected for the next render")
        if hasattr(self, "num_samples"):
            self.num_samples.setMaximum(max(1, n))

    def _song_videos(self):
        """Backdrop pool for songs: the ticked videos, never the stock clip
        unless it is the only one. Nothing ticked falls back to every video
        so the backdrop never goes dark."""
        pool = [Path(p) for p in self.videos_tree.checked()] or video_files()
        return [f for f in pool if f != DEFAULT_VIDEO] or pool

    def _on_videos_changed(self):
        n = len(self.videos_tree.checked())
        how = ("any clip may play" if n == 0 else "this video only" if n == 1
               else "switching between them on drops")
        self.videos_status.setText(
            f"{n} / {self.videos_tree.total} videos selected, {how}")

    def _show_video(self, path):
        """Double-click in the Videos tab: loop that clip right now."""
        self.backdrop.set_video(Path(path))

    def _pick_song_video(self):
        """A new song starts: one ticked video plays as is, several pick a
        random one (a different one than now when possible)."""
        pool = self._song_videos()
        if not pool:
            return
        if len(pool) == 1:
            if self.backdrop.current != pool[0]:
                self.backdrop.set_video(pool[0], random_start=True)
        else:
            self.backdrop.pick_random()
        self._last_switch_ms = 0.0

    def _on_section_change(self, idx, pos):
        """The playhead crossed into section idx (1-based count of bounds
        passed). With several videos ticked the clip may change here: always
        on a drop (a groove after an intro or break, or a section with a
        transition effect) once 4 bars have passed since the last switch,
        and now and then on an ordinary section boundary after 16 quiet
        bars. Otherwise the current clip jumps to a random spot."""
        switch = False
        if self.video_switch.isChecked() and len(self._song_videos()) > 1 \
                and self._sections and 0 < idx <= len(self._sections):
            sec = self._sections[idx - 1]
            prev = self._sections[idx - 2] if idx >= 2 else None
            drop = bool(sec.get("transition")) or (
                sec.get("kind") == "groove" and prev is not None
                and prev.get("kind") in ("intro", "break"))
            bars_since = (pos - self._last_switch_ms) / (4 * self._beat_ms)
            if drop and bars_since >= 4:
                switch = True
            elif not drop and bars_since >= 16 and random.random() < 0.35:
                switch = True
        if switch:
            self.backdrop.pick_random()          # lands on a random spot itself
            self._last_switch_ms = pos
        elif self.video_jump.isChecked():
            self.backdrop.jump()

    # ---------- browsing / playback ----------
    def refresh_lists(self):
        self.outputs_list.clear()
        OUT_DIR.mkdir(exist_ok=True)
        files = sorted(OUT_DIR.glob("*.wav"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.UserRole, str(f))
            self.outputs_list.addItem(item)

        self.samples_tree.refresh()
        self.videos_tree.refresh()

        current = self.mixer_renders.currentItem()
        current_path = current.data(Qt.UserRole) if current else None
        self.mixer_renders.clear()
        for f in files:
            if not f.with_suffix(".json").exists():
                continue
            item = QListWidgetItem(f.name)
            item.setData(Qt.UserRole, str(f))
            self.mixer_renders.addItem(item)
            if str(f) == current_path:
                self.mixer_renders.setCurrentItem(item)

    def _load_sections(self, wav):
        """Called whenever a new file starts: pick a fresh backdrop video,
        then load the section map and loudness envelope for the video
        jumps and effects. Everything degrades to nothing."""
        self._sec_idx = -1
        self._sections, self._sec_bounds, self._env = [], [], None
        self._pick_song_video()
        self._outro = None
        parts = Path(wav).stem.split("_")
        self._song_word = parts[1] if len(parts) > 2 and parts[0] == "gacha" else None
        try:
            meta = json.loads(Path(wav).with_suffix(".json").read_text())
            self._sections = meta["sections"]
            self._sec_bounds = [int(s["start_sec"] * 1000)
                                for s in self._sections]
            self._beat_ms = 60000.0 / meta["bpm"]
            o = meta.get("outro") or {}
            if o.get("style") in (None, "none") or "tail_sec" not in o:
                raise KeyError            # no ending effect: no fade
            end_ms = self._sections[-1]["end_sec"] * 1000
            body_end = end_ms - o["tail_sec"] * 1000
            start_ms = body_end - o.get("onset_bars", 2.0) * 4 * self._beat_ms
            self._outro = (start_ms, end_ms)   # video fades to black over it
        except Exception:
            pass
        try:
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            hop = int(0.05 * sr)
            m = audio.mean(axis=1)
            n = len(m) // hop
            env = np.sqrt((m[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
            ref = np.percentile(env, 95) or 1.0
            self._env = np.clip(env / ref, 0, 1)
        except Exception:
            pass

    def _roll_look(self, idx, phrase):
        """Random look: which stylistic effects are on and how hard. Re-rolled
        every 4-bar phrase so long sections keep moving; section-level
        choices (kaleidoscope, mono tones, shader) change per section."""
        key = (idx, phrase)
        if key == self._look_idx:
            return
        section_changed = self._look_idx is None or self._look_idx[0] != idx
        self._look_idx = key
        # reverse: this phrase runs a bar or two backwards with probability
        # = strength / 2, starting on a random bar of the phrase
        self._reverse_plan = None
        if random.random() < self.vfx["reverse"].value() * 0.5:
            self._reverse_plan = (key, random.randrange(4), random.choice([1, 1, 2]))
        # word cloud: this phrase shows one with probability = strength
        self._words_phrase = None
        if random.random() < self.vfx["words"].value() and hasattr(self.backdrop, "set_overlay"):
            self._words_phrase = key
            self._words_slot = None                      # drawn per half-beat below
        if self.shuffle_look.isChecked():
            self._look = {k: (0.0 if random.random() < 0.5
                              else random.uniform(0.4, 1.3))
                          for k, *_ in VIDEO_EFFECTS if k not in LOOK_EXEMPT}
        else:
            self._look = {}
        if not section_changed:
            return
        # kaleidoscope: strength is the chance a section gets it
        self._kaleido = random.choice([4, 6, 8]) \
            if random.random() < self.vfx["kaleido"].value() else 0
        if self.shader.currentText() == "random":
            self._pick_section_shader()
        # mono: hard tones with probability = strength, else smooth gray
        hard = random.random() < self.vfx["mono"].value()
        self._mono = (random.choice([2, 2, 3, 4, 6, 8]) if hard else 0,
                      random.uniform(0.3, 0.7))
        self._pix_seed = random.random()
        # glitch bands per section: thin sharp lines (0.4-1% of the frame),
        # medium bands (4-14%), or anything from 4% up to 38% slabs
        self._glitch_mode = random.choice([0, 0, 1, 1, 2])

    REPEAT_DIVS = {"accelerate": (2, 4, 8, 16), "halving": (1, 2, 4, 8),
                   "stutter": (4, 4, 4, 4)}

    def _drop_fx(self, st, pos, boundary, transitions):
        """Visuals for the run-up to a drop, timed exactly like the audio
        transitions written in the render's JSON."""
        beat, bar = self._beat_ms, 4 * self._beat_ms
        left = boundary - pos                        # ms until the drop
        for t in transitions:
            name, _, arg = t.partition(":")
            if name == "cymbal":                     # reverse cymbal swell
                length = float(arg.rstrip("bar")) * bar
                if 0 <= left < length:
                    prog = 1 - left / length
                    st["swell"] = prog
                    st["zoom"] = st.get("zoom", 0.0) + 0.25 * prog
            elif name == "repeat" and 0 <= left < bar:   # beat-repeat roll
                divs = self.REPEAT_DIVS.get(arg, (4, 4, 4, 4))
                t_in = bar - left                    # ms into the roll bar
                beat_i = min(3, int(t_in // beat))
                div = divs[beat_i]
                slice_ms = beat / div
                sub = int((t_in - beat_i * beat) // slice_ms)
                frac = ((t_in - beat_i * beat) % slice_ms) / slice_ms
                k = sum(divs[:beat_i]) + sub         # retrigger count so far
                st["repeat"] = (k, frac)
            elif name == "gap":                      # silence = frozen frame
                beats = float(arg.rstrip("beat"))
                if 0 <= left < beats * beat:
                    st["freeze"] = True
        if st.get("freeze"):
            st.pop("repeat", None)                   # the gap cuts the roll

    def _outro_fade(self, pos):
        """1.0 normally; falls to 0.0 across the ending effect so the video
        fades to black together with the song."""
        if not self._outro:
            return 1.0
        start, end = self._outro
        if pos <= start:
            return 1.0
        return max(0.0, 1.0 - (pos - start) / max(1.0, end - start))

    def _video_state(self):
        """Effect state for the frame being drawn, from the playhead."""
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            return None
        pos = self.player.position()
        fade = self._outro_fade(pos)
        v = {k: sp.value() for k, sp in self.vfx.items()}
        env = 0.5
        if self._env is not None and len(self._env):
            env = float(self._env[min(len(self._env) - 1, int(pos // 50))])
        idx, sec, kind = 0, None, "groove"
        if self._sections and self._sec_bounds:
            idx = max(0, bisect.bisect_right(self._sec_bounds, pos) - 1)
            sec = self._sections[idx]
            kind = sec["kind"]
        # bars counted from the section's own first bar, so phrases line up
        sec_start_ms = self._sec_bounds[idx] if self._sec_bounds else 0
        bar_i = int(max(0, pos - sec_start_ms) // (4 * self._beat_ms))
        self._roll_look(idx, bar_i // 4)
        lk = lambda k: v[k] * self._look.get(k, 1.0)     # strength x look

        # music data for generative shaders (iLoud, iBeat, ...)
        sec_id = {"intro": 0, "groove": 1, "break": 2, "outro": 3}.get(kind, 1)
        root = sec.get("root") if sec is not None else None
        total_ms = self._sections[-1]["end_sec"] * 1000 if self._sections else 0
        drop = 0.0
        if sec is not None and sec.get("transition"):
            drop = max(0.0, 1 - (pos - self._sec_bounds[idx]) / 400.0)
        st = {"music": {
            "loud": env, "beat": (pos % self._beat_ms) / self._beat_ms,
            "bar": (pos % (4 * self._beat_ms)) / (4 * self._beat_ms),
            "bpm": 60000.0 / self._beat_ms, "section": sec_id,
            "root": int(round(NOTE_HUES[root] * 12)) if root in NOTE_HUES else -1,
            "drop": drop, "song_pos": pos / total_ms if total_ms else 0.0}}
        if v["pump"]:
            st["brightness"] = 0.7 + 0.7 * v["pump"] * env
        if sec is not None:
            if v["color"]:
                st["hue"] = v["color"] * ((idx * 0.37) % 1.0) * 0.5
                if kind == "intro":
                    span = max(1.0, sec["end_sec"] * 1000 - self._sec_bounds[idx])
                    t = (pos - self._sec_bounds[idx]) / span
                    st["saturation"] = 1 - v["color"] * (1 - t) * 0.8
                elif kind == "break":
                    st["saturation"] = 1 - 0.7 * v["color"]
                else:
                    st["saturation"] = 1 + 0.5 * v["color"]
            if v["pixel"] and kind == "break":
                # a rhythm inside the break: per 2-bar phrase, on ~2/3 of the
                # time with a fresh block size, breathing with the loudness
                r = random.Random(hash((idx, bar_i // 2, self._pix_seed)))
                if r.random() < 0.66:
                    size = (2 + 10 * v["pixel"]) * r.uniform(0.5, 1.5)
                    st["pixelate"] = max(2, int(size * (0.7 + 0.6 * env)))
            if lk("tint") and sec.get("root") in NOTE_HUES:
                st["tint"] = (NOTE_HUES[sec["root"]], 0.6 * lk("tint"))
            if v["flash"]:
                start = self._sec_bounds[idx]
                if sec.get("transition") and pos - start < 120 * v["flash"]:
                    st["invert"] = True            # the drop hits
                nxt = idx + 1
                if nxt < len(self._sections):
                    self._drop_fx(st, pos, self._sec_bounds[nxt],
                                  self._sections[nxt].get("transition", []))
        if lk("bits"):
            st["bits"] = max(1, round(5 - 4 * min(1.0, lk("bits"))))
        if lk("dither"):
            st["dither"] = min(1.0, lk("dither"))
        if lk("mono"):
            st["mono"] = self._mono
        if lk("solar"):
            st["solarize"] = min(1.0, lk("solar"))
        if lk("edges"):
            st["edges"] = min(1.0, lk("edges"))
        if lk("lines"):
            st["scanlines"] = min(1.0, lk("lines"))
        if lk("grain"):
            st["grain"] = min(1.0, lk("grain"))
        if lk("zoom"):
            st["zoom"] = 0.3 * lk("zoom") * env
        if lk("trails"):
            st["trails"] = min(1.0, lk("trails"))
        if lk("glitch"):
            st["glitch"] = min(1.0, lk("glitch")) * (0.3 + 0.7 * env)
            st["glitch_mode"] = self._glitch_mode
        if lk("rgb"):
            st["rgbshift"] = int(14 * lk("rgb") * (0.3 + 0.7 * env))
        if v["reverse"]:
            plan = self._reverse_plan
            if plan and plan[0] == self._look_idx \
                    and plan[1] <= bar_i % 4 < plan[1] + plan[2]:
                st["reverse"] = 1.0                 # a bar or two, real time
            nxt = idx + 1
            if nxt < len(self._sections):           # cymbal swell: rewind into
                for t in self._sections[nxt].get("transition", []):   # the drop
                    if t.startswith("cymbal:"):
                        length = float(t[7:].rstrip("bar")) * 4 * self._beat_ms
                        left = self._sec_bounds[nxt] - pos
                        if 0 <= left < length:
                            st["reverse"] = 1.0 + 2.0 * (1 - left / length)
        if lk("vign"):
            st["vignette"] = min(1.0, lk("vign"))
        if self._kaleido:
            st["kaleido"] = (self._kaleido, 0.0)      # upright, no rotation
        if self._words_phrase == self._look_idx:
            # a new cloud every half beat (1/8 of a bar): new words, new
            # places; loudness sets how many words and how bright
            phrase_ms = 16 * self._beat_ms
            t_in = (pos - sec_start_ms) - (bar_i // 4) * phrase_ms
            slot = int(t_in // (self._beat_ms / 2))
            if slot != self._words_slot:
                self._words_slot = slot
                sec_d = sec if sec is not None else {}
                root = sec_d.get("root")
                hue = NOTE_HUES.get(root) if root in NOTE_HUES else None
                words = list(WORDS)
                if self._song_word:
                    words += [self._song_word] * 6       # the song's own name, often
                n = int(40 + 140 * env)
                rng = random.Random(random.random())
                frame = self.backdrop.last_frame() if hasattr(self.backdrop, "last_frame") else None
                self.backdrop.set_overlay(make_word_cloud(
                    self.backdrop.size(), words, self.font_families, rng, hue, n,
                    frame_palette(frame, rng)))
            edge = min(1.0, t_in / (0.2 * self._beat_ms),
                       (phrase_ms - t_in) / (0.2 * self._beat_ms))
            st["words"] = min(1.0, max(0.0, edge) * (0.6 + 0.6 * env)
                              * min(1.0, 0.5 + self.vfx["words"].value()))
        if fade < 1.0:                                 # outro: fade to black
            st["brightness"] = st.get("brightness", 1.0) * fade
        st["music"]["swell"] = st.get("swell", 0.0)
        return st

    def _shader_on_for_song(self):
        """A song is starting: the shader layer comes on, unless you switched
        it off yourself (S, 0 or the combo)."""
        if self._sections and self.shader.currentText() == "off" \
                and not self._shader_user_off:
            self.shader.setCurrentText(self._prev_shader_choice)

    def _play_item(self, item):
        self._play_path(item.data(Qt.UserRole))

    def _play_path(self, path):
        self._segment = None
        self._load_sections(path)
        self._shader_on_for_song()
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        self.now_playing.setText(Path(path).name)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            if self.player.source().isEmpty():
                item = (self.outputs_list.currentItem()
                        or (self.outputs_list.count()
                            and self.outputs_list.item(0)))
                if item:
                    self._play_item(item)
                    return
                cur = self.samples_tree.currentItem()
                if cur and cur.data(0, Qt.UserRole):
                    self._play_path(cur.data(0, Qt.UserRole))
                    return
            self.player.play()

    def _on_state(self, state):
        self.play_btn.setText(
            "Pause" if state == QMediaPlayer.PlayingState else "Play")
        if state == QMediaPlayer.PlayingState and self.auto_hide.isChecked():
            self._hide_ui()                 # straight away when a song starts
        else:
            self._arm_hide()                # song over, paused or stopped: show

    # ---------- auto-hide of the interface ----------
    def _arm_hide(self, *_):
        """(Re)start the idle timer while playing; otherwise show the UI."""
        playing = self.player.playbackState() == QMediaPlayer.PlayingState
        if playing and self.auto_hide.isChecked():
            self.hide_timer.start(int(self.hide_after.value() * 1000))
        else:
            self.hide_timer.stop()
            self._show_ui()

    def _hide_ui(self, force=False):
        if force or (self.player.playbackState() == QMediaPlayer.PlayingState
                     and self.auto_hide.isChecked()):
            self._travel = 0
            self._last_mouse = None
            self.ui.hide()
            if not self._cursor_hidden:         # the pointer goes too
                QApplication.setOverrideCursor(Qt.BlankCursor)
                self._cursor_hidden = True
            self.backdrop.update()

    def _show_ui(self):
        if self._cursor_hidden:
            QApplication.restoreOverrideCursor()
            self._cursor_hidden = False
        if not self.ui.isVisible():
            self.ui.show()
            self.backdrop.update()

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.MouseMove:
            if self.ui.isVisible():
                self._arm_hide()            # activity: postpone the hide
            else:
                pos = event.globalPosition().toPoint()
                if self._last_mouse is not None:
                    d = pos - self._last_mouse
                    self._travel += abs(d.x()) + abs(d.y())
                self._last_mouse = pos
                if self._travel >= SHOW_AFTER_PX:
                    self._show_ui()
                    self._arm_hide()
        elif t in (QEvent.KeyPress, QEvent.MouseButtonPress) \
                and isinstance(obj, QWidget):
            # only the widget-level copy: the same press also arrives on the
            # native QWindow object, which would count it twice
            if t == QEvent.KeyPress and event.key() == Qt.Key_Escape \
                    and self.isFullScreen() and not event.isAutoRepeat():
                self.showNormal()
            if not self.ui.isVisible():
                self._show_ui()
            self._arm_hide()
        return super().eventFilter(obj, event)

    def _on_media_status(self, status):
        # seek to the slice start once the file is actually loaded
        if self._segment and status in (QMediaPlayer.LoadedMedia,
                                        QMediaPlayer.BufferedMedia):
            if self.player.position() < self._segment[0]:
                self.player.setPosition(self._segment[0])
    def _on_position(self, pos):
        if self._segment and pos >= self._segment[1]:
            self._segment = None
            self.player.stop()
            return
        # section index of the playhead; a change (incl. song start and
        # manual seeks) throws the backdrop somewhere new or swaps the clip
        idx = bisect.bisect_right(self._sec_bounds, pos) if self._sec_bounds else 0
        if idx != self._sec_idx:
            self._sec_idx = idx
            self._on_section_change(idx, pos)
        if not self.pos_slider.isSliderDown():
            self.pos_slider.setValue(pos)
        self.time_lbl.setText(
            f"{self._fmt(pos)} / {self._fmt(self.player.duration())}")

    def _on_duration(self, dur):
        self.pos_slider.setRange(0, dur)

    @staticmethod
    def _fmt(ms):
        s = int(ms / 1000)
        return f"{s // 60}:{s % 60:02d}"

    def closeEvent(self, event):
        self._show_ui()                         # never leave the cursor hidden
        self.player.stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec())
