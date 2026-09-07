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
import math
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
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox,
    QStyle,
    QScrollArea, QSizePolicy, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gacha_gl import BLEND_MODES, GLBackdrop, gl_available, shader_files
from gacha_engine import (AUDIO_EXTS, RANDOM_START_MODES, SYNTH_STYLES, WORDS,
                          sample_name)
from gacha_live import NOTE_NAMES, LiveInput, TapTempo, audio_inputs
from gacha_audio import LiveAudio, audio_devices
from gacha_takes import TakeStore
from gacha_section import Material, render_section, render_oneshot, KINDS, STEMS
import gacha_engine as E
from scipy.signal import resample_poly
import threading
from gacha_video import VideoSource, video_inputs
from gacha_midi import (MIDI_AVAILABLE, MidiIn, format_binding, midi_inputs,
                        parse_binding)

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


def _fact_value(v):
    """A JSON value as a short word; None for the ones not worth showing."""
    if v is None or v == "" or isinstance(v, dict):
        return None
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, list):
        if not v or any(isinstance(x, (dict, list)) for x in v):
            return None
        return " ".join(str(x) for x in v)
    return str(v)


def json_facts(meta):
    """(key, value) pairs from a render's JSON, for the word cloud: the song's
    seed, bpm, style, outro, pan, every knob that was set and the samples it
    was made from. Sections are handled per section by section_facts()."""
    facts = []
    for k, v in meta.items():
        if k in ("sections", "samples"):
            continue
        if isinstance(v, dict):                    # outro, pan, knobs: flatten
            for kk, vv in v.items():
                val = _fact_value(vv)
                if val is not None:
                    facts.append((kk, val))
        else:
            val = _fact_value(v)
            if val is not None:
                facts.append((k, val))
    for smp in meta.get("samples") or []:
        facts.append(("sample", Path(str(smp)).stem))
    return facts


def section_facts(sec):
    """(key, value) pairs describing one section of the section map."""
    facts = []
    for k, v in sec.items():
        if k == "i":
            continue
        val = _fact_value(v)
        if val is not None:
            facts.append((k, val))
    return facts


def fact_words(facts, rng):
    """Render (key, value) pairs as cloud words in a mix of JSON-ish shapes:
    `bpm: 121`, `bpm = 121`, `"bpm": 121`, or just the value or the key."""
    out = []
    for k, v in facts:
        shape = rng.random()
        if shape < 0.35:
            out.append(f"{k}: {v}")
        elif shape < 0.55:
            out.append(f"{k} = {v}")
        elif shape < 0.7:
            out.append(f'"{k}": {v}')
        elif shape < 0.9:
            out.append(v)
        else:
            out.append(k)
    return out


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
# every action the performer has, in one table: the Controls tab shows it,
# the keyboard shortcuts and the MIDI bindings hang on the same ids.
# (group, id, key, what); "" for key = MIDI only (a continuous control)
KEYMAP = [
    ("show", "fullscreen", "F11", "fullscreen on/off; Esc leaves it too"),
    ("show", "video_only", "C", "video only: no effects, no shader layer, no words"),
    ("show", "next_fx", "Space", "next video FX look, and the next shader"),
    ("show", "shader", "S", "shader layer off/on"),
    ("show", "shader_1", "1", "shader 1 in list order"),
    ("show", "shader_2", "2", "shader 2 in list order"),
    ("show", "shader_3", "3", "shader 3 in list order"),
    ("show", "shader_4", "4", "shader 4 in list order"),
    ("show", "shader_5", "5", "shader 5 in list order"),
    ("show", "shader_6", "6", "shader 6 in list order"),
    ("show", "shader_7", "7", "shader 7 in list order"),
    ("show", "shader_8", "8", "shader 8 in list order"),
    ("show", "shader_9", "9", "shader 9 in list order"),
    ("show", "shader_off", "0", "shader layer off"),
    ("tape", "through", "A", "audio through off/on (tape -> output)"),
    ("tape", "record", "R", "take: start / stop recording the live feed into RAM"),
    ("clock", "tap", "T", "tap tempo, first tap on the downbeat"),
    ("clock", "downbeat", "D", "restart the bar now, tempo kept"),
    ("clock", "live", "L", "live mode (audio drives the visuals) off/on"),
    ("clock", "drop", "X", "mark a drop by hand"),
    ("section", "next", "N", "next section on the bar (pre-rendered)"),
    ("section", "stop", "Shift+N", "stop the section on the bar"),
    ("section", "event", "E", "fire a one-shot from the material on the beat"),
    ("section", "ab_down", "[", "A/B 10 % toward the tape"),
    ("section", "ab_up", "]", "A/B 10 % toward the section"),
    ("section", "ab", "", "A/B position, an expression pedal (CC 0..127)"),
    ("section", "stem_drums", "F1", "drums stem in/out"),
    ("section", "stem_layers", "F2", "layers stem in/out"),
    ("section", "stem_chops", "F3", "chops stem in/out"),
    ("section", "stem_events", "F4", "events stem in/out"),
]
# ids that take a value (0..1) instead of firing; only a CC can drive them
CONTINUOUS = {"ab"}
KEYMAP_GROUPS = {"show": "picture", "tape": "tape", "clock": "clock",
                 "section": "section engine"}


def default_midi_map():
    """The bindings before anyone learns their own: one note per button
    action, from C1 (36) up in table order, on channel 1, which is what a
    pad or a keyboard sends; the continuous actions on CC 11 (expression),
    12, ... on channel 1. Any controller overwrites these by MIDI learn."""
    out, note, cc = {}, 36, 11
    for _g, ident, _k, _w in KEYMAP:
        if ident in CONTINUOUS:
            out[ident] = format_binding("cc", 1, cc)
            cc += 1
        else:
            out[ident] = format_binding("note", 1, note)
            note += 1
    return out

# source grade: colour correction of the picture itself (the VHS input above
# all), applied before every effect and kept in "video only" mode.
# key, label, (min, max, step, neutral), tooltip
VIDEO_GRADE = [
    ("exposure", "exposure", (0.25, 3.0, 0.05, 1.0), "Overall light: multiplies the picture"),
    ("black", "black", (-0.25, 0.25, 0.01, 0.0), "Black level: negative pulls VHS's grey "
                                                  "blacks down, positive lifts them"),
    ("gamma", "gamma", (0.4, 2.5, 0.05, 1.0), "Mid-tones: above 1 brightens the middle "
                                              "without touching black or white"),
    ("contrast", "contrast", (0.25, 2.5, 0.05, 1.0), "Contrast around mid grey"),
    ("saturation", "saturation", (0.0, 2.5, 0.05, 1.0), "Colour strength; 0 is black and white"),
    ("warmth", "warmth", (-1.0, 1.0, 0.02, 0.0), "Colour temperature: positive warmer "
                                                 "(amber), negative cooler (blue)"),
    ("tint", "tint", (-1.0, 1.0, 0.02, 0.0), "Green (negative) to magenta (positive)"),
    ("react", "react", (0.0, 1.0, 0.05, 0.0), "How much the grade breathes with the music, "
                                              "very subtly: exposure and saturation with "
                                              "loudness, warmth toward the root note's colour"),
]

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
    ("reverse", "reverse", 0.3, "Rewind: on a downbeat, with probability = "
                                "strength, the video runs backwards for the "
                                "length set next to it at a random speed "
                                "(1x to 3x), then carries on forwards from "
                                "there; a live video input fast-forwards "
                                "back to live. Every clean flash jogs the "
                                "same way. A cymbal swell also rewinds "
                                "faster and faster into the drop. 0 = "
                                "never, 1 = every bar"),
    ("vign", "vignette", 0.5, "Dark corners"),
    ("words", "words", 0.25, "Word cloud: now and then a 4-bar phrase fills the "
                             "screen with random words in the fonts from fonts/. "
                             "Strength = how often"),
    ("kaleido", "kaleido", 0.15, "Kaleidoscope with 4, 6 or 8 mirrored "
                                 "segments, upright. Strength = how often a "
                                 "section gets it"),
    ("clean", "clean", 0.3, "Clean video: on a downbeat, with probability = "
                            "strength, a 1/16-note flash of the video as it "
                            "is, with no effects, no shader layer and no "
                            "words. 0 = never, 1 = every bar"),
]
NOTE_HUES = {n: i / 12 for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}
# these follow the music directly and are not part of the random per-section look
# glitch band height ranges as fractions of the frame, one per section mode:
# thin sharp lines, medium bands, anything up to big slabs
GLITCH_MODES = [(0.004, 0.01), (0.04, 0.14), (0.04, 0.38)]

# what the picture follows in a song built from a video's own audio: the
# groups of timeline events, in priority order (a hit wins over a texture).
# "varies" re-rolls that order per section and per 4-bar phrase, so the
# picture is led by the textures for a while, then the chops, then the drums
FOLLOW_GROUPS = {"off": (), "textures": ("textures",),
                 "textures and chops": ("chops", "textures"),
                 "everything": ("hits", "chops", "textures"),
                 "varies": ("hits", "chops", "textures")}
# varies: how often each group gets to lead (drums are sounding almost all
# the time, so they would win every phrase on an even roll)
FOLLOW_LEAD_WEIGHTS = {"hits": 1.0, "chops": 1.4, "textures": 1.6}
EVENT_GROUP = {"kick": "hits", "snare": "hits", "hihat": "hits", "ohat": "hits",
               "ride": "hits", "glitch": "hits", "chop": "chops",
               "texture": "textures", "swell": "textures",
               "collage": "textures", "oneshot": "textures"}


class FollowPlan:
    """The render's sample timeline (meta["events"]) reduced to the events
    that came from a video: for any playhead position, the event whose
    frames should be on screen. Rows are (start ms, video, source offset s,
    duration ms, rate), per group, sorted by start."""

    def __init__(self, events, sources):
        self.lists = {g: ([], []) for g in ("hits", "chops", "textures")}
        ok = {}
        for t, src, off, dur, rate, role in events:
            video = sources.get(src)
            group = EVENT_GROUP.get(role)
            if video is None or group is None:
                continue
            if video not in ok:
                ok[video] = Path(video).is_file()
            if not ok[video]:
                continue
            ts, rows = self.lists[group]
            ts.append(t * 1000.0)
            rows.append((t * 1000.0, Path(video), float(off), dur * 1000.0,
                         float(rate)))
        self.n = sum(len(rows) for _, rows in self.lists.values())

    def pick(self, pos_ms, groups):
        """The latest event of the first group that is sounding at pos_ms."""
        for g in groups:
            ts, rows = self.lists[g]
            i = bisect.bisect_right(ts, pos_ms) - 1
            if i >= 0 and pos_ms < rows[i][0] + rows[i][3]:
                return rows[i]
        return None


# rewind lengths in bars; None = a random one per bar
REVERSE_LENGTHS = {"1/8 bar": 1 / 8, "1/4 bar": 1 / 4, "1/2 bar": 1 / 2,
                   "1 bar": 1.0, "2 bars": 2.0, "random": None}

LOOK_EXEMPT = {"color", "pump", "flash", "pixel", "kaleido", "words", "reverse",
               "clean"}


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
        self._live = None           # Frame the image was made from
        self.state_fn = None
        self.fx = VideoFX()
        self.override = None       # RAM frame from the section engine (GL only shows it)
        self.grade = {}            # source grade (GL only applies it)
        self._last_fx = 0.0
        self._zoom = 0.0
        self._frozen = None         # captured frame while a beat-repeat runs
        self.child = child
        self.current = None         # path of the video now looping
        self.input = None           # live capture device while video in is on
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(child)
        self.source = None
        if video_files():
            self.source = VideoSource(parent=self)   # decodes on its own thread
            self.source.frameChanged.connect(self._on_frame)
            if DEFAULT_VIDEO.exists():
                self.set_video(DEFAULT_VIDEO)   # always the stock clip at start
            else:
                self.pick_random()

    def set_input(self, device):
        """Live video in: show a V4L2 capture device (`/dev/videoN`) instead
        of the clips, and ignore clip changes until `None` switches back."""
        if self.source is None:
            return
        self.input = device
        if device:
            self.source.open(device)
        elif self.current is not None:
            self.source.open(self.current, random_start=True)

    def set_video(self, path, random_start=False):
        if self.source is None or self.input:
            return None
        self.current = Path(path)
        self.source.open(self.current, random_start)
        return self.current

    def pick_random(self):
        """Loop a random video from videos/ for a song: never the stock clip
        (unless it is the only one), and a different one than now if
        possible. Rescans the folder, so new files are picked up without a
        restart."""
        files = list(self.videos_fn())
        if self.source is None or not files:
            return None
        files = [f for f in files if f != DEFAULT_VIDEO] or files
        pool = [f for f in files if f != self.current] or files
        return self.set_video(random.choice(pool), random_start=True)

    def _on_frame(self):
        fr = self.source.latest()
        if fr is None:
            return
        st = self.state_fn() if self.state_fn else None
        rev = st.get("reverse") if st else None
        if st and "speed" in st:                # the song's sample timeline
            self.source.set_speed(float(st["speed"]))
        else:
            self.source.set_speed(-float(rev) if rev else 1.0)   # a real rewind
        live = fr.img
        if st and any(k not in ("music", "clean") for k in st):
            now = time.monotonic()
            if now - self._last_fx < FX_INTERVAL:
                return              # throttle: keep showing the last frame
            self._last_fx = now
            self._zoom = st.get("zoom", 0.0)
            if st.get("repeat") or st.get("freeze"):
                # beat-repeat: hold the frame the roll started on, like the
                # audio holds its slice; a pre-drop gap holds it still
                if self._frozen is None:
                    self._frozen = fr
                src = self._frozen.img
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
        self._live = fr                 # the QImage borrows fr's buffer
        self.update()

    def set_shader(self, path):          # generative layer: GL backdrop only
        self.shader_path = Path(path) if path else None    # remembered for the log

    def jump(self):
        """Seek the backdrop video to a random position (no-op without video)."""
        if self.source is not None and not self.input:
            self.source.jump()

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
        self._reverse_bar = None    # (section, bar) last rolled for a rewind
        self._reverse_on = False    # this bar opens with a rewind
        self._reverse_ms = 0.0      # and for how long
        self._reverse_speed = 1.0   # and how fast
        self._follow = None         # FollowPlan of the playing song, if any
        self._pos_last, self._pos_t = -1, 0.0   # playhead interpolation
        self._follow_cur = None     # the timeline event now on screen
        self._follow_key = None     # (section, phrase) the lead was rolled for
        self._follow_order = FOLLOW_GROUPS["everything"]   # varies: this phrase
        self._follow_seek_t = 0.0   # when the picture was last seeked to it
        self._words_slot = None     # half-beat slot of the cloud on screen
        self._clean_bar = None      # (section, bar) last rolled for clean video
        self._clean_on = False      # this bar shows the video with no effects
        self._song_word = None      # the word in the playing render's file name
        self._song_facts = []       # (key, value) pairs from the render's JSON
        self.font_families = load_fonts()
        self._kaleido = 0
        self._mono = (0, 0.5)
        self._pix_seed = 0.0
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self._hide_ui)
        # live mode: an audio input and a tap-tempo clock stand in for the
        # song's loudness envelope and section map
        self.live = LiveInput(self)
        self.tempo = TapTempo()
        self._live_timer = QTimer(self)             # level meter refresh
        self._live_timer.timeout.connect(self._update_live_meter)
        self.through = None                         # LiveAudio while audio through is on
        self.takes = TakeStore(parent=self)         # moments of the show, in RAM
        # the live section engine: material carved from the takes, sections
        # rendered on a thread, played by the audio thread's SectionPlayer
        self._perf_material = None                  # (key, Material)
        self._perf_ready = None                     # (key, Section, stems48) pre-rendered
        self._perf_busy = None                      # ("section"|"shot", key) rendering now
        self._perf_result = None                    # set by the render thread
        self._perf_want = False                     # play the next result as soon as it lands
        self._perf_playing = None                   # Section on the player
        self._perf_timer = QTimer(self)
        self._perf_timer.timeout.connect(self._poll_perform)
        self._perf_timer.start(20)
        self.takes.changed.connect(self._takes_changed)
        self._take_timer = QTimer(self)             # limit check + status while recording
        self._take_timer.timeout.connect(self._poll_take)
        self._through_timer = QTimer(self)          # feeds the analyzer, status line
        self._through_timer.timeout.connect(self._poll_through)
        self._through_ticks = 0
        self._media_devices = QMediaDevices(self)
        self._media_devices.audioInputsChanged.connect(self._refresh_live_devices)

        # ---------- parameters panel ----------
        self.settings = QSettings("gacha", "gacha")
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
        tabs.addTab(self._scrolling(self._build_mixer()), "Mixer")
        # the lists pane is a quarter of the interface: pages scroll rather
        # than force the pane wide
        for i in range(tabs.count()):
            if tabs.tabText(i) in ("Samples", "Videos"):
                tabs.widget(i).setMinimumWidth(1)
        right_min = 300

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
        self.vol = QSlider(Qt.Horizontal, maximumWidth=100)
        self.vol.setRange(0, 100)
        self.vol.setValue(90)
        self.vol.setToolTip("Volume of songs, and of the audio through")
        self.vol.valueChanged.connect(self._set_volume)
        vol = self.vol

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
        right_w.setMinimumWidth(right_min)

        split = QSplitter()
        split.addWidget(left_w)
        split.addWidget(right_w)
        left_w.setMinimumWidth(560)      # four-spinbox rows need the room
        # parameters 75 %, the file lists 25 %; the ratio survives resizes
        # and follows the handle when it is dragged
        self.split = split
        self._split_ratio = 0.75
        split.splitterMoved.connect(self._split_dragged)
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
        # the performer's actions, by id: the keys in KEYMAP and the MIDI
        # bindings both land here. Application shortcuts fire exactly once
        # per press, whatever has focus, even with the interface hidden.
        self._prev_shader_choice = "random"
        self._shader_user_off = False     # off by your own hand: songs leave it off
        self.shader.activated.connect(
            lambda _i: setattr(self, "_shader_user_off",
                               self.shader.currentText() == "off"))
        self.actions = {
            "fullscreen": self.toggle_fullscreen,
            "video_only": self.video_plain.toggle,
            "next_fx": self.next_video_fx,
            "shader": self.toggle_shader,
            "shader_off": lambda: self.select_shader_key(0),
            "through": self.through_on.toggle,
            "record": self.toggle_take,
            "tap": self.tap_tempo,
            "downbeat": self.live_downbeat,
            "live": self.live_on.toggle,
            "drop": self.live_drop_key,
            "next": self.perform_next,
            "stop": self.perform_stop,
            "event": self.perform_event,
            "ab_down": lambda: self.perf_ab.setValue(self.perf_ab.value() - 10),
            "ab_up": lambda: self.perf_ab.setValue(self.perf_ab.value() + 10),
            "ab": lambda x: self.perf_ab.setValue(round(x * 100)),
        }
        for n in range(1, 10):
            self.actions[f"shader_{n}"] = lambda n=n: self.select_shader_key(n)
        for stem in STEMS:
            self.actions[f"stem_{stem}"] = self.perf_stems[stem].toggle
        self.shortcuts = []
        for _g, ident, key, _w in KEYMAP:
            if not key or ident in CONTINUOUS:
                continue
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(self.actions[ident])
            self.shortcuts.append(sc)
        self._midi_setup()

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
        tabs.addTab(self._tab_perform(), "Perform")
        tabs.addTab(self._tab_controls(), "Controls")
        return tabs

    def _split_dragged(self, *_):
        a, b = self.split.sizes()
        if a + b > 0:
            self._split_ratio = a / (a + b)

    def _apply_split(self):
        w = self.split.width()
        if w > 0:
            self.split.blockSignals(True)
            self.split.setSizes([int(w * self._split_ratio), int(w * (1 - self._split_ratio))])
            self.split.blockSignals(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "split"):
            self._apply_split()

    def _tab_controls(self):
        """Every action, grouped, with its key and its MIDI binding. Click
        the MIDI cell of a row and the next message from a ticked controller
        becomes that row's binding (MIDI learn); Delete clears the row."""
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        self.midi_ports = QListWidget()
        self.midi_ports.setToolTip(
            "MIDI inputs. Tick every controller you play from: a foot "
            "controller and a pad at once is fine, they all feed the same "
            "map. A port that is not plugged in shows greyed and opens by "
            "itself when it appears.")
        self.midi_ports.setMaximumHeight(72)
        self.midi_ports.itemChanged.connect(self._midi_ports_changed)
        rescan = QPushButton("rescan")
        rescan.setToolTip("Look for controllers plugged in since launch "
                          "(done every 2 s anyway)")
        rescan.clicked.connect(self._midi_refresh_ports)
        self.midi_last = QLabel("no MIDI yet" if MIDI_AVAILABLE else
                                "MIDI off: pip install mido python-rtmidi")
        self.midi_last.setProperty("role", "sub")
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.midi_ports, stretch=1)
        side = QVBoxLayout()
        side.addWidget(rescan)
        side.addStretch(1)
        top.addLayout(side)
        outer.addWidget(QLabel("MIDI inputs"))
        outer.addLayout(top)
        outer.addWidget(self.midi_last)

        tree = QTreeWidget()
        tree.setColumnCount(4)
        tree.setHeaderLabels(["key", "action", "what it does", "MIDI"])
        tree.setRootIsDecorated(True)
        tree.setUniformRowHeights(True)
        tree.setToolTip("The performer's actions. Click a row's MIDI cell, "
                        "then press the switch, hit the pad or move the "
                        "pedal that should do it. Delete clears the row.")
        groups = {}
        self.midi_items = {}
        for group, ident, key, what in KEYMAP:
            if group not in groups:
                g = QTreeWidgetItem([KEYMAP_GROUPS.get(group, group), "", "", ""])
                g.setFlags(g.flags() & ~Qt.ItemIsSelectable)
                tree.addTopLevelItem(g)
                groups[group] = g
            item = QTreeWidgetItem([key, ident, what, ""])
            item.setData(0, Qt.UserRole, ident)
            groups[group].addChild(item)
            self.midi_items[ident] = item
        tree.expandAll()
        for c in range(3):
            tree.resizeColumnToContents(c)
        tree.itemClicked.connect(self._midi_tree_clicked)
        for key in (Qt.Key_Delete, Qt.Key_Backspace):
            sc = QShortcut(QKeySequence(key), tree)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(self._midi_clear_selected)
        self.controls_tree = tree
        tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tree.setMinimumHeight(240)          # the tab area gives it the rest
        outer.addWidget(tree, stretch=1)
        clear_btn = QPushButton("clear binding")
        clear_btn.setToolTip("Forget the MIDI binding of the selected row")
        clear_btn.clicked.connect(self._midi_clear_selected)
        defaults_btn = QPushButton("defaults")
        defaults_btn.setToolTip("Back to the stock map: notes from C1 up, "
                                "one per button, CC 11 for A/B, channel 1")
        defaults_btn.clicked.connect(self._midi_defaults)
        hint = QLabel("none of these bring the hidden interface back; a button "
                      "fires on note on, program change or a CC going above 63")
        hint.setProperty("role", "sub")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(hint, stretch=1)
        row.addWidget(clear_btn)
        row.addWidget(defaults_btn)
        outer.addLayout(row)
        w = QWidget()
        w.setLayout(outer)
        return w

    # ---------- MIDI ----------
    def _midi_setup(self):
        """Load the map and the ticked ports, open them, start watching for
        controllers plugged in later. Called once the actions exist."""
        self.midi = MidiIn(self)
        self.midi.message.connect(self._on_midi)
        self._midi_learn = None                 # ident being learnt, or None
        self._midi_cc_last = {}                 # (ch, cc) -> last value, for edges
        self.midi_map = default_midi_map()
        saved = self.settings.value("midi/map", "")
        if saved:
            try:
                for ident, text in json.loads(saved).items():
                    if ident in self.midi_map and (text == "" or parse_binding(text)):
                        self.midi_map[ident] = text
            except (ValueError, AttributeError):
                pass
        ticked = self.settings.value("midi/ports", None)
        self._midi_ticked = set(ticked) if ticked is not None else None   # None: first run
        self._midi_refresh_tree()
        self._midi_refresh_ports()
        self._midi_timer = QTimer(self)
        self._midi_timer.timeout.connect(self._midi_refresh_ports)
        self._midi_timer.start(2000)

    @staticmethod
    def _midi_hw(label):
        return "Midi Through" not in label

    def _midi_refresh_ports(self):
        """Rescan the inputs, keep the ticks, open what is ticked and there."""
        if not MIDI_AVAILABLE:
            return
        ports = midi_inputs()
        present = [lab for lab, _ in ports]
        if self._midi_ticked is None:           # first run: every real controller
            self._midi_ticked = {lab for lab in present if self._midi_hw(lab)}
        labels = list(present) + sorted(self._midi_ticked - set(present))
        shown = [self.midi_ports.item(i).text() for i in range(self.midi_ports.count())]
        if shown != labels:
            self.midi_ports.blockSignals(True)
            self.midi_ports.clear()
            for lab in labels:
                it = QListWidgetItem(lab)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if lab in self._midi_ticked else Qt.Unchecked)
                if lab not in present:
                    it.setForeground(QColor(242, 233, 229, 90))
                    it.setToolTip("not plugged in")
                self.midi_ports.addItem(it)
            self.midi_ports.blockSignals(False)
        self._midi_open(ports)

    def _midi_ports_changed(self, item):
        lab = item.text()
        if item.checkState() == Qt.Checked:
            self._midi_ticked.add(lab)
        else:
            self._midi_ticked.discard(lab)
        self.settings.setValue("midi/ports", sorted(self._midi_ticked))
        self._midi_open()

    def _midi_open(self, ports=None):
        before = set(self.midi.open_labels)
        errors = self.midi.open(self._midi_ticked, ports)
        for lab, err in errors.items():
            if lab not in getattr(self, "_midi_errored", set()):
                self.log.appendPlainText(f"midi: cannot open {lab}: {err}")
        self._midi_errored = set(errors)
        after = set(self.midi.open_labels)
        for lab in sorted(after - before):
            self.log.appendPlainText(f"midi: listening to {lab}")
        for lab in sorted(before - after):
            self.log.appendPlainText(f"midi: {lab} closed")

    def _midi_refresh_tree(self):
        rev = {}
        for ident, item in self.midi_items.items():
            text = self.midi_map.get(ident, "")
            if self._midi_learn == ident:
                item.setText(3, "learning: press or move a control...")
                item.setForeground(3, QColor(255, 120, 90))
            else:
                item.setText(3, text)
                item.setForeground(3, QColor(242, 233, 229, 230 if text else 90))
            b = parse_binding(text)
            if b:
                rev[b] = ident
        self._midi_rev = rev
        self.controls_tree.resizeColumnToContents(3)

    def _midi_save(self):
        self.settings.setValue("midi/map", json.dumps(self.midi_map))

    def _midi_tree_clicked(self, item, column):
        ident = item.data(0, Qt.UserRole)
        if ident is None:
            return
        if column != 3:
            if self._midi_learn is not None:    # any other click ends the learn
                self._midi_learn = None
                self._midi_refresh_tree()
            return
        self._midi_learn = None if self._midi_learn == ident else ident
        if self._midi_learn:
            what = "move a knob or a pedal" if ident in CONTINUOUS \
                else "press the switch or hit the pad"
            self.midi_last.setText(f"learning {ident}: {what}; click again to cancel")
        self._midi_refresh_tree()

    def _midi_clear_selected(self):
        item = self.controls_tree.currentItem()
        ident = item.data(0, Qt.UserRole) if item is not None else None
        if ident is None:
            return
        self.midi_map[ident] = ""
        self._midi_learn = None
        self._midi_save()
        self._midi_refresh_tree()
        self.log.appendPlainText(f"midi: {ident} unbound")

    def _midi_defaults(self):
        self.midi_map = default_midi_map()
        self._midi_learn = None
        self._midi_save()
        self._midi_refresh_tree()
        self.log.appendPlainText("midi: default map")

    def _on_midi(self, port, kind, ch, num, val):
        """Every message from every ticked controller, on the GUI thread."""
        text = format_binding(kind, ch, num)
        self.midi_last.setText(f"last: {text} = {val} from {port}")
        if self._midi_learn is not None:
            ident = self._midi_learn
            if ident in CONTINUOUS and kind != "cc":
                return                          # a pedal, not a switch
            other = self._midi_rev.get((kind, ch, num))
            if other and other != ident:
                self.midi_map[other] = ""       # one control, one action
                self.log.appendPlainText(f"midi: {text} taken from {other}")
            self.midi_map[ident] = text
            self._midi_learn = None
            self._midi_save()
            self._midi_refresh_tree()
            self.midi_last.setText(f"{ident} <- {text} from {port}")
            self.log.appendPlainText(f"midi: {ident} <- {text} ({port})")
            if kind == "cc":
                self._midi_cc_last[(ch, num)] = val   # no trigger from the learn press
            return
        ident = self._midi_rev.get((kind, ch, num))
        if ident is None:
            return
        if ident in CONTINUOUS:
            self.actions[ident](val / 127.0)
            return
        if kind == "cc":                        # a switch on a CC: rising edge only
            prev = self._midi_cc_last.get((ch, num), 0)
            self._midi_cc_last[(ch, num)] = val
            if not (val >= 64 and prev < 64):
                return
        self.actions[ident]()

    def _tab_perform(self):
        """The live section engine: what the feet will drive on stage."""
        form = QFormLayout()
        self.perf_material = QComboBox()
        self.perf_material.addItems(["takes, else ticked samples", "takes only",
                                     "ticked samples"])
        self.perf_material.setToolTip(
            "What the sections are built from. The takes recorded during the "
            "show are the point; the ticked sample sets stand in for rehearsal "
            "when there are none (nothing ticked = every sample).")
        self.perf_kind = QComboBox()
        self.perf_kind.addItems(["random"] + list(KINDS))
        self.perf_kind.setToolTip(
            "Shape of the next section. groove: full kit, one to three layers, "
            "maybe chops. break: one or two drum voices under more layers. "
            "build: the voices stack up bar by bar. sparse: one voice, one layer.")
        form.addRow("Material", self._row([("", self.perf_material), ("next kind", self.perf_kind)]))
        next_btn = QPushButton("next section (N)")
        next_btn.setMinimumHeight(30)
        next_btn.setToolTip(
            "A new 8-bar section from the material at the live tempo, on the "
            "next bar. One is rendered ahead while the current one loops, so "
            "this is instant; the very first waits about a second.")
        next_btn.clicked.connect(self.perform_next)
        stop_btn = QPushButton("stop (Shift+N)")
        stop_btn.setToolTip("Silence the section on the next bar; the tape stays")
        stop_btn.clicked.connect(self.perform_stop)
        event_btn = QPushButton("event (E)")
        event_btn.setToolTip("Fire a one-shot cut from the material on the next beat")
        event_btn.clicked.connect(self.perform_event)
        form.addRow("Section", self._row([("", next_btn), ("", stop_btn), ("", event_btn)]))
        self.perf_stems = {}
        keys = {"drums": "F1", "layers": "F2", "chops": "F3", "events": "F4"}
        pairs = []
        for stem in STEMS:
            cb = QCheckBox(f"{stem} ({keys[stem]})")
            cb.setChecked(True)
            cb.toggled.connect(lambda on, st=stem: self._perf_stem(st, on))
            self.perf_stems[stem] = cb
            pairs.append(("", cb))
        form.addRow("Stems", self._row(pairs))
        self.perf_ab = QSlider(Qt.Horizontal)
        self.perf_ab.setRange(0, 100)
        self.perf_ab.setValue(0)
        self.perf_ab.setToolTip("A/B: 0 = the tape through only, 100 = the section "
                                "only, equal power in between. Keys [ and ] step "
                                "by 10; a MIDI expression pedal drives it "
                                "through the Controls tab.")
        self.perf_ab.valueChanged.connect(self._perf_ab_changed)
        self.perf_ab_lbl = QLabel("A 100 %  ·  B 0 %")
        self.perf_ab_lbl.setProperty("role", "sub")
        form.addRow("A/B", self._row([("", self.perf_ab), ("", self.perf_ab_lbl)]))
        self.perf_picture = QComboBox()
        self.perf_picture.addItems(["auto", "take", "live"])
        self.perf_picture.setToolTip(
            "What the screen shows while a section plays: take = the frames "
            "the sounds were cut from (needs takes with video), live = the "
            "tape as it comes in, auto = the take frames once A/B passes 50 %.")
        form.addRow("Picture", self._row([("", self.perf_picture)]))
        self.perf_status = QLabel("no section")
        self.perf_status.setProperty("role", "sub")
        self.perf_status.setWordWrap(True)
        form.addRow("", self.perf_status)
        hint = QLabel("needs audio through on (A): the section is mixed into that "
                      "path. Tempo = the live clock (tap T, or the bpm box).")
        hint.setProperty("role", "sub")
        form.addRow("", hint)
        return self._wrap(form)

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
        self.video_audio = QCheckBox("use the audio of the ticked videos as "
                                     "sample material")
        self.video_audio.setToolTip(
            "The audio tracks of the videos ticked in the Videos tab (all of "
            "them when none is ticked) join the ticked samples for the next "
            "render, so drums, textures and chops are cut out of the film's "
            "own sound. Works with no samples ticked at all. Videos without "
            "an audio track are skipped. Ticking this sets Random start "
            "(Layers & FX) to 'all' if it was off, so cuts land anywhere in "
            "the film rather than at its first seconds.")
        self.video_audio.toggled.connect(self._video_audio_toggled)
        form.addRow("Video audio", self.video_audio)
        self.video_follow = QComboBox()
        self.video_follow.addItems(list(FOLLOW_GROUPS))
        self.video_follow.setCurrentText("varies")
        self.video_follow.setToolTip(
            "While a song built from video audio plays, the picture shows "
            "the frames the sound on the speakers was cut from: a hit cuts "
            "to the moment it was carved at, a texture runs (backwards, "
            "stretched) along with its loop, a chop jumps slice by slice. "
            "Pick which layers the picture follows. everything: a hit wins "
            "over a chop, a chop over a texture (drums nearly always). "
            "varies: which layer leads is re-rolled at every section and "
            "every 4 bars, textures and chops a little more often than the "
            "drums. Off: the backdrop behaves as usual.")
        form.addRow("", self._row([("picture follows", self.video_follow)]))
        self.video_in = QComboBox()
        self.video_in.setToolTip(
            "Live video in: a V4L2 capture device (a webcam, or a USB "
            "composite grabber with a VHS deck on it) replaces the clips as "
            "the backdrop. Everything else runs on it as usual: the effects, "
            "the shader layer, the words, and rewinds jog the live picture "
            "through the last seconds captured. Clip changes are held off "
            "until this is back to off.")
        self.video_in.currentIndexChanged.connect(self._video_in_changed)
        form.addRow("Video input", self._row([("device", self.video_in)]))

        self.live_on = QCheckBox("live: an audio input drives the effects "
                                 "instead of a song (L)")
        self.live_on.setToolTip(
            "VJ mode. Loudness comes from the chosen input (a mic, a line "
            "in, or on PipeWire/Pulse a monitor of what the machine plays) "
            "and the beat from tap tempo: press T on the beat four times, "
            "starting on the one. D restarts the bar on the current tempo. "
            "Every 8 bars count as a section for the look, shader and clip "
            "changes. Turning a song on switches live off.")
        self.live_on.toggled.connect(self._toggle_live)
        form.addRow("Audio input", self.live_on)
        self.live_device = QComboBox()
        self.live_device.setToolTip("Capture device; the list follows what "
                                    "the system offers")
        self.live_level = QProgressBar()
        self.live_level.setRange(0, 100)
        self.live_level.setTextVisible(False)
        self.live_level.setFixedHeight(10)
        self.live_level.setToolTip("Normalised loudness, what the effects see")
        form.addRow("", self._row([("device", self.live_device),
                                   ("level", self.live_level)]))
        self.through_on = QCheckBox("through: the input plays out of an output "
                                    "device, via the live FX rack (A)")
        self.through_on.setToolTip(
            "The live path of the show: the chosen input (the VHS grabber) "
            "goes to the chosen output in realtime, through a rack of effects "
            "that is empty for now, so what comes out is the deck's own sound. "
            "Two 256-frame buffers make about 11 ms plus the devices' own "
            "latency. Only ALSA hardware devices are listed; one that PipeWire "
            "or another app is playing through cannot be opened. With live "
            "mode on, the analysis listens to this stream. The main volume "
            "slider sets its level. Key A toggles it.")
        self.through_on.toggled.connect(self._toggle_through)
        form.addRow("Audio through", self.through_on)
        self.through_in = QComboBox()
        self.through_in.setToolTip("Hardware input: the grabber's audio")
        self.through_out = QComboBox()
        self.through_out.setToolTip("Hardware output: the PA / the desk")
        self.through_block = QComboBox()
        self.through_block.addItems(["256", "512", "1024"])
        self.through_block.setToolTip("Frames per buffer: 256 = 5.3 ms each "
                                      "way; raise it if drops appear")
        self.through_sync = QSpinBox(minimum=0, maximum=2000, singleStep=5,
                                     suffix=" ms")
        self.through_sync.setValue(0)
        self.through_sync.setToolTip(
            "Lip sync: hold the sound back by this much so it meets the "
            "picture, which arrives late through the grabber, the decoder "
            "and the display. Adjust while watching a tape: mouths, cuts, "
            "hits. Changes live without a click. Typical: 60 to 150 ms.")
        self.through_sync.valueChanged.connect(self._through_sync_changed)
        form.addRow("", self._row([("in", self.through_in),
                                   ("out", self.through_out),
                                   ("block", self.through_block),
                                   ("sync", self.through_sync)]))
        self.through_status = QLabel("")
        self.through_status.setProperty("role", "sub")
        form.addRow("", self.through_status)

        self.rec_btn = QPushButton("● record (R)")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setMinimumHeight(30)
        self.rec_btn.setToolTip(
            "A take: the live sound and picture from a few seconds before "
            "this press until the next one, kept in RAM. The takes together "
            "are the sample bank the engine builds sections from at the end "
            "of the piece; short ones are fine. Stops by itself at the limit; "
            "the next press then starts a new take. Needs audio through and/or "
            "the video input on. Key R.")
        self.rec_btn.clicked.connect(self.toggle_take)
        self.take_pre = QDoubleSpinBox(minimum=0.0, maximum=12.0, singleStep=1.0,
                                       decimals=0, suffix=" s")
        self.take_pre.setValue(10.0)
        self.take_pre.setToolTip("Pre-roll: how many seconds before the press "
                                 "each take starts with")
        self.take_limit = QDoubleSpinBox(minimum=5.0, maximum=300.0, singleStep=5.0,
                                         decimals=0, suffix=" s")
        self.take_limit.setValue(60.0)
        self.take_limit.setToolTip("A take stops by itself after this long")
        save_btn = QPushButton("save takes")
        save_btn.setToolTip("Write every take as wav + mov into takes/<date>/ "
                            "for rehearsal and offline work")
        save_btn.clicked.connect(self._save_takes)
        clear_btn = QPushButton("clear")
        clear_btn.setToolTip("Forget every take")
        clear_btn.clicked.connect(self.takes.clear)
        form.addRow("Takes", self._row([("", self.rec_btn), ("pre-roll", self.take_pre),
                                        ("limit", self.take_limit), ("", save_btn),
                                        ("", clear_btn)]))
        self.take_status = QLabel("no takes")
        self.take_status.setProperty("role", "sub")
        form.addRow("", self.take_status)
        self._refresh_through_devices()
        self.live_bpm = QDoubleSpinBox(minimum=40.0, maximum=240.0,
                                       singleStep=1.0, decimals=1, suffix=" bpm")
        self.live_bpm.setValue(self.tempo.bpm)
        self.live_bpm.setToolTip("Tempo of the live clock; tapping sets it")
        self.live_bpm.valueChanged.connect(self._live_bpm_edited)
        tap_btn = QPushButton("Tap (T)")
        tap_btn.setToolTip("Tap on the beat, four or more times, the first "
                           "one on the downbeat")
        tap_btn.clicked.connect(self.tap_tempo)
        down_btn = QPushButton("Downbeat (D)")
        down_btn.setToolTip("Restart the bar now, keeping the tempo")
        down_btn.clicked.connect(self.live_downbeat)
        form.addRow("", self._row([("tempo", self.live_bpm), ("", tap_btn),
                                   ("", down_btn)]))
        self.live_auto = QCheckBox("auto: tempo, beat and root note from the audio")
        self.live_auto.setChecked(True)
        self.live_auto.setToolTip(
            "Listens along: the tempo and beat phase from the onsets (kicks "
            "first), the root note from a running chroma for the tint. Needs "
            "about 4 s of music. Tapping T switches this off and follows "
            "your taps instead; X marks a drop by hand and restarts the bar.")
        self.live_status = QLabel("")
        self.live_status.setProperty("role", "sub")
        form.addRow("", self._row([("", self.live_auto), ("", self.live_status)]))
        self._refresh_live_devices()

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

        self.grade = {}
        for key, label, (lo, hi, step, val), tip in VIDEO_GRADE:
            sp = QDoubleSpinBox(minimum=lo, maximum=hi, singleStep=step, decimals=2)
            sp.setValue(val)
            sp.setToolTip(tip)
            sp.valueChanged.connect(self._apply_grade)
            self.grade[key] = sp
        keys = [k for k, *_ in VIDEO_GRADE]
        labels = {k: lbl for k, lbl, _, _ in VIDEO_GRADE}
        form.addRow("Source grade", self._row([(labels[k], self.grade[k]) for k in keys[:4]]))
        form.addRow("", self._row([(labels[k], self.grade[k]) for k in keys[4:]]))
        grade_hint = QLabel("colour correction of the picture itself, before every "
                            "effect and kept in video-only mode; neutral as set")
        grade_hint.setProperty("role", "sub")
        reset_btn = QPushButton("neutral")
        reset_btn.setToolTip("Back to no correction")
        reset_btn.clicked.connect(self._reset_grade)
        form.addRow("", self._row([("", grade_hint), ("", reset_btn)]))

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
        self.video_plain = QCheckBox("video only: no effects, no shader layer, "
                                     "no words (C)")
        self.video_plain.setToolTip(
            "Show the video as it is, for as long as this is ticked: every "
            "effect below, the shader layer and the words are bypassed. The "
            "picture still follows the song (sample timeline, section jumps, "
            "clip switches, rewinds, the outro fade). Key C toggles it.")
        form.addRow("", self.video_plain)

        self.vfx = {}
        for key, label, default, tip in VIDEO_EFFECTS:
            self.vfx[key] = self._prob_spin(default, tip)
        self.reverse_len = QComboBox()
        self.reverse_len.addItems(list(REVERSE_LENGTHS))
        self.reverse_len.setToolTip("How long each rewind runs, from its "
                                    "downbeat; random picks per bar")
        rows = [("Color", ("color", "tint", "pump", "flash")),
                ("Pixels", ("pixel", "bits", "dither")),
                ("Tone", ("mono", "solar", "edges", "lines")),
                ("Grain", ("grain",)),
                ("Motion", ("zoom", "trails", "glitch", "rgb", "reverse")),
                ("Frame", ("vign", "kaleido", "words", "clean"))]
        labels = {k: lbl for k, lbl, _, _ in VIDEO_EFFECTS}
        for title, keys in rows:
            pairs = [(labels[k], self.vfx[k]) for k in keys]
            if "reverse" in keys:
                pairs.append(("length", self.reverse_len))
            form.addRow(title, self._row(pairs))
        hint = QLabel("strength 0 = off; driven by the song's sections and "
                      "loudness, or by the live input and tap tempo")
        hint.setProperty("role", "sub")
        form.addRow("", hint)
        return self._wrap(form, self._randomize_video)

    def _randomize_video(self):
        self._shuffle(*self.vfx.values(), self.shader_blend, self.reverse_len)
        self.shader_mix.setValue(random.choice([0.3, 0.5, 0.7, 1.0]))
        self.shader.setCurrentText("random")
        self._look_idx = None                  # re-roll the look right away

    def _apply_grade(self, *_):
        """Grade spinboxes -> backdrop, as one dict."""
        if hasattr(self, "backdrop"):
            self.backdrop.grade = {k: sp.value() for k, sp in self.grade.items()}

    def _reset_grade(self):
        for (key, _, (_, _, _, val), _) in VIDEO_GRADE:
            self.grade[key].setValue(val)

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
    @staticmethod
    def _scrolling(widget):
        """A page that scrolls when the pane is narrower than its content."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setWidget(widget)
        return area

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
        video_audio = [str(p) for p in self._song_videos()] \
            if self.video_audio.isChecked() else []
        if not samples and not video_audio:
            self.log.appendPlainText(
                "✗ no samples selected — tick some in the Samples tab, or "
                "use the audio of the videos (Video tab)")
            return
        job = {"count": self.count.value(),
               "duration": self.duration.value(),
               "samples": samples,
               "video_audio": video_audio,
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

    # ---------- live input (VJ mode) ----------
    LIVE_SECTION_BARS = 8       # bars per pseudo-section on the live clock
    # the analyzer's loudness-based drop detector is off: on rendered songs
    # it found 4 drops in 22 and fired falsely every 30-60 s. X marks drops.
    LIVE_AUTO_DROPS = False

    def _live_active(self):
        return self.live_on.isChecked() and self.live.running

    def _refresh_live_devices(self):
        """Fill the device combo, keeping the current pick when it survives."""
        cur = self.live_device.currentText()
        self.live_device.blockSignals(True)
        self.live_device.clear()
        for name, dev in audio_inputs():
            self.live_device.addItem(name, dev)
        if cur:
            i = self.live_device.findText(cur)
            if i >= 0:
                self.live_device.setCurrentIndex(i)
        self.live_device.blockSignals(False)
        cur = self.video_in.currentData()
        self.video_in.blockSignals(True)
        self.video_in.clear()
        self.video_in.addItem("off", None)
        for name, dev in video_inputs():
            self.video_in.addItem(name, dev)
        if cur:
            i = self.video_in.findData(cur)
            if i >= 0:
                self.video_in.setCurrentIndex(i)
        self.video_in.blockSignals(False)
        if hasattr(self, "through_in"):
            self._refresh_through_devices()
        if self.video_in.currentData() != cur:
            self._video_in_changed()             # the device went away

    def _video_audio_toggled(self, on):
        """Video audio on: a soundtrack is one long take, so cuts should
        start anywhere in it, not always at its first seconds."""
        if on and self.random_start.currentText() == "off":
            self.random_start.setCurrentText("all")
            self.log.appendPlainText("video audio: random start set to 'all' "
                                     "so cuts land anywhere in the film")

    def _video_in_changed(self, *_):
        """Video input combo -> backdrop: a capture device, or back to clips."""
        if hasattr(self, "backdrop"):
            self.backdrop.set_input(self.video_in.currentData())

    def _toggle_live(self, on):
        """Live mode on: the song stops and its analysis is dropped, the
        input opens, the clock restarts on a downbeat and the clip is
        re-picked as for a new song. Off: capture closes, songs work again."""
        self._sec_idx = -1
        if on:
            self.player.stop()
            self._sections, self._sec_bounds, self._env = [], [], None
            self._outro, self._song_word = None, None
            if self.through is not None:        # the through path owns the input
                err = self.live.start_external(self.through_in.currentText(),
                                               self.through.sr)
            else:
                err = self.live.start(self.live_device.currentData())
            if err:
                self.log.appendPlainText(f"live: {err}")
                self.live_on.setChecked(False)
                return
            self.tempo.downbeat()
            self._live_drop_t = None        # monotonic time of the last drop
            self._live_switch_t = time.monotonic()   # last clip switch
            self._last_beat_sync = None     # analyzer beat already applied
            self._last_auto_drop = None
            self._pick_song_video()
            self._look_idx = None
            if self.shader.currentText() == "off" and not self._shader_user_off:
                self.shader.setCurrentText(self._prev_shader_choice)
            self._live_timer.start(50)
            self.log.appendPlainText(
                f"live: on, {self.live.device_name}, {self.tempo.bpm:.1f} bpm. "
                "Tap T on the beat, D restarts the bar, L switches off")
            if self.auto_hide.isChecked():
                self._hide_ui()
        else:
            self._live_timer.stop()
            if self.live.running:
                self.live.stop()
                self.log.appendPlainText("live: off")
            self.live_level.setValue(0)
            self._arm_hide()

    # ---------- perform: live sections ----------
    def _perf_material_key(self):
        mode = self.perf_material.currentIndex()
        take_ids = tuple(t.idx for t in self.takes.takes if len(t.audio))
        if take_ids and mode != 2:
            return ("takes", take_ids)
        if mode == 1:
            return None
        files = tuple(str(p) for p in self.samples_tree.checked()) or \
            tuple(sorted(str(p) for p in SAMPLES_DIR.rglob("*") if p.suffix.lower() in AUDIO_EXTS))
        return ("samples", files) if files else None

    def _perf_bank(self, key):
        """Material for `key`, built if needed (call on the render thread)."""
        if self._perf_material and self._perf_material[0] == key:
            return self._perf_material[1]
        if key[0] == "takes":
            bank = self.takes.bank()
        else:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                bank = E.load_samples(list(key[1]))
        mat = Material(bank, seed=random.randrange(1 << 30))
        self._perf_material = (key, mat)
        return mat

    def _perf_key(self):
        kind = self.perf_kind.currentText()
        return (self._perf_material_key(), kind, round(self.tempo.bpm, 1))

    def _perf_render(self, key, kind_choice):
        """Render thread: a section for `key`, stems resampled to the audio
        thread's rate; result lands in _perf_result."""
        try:
            mat = self._perf_bank(key[0])
            if not mat.ok:
                self._perf_result = ("error", "no material with sound (record a take, "
                                              "or tick some samples)")
                return
            kind = random.choice(KINDS) if kind_choice == "random" else kind_choice
            sec = render_section(mat, seed=random.randrange(1 << 30), bpm=key[2],
                                 bars=8, kind=kind)
            sr = self.through.sr if self.through is not None else 48000
            stems = {}
            for name, data in sec.stems.items():
                d = data if sr == E.SR else resample_poly(data, sr // 300, E.SR // 300, axis=0)
                stems[name] = np.ascontiguousarray(d.T.astype(np.float32))
            self._perf_result = ("section", key, sec, stems)
        except Exception as e:                        # noqa: BLE001
            self._perf_result = ("error", f"render failed: {e}")

    def _perf_render_shot(self, key):
        try:
            mat = self._perf_bank(key[0])
            if not mat.ok:
                self._perf_result = ("error", "no material with sound")
                return
            shot = render_oneshot(mat, random.randrange(1 << 30), 60.0 / key[2])
            sr = self.through.sr if self.through is not None else 48000
            clip = shot["clip"] * shot["gain"]
            if sr != E.SR:
                clip = resample_poly(clip, sr // 300, E.SR // 300, axis=0)
            self._perf_result = ("shot", np.ascontiguousarray(clip.T.astype(np.float32)), shot)
        except Exception as e:                        # noqa: BLE001
            self._perf_result = ("error", f"one-shot failed: {e}")

    def _perf_start(self, what, key):
        if self._perf_busy is not None:
            return False
        self._perf_busy = (what, key)
        target = self._perf_render if what == "section" else self._perf_render_shot
        args = (key, self.perf_kind.currentText()) if what == "section" else (key,)
        threading.Thread(target=target, args=args, daemon=True, name="gacha-render").start()
        return True

    def perform_next(self):
        """N: the next section, on the bar. Uses the pre-rendered one when it
        matches the material, kind and tempo; else renders and plays when done."""
        if self.through is None or not self.through.running:
            self.log.appendPlainText("perform: switch audio through on first (A)")
            return
        key = self._perf_key()
        if key[0] is None:
            self.log.appendPlainText("perform: no material (record a take or tick samples)")
            return
        if self._perf_ready is not None and self._perf_ready[0] == key:
            self._perf_play(*self._perf_ready[1:])
            self._perf_ready = None
            self._perf_start("section", key)          # the one after, ahead of time
        else:
            self._perf_want = True
            self._perf_ready = None
            if not self._perf_start("section", key):
                self.perf_status.setText("rendering, the next lands when ready...")

    def _perf_play(self, sec, stems):
        self.through.player.queue(stems, sec.events, sec.bpm, sec.bars)
        self._perf_playing = sec
        self.log.appendPlainText(f"section: {sec.kind} {sec.bpm:.0f} bpm, "
                                 f"{'; '.join(sec.notes)}")

    def perform_stop(self):
        if self.through is not None:
            self.through.player.clear()
        self._perf_playing = None
        self.backdrop.override = None
        self.perf_status.setText("no section")

    def perform_event(self):
        """E: a one-shot from the material, on the next beat."""
        if self.through is None or not self.through.running:
            self.log.appendPlainText("perform: switch audio through on first (A)")
            return
        key = self._perf_key()
        if key[0] is None:
            return
        self._perf_start("shot", key)

    def _perf_stem(self, stem, on):
        if self.through is not None:
            self.through.player.on[stem] = on

    def _perf_ab_changed(self, v):
        self.perf_ab_lbl.setText(f"A {100 - v} %  ·  B {v} %")
        if self.through is not None:
            self.through.ab = v / 100.0

    def _poll_perform(self):
        """20 ms: collect render results, keep the live clock on the section,
        and put the take's frames on screen for the sound that plays."""
        r, self._perf_result = self._perf_result, None
        if r is not None:
            what = r[0]
            self._perf_busy = None
            if what == "error":
                self.log.appendPlainText(f"perform: {r[1]}")
                self._perf_want = False
            elif what == "shot":
                if self.through is not None:
                    self.through.player.fire(r[1])
                    self.log.appendPlainText(f"event <- {r[2]['src']}")
            elif what == "section":
                _, key, sec, stems = r
                if self._perf_want and self.through is not None and key == self._perf_key():
                    self._perf_want = False
                    self._perf_play(sec, stems)
                    self._perf_start("section", key)  # pre-render the one after
                else:
                    self._perf_ready = (key, sec, stems)
        la = self.through
        pl = la.player if la is not None and la.running else None
        if pl is None or not pl.playing:
            if self.backdrop.override is not None:
                self.backdrop.override = None
            if self._perf_playing is not None and pl is None:
                self._perf_playing = None
            if self._perf_busy:
                self.perf_status.setText("rendering...")
            return
        if pl.swapped:                                 # the live clock follows the loop
            pl.swapped = False
            self.tempo.set_bpm(pl.bpm)
            self.tempo.anchor = pl.loop_t0 if pl.loop_t0 else time.monotonic()
            self.live_bpm.blockSignals(True)
            self.live_bpm.setValue(self.tempo.bpm)
            self.live_bpm.blockSignals(False)
        sec = self._perf_playing
        pos = pl.pos_s()
        bar = int(pos * pl.bpm / 60.0 / 4) + 1
        ready = "next ready" if self._perf_ready else ("rendering next..." if self._perf_busy else "")
        if sec is None:                                # stop asked: it lands on the bar
            self.perf_status.setText("stopping on the bar...")
        else:
            self.perf_status.setText(
                f"playing {sec.kind} {pl.bpm:.0f} bpm  ·  bar {bar}/{pl.bars}  ·  {ready}")
        # picture: the frame the sounding sample was cut from
        mode = self.perf_picture.currentText()
        show = mode == "take" or (mode == "auto" and la.ab >= 0.5)
        fr = self._perf_frame(pos) if (show and sec is not None) else None
        self.backdrop.override = fr

    def _perf_frame(self, pos):
        """Frame of the take the loudest-priority event sounding at `pos`
        (seconds into the loop) was cut from, or None (no video in it)."""
        sec = self._perf_playing
        takes = {t.name: t for t in self.takes.takes}
        order = {"kick": 0, "snare": 0, "hihat": 0, "ohat": 0, "ride": 0, "glitch": 0,
                 "chop": 1, "oneshot": 1, "texture": 2}
        best = None
        for t, src, off, dur, rate, role in sec.events:
            if t <= pos < t + dur and src in takes and takes[src].frames:
                pri = order.get(role, 3)
                if best is None or pri < best[0]:
                    best = (pri, t, src, off, rate)
        if best is None:
            return None
        _, t, src, off, rate = best
        return takes[src].frame_at(off + (pos - t) * rate)

    # ---------- takes ----------
    def toggle_take(self):
        """R / the button / a footswitch later: start or stop a take."""
        audio = self.through if self.through is not None and self.through.running else None
        src = getattr(self.backdrop, "source", None)
        video = src if (self.backdrop.input and src is not None and src.is_live) else None
        if not self.takes.recording and audio is None and video is None:
            self.log.appendPlainText("take: nothing to record, switch audio through "
                                     "and/or the video input on")
            self.rec_btn.setChecked(False)
            return
        self.takes.pre_roll = self.take_pre.value()
        self.takes.limit = self.take_limit.value()
        take = self.takes.toggle(audio, video)
        if take is not None:
            self._log_take(take)
        elif self.takes.recording:
            what = " + ".join(w for w, on in (("audio", audio is not None),
                                              ("video", video is not None)) if on)
            self.log.appendPlainText(f"take {len(self.takes.takes) + 1}: recording {what} "
                                     f"(pre-roll {self.takes.pre_roll:.0f} s, "
                                     f"limit {self.takes.limit:.0f} s)")

    def _log_take(self, take):
        self.log.appendPlainText(
            f"{take.name}: {take.seconds:.1f} s, {len(take.frames)} frames, "
            f"{take.nbytes / 1e9:.2f} GB  ·  {len(self.takes.takes)} takes, "
            f"{self.takes.nbytes / 1e9:.1f} GB in RAM"
            + ("  (limit reached)" if self.takes.limit_hit else ""))

    def _takes_changed(self):
        rec = self.takes.recording
        self.rec_btn.setChecked(rec)
        self.rec_btn.setText("■ stop (R)" if rec else "● record (R)")
        if rec:
            self._take_timer.start(100)
        else:
            self._take_timer.stop()
            n = len(self.takes.takes)
            self.take_status.setText(
                "no takes" if not n else
                f"{n} take{'s' if n > 1 else ''}, {sum(t.seconds for t in self.takes.takes):.0f} s, "
                f"{self.takes.nbytes / 1e9:.1f} GB in RAM")

    def _poll_take(self):
        was = self.takes.recording
        self.takes.tick()
        if was and not self.takes.recording:           # the limit stopped it
            self._log_take(self.takes.takes[-1])
            return
        self.take_status.setText(f"● recording {self.takes.elapsed:.1f} s / "
                                 f"{self.takes.limit:.0f} s")

    def _save_takes(self):
        if not self.takes.takes:
            self.log.appendPlainText("takes: nothing to save")
            return
        folder = Path("takes") / time.strftime("%Y%m%d_%H%M%S")
        paths = self.takes.save(folder)
        self.log.appendPlainText(f"takes: wrote {len(paths)} files to {folder}/")

    # ---------- audio through ----------
    def _set_volume(self, v):
        self.audio_out.setVolume(v / 100)
        if self.through is not None:
            self.through.gain = v / 100

    def _through_sync_changed(self, ms):
        if self.through is not None:
            self.through.delay_ms = ms

    def _refresh_through_devices(self):
        """Hardware devices pedalboard can open, the grabber preselected."""
        ins, outs = audio_devices()
        for combo, items, prefer in ((self.through_in, ins, ("MS210x", "USB Video", "Grabber")),
                                     (self.through_out, outs, ())):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for short, full in items:
                combo.addItem(short, full)
            i = combo.findData(cur) if cur else -1
            if i < 0:
                for k, (short, _) in enumerate(items):
                    if any(p in short for p in prefer):
                        i = k
                        break
            combo.setCurrentIndex(max(0, i))
            combo.blockSignals(False)

    def _toggle_through(self, on):
        if on:
            in_name, out_name = self.through_in.currentData(), self.through_out.currentData()
            if not in_name or not out_name:
                self.log.appendPlainText("audio through: pick an input and an output")
                self.through_on.setChecked(False)
                return
            self.through = LiveAudio(in_name, out_name, int(self.through_block.currentText()))
            self.through.gain = self.vol.value() / 100
            self.through.delay_ms = self.through_sync.value()
            self.through.start()
            self.through_status.setText("opening...")
            self._through_ticks = 0
            self._through_timer.start(20)
            self.log.appendPlainText(
                f"audio through: {self.through_in.currentText()} -> "
                f"{self.through_out.currentText()}, {self.through.block} frames")
            if self.live_on.isChecked():          # analysis moves to this stream
                self.live.start_external(self.through_in.currentText(), self.through.sr)
        else:
            self._through_timer.stop()
            if self.through is not None:
                self.through.stop()
                self.through = None
                self.log.appendPlainText("audio through: off")
            self.through_status.setText("")
            if self.live_on.isChecked() and self.live.external:   # back to capture
                err = self.live.start(self.live_device.currentData())
                if err:
                    self.log.appendPlainText(f"live: {err}")

    def _poll_through(self):
        la = self.through
        if la is None:
            return
        if not la.running:
            if la.error:
                self.log.appendPlainText(f"audio through: {la.error}")
                self.through_on.setChecked(False)
            return
        if self._live_active() and self.live.external:
            for block, t in la.drain():
                self.live.feed(block, t)
        self._through_ticks += 1
        if self._through_ticks % 10 == 0:
            s = la.stats()
            db = lambda p: f"{20 * math.log10(max(p, 1e-5)):.0f} dB"
            text = (f"in {db(s['in_peak'])}  out {db(s['out_peak'])}  ·  dsp {s['load']:.0%}  ·  "
                    f"buffers {s['buffer_ms']:.1f} ms  ·  sync {s['delay_ms']:.0f} ms  ·  "
                    f"drops {s['dropped']}  late {s['late']}")
            src = getattr(self.backdrop, "source", None)
            lat = getattr(src, "capture_latency_ms", None) if self.backdrop.input else None
            if lat is not None:
                # what the picture is already late by before paint and display
                text += f"  ·  picture {lat:.0f} ms + paint + display"
            self.through_status.setText(text)

    def _update_live_meter(self):
        self.live_level.setValue(int(100 * self.live.loud))
        an = self.live.analyzer
        if an is not None and self.live_auto.isChecked():
            bpm = f"{an.bpm:.0f} bpm" if an.bpm else "listening..."
            key = NOTE_NAMES[an.root] if an.root >= 0 else "-"
            self.live_status.setText(f"{bpm}, root {key}")
        else:
            self.live_status.setText(f"{self.tempo.bpm:.0f} bpm, tapped")

    def _follow_analyzer(self, now):
        """Auto mode: pull the clock toward the analyzer's tempo and beat
        each time it has a new estimate, and take its drops."""
        an = self.live.analyzer
        if an is None:
            return
        if an.bpm and an.beat_time is not None \
                and an.beat_time != self._last_beat_sync:
            self._last_beat_sync = an.beat_time
            self.tempo.align(an.bpm, an.beat_time)
            self.live_bpm.blockSignals(True)
            self.live_bpm.setValue(self.tempo.bpm)
            self.live_bpm.blockSignals(False)
        if self.LIVE_AUTO_DROPS and an.drop_time is not None \
                and an.drop_time != self._last_auto_drop:
            self._last_auto_drop = an.drop_time
            self._live_drop(an.drop_time)

    def live_drop_key(self):
        """X: a drop, by hand, on the one."""
        if self._live_active():
            self._live_drop(time.monotonic())

    def _live_drop(self, t):
        """A drop at monotonic time t: the flash and the shaders' iDrop, the
        bar restarts there (drops land on the one), and as in a song the
        clip may switch, or jumps. The restart also counts as a new section,
        so the look and the shader roll again."""
        self._live_drop_t = t
        self.tempo.downbeat(t)
        bars_since = (t - self._live_switch_t) / (4 * self.tempo.beat_ms / 1000)
        if self.video_switch.isChecked() and len(self._song_videos()) > 1 \
                and bars_since >= 4:
            self.backdrop.pick_random()
            self._live_switch_t = t
        elif self.video_jump.isChecked():
            self.backdrop.jump()

    def _live_bpm_edited(self, bpm):
        self.tempo.set_bpm(bpm)

    def tap_tempo(self):
        """T: one tap on the beat. The first tap of a series is the downbeat,
        the intervals set the tempo (shown in the spinbox)."""
        if self.live_auto.isChecked():
            self.live_auto.setChecked(False)
            self.log.appendPlainText("live: auto off, following your taps")
        n = self.tempo.tap()
        if n >= 2:
            self.live_bpm.blockSignals(True)
            self.live_bpm.setValue(self.tempo.bpm)
            self.live_bpm.blockSignals(False)

    def live_downbeat(self):
        """D: the bar starts now, tempo unchanged."""
        self.tempo.downbeat()

    def _on_live_section(self, idx, pos):
        """The live clock crossed into pseudo-section idx: swap the clip
        now and then (as a song does on quiet section boundaries) or jump."""
        if idx == 0:
            return                      # the clip was just picked
        now = time.monotonic()
        bars_since = (now - self._live_switch_t) / (4 * self.tempo.beat_ms / 1000)
        if self.video_switch.isChecked() and len(self._song_videos()) > 1 \
                and bars_since >= 16 and random.random() < 0.35:
            self.backdrop.pick_random()
            self._live_switch_t = now
        elif self.video_jump.isChecked():
            self.backdrop.jump()

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
        if self._following():
            return                  # the sample timeline picks clip and spot
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

    def _playhead_ms(self):
        """The song position for this frame. QMediaPlayer reports it only
        about ten times a second, so between reports it is carried forward
        on the wall clock (up to a quarter second), which keeps hits, cuts
        and the sample timeline within a frame of the sound."""
        pos = self.player.position()
        now = time.monotonic()
        if pos != self._pos_last:
            self._pos_last, self._pos_t = pos, now
            return pos
        return pos + min(250.0, (now - self._pos_t) * 1000.0)

    def _following(self):
        """True while the playing song's picture is driven by its sample
        timeline (a song built from video audio, follow mode not off)."""
        return self._follow is not None \
            and bool(FOLLOW_GROUPS.get(self.video_follow.currentText()))

    def _follow_groups(self, phrase_key):
        """The priority order of event groups for this frame. Fixed for the
        plain modes; in "varies" a lead group is rolled once per (section,
        4-bar phrase), among the groups the song actually has events in,
        weighted towards the layers, the other groups behind it as fallback
        so something is always on screen."""
        mode = self.video_follow.currentText()
        if mode != "varies":
            return FOLLOW_GROUPS[mode]
        if phrase_key != self._follow_key:
            # a new section (or song) always re-rolls; a new phrase half the time
            new_sec = self._follow_key is None or phrase_key[0] != self._follow_key[0]
            self._follow_key = phrase_key
            if new_sec or random.random() < 0.5:
                have = [g for g in FOLLOW_GROUPS["everything"]
                        if self._follow.lists[g][1]]
                if have:
                    lead = random.choices(
                        have, [FOLLOW_LEAD_WEIGHTS[g] for g in have])[0]
                    rest = [g for g in FOLLOW_GROUPS["everything"] if g != lead]
                    self._follow_order = (lead, *rest)
        return self._follow_order

    def _follow_video(self, pos, st, phrase_key=(0, 0)):
        """Put the frames the sounding sample was cut from on screen: on a
        new event, switch clip if needed and seek to its source offset (as
        of now); every frame, hand its rate to the backdrop as st['speed'].
        Nothing sounding from a video = the backdrop free-runs as usual."""
        source = getattr(self.backdrop, "source", None)
        if source is None or self.backdrop.input or not self._following():
            self._follow_cur = None
            return
        ev = self._follow.pick(pos, self._follow_groups(phrase_key))
        source.catch_up = ev is None       # a deliberate slow-down is not a rewind
        if ev is None:
            self._follow_cur = None
            return
        t_ms, video, off, _dur_ms, rate = ev
        want = off + (pos - t_ms) / 1000.0 * rate
        now = time.monotonic()
        if ev is not self._follow_cur:
            self._follow_cur = ev
            if self.backdrop.current != video:
                self.backdrop.set_video(video)
            source.seek(want)
            self._follow_seek_t = now
        elif now - self._follow_seek_t > 0.15 and source.duration:
            # drift check: a seek in the song, or a stalled decode, and the
            # picture is somewhere else; the wrap-around is a short distance
            d = abs(source.position() - want % source.duration)
            if min(d, source.duration - d) > 0.15:
                source.seek(want)
                self._follow_seek_t = now
        st["speed"] = rate

    def _load_sections(self, wav):
        """Called whenever a new file starts: pick a fresh backdrop video,
        then load the section map and loudness envelope for the video
        jumps and effects. Everything degrades to nothing."""
        self._sec_idx = -1
        self._sections, self._sec_bounds, self._env = [], [], None
        self._follow = self._follow_cur = self._follow_key = None
        self._pick_song_video()
        self._outro = None
        parts = Path(wav).stem.split("_")
        self._song_word = parts[1] if len(parts) > 2 and parts[0] == "gacha" else None
        self._song_facts = []
        try:
            meta = json.loads(Path(wav).with_suffix(".json").read_text())
            self._song_facts = json_facts(meta)
            self._sections = meta["sections"]
            self._sec_bounds = [int(s["start_sec"] * 1000)
                                for s in self._sections]
            self._beat_ms = 60000.0 / meta["bpm"]
            plan = FollowPlan(meta.get("events") or [],
                              meta.get("video_sources") or {})
            self._follow = plan if plan.n else None
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
        """Effect state for the frame being drawn, from the playhead: the
        song's, or in live mode the tap-tempo clock with the input's loudness."""
        live = self._live_active()
        if live:
            now = time.monotonic()
            if self.live_auto.isChecked():
                self._follow_analyzer(now)
            pos = self.tempo.pos_ms(now)
            fade = 1.0
            env = self.live.loud
            beat_ms = self.tempo.beat_ms
        else:
            if self.player.playbackState() != QMediaPlayer.PlayingState:
                return None
            pos = self._playhead_ms()
            fade = self._outro_fade(pos)
            beat_ms = self._beat_ms
            env = 0.5
            if self._env is not None and len(self._env):
                env = float(self._env[min(len(self._env) - 1, int(pos // 50))])
        v = {k: sp.value() for k, sp in self.vfx.items()}
        idx, sec, kind = 0, None, "groove"
        sec_start_ms = 0
        if live:
            # no section map: every 8 bars count as a section, so the look,
            # the shader and the clip keep changing the way they do in a song
            sec_len = self.LIVE_SECTION_BARS * 4 * beat_ms
            idx = int(pos // sec_len)
            sec_start_ms = idx * sec_len
            if idx != self._sec_idx:
                self._sec_idx = idx
                self._on_live_section(idx, pos)
        elif self._sections and self._sec_bounds:
            idx = max(0, bisect.bisect_right(self._sec_bounds, pos) - 1)
            sec = self._sections[idx]
            kind = sec["kind"]
            sec_start_ms = self._sec_bounds[idx]
        # bars counted from the section's own first bar, so phrases line up
        bar_i = int(max(0, pos - sec_start_ms) // (4 * beat_ms))
        self._roll_look(idx, bar_i // 4)
        lk = lambda k: v[k] * self._look.get(k, 1.0)     # strength x look

        # music data for generative shaders (iLoud, iBeat, ...)
        sec_id = {"intro": 0, "groove": 1, "break": 2, "outro": 3}.get(kind, 1)
        root = sec.get("root") if sec is not None else None
        total_ms = self._sections[-1]["end_sec"] * 1000 if self._sections else 0
        drop = 0.0
        drop_age = None                       # ms since a live drop
        if live:
            an = self.live.analyzer
            if self.live_auto.isChecked() and an is not None and an.root >= 0:
                root = NOTE_NAMES[an.root]
            if self._live_drop_t is not None:
                drop_age = (now - self._live_drop_t) * 1000.0
                drop = max(0.0, 1 - drop_age / 400.0)
        elif sec is not None and sec.get("transition"):
            drop = max(0.0, 1 - (pos - self._sec_bounds[idx]) / 400.0)
        st = {"music": {
            "loud": env, "beat": (pos % beat_ms) / beat_ms,
            "bar": (pos % (4 * beat_ms)) / (4 * beat_ms),
            "bpm": 60000.0 / beat_ms, "section": sec_id,
            "root": int(round(NOTE_HUES[root] * 12)) if root in NOTE_HUES else -1,
            "drop": drop,
            # live: no song to be a fraction of; a slow 5-minute loop keeps
            # the shaders that age with it (vhs) moving
            "song_pos": (pos / 300000.0) % 1.0 if live
            else pos / total_ms if total_ms else 0.0}}
        # clean video: rolled once per bar, on its downbeat, with probability
        # = strength; the first 1/16 note of the bar then flashes the video as
        # it is (no effects, no shader layer, no words), only the outro fade
        # still applies
        bar_key = (idx, bar_i)
        if bar_key != self._clean_bar:
            self._clean_bar = bar_key
            self._clean_on = random.random() < v["clean"]
        in_bar = (pos - sec_start_ms) - bar_i * 4 * beat_ms
        if not live:
            self._follow_video(pos, st, (idx, bar_i // 4))
        if v["reverse"]:
            # rolled once per bar on its downbeat, with probability =
            # strength; the bar then opens with a rewind of the chosen length
            if bar_key != self._reverse_bar:
                self._reverse_bar = bar_key
                self._reverse_on = random.random() < v["reverse"]
                bars = REVERSE_LENGTHS[self.reverse_len.currentText()]
                if bars is None:
                    bars = random.choice([b for b in REVERSE_LENGTHS.values() if b])
                self._reverse_ms = bars * 4 * beat_ms
                self._reverse_speed = random.choice([1.0, 1.0, 1.5, 2.0, 3.0])
            # a clean flash jogs too: the bare video runs back, then on
            if (self._reverse_on or self._clean_on) and in_bar < self._reverse_ms \
                    and "speed" not in st:          # not while following a sample
                st["reverse"] = self._reverse_speed  # backwards, at that speed
            nxt = idx + 1
            if nxt < len(self._sections):           # cymbal swell: rewind into
                for t in self._sections[nxt].get("transition", []):   # the drop
                    if t.startswith("cymbal:"):
                        length = float(t[7:].rstrip("bar")) * 4 * beat_ms
                        left = self._sec_bounds[nxt] - pos
                        if 0 <= left < length:
                            st["reverse"] = 1.0 + 2.0 * (1 - left / length)
                            st.pop("speed", None)   # the swell wins over the timeline
        if self.video_plain.isChecked() or (self._clean_on and in_bar < beat_ms / 4):
            st["clean"] = True
            if fade < 1.0:
                st["brightness"] = fade
            st["music"]["swell"] = 0.0
            return st
        if v["pump"]:
            st["brightness"] = 0.7 + 0.7 * v["pump"] * env
        if live:
            # the song-side colour moves, on the live clock's pseudo-sections
            if v["color"]:
                st["hue"] = v["color"] * ((idx * 0.37) % 1.0) * 0.5
                st["saturation"] = 1 + 0.5 * v["color"]
            if lk("tint") and root in NOTE_HUES:
                st["tint"] = (NOTE_HUES[root], 0.6 * lk("tint"))
            if v["flash"] and drop_age is not None and drop_age < 120 * v["flash"]:
                st["invert"] = True            # the drop hits
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
        if lk("vign"):
            st["vignette"] = min(1.0, lk("vign"))
        if self._kaleido:
            st["kaleido"] = (self._kaleido, 0.0)      # upright, no rotation
        if self._words_phrase == self._look_idx:
            # a new cloud every half beat (1/8 of a bar): new words, new
            # places; loudness sets how many words and how bright
            phrase_ms = 16 * beat_ms
            t_in = (pos - sec_start_ms) - (bar_i // 4) * phrase_ms
            slot = int(t_in // (beat_ms / 2))
            if slot != self._words_slot:
                self._words_slot = slot
                sec_d = sec if sec is not None else {}
                root = sec_d.get("root")
                hue = NOTE_HUES.get(root) if root in NOTE_HUES else None
                rng = random.Random(random.random())
                # the render's own JSON: the song's keys and values, some of
                # its samples, the playing section's facts (kind, bars, root,
                # transition) several times over, and the song's own name.
                # Only the live input, which has no JSON, gets random words.
                core = [f for f in self._song_facts if f[0] != "sample"]
                smp = [f for f in self._song_facts if f[0] == "sample"]
                facts = core + rng.sample(smp, min(len(smp), 16))
                if sec is not None:
                    facts += section_facts(sec) * 3
                words = fact_words(facts, rng) if facts else list(WORDS)
                if self._song_word:
                    words += [self._song_word] * 6       # the song's own name, often
                n = int(40 + 140 * env)
                frame = self.backdrop.last_frame() if hasattr(self.backdrop, "last_frame") else None
                self.backdrop.set_overlay(make_word_cloud(
                    self.backdrop.size(), words, self.font_families, rng, hue, n,
                    frame_palette(frame, rng)))
            edge = min(1.0, t_in / (0.2 * beat_ms),
                       (phrase_ms - t_in) / (0.2 * beat_ms))
            st["words"] = min(1.0, max(0.0, edge) * (0.6 + 0.6 * env)
                              * min(1.0, 0.5 + self.vfx["words"].value())) * 0.85
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
        if self.live_on.isChecked():
            self.live_on.setChecked(False)          # a song takes over
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
    def _visuals_running(self):
        """A song is playing or live mode is on: the effects are moving."""
        return self.player.playbackState() == QMediaPlayer.PlayingState \
            or self._live_active()

    def _arm_hide(self, *_):
        """(Re)start the idle timer while playing; otherwise show the UI."""
        if self._visuals_running() and self.auto_hide.isChecked():
            self.hide_timer.start(int(self.hide_after.value() * 1000))
        else:
            self.hide_timer.stop()
            self._show_ui()

    def _hide_ui(self, force=False):
        if force or (self._visuals_running() and self.auto_hide.isChecked()):
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
        self.live.stop()
        if self.takes.recording:
            self.takes.stop()
        if self.through is not None:
            self.through.stop()
        if getattr(self, "midi", None) is not None:
            self.midi.close_all()
        if getattr(self.backdrop, "source", None) is not None:
            self.backdrop.source.stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec())
