#!/usr/bin/env python3
"""
gacha.py — Qt front-end for the gacha_engine.py render engine.

Left side: all generation parameters in tabs (Perform / Drums / ... / Settings /
Sections, the layers & effects on the same page / Video / Audio FX / Video FX /
Controls) and the live render log in a tab of its own.
Right side: the sample library (samples/, one subfolder per set, with
checkboxes picking what a render may use), the video library (videos/,
same tree, picking the backdrop clips), past renders
(output/) and the section mixer — double-click anything to listen. New
renders start playing automatically when generation finishes.

Usage:  python3 gacha.py
"""

import bisect
import html
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from PySide6.QtCore import (QEvent, QObject, QPoint, QPointF, QSize, Qt, QRectF, QSettings,
                            QThread, QTimer, QUrl, Signal)
from PySide6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                           QPolygonF, QRegularExpressionValidator, QShortcut)
from PySide6.QtCore import QRegularExpression
from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PySide6.QtWidgets import (
    QFileDialog, QFrame,
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QCompleter, QDoubleSpinBox,
    QFormLayout, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QProgressDialog, QPushButton,
    QSlider, QSpinBox,
    QStyle,
    QScrollArea, QSizePolicy, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gacha_gl import BLEND_MODES, GLBackdrop, gl_available, shader_files
from gacha_engine import (AUDIO_EXTS, RANDOM_START_MODES, SYNTH_STYLES,
                          sample_name)
from gacha_live import NOTE_NAMES, LiveInput, TapTempo, audio_inputs
from gacha_link import INSTALL_HINT as LINK_INSTALL_HINT, LINK_AVAILABLE, LinkClock
from gacha_audio import audio_apis, audio_devices, audio_status, PHRASE_BARS
from gacha_audioproc import AudioProxy, audio_host      # the engine, in its own process
from gacha_takes import TakeStore
from gacha_section import (EXTRA_KINDS, Material, cut_burst, render_section, render_oneshot,
                           render_swell, KINDS, STEMS, SHUFFLE_STEMS, SHUFFLE_ROLES, KIT_PARTS)
import gacha_engine as E
from scipy.signal import resample_poly
import threading
from gacha_video import VideoSource, video_has_audio, video_input_modes, video_inputs
from gacha_fetch import (INSTALL_HINT as FETCH_INSTALL_HINT, FetchWorker,
                         fetch_available, looks_like_url)
from gacha_midi import (CYCLES_CHANNELS, CYCLES_FX_PARAMS, CYCLES_LISTS, CYCLES_MACHINES,
                        CYCLES_PARAMS, CYCLES_RANGES, CYCLES_RETRIG_LENS, CYCLES_RETRIG_RATES,
                        CYCLES_ROOT, CYCLES_SAMPLER_KNOBS, CYCLES_SAMPLER_MODES, CYCLES_SLICE_NOTE,
                        DRUM_LABELS, DRUM_NOTES, DRUM_ROLES, MIDI_AVAILABLE, MidiIn, MidiOut,
                        cycles_cc_value, format_binding, midi_inputs, midi_outputs,
                        parse_binding)
from gacha_freeze import FREEZE
from gacha_frag import (FRAG_DIRS, FRAG_DIVS, FRAG_PLAYS, FRAG_RATES, FRAG_UNITS, FRAGMENTS,
                        MAX_VOICES)
from gacha_fx import (ALL_CHANNELS, AUDIO_FX, BIPOLAR, CHANNELS, FX_SET, FX_SETTINGS,
                      VOICE_LABELS, VOICES)
from gacha_erae import (ERAE_AVAILABLE, PORT as ERAE_PORT, EraeSurface, PAGES as ERAE_PAGES,
                        DEFAULT_PAGES as ERAE_DEFAULT_PAGES, SETUP_DOC as ERAE_SETUP_DOC,
                        make_page as erae_make_page, W as ERAE_W, H as ERAE_H)
from gacha_vst import KNOBS as VST_KNOBS, scan_plugins
from gacha_lfo import (LfoEngine, SHAPES as LFO_SHAPES, RATES as LFO_RATES,
                       RATE_DEFAULT as LFO_RATE_DEFAULT, MODES as LFO_MODES, N_SLOTS as LFO_SLOTS)
from gacha_scene import (SceneBank, Morph, GROUPS as SCENE_GROUPS,
                         GROUP_LABELS as SCENE_GROUP_LABELS, MORPH_BARS, MORPH_DEFAULT,
                         N_SLOTS as SCENE_SLOTS)
from gacha_macro import MacroEngine, N_SLOTS as MACRO_SLOTS, N_DESTS as MACRO_ROWS
try:
    import psutil                    # cpu and memory for the stats HUD
except ImportError:                  # pragma: no cover
    psutil = None

BASE = Path(__file__).parent
GACHA = BASE / "gacha_engine.py"
SAMPLES_DIR = BASE / "samples"
OUT_DIR = BASE / "output"
VIDEOS_DIR = BASE / "videos"
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
QPushButton:checked {
    background: rgba(255, 120, 60, 215); border: 1px solid rgba(255, 210, 170, 240);
    color: rgb(28, 11, 9); font-weight: bold;
}
QPushButton:checked:hover { background: rgba(255, 150, 90, 230); }
QPushButton#solo { padding: 2px 2px; }
QPushButton#generate {
    background: rgba(212, 48, 30, 198); border: 1px solid rgba(255, 120, 90, 240);
    font-weight: bold;
}
QPushButton#generate:hover { background: rgba(255, 96, 70, 220); }
QPushButton#next { font-weight: bold; font-size: 20px; padding: 4px 14px; }
QPushButton#small { padding: 2px 10px; font-size: 12px; }
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
QFrame#info {
    background: rgba(255, 255, 255, 10); border: 1px solid rgba(255, 255, 255, 22);
    border-radius: 8px;
}
QLabel#info_icon { color: rgba(242,233,229,77); font-size: 16px; }
QCheckBox[role="pool"] { font-size: 19px; spacing: 10px; padding: 3px 0; }
QCheckBox[role="pool"]::indicator { width: 28px; height: 28px; border-radius: 6px; }
QComboBox QAbstractItemView { background: rgb(42, 18, 14); }
QComboBox QLineEdit { background: transparent; border: none; padding: 0; }
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
    ("sharpen", "sharpen", (-10.0, 10.0, 0.1, 0.0), "Positive: unsharp mask on the source "
                                                   "pixels, lifts the detail a tape or a "
                                                   "soft grabber lost (0.3 to 0.8 is gentle, "
                                                   "10 is pure grit). Negative: blur, a box "
                                                   "that widens up to 10 source pixels"),
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
    ("gran", "granular", 0.5, "Granular picture: while a fragment sounds (Perform "
                              "tab) the frame is the slice and every grain is a lit "
                              "band at its spot, as wide as the grain, swelling and "
                              "fading with it, a playhead line crossing it (backwards "
                              "for a reversed grain); the motion scatters the bands, "
                              "the size widens them, a frozen playhead holds the "
                              "frame. With none sounding, bands on the Fragments "
                              "rows' settings alone"),
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
    ("kaleido", "kaleido", 0.15, "Kaleidoscope with 4, 6 or 8 mirrored "
                                 "segments, upright. Strength = how often a "
                                 "section gets it"),
    ("clean", "clean", 0.3, "Clean video: on a downbeat, with probability = "
                            "strength, a 1/16-note flash of the video as it "
                            "is, with no effects and no shader layer. "
                            "0 = never, 1 = every bar"),
]
# every action the performer has, in one table: the Controls tab shows it,
# the keyboard shortcuts and the MIDI bindings hang on the same ids.
# (group, id, key, what); "" for key = MIDI only (a continuous control)
KEYMAP = [
    ("show", "fullscreen", "F11", "fullscreen on/off; Esc leaves it too"),
    ("show", "video_only", "C", "video only: no effects, no shader layer"),
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
    ("tape", "through", "A", "live input through its rack to the output, off/on"),
    ("tape", "record", "R", "take: start / stop recording the live feed into RAM"),
    ("tape", "discard", "Shift+R", "forget the last take, or the one recording (RAM only)"),
    ("tape", "clear_takes", "Ctrl+Shift+R", "forget every take (RAM only, the files stay)"),
    ("clock", "tap", "T", "tap tempo, first tap on the downbeat"),
    ("clock", "downbeat", "D", "restart the bar now, tempo kept"),
    ("clock", "live", "L", "live mode (audio drives the visuals) off/on"),
    ("clock", "drop", "X", "mark a drop by hand"),
    ("section", "next", "N", "next section on the bar (pre-rendered)"),
    ("section", "stop", "Shift+N", "stop the section on the bar; stopped, start it again"),
    ("section", "event", "E", "fire a one-shot from the material on the beat"),
    ("section", "ab_down", "[", "A/B 10 % toward the tape"),
    ("section", "ab_up", "]", "A/B 10 % toward the section"),
    ("section", "ab", "", "A/B position, an expression pedal (CC 0..127)"),
    ("section", "stem_drums", "F1", "drums stem in/out"),
    ("section", "stem_layers", "F2", "layers stem in/out"),
    ("section", "stem_chops", "F3", "chops stem in/out"),
    ("section", "stem_events", "F4", "events stem in/out"),
    ("section", "randomize_drums", "F5", "randomize: the drums rendered again, the rest kept"),
    ("section", "randomize_layers", "F6", "randomize: the layers rendered again, the rest kept"),
    ("section", "randomize_chops", "F7", "randomize: the chops rendered again, the rest kept"),
    ("section", "randomize_events", "F8", "randomize: the events rendered again, the rest kept"),
    # MIDI only from here on (no key): more buttons, then the knobs
    ("show", "randomize_fx", "", "new random effect strengths (the Randomize button)"),
    ("show", "grade_neutral", "", "source grade back to neutral"),
    ("show", "blend_next", "", "next shader blend mode (mix, add, screen)"),
    ("section", "kind_next", "", "next section kind (random, groove, break, ...)"),
    ("clock", "link", "", "Ableton Link: tempo and start/stop shared with the network, off/on"),
    ("clock", "link_play", "", "Ableton Link: play / stop sent to the session (the section follows)"),
    ("section", "picture_next", "", "picture while a section plays: follow, take, live"),
    ("section", "dissolve_next", "", "how the A/B pedal blends the two pictures: mix, add, screen"),
    ("mix", "volume", "", "main volume: the audio stream (songs, sections, live input)"),
    ("mix", "sync", "", "audio sync delay, 0 to 2000 ms"),
    ("mix", "shader_mix", "", "shader layer opacity"),
    ("mix", "shader_pick", "", "shader by knob: off at the bottom, then each shader in list order"),
]
# the source grade knobs and every effect strength, one row each, from the
# tables that build their widgets
KEYMAP += [("grade", f"grade_{k}", "", f"{lbl}: {lo:g} to {hi:g}, neutral {val:g}")
           for k, lbl, (lo, hi, _s, val), _t in VIDEO_GRADE]
KEYMAP += [("fx", f"fx_{k}", "", f"{lbl} strength, 0 (off) to 1")
           for k, lbl, _d, _t in VIDEO_EFFECTS]
# the audio racks: eleven effects and a volume per channel (the live input and
# the four stems), every one a knob; then a focused set (the same knobs for
# whichever channel was focused last) and the five focus buttons
for _ch in CHANNELS:
    KEYMAP += [(f"audio_{_ch}", f"afx_{_ch}_{k}", "", f"{_ch}: {lbl}")
               for k, lbl, _d, _t in AUDIO_FX]
    KEYMAP += [(f"audio_{_ch}", f"avol_{_ch}", "", f"{_ch}: volume")]
KEYMAP += [("audio_focus", f"focus_{_ch}", "", f"focus the {_ch} channel for the knobs below")
           for _ch in CHANNELS]
KEYMAP += [("audio_focus", f"afx_{k}", "", f"{lbl} of the focused channel")
           for k, lbl, _d, _t in AUDIO_FX]
KEYMAP += [("audio_focus", "avol", "", "volume of the focused channel")]
# the master inserts (VST3 plugins after the racks): a bypass switch and
# eight knobs, each pointed at a plugin parameter in the Audio FX tab
KEYMAP += [("vst", "inserts", "", "master inserts (VST3 plugins) off/on")]
KEYMAP += [("vst", f"vst_{n}", "", f"insert knob {n}: the plugin parameter it is "
                                  "pointed at in the Audio FX tab")
           for n in range(1, VST_KNOBS + 1)]
# the Video FX tab's pool: every video effect and every shader file can be
# taken out of the show; a button per row toggles that tick
KEYMAP += [("vfx", f"vfx_{k}", "", f"{lbl}: video effect in the show, off/on")
           for k, lbl, _d, _t in VIDEO_EFFECTS]
KEYMAP += [("vfx", f"shd_{p.stem}", "", f"{p.stem}: shader in the pool, off/on")
           for p in shader_files()]
# the clip on screen by hand: the ticked videos of the Videos tab
KEYMAP += [("show", "video_prev", "", "previous ticked video, in list order"),
           ("show", "video_next", "", "next ticked video, in list order"),
           ("show", "video_random", "", "a random ticked video: a shuffled order, each "
                                        "one once before any comes again")]
# the burst: a few ms of the material, now, its picture over the screen
KEYMAP += [("section", "burst", "B", "burst: a slice of the material from a random "
                                    "spot, now, its frame over the picture"),
           ("section", "burst_len", "", "burst length, 10 ms to 1 s (a knob, log scale)"),
           ("section", "glitch", "G", "glitch: on the 4-bar phrase's last bar, sixteenths "
                                      "of it thrown about, then the next section on the one"),
           ("section", "break", "Shift+G", "break: on the 4-bar phrase's last bar, beat-repeat, "
                                           "gap and cymbal swell into the drop, then the next "
                                           "section on the one")]
# the second video input: a webcam next to the VHS grabber, both kept open,
# a cut between them (last in the table so the stock MIDI notes above stay put)
KEYMAP += [("show", "video_second", "W", "second video input (the webcam) on the B side "
                                         "of the pedal, in place of the take frames, off/on")]
# the drum hits as MIDI notes to another program (the Controls tab's MIDI out panel)
KEYMAP += [("section", "metronome", "", "metronome off/on: a click on every beat into the "
                                        "mix, and a note per beat on the MIDI out port"),
           ("section", "midi_out", "", "MIDI out: the drum hits of the section and the "
                                       "song as notes to the chosen port, off/on")]
# the whole picture off: no decoding, no effects, no GPU, for a sound-only night
KEYMAP += [("show", "video_off", "V", "video off: no picture, no decoding, no GPU work "
                                      "(the sound runs as usual), off/on")]
# the drum voices of a section (gacha_fx.VOICES): each in or out and soloed,
# all back in, the grid the switches, events and bursts land on, and a rack
# per voice (eleven effects and a volume, like the five channels above) with
# its focus button
KEYMAP += [("voices", f"voice_{_v}", f"Shift+F{_i}",
            f"{VOICE_LABELS[_v]} in/out, on the quantize grid; its MIDI notes follow")
           for _i, _v in enumerate(VOICES, 1)]
KEYMAP += [("voices", f"solo_{_v}", f"Ctrl+F{_i}",
            f"{VOICE_LABELS[_v]} solo off/on: the other voices out while any is soloed")
           for _i, _v in enumerate(VOICES, 1)]
KEYMAP += [("voices", f"random_{_v}", "",
            f"{VOICE_LABELS[_v]}: another sample for its sound, the groove kept (the R button)")
           for _v in KIT_PARTS]
KEYMAP += [("voices", "voices_reset", "Shift+F8", "every voice back in, no solo, the voice "
                                                  "racks' effects off and their volumes at 1"),
           ("voices", "quant_next", "", "next quantize grid: off, 1/16, 1/8, 1/4, 1/2, bar")]
for _v in VOICES:
    KEYMAP += [(f"rack_{_v}", f"afx_{_v}_{k}", "", f"{VOICE_LABELS[_v]}: {lbl}")
               for k, lbl, _d, _t in AUDIO_FX]
    KEYMAP += [(f"rack_{_v}", f"avol_{_v}", "", f"{VOICE_LABELS[_v]}: volume")]
KEYMAP += [("audio_focus", f"focus_{_v}", "",
            f"focus the {VOICE_LABELS[_v]} voice for the knobs above")
           for _v in VOICES]
# the drum pads: a voice's hit from the material, now, with a velocity (a
# pad's note velocity, the Erae's pressure); the Model:Cycles plays it too
KEYMAP += [("pads", f"hit_{_v}", "", f"{VOICE_LABELS[_v]} hit: the voice's sample from the "
                                      "material, on the quantize grid, through its rack, "
                                      "and its note to the MIDI out, at the pad's velocity")
           for _v in DRUM_ROLES]
# the repeat: a cell of the bar held over the running section (Perform tab)
KEYMAP += [("section", "repeat", "H", "repeat (hold): the cell of the bar playing at the "
                                     "press loops while the key or the note is down, at "
                                     "the division next to it; the loop runs on underneath"),
           ("section", "rep_finer", "-", "repeat division one finer: bar, 1/2, 1/4, 1/8, "
                                        "1/16, 1/32, 1/64"),
           ("section", "rep_coarser", "=", "repeat division one coarser"),
           ("section", "rep_div", "", "repeat division as a knob: bar at the bottom, 1/64 "
                                     "at the top"),
           ("section", "rep_earlier", ",", "repeat position one cell earlier, round the loop"),
           ("section", "rep_later", ".", "repeat position one cell later"),
           ("section", "rep_pos", "", "repeat position as a knob: the captured cell in the "
                                     "middle, 16 cells earlier to 16 later"),
           ("section", "rep_rev", "/", "repeat backwards off/on"),
           ("section", "half", "Shift+H", "half time off/on: the bar at half speed from its "
                                          "start, on the grid; the clock and the bar stay"),
           ("section", "double", "Ctrl+H", "double time off/on: the bar at double speed from "
                                           "its start, on the grid"),
           ("section", "speed_sync", "", "half and double time resync every: 1/2 bar, 1, 2, "
                                         "4 bars or the loop, as a knob"),
           ("section", "pitch_lock", "", "pitch locked off/on: half and double time, a nudge "
                                         "and a tempo change keep the pitch (the stretch) "
                                         "instead of following the speed"),
           ("section", "brake", "Shift+B", "tape stop, held: the section decelerates to a stop "
                                           "over the length next to it; let go, it spins back "
                                           "up to the transport, which ran on"),
           ("section", "brake_len", "", "tape stop length as a knob: 1/4 bar at the bottom, "
                                        "4 bars at the top"),
           ("section", "freeze", "Z", "freeze, held: the whole mix (tape, section, song) held "
                                      "on the instant of the press as a drone of grains, the "
                                      "picture on its frame; let go, the mix comes back where "
                                      "it is"),
           ("section", "freeze_size", "", "freeze grain size, 20 to 250 ms"),
           ("section", "freeze_smear", "", "freeze smear, 0 to 100 %: a jitter round the frozen "
                                           "spot, detune and stereo spray; past half, octave "
                                           "and fifth scatter and reversed grains"),
           ("section", "freeze_pitch", "", "freeze pitch as a knob: -24 semitones at the "
                                           "bottom, +24 at the top"),
           ("section", "freeze_a", "", "freeze attack, 1 ms to 2 s: the mix out, the drone in"),
           ("section", "freeze_r", "", "freeze release, 1 ms to 4 s: the drone out, the mix back"),
           ("section", "nudge_down", "J", "nudge down, held: the section a few percent slower "
                                          "and the clock with it, to line up with something "
                                          "on its own clock (not under Link)"),
           ("section", "nudge_up", "K", "nudge up, held: a few percent faster")]
# the fragments (Perform tab): the bar sliced, a slice held sounds as grains
KEYMAP += [("frag", "frag", "F", "fragments (hold): the slice of the bar under the playhead "
                                 "(or the one picked in the Perform tab) sounds as grains in "
                                 "place of the section, which runs on underneath; let go, "
                                 "the release, then the section")]
KEYMAP += [("frag", f"frag_{_n}", "", f"fragments (hold): slice {_n} of the bar, of the "
                                      "slices set in the Perform tab")
           for _n in range(1, 17)]
KEYMAP += [("frag", "frag_div", "", "fragments slices per bar as a knob: 4, 8, 16, 32"),
           ("frag", "frag_slice", "", "fragments slice for the F key as a knob: follow at the "
                                      "bottom, then slice 1 up to the last"),
           ("frag", "frag_size", "", "fragments grain size as a knob: 1/64 of the bar at the "
                                     "bottom up to a bar"),
           ("frag", "frag_rate", "", "fragments grain rate as a knob: dense at the bottom, "
                                     "then a grain every 1/64 of the bar up to every bar"),
           ("frag", "frag_motion", "", "fragments motion, 0 to 800 %: how far from the "
                                       "playhead the grains land, in grain sizes"),
           ("frag", "frag_pitch", "", "fragments pitch as a knob: -24 semitones at the "
                                      "bottom, +24 at the top, none in the middle"),
           ("frag", "frag_speed", "", "fragments playhead speed as a knob: -200 % at the "
                                      "bottom, frozen in the middle, +200 % at the top"),
           ("frag", "frag_repeats", "", "fragments repeats, 1 to 16: each grain that many "
                                        "times over"),
           ("frag", "frag_shape", "", "fragments grain envelope, 0 to 100 %: a pluck, a "
                                      "Hann in the middle, open"),
           ("frag", "frag_dir", "", "fragments grain direction, next: forward, reverse, "
                                    "random, alternate"),
           ("frag", "frag_play", "", "fragments playback at the slice's ends, next: loop, "
                                     "ping-pong, once"),
           ("frag", "frag_a", "", "fragments attack, 0 to 2 s"),
           ("frag", "frag_d", "", "fragments decay, 0 to 2 s"),
           ("frag", "frag_s", "", "fragments sustain, 0 to 100 %"),
           ("frag", "frag_r", "", "fragments release, 0 to 4 s")]
# the effect settings (Audio FX tab, under the racks), every channel's: the
# reverse delay's pair here, where it was, so the default map after it stands;
# the rest after the whole table
FX_LABEL = {k: lbl for k, lbl, _d, _t in AUDIO_FX}
KEYMAP += [("fxset", "rev_time", "", "reverse delay time as a knob: 1/64 of the bar at the "
                                     "bottom up to a bar, every channel's"),
           ("fxset", "rev_fb", "", "reverse delay feedback, 0 to 95 %, every channel's")]
# the LFOs (LFO tab): each slot in or out and its depth as a knob, every
# trig and one-shot slot restarted, and the panic key
KEYMAP += [row for _n in range(1, LFO_SLOTS + 1)
           for row in (("lfo", f"lfo_{_n}", "", f"LFO {_n} off/on"),
                       ("lfo", f"lfo_{_n}_depth", "", f"LFO {_n} depth, 0 to 100 % (the "
                                                      "sign stays as set in the LFO tab)"))]
KEYMAP += [("lfo", "lfo_trig", "", "retrigger: every trig and one-shot LFO restarts its "
                                   "cycle now (a section start and a drop do this too)"),
           ("lfo", "lfo_off", "Shift+L", "every LFO off, the knobs back where you set them")]
# the scenes (Scenes tab): every slot loaded and saved from a button, and
# the morph switch
KEYMAP += [("scene", f"scene_load_{_n}", "", f"scene {_n}: load it (the groups it ticks), at "
                                             "once or morphed over the time set in the tab")
           for _n in range(1, SCENE_SLOTS + 1)]
KEYMAP += [("scene", f"scene_save_{_n}", "", f"scene {_n}: save the state as it is now into "
                                             "the slot")
           for _n in range(1, SCENE_SLOTS + 1)]
KEYMAP += [("scene", "scene_morph", "", "scene morph off/on: a load glides the knobs over the "
                                        "time set in the Scenes tab instead of jumping")]
# the macros (Macros tab): each one a knob, 0 to 100 %, that moves every
# destination of its row through its range
KEYMAP += [("macro", f"macro_{_n}", "", f"macro {_n}, 0 to 100 %: every knob it points at "
                                        "(Macros tab) follows through its own range")
           for _n in range(1, MACRO_SLOTS + 1)]
# the Model:Cycles tab (gacha_midi.CYCLES_PARAMS): a knob per parameter per
# track of the box and a retrig hold, the same set again for the focused
# track with six focus buttons, the box's delay and reverb, the retrig's
# settings. The tracks are the box's six, on MIDI channels 1 to 6, each the
# drum the MIDI out's Model:Cycles tick sends there
MC_TRACKS = tuple(range(1, 7))
MC_ROLE = {ch: role for role, ch in CYCLES_CHANNELS.items()}
MC_PARAM = {key: (cc, label, kind, default, what)
            for key, cc, label, kind, default, what in CYCLES_PARAMS + CYCLES_FX_PARAMS}
MC_RETRIG_KNOBS = ("mc_retrig_rate", "mc_retrig_len", "mc_retrig_fade", "mc_retrig_vel")
for _t in MC_TRACKS:
    KEYMAP += [(f"mc_{_t}", f"mc_{_t}_{key}", "", f"track {_t} {label}: {what}")
               for key, _cc, label, _kind, _d, what in CYCLES_PARAMS]
    KEYMAP += [(f"mc_{_t}", f"mc_retrig_{_t}", "",
                f"track {_t} retrig (hold): the track's note again at the retrig rate "
                "while held, the velocity fading as set (the box's own retrig has no CC)")]
KEYMAP += [("mc_focus", f"mc_focus_{_t}", "", f"focus track {_t} for the knobs below")
           for _t in MC_TRACKS]
KEYMAP += [("mc_focus", f"mc_{key}", "", f"{label} of the focused track: {what}")
           for key, _cc, label, _kind, _d, what in CYCLES_PARAMS]
KEYMAP += [("mc_fx", f"mc_fx_{key}", "", f"{label}: {what}")
           for key, _cc, label, _kind, _d, what in CYCLES_FX_PARAMS]
KEYMAP += [("mc_retrig", "mc_retrig_rate", "", "retrig rate, as a knob: a note every bar at "
                                                "the bottom up to every 1/80 of it"),
           ("mc_retrig", "mc_retrig_len", "", "retrig length, as a knob: the velocity curve "
                                               "over 1/8 of a step at the bottom up to 16 "
                                               "steps, flat (infinite) at the top"),
           ("mc_retrig", "mc_retrig_fade", "", "retrig fade, -128 to 127: negative fades the "
                                                "velocity out over the length, positive in"),
           ("mc_retrig", "mc_retrig_vel", "", "retrig velocity, 1 to 127: what the curve "
                                               "starts from")]
# the rest of the effect settings, every channel's: the menus as knobs
KEYMAP += [("fxset", ident, "", f"{FX_LABEL[fx]} {label}: {what}"
            + (", as a knob over the menu" if kind == "list" else ""))
           for ident, fx, label, kind, _lo, _hi, _d, _s, what in FX_SETTINGS
           if ident not in ("rev_time", "rev_fb")]
_MC_KNOBS = [key for key, _cc, _l, kind, *_ in CYCLES_PARAMS if kind != "on"]
MC_CONTINUOUS = {f"mc_{_t}_{k}" for _t in MC_TRACKS for k in _MC_KNOBS} \
    | {f"mc_{k}" for k in _MC_KNOBS} \
    | {f"mc_fx_{key}" for key, _cc, _l, kind, *_ in CYCLES_FX_PARAMS if kind != "on"} \
    | set(MC_RETRIG_KNOBS)
# ids that take a velocity (0..1) from a note's velocity or a pad's pressure
VELOCITY = {f"hit_{_v}" for _v in DRUM_ROLES}
# ids held rather than pressed: the key's or the note's release matters too
# (no QShortcut, the event filter sees both; a CC's falling edge releases)
HOLD = {"repeat", "brake", "nudge_down", "nudge_up", "frag", "freeze"} \
    | {f"frag_{_n}" for _n in range(1, 17)} | {f"mc_retrig_{_t}" for _t in MC_TRACKS}
# ids that take a value (0..1) instead of firing; only a CC can drive them
CONTINUOUS = {"ab", "volume", "sync", "shader_mix", "shader_pick", "avol", "burst_len",
              "rep_div", "rep_pos", "speed_sync", "brake_len", "frag_div", "frag_slice",
              "frag_size", "frag_rate", "frag_motion", "frag_pitch", "frag_speed",
              "frag_repeats", "frag_shape", "frag_a", "frag_d", "frag_s", "frag_r",
              "rev_time", "rev_fb", "freeze_size", "freeze_smear", "freeze_pitch",
              "freeze_a", "freeze_r"} \
    | {f"lfo_{_n}_depth" for _n in range(1, LFO_SLOTS + 1)} \
    | {f"macro_{_n}" for _n in range(1, MACRO_SLOTS + 1)} \
    | {f"grade_{k}" for k, *_ in VIDEO_GRADE} | {f"fx_{k}" for k, *_ in VIDEO_EFFECTS} \
    | {f"afx_{k}" for k, *_ in AUDIO_FX} \
    | {f"afx_{_ch}_{k}" for _ch in ALL_CHANNELS for k, *_ in AUDIO_FX} \
    | {f"avol_{_ch}" for _ch in ALL_CHANNELS} | {f"vst_{n}" for n in range(1, VST_KNOBS + 1)} \
    | {ident for ident, *_ in FX_SETTINGS} | MC_CONTINUOUS
# the faint mark after a Sections page row's name: what only the generated
# songs use; everything else on the page the live render reads at each N too
MODE_SONG = "\u266a"                                      # a note
MODE_TIP = "the generated songs only; a live section does not read it"

KEYMAP_GROUPS = {"show": "picture", "tape": "tape", "clock": "clock",
                 "section": "section engine", "mix": "mix and shader",
                 "grade": "source grade", "fx": "video effects",
                 "audio_focus": "audio racks: focused channel",
                 "fxset": "audio racks: effect settings",
                 "vst": "master inserts (VST3)", "vfx": "video fx: the pool",
                 "voices": "drum voices", "pads": "drum pads", "lfo": "LFOs", "scene": "scenes",
                 "macro": "macros", "frag": "fragments: the sliced bar",
                 **{f"audio_{_ch}": f"audio rack: {_ch}" for _ch in CHANNELS},
                 **{f"rack_{_v}": f"audio rack: {VOICE_LABELS[_v]} voice" for _v in VOICES},
                 **{f"mc_{_t}": f"Model:Cycles: track {_t} ({DRUM_LABELS[MC_ROLE[_t]]})"
                    for _t in MC_TRACKS},
                 "mc_focus": "Model:Cycles: focused track",
                 "mc_fx": "Model:Cycles: delay and reverb", "mc_retrig": "Model:Cycles: retrig"}
VOICE_SHORT = {"kick": "kick", "snare": "snare", "hihat": "c.hat", "ohat": "o.hat",
               "ride": "ride", "glitch": "stutter", "sub": "sub"}
LFO_LABEL_FIX = {"ab": "A/B", "volume": "main volume", "sync": "audio sync delay",
                 "frag_div": "fragments: slices", "frag_slice": "fragments: slice",
                 "frag_size": "fragments: grain size", "frag_rate": "fragments: grain rate",
                 "frag_motion": "fragments: motion", "frag_pitch": "fragments: pitch",
                 "frag_speed": "fragments: speed", "frag_repeats": "fragments: repeats",
                 "frag_shape": "fragments: shape", "frag_a": "fragments: attack",
                 "frag_d": "fragments: decay", "frag_s": "fragments: sustain",
                 "frag_r": "fragments: release",
                 **{ident: f"{FX_LABEL[fx]}: {label}" for ident, fx, label, *_ in FX_SETTINGS},
                 "freeze_size": "freeze: grain size", "freeze_smear": "freeze: smear",
                 "freeze_pitch": "freeze: pitch", "freeze_a": "freeze: attack",
                 "freeze_r": "freeze: release",
                 "shader_mix": "shader opacity", "shader_pick": "shader pick",
                 "burst_len": "burst length", "rep_div": "repeat division",
                 "rep_pos": "repeat position", "speed_sync": "speed resync",
                 "brake_len": "tape stop length",
                 **{f"macro_{_n}": f"macro {_n}" for _n in range(1, MACRO_SLOTS + 1)}}


def _knob_label(group, ident, what):
    """A short name for a knob in the LFO destination list."""
    if ident in LFO_LABEL_FIX:
        return LFO_LABEL_FIX[ident]
    if group.startswith("audio_") or group.startswith("rack_"):
        return f"rack {what}"                       # "drums: lowpass"
    short = what
    if group.startswith("mc_"):                     # "M:C track 1 pitch", "M:C delay time"
        return "M:C " + short[:min(i for i in (short.find(","), short.find(":")) if i > 0)]
    for cut in (",", " (", ":"):
        i = short.find(cut)
        if i > 0:
            short = short[:i]
    short = short.replace(" strength", "").strip()
    return f"{ {'grade': 'grade', 'fx': 'fx', 'vst': 'vst'}.get(group, group)}: {short}"


# the LFO destinations: (ident, label) for every knob, in table order
LFO_DESTS = [(ident, _knob_label(group, ident, what)) for group, ident, _k, what in KEYMAP
             if ident in CONTINUOUS and group not in ("audio_focus", "lfo", "scene", "mc_focus")]
# the macro destinations: the same knobs, the macros themselves left out
# (a macro on a macro would loop), plus the LFOs' depths: a macro opening
# an LFO is the classic pairing (the LFOs do not get them: an LFO on its
# own depth would chase itself)
MACRO_DESTS = [(ident, label) for ident, label in LFO_DESTS if not ident.startswith("macro_")]     + [(f"lfo_{_n}_depth", f"LFO {_n} depth") for _n in range(1, LFO_SLOTS + 1)]
# the quantize grid: what a switch, an event or a burst waits for while a
# section plays, as a fraction of a bar (0 = at once)
QUANT_GRIDS = [("off", 0.0), ("1/16", 1 / 16), ("1/8", 1 / 8), ("1/4", 1 / 4),
               ("1/2", 1 / 2), ("bar", 1.0)]
QUANT_DEFAULT = "1/8"
# the repeat's cell: a division of the bar
REPEAT_DIVS = [("bar", 1.0), ("1/2", 1 / 2), ("1/4", 1 / 4), ("1/8", 1 / 8),
               ("1/16", 1 / 16), ("1/32", 1 / 32), ("1/64", 1 / 64)]
REPEAT_DEFAULT = "bar"
REPEAT_RANGE = 16                 # cells earlier or later the position knob reaches
# the speed: what half and double time resync on, and a tape stop's length
SPEED_SYNCS = [("1/2", 0.5), ("1 bar", 1.0), ("2 bars", 2.0), ("4 bars", 4.0), ("loop", 0.0)]
SPEED_SYNC_DEFAULT = "1 bar"
BRAKE_LENS = [("1/4", 0.25), ("1/2", 0.5), ("1 bar", 1.0), ("2 bars", 2.0), ("4 bars", 4.0)]
BRAKE_DEFAULT = "1 bar"
PITCH_MODES = ["follows", "locked"]
NUDGE_DEFAULT = 4                 # percent
GRAN_VIEW_S = 1.25                # the granular picture, idle: seconds across the frame
GRAN_HEADS = 16                   # bands drawn at the most
FRAG_DEFAULTS = {"div": 16, "size": "1/16", "rate": "dense", "motion": 100, "pitch": 0,
                 "speed": 100, "repeats": 1, "shape": 50, "dir": "forward", "play": "loop",
                 "a": 5, "d": 200, "s": 100, "r": 250}


class _ComboSearch(QObject):
    """The typing side of searchable(): Enter takes the first name that
    matches what was typed, Escape or leaving with a name that matches
    nothing puts the chosen one back (an editable combo would otherwise
    keep the typed text, and a highlight walked with the arrows then
    Escape would stick). A combo with the "clearable" property has no
    empty entry in its list: an empty field is "nothing", and deleting
    the name then Enter, or leaving, clears it (index -1)."""

    def __init__(self, combo, comp):
        super().__init__(combo)
        self.combo, self.comp, self.le = combo, comp, combo.lineEdit()
        self.le.returnPressed.connect(self.pick)
        self.le.editingFinished.connect(self.settle)
        comp.popup().installEventFilter(self)
        self.le.installEventFilter(self)

    def chosen(self):
        return self.combo.itemText(self.combo.currentIndex())

    def clear_if_empty(self):
        if self.combo.property("clearable") and not self.le.text().strip():
            self.combo.setCurrentIndex(-1)
            self.le.setText("")
            return True
        return False

    def pick(self):
        c = self.combo
        if self.clear_if_empty():
            return
        if c.findText(self.le.text()) < 0 and self.comp.completionCount():
            i = c.findText(self.comp.currentCompletion())
            if i >= 0:
                c.setCurrentIndex(i)
                self.le.setText(c.itemText(i))

    def settle(self):
        try:
            if self.clear_if_empty():
                return
            if self.combo.findText(self.le.text()) < 0:
                self.le.setText(self.chosen())
        except RuntimeError:                   # the widgets going away
            pass

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            QTimer.singleShot(0, self.settle_all)
        return False

    def settle_all(self):
        try:
            self.le.setText(self.chosen())
        except RuntimeError:
            pass


def searchable(combo):
    """A combo you can type into: click it, type, and its list narrows to
    the names holding what you typed (any part, case blind); Enter takes
    the first, a click any; Escape, or leaving with a name that matches
    nothing, keeps what was chosen. The arrows and the wheel work as
    before. For the long lists (the LFO and macro destinations, the M:C
    menus)."""
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    comp = QCompleter(combo.model(), combo)
    comp.setCompletionMode(QCompleter.PopupCompletion)
    comp.setFilterMode(Qt.MatchContains)
    comp.setCaseSensitivity(Qt.CaseInsensitive)
    combo.setCompleter(comp)
    pop = comp.popup()                 # a window of its own, above a fullscreen picture
    pop.setWindowFlags(pop.windowFlags() | Qt.WindowStaysOnTopHint)
    _ComboSearch(combo, comp)
    combo.setToolTip((combo.toolTip() + " Type to search the list.").strip())
    return combo


def default_midi_map():
    """The bindings before anyone learns their own: one note per button
    action, from C1 (36) up in table order, on channel 1, which is what a
    pad or a keyboard sends; the knobs on CC 11 (expression) for A/B, then
    12, 13, ... in table order, channel 1. The audio racks live on MIDI
    channel 2: CC 1 upward for their knobs, notes from 36 for the focus
    buttons; the drum voices on channel 3 the same way, their switches as
    notes from 36 and their racks' knobs on CC 1 upward. Channel 1's notes
    ran out at 127 with the repeat's buttons: the buttons after that go on
    channel 4 from note 36, and a channel's CCs past 127 the same way. The
    Model:Cycles tab's rows come last and take a channel of their own, the
    one after the last in use by then, notes from 36 and CCs from 1. Any
    controller overwrites these by MIDI learn."""
    out = {}
    # (kind, channel) -> next number; a channel that runs past 127 spills
    # into the one after the last in use, 4 at first
    nxt = {("note", 1): 36, ("cc", 1): 11, ("note", 2): 36, ("cc", 2): 1,
           ("note", 3): 36, ("cc", 3): 1}
    spill = {}

    def take(kind, ch):
        if ch == "mc":                      # the tab's own channel, made on first use
            if "mc" not in spill:
                spill["mc"] = max(c for _k, c in nxt) + 1
                nxt[("note", spill["mc"])], nxt[("cc", spill["mc"])] = 36, 1
            ch = spill["mc"]
        while nxt[(kind, ch)] > 127:
            if (kind, ch) not in spill:
                new = max(c for _k, c in nxt) + 1
                spill[(kind, ch)] = new
                nxt[(kind, new)] = 36 if kind == "note" else 1
            ch = spill[(kind, ch)]
        n = nxt[(kind, ch)]
        nxt[(kind, ch)] = n + 1
        return format_binding(kind, ch, n)
    for group, ident, _k, _w in KEYMAP:
        audio = group.startswith("audio")
        voice = group == "voices" or group.startswith("rack_")
        kind = "cc" if ident in CONTINUOUS else "note"
        out[ident] = take(kind, "mc" if group.startswith("mc_") else 3 if voice
                          else 2 if audio else 1)
    return out


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
# drums: kick and snare, the backbone; hits: hats, ride, glitches, too dense
# to cut on by themselves (sixteenth hats are a strobe)
FOLLOW_GROUPS = {"off": (), "textures": ("textures",),
                 "textures and chops": ("chops", "textures"),
                 "drums": ("drums", "chops", "textures"),
                 "everything": ("drums", "hits", "chops", "textures"),
                 "varies": ("drums", "hits", "chops", "textures")}
# varies: how often each group gets to lead (the drums are sounding almost
# all the time, so they would win every phrase on an even roll; the hats
# rarely, they flicker)
FOLLOW_LEAD_WEIGHTS = {"drums": 1.0, "hits": 0.4, "chops": 1.4, "textures": 1.6}
# drums leading: a kick's or snare's frame holds until the next one, this
# long at most (ms), so the cut is on the beat and the picture then runs on
FOLLOW_DRUM_HOLD_MS = 2000.0
EVENT_GROUP = {"kick": "drums", "snare": "drums", "hihat": "hits", "ohat": "hits",
               "ride": "hits", "glitch": "hits", "chop": "chops",
               "texture": "textures", "swell": "textures",
               "collage": "textures", "oneshot": "textures"}


class FollowPlan:
    """The render's sample timeline (meta["events"]) reduced to the events
    that came from a video: for any playhead position, the event whose
    frames should be on screen. Rows are (start ms, video, source offset s,
    duration ms, rate), per group, sorted by start."""

    def __init__(self, events, sources):
        self.lists = {g: ([], []) for g in ("drums", "hits", "chops", "textures")}
        ok = {}
        for t, src, off, dur, rate, role, *_vel in events:
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

    def pick(self, pos_ms, groups, hold_ms=0.0):
        """The latest event of the first group that is sounding at pos_ms.
        With `hold_ms`, an event of the first group stays on screen past its
        own length, until the next event of that group or `hold_ms` at most:
        the drums cut on the hit, then the picture runs on from there."""
        for g in groups:
            ts, rows = self.lists[g]
            i = bisect.bisect_right(ts, pos_ms) - 1
            if i < 0:
                continue
            end = rows[i][0] + rows[i][3]
            if hold_ms and g == groups[0]:
                nxt = ts[i + 1] if i + 1 < len(ts) else float("inf")
                end = max(end, min(nxt, rows[i][0] + hold_ms))
            if pos_ms < end:
                return rows[i]
        return None


# rewind lengths in bars; None = a random one per bar
REVERSE_LENGTHS = {"1/8 bar": 1 / 8, "1/4 bar": 1 / 4, "1/2 bar": 1 / 2,
                   "1 bar": 1.0, "2 bars": 2.0, "random": None}

LOOK_EXEMPT = {"color", "pump", "flash", "pixel", "kaleido", "reverse",
               "clean", "gran"}


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

    def granulate(self, px, gr):
        """The granular picture: the frame is the buffer (now at its right
        edge) and every grain a band at its spot, as wide as the grain,
        lifted by its window with soft edges, a bright playhead line where
        it reads; outside the bands the frame darkens a little."""
        h, w = px.shape[:2]
        amt = gr["amt"]
        x = (np.arange(w) + 0.5) / w
        lit = np.zeros(w, "float32")
        line = np.zeros(w, "float32")
        for x0, bw, hx, l in gr.get("heads", []):
            if bw <= 0:
                continue
            e = max(1.5 / w, 0.08 * bw)                 # the edge's softness
            inside = np.clip((x - x0) / e, 0, 1) * np.clip((x0 + bw - x) / e, 0, 1)
            lit = np.maximum(lit, inside * l)
            line = np.maximum(line, np.clip(1.0 - np.abs(x - hx) * w / 1.5, 0, 1) * inside)
        gain = 1.0 - 0.3 * amt + 0.6 * amt * lit
        out = px * gain[None, :, None] + (40.0 * amt * lit)[None, :, None]
        return out * (1 - line[None, :, None] * amt) + (255.0 - out) * (line[None, :, None] * amt)

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
        gr = g("gran")                        # the granular picture
        if gr:
            px = self.granulate(px, gr)
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
        self.override_ab = 0.0     # how far the picture dissolves into it (GL only)
        self.override_blend = 0    # mix, add or screen (GL only)
        self.grade = {}            # source grade (GL only applies it)
        self._last_fx = 0.0
        self._zoom = 0.0
        self._frozen = None         # captured frame while a beat-repeat runs
        self.child = child
        self.frames = 0             # paints so far (the stats HUD reads the rate)
        self.current = None         # path of the video now looping
        self.input = None           # live capture device while video in is on
        self._input_opts = None     # its capture mode
        self.source2 = None         # a second capture device (the webcam), kept open
        self.input2 = None          # its device string while it is on
        self._input2_opts = None
        self.enabled = True         # False: no decoding, no repaint, a black widget
        self.second_on_b = False    # the second input as the B layer (GL only shows it)
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

    def set_second(self, device, options=None):
        """A second capture device (a webcam next to the VHS grabber) opened
        alongside the first; the GL backdrop puts it on the B side of the
        pedal, this one only keeps it open. None closes it."""
        self.input2, self._input2_opts = device, options
        if device:
            if self.source2 is None:
                self.source2 = VideoSource(parent=self)
            if self.enabled:
                self.source2.open(device, options=options)
        elif self.source2 is not None:
            self.source2.close()

    def set_second_on_b(self, on):
        self.second_on_b = bool(on)     # the B layer is GL only, like the take frames

    def set_enabled(self, on):
        """Video off: decoders closed, no frames processed, a black widget;
        on again reopens what was chosen (see GLBackdrop.set_enabled)."""
        on = bool(on)
        if on == self.enabled:
            return
        self.enabled = on
        if on:
            if self.source is not None:
                self._open_main(random_start=True)
            if self.input2 and self.source2 is not None:
                self.source2.open(self.input2, options=self._input2_opts)
        else:
            for s in (self.source, self.source2):
                if s is not None:
                    s.close()
            self._image = self._live = self._arr = None
            self.update()

    def _open_main(self, random_start=False):
        if self.input:
            self.source.open(self.input, options=self._input_opts)
        elif self.current is not None:
            self.source.open(self.current, random_start)

    def second_frame(self):
        if self.second_on_b and self.input2 and self.source2 is not None:
            return self.source2.latest()
        return None

    def live_source(self):
        """The capture device behind the picture, or None when a clip is
        (or the video is off: the device is closed then)."""
        src = self.source
        return src if (self.enabled and self.input and src is not None and src.is_live) else None

    def set_input(self, device, options=None):
        """Live video in: show a capture device (see video_inputs()) instead
        of the clips, at the capture mode `options` (a video_input_modes()
        entry, None = the device's default), and ignore clip changes until
        `None` switches back."""
        if self.source is None:
            return
        self.input, self._input_opts = device, options
        if self.enabled:
            self._open_main(random_start=True)

    def set_video(self, path, random_start=False):
        if self.source is None or self.input:
            return None
        self.current = Path(path)
        if self.enabled:
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
        if not self.enabled:
            return
        src = self.source
        fr = src.latest() if src is not None else None
        if fr is None or fr is self._live:
            return                              # nothing new
        st = self.state_fn() if self.state_fn else None
        rev = st.get("reverse") if st else None
        if st and "speed" in st:                # the song's sample timeline
            src.set_speed(float(st["speed"]))
        else:
            src.set_speed(-float(rev) if rev else 1.0)   # a real rewind
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
        self.frames += 1
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

    def restore(self):
        """The ticks from the settings again, as at the first refresh (a
        set loaded)."""
        self._seen = None
        self.refresh()

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


class StreamPlayer(QObject):
    """A song through the app's own audio stream (LiveAudio.song), with the
    QMediaPlayer face the window uses: the same methods and signals. `la_fn`
    returns the running LiveAudio or None."""

    positionChanged = Signal(int)
    durationChanged = Signal(int)
    playbackStateChanged = Signal(object)
    mediaStatusChanged = Signal(object)

    def __init__(self, la_fn, parent=None):
        super().__init__(parent)
        self._la_fn = la_fn
        self._src = QUrl()
        self._state = QMediaPlayer.StoppedState
        self._loaded = False
        self._pending_play = False
        self._error = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def _song(self):
        la = self._la_fn()
        return la.song if la is not None else None

    def _set_state(self, st):
        if st != self._state:
            self._state = st
            self.playbackStateChanged.emit(st)

    def setSource(self, url):
        self._src = url
        self._loaded, self._error = False, None
        song = self._song()
        if song is not None:
            song.load(url.toLocalFile(), play=False)
            self.mediaStatusChanged.emit(QMediaPlayer.LoadingMedia)

    def source(self):
        return self._src

    def play(self):
        """Playing is reported when the sound actually starts: a song still
        loading plays as soon as it is ready, and the state (which starts
        the visuals) follows in _tick, so picture and sound begin together."""
        song = self._song()
        if song is None:
            return
        song.play()
        if song.loaded:
            self._set_state(QMediaPlayer.PlayingState)
        else:
            self._pending_play = True

    def pause(self):
        song = self._song()
        if song is not None:
            song.pause()
        self._pending_play = False
        self._set_state(QMediaPlayer.PausedState)

    def stop(self):
        song = self._song()
        if song is not None:
            song.stop()
        self._pending_play = False
        self._set_state(QMediaPlayer.StoppedState)

    def setPosition(self, ms):
        song = self._song()
        if song is not None:
            song.seek_ms(ms)
            self.positionChanged.emit(int(ms))

    def position(self):
        """Where the listener is: frames handed to the device, minus the
        output buffer not yet heard, so the picture follows the sound."""
        la = self._la_fn()
        if la is None:
            return 0
        return max(0, int(la.song.position_ms() - la.out_latency_ms))

    def duration(self):
        song = self._song()
        return int(song.duration_ms()) if song is not None else 0

    def playbackState(self):
        return self._state

    def _tick(self):
        song = self._song()
        if song is None:                          # the stream went away
            if self._state != QMediaPlayer.StoppedState:
                self._set_state(QMediaPlayer.StoppedState)
            return
        if song.error and song.error != self._error:
            self._error = song.error
            self.mediaStatusChanged.emit(QMediaPlayer.InvalidMedia)
            self._set_state(QMediaPlayer.StoppedState)
            return
        if song.loaded and not self._loaded:
            self._loaded = True
            self.durationChanged.emit(self.duration())
            self.mediaStatusChanged.emit(QMediaPlayer.LoadedMedia)
        if self._pending_play and song.state == "playing":
            self._pending_play = False
            self._set_state(QMediaPlayer.PlayingState)   # the sound has started
        if self._state == QMediaPlayer.PlayingState:
            self.positionChanged.emit(self.position())
            if song.ended:
                self._set_state(QMediaPlayer.StoppedState)
                self.mediaStatusChanged.emit(QMediaPlayer.EndOfMedia)


class AppPlayer(QObject):
    """The window's one player: Qt's media player when the app has no audio
    stream open, the stream (racks, stems, F1-F4) when it has. The backend
    is chosen when a source is set; the signals of whichever is active are
    relayed under one name."""

    positionChanged = Signal(int)
    durationChanged = Signal(int)
    playbackStateChanged = Signal(object)
    mediaStatusChanged = Signal(object)

    def __init__(self, la_fn, parent=None):
        super().__init__(parent)
        self._la_fn = la_fn
        self.qt = QMediaPlayer()
        self.stream = StreamPlayer(la_fn, self)
        self.active = self.qt
        for backend in (self.qt, self.stream):
            backend.positionChanged.connect(lambda v, b=backend: self._relay(b, self.positionChanged, v))
            backend.durationChanged.connect(lambda v, b=backend: self._relay(b, self.durationChanged, v))
            backend.playbackStateChanged.connect(lambda v, b=backend: self._relay(b, self.playbackStateChanged, v))
            backend.mediaStatusChanged.connect(lambda v, b=backend: self._relay(b, self.mediaStatusChanged, v))

    def _relay(self, backend, sig, value):
        if backend is self.active:
            sig.emit(value)

    @property
    def streaming(self):
        return self.active is self.stream

    def setAudioOutput(self, out):
        self.qt.setAudioOutput(out)

    def setSource(self, url):
        want = self.stream if self._la_fn() is not None else self.qt
        if want is not self.active:
            self.active.stop()
            self.active = want
            self.playbackStateChanged.emit(QMediaPlayer.StoppedState)
        self.active.setSource(url)

    def source(self):
        return self.active.source()

    def play(self):
        self.active.play()

    def pause(self):
        self.active.pause()

    def stop(self):
        self.active.stop()

    def setPosition(self, ms):
        self.active.setPosition(ms)

    def position(self):
        return self.active.position()

    def duration(self):
        return self.active.duration()

    def playbackState(self):
        return self.active.playbackState()


class LfoBar(QWidget):
    """An LFO's wave right now: a bar from the middle, right for positive,
    left for negative, lit while the slot is on."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value, self.on = 0.0, False
        self.setFixedSize(64, 14)

    def set_value(self, value, on):
        value = max(-1.0, min(1.0, float(value)))
        if (value, on) != (self.value, self.on):
            self.value, self.on = value, on
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(28, 11, 9, 117))
        p.drawRoundedRect(QRectF(0, 0, w, h), 4, 4)
        mid = w / 2
        if self.on:
            p.setBrush(QColor(255, 214, 80, 230))
            x = mid * self.value
            p.drawRect(QRectF(min(mid, mid + x), 2, abs(x), h - 4))
        p.setBrush(QColor(242, 233, 229, 120))
        p.drawRect(QRectF(mid - 0.5, 0, 1, h))


class EraeMirror(QWidget):
    """The page the Erae shows, as the app draws it: the surface's last
    frame scaled up, the fingers on it, and the control under the mouse
    named through `hovered`. `surface` is the EraeSurface, or None."""

    hovered = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.surface = None
        self.fingers = {}                # (zone, fid) -> (x, y, z)
        self.setMinimumSize(ERAE_W * 7, ERAE_H * 7)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(100)
        self._bound = None

    def _geometry(self):
        cell = min(self.width() / ERAE_W, (self.height() - 18) / ERAE_H)
        x0 = (self.width() - cell * ERAE_W) / 2
        y0 = 18 + (self.height() - 18 - cell * ERAE_H) / 2
        return cell, x0, y0

    def _on_finger(self, zone, code, fid, x, y, z):
        if code == 2:
            self.fingers.pop((zone, fid), None)
        else:
            self.fingers[(zone, fid)] = (x, y, z)

    def _bind(self):
        surf = self.surface
        if surf is self._bound:
            return
        if self._bound is not None:
            try:
                self._bound.finger.disconnect(self._on_finger)
            except (RuntimeError, TypeError):
                pass
        if surf is not None:
            surf.finger.connect(self._on_finger)
        self._bound = surf
        self.fingers.clear()

    def paintEvent(self, _event):
        self._bind()
        surf = self.surface
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(12, 10, 10))
        cell, x0, y0 = self._geometry()
        frame = surf.last_frame() if surf is not None else None
        zone = surf.active_zone() if surf is not None else None
        if frame is None:
            p.setPen(QColor(242, 233, 229, 110))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "no page" if surf is not None else "surface off")
            return
        for row, cells in enumerate(frame):
            yy = y0 + (ERAE_H - 1 - row) * cell
            for col, (r, g, b) in enumerate(cells):
                if (r, g, b) == (0, 0, 0):
                    c = QColor(22, 20, 20)
                else:                    # the LEDs read brighter than their numbers
                    c = QColor(*(int(255 * (v / 127.0) ** 0.6) for v in (r, g, b)))
                p.fillRect(QRectF(x0 + col * cell + 0.5, yy + 0.5, cell - 1, cell - 1), c)
        p.setPen(QPen(QColor(255, 255, 255, 220), 2))
        p.setBrush(Qt.NoBrush)
        for (z, _fid), (fx, fy, fz) in self.fingers.items():
            if z != zone:
                continue
            r = cell * (0.6 + 2.0 * min(1.0, fz / 0.25))
            p.drawEllipse(QPointF(x0 + fx * cell, y0 + (ERAE_H - fy) * cell), r, r)
        pg = surf.active_page()
        if pg is not None:
            p.setPen(QColor(242, 233, 229, 160))
            p.drawText(QRectF(x0, 0, cell * ERAE_W, 16), Qt.AlignLeft | Qt.AlignVCenter,
                       f"N{surf.active + 1}  {pg.title}")

    def mouseMoveEvent(self, event):
        surf = self.surface
        pg = surf.active_page() if surf is not None else None
        if pg is None:
            self.hovered.emit("")
            return
        cell, x0, y0 = self._geometry()
        pos = event.position()
        fx = (pos.x() - x0) / cell
        fy = ERAE_H - (pos.y() - y0) / cell
        wd = pg.widget_at(fx, fy)
        self.hovered.emit("" if wd is None else f"{wd.label}: " + ", ".join(wd.idents()))


class PerfMeter(QWidget):
    """The section at a glance, big enough to read from the pedals: what
    plays (kind, root, tempo) in large type, and the loop's bars as a row of
    cells, the current one lit with a sweep running through it over the
    bar, four beat dots under each, a mark after every phrase. A held
    repeat draws its window over the cells in yellow, the cell's own
    playhead inside it and a chevron for the direction, while the sweep
    underneath goes on showing the transport the loop will come back to."""

    ACCENT = QColor(212, 48, 30, 230)
    REPEAT = QColor(255, 214, 80, 70)
    REPEAT_EDGE = QColor(255, 214, 80, 240)
    SWEEP = QColor(255, 130, 90, 170)
    PASSED = QColor(134, 28, 20, 110)
    IDLE = QColor(28, 11, 9, 117)
    TEXT = QColor(242, 233, 229)
    DIM = QColor(242, 233, 229, 110)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = "no section"
        self.bars = 8
        self.pos = None                  # bars into the loop (float), None = nothing plays
        self.repeat = None               # (start, length, backwards, heard), in bars
        self.setMinimumHeight(132)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_state(self, title, bars=None, pos=None, repeat=None):
        """`title` the big line; `bars` the loop's length; `pos` the
        position in bars (0 <= pos < bars), or None while nothing plays;
        `repeat` a held repeat's window as (start, length, backwards,
        heard), start and heard in bars into the loop, length in bars."""
        bars = self.bars if bars is None else max(1, int(bars))
        if (title, bars, pos, repeat) == (self.title, self.bars, self.pos, self.repeat):
            return
        self.title, self.bars, self.pos, self.repeat = title, bars, pos, repeat
        self.update()

    def _x(self, b, cw, gap, end=False):
        """The x of bar position `b` (bars into the loop); `end` for the
        end of a span, so a whole bar ends at its own cell's right edge."""
        i = int(b)
        f = b - i
        if end and f == 0.0 and i > 0:
            i, f = i - 1, 1.0
        return i * (cw + gap) + cw * f

    def _paint_repeat(self, p, cw, gap, top, ch):
        """The repeat's window over the cells (two pieces round the loop's
        end), the playhead in it, a chevron for the direction."""
        start, length, rev, heard = self.repeat
        n = self.bars
        end = start + length
        spans = [(start, min(end, n))]
        if end > n:
            spans.append((0.0, end - n))
        p.setPen(QPen(self.REPEAT_EDGE, 2))
        p.setBrush(self.REPEAT)
        for a, b in spans:
            x0, x1 = self._x(a, cw, gap), self._x(b, cw, gap, end=True)
            p.drawRoundedRect(QRectF(x0, top - 4, max(4.0, x1 - x0), ch + 8), 5, 5)
        # the chevron at the leading edge: where the cell runs towards
        a, b = spans[0] if not rev else spans[-1]
        x0, x1 = self._x(a, cw, gap), self._x(b, cw, gap, end=True)
        s = min(10.0, max(4.0, (x1 - x0) / 3))
        cy = top - 4 + s
        xe = x1 - 3 if not rev else x0 + 3
        d = -s if not rev else s
        p.setPen(Qt.NoPen)
        p.setBrush(self.REPEAT_EDGE)
        p.drawPolygon(QPolygonF([QPointF(xe, cy), QPointF(xe + d, cy - s),
                                 QPointF(xe + d, cy + s)]))
        if heard is not None:                      # the cell's own playhead, white
            xh = self._x(heard % n, cw, gap)
            p.setBrush(self.TEXT)
            p.drawRect(QRectF(xh - 1.5, top - 4, 3, ch + 8))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        f = p.font()
        f.setPixelSize(28)
        f.setBold(True)
        p.setFont(f)
        p.setPen(self.TEXT if self.pos is not None else self.DIM)
        p.drawText(QRectF(2, 0, w - 4, 40), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        n = self.bars
        gap, top = 6.0, 48.0
        ch = h - top - 6
        cw = (w - gap * (n - 1)) / n
        cur = None if self.pos is None else int(self.pos) % n
        frac = 0.0 if self.pos is None else self.pos - int(self.pos)
        num = QFont(f)
        num.setPixelSize(max(12, int(ch * 0.5)))
        for i in range(n):
            x = i * (cw + gap)
            r = QRectF(x, top, cw, ch)
            p.setPen(Qt.NoPen)
            if i == cur:
                p.setBrush(self.ACCENT)
            elif cur is not None and i < cur:
                p.setBrush(self.PASSED)
            else:
                p.setBrush(self.IDLE)
            p.drawRoundedRect(r, 8, 8)
            if i == cur and frac > 0:                 # the sweep through the bar
                p.setBrush(self.SWEEP)
                p.drawRoundedRect(QRectF(x, top, max(8.0, cw * frac), ch), 8, 8)
            p.setFont(num)
            p.setPen(self.TEXT if i == cur else self.DIM)
            p.drawText(QRectF(x, top, cw, ch - 10), Qt.AlignCenter, str(i + 1))
            p.setPen(Qt.NoPen)
            for b in range(4):                        # the beats of the bar
                lit = i == cur and frac * 4 >= b
                p.setBrush(self.TEXT if lit else QColor(242, 233, 229, 50))
                cx = x + cw * (b + 0.5) / 4
                p.drawEllipse(QRectF(cx - 3, top + ch - 10, 6, 6))
            if (i + 1) % PHRASE_BARS == 0 and i + 1 < n:   # the phrase's end
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 120, 90, 220))
                p.drawRect(QRectF(x + cw + gap / 2 - 1, top - 6, 2, ch + 12))
        if self.repeat is not None and self.pos is not None:
            self._paint_repeat(p, cw, gap, top, ch)
        p.end()


class DragSpin(QObject):
    """Every number box a fader for a finger: press on one and drag up or
    down, the value follows, the box's whole range over DRAG_PX (300 px)
    so a short swipe is a nudge and a long one the whole travel. A tap or
    a sideways drag is left to the box (the caret, the text selection),
    the drag takes over past a few pixels up or down. One filter on the
    application, so every spinbox has it, touch or mouse (a finger is a
    mouse to a widget that takes no touch events)."""
    DRAG_PX = 300.0                       # the full range over this much travel
    ARM_PX = 8                            # vertical travel before the drag takes over

    def __init__(self, parent=None):
        super().__init__(parent)
        self._box = None                  # the spinbox under the finger
        self._y0 = 0.0
        self._v0 = 0.0
        self._armed = False

    @staticmethod
    def _spin_of(obj):
        if isinstance(obj, QAbstractSpinBox):
            return obj
        if isinstance(obj, QLineEdit) and isinstance(obj.parent(), QAbstractSpinBox):
            return obj.parent()
        return None

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            box = self._spin_of(obj)
            if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
                box = None                 # a Ctrl- or Alt-click: the MIDI learn's, not a drag
            if box is not None and box.isEnabled() and not box.isReadOnly():
                self._box, self._armed = box, False
                self._y0 = event.globalPosition().y()
                self._v0 = float(box.value())
            return False
        if self._box is None:
            return False
        if t == QEvent.MouseMove:
            try:
                dy = self._y0 - event.globalPosition().y()
            except RuntimeError:
                self._box = None
                return False
            if not self._armed:
                if abs(dy) < self.ARM_PX:
                    return False
                self._armed = True
                le = self._box.lineEdit() if hasattr(self._box, "lineEdit") else None
                if le is not None:
                    le.deselect()
            box = self._box
            lo, hi = box.minimum(), box.maximum()
            v = self._v0 + dy / self.DRAG_PX * (hi - lo)
            v = max(lo, min(hi, v))
            if isinstance(box, QDoubleSpinBox):
                box.setValue(round(v, box.decimals()))
            else:
                box.setValue(int(round(v)))
            return True                    # not a text selection
        if t == QEvent.MouseButtonRelease:
            armed, self._box, self._armed = self._armed, None, False
            return armed
        return False


class HoverTabs(QObject):
    """Switch a QTabWidget's page by hovering its tabs: after `delay` ms on
    one tab it becomes current, no click needed. Install with
    HoverTabs(tabwidget). The dwell keeps a sweep across the bar from
    flipping through every page."""

    def __init__(self, tabs, delay=120):
        super().__init__(tabs)
        self.tabs, self.bar = tabs, tabs.tabBar()
        self._idx = -1
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay)
        self._timer.timeout.connect(self._fire)
        self.bar.setMouseTracking(True)
        self.bar.installEventFilter(self)

    def eventFilter(self, obj, event):
        t = event.type()
        if t in (QEvent.HoverMove, QEvent.MouseMove):
            idx = self.bar.tabAt(event.position().toPoint())
            if idx != self._idx:
                self._idx = idx
                if idx >= 0 and idx != self.tabs.currentIndex():
                    self._timer.start()
                else:
                    self._timer.stop()
        elif t in (QEvent.HoverLeave, QEvent.Leave):
            self._idx = -1
            self._timer.stop()
        return False

    def _fire(self):
        if 0 <= self._idx < self.tabs.count():
            self.tabs.setCurrentIndex(self._idx)


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
                [sys.executable, "-u", "-X", "utf8", str(GACHA), job_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", **E.NO_WINDOW)
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
    vst_params_changed = Signal()               # from the inserts' watch thread
    vst_editor_closed = Signal(str)             # from the engine: a plugin's window closed
    engine_log = Signal(str)                    # the engine process's log lines

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gacha")
        self.resize(1050, 640)
        self.worker = None

        # ---------- audio player ----------
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(1.0)
        # songs play through the app's audio stream when one is open (racks,
        # stems, F1-F4), through Qt otherwise
        self.player = AppPlayer(lambda: self.through if (getattr(self, "through", None)
                                                        is not None and self.through.running)
                                else None, self)
        self.player.setAudioOutput(self.audio_out)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self._segment = None        # (start_ms, end_ms) while auditioning a slice
        self._sec_bounds = []       # section start times (ms) of the playing file
        self._sec_idx = -1          # which section the playhead is in
        self._last_switch_ms = 0.0  # song position of the last clip switch
        self._video_deck = []       # video_random: the shuffled order left to deal
        self._video_deck_pool = []  # ...and the ticked videos it was dealt from
        self._live_switch_t = time.monotonic()   # last automatic clip switch (live, sections)
        self._sections = []         # full section dicts of the playing file
        self._beat_ms = 500.0
        self._env = None            # loudness envelope, one value per 50 ms
        self._outro = None          # (start_ms, end_ms) of the ending effect
        self._travel = 0            # mouse distance since the interface hid
        self._last_mouse = None
        self._cursor_hidden = False
        self._fullscreen = False                    # see toggle_fullscreen
        self._was_maximized = False                 # ...and what to go back to
        self._look = {}             # per-section random multipliers per effect
        self._look_idx = None       # (section, 4-bar phrase) of the current look
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
        self._clean_bar = None      # (section, bar) last rolled for clean video
        self._song_visuals_pending = False   # a song is loading: its clip and shader wait
        self._clean_on = False      # this bar shows the video with no effects
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
        # Ableton Link: with the Settings tab's Link button on, the live clock
        # shares the network session's tempo (gacha_link.py), polled
        self.link = None
        if LINK_AVAILABLE:
            try:
                self.link = LinkClock(self.tempo.bpm)
            except Exception as e:                  # a broken wheel must not take the app down
                print(f"ableton link unavailable: {e}", file=sys.stderr)
        self._link_timer = QTimer(self)
        self._link_timer.timeout.connect(self._poll_link)
        self._link_ticks = 0
        self._link_peers = -1                       # last peer count shown in the log
        self._link_playing = False                  # the session's transport as last seen / sent
        self._live_timer = QTimer(self)             # level meter refresh
        self._live_timer.timeout.connect(self._update_live_meter)
        self.through = None                         # AudioProxy: the audio stream, when open
        host = audio_host()                         # the engine process (gacha_audioproc), from now
        host.log = self.engine_log.emit
        host.on_event("editor_closed", self.vst_editor_closed.emit)
        self.engine_log.connect(lambda t: self.log.appendPlainText(f"engine: {t}"))
        self.vst_editor_closed.connect(self._vst_after_editor)
        # moments of the show, in RAM; each one also lands in videos/ as a set
        # folder named after tonight, so tomorrow the takes are backdrops
        self.takes = TakeStore(parent=self, archive_dir=VIDEOS_DIR /
                               f"takes_{time.strftime('%Y%m%d')}")
        self.takes.saved.connect(self._take_saved)
        # the live section engine: material carved from the takes, sections
        # rendered on a thread, played by the audio thread's SectionPlayer
        self._perf_material = None                  # (key, Material)
        self._pad_rr = {}                            # the drum pads: round-robin per role
        self._pad_cache = {}                         # (material, role, variant, layer, sr) -> clip
        self._burst = None                          # (t0, until, take, off): a burst's picture
        self._perf_hold = None                      # the last take frame shown: held over gaps
        self._head_tape = False
        self._gran_clock = 0.0                 # the granular picture's grain clock
        self._gran_t = None                     # a file tape slowed under the section's head
        self._perf_drop_t = None                    # monotonic time a break dropped into a section
        self._perf_ready = None                     # (key, Section, stems48) pre-rendered
        self._perf_key_last, self._perf_key_t = None, 0.0   # the knobs' last change, see _perf_refresh_ready
        self._perf_poll_n = 0
        self._perf_busy = None                      # ("section"|"shot", key) rendering now
        self._perf_result = None                    # set by the render thread
        self._perf_want = False                     # play the next result as soon as it lands
        self._shuffle_queue = []                    # parts asked for a shuffle, in turn
        self._shuffle_landing = []                  # parts handed to the player, not yet on the bar
        self._perf_playing = None                   # Section on the player
        self._perf_current = None                   # (Section, stems, sr): the last one played,
                                                    # what stop / Link start bring back
        self._perf_sounding = False                 # a section was on the player at the last poll
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
        self.log = QPlainTextEdit(readOnly=True)     # its own tab, see _tab_log
        self.log.setMaximumBlockCount(2000)
        params_box = self._build_params()

        self.generate_btn = QPushButton("Generate song")
        self.generate_btn.setToolTip("Render a song from the ticked samples with the "
                                     "Generate and Sections tabs' settings, in a "
                                     "subprocess (the log follows it); it lands in "
                                     "output/ and in the list, ready to play")
        self.generate_btn.setObjectName("generate")     # the orange one, see STYLE
        self.generate_btn.setMinimumHeight(36)
        self.generate_btn.clicked.connect(self.generate)
        fs_btn = QPushButton("Fullscreen (F11)")
        fs_btn.setObjectName("small")
        fs_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        fs_btn.clicked.connect(self.toggle_fullscreen)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addWidget(self.generate_btn, stretch=1)
        top_row.addWidget(fs_btn, alignment=Qt.AlignTop)

        left = QVBoxLayout()
        left.addWidget(params_box, stretch=1)
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
        refresh_btn.setObjectName("small")
        refresh_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        refresh_btn.setToolTip("Read the samples, videos and outputs folders again")
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
            if tree is self.videos_tree:
                box.addLayout(self._build_fetch_row())
            else:
                box.addLayout(self._build_prep_row())
            w = QWidget()
            w.setLayout(box)
            tabs.addTab(w, title)
        tabs.addTab(self.outputs_list, "Outputs")
        tabs.addTab(self._scrolling(self._build_mixer()), "Mixer")
        # the lists pane is 40 % of the interface: pages scroll rather
        # than force the pane wide
        for i in range(tabs.count()):
            if tabs.tabText(i) in ("Samples", "Videos"):
                tabs.widget(i).setMinimumWidth(1)
        right_min = 300

        # ---------- player bar ----------
        self.now_playing = QLabel("—")
        self.now_playing.setStyleSheet("font-weight: bold;")
        self.play_btn = QPushButton("Play")
        self.play_btn.setToolTip("Play the selected render through the audio stream "
                                 "and the racks; pressed while playing, pause")
        self.play_btn.clicked.connect(self.toggle_play)
        stop_btn = QPushButton("Stop")
        stop_btn.setToolTip("Stop the render and go back to its start")
        stop_btn.clicked.connect(self.player.stop)
        self.pos_slider = SeekSlider(Qt.Horizontal)
        self.pos_slider.setToolTip("Where the render is; click anywhere on it to jump there")
        self.pos_slider.sliderMoved.connect(self.player.setPosition)
        self.time_lbl = QLabel("0:00 / 0:00")
        self.vol = QSlider(Qt.Horizontal, maximumWidth=100)
        self.vol.setRange(0, 100)
        self.vol.setValue(100)
        self.vol.setToolTip("Volume of the audio stream: songs, sections, the live input. "
                            "Unity at the top; the level of the room is the sound system's business")
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
        right.addLayout(top_row)
        right.addWidget(tabs, stretch=1)
        right.addWidget(refresh_btn, alignment=Qt.AlignRight)
        right.addWidget(self.now_playing)
        right.addLayout(bar)
        right_w = QWidget()
        right_w.setLayout(right)
        right_w.setMinimumWidth(right_min)

        split = QSplitter()
        split.addWidget(left_w)
        split.addWidget(right_w)
        left_w.setMinimumWidth(560)      # four-spinbox rows need the room
        # the file lists as narrow as they go at launch, the parameters
        # take the rest; a dragged handle sets a ratio that then survives
        # resizes
        self.split = split
        self._split_right_min = right_min
        self._split_ratio = None            # None: the lists at their minimum width
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
        if self.settings.value("video/off", False, type=bool):
            self.video_off.setChecked(True)      # the backdrop exists now: it applies
        self.setCentralWidget(self.backdrop)
        # the stats HUD: a label over the picture, top right, above the interface
        self.hud = QLabel(self.backdrop)
        self.hud.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hud.setStyleSheet(
            "QLabel { background: rgba(16, 9, 7, 170); color: rgba(242, 233, 229, 230); "
            "font-family: monospace; font-size: 12px; padding: 4px 8px; "
            "border-radius: 6px; }")
        self.hud.hide()
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_prev = None                  # (time, paints, video frames)
        self._proc = psutil.Process() if psutil else None
        self._mem_free = None                       # (monotonic, GB) the last free-memory reading
        if self._proc is not None:
            self._proc.cpu_percent(None)          # prime the counters
            psutil.cpu_percent(None)
        self._apply_shader_choice()
        if hasattr(self.backdrop, "prewarm"):
            self.backdrop.prewarm(shader_files())   # compile all shaders early
        # shader compile errors from the GL backdrop land in the render log
        self._gl_log_timer = QTimer(self)
        self._gl_log_timer.timeout.connect(self._drain_gl_log)
        self._gl_log_timer.start(500)
        self.ui = split
        QApplication.instance().installEventFilter(self)
        # a finger (or the mouse) dragging up or down on any number box
        self._drag_spin = DragSpin(self)
        QApplication.instance().installEventFilter(self._drag_spin)
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
            "discard": self.discard_take,
            "clear_takes": self.clear_takes,
            "tap": self.tap_tempo,
            "downbeat": self.live_downbeat,
            "live": self.live_on.toggle,
            "drop": self.live_drop_key,
            "next": self.perform_next,
            "stop": self.perform_stop,
            "event": self.perform_event,
            "burst": self.perform_burst,
            "glitch": lambda: self.perform_break("glitch"),
            "break": lambda: self.perform_break("break"),
            "burst_len": self._burst_len_frac,
            "repeat": lambda: self._rep_hold("midi", True),
            "rep_finer": lambda: self._rep_div_step(1),
            "rep_coarser": lambda: self._rep_div_step(-1),
            "rep_div": self._rep_div_frac,
            "half": self.half_btn.toggle,
            "double": self.double_btn.toggle,
            "speed_sync": lambda x: self._combo_frac(self.speed_sync, x),
            "pitch_lock": lambda: self.pitch_mode.setCurrentIndex(
                1 - self.pitch_mode.currentIndex()),
            "brake": lambda: self._brake_hold("midi", True),
            "brake_len": lambda x: self._combo_frac(self.brake_len, x),
            "freeze": lambda: self._freeze_hold("midi", True),
            "nudge_down": lambda: self._nudge_hold(-1, "midi", True),
            "nudge_up": lambda: self._nudge_hold(1, "midi", True),
            "rep_earlier": lambda: self.rep_pos.setValue(self.rep_pos.value() - 1),
            "rep_later": lambda: self.rep_pos.setValue(self.rep_pos.value() + 1),
            "rep_pos": lambda x: self.rep_pos.setValue(
                round((max(0.0, min(1.0, x)) - 0.5) * 2 * REPEAT_RANGE)),
            "rep_rev": self.rep_rev.toggle,
            "ab_down": lambda: self.perf_ab.setValue(self.perf_ab.value() - 10),
            "ab_up": lambda: self.perf_ab.setValue(self.perf_ab.value() + 10),
            "ab": lambda x: self.perf_ab.setValue(round(x * 100)),
        }
        for n in range(1, 10):
            self.actions[f"shader_{n}"] = lambda n=n: self.select_shader_key(n)
        for stem in STEMS:
            self.actions[f"stem_{stem}"] = self.perf_stems[stem].toggle
            self.actions[f"randomize_{stem}"] = lambda stem=stem: self.perform_shuffle(stem)
        for v in VOICES:
            self.actions[f"voice_{v}"] = self.perf_voices[v].toggle
            self.actions[f"solo_{v}"] = self.perf_solos[v].toggle
        for v in KIT_PARTS:
            self.actions[f"random_{v}"] = lambda v=v: self.perform_shuffle(v)
        self.actions["voices_reset"] = self._voices_reset
        for v in DRUM_ROLES:
            self.actions[f"hit_{v}"] = lambda vel=1.0, v=v: self.perform_hit(v, vel)
        self.actions["quant_next"] = lambda: self._combo_next(self.perf_quant)
        # MIDI-only buttons
        self.actions.update({
            "randomize_fx": self._randomize_video,
            "grade_neutral": self._reset_grade,
            "blend_next": lambda: self._combo_next(self.shader_blend),
            "kind_next": lambda: self._combo_next(self.perf_kind),
            "link": self.link_on.toggle,
            "link_play": self._link_press,
            "picture_next": lambda: self._combo_next(self.perf_picture),
            "dissolve_next": lambda: self._combo_next(self.perf_dissolve),
        })
        # knobs: a CC's 0..1 spread over the widget's range
        self.actions.update({
            "volume": self._frac_setter(self.vol),
            "sync": self._frac_setter(self.through_sync),
            "shader_mix": self._frac_setter(self.shader_mix),
            "shader_pick": self._pick_shader_frac,
        })
        for k, sp in self.grade.items():
            self.actions[f"grade_{k}"] = self._frac_setter(sp)
        for k, sp in self.vfx.items():
            self.actions[f"fx_{k}"] = self._frac_setter(sp)
        # the audio racks: every cell, the focused set, the focus buttons
        for (ch, key), sp in self.afx.items():
            ident = f"avol_{ch}" if key == "vol" else f"afx_{ch}_{key}"
            self.actions[ident] = self._frac_setter(sp)
        for k, *_ in AUDIO_FX:
            self.actions[f"afx_{k}"] = self._afx_focused(k)
        self.actions["avol"] = self._afx_focused("vol")
        for ch in ALL_CHANNELS:
            self.actions[f"focus_{ch}"] = lambda ch=ch: self.afx_focus.setCurrentText(ch)
        self.actions["inserts"] = self.vst_bypass.toggle
        for k, cb in self.vfx_on.items():
            self.actions[f"vfx_{k}"] = cb.toggle
        for name, cb in self.shader_on.items():
            self.actions[f"shd_{name}"] = cb.toggle
        self.actions["video_prev"] = lambda: self._video_step(-1)
        self.actions["video_next"] = lambda: self._video_step(1)
        self.actions["video_random"] = self._video_random
        self.actions["video_second"] = self.video_second.toggle
        self.actions["midi_out"] = self.midi_out_on.toggle
        self.actions["metronome"] = self.midi_metro.toggle
        self.actions["video_off"] = self.video_off.toggle
        for n in range(1, VST_KNOBS + 1):
            self.actions[f"vst_{n}"] = lambda x, n=n: self.inserts.knob(n - 1, x)
        # the LFOs: each slot's switch and depth, the trigger, the panic
        for n in range(1, LFO_SLOTS + 1):
            self.actions[f"lfo_{n}"] = self.lfo_rows[n - 1]["on"].toggle
            self.actions[f"lfo_{n}_depth"] = lambda x, n=n: self._lfo_depth_frac(n - 1, x)
        self.actions["lfo_trig"] = lambda: self.lfo.trigger(time.monotonic())
        self.actions["lfo_off"] = self._lfo_all_off
        # the scenes: a load and a save button per slot, the morph switch
        for n in range(1, SCENE_SLOTS + 1):
            self.actions[f"scene_load_{n}"] = lambda n=n: self._scene_load(n - 1)
            self.actions[f"scene_save_{n}"] = lambda n=n: self._scene_save(n - 1)
        self.actions["scene_morph"] = self.scene_morph_on.toggle
        # the macros: each one's knob (its slider fans out on every change)
        for n in range(1, MACRO_SLOTS + 1):
            self.actions[f"macro_{n}"] = self._frac_setter(self.macro_rows[n - 1]["knob"])
        # the knobs the LFOs read: every widget-backed one by its id (the
        # rest, shader pick, burst length, the repeat division and the VST
        # knobs, are read by _lfo_get on their own terms)
        self.knob_widgets = {"volume": self.vol, "sync": self.through_sync,
                             "shader_mix": self.shader_mix, "ab": self.perf_ab,
                             "rep_pos": self.rep_pos}
        self.knob_widgets.update({f"grade_{k}": sp for k, sp in self.grade.items()})
        self.knob_widgets.update({f"fx_{k}": sp for k, sp in self.vfx.items()})
        self.knob_widgets.update({(f"avol_{ch}" if key == "vol" else f"afx_{ch}_{key}"): sp
                                  for (ch, key), sp in self.afx.items()})
        self.knob_widgets.update({f"macro_{n}": self.macro_rows[n - 1]["knob"]
                                  for n in range(1, MACRO_SLOTS + 1)})
        # the holds: a release handler next to the press, and the keys the
        # event filter watches (a shortcut would fire on the press alone)
        # the fragments: the F key's slice, a note per slice, the rows' knobs
        self.actions["frag"] = lambda: self._frag_hold_pick("midi", True)
        for n in range(1, 17):
            self.actions[f"frag_{n}"] = lambda n=n: self._frag_hold(n, "midi", True)
        self.frag_combos = {"frag_div": self.frag_div, "frag_slice": self.frag_slice,
                            "frag_size": self.frag_size, "frag_rate": self.frag_rate}
        # the effect settings (Audio FX tab): the menus join the combos read
        # as knobs (their actions below), the rest are knobs outright
        for ident, w in self.fxset.items():
            if isinstance(w, QComboBox):
                self.frag_combos[ident] = w
            else:
                self.actions[ident] = self._frac_setter(w)
                self.knob_widgets[ident] = w
        for ident, w in (("freeze_size", self.freeze_size), ("freeze_smear", self.freeze_smear),
                         ("freeze_pitch", self.freeze_pitch), ("freeze_a", self.freeze_a),
                         ("freeze_r", self.freeze_r)):
            self.actions[ident] = self._frac_setter(w)
            self.knob_widgets[ident] = w
        for ident, combo in self.frag_combos.items():
            self.actions[ident] = lambda x, c=combo: self._combo_frac(c, x)
        for ident, w in (("frag_motion", self.frag_motion), ("frag_pitch", self.frag_pitch),
                         ("frag_speed", self.frag_speed), ("frag_repeats", self.frag_repeats),
                         ("frag_shape", self.frag_shape), ("frag_a", self.frag_a),
                         ("frag_d", self.frag_d), ("frag_s", self.frag_s), ("frag_r", self.frag_r)):
            self.actions[ident] = self._frac_setter(w)
            self.knob_widgets[ident] = w
        self.actions["frag_dir"] = lambda: self._combo_next(self.frag_dir)
        self.actions["frag_play"] = lambda: self._combo_next(self.frag_play)
        # the Model:Cycles tab: every cell, the focused set and its focus
        # buttons, the retrig holds and the retrig's settings
        for (t, key), w in self.mc.items():
            ident = self._mc_ident(t, key)
            if isinstance(w, QCheckBox):
                self.actions[ident] = w.toggle
            elif isinstance(w, QComboBox):
                self.actions[ident] = lambda x, c=w: self._combo_frac(c, x)
                self.frag_combos[ident] = w
            else:
                self.actions[ident] = self._frac_setter(w)
                self.knob_widgets[ident] = w
        for key, *_ in CYCLES_PARAMS:
            self.actions[f"mc_{key}"] = self._mc_focused(key)
        for t in MC_TRACKS:
            self.actions[f"mc_focus_{t}"] = lambda t=t: self.mc_focus.setCurrentIndex(t - 1)
            self.actions[f"mc_retrig_{t}"] = lambda t=t: self._mc_retrig_hold(t, "midi", True)
        for ident, c in (("mc_retrig_rate", self.mc_retrig_rate),
                         ("mc_retrig_len", self.mc_retrig_len)):
            self.actions[ident] = lambda x, c=c: self._combo_frac(c, x)
            self.frag_combos[ident] = c
        for ident, w in (("mc_retrig_fade", self.mc_retrig_fade),
                         ("mc_retrig_vel", self.mc_retrig_vel)):
            self.actions[ident] = self._frac_setter(w)
            self.knob_widgets[ident] = w
        self.releases = {"repeat": lambda: self._rep_hold("midi", False),
                         "brake": lambda: self._brake_hold("midi", False),
                         "freeze": lambda: self._freeze_hold("midi", False),
                         "frag": lambda: self._frag_hold_pick("midi", False),
                         **{f"frag_{n}": (lambda n=n: self._frag_hold(n, "midi", False))
                            for n in range(1, 17)},
                         "nudge_down": lambda: self._nudge_hold(-1, "midi", False),
                         "nudge_up": lambda: self._nudge_hold(1, "midi", False)}
        self.holds = {"repeat": self._rep_hold, "brake": self._brake_hold,
                      "freeze": self._freeze_hold,
                      "frag": self._frag_hold_pick,
                      **{f"frag_{n}": (lambda src, on, n=n: self._frag_hold(n, src, on))
                         for n in range(1, 17)},
                      "nudge_down": lambda src, on: self._nudge_hold(-1, src, on),
                      "nudge_up": lambda src, on: self._nudge_hold(1, src, on)}
        for t in MC_TRACKS:
            self.releases[f"mc_retrig_{t}"] = lambda t=t: self._mc_retrig_hold(t, "midi", False)
            self.holds[f"mc_retrig_{t}"] = lambda src, on, t=t: self._mc_retrig_hold(t, src, on)
        self._hold_keys = {QKeySequence(key).toString(): ident
                           for _g, ident, key, _w in KEYMAP if key and ident in HOLD}
        missing = [ident for _g, ident, _k, _w in KEYMAP if ident not in self.actions]
        assert not missing, f"actions without a handler: {missing}"
        self.shortcuts = []
        for _g, ident, key, _w in KEYMAP:
            if not key or ident in CONTINUOUS or ident in HOLD:
                continue
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(self.actions[ident])
            self.shortcuts.append(sc)
        self._midi_setup()
        self._update_afx_status()
        self._lfo_timer = QTimer(self)
        self._lfo_timer.setTimerType(Qt.PreciseTimer)
        self._lfo_timer.timeout.connect(self._lfo_tick)
        self._lfo_timer.start(10)
        # every combo's list is a window of its own, and Windows does not
        # keep it above a fullscreen window: in F11 the list opened out of
        # sight, behind the picture. Stay-on-top puts it in the band above.
        for cb in self.findChildren(QComboBox):
            popup = cb.view().window()
            popup.setWindowFlags(popup.windowFlags() | Qt.WindowStaysOnTopHint)

        self.refresh_lists()
        # the audio stream: open at launch on the remembered output, with the
        # live input if it was on; songs use Qt only while it is not running
        self.through_on.blockSignals(True)
        self.through_on.setChecked(self.settings.value("audio/live_in", False, type=bool))
        self.through_on.blockSignals(False)
        self.through_excl.blockSignals(True)
        self.through_excl.setChecked(self.settings.value("audio/exclusive", True, type=bool))
        self.through_excl.blockSignals(False)
        blk = self.settings.value("audio/block", "480" if sys.platform == "win32" else "256")
        if self.through_block.findText(str(blk)) >= 0:
            self.through_block.blockSignals(True)
            self.through_block.setCurrentText(str(blk))
            self.through_block.blockSignals(False)
        sr = self.settings.value("audio/sr", "48000")
        if self.through_sr.findText(str(sr)) >= 0:
            self.through_sr.blockSignals(True)
            self.through_sr.setCurrentText(str(sr))
            self.through_sr.blockSignals(False)
        ring = self.settings.value("audio/ring", 0, type=int)
        if self.through_ring.findData(ring) >= 0:
            self.through_ring.blockSignals(True)
            self.through_ring.setCurrentIndex(self.through_ring.findData(ring))
            self.through_ring.blockSignals(False)
        for cb, key in ((self.audio_pro, "audio/pro_audio"), (self.audio_high, "audio/high_priority"),
                        (self.audio_gc, "audio/gc_off")):
            cb.blockSignals(True)
            cb.setChecked(self.settings.value(key, False, type=bool))
            cb.blockSignals(False)
        self._priority_changed(save=False)
        self._io_ready = True
        self._keys_into_tips()                  # "(N)" off the buttons, into the tips
        self._live_defaults = self._live_state()  # as built: the values-only reset's target
        self._live_restore()                    # the knobs as they were left
        self._live_timer = QTimer(self)         # and kept from then on, every few seconds
        self._live_timer.setInterval(3000)
        self._live_timer.timeout.connect(self._live_save)
        self._live_timer.start()
        QTimer.singleShot(600, self._vst_restore)      # slow plugin loads, after the window shows
        QTimer.singleShot(0, self._reopen_stream)

    def _priority_changed(self, save=True):
        """The Priority ticks of the Settings tab, all three for the engine
        process (gacha_audioproc), where the sound is: its process class
        and its GC at once, its audio thread's MMCSS class on the next
        block (gacha_audio reads the flag there); remembered. This process
        is the interface, it runs as any desktop app."""
        high, off, pro = self.audio_high.isChecked(), self.audio_gc.isChecked(), self.audio_pro.isChecked()
        audio_host().priority(high, off)
        if self.through is not None:
            self.through.pro_audio = pro
        if save:
            self.settings.setValue("audio/pro_audio", pro)
            self.settings.setValue("audio/high_priority", high)
            self.settings.setValue("audio/gc_off", off)
            self.log.appendPlainText(f"audio: priority: pro audio {'on' if pro else 'off'}, "
                                     f"process {'high' if high else 'normal'}, "
                                     f"gc {'off' if off else 'on'}")

    # ---------- stats HUD ----------
    def _toggle_stats(self, on):
        if on:
            self._stats_prev = None
            self._update_stats()
            self._stats_timer.start(500)
            self.hud.show()
            self.hud.raise_()
        else:
            self._stats_timer.stop()
            self.hud.hide()

    def _update_stats(self):
        """Half a second of the machine: paint and video rates from the frame
        counters, GPU paint time, CPU, memory, and the audio thread's load."""
        now = time.monotonic()
        paints = getattr(self.backdrop, "frames", 0)
        src = getattr(self.backdrop, "source", None)
        vframes = src.frames if src is not None else 0
        parts = []
        if self._stats_prev is not None:
            t0, p0, v0 = self._stats_prev
            dt = max(1e-3, now - t0)
            parts.append(f"paint {(paints - p0) / dt:5.1f} fps")
            gpu = getattr(self.backdrop, "frame_ms", None)
            if gpu:
                parts[-1] += f"  {gpu:4.1f} ms"
            parts.append(f"video {(vframes - v0) / dt:5.1f} fps")
        else:
            parts.append("paint  ... fps")
        self._stats_prev = (now, paints, vframes)
        if src is not None and getattr(src, "_size", None):
            w, h = src._size
            parts[-1] += f"  {w}x{h}"
        if self._proc is not None:
            cpu_all = psutil.cpu_percent(None)
            cpu_app = self._proc.cpu_percent(None) / max(1, psutil.cpu_count() or 1)
            parts.append(f"cpu   {cpu_all:5.1f} %  app {cpu_app:5.1f} %")
            rss = self._proc.memory_info().rss
            parts.append(f"mem   {rss / 1e9:5.2f} GB app  {self._free_memory_gb():5.1f} GB free")
            takes = self.takes.nbytes
            if takes:
                parts[-1] += f"  takes {takes / 1e9:.2f} GB"
        else:
            parts.append("cpu/mem: pip install psutil")
        la = self.through
        if la is not None and la.running:
            st = la.stats()
            db = lambda p: f"{20 * math.log10(max(p, 1e-5)):4.0f} dB"
            src = f"in {db(st['in_peak'])}" if la.in_name else "no input"
            mode = st.get("mode") or "audio"
            parts.append(f"audio {mode}  {src}  out {db(st['out_peak'])}")
            parts.append(f"      dsp {100 * st['load']:4.1f} %  peak {100 * st['peak']:4.0f} %  "
                         f"buf {st['buffer_ms']:.1f} ms  "
                         f"sync {st['delay_ms']:.0f} ms  late {st['late']}  "
                         f"gap {st['gap_ms']:.1f} ms  xruns {st['dropped']}/{st['underruns']}")
            song = la.song
            if song.loaded and song.state in ("playing", "paused"):
                stems = "+".join(n[0] for n in song.stems if la.on.get(n, True)) or "none"
                parts.append(f"song  {song.position_ms() / 1000:6.1f} s / {song.duration_ms() / 1000:.0f} s  "
                             f"stems {stems}  ab {la.ab:.2f}")
            elif la.player.playing:
                sp = la.player
                parts.append(f"section {sp.bpm:.0f} bpm  {sp.pos_s():5.1f} s / {sp.n / la.sr:.0f} s  ab {la.ab:.2f}")
        elif la is not None and la.error:
            parts.append(f"audio not open: {la.error[:50]}")
        # the clock and what the next N would render, then the song-only
        # structure settings (a live section is always 8 bars; the intro
        # and outro styles shape the live kinds of those names too)
        bpm = self.tempo.bpm
        bar_s = 4 * 60.0 / bpm
        parts.append(f"clock {bpm:6.1f} bpm  bar {bar_s:.2f} s  beat {bar_s / 4:.3f} s")
        style = self.drum_style.currentText()
        sec = self._perf_playing
        if sec is not None and getattr(sec, "drum_style", None) and style == "random":
            style = f"random ({sec.drum_style} now)"
        parts.append(f"next N  {self.perf_kind.currentText()} 8 bars {8 * bar_s:.1f} s  "
                     f"drums {style}")
        intro = self.intro_style.currentText()
        if self.intro_cap.isChecked():
            intro += f" <={self.intro_bars.value()} bars"
        tail = "auto" if self.outro_tail_auto.isChecked() else f"{self.outro_tail.value():.1f} s"
        parts.append(f"song  intro {intro}  outro {self.outro_style.currentText()} "
                     f"{self.outro_bars.value():g} bars tail {tail}  "
                     f"lengths {self.section_bars.text().replace(' ', '')} bars")
        self.hud.setText("\n".join(parts))
        self.hud.adjustSize()
        self.hud.move(self.backdrop.width() - self.hud.width() - 12, 12)

    def _free_memory_gb(self):
        """The machine's free memory. Not psutil's virtual_memory() on
        Windows: that call takes 5 ms (measured) twice a second, a stutter
        in the interface for a number; GlobalMemoryStatusEx through ctypes
        takes 20 us. Elsewhere psutil, every few seconds."""
        now = time.monotonic()
        if self._mem_free is not None and now - self._mem_free[0] < 2.0:
            return self._mem_free[1]
        gb = 0.0
        try:
            if sys.platform == "win32":
                import ctypes

                class MemoryStatusEx(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                                ("ullTotalPhys", ctypes.c_ulonglong),
                                ("ullAvailPhys", ctypes.c_ulonglong),
                                ("ullTotalPageFile", ctypes.c_ulonglong),
                                ("ullAvailPageFile", ctypes.c_ulonglong),
                                ("ullTotalVirtual", ctypes.c_ulonglong),
                                ("ullAvailVirtual", ctypes.c_ulonglong),
                                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
                m = MemoryStatusEx()
                m.dwLength = ctypes.sizeof(m)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                    gb = m.ullAvailPhys / 1e9
            elif psutil is not None:
                gb = psutil.virtual_memory().available / 1e9
        except Exception:
            pass
        self._mem_free = (now, gb)
        return gb

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
        # built in this order: the Perform tab makes the voice switches the
        # Audio FX tab's column heads link to, and the Video tab the widgets
        # the Perform tab reads
        self._mode_forms = []                   # the forms with song-only rows to mark
        general = self._tab_general()
        sections = self._tab_sections_layers()   # the drums form inside
        video = self._tab_video()
        perform = self._tab_perform()
        self._mark_modes(self._mode_forms)
        tabs.addTab(perform, "Perform")
        tabs.addTab(sections, "Sections")
        tabs.addTab(video, "Video")
        audio_fx = self._tab_audio_fx()  # built here, shown after Video FX
        cycles = self._tab_cycles()  # built here, shown after Macros
        tabs.addTab(self._tab_video_fx(), "Video FX")
        tabs.addTab(audio_fx, "Audio FX")
        tabs.addTab(self._tab_lfo(), "LFO")
        scenes = self._tab_scenes()  # built here, shown after Macros
        tabs.addTab(self._tab_macros(), "Macros")
        tabs.addTab(scenes, "Scenes")
        tabs.addTab(cycles, "M:C")
        tabs.addTab(self._tab_erae(), "Erae")
        tabs.addTab(self._tab_controls(), "Controls")
        tabs.addTab(general, "Settings")
        tabs.addTab(self._tab_log(), "Log")
        HoverTabs(tabs)                  # hover a tab to open it, no click
        self.params_tabs = tabs
        return tabs

    @staticmethod
    def _frac_setter(widget):
        """x in 0..1 -> widget.setValue over its min..max (spinbox or slider)."""
        def set_frac(x):
            lo, hi = widget.minimum(), widget.maximum()
            v = lo + max(0.0, min(1.0, x)) * (hi - lo)
            # an integer widget rounds (setValue would truncate 9.99 to 9)
            widget.setValue(v if isinstance(widget, QDoubleSpinBox) else int(round(v)))
        return set_frac

    @staticmethod
    def _combo_next(combo):
        if combo.count():
            combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())

    @staticmethod
    def _combo_frac(combo, x):
        """A knob over a combo's items: the first at the bottom, the last
        at the top."""
        n = combo.count()
        if n:
            combo.setCurrentIndex(min(n - 1, int(max(0.0, min(1.0, x)) * n)))

    def _pick_shader_frac(self, x):
        """A knob over the shaders: the bottom of its travel is off, the rest
        is split evenly between the shader files in list order."""
        files = self._shader_pool()
        n = len(files) + 1
        idx = min(n - 1, int(max(0.0, min(1.0, x)) * n))
        if idx == 0:
            if self.shader.currentText() != "off":
                self.select_shader_key(0)
        else:
            name = files[idx - 1].stem
            if self.shader.currentText() != name:
                self.shader.setCurrentText(name)
                self._shader_user_off = False

    def _show_log(self):
        """Bring the Log tab to the front."""
        self.params_tabs.setCurrentWidget(self.log_page)

    def _tab_log(self):
        """The render log on its own page: the engine's lines, the video and
        shader messages, the live and MIDI notes; the last 2000 lines."""
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.addWidget(self.log, stretch=1)
        clear_btn = QPushButton("clear")
        clear_btn.setToolTip("Empty the log")
        clear_btn.clicked.connect(self.log.clear)
        hint = QLabel("render engine, video, shaders, live, MIDI: everything "
                      "the app has to say lands here")
        hint.setProperty("role", "sub")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(hint, stretch=1)
        row.addWidget(clear_btn)
        outer.addLayout(row)
        w = QWidget()
        w.setLayout(outer)
        self.log_page = w
        return w

    def _split_dragged(self, *_):
        a, b = self.split.sizes()
        if a + b > 0:
            self._split_ratio = a / (a + b)

    def _apply_split(self):
        w = self.split.width()
        if w > 0:
            right = self._split_right_min if self._split_ratio is None                 else int(w * (1 - self._split_ratio))
            self.split.blockSignals(True)
            self.split.setSizes([max(0, w - right), right])
            self.split.blockSignals(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "split"):
            # the splitter gets its new width after this event: apply then
            QTimer.singleShot(0, self._apply_split)

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
        outer.addWidget(QLabel("MIDI inputs"))
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.midi_ports, stretch=1)
        side = QVBoxLayout()
        side.addWidget(rescan)
        side.addStretch(1)
        top.addLayout(side)
        outer.addLayout(top)
        outer.addWidget(self.midi_last)
        outer.addWidget(self._midi_out_panel())

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
        # the panel is short of room: rows 20 % lower than the style gives
        # them, with the font down by as much (15 -> 12 px) so it still fits
        row_h = tree.sizeHintForRow(0)           # what the style would give a row
        if row_h > 0:                            # (the app's stylesheet ignores item hints)
            tree.setStyleSheet(f"QTreeWidget {{ font-size: 12px; }} "
                               f"QTreeWidget::item {{ height: {max(13, round(row_h * 0.8))}px; }}")
        for group in groups:                     # long knob lists: folded at first
            if group in ("grade", "fx", "vfx", "lfo", "scene", "macro") \
                    or group.startswith("rack_") \
                    or group.startswith("mc_") and group != "mc_focus" \
                    or group.startswith("audio_") and group != "audio_focus":
                groups[group].setExpanded(False)
        for c in range(3):
            tree.resizeColumnToContents(c)
        tree.itemClicked.connect(self._midi_tree_clicked)
        for key in (Qt.Key_Delete, Qt.Key_Backspace):
            sc = QShortcut(QKeySequence(key), tree)
            sc.setContext(Qt.WidgetShortcut)
            sc.activated.connect(self._midi_clear_selected)
        self.controls_tree = tree
        tree.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tree.setMinimumHeight(160)          # the page gives it the rest, and scrolls
        outer.addWidget(tree, stretch=1)
        clear_btn = QPushButton("clear binding")
        clear_btn.setToolTip("Forget the MIDI binding of the selected row")
        clear_btn.clicked.connect(self._midi_clear_selected)
        defaults_btn = QPushButton("defaults")
        defaults_btn.setToolTip("Back to the stock map: notes from C1 up, "
                                "one per button, CC 11 for A/B, channel 1; the "
                                "audio racks on channel 2, the drum voices on 3")
        defaults_btn.clicked.connect(self._midi_defaults)
        hint = self._info("none of these bring the hidden interface back; a button "
                      "fires on note on, program change or a CC going above 63; "
                      "a knob row takes a CC over the widget's whole range. "
                      "Learn from the interface: Ctrl-click any knob, switch, menu or "
                      "slider, then move or press the controller; this adds to what the "
                      "controller drives already, so one knob can turn several cells "
                      "(a shared row shows (+1), its tooltip names the others). "
                      "Alt-click a control to unbind its row. The MIDI cell here still "
                      "replaces: it takes the controller away from its other rows.")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(hint, stretch=1)
        row.addWidget(clear_btn)
        row.addWidget(defaults_btn)
        outer.addLayout(row)
        w = QWidget()
        w.setLayout(outer)
        # the page scrolls, vertically only, when the pane is shorter than
        # the ports, the MIDI out panel and the table together, instead of
        # pushing the bottom out of sight
        area = self._scrolling(w)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return area

    def _midi_out_panel(self):
        """The drum hits of the section and the song as notes on one output
        port, for a drum machine elsewhere (Bitwig over a loopMIDI cable) to
        play them; see the module note of gacha_audio.py."""
        # a plain widget with its title as a label, like the VST section of
        # the Audio FX tab (a QGroupBox draws its title over the first row
        # under the app's stylesheet)
        box = QVBoxLayout()
        box.setContentsMargins(0, 4, 0, 0)
        box.setSpacing(2)
        box.addWidget(QLabel("MIDI out: the drum hits as notes"))
        form = QFormLayout()
        form.setContentsMargins(8, 2, 8, 4)
        form.setVerticalSpacing(2)
        self.midi_out_on = QPushButton("off")
        self.midi_out_on.setCheckable(True)
        self.midi_out_on.setFixedWidth(52)              # "on" / "off", no more
        self.midi_out_on.setToolTip(
            "Send every drum hit of the playing section and song as a MIDI "
            "note on the port to the right, as it sounds (or ahead of it, see "
            "lead). Also a MIDI button, midi_out in the section engine group. "
            "Remembered.")
        self.midi_out_port = QComboBox()
        self.midi_out_port.setSizeAdjustPolicy(QComboBox.AdjustToContents)   # as wide as its names
        self.midi_out_port.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.midi_out_port.setToolTip(
            "The MIDI output the notes go to. Windows has no virtual MIDI port "
            "of its own: install loopMIDI (free), make a port, pick it here and "
            "as an input in Bitwig. Rescanned every 2 s with the inputs.")
        self.midi_out_status = QLabel("off")
        self.midi_out_status.setProperty("role", "sub")
        # takes what is left of the row; a long status never widens the tab
        self.midi_out_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        form.addRow("send", self._row([("", self.midi_out_on), ("port", self.midi_out_port),
                                       ("", self.midi_out_status)]))
        self.midi_out_channel = QSpinBox(minimum=1, maximum=16, value=10)
        self.midi_out_channel.setToolTip("MIDI channel of the notes; 10 is the GM drum channel")
        self.midi_out_cycles = QCheckBox("Model:Cycles")
        self.midi_out_cycles.setToolTip(
            "The receiver is an Elektron Model:Cycles: each drum goes out on its "
            "own track channel instead of the one channel, kick 1, snare 2, closed "
            "hat 3, open hat 4, ride 5, hat stutter 6 (the box's default T1-6 IN "
            "channels; RCV NOTE must be on). A note on a track channel plays that "
            "track chromatically, so the map below is the pitch of each: M:C puts "
            "the root (C4, 60) everywhere, which leaves the track's own tuning alone.")
        self.midi_out_lead = QSpinBox(minimum=-100, maximum=500, value=0, suffix=" ms")
        self.midi_out_lead.setToolTip(
            "How early each note leaves, in ms, counted from the moment our "
            "own hit is heard. The hits are known ahead, so any lead works: "
            "set it to the receiver's output latency (Bitwig shows it in its "
            "audio settings) and its drums land on ours; negative sends late.")
        self.midi_out_gate = QSpinBox(minimum=5, maximum=1000, value=50, suffix=" ms")
        self.midi_out_gate.setToolTip("Note on to note off; drum samplers usually ignore it")
        self.midi_out_only = QCheckBox("silence own drums")
        self.midi_out_only.setToolTip(
            "While sending, our six drum voices are silent so the receiver's "
            "drums replace them instead of doubling them; the sub bass stays. "
            "F1 still switches everything out.")
        self.midi_out_follow = QCheckBox("follow mutes")
        self.midi_out_follow.setToolTip(
            "Ticked, F1 and the Perform tab's voice switches and solos take the "
            "notes out with the sound. Off (the default) every hit goes out "
            "whatever is muted or soloed here: the receiver mutes its own tracks. "
            "Remembered.")
        form.addRow("notes", self._row([("channel", self.midi_out_channel),
                                        ("", self.midi_out_cycles),
                                        ("lead", self.midi_out_lead),
                                        ("gate", self.midi_out_gate),
                                        ("", self.midi_out_only),
                                        ("", self.midi_out_follow)]))
        self.midi_out_notes = {}
        pairs = []
        for role in DRUM_ROLES:
            sp = QSpinBox(minimum=-1, maximum=127, value=DRUM_NOTES[role])
            sp.setSpecialValueText("off")
            sp.setToolTip(f"The note the {DRUM_LABELS[role]} goes out as (General MIDI: "
                          f"{DRUM_NOTES[role]}); off sends none")
            self.midi_out_notes[role] = sp
            pairs.append((DRUM_LABELS[role], sp))
        gm = QPushButton("GM")
        gm.setToolTip("The General MIDI drum notes back: kick 36, snare 38, closed hat 42, "
                      "open hat 46, ride 51, hat stutter 44 (pedal hat)")
        gm.clicked.connect(self._midi_out_gm)
        pairs.append(("", gm))
        mc = QPushButton("M:C")
        mc.setToolTip(f"Every drum at the Model:Cycles root note {CYCLES_ROOT}: each track "
                      "plays at its own tuning. Move one up or down to pitch that track.")
        mc.clicked.connect(self._midi_out_cycles_root)
        pairs.append(("", mc))
        form.addRow("map", self._row(pairs))
        self.midi_out_clock = QCheckBox("MIDI clock")
        self.midi_out_clock.setToolTip(
            "Send MIDI clock (24 ticks a beat) on the port at all times: at the "
            "live tempo while nothing plays, and on the playing section's own beat "
            "grid, under the same lead as the notes, so a receiver's beat sits on "
            "our hits. The Model:Cycles needs CLK IN on and follows the tempo from "
            "the ticks alone. Remembered.")
        self.midi_out_transport = QCheckBox("start / stop")
        self.midi_out_transport.setToolTip(
            "Send Start when a section starts from silence (just before the tick "
            "on its downbeat, so the receiver's step 1 is our bar 1; entered late "
            "on a Link bar, on the next bar) and Stop when the sections stop or "
            "the port closes. Only with the clock: a receiver that gets Start "
            "without ticks stands still. Remembered.")
        self.midi_out_sync_status = QLabel("")
        self.midi_out_sync_status.setProperty("role", "sub")
        self.midi_out_sync_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        form.addRow("sync", self._row([("", self.midi_out_clock), ("", self.midi_out_transport),
                                       ("", self.midi_out_sync_status)]))
        # the metronome (gacha_metro): a click on every beat into the mix
        # (the audio output open), and a note per beat on the port
        self.midi_metro = QCheckBox("metronome")
        self.midi_metro.setToolTip(
            "A click on every beat, higher and louder on the one, into the mix (the audio "
            "output open, Settings tab): on the playing section's own beat grid, through the "
            "gears and a tape stop, or on the live clock (tap, downbeat, BPM box, Link) with "
            "nothing playing. With the port open every beat also goes out as a note on the "
            "notes' channel, under the same lead as the hits: the one as the first note "
            "next to it at velocity 127, beats 2, 3 and 4 as the second at 90. Remembered; "
            "a MIDI button, metronome in the section engine group.")
        self.midi_metro_one = QSpinBox(minimum=-1, maximum=127, value=76)
        self.midi_metro_one.setSpecialValueText("off")
        self.midi_metro_one.setToolTip("The downbeat's note on the MIDI out (General MIDI: "
                                       "76, hi wood block); off sends none for the one")
        self.midi_metro_note = QSpinBox(minimum=-1, maximum=127, value=77)
        self.midi_metro_note.setSpecialValueText("off")
        self.midi_metro_note.setToolTip("The note of beats 2, 3 and 4 on the MIDI out (General "
                                        "MIDI: 77, low wood block); off sends none for them")
        form.addRow("click", self._row([("", self.midi_metro), ("one", self.midi_metro_one),
                                        ("beats", self.midi_metro_note)]))
        box.addLayout(form)
        w = QWidget()
        w.setLayout(box)
        if not MIDI_AVAILABLE:
            w.setEnabled(False)
        return w

    # ---------- Model:Cycles ----------
    def _tab_cycles(self):
        """The Model:Cycles' sound over CC (gacha_midi.CYCLES_PARAMS): a
        knob per parameter per track, out on the MIDI out's port on the
        track's channel, the box's delay and reverb on the FX channel, and a
        retrig done from here (the box's own has no CC). Every cell is a
        knob the MIDI map, the LFOs, the macros and the scenes reach."""
        sv = self.settings.value
        self.mc = {}                        # (track 1..6 or "fx", key) -> widget
        self.mc_toggles = {}                # ident -> the switches among them
        self._mc_retrig_src = {t: set() for t in MC_TRACKS}
        self._mc_save_timer = QTimer(self)
        self._mc_save_timer.setSingleShot(True)
        self._mc_save_timer.setInterval(400)
        self._mc_save_timer.timeout.connect(self._mc_save)
        try:
            saved = json.loads(sv("mc/values", "") or "{}")
            if not isinstance(saved, dict):
                saved = {}
        except (ValueError, TypeError):
            saved = {}
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        hint = self._info(
            "The box's sound from here, over CC on the MIDI out's port (Controls tab, on "
            "and pointed at the box): a column per track on its channel, 1 to 6, the drum "
            "the Model:Cycles tick sends there under its number. Every cell is a knob for "
            "the MIDI map, the LFOs, the macros and the scenes; the box needs RCV CC on. "
            "Pitch glides on the box (it slews CC), machine and the LFO menus go out as the "
            "menu's index, to verify by ear. Retrig: hold a track's button, the track's note "
            "again at the rate below until let go. With the Model-TG firmware a track's "
            "machine can be the sampler: its four knobs are then the color to contour cells, "
            "named after the mode row (set it to what the box shows, it is not sent), its "
            "pitch cell is left alone, and in slice mode the note picks the slice "
            f"({CYCLES_SLICE_NOTE} is slice 1, the root {CYCLES_ROOT} slice "
            f"{CYCLES_ROOT - CYCLES_SLICE_NOTE + 1}).")
        outer.addWidget(hint)
        self.mc_status = QLabel("")
        self.mc_status.setProperty("role", "sub")
        self.mc_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.mc_focus = QComboBox()
        self.mc_focus.addItems([f"T{t} {VOICE_SHORT[MC_ROLE[t]]}" for t in MC_TRACKS])
        self.mc_focus.setToolTip("The track the focused knob rows of the Controls tab act "
                                 "on; the focus buttons switch it from a controller")
        self.mc_fx_channel = QSpinBox(minimum=1, maximum=16)
        self.mc_fx_channel.setValue(sv("mc/fx_channel", 9, type=int))
        self.mc_fx_channel.setToolTip("The channel the delay and reverb CCs go out on: the "
                                      "box's CONFIG > MIDI > IN CHAN > FX IN. Remembered.")
        self.mc_fx_channel.valueChanged.connect(self._mc_fx_channel_changed)
        send_btn = QPushButton("send all")
        send_btn.setToolTip("Every value of the tab to the box again: after it powers up, "
                            "a pattern change or a knob turned on the box itself")
        send_btn.clicked.connect(self._mc_send_all)
        defaults_btn = QPushButton("defaults")
        defaults_btn.setToolTip("Every cell back to a plain setting (the box's own defaults "
                                "as far as known) and sent")
        defaults_btn.clicked.connect(self._mc_defaults)
        outer.addWidget(self._row([("", self.mc_status), ("focus", self.mc_focus),
                                   ("FX channel", self.mc_fx_channel), ("", send_btn),
                                   ("", defaults_btn)]))
        grid = QGridLayout()
        grid.setContentsMargins(8, 4, 8, 4)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        for c, t in enumerate(MC_TRACKS):
            hdr = QLabel(f"T{t}  {VOICE_SHORT[MC_ROLE[t]]}")
            hdr.setAlignment(Qt.AlignCenter)
            hdr.setToolTip(f"Track {t} of the box, MIDI channel {t}: where the "
                           f"{DRUM_LABELS[MC_ROLE[t]]} goes with the Model:Cycles tick on")
            grid.addWidget(hdr, 0, c + 1)
        for r, (key, cc, label, kind, default, what) in enumerate(CYCLES_PARAMS, start=1):
            lbl = QLabel(label)
            lbl.setProperty("role", "sub")
            lbl.setToolTip(f"CC {cc}: {what}")
            grid.addWidget(lbl, r, 0)
            for c, t in enumerate(MC_TRACKS):
                w = self._mc_cell(t, key, saved.get(f"{t}_{key}"))
                w.setToolTip(f"Track {t} {label} (CC {cc} on channel {t}): {what}")
                grid.addWidget(w, r, c + 1)
        r = len(CYCLES_PARAMS) + 1
        lbl = QLabel("retrig")
        lbl.setProperty("role", "sub")
        lbl.setToolTip("Hold: the track's note (the map in the Controls tab, the root when "
                       "off) again and again at the retrig rate below, the velocity fading "
                       "as set, until let go. The box's RETRIG key done from here, since the "
                       "box takes no CC for it. A MIDI button per track, held.")
        grid.addWidget(lbl, r, 0)
        self.mc_retrig_btns = {}
        for c, t in enumerate(MC_TRACKS):
            b = QPushButton("hold")
            b.setCheckable(True)
            b.setToolTip(f"Track {t} retrig: latched for the mouse; a MIDI note or CC holds "
                         "it while down")
            b.toggled.connect(lambda on, t=t: self._mc_retrig_hold(t, "ui", on))
            self.mc_retrig_btns[t] = b
            grid.addWidget(b, r, c + 1)
        r += 1
        lbl = QLabel("mode")
        lbl.setProperty("role", "sub")
        lbl.setToolTip("The sampler track's playback mode as set on the box (Model-TG, "
                       "SETTINGS + PRESET): one shot, loop, slice, granular, stretch, pluck or "
                       "wave. Nothing goes out; it names the four knob cells above after what "
                       "they do in that mode. Only a sampler track has it.")
        grid.addWidget(lbl, r, 0)
        self.mc_mode = {}
        for c, t in enumerate(MC_TRACKS):
            m = QComboBox()
            m.addItems([name for name, *_ in CYCLES_SAMPLER_MODES])
            try:
                idx = int(saved.get(f"{t}_mode", 0))
            except (TypeError, ValueError):
                idx = 0
            m.setCurrentIndex(max(0, min(m.count() - 1, idx)))
            m.setToolTip(f"Track {t} sampler mode, as the box has it (not sent); names the "
                         "cells above")
            m.currentIndexChanged.connect(lambda *_a, t=t: self._mc_mode_changed(t))
            self.mc_mode[t] = m
            grid.addWidget(m, r, c + 1)
        for t in MC_TRACKS:
            self._mc_relabel(t)
        grid.setColumnStretch(0, 0)
        for c in range(len(MC_TRACKS)):
            grid.setColumnStretch(c + 1, 1)
        outer.addLayout(grid)
        form = QFormLayout()
        form.setContentsMargins(8, 2, 8, 4)
        form.setVerticalSpacing(2)
        self.mc_retrig_rate = QComboBox()
        self.mc_retrig_rate.addItems([name for name, _f in CYCLES_RETRIG_RATES])
        rate = str(sv("mc/retrig_rate", "1/16"))
        self.mc_retrig_rate.setCurrentText(
            rate if self.mc_retrig_rate.findText(rate) >= 0 else "1/16")
        self.mc_retrig_rate.setToolTip("A note every this much of a bar, at the live tempo: "
                                       "1/16 one a step, 1/32 two, 1/12 triplets, the box's "
                                       "own list. Remembered; a MIDI knob.")
        self.mc_retrig_len = QComboBox()
        self.mc_retrig_len.addItems([name for name, _f in CYCLES_RETRIG_LENS])
        ln = str(sv("mc/retrig_len", "1 step"))
        self.mc_retrig_len.setCurrentText(
            ln if self.mc_retrig_len.findText(ln) >= 0 else "1 step")
        self.mc_retrig_len.setToolTip("How long the velocity fade takes, in sixteenth "
                                      "steps; inf is a flat curve. Remembered; a MIDI knob.")
        self.mc_retrig_fade = QSpinBox(minimum=-128, maximum=127)
        self.mc_retrig_fade.setValue(sv("mc/retrig_fade", 0, type=int))
        self.mc_retrig_fade.setToolTip("The box's FDE: -128 fades the velocity out to "
                                       "nothing over the length, -64 to half, 0 flat, 64 "
                                       "fades in from half, 127 from nothing. Remembered; a "
                                       "MIDI knob.")
        self.mc_retrig_vel = QSpinBox(minimum=1, maximum=127)
        self.mc_retrig_vel.setValue(sv("mc/retrig_vel", 100, type=int))
        self.mc_retrig_vel.setToolTip("The velocity the curve starts from (the box's PAD "
                                      "menu says what it shapes). Remembered; a MIDI knob.")
        for w in (self.mc_retrig_rate, self.mc_retrig_len):
            w.currentIndexChanged.connect(self._mc_retrig_settings_changed)
        for w in (self.mc_retrig_fade, self.mc_retrig_vel):
            w.valueChanged.connect(self._mc_retrig_settings_changed)
        form.addRow("retrig", self._row([("rate", self.mc_retrig_rate),
                                         ("length", self.mc_retrig_len),
                                         ("fade", self.mc_retrig_fade),
                                         ("velocity", self.mc_retrig_vel)]))
        pairs = []
        for key, cc, label, kind, default, what in CYCLES_FX_PARAMS:
            w = self._mc_cell("fx", key, saved.get(f"fx_{key}"))
            w.setToolTip(f"{label} (CC {cc} on the FX channel): {what}")
            pairs.append((label, w))
        form.addRow("delay / reverb", self._row(pairs))
        outer.addLayout(form)
        w = QWidget()
        w.setLayout(outer)
        if not MIDI_AVAILABLE:
            w.setEnabled(False)
        area = self._scrolling(w)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return area

    @staticmethod
    def _mc_ident(t, key):
        return f"mc_fx_{key}" if t == "fx" else f"mc_{t}_{key}"

    def _mc_cell(self, t, key, saved):
        """One cell of the tab: a switch, a menu or a spinbox after the
        parameter's kind, at its saved value (the default without one)."""
        _cc, _label, kind, default, _what = MC_PARAM[key]
        if kind == "on":
            w = QCheckBox("")
            w.setChecked(bool(saved) if saved is not None else bool(default))
            w.toggled.connect(lambda *_a, t=t, key=key: self._mc_changed(t, key))
            self.mc_toggles[self._mc_ident(t, key)] = w
        elif kind == "list":
            w = QComboBox()
            w.addItems(CYCLES_LISTS[key])
            if w.count() > 8:
                searchable(w)
            try:
                idx = int(saved) if saved is not None else int(default)
            except (TypeError, ValueError):
                idx = int(default)
            w.setCurrentIndex(max(0, min(w.count() - 1, idx)))
            w.currentIndexChanged.connect(lambda *_a, t=t, key=key: self._mc_changed(t, key))
        else:
            lo, hi = CYCLES_RANGES[kind]
            w = QSpinBox(minimum=lo, maximum=hi)
            if kind in ("pct", "swing"):
                w.setSuffix(" %")
            elif kind == "semi":
                w.setSuffix(" st")
            w.setButtonSymbols(QAbstractSpinBox.NoButtons)
            try:
                v = int(saved) if saved is not None else int(default)
            except (TypeError, ValueError):
                v = int(default)
            w.setValue(max(lo, min(hi, v)))
            w.valueChanged.connect(lambda *_a, t=t, key=key: self._mc_changed(t, key))
        self.mc[(t, key)] = w
        return w

    def _mc_value(self, t, key):
        """A cell's value as the tab shows it (a switch as a bool, a menu
        as its index)."""
        w = self.mc[(t, key)]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentIndex()
        return w.value()

    def _mc_changed(self, t, key):
        self._mc_save_timer.start()
        self._mc_send(t, key)
        if key == "machine":
            self._mc_relabel(t)

    def _mc_mode_changed(self, t):
        self._mc_save_timer.start()
        self._mc_relabel(t)

    def _mc_is_sampler(self, t):
        """Track t's machine cell says the Model-TG sampler."""
        return t != "fx" and CYCLES_MACHINES[self._mc_value(t, "machine")] == "sampler"

    def _mc_relabel(self, t):
        """Track t's cells after its machine: on the sampler the four machine
        knobs carry the mode's names (a prefix in the cell, the job in the
        tooltip), the pitch cell is greyed and the mode menu live; on an FM
        machine the plain cells and no mode."""
        sampler = self._mc_is_sampler(t)
        mode = self.mc_mode[t]
        mode.setEnabled(sampler)
        name, shorts, longs = CYCLES_SAMPLER_MODES[mode.currentIndex()]
        for key, short, long_ in zip(CYCLES_SAMPLER_KNOBS, shorts, longs):
            cc, label, _kind, _d, what = MC_PARAM[key]
            w = self.mc[(t, key)]
            if sampler:
                w.setPrefix(short + " ")
                w.setToolTip(f"Track {t} sampler, {name} mode: {long_} (the {label} slot, "
                             f"CC {cc} on channel {t}, to verify on the box)")
            else:
                w.setPrefix("")
                w.setToolTip(f"Track {t} {label} (CC {cc} on channel {t}): {what}")
        cc, label, _kind, _d, what = MC_PARAM["pitch"]
        w = self.mc[(t, "pitch")]
        w.setEnabled(not sampler)
        w.setToolTip(f"Track {t} sampler: not sent, the pitch CC would pin the pitch and end "
                     "note tracking; tune it from the note map in the Controls tab" if sampler
                     else f"Track {t} {label} (CC {cc} on channel {t}): {what}")

    def _mc_send(self, t, key, force=False):
        """A cell to the box: its CC on the track's channel (the FX channel
        for the delay and reverb); the LFO depth's fine LSB as 0."""
        mo = getattr(self, "midi_out", None)
        if mo is None or not mo.is_open:
            return
        if key == "pitch" and self._mc_is_sampler(t):
            return                      # it would pin the sampler's pitch
        cc, _label, kind, _default, _what = MC_PARAM[key]
        ch = self.mc_fx_channel.value() if t == "fx" else t
        mo.cc(ch, cc, cycles_cc_value(kind, self._mc_value(t, key)), force)
        if key == "lfo_depth":
            mo.cc(ch, 110, 0, force)

    def _mc_send_all(self):
        if not self.midi_out.is_open:
            self.log.appendPlainText("m:c: the MIDI out is closed, nothing sent")
            return
        for t, key in self.mc:
            self._mc_send(t, key, force=True)
        self.log.appendPlainText(f"m:c: every value sent to {self.midi_out.label}")

    def _mc_defaults(self):
        for (t, key), w in self.mc.items():
            default = MC_PARAM[key][3]
            if isinstance(w, QCheckBox):
                w.setChecked(bool(default))
            elif isinstance(w, QComboBox):
                w.setCurrentIndex(int(default))
            else:
                w.setValue(int(default))

    def _mc_save(self):
        values = {("fx_" if t == "fx" else f"{t}_") + key: self._mc_value(t, key)
                  for t, key in self.mc}
        values.update({f"{t}_mode": m.currentIndex() for t, m in self.mc_mode.items()})
        self.settings.setValue("mc/values", json.dumps(values))

    def _mc_fx_channel_changed(self, ch):
        self.settings.setValue("mc/fx_channel", int(ch))
        for key, *_ in CYCLES_FX_PARAMS:
            self._mc_send("fx", key, force=True)
        self._mc_status()

    def _mc_focused(self, key):
        """An action on the focused track's cell `key`: a knob's 0..1 for
        a spinbox or a menu, a toggle for a switch."""
        kind = MC_PARAM[key][2]

        def act(x=None):
            w = self.mc[(self.mc_focus.currentIndex() + 1, key)]
            if kind == "on":
                w.toggle()
            elif kind == "list":
                self._combo_frac(w, x)
            else:
                self._frac_setter(w)(x)
        return act

    def _mc_retrig_hold(self, t, source, on):
        """Track t's retrig held (`on`) or let go by `source`: "midi" or
        "ui" (the latching button). It runs while any of them holds it; a
        new hold restarts the velocity curve."""
        srcs = self._mc_retrig_src[t]
        if on:
            srcs.add(source)
        else:
            srcs.discard(source)
        held = bool(srcs)
        b = self.mc_retrig_btns[t]
        if b.isChecked() != held:
            b.blockSignals(True)
            b.setChecked(held)
            b.blockSignals(False)
        if held:
            self._mc_retrig_start(t)
        else:
            self.midi_out.retrig_stop(t)
        self._mc_status()

    def _mc_retrig_start(self, t):
        note = self.midi_out_notes[MC_ROLE[t]].value()
        if note < 0:
            note = CYCLES_ROOT
        bar_s = 240.0 / max(1e-6, self.tempo.bpm)
        rate = dict(CYCLES_RETRIG_RATES).get(self.mc_retrig_rate.currentText(), 1 / 16)
        steps = dict(CYCLES_RETRIG_LENS).get(self.mc_retrig_len.currentText(), 1.0)
        self.midi_out.retrig_start(t, note, self.mc_retrig_vel.value(), rate * bar_s,
                                   steps * bar_s / 16.0, self.mc_retrig_fade.value())

    def _mc_retrig_release_all(self):
        """Every retrig let go (the port closing)."""
        for t in MC_TRACKS:
            if self._mc_retrig_src[t]:
                self._mc_retrig_src[t].clear()
                self._mc_retrig_hold(t, "ui", False)

    def _mc_retrig_settings_changed(self, *_a):
        self.settings.setValue("mc/retrig_rate", self.mc_retrig_rate.currentText())
        self.settings.setValue("mc/retrig_len", self.mc_retrig_len.currentText())
        self.settings.setValue("mc/retrig_fade", self.mc_retrig_fade.value())
        self.settings.setValue("mc/retrig_vel", self.mc_retrig_vel.value())
        for t in MC_TRACKS:                 # a held one takes the new settings
            if self._mc_retrig_src[t] and self.midi_out.is_open:
                self._mc_retrig_start(t)

    def _mc_status(self):
        mo = getattr(self, "midi_out", None)
        if not MIDI_AVAILABLE:
            text = "mido / python-rtmidi missing"
        elif mo is None or not mo.is_open:
            text = "MIDI out closed (Controls tab): nothing leaves"
        else:
            held = [str(t) for t in MC_TRACKS if self._mc_retrig_src[t]]
            text = (f"{mo.label}: tracks on channels 1-6, FX on {self.mc_fx_channel.value()}, "
                    f"{mo.ccs} CCs sent" + (", retrig T" + " T".join(held) if held else ""))
        self.mc_status.setText(text)

    # ---------- MIDI ----------
    def _midi_out_setup(self):
        """The output port and its settings back from the last launch; the
        widgets drive the stream's MidiOut from here on."""
        self.midi_out = MidiOut()
        self._midi_out_want = self.settings.value("midi/out_port", "") or ""
        self._midi_out_present = []
        self.midi_out_channel.setValue(self.settings.value("midi/out_channel", 10, type=int))
        self.midi_out_lead.setValue(self.settings.value("midi/out_lead", 0, type=int))
        self.midi_out_gate.setValue(self.settings.value("midi/out_gate", 50, type=int))
        self.midi_out_only.setChecked(self.settings.value("midi/out_only", False, type=bool))
        self.midi_out_follow.setChecked(self.settings.value("midi/out_follow", False, type=bool))
        self.midi_out_cycles.setChecked(self.settings.value("midi/out_cycles", False, type=bool))
        self.midi_out_channel.setEnabled(not self.midi_out_cycles.isChecked())
        self.midi_out_clock.setChecked(self.settings.value("midi/out_clock", False, type=bool))
        self.midi_out_transport.setChecked(
            self.settings.value("midi/out_transport", False, type=bool))
        self.midi_metro.setChecked(self.settings.value("midi/metro", False, type=bool))
        self.midi_metro_one.setValue(self.settings.value("midi/metro_one", 76, type=int))
        # the beats' note; before the one had its own, one note (76) did both
        beats = self.settings.value("midi/metro_note", 77, type=int)
        if beats == 76 and not self.settings.contains("midi/metro_one"):
            beats = 77
        self.midi_metro_note.setValue(beats)
        saved = self.settings.value("midi/out_notes", "")
        if saved:
            try:
                for role, note in json.loads(saved).items():
                    if role in self.midi_out_notes and -1 <= int(note) <= 127:
                        self.midi_out_notes[role].setValue(int(note))
            except (ValueError, AttributeError, TypeError):
                pass
        # a map left at the first Model:Cycles root (36, every drum) moves to the real one
        if self.midi_out_cycles.isChecked()                 and all(sp.value() == 36 for sp in self.midi_out_notes.values()):
            for sp in self.midi_out_notes.values():
                sp.setValue(CYCLES_ROOT)
        for sp in (self.midi_out_channel, self.midi_out_lead, self.midi_out_gate):
            sp.valueChanged.connect(self._midi_out_apply)
        self.midi_out_only.toggled.connect(self._midi_out_apply)
        self.midi_out_follow.toggled.connect(self._midi_out_apply)
        self.midi_out_cycles.toggled.connect(self._midi_out_cycles_toggled)
        self.midi_out_clock.toggled.connect(self._midi_out_apply)
        self.midi_metro.toggled.connect(self._midi_out_apply)
        self.midi_metro_note.valueChanged.connect(self._midi_out_apply)
        self.midi_metro_one.valueChanged.connect(self._midi_out_apply)
        self.midi_out_transport.toggled.connect(self._midi_out_apply)
        for sp in self.midi_out_notes.values():
            sp.valueChanged.connect(self._midi_out_apply)
        self.midi_out_port.currentIndexChanged.connect(self._midi_out_port_changed)
        self.midi_out_on.toggled.connect(self._midi_out_toggled)
        self._midi_out_refresh_ports()
        if MIDI_AVAILABLE and self.settings.value("midi/out_on", False, type=bool):
            self.midi_out_on.setChecked(True)

    def _midi_out_refresh_ports(self):
        """The outputs list, the remembered one kept (greyed when absent)."""
        if not MIDI_AVAILABLE:
            return
        ports = midi_outputs()
        present = [lab for lab, _ in ports]
        self._midi_out_present = ports
        labels = list(present)
        if self._midi_out_want and self._midi_out_want not in labels:
            labels.append(self._midi_out_want)
        shown = [self.midi_out_port.itemData(i) for i in range(self.midi_out_port.count())]
        if shown != labels:
            self.midi_out_port.blockSignals(True)
            self.midi_out_port.clear()
            for lab in labels:
                text = lab if lab in present else f"{lab} (not plugged in)"
                self.midi_out_port.addItem(text, lab)
            idx = labels.index(self._midi_out_want) if self._midi_out_want in labels else -1
            self.midi_out_port.setCurrentIndex(idx)
            self.midi_out_port.blockSignals(False)
        if self.midi_out_on.isChecked():
            want = self._midi_out_want
            if want and not self.midi_out.is_open and want in present:
                self._midi_out_open()               # plugged in since: opens by itself
            elif self.midi_out.is_open and want not in present:
                self.midi_out.close_port()
                self.midi_out_status.setText(f"{want} vanished; waiting for it")
                self.log.appendPlainText(f"midi out: {want} vanished")
            elif self.midi_out.is_open:
                self.midi_out_status.setText(
                    f"{want}: {self.midi_out.sent} notes sent, {self._midi_out_channels_text()}, "
                    f"late {self.midi_out.late_ms:.1f} ms")
        self._midi_out_sync_text()

    def _midi_out_port_changed(self, idx):
        lab = self.midi_out_port.itemData(idx) if idx >= 0 else ""
        self._midi_out_want = lab or ""
        self.settings.setValue("midi/out_port", self._midi_out_want)
        if self.midi_out_on.isChecked():
            self._midi_out_open()

    def _midi_out_open(self):
        want = self._midi_out_want
        if not want:
            self.midi_out.close_port()
            self.midi_out_status.setText("no port chosen")
            return
        err = self.midi_out.open(want, self._midi_out_present)
        if err:
            self.midi_out_status.setText(err)
            self.log.appendPlainText(f"midi out: cannot open {want}: {err}")
        else:
            self.midi_out_status.setText(f"{want}: open, {self._midi_out_channels_text()}")
            self.log.appendPlainText(f"midi out: {want}, {self._midi_out_channels_text()}, "
                                     f"lead {self.midi_out_lead.value()} ms")
        self._midi_out_apply()

    def _midi_out_toggled(self, on):
        self.settings.setValue("midi/out_on", on)
        self.midi_out_on.setText("on" if on else "off")
        if on:
            self._midi_out_refresh_ports()      # opens the port when it is there
            if not self.midi_out.is_open:
                self._midi_out_open()           # else say why not
        else:
            self._mc_retrig_release_all()
            self.midi_out.close_port()              # note offs for what sounds
            self.midi_out_status.setText("off")
            self.log.appendPlainText("midi out: off")
        self._midi_out_apply()

    def _midi_out_gm(self):
        for role, sp in self.midi_out_notes.items():
            sp.setValue(DRUM_NOTES[role])

    def _midi_out_cycles_root(self):
        for sp in self.midi_out_notes.values():
            sp.setValue(CYCLES_ROOT)

    def _midi_out_sync_text(self):
        """What the clock and the transport are doing, on the sync row."""
        mo = self.midi_out
        if not mo.is_open or not (self.midi_out_clock.isChecked()
                                  or self.midi_out_transport.isChecked()):
            text = ""
        else:
            parts = []
            if self.midi_out_clock.isChecked():
                parts.append(f"clock {mo.bpm:.1f} bpm, {mo.ticks} ticks")
            if self.midi_out_transport.isChecked():
                parts.append("running" if mo.running else "stopped")
            text = ", ".join(parts)
        self.midi_out_sync_status.setText(text)
        self._mc_status()

    def _midi_out_channels_text(self):
        if self.midi_out_cycles.isChecked():
            return "Model:Cycles channels 1-6"
        return f"channel {self.midi_out_channel.value()}"

    def _midi_out_cycles_toggled(self, on):
        """A Model:Cycles at the other end: the channel box gives way to the
        box's track channels; the notes of a fresh tick go to the root, the
        GM ones would pitch every track up."""
        self.midi_out_channel.setEnabled(not on)
        if on and all(sp.value() == DRUM_NOTES[r] for r, sp in self.midi_out_notes.items()):
            self._midi_out_cycles_root()        # applies through the spinboxes
        self._midi_out_apply()
        if self.midi_out.is_open:
            self.midi_out_status.setText(f"{self._midi_out_want}: open, "
                                         f"{self._midi_out_channels_text()}")

    def _midi_out_apply(self, *_a):
        """The panel's values to the settings and the stream."""
        notes = {role: sp.value() for role, sp in self.midi_out_notes.items()}
        self.settings.setValue("midi/out_channel", self.midi_out_channel.value())
        self.settings.setValue("midi/out_lead", self.midi_out_lead.value())
        self.settings.setValue("midi/out_gate", self.midi_out_gate.value())
        self.settings.setValue("midi/out_only", self.midi_out_only.isChecked())
        self.settings.setValue("midi/out_cycles", self.midi_out_cycles.isChecked())
        self.settings.setValue("midi/out_follow", self.midi_out_follow.isChecked())
        self.settings.setValue("midi/out_clock", self.midi_out_clock.isChecked())
        self.settings.setValue("midi/out_transport", self.midi_out_transport.isChecked())
        self.settings.setValue("midi/out_notes", json.dumps(notes))
        self.settings.setValue("midi/metro", self.midi_metro.isChecked())
        self.settings.setValue("midi/metro_note", self.midi_metro_note.value())
        self.settings.setValue("midi/metro_one", self.midi_metro_one.value())
        if self.midi_out.clock_on != self.midi_out_clock.isChecked():
            self.midi_out.clock_on = self.midi_out_clock.isChecked()
        self.midi_out.transport_on = self.midi_out_transport.isChecked()
        if not self.midi_out.transport_on:
            self.midi_out.stop()                    # a Start out there is taken back
        self._midi_out_sync_text()
        la = self.through
        if la is None:
            return
        la.midi_out = self.midi_out
        la.midi_channel = self.midi_out_channel.value()
        la.midi_channels = dict(CYCLES_CHANNELS) if self.midi_out_cycles.isChecked() else None
        la.midi_lead_ms = float(self.midi_out_lead.value())
        la.midi_gate_ms = float(self.midi_out_gate.value())
        la.midi_notes = notes
        la.midi_drums_only = self.midi_out_only.isChecked()
        la.midi_follow_mutes = self.midi_out_follow.isChecked()
        la.metro.on = self.midi_metro.isChecked()
        la.metro.note = self.midi_metro_note.value()
        la.metro.note_one = self.midi_metro_one.value()

    def _midi_setup(self):
        """Load the map and the ticked ports, open them, start watching for
        controllers plugged in later. Called once the actions exist."""
        self.midi = MidiIn(self)
        self.midi.message.connect(self._on_midi)
        self._midi_learn = None                 # ident being learnt, or None
        self._midi_learn_add = False            # that learn leaves the control's other rows
        self._pad_learn = None                  # the pad rows still to learn in a row, or None
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
        self._midi_out_setup()                  # first: the Erae mirrors its clock
        self.erae = None                        # EraeSurface while the Erae row is on
        self.erae_on.toggled.connect(self._erae_toggled)
        if ERAE_AVAILABLE and self.settings.value("erae/on", False, type=bool):
            self.erae_on.setChecked(True)       # before the MIDI ports open: the port is the surface's
        self._midi_refresh_ports()
        self._midi_timer = QTimer(self)
        self._midi_timer.timeout.connect(self._midi_refresh_ports)
        self._midi_timer.timeout.connect(self._midi_out_refresh_ports)
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
        want = set(self._midi_ticked or ())
        if self.erae is not None:               # the surface has the Erae's port
            want = {lab for lab in want if not lab.startswith(ERAE_PORT)}
        errors = self.midi.open(want, ports)
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
                rev.setdefault(b, []).append(ident)
        for b, idents in rev.items():
            if len(idents) > 1:                 # one control, several rows: each says so
                for ident in idents:
                    item = self.midi_items[ident]
                    item.setText(3, f"{self.midi_map[ident]}  (+{len(idents) - 1})")
                    item.setToolTip(3, "also drives: " + ", ".join(i for i in idents if i != ident))
            else:
                self.midi_items[idents[0]].setToolTip(3, "")
        self._midi_rev = rev
        self.controls_tree.resizeColumnToContents(3)

    def _midi_save(self):
        self.settings.setValue("midi/map", json.dumps(self.midi_map))

    def _learn_widgets(self):
        """Widget -> action id for every control with a row in the Controls
        tab, read off the actions themselves (the widget a setter closes
        over, a toggle's own widget, a combo passed as a default)."""
        m = getattr(self, "_learn_map", None)
        if m is not None:
            return m
        m = {}
        for ident, fn in self.actions.items():
            if ident not in self.midi_items:
                continue
            found = [getattr(fn, "__self__", None)]
            for cell in getattr(fn, "__closure__", None) or ():
                try:
                    found.append(cell.cell_contents)
                except ValueError:
                    pass
            found.extend(getattr(fn, "__defaults__", None) or ())
            for w in found:
                if isinstance(w, QWidget) and w is not self and not w.isWindow() \
                        and w not in m:
                    m[w] = ident
                    break
        self._learn_map = m
        return m

    def _learn_click(self, obj):
        """A Ctrl-click on a control: its row in the Controls tab learns
        the next MIDI message (a second Ctrl-click, or a click in the tree,
        cancels). True when the click landed on a control."""
        m = self._learn_widgets()
        w, ident = obj, None
        for _ in range(5):
            if w is None:
                break
            ident = m.get(w)
            if ident is not None:
                break
            w = w.parentWidget()
        if ident is None:
            return False
        self._midi_learn = None if self._midi_learn == ident else ident
        self._midi_learn_add = True             # on top of what the control drives already
        if self._midi_learn:
            what = "move a knob or a pedal" if ident in CONTINUOUS \
                else "press the switch or hit the pad"
            self.midi_last.setText(f"learning {ident}: {what}; Ctrl-click again to cancel")
            self.log.appendPlainText(f"midi: learning {ident} ({what}), added to the control's rows")
            self.controls_tree.scrollToItem(self.midi_items[ident])
        else:
            self.midi_last.setText(f"learning {ident}: cancelled")
        self._midi_refresh_tree()
        return True

    def _learn_clear_click(self, obj):
        """An Alt-click on a control: its row's binding goes. True when the
        click landed on a control."""
        m = self._learn_widgets()
        w, ident = obj, None
        for _ in range(5):
            if w is None:
                break
            ident = m.get(w)
            if ident is not None:
                break
            w = w.parentWidget()
        if ident is None:
            return False
        text = self.midi_map.get(ident, "")
        if self._midi_learn == ident:
            self._midi_learn = None
        if text:
            self.midi_map[ident] = ""
            self._midi_save()
            self.midi_last.setText(f"{ident}: {text} unbound")
            self.log.appendPlainText(f"midi: {ident}: {text} unbound")
        else:
            self.midi_last.setText(f"{ident}: nothing bound")
        self._midi_refresh_tree()
        return True

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
        self._midi_learn_add = False
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

    # ------------------------------------------------------------ Erae 2 surface
    def _tab_erae(self):
        """The Erae 2: the surface's switch and status, the clock and the
        transport to the device, which page sits on each N button, a
        mirror of the page on screen naming the control under the mouse,
        and the documentation of every page."""
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        self.erae_on = QCheckBox("Erae 2 surface")
        self.erae_on.setToolTip(
            "Needs an Erae 2 whose eight layouts each hold one API zone over the "
            "whole surface (Erae Lab: zone index 0 on N1 up to 7 on N8), and Erae "
            "Lab closed. Each layout becomes the page chosen below, drawn and read "
            "from here; the layout's own pads and faders still reach the Controls "
            "tab's map as plain MIDI. Remembered.")
        self.erae_status = QLabel("off")
        self.erae_status.setProperty("role", "sub")
        self.erae_status.setWordWrap(True)
        if not ERAE_AVAILABLE:
            self.erae_on.setEnabled(False)
            self.erae_status.setText("python-rtmidi missing")
        sv = self.settings.value
        self.erae_clock = QCheckBox("MIDI clock to the Erae")
        self.erae_clock.setToolTip(
            "The app's MIDI clock (24 ticks a beat: the section's grid while one "
            "plays, the live clock otherwise) on the Erae's port, so its looper "
            "locks to the show. On the device: Settings > MIDI clock > clock source "
            "= USB device. Runs whether or not the MIDI out panel's own clock is on.")
        self.erae_clock.setChecked(sv("erae/clock", True, type=bool))
        self.erae_transport = QCheckBox("transport")
        self.erae_transport.setToolTip(
            "Start to the Erae on the live clock's next downbeat when this goes on, "
            "again at D (downbeat) and with every section's first bar, so the "
            "device's loop starts on our one. Nothing stops it when a section ends "
            "unless the next tick says so.")
        self.erae_transport.setChecked(sv("erae/transport", True, type=bool))
        self.erae_stop = QCheckBox("stop with the section")
        self.erae_stop.setToolTip("Stop to the Erae when a section clears: its looper "
                                  "stops and starts from the top at the next Start")
        self.erae_stop.setChecked(sv("erae/stop", False, type=bool))
        for cb in (self.erae_clock, self.erae_transport, self.erae_stop):
            cb.toggled.connect(self._erae_sync_apply)
        self.erae_learn = QPushButton("learn the pads")
        self.erae_learn.setCheckable(True)
        self.erae_learn.setToolTip(
            "The device's own pads (a layout built in Erae Lab) to the drum rows in one go: "
            "click, then tap the pads in this order: kick, snare, closed hat, open hat, ride, "
            "hat stutter, burst, event. Click again to stop early and keep what was learned.")
        self.erae_learn.toggled.connect(self._pad_learn_toggled)
        outer.addWidget(self._row([("", self.erae_on), ("", self.erae_clock),
                                   ("", self.erae_transport), ("", self.erae_stop),
                                   ("", self.erae_learn)]))
        outer.addWidget(self.erae_status)
        # the pages on the N buttons on the left, the mirror on the right
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        self.erae_page_combos = []
        names = self._erae_pages()
        for i in range(8):
            lbl = QLabel(f"N{i + 1}")
            lbl.setProperty("role", "sub")
            combo = QComboBox()
            for key in ERAE_PAGES:
                combo.addItem(erae_make_page(key).title, key)
            combo.setCurrentIndex(max(0, combo.findData(names[i])))
            combo.currentIndexChanged.connect(self._erae_pages_changed)
            combo.setToolTip("The page this layout button shows; the documentation "
                             "below says what is on it")
            grid.addWidget(lbl, i, 0)
            grid.addWidget(combo, i, 1)
            self.erae_page_combos.append(combo)
        grid.setColumnStretch(1, 1)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(QLabel("pages"))
        left.addLayout(grid)
        left.addStretch(1)
        self.erae_mirror = EraeMirror()
        self.erae_hover = QLabel("hover the page: the control under the mouse")
        self.erae_hover.setWordWrap(True)
        self.erae_hover.setMinimumHeight(44)
        self.erae_hover.setStyleSheet(
            "QLabel { background: rgb(28, 22, 20); color: rgb(242, 233, 229); "
            "font-size: 17px; padding: 6px 10px; border-radius: 4px; }")
        self.erae_mirror.hovered.connect(
            lambda t: self.erae_hover.setText(t or "hover the page: the control under the mouse"))
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(QLabel("on screen"))
        right.addWidget(self.erae_mirror, stretch=1)
        right.addWidget(self.erae_hover)
        cols = QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(12)
        cols.addLayout(left, stretch=2)
        cols.addLayout(right, stretch=3)
        outer.addLayout(cols, stretch=2)
        self.erae_doc = QPlainTextEdit()
        self.erae_doc.setReadOnly(True)
        self.erae_doc.setPlainText(self._erae_doc_text())
        outer.addWidget(QLabel("documentation"))
        outer.addWidget(self.erae_doc, stretch=1)
        w = QWidget()
        w.setLayout(outer)
        return w

    def _erae_pages(self):
        """The page names on N1..N8, from the settings (the catalogue's
        defaults where nothing is saved)."""
        try:
            names = json.loads(self.settings.value("erae/pages", "") or "[]")
        except (TypeError, ValueError):
            names = []
        names = [n if n in ERAE_PAGES else "empty" for n in names][:8]
        return names + ERAE_DEFAULT_PAGES[len(names):]

    def _erae_pages_changed(self, *_a):
        names = [c.currentData() for c in self.erae_page_combos]
        self.settings.setValue("erae/pages", json.dumps(names))
        if self.erae is not None:
            self.erae.set_pages(names)
        self.erae_doc.setPlainText(self._erae_doc_text())

    def _erae_doc_text(self):
        names = [c.currentData() for c in self.erae_page_combos]
        parts = [ERAE_SETUP_DOC, ""]
        for i, name in enumerate(names):
            parts.append(f"N{i + 1}  " + erae_make_page(name).describe())
            parts.append("")
        return "\n".join(parts)

    def _erae_page_changed(self, k):
        pg = self.erae.active_page() if self.erae is not None else None
        if pg is not None:
            self.midi_last.setText(f"erae: N{k + 1} {pg.title}")

    def _erae_toggled(self, on):
        """The Erae tab's switch: the surface takes the Erae's port (the
        MIDI list lets go of it first), draws the pages and gets the clock;
        off, it blanks them, leaves API mode, and the port is a plain input
        again."""
        self.settings.setValue("erae/on", on)
        if on and self.erae is None:
            self.erae = EraeSurface(self._erae_read, pages=self._erae_pages(), parent=self)
            self.erae.action.connect(self._erae_action)
            self.erae.hold.connect(self._erae_hold)
            self.erae.midi.connect(self._on_midi)
            self.erae.page.connect(self._erae_page_changed)
            self.erae.log.connect(self.log.appendPlainText)
            self.erae.log.connect(self.erae_status.setText)
            if self._midi_ticked is not None:
                self._midi_open()               # drop the Erae from the plain inputs
            err = self.erae.open()
            if err:
                self.erae_status.setText(err)
                self.log.appendPlainText(f"erae: {err}")
                self.erae.close()
                self.erae = None
                if self._midi_ticked is not None:
                    self._midi_open()
            else:
                self.erae_status.setText("joining: asking the zones their size...")
                self.log.appendPlainText(f"erae: on, {self.erae.label}")
                self.erae_mirror.surface = self.erae
                self._erae_sync_apply()
                self._erae_start()
        elif not on and self.erae is not None:
            self.midi_out.mirror(self.erae, False, False)
            self.erae_mirror.surface = None
            self.erae.close()
            self.erae = None
            self.erae_status.setText("off")
            self.log.appendPlainText("erae: off")
            if self._midi_ticked is not None:
                self._midi_open()               # the port is a plain input again, if ticked

    def _erae_sync_apply(self, *_a):
        """The clock and transport ticks to the settings and the mirror."""
        clock = self.erae_clock.isChecked()
        transport = self.erae_transport.isChecked()
        stop = self.erae_stop.isChecked()
        self.settings.setValue("erae/clock", clock)
        self.settings.setValue("erae/transport", transport)
        self.settings.setValue("erae/stop", stop)
        if self.erae is None:
            return
        was = self.midi_out.mirrored
        self.midi_out.mirror(self.erae, clock=clock, transport=transport, stop=stop)
        if transport and not was:
            self._erae_start()

    def _erae_start(self, at_next_bar=True):
        """Start to the Erae on the live clock's next downbeat (or now),
        so the device's loop begins on our one."""
        if self.erae is None or not self.erae_transport.isChecked():
            return
        bar_s = 4 * 60.0 / max(1e-6, self.tempo.bpm)
        now_m = time.monotonic()
        due_m = now_m
        if at_next_bar:
            n = math.ceil((now_m - self.tempo.anchor) / bar_s - 1e-6)
            due_m = self.tempo.anchor + n * bar_s
        self.midi_out.start(time.perf_counter() + max(0.0, due_m - now_m))

    PAD_LEARN = ["hit_kick", "hit_snare", "hit_hihat", "hit_ohat", "hit_ride", "hit_glitch",
                 "burst", "event"]

    def _pad_learn_toggled(self, on):
        """The pads in a row: each note that comes in maps the next row of
        PAD_LEARN; off (or the end) keeps what was learned so far."""
        if on:
            self._midi_learn = None
            self._midi_refresh_tree()
            self._pad_learn = list(self.PAD_LEARN)
            self._pad_learn_prompt()
        else:
            if self._pad_learn is not None:
                done = len(self.PAD_LEARN) - len(self._pad_learn)
                self._pad_learn = None
                self.erae_status.setText(f"pads: {done} learned")
                self.log.appendPlainText(f"pads: {done} of {len(self.PAD_LEARN)} learned")

    def _pad_learn_prompt(self):
        ident = self._pad_learn[0]
        n = len(self.PAD_LEARN) - len(self._pad_learn) + 1
        name = VOICE_LABELS.get(ident[4:], ident) if ident.startswith("hit_") else ident
        self.erae_status.setText(f"pads: tap the pad for {name} ({n} of {len(self.PAD_LEARN)}), "
                                 "click the button again to stop here")

    def _pad_learn_note(self, port, kind, ch, num, val):
        """A message while the pads are being learned: True when it was taken."""
        if kind != "note" or val == 0:
            return kind == "note"               # releases: eaten, the rest passes
        text = format_binding(kind, ch, num)
        ident = self._pad_learn[0]
        done = self.PAD_LEARN[:len(self.PAD_LEARN) - len(self._pad_learn)]
        if any(i in done for i in self._midi_rev.get((kind, ch, num), ())):
            return True                         # the same pad again: a double hit
        for other in self._midi_rev.get((kind, ch, num), ()):
            if other != ident:
                self.midi_map[other] = ""
                self.log.appendPlainText(f"midi: {text} taken from {other}")
        self.midi_map[ident] = text
        self._midi_save()
        self._midi_refresh_tree()
        self.log.appendPlainText(f"midi: {ident} <- {text} ({port})")
        self._pad_learn.pop(0)
        if self._pad_learn:
            self._pad_learn_prompt()
        else:
            self._pad_learn = None
            self.erae_learn.setChecked(False)
            self.erae_status.setText(f"pads: all {len(self.PAD_LEARN)} learned")
            self.log.appendPlainText("pads: all learned")
        return True

    def _erae_action(self, ident, value):
        """A page asked for an action: a knob gets 0..1, a button a call."""
        fn = self.actions.get(ident)
        if fn is None:
            return
        if value is not None:
            fn(float(value))
            self.midi_last.setText(f"last: erae {ident} = {float(value):.2f}")
        else:
            fn()
            self.midi_last.setText(f"last: erae {ident}")

    def _erae_hold(self, ident, down):
        """A page's held action going down or up; an action without a
        hold handler is pressed on the down."""
        h = self.holds.get(ident)
        if h is not None:
            h("erae", down)
        elif down:
            fn = self.actions.get(ident)
            if fn is not None:
                fn()
        self.midi_last.setText(f"last: erae {ident} {'down' if down else 'up'}")

    @staticmethod
    def _frac_of(widget):
        """A widget's value as 0..1 of its range (spinbox or slider)."""
        lo, hi = widget.minimum(), widget.maximum()
        return 0.0 if hi <= lo else (widget.value() - lo) / (hi - lo)

    @staticmethod
    def _combo_at(combo):
        return (combo.currentIndex() + 0.5) / max(1, combo.count())

    def _erae_read(self, ident):
        """The app's state for the LEDs: 0..1 for a knob, True/False for a
        switch, the clock's phases, the section, the scenes and the
        fragments for the strips, None when there is nothing to show."""
        if ident == "ab":
            return self._frac_of(self.perf_ab)
        if ident == "beat":
            la = self.through
            if la is not None and la.running and la.player.playing:
                return ((time.monotonic() - self.tempo.anchor) / (60.0 / self.tempo.bpm)) % 1.0
            return None
        if ident == "bar":
            pos = self.perf_meter.pos
            if pos is not None:
                return pos % 1.0
            return ((time.monotonic() - self.tempo.anchor) / (240.0 / self.tempo.bpm)) % 1.0
        if ident == "pos":
            return self.perf_meter.pos
        if ident == "bars":
            return self.perf_meter.bars
        if ident == "kind":
            sec = self._perf_playing
            return sec.kind if sec is not None and self.perf_meter.pos is not None else None
        if ident == "level":
            la = self.through
            return min(1.0, float(getattr(la, "out_peak", 0.0))) if la is not None else None
        if ident == "scene_cur":
            return self.scenes.current
        if ident.startswith("scene_empty_"):
            return self.scenes.slots[int(ident[12:]) - 1].empty
        if ident == "morph":
            return None if self.scene_morph is None else self.scene_morph.frac
        if ident == "frag_divs":
            return int(self.frag_div.currentText() or 16)
        if ident == "frag_held":
            return {k for k, srcs in self._frag_sources.items() if srcs}
        if ident == "frag_slice":
            pl = self._player()
            st = pl.frag_state() if pl is not None else None
            return st["slice"] if st else None
        if ident == "repeat":
            return bool(self._rep_sources)
        if ident == "brake":
            return bool(self._brake_sources)
        if ident == "freeze":
            return bool(self._freeze_sources)
        if ident == "nudge_down":
            return bool(self._nudge_sources[-1])
        if ident == "nudge_up":
            return bool(self._nudge_sources[1])
        if ident.startswith("frag_") and ident[5:].isdigit():
            return bool(self._frag_sources.get(int(ident[5:])))
        if ident.startswith("mc_retrig_") and ident[10:].isdigit():
            return bool(self._mc_retrig_src.get(int(ident[10:])))
        switches = {"through": self.through_on, "video_only": self.video_plain,
                    "video_off": self.video_off, "video_second": self.video_second,
                    "link": self.link_on, "midi_out": self.midi_out_on,
                    "metronome": self.midi_metro, "scene_morph": self.scene_morph_on,
                    "rep_rev": self.rep_rev, "half": self.half_btn, "double": self.double_btn,
                    "live": self.live_on, "inserts": self.vst_bypass}
        w = switches.get(ident)
        if w is not None:
            return w.isChecked()
        if ident == "record":
            return self.takes.recording
        if ident == "shader":
            return self.shader.currentText() != "off"
        if ident == "pitch_lock":
            return self.pitch_mode.currentIndex() == 1
        if ident.startswith("stem_"):
            cb = self.perf_stems.get(ident[5:])       # the M button: lit while out
            return None if cb is None else not cb.isChecked()
        if ident.startswith("voice_"):
            cb = self.perf_voices.get(ident[6:])      # the M button: lit while out
            return None if cb is None else not cb.isChecked()
        if ident.startswith("solo_"):
            b = self.perf_solos.get(ident[5:])
            return None if b is None else b.isChecked()
        if ident.startswith("focus_"):
            return self.afx_focus.currentText() == ident[6:]
        if ident.startswith("mc_focus_") and ident[9:].isdigit():
            return self.mc_focus.currentIndex() == int(ident[9:]) - 1
        if ident.startswith("lfo_") and ident[4:].isdigit():
            return self.lfo_rows[int(ident[4:]) - 1]["on"].isChecked()
        if ident.startswith("afx_") and ident.count("_") == 1:      # the focused rack
            sp = self.afx.get((self.afx_focus.currentText(), ident[4:]))
            return None if sp is None else self._frac_of(sp)
        if ident == "avol":
            sp = self.afx.get((self.afx_focus.currentText(), "vol"))
            return None if sp is None else self._frac_of(sp)
        if ident == "brake_len":
            return self._combo_at(self.brake_len)
        if ident == "speed_sync":
            return self._combo_at(self.speed_sync)
        return self._lfo_get(ident)

    def _on_midi(self, port, kind, ch, num, val):
        """Every message from every ticked controller, on the GUI thread."""
        text = format_binding(kind, ch, num)
        self.midi_last.setText(f"last: {text} = {val} from {port}")
        if self._pad_learn is not None and self._pad_learn_note(port, kind, ch, num, val):
            return
        if self._midi_learn is not None:
            ident = self._midi_learn
            if ident in CONTINUOUS and kind != "cc":
                return                          # a pedal, not a switch
            if kind == "note" and val == 0:
                return                          # a release, not a press
            others = [i for i in self._midi_rev.get((kind, ch, num), ()) if i != ident]
            if others and not self._midi_learn_add:
                for other in others:
                    self.midi_map[other] = ""   # one control, one action
                self.log.appendPlainText(f"midi: {text} taken from {', '.join(others)}")
            elif others:
                self.log.appendPlainText(f"midi: {text} also drives {', '.join(others)}")
            self.midi_map[ident] = text
            self._midi_learn = None
            self._midi_save()
            self._midi_refresh_tree()
            self.midi_last.setText(f"{ident} <- {text} from {port}")
            self.log.appendPlainText(f"midi: {ident} <- {text} ({port})")
            if kind == "cc":
                self._midi_cc_last[(ch, num)] = val   # no trigger from the learn press
            return
        idents = self._midi_rev.get((kind, ch, num))
        if not idents:
            return
        prev = None
        if kind == "cc":                        # the edge is read once for every row
            prev = self._midi_cc_last.get((ch, num), 0)
            self._midi_cc_last[(ch, num)] = val
        for ident in list(idents):
            self._midi_fire(ident, kind, val, prev)

    def _midi_fire(self, ident, kind, val, prev):
        """One row's action from a message bound to it (`prev`: the CC's
        value before this one)."""
        if ident in CONTINUOUS:
            self.actions[ident](val / 127.0)
            return
        if ident in VELOCITY and kind == "note":   # a pad: its velocity, no release
            if val > 0:
                self.actions[ident](val / 127.0)
            return
        if kind == "cc":                        # a switch on a CC: rising edge only
            if ident in HOLD:                   # and the falling edge lets go
                if val >= 64 and prev < 64:
                    self.actions[ident]()
                elif val < 64 and prev >= 64:
                    self.releases[ident]()
                return
            if not (val >= 64 and prev < 64):
                return
        elif kind == "note":
            if ident in HOLD:                   # the note off is the release
                (self.actions if val else self.releases)[ident]()
                return
            if val == 0:
                return                          # a note off: nothing for a press
        self.actions[ident]()

    def _afx_grid(self, channels, header):
        """A grid of knobs: a column per channel of `channels` headed by
        `header(ch)`, a row per effect plus the volume. Every cell is a knob
        the MIDI map reaches, kept in self.afx; all effects start off."""
        grid = QGridLayout()
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        for c, ch in enumerate(channels):
            grid.addWidget(header(ch), 0, c + 1)
        rows = list(AUDIO_FX) + [("vol", "volume", 1.0, "Level of this channel, unity at 1; "
                                                        "a gentle taper below")]
        for r, (key, label, default, tip) in enumerate(rows, start=1):
            lbl = QLabel(label)
            lbl.setProperty("role", "sub")
            lbl.setToolTip(tip)
            grid.addWidget(lbl, r, 0)
            for c, ch in enumerate(channels):
                lo = -1.0 if key in BIPOLAR else 0.0
                sp = QDoubleSpinBox(minimum=lo, maximum=1.0, singleStep=0.05, decimals=2)
                sp.setValue(default)
                sp.setToolTip(f"{ch}: {tip}")
                sp.setButtonSymbols(QAbstractSpinBox.NoButtons)
                sp.valueChanged.connect(lambda v, ch=ch, key=key: self._apply_afx(ch, key, v))
                self.afx[(ch, key)] = sp
                grid.addWidget(sp, r, c + 1)
        grid.setColumnStretch(0, 0)
        for c in range(len(channels)):
            grid.setColumnStretch(c + 1, 1)
        return grid

    def _voice_header(self, v):
        """A voice's column head in the Audio FX grid: the same switch as
        the Perform tab's, lit while the voice is in."""
        b = QPushButton(VOICE_SHORT[v])
        b.setCheckable(True)
        b.setChecked(True)
        m = self.perf_voices[v]                       # the Perform tab's M: lit while out
        b.toggled.connect(lambda on, m=m: m.setChecked(not on))
        m.toggled.connect(lambda muted, b=b: b.setChecked(not muted))
        b.setToolTip(f"{VOICE_LABELS[v]}: in or out, the Perform tab's switch; the knobs "
                     "below are this voice's own rack, before the drums rack (the bus)")
        return b

    def _tab_audio_fx(self):
        """The racks as grids: a column per channel (the live input and the
        four stems), then one per drum voice, a row per effect plus the
        volume."""
        self.afx = {}

        def plain(ch):
            hdr = QLabel(ch)
            hdr.setAlignment(Qt.AlignCenter)
            return hdr
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._afx_grid(CHANNELS, plain))
        voices_lbl = QLabel("drum voices: each through its own rack, gated by its switch "
                            "before it (a delay rings out after the mute), then summed into "
                            "the drums channel above, its rack and F1")
        voices_lbl.setProperty("role", "sub")
        voices_lbl.setWordWrap(True)
        voices_lbl.setContentsMargins(8, 4, 8, 0)
        outer.addWidget(voices_lbl)
        outer.addLayout(self._afx_grid(VOICES, self._voice_header))
        self.afx_focus = QComboBox()
        self.afx_focus.addItems(list(ALL_CHANNELS))
        self.afx_focus.setToolTip("The channel the focused knob rows of the Controls tab "
                                  "act on; the focus buttons switch it from a controller")
        off_btn = QPushButton("all effects off")
        off_btn.setToolTip("Every effect knob back to 0; volumes stay")
        off_btn.clicked.connect(self._afx_all_off)
        hint = self._info("0 = bypassed, small = a little on top of the dry sound, 1 = fully "
                      "wet; pitch is off in the middle. The live rack takes the live "
                      "input, the four stem racks take the section engine and any song "
                      "played (the stream opens at launch on the chosen output); the "
                      "voice racks take the section's drums, one voice each (a song's "
                      "drums go straight to the drums rack).")
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 8, 8)
        row.addWidget(self._row([("focused channel", self.afx_focus)]))
        row.addStretch(1)
        row.addWidget(off_btn)
        outer.addLayout(row)
        self.afx_status = QLabel("")
        self.afx_status.setStyleSheet("color: rgb(255, 150, 120); padding: 0 8px;")
        self.afx_status.setWordWrap(True)
        outer.addWidget(self.afx_status)
        hint.setContentsMargins(8, 0, 8, 8)
        outer.addWidget(hint)
        outer.addWidget(self._afx_settings())
        outer.addWidget(self._vst_section())
        outer.addStretch(1)
        w = QWidget()
        w.setLayout(outer)
        return self._scrolling(w) if hasattr(self, "_scrolling") else w

    # ---------- effect settings ----------
    def _afx_settings(self):
        """The effects' settings (gacha_fx.FX_SETTINGS), one for every
        channel's rack, between two rules under the racks: what the knob in
        the grid does not say (the delay's time and feedback, the reverb's
        room, the chorus rate...). A row per effect. Every one a MIDI knob,
        an LFO, macro and scene destination; remembered."""
        sv = self.settings.value
        self.fxset = {}
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        def rule():
            hr = QFrame()
            hr.setFrameShape(QFrame.HLine)
            hr.setStyleSheet("color: rgba(255, 255, 255, 50);")
            return hr
        box.addWidget(rule())
        title = QLabel("effect settings")
        title.setContentsMargins(8, 4, 8, 0)
        box.addWidget(title)
        hint = self._info("What the grid's knobs do not say: a knob is the amount, these shape "
                      "the effect under it, the same for every channel's rack. Each is a "
                      "MIDI knob (Controls tab, the effect settings group), an LFO, macro "
                      "and scene destination; remembered.")
        hint.setContentsMargins(8, 0, 8, 4)
        box.addWidget(hint)
        form = QFormLayout()
        form.setContentsMargins(8, 2, 8, 4)
        form.setVerticalSpacing(2)
        rows = {}
        for ident, fx, label, kind, lo, hi, default, suffix, what in FX_SETTINGS:
            saved = sv(f"afx/{ident}", None)
            if saved is None and ident in ("rev_time", "rev_fb"):      # the row's old home
                saved = sv(f"perform/{ident}", None)
            if kind == "list":
                w = QComboBox()
                w.addItems([name for name, _v in lo])
                name = str(saved) if saved is not None else default
                w.setCurrentText(name if w.findText(name) >= 0 else default)
                w.currentTextChanged.connect(self._fxset_changed)
            else:
                if kind == "int":
                    w = QSpinBox(minimum=int(lo), maximum=int(hi))
                else:
                    w = QDoubleSpinBox(minimum=float(lo), maximum=float(hi), decimals=1,
                                       singleStep=0.1)
                if suffix:
                    w.setSuffix(suffix)
                try:
                    v = float(saved) if saved is not None else float(default)
                except (TypeError, ValueError):
                    v = float(default)
                w.setValue(int(round(v)) if kind == "int" else v)
                w.valueChanged.connect(self._fxset_changed)
            w.setToolTip(f"{FX_LABEL[fx]} {label}: {what}. Every channel's; remembered; a "
                         "MIDI knob.")
            self.fxset[ident] = w
            rows.setdefault(fx, []).append((label, w))
        for fx, pairs in rows.items():
            form.addRow(FX_LABEL[fx], self._row(pairs))
        box.addLayout(form)
        box.addWidget(rule())
        self._fxset_changed()
        w = QWidget()
        w.setLayout(box)
        return w

    def _fxset_changed(self, *_a):
        """The settings block into every rack (gacha_fx.FX_SET, read per
        block), and remembered."""
        st = self.settings.setValue
        for ident, _fx, _label, kind, lo, *_r in FX_SETTINGS:
            w = self.fxset[ident]
            if kind == "list":
                FX_SET[ident] = dict(lo).get(w.currentText(), FX_SET[ident])
                st(f"afx/{ident}", w.currentText())
            else:
                FX_SET[ident] = float(w.value())
                st(f"afx/{ident}", w.value())

    # ---------- master inserts (VST3) ----------
    def _vst_section(self):
        """VST3 effect plugins on the master, after the racks: pick one from
        ./vst or the system folders (or browse), open its editor, order them,
        point the eight MIDI knobs at their parameters. Everything is
        remembered, plugin state included."""
        self.inserts = audio_host().inserts         # the chain, in the engine process
        box = QVBoxLayout()
        box.setContentsMargins(8, 8, 8, 8)
        box.setSpacing(4)
        title = QLabel("master inserts (VST3)")
        box.addWidget(title)
        self.vst_pick = QComboBox()
        self.vst_pick.setToolTip("Plugins found in ./vst and the system's VST3 folders; "
                                 "browse... for one elsewhere")
        add_btn = QPushButton("add")
        add_btn.setToolTip("Load the chosen plugin at the end of the chain (a second "
                           "or two for a JUCE plugin, the interface waits)")
        add_btn.clicked.connect(self._vst_add)
        rescan = QPushButton("rescan")
        rescan.setToolTip("Look again in vst/ and the system's VST3 folders, after a "
                          "plugin was installed or copied in")
        rescan.clicked.connect(self._vst_rescan)
        self.vst_bypass = QCheckBox("bypass all")
        self.vst_bypass.setToolTip("The whole chain out of the path, faded over one block")
        self.vst_bypass.toggled.connect(self._vst_bypass_changed)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.vst_pick, stretch=1)
        top.addWidget(add_btn)
        top.addWidget(rescan)
        top.addSpacing(12)
        top.addWidget(self.vst_bypass)
        box.addLayout(top)
        self.vst_rows = QVBoxLayout()
        self.vst_rows.setContentsMargins(0, 0, 0, 0)
        self.vst_rows.setSpacing(2)
        box.addLayout(self.vst_rows)
        self.vst_knob_combos = []
        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        for n in range(VST_KNOBS):
            lbl = QLabel(f"knob {n + 1}")
            lbl.setProperty("role", "sub")
            cb = QComboBox()
            cb.setToolTip(f"The plugin parameter MIDI knob {n + 1} drives (Controls tab, "
                          "master inserts)")
            cb.currentIndexChanged.connect(lambda _i, n=n: self._vst_knob_changed(n))
            self.vst_knob_combos.append(cb)
            grid.addWidget(lbl, n % 4, (n // 4) * 2)
            grid.addWidget(cb, n % 4, (n // 4) * 2 + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self._vst_choices = None                    # what the knob combos list now
        # plugins rename or add parameters at any time; a thread looks once a
        # second (7 ms for plugdata's 2595 slots, too much for the GUI thread)
        # and the combos are rebuilt here when it saw a change
        self.vst_params_changed.connect(self._vst_params_changed)
        self._vst_watch_stop = self.inserts.watch(self.vst_params_changed.emit)
        self._closing = False
        box.addLayout(grid)
        self.vst_status = QLabel("")
        self.vst_status.setProperty("role", "sub")
        self.vst_status.setWordWrap(True)
        box.addWidget(self.vst_status)
        hint = self._info("Effect plugins after the five racks, before the master gain and "
                      "the limiter, in the audio thread: no added buffer. A plugin's "
                      "editor blocks this interface while it is open (the sound keeps "
                      "running); a plugin that crashes takes the app with it. plugdata: "
                      "its [param] objects become the parameters the knobs can drive.")
        box.addWidget(hint)
        w = QWidget()
        w.setLayout(box)
        self._vst_rescan()
        return w

    def _vst_rescan(self):
        cur = self.vst_pick.currentData()
        self.vst_pick.blockSignals(True)
        self.vst_pick.clear()
        for label, path in scan_plugins():
            self.vst_pick.addItem(label, path)
        self.vst_pick.addItem("browse...", "")
        if cur:
            i = self.vst_pick.findData(cur)
            if i >= 0:
                self.vst_pick.setCurrentIndex(i)
        self.vst_pick.blockSignals(False)

    def _vst_add(self):
        path = self.vst_pick.currentData()
        if not path:
            if sys.platform == "win32":                # a .vst3 is a file on Windows...
                path, _f = QFileDialog.getOpenFileName(self, "VST3 plugin", str(Path.home()),
                                                       "VST3 (*.vst3)")
            else:                                      # ...and a folder on Linux and macOS
                path = QFileDialog.getExistingDirectory(self, "VST3 plugin bundle (.vst3 folder)",
                                                        str(Path.home()))
            if not path or not path.lower().endswith(".vst3"):
                if path:
                    self.vst_status.setText("that is not a .vst3 bundle")
                return
        self.vst_status.setText(f"loading {Path(path).stem}...")
        QApplication.processEvents()
        try:
            ins = self.inserts.add(path)
        except Exception as e:
            msg = str(e) or type(e).__name__
            self.vst_status.setText(f"could not load {Path(path).stem}: {msg[:200]}")
            self.log.appendPlainText(f"vst: {Path(path).stem}: {msg}")
            return
        self.log.appendPlainText(f"vst: {ins.name} loaded ({len(ins.params())} parameters, "
                                 f"{ins.latency_samples()} samples of latency, compensated)")
        self._vst_refresh()
        self._vst_save()

    def _vst_refresh(self):
        """Rebuild the rows and the knob combos from the chain."""
        while self.vst_rows.count():
            item = self.vst_rows.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for i, ins in enumerate(self.inserts.items):
            on = QCheckBox(f"{i + 1}. {ins.name}")
            on.setChecked(bool(ins.on))
            on.setToolTip(ins.path)
            on.toggled.connect(lambda v, ins=ins: self._vst_on(ins, v))
            edit = QPushButton("edit")
            edit.setToolTip("Open the plugin's own window; this interface waits until "
                            "it is closed, the sound keeps running")
            edit.clicked.connect(lambda _c=False, ins=ins: self._vst_edit(ins))
            up = QPushButton("up")
            up.clicked.connect(lambda _c=False, ins=ins: self._vst_move(ins, -1))
            down = QPushButton("down")
            down.clicked.connect(lambda _c=False, ins=ins: self._vst_move(ins, 1))
            rm = QPushButton("remove")
            rm.clicked.connect(lambda _c=False, ins=ins: self._vst_remove(ins))
            for b in (edit, up, down, rm):
                b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                b.setStyleSheet("padding: 2px 8px; font-size: 11px;")
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(on, stretch=1)
            if ins.error:
                err = QLabel(f"error, bypassed: {ins.error[:60]}")
                err.setStyleSheet("color: rgb(255, 150, 120);")
                row.addWidget(err)
            for b in (edit, up, down, rm):
                row.addWidget(b)
            w = QWidget()
            w.setLayout(row)
            self.vst_rows.addWidget(w)
        self._vst_fill_knobs(self._vst_choices_now())
        self.vst_bypass.blockSignals(True)
        self.vst_bypass.setChecked(self.inserts.bypass)
        self.vst_bypass.blockSignals(False)
        n = len(self.inserts.items)
        self.vst_status.setText("no insert" if not n else
                                f"{n} insert{'s' if n > 1 else ''} in the chain"
                                + (", bypassed" if self.inserts.bypass else ""))

    def _vst_choices_now(self):
        """[(label, data)] for the knob combos: off, then every parameter of
        every plugin in the chain, read from the plugins now. Names are
        live: a plugin may rename or add parameters at any time (plugdata
        when a [param] object is put in the patch, Surge XT Effects when
        its type changes), so this is asked again by _vst_poll."""
        choices = [("off", None)]
        for i, ins in enumerate(self.inserts.items):
            for name in ins.params():
                choices.append((f"{i + 1}. {ins.name}: {name}", (i, name)))
        return choices

    def _vst_fill_knobs(self, choices):
        """The knob combos from `choices`. A slot pointing at a name the
        plugin does not offer right now stays assigned (the name may come
        back, as when a plugin's type is switched and switched back) and is
        shown as such, rather than silently reading as off."""
        self._vst_choices = choices
        for n, cb in enumerate(self.vst_knob_combos):
            cb.blockSignals(True)
            cb.clear()
            for label, data in choices:
                cb.addItem(label, data)
            k = self.inserts.knobs[n]
            idx = next((j for j, (_l, d) in enumerate(choices) if d == k), None)
            if idx is None and k is not None:
                if k[0] < len(self.inserts.items):
                    cb.addItem(f"{k[0] + 1}. {self.inserts.items[k[0]].name}: {k[1]}"
                               "  (not a parameter now)", k)
                    idx = cb.count() - 1
                else:
                    idx = 0
            cb.setCurrentIndex(idx or 0)
            cb.blockSignals(False)

    def _vst_params_changed(self):
        """The watch thread saw a plugin's parameter names change: rebuild
        the knob combos if what they list differs. Not while a combo's list
        is open under the mouse (the next change, or an add/remove, catches
        up), nor in the middle of a restore."""
        if not self.inserts.items or getattr(self, "_vst_restoring", False):
            return
        if any(cb.view().isVisible() for cb in self.vst_knob_combos):
            return
        choices = self._vst_choices_now()
        if choices != self._vst_choices:
            self._vst_fill_knobs(choices)
            self.log.appendPlainText("vst: parameters changed, knob choices updated")

    def _vst_on(self, ins, on):
        ins.on = bool(on)
        if on and ins.error:
            ins.error = None                      # try again
        self._vst_save()

    def _vst_move(self, ins, delta):
        self.inserts.move(ins, delta)
        self._vst_refresh()
        self._vst_save()

    def _vst_remove(self, ins):
        self.inserts.remove(ins)
        self.log.appendPlainText(f"vst: {ins.name} removed")
        self._vst_refresh()
        self._vst_save()

    def _vst_edit(self, ins):
        """The plugin's own window, opened by the engine process (the plugin
        lives there) on its main thread; this interface stays free and the
        sound runs on. When it closes the engine says so (editor_closed)
        and _vst_after_editor takes the patch; closing the app closes it."""
        self.log.appendPlainText(f"vst: {ins.name} editor open")
        self.vst_status.setText(f"{ins.name}: editor open")
        self.inserts.edit(ins, int(self.winId()))  # Windows: it also moves the window on screen

    def _vst_after_editor(self, name):
        if self._closing:
            return
        self._vst_refresh()                       # parameters may have new names
        self._vst_save()                          # the patch lives in the state
        self.vst_status.setText(f"{name}: editor closed, the patch is kept")

    def _vst_knob_changed(self, n):
        self.inserts.knobs[n] = self.vst_knob_combos[n].currentData()
        self._vst_save()

    def _vst_bypass_changed(self, on):
        self.inserts.bypass = bool(on)
        self._vst_refresh()
        self._vst_save()

    def _vst_save(self):
        if getattr(self, "_vst_restoring", False):
            return
        self.settings.setValue("vst/inserts", self.inserts.to_json())

    def _vst_restore(self):
        """Load the remembered chain, plugin state included; slow, so it
        runs once the window is up."""
        text = self.settings.value("vst/inserts", "")
        if not text:
            return
        self._vst_restoring = True
        self.vst_status.setText("loading the remembered inserts...")
        QApplication.processEvents()
        try:
            errors = self.inserts.load_json(text)
        finally:
            self._vst_restoring = False
        for path, err in errors:
            self.log.appendPlainText(f"vst: {Path(path).stem} not loaded: {err}")
        for ins in self.inserts.items:
            self.log.appendPlainText(f"vst: {ins.name} back ({len(ins.params())} parameters)")
        self._vst_refresh()
        if errors:
            self.vst_status.setText(self.vst_status.text() + "; " +
                                    ", ".join(f"{Path(p).stem} failed" for p, _e in errors))

    def _update_afx_status(self):
        """Which path the sound is on: the racks only touch what goes through
        the app's audio stream."""
        la = getattr(self, "through", None)
        playing = self.player.playbackState() == QMediaPlayer.PlayingState
        if la is not None and la.running:
            if playing and not self.player.streaming:
                text = ("racks ON for the live input and the section engine; the song now "
                        "playing started before the stream opened and runs through Qt, "
                        "untouched: start it again to bring it through the racks")
            else:
                text = ("racks ON: the stream is open; the live input, the section engine "
                        "and songs started from now on pass through them")
            self.afx_status.setStyleSheet("color: rgba(242,233,229,160); padding: 0 8px;")
        else:
            text = ("racks OFF: the audio stream is not open (no output device, or it "
                    "could not be opened: see Audio output in the Settings tab), so songs "
                    "play through Qt and none of these knobs, volumes or F1-F4 apply.")
            self.afx_status.setStyleSheet("color: rgb(255, 150, 120); padding: 0 8px;")
        self.afx_status.setText(text)

    def _apply_afx(self, ch, key, v):
        la = getattr(self, "through", None)
        if la is None:
            return
        rack = la.racks[ch]
        if key == "vol":
            rack.vol = float(v)
        else:
            rack.amounts[key] = float(v)

    def _push_afx(self):
        """Every knob of the grid into a freshly opened stream."""
        for (ch, key), sp in self.afx.items():
            self._apply_afx(ch, key, sp.value())

    def _afx_all_off(self):
        for (ch, key), sp in self.afx.items():
            if key != "vol":
                sp.setValue(0.0)

    def _afx_focused(self, key):
        """x -> the focused channel's knob `key`."""
        def set_frac(x):
            self._frac_setter(self.afx[(self.afx_focus.currentText(), key)])(x)
        return set_frac

    def _tab_perform(self):
        """The live section engine: what the feet will drive on stage."""
        form = QFormLayout()
        # the song shapes the live set borrows, at hand: the intro and the
        # outro style and the drum style, each the Generate / Sections tab's
        # own box again, the pairs one setting
        self.perf_intro = QComboBox()
        self.perf_outro = QComboBox()
        self.perf_drum_style = QComboBox()
        for mine, theirs, tip in (
                (self.perf_intro, self.intro_style,
                 "How an intro section is shaped (next kind: intro): random picks one of "
                 "the rest; ambient (textures, "
                 "the drums now and then), sparse (one voice), build (the voices stack "
                 "up bar by bar), drums-first (no layers), filtered (a lowpass opens over "
                 "the 8 bars), reverse-swell (a reversed sample swells into the loop's "
                 "end, no drums: N on its last bar is the drop), collage (scattered "
                 "one-shots, no beat). The Generate tab's box, without its none: a live "
                 "intro always has a shape."),
                (self.perf_outro, self.outro_style,
                 "How an outro section ends (next kind: outro): the effect blends in over "
                 "the last bars of a shorter groove and rings out to silence by bar 8, "
                 "where the sections stop by themselves; random picks one. The Generate "
                 "tab's box without its none, with its tail, bars, wet and amount."),
                (self.perf_drum_style, self.drum_style,
                 "The drum style the next section is patterned in: the Sections tab's box; "
                 "random picks one per section (the one playing shows in the meter).")):
            # the same names but none (a live intro or outro always has a
            # shape), so the pair is kept by name: the Generate tab on none
            # leaves this box where it was, and the live render reads this one
            mine.addItems([t for t in (theirs.itemText(i) for i in range(theirs.count()))
                           if t != "none"])
            if mine.findText(theirs.currentText()) >= 0:
                mine.setCurrentText(theirs.currentText())
            mine.setToolTip(tip)
            mine.currentTextChanged.connect(theirs.setCurrentText)
            theirs.currentTextChanged.connect(
                lambda t, m=mine: m.setCurrentText(t) if m.findText(t) >= 0 else None)
        form.addRow("Shapes", self._row([("intro", self.perf_intro), ("drums", self.perf_drum_style),
                                         ("outro", self.perf_outro)]))
        self.perf_material = QComboBox()
        self.perf_material.addItems(["takes, else ticked samples", "takes only",
                                     "ticked samples"])
        self.perf_material.setToolTip(
            "What the sections are built from. The takes recorded during the "
            "show are the point; the ticked samples stand in for rehearsal "
            "when there are none. The ticked videos' soundtracks join them "
            "with Video audio on (Video tab), and are the material on their "
            "own when no sample is ticked. Nothing ticked anywhere: nothing.")
        # as narrow as it goes: the row has three boxes, the list opens full width
        self.perf_material.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.perf_material.setMinimumContentsLength(9)
        self.perf_material.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.perf_kind = QComboBox()
        self.perf_kind.addItems(["random"] + list(KINDS) + list(EXTRA_KINDS))
        self.perf_kind.setToolTip(
            "Shape of the next section. groove: full kit, one to three layers, "
            "maybe chops. break: one or two drum voices under more layers. "
            "build: the voices stack up bar by bar. sparse: one voice, one layer. "
            "random: one of those four. intro and outro: 8 bars shaped by the intro "
            "and outro styles above; an outro is entered at bar 5, ends in silence and "
            "stops the sections on its last bar.")
        # the live clock's bpm box again (the Sections tab's), the two one value
        self.perf_bpm = QDoubleSpinBox(minimum=40.0, maximum=240.0, singleStep=1.0,
                                       decimals=1, suffix=" bpm")
        self.perf_bpm.setValue(self.live_bpm.value())
        self.perf_bpm.setToolTip("The live clock's tempo, the bpm box of the Sections tab: "
                                 "the next section renders at it. Type it or tap it (T).")
        self.perf_bpm.valueChanged.connect(self.live_bpm.setValue)
        self.live_bpm.valueChanged.connect(self.perf_bpm.setValue)
        form.addRow("Material", self._row([("", self.perf_material),
                                           ("next kind", self.perf_kind),
                                           ("", self.perf_bpm)]))
        # the takes above the section buttons: what the section is built from,
        # then what plays it
        self.rec_btn = QPushButton("● record (R)")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setMinimumHeight(30)
        self.rec_btn.setToolTip(
            "A take: the live sound and picture from a few seconds before "
            "this press until the next one, kept in RAM. The takes together "
            "are the sample bank the engine builds sections from at the end "
            "of the piece; short ones are fine. Stops by itself at the limit; "
            "the next press then starts a new take. Needs the live input and/or "
            "the video input on. Key R.")
        self.rec_btn.clicked.connect(self.toggle_take)
        drop_btn = QPushButton("forget last (Shift+R)")
        drop_btn.setToolTip(
            "Forget the last take, or drop the one recording, from RAM: the "
            "material for the next section no longer has it. A take already "
            "written to videos/ keeps its file. Key Shift+R, and a MIDI button.")
        drop_btn.clicked.connect(self.discard_take)
        self.take_pre = QDoubleSpinBox(minimum=0.0, maximum=12.0, singleStep=1.0,
                                       decimals=0, suffix=" s")
        self.take_pre.setValue(self.settings.value("takes/pre_roll", 0.0, type=float))
        self.take_pre.valueChanged.connect(
            lambda v: self.settings.setValue("takes/pre_roll", float(v)))
        self.take_pre.setToolTip("Pre-roll: how many seconds before the press "
                                 "each take starts with")
        self.take_limit = QDoubleSpinBox(minimum=5.0, maximum=300.0, singleStep=5.0,
                                         decimals=0, suffix=" s")
        self.take_limit.setValue(60.0)
        self.take_limit.setToolTip("A take stops by itself after this long")
        save_btn = QPushButton("save takes")
        save_btn.setToolTip(
            "Write the takes not yet on disk to videos/takes_<date>/ now, in the "
            "background (the encode can glitch the live sound: between pieces). "
            "With 'keep the takes' ticked in Settings they are written when the "
            "app quits anyway.")
        save_btn.clicked.connect(self._save_takes)
        clear_btn = QPushButton("clear (Ctrl+Shift+R)")
        clear_btn.setToolTip("Forget every take, the one recording included, from RAM: "
                             "the files already written to videos/ stay. Key "
                             "Ctrl+Shift+R, and a MIDI button.")
        clear_btn.clicked.connect(self.clear_takes)
        form.addRow("Takes", self._row([("", self.rec_btn), ("", drop_btn),
                                        ("pre-roll", self.take_pre),
                                        ("limit", self.take_limit), ("", save_btn),
                                        ("", clear_btn)]))
        self.take_status = QLabel("no takes")
        self.take_status.setProperty("role", "sub")
        form.addRow("", self.take_status)
        next_btn = QPushButton("play (N)")     # "next section (N)" once one has played
        next_btn.setObjectName("next")          # the set's own colour, big type
        self.perf_next_btn = next_btn
        next_btn.setMinimumHeight(40)
        next_btn.setToolTip(
            "A new 8-bar section from the material at the live tempo, on the "
            "next bar. One is rendered ahead while the current one loops, so "
            "this is instant; the very first waits about a second.")
        next_btn.clicked.connect(self.perform_next)
        stop_btn = QPushButton("stop / start (Shift+N)")
        stop_btn.setToolTip(
            "Silence the section on the next bar; the tape stays. Pressed "
            "with no section playing, the last one starts again as it was, "
            "nothing new is rendered. With Ableton Link on, this is the "
            "session's stop and play, both ways.")
        stop_btn.clicked.connect(self.perform_stop)
        event_btn = QPushButton("event (E)")
        event_btn.setToolTip("Fire a one-shot cut from the material on the next beat")
        event_btn.clicked.connect(self.perform_event)
        form.addRow("Section", self._row([("", next_btn), ("", stop_btn), ("", event_btn)]))
        self.perf_stems = {}
        keys = {"drums": "F1", "layers": "F2", "chops": "F3", "events": "F4"}
        shuffle_keys = {"drums": "F5", "layers": "F6", "chops": "F7", "events": "F8"}
        pairs = []
        self.perf_shuffles = {}
        # a mixer's way, as the voices below: M (lit while the stem is out),
        # the name, the shuffle. perf_stems holds the M buttons: checked =
        # out; the stem_* actions, the scenes and the pads speak of the
        # stem being in (see _erae_read)
        for stem in STEMS:
            cb = QPushButton("M")
            cb.setObjectName("solo")
            cb.setCheckable(True)
            cb.setFixedWidth(24)
            cb.setToolTip(f"mute the {stem} stem ({keys[stem]}): out with a ramp, on the "
                          "quantize grid, for the section and a song alike, after its "
                          "rack; lit while out.")
            cb.toggled.connect(lambda muted, st=stem: self._perf_stem(st, not muted))
            self.perf_stems[stem] = cb
            lab = QLabel(stem.upper())
            lab.setStyleSheet("font-weight: bold;")
            pairs.append(("", self._pair(cb, lab)))
            sh = QPushButton(f"randomize ({shuffle_keys[stem]})")
            sh.setObjectName("small")
            sh.setToolTip(
                f"The {stem} of the section playing rendered again from the "
                "material: a new sound in that part, the other three as they "
                "are, bar for bar, and the loop where it is. Lands on the next "
                "bar (a second to render). The sub bass follows the drums, the "
                "layers and the chops. Stopped, it waits in the "
                "section for the next start.")
            sh.setCheckable(True)                     # lit while a shuffle is on its way
            sh.clicked.connect(lambda _=False, st=stem: self.perform_shuffle(st))
            self.perf_shuffles[stem] = sh
            pairs.append(("", sh))
        stems_row = self._row(pairs)                   # added under the Break row
        glitch_btn = QPushButton("glitch (G)")
        glitch_btn.setToolTip(
            "A glitch break, on the phrase: it waits for the last bar of the "
            "4-bar phrase (bar 4 or 8 of the loop), then sixteenths of the bar "
            "are thrown about, some backwards, a "
            "stutter roll, a sixteenth of silence, and the next section drops "
            "in on the one. Pressed inside the last bar it runs from the next "
            "beat. With no section playing: plain next.")
        glitch_btn.clicked.connect(lambda: self.perform_break("glitch"))
        break_btn = QPushButton("break (Shift+G)")
        break_btn.setToolTip(
            "The song's break, on the phrase: it waits for the last bar of the "
            "4-bar phrase (bar 4 or 8 of the loop), then the bar's first slice "
            "is beat-repeated, accelerating, steady "
            "or halving, a reverse cymbal swell rises under it and the picture "
            "with it to white, silence right before the drop, then the next "
            "section on the one, with the flash. Pressed inside the last bar it "
            "runs from the next beat. With no section playing: plain next. "
            "Video FX flash sets the picture's share.")
        break_btn.clicked.connect(lambda: self.perform_break("break"))
        self.perf_gap = QCheckBox("gap before the drop")
        self.perf_gap.setToolTip(
            "The silence right before the one: a sixteenth after the glitch's "
            "roll, half a beat or a beat after the break's repeat, as in the "
            "generated songs. Off, the roll and the repeat run straight into "
            "the drop. Remembered; takes effect from the next break.")
        self.perf_gap.setChecked(self.settings.value("perform/gap", False, type=bool))
        self.perf_gap.toggled.connect(self._perf_gap_changed)
        form.addRow("Break", self._row([("", glitch_btn), ("", break_btn), ("", self.perf_gap)]))
        form.addRow("Stems", stems_row)
        drums_at = form.rowCount()                     # the Drums row goes right under
        # the repeat: held from a key, a note or this latching button
        self._rep_sources = set()
        self.rep_btn = QPushButton("repeat (H)")
        self.rep_btn.setCheckable(True)
        self.rep_btn.setToolTip(
            "Hold H (or the MIDI note): the cell of the bar playing at the press, "
            "the division next to it long, loops for as long as it is held; the "
            "section runs on underneath, so letting go lands where the loop would "
            "have been, on the bar. The cells sit on the bar's grid, so a change "
            "of division or position is in time at once. This button latches it "
            "for the mouse. A glitch or a break wins over it; a stop ends it.")
        self.rep_btn.toggled.connect(lambda on: self._rep_hold("ui", on))
        self.rep_div = QComboBox()
        self.rep_div.addItems([name for name, _f in REPEAT_DIVS])
        self.rep_div.setToolTip(
            "The cell the repeat loops: the bar down to a 1/64 note (a pitched "
            "buzz by then). Finer while held tightens the roll from the same "
            "one; coarser widens it. - and = step it, a MIDI knob sweeps it. "
            "Remembered.")
        d = self.settings.value("perform/rep_div", REPEAT_DEFAULT)
        if self.rep_div.findText(str(d)) >= 0:
            self.rep_div.setCurrentText(str(d))
        self.rep_div.currentTextChanged.connect(self._rep_div_changed)
        self.rep_pos = QSlider(Qt.Horizontal)
        self.rep_pos.setRange(-REPEAT_RANGE, REPEAT_RANGE)
        self.rep_pos.setValue(0)
        self.rep_pos.setTickPosition(QSlider.TicksBelow)
        self.rep_pos.setTickInterval(4)
        self.rep_pos.setToolTip(
            "Where the repeat reads: 0 is the cell captured at the press, each "
            "step one cell of the current division earlier (left) or later "
            "(right), round the loop's ends. , and . step it, a MIDI knob sweeps "
            "it with the captured cell in the middle.")
        self.rep_pos.valueChanged.connect(self._rep_pos_changed)
        self.rep_pos_lbl = QLabel("0")
        self.rep_pos_lbl.setProperty("role", "sub")
        self.rep_pos_lbl.setFixedWidth(28)
        self.rep_pos_lbl.setAlignment(Qt.AlignCenter)
        zero_btn = QPushButton("0")
        zero_btn.setFixedWidth(24)
        zero_btn.setToolTip("The position back to the captured cell")
        zero_btn.clicked.connect(lambda: self.rep_pos.setValue(0))
        self.rep_rev = QCheckBox("reverse (/)")
        self.rep_rev.setToolTip("The repeated cell played backwards; / switches it")
        self.rep_rev.toggled.connect(lambda _on: self._rep_apply())
        form.addRow("Repeat", self._row([("", self.rep_btn), ("", self.rep_div),
                                         ("position", self.rep_pos), ("", self.rep_pos_lbl),
                                         ("", zero_btn), ("", self.rep_rev)]))
        # the speed: half and double time, the tape stop, the nudge, the pitch
        self.half_btn = QPushButton("half time (Shift+H)")
        self.half_btn.setCheckable(True)
        self.half_btn.setToolTip(
            "The bar at half speed from its start, resynced every sync below, so the "
            "clock, the bar, Link and the MIDI clock never move: the Gross Beat way. "
            "An octave down with the pitch following; at pitch with it locked. Lands "
            "on the quantize grid. A MIDI button; never remembered.")
        self.half_btn.toggled.connect(lambda on: self._speed_toggled("half", on))
        self.double_btn = QPushButton("double (Ctrl+H)")
        self.double_btn.setCheckable(True)
        self.double_btn.setToolTip(
            "The bar at double speed from its start: this bar and the next inside one, "
            "resynced every sync below. An octave up, or at pitch when locked. Lands on "
            "the quantize grid. A MIDI button; never remembered.")
        self.double_btn.toggled.connect(lambda on: self._speed_toggled("double", on))
        self.speed_sync = QComboBox()
        self.speed_sync.addItems([name for name, _b in SPEED_SYNCS])
        s = self.settings.value("perform/speed_sync", SPEED_SYNC_DEFAULT)
        self.speed_sync.setCurrentText(
            str(s) if self.speed_sync.findText(str(s)) >= 0 else SPEED_SYNC_DEFAULT)
        self.speed_sync.setToolTip(
            "How often half and double time jump back to the transport: every half "
            "bar, bar, 2 or 4 bars, or only at the loop's end (half time then plays the "
            "whole loop over two loops). Remembered; a MIDI knob.")
        self.speed_sync.currentTextChanged.connect(self._speed_sync_changed)
        self.pitch_mode = QComboBox()
        self.pitch_mode.addItems(PITCH_MODES)
        p = self.settings.value("perform/pitch", PITCH_MODES[0])
        self.pitch_mode.setCurrentText(str(p) if str(p) in PITCH_MODES else PITCH_MODES[0])
        self.pitch_mode.setToolTip(
            "What the pitch does under half and double time, a nudge and a tempo change: "
            "follows the speed (varispeed, the Looper's way: half time is an octave "
            "down), or locked (a granular stretch keeps it, the warped clip's way; the "
            "drums smear a little at half speed). The tape stop always drops, that is "
            "the point. Remembered; the MIDI button pitch_lock flips it.")
        self.pitch_mode.currentTextChanged.connect(self._pitch_changed)
        self._brake_sources = set()
        self.brake_btn = QPushButton("tape stop (Shift+B)")
        self.brake_btn.setCheckable(True)
        self.brake_btn.setToolTip(
            "Hold Shift+B (or the MIDI note): the section decelerates to a stop over the "
            "length next to it, the pitch falling with it and the picture slowing to a "
            "freeze; let go, it spins back up over the same slope and the transport, "
            "which ran on underneath, is heard again on the grid: the bar, the clock, "
            "Link and a queued section are untouched. This button latches it for the "
            "mouse. A break wins over it; a stop ends it.")
        self.brake_btn.toggled.connect(lambda on: self._brake_hold("ui", on))
        self.brake_len = QComboBox()
        self.brake_len.addItems([name for name, _b in BRAKE_LENS])
        b = self.settings.value("perform/brake_len", BRAKE_DEFAULT)
        self.brake_len.setCurrentText(
            str(b) if self.brake_len.findText(str(b)) >= 0 else BRAKE_DEFAULT)
        self.brake_len.setToolTip("From full speed to the stop, in bars at the live tempo; "
                                  "the spin-up takes as long. Remembered; a MIDI knob.")
        self.brake_len.currentTextChanged.connect(self._brake_len_changed)
        self._nudge_sources = {-1: set(), 1: set()}
        self.nudge_down_btn = QPushButton("nudge − (J)")
        self.nudge_down_btn.setToolTip(
            "Held: the section a few percent slower (the amount next to it) and the "
            "live clock with it, so the shaders, the LFOs and the MIDI clock slide too: "
            "to line up with a tape or a drum machine running on its own clock. Not "
            "under Link, which owns the beat. The pitch follows unless locked.")
        self.nudge_down_btn.pressed.connect(lambda: self._nudge_hold(-1, "ui", True))
        self.nudge_down_btn.released.connect(lambda: self._nudge_hold(-1, "ui", False))
        self.nudge_up_btn = QPushButton("nudge + (K)")
        self.nudge_up_btn.setToolTip("Held: a few percent faster; see nudge −")
        self.nudge_up_btn.pressed.connect(lambda: self._nudge_hold(1, "ui", True))
        self.nudge_up_btn.released.connect(lambda: self._nudge_hold(1, "ui", False))
        self.nudge_pct = QSpinBox()
        self.nudge_pct.setRange(1, 10)
        self.nudge_pct.setSuffix(" %")
        self.nudge_pct.setValue(self.settings.value("perform/nudge_pct", NUDGE_DEFAULT, type=int))
        self.nudge_pct.setToolTip("How much a nudge speeds up or slows down. Remembered.")
        self.nudge_pct.valueChanged.connect(
            lambda v: self.settings.setValue("perform/nudge_pct", int(v)))
        form.addRow("Speed", self._row([("", self.half_btn), ("", self.double_btn),
                                        ("sync", self.speed_sync), ("pitch", self.pitch_mode)]))
        form.addRow("", self._row([("", self.brake_btn), ("", self.brake_len),
                                   ("", self.nudge_down_btn), ("", self.nudge_up_btn),
                                   ("", self.nudge_pct)]))
        # the fragments: the bar sliced, a slice held sounding as grains
        # (gacha_frag), Bitwig's Sampler in Sliced mode playing Fragments
        sv = self.settings.value
        self.frag_div = QComboBox()
        self.frag_div.addItems([str(d) for d in FRAG_DIVS])
        self.frag_div.setCurrentText(str(sv("perform/frag_div", FRAG_DEFAULTS["div"], type=int)))
        self.frag_div.setToolTip(
            "Slices per bar: the bar playing is cut into that many equal pieces, each a "
            "MIDI note (frag_1 to frag_16, the fragments group of the Controls tab; past "
            "16 only the slice pick reaches them). Remembered; a MIDI knob.")
        self.frag_slice = QComboBox()
        self._frag_fill_slices(str(sv("perform/frag_slice", "follow")))
        self.frag_slice.setToolTip(
            "The slice the F key and the button hold: follow is the slice under the "
            "playhead at the press, the sampler's way of playing what is there; or a "
            "fixed one. Remembered; a MIDI knob.")
        self.frag_size = QComboBox()
        self.frag_size.addItems([name for name, _f in FRAG_UNITS])
        self.frag_size.setCurrentText(str(sv("perform/frag_size", FRAG_DEFAULTS["size"])))
        self.frag_size.setToolTip(
            "A grain's length, as a fraction of the bar (Bitwig's beat units; a 16th by "
            "default), so a tempo change keeps the grains in time: a 64th buzzes, an 8th "
            "keeps whole hits. Remembered; a MIDI knob.")
        self.frag_rate = QComboBox()
        self.frag_rate.addItems([name for name, _f in FRAG_RATES])
        self.frag_rate.setCurrentText(str(sv("perform/frag_rate", FRAG_DEFAULTS["rate"])))
        self.frag_rate.setToolTip(
            "How often a grain starts: dense is eight of them overlapping whatever the "
            "size (a cloud); a beat unit starts one on every such line of the bar's grid, "
            "in time with the loop (a 16th with 16th grains is the slice stuttered on the "
            "grid, a 16th with 64th grains a rhythm of blips). Remembered; a MIDI knob.")
        self.frag_motion = QSpinBox(minimum=0, maximum=800, suffix=" %")
        self.frag_motion.setValue(sv("perform/frag_motion", FRAG_DEFAULTS["motion"], type=int))
        self.frag_motion.setToolTip(
            "How far from the playhead each grain lands, up to that many grain sizes "
            "either way at random, and its place in the stereo field with it: 0 is the "
            "slice replayed as it is, 100 % a smear, 800 % the whole slice as a cloud. "
            "Remembered.")
        self.frag_pitch = QSpinBox(minimum=-24, maximum=24, suffix=" st")
        self.frag_pitch.setValue(sv("perform/frag_pitch", FRAG_DEFAULTS["pitch"], type=int))
        self.frag_pitch.setToolTip("The grains' pitch, two octaves either way; the playhead's "
                                   "speed is untouched by it. Remembered.")
        self.frag_speed = QSpinBox(minimum=-200, maximum=200, suffix=" %")
        self.frag_speed.setValue(sv("perform/frag_speed", FRAG_DEFAULTS["speed"], type=int))
        self.frag_speed.setToolTip(
            "The playhead's speed through the slice: 100 % is the section's own, 50 % "
            "half time at pitch, 0 frozen on the spot (a drone of that instant), "
            "negative backwards. Remembered.")
        self.frag_repeats = QSpinBox(minimum=1, maximum=16)
        self.frag_repeats.setValue(sv("perform/frag_repeats", FRAG_DEFAULTS["repeats"], type=int))
        self.frag_repeats.setToolTip("Each grain triggered that many times over, the same spot "
                                     "and place: a roll, denser. Remembered.")
        self.frag_shape = QSpinBox(minimum=0, maximum=100, suffix=" %")
        self.frag_shape.setValue(sv("perform/frag_shape", FRAG_DEFAULTS["shape"], type=int))
        self.frag_shape.setToolTip("The grain's envelope: 0 a pluck (a short attack, the rest "
                                   "falling), 50 % a Hann, 100 % open (short ramps, flat "
                                   "between). Remembered.")
        self.frag_dir = QComboBox()
        self.frag_dir.addItems(FRAG_DIRS)
        self.frag_dir.setCurrentText(str(sv("perform/frag_dir", FRAG_DEFAULTS["dir"])))
        self.frag_dir.setToolTip("The grains' direction: forward, reverse, each at random, or "
                                 "in turn. Remembered; a MIDI button steps it.")
        self.frag_play = QComboBox()
        self.frag_play.addItems(FRAG_PLAYS)
        self.frag_play.setCurrentText(str(sv("perform/frag_play", FRAG_DEFAULTS["play"])))
        self.frag_play.setToolTip("What the playhead does at the slice's ends: loop round, "
                                  "ping-pong, or once (the voice releases there). Remembered; "
                                  "a MIDI button steps it.")
        self.frag_a = QSpinBox(minimum=0, maximum=2000, suffix=" ms")
        self.frag_a.setValue(sv("perform/frag_a", FRAG_DEFAULTS["a"], type=int))
        self.frag_a.setToolTip("Attack: the voice fades in over it on the press, the section "
                               "out. Remembered.")
        self.frag_d = QSpinBox(minimum=0, maximum=2000, suffix=" ms")
        self.frag_d.setValue(sv("perform/frag_d", FRAG_DEFAULTS["d"], type=int))
        self.frag_d.setToolTip("Decay, down to the sustain. Remembered.")
        self.frag_s = QSpinBox(minimum=0, maximum=100, suffix=" %")
        self.frag_s.setValue(sv("perform/frag_s", FRAG_DEFAULTS["s"], type=int))
        self.frag_s.setToolTip("Sustain, the level held while the key is down. Remembered.")
        self.frag_r = QSpinBox(minimum=0, maximum=4000, suffix=" ms")
        self.frag_r.setValue(sv("perform/frag_r", FRAG_DEFAULTS["r"], type=int))
        self.frag_r.setToolTip("Release: the voice fades out over it when let go, the section "
                               "back in. Remembered.")
        for w in (self.frag_slice, self.frag_size, self.frag_rate, self.frag_dir, self.frag_play):
            w.currentTextChanged.connect(self._frag_changed)
        for w in (self.frag_motion, self.frag_pitch, self.frag_speed, self.frag_repeats,
                  self.frag_shape, self.frag_a, self.frag_d, self.frag_s, self.frag_r):
            w.valueChanged.connect(self._frag_changed)
        self.frag_div.currentTextChanged.connect(self._frag_div_changed)
        self._frag_sources = {}                 # slice key -> who holds it
        self._frag_pick_key = 0                 # the key the F key is holding
        self.frag_btn = QPushButton("fragments (F)")
        self.frag_btn.setCheckable(True)
        self.frag_btn.setToolTip(
            "Hold F (or the MIDI note; a note per slice is in the Controls tab's fragments "
            "group): the slice picked next to it, of the bar playing, sounds as grains in "
            "place of the section, through the same stems, racks and switches; the "
            "section runs on underneath and comes back with the release. Up to four "
            "slices at once from the notes. This button latches it for the mouse.")
        self.frag_btn.toggled.connect(lambda on: self._frag_hold_pick("ui", on))
        form.addRow("Fragments", self._row([("", self.frag_btn), ("slices", self.frag_div),
                                            ("slice", self.frag_slice), ("size", self.frag_size),
                                            ("rate", self.frag_rate), ("motion", self.frag_motion),
                                            ("pitch", self.frag_pitch), ("speed", self.frag_speed)]))
        form.addRow("", self._row([("repeats", self.frag_repeats), ("shape", self.frag_shape),
                                   ("dir", self.frag_dir), ("play", self.frag_play),
                                   ("A", self.frag_a), ("D", self.frag_d), ("S", self.frag_s),
                                   ("R", self.frag_r)]))
        self._frag_changed()
        # the freeze (gacha_freeze): the whole mix held on the instant of
        # the press, on the master, the picture on its frame
        self._freeze_sources = set()
        self.freeze_btn = QPushButton("freeze (Z)")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.setToolTip(
            "Hold Z (or the MIDI note): the whole mix as heard (the tape through, the "
            "section, the song, the bursts, after the racks) is held on the instant of the "
            "press, a drone of grains read from that spot, the live mix fading out under "
            "it over the attack while the transport runs on; let go, the mix comes back "
            "over the release right where the loop is. The picture holds its frame. The "
            "MIDI out's hits pause under it. This button latches it for the mouse.")
        self.freeze_btn.toggled.connect(lambda on: self._freeze_hold("ui", on))
        self.freeze_size = QSpinBox(minimum=20, maximum=250, suffix=" ms")
        self.freeze_size.setValue(sv("perform/freeze_size", 80, type=int))
        self.freeze_size.setToolTip("A grain's length: short is a buzz at the grain's pitch, "
                                    "long keeps the moment's texture. Remembered; a MIDI knob.")
        self.freeze_smear = QSpinBox(minimum=0, maximum=100, suffix=" %")
        self.freeze_smear.setValue(sv("perform/freeze_smear", 0, type=int))
        self.freeze_smear.setToolTip(
            "0 is the instant itself, a clean drone (the grains lined up in phase); up to "
            "half it jitters the spot (up to half a second back), detunes a little and "
            "sprays the stereo; past half, octave and fifth scatter and reversed grains, a "
            "cloud. Remembered; a MIDI knob.")
        self.freeze_pitch = QSpinBox(minimum=-24, maximum=24, suffix=" st")
        self.freeze_pitch.setValue(sv("perform/freeze_pitch", 0, type=int))
        self.freeze_pitch.setToolTip("The drone's pitch, two octaves either way. "
                                     "Remembered; a MIDI knob.")
        self.freeze_a = QSpinBox(minimum=1, maximum=2000, suffix=" ms")
        self.freeze_a.setValue(sv("perform/freeze_a", 10, type=int))
        self.freeze_a.setToolTip("The live mix out and the drone in, on the press. "
                                 "Remembered; a MIDI knob.")
        self.freeze_r = QSpinBox(minimum=1, maximum=4000, suffix=" ms")
        self.freeze_r.setValue(sv("perform/freeze_r", 250, type=int))
        self.freeze_r.setToolTip("The drone out and the live mix back, on the let-go. "
                                 "Remembered; a MIDI knob.")
        for w in (self.freeze_size, self.freeze_smear, self.freeze_pitch, self.freeze_a,
                  self.freeze_r):
            w.valueChanged.connect(self._freeze_changed)
        form.addRow("Freeze", self._row([("", self.freeze_btn), ("size", self.freeze_size),
                                         ("smear", self.freeze_smear),
                                         ("pitch", self.freeze_pitch), ("A", self.freeze_a),
                                         ("R", self.freeze_r)]))
        self._freeze_changed()
        burst_btn = QPushButton("burst (B)")
        burst_btn.setToolTip(
            "A slice of the material from a random spot, right now, at the "
            "tape's own level and heard whatever the A/B pedal says; its frame "
            "takes the whole picture while it sounds (from a take with video).")
        burst_btn.clicked.connect(self.perform_burst)
        self.burst_len = QDoubleSpinBox(minimum=10.0, maximum=1000.0, singleStep=10.0,
                                        decimals=0, suffix=" ms")
        self.burst_len.setValue(self.settings.value("perform/burst_ms", 100.0, type=float))
        self.burst_len.setToolTip("How long a burst lasts, 10 ms to 1 s. Remembered; "
                                  "a MIDI knob sweeps it on a log scale (Controls tab).")
        self.burst_len.valueChanged.connect(
            lambda v: self.settings.setValue("perform/burst_ms", float(v)))
        form.addRow("Burst", self._row([("", burst_btn), ("length", self.burst_len)]))
        # the drum voices, a mixer's way: the name, M (lit while muted), S
        # (lit while soloed), landing on the quantize grid. perf_voices holds
        # the M buttons: checked = muted; the voice_* actions, the scenes and
        # the pads still speak of the voice being in (see _erae_read)
        self.perf_voices, self.perf_solos, self.perf_randoms = {}, {}, {}
        pairs = []
        for i, v in enumerate(VOICES, 1):
            lab = QLabel(VOICE_SHORT[v].upper())
            cb = QPushButton("M")
            cb.setObjectName("solo")
            cb.setCheckable(True)
            cb.setFixedWidth(24)
            cb.setToolTip(f"mute the {VOICE_LABELS[v]} (Shift+F{i}): out on the quantize grid "
                          "while a section plays, lit while out; its MIDI notes stop with it "
                          "only with 'follow mutes' ticked in the MIDI out panel. A delay "
                          "or a reverb on its rack (Audio FX tab) rings out after the "
                          "mute. Every voice starts in at launch.")
            cb.toggled.connect(lambda muted, v=v: self._perf_voice(v, not muted))
            so = QPushButton("S")
            so.setObjectName("solo")
            so.setCheckable(True)
            so.setFixedWidth(24)
            so.setToolTip(f"solo the {VOICE_LABELS[v]} (Ctrl+F{i}): every voice not soloed "
                          "is out while any is, on the quantize grid")
            so.toggled.connect(lambda on, v=v: self._perf_solo(v, on))
            self.perf_voices[v], self.perf_solos[v] = cb, so
            cells = [cb, so]
            if v in KIT_PARTS:                        # the sub is synthesized: no sample to swap
                rb = QPushButton("R")
                rb.setObjectName("solo")
                rb.setCheckable(True)                 # lit while the new sound is on its way
                rb.setFixedWidth(24)
                rb.setToolTip(
                    f"Another sample for the {VOICE_LABELS[v]} of the section playing, "
                    "carved into that drum, on the same hits: the sound changes, the "
                    "groove does not. Lands on the next bar, a second to render; lit "
                    "until then. Pressed again, another one. The drum pad follows it.")
                rb.clicked.connect(lambda _=False, v=v: self.perform_shuffle(v))
                self.perf_randoms[v] = rb
                cells.append(rb)
            box = QHBoxLayout()                       # [M] [S] [R] [KICK]
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(4)
            for x in (*cells, lab):
                box.addWidget(x)
            cell = QWidget()
            cell.setLayout(box)
            pairs.append(("", cell))
        form.insertRow(drums_at, "Drums", self._row(pairs))   # under the Stems row
        self.perf_quant = QComboBox()
        self.perf_quant.addItems([name for name, _f in QUANT_GRIDS])
        self.perf_quant.setToolTip(
            "Where a voice, solo or stem switch, an event (E) and a burst (B) "
            "land while a section plays: at once, or on the loop's next "
            "sixteenth, eighth, beat, half bar or bar. With no section playing "
            "everything is at once. Remembered; quant_next in the Controls tab "
            "steps through it.")
        q = self.settings.value("perform/quant", QUANT_DEFAULT)
        if self.perf_quant.findText(str(q)) >= 0:
            self.perf_quant.setCurrentText(str(q))
        self.perf_quant.currentTextChanged.connect(self._quant_changed)
        reset_btn = QPushButton("all voices in (Shift+F8)")
        reset_btn.setToolTip("Every drum voice back in, no solo, the voice racks' effects "
                             "off and their volumes at 1")
        reset_btn.clicked.connect(self._voices_reset)
        form.insertRow(drums_at + 1, "Quantize",            # under the Drums row
                       self._row([("", self.perf_quant), ("", reset_btn)]))
        self.perf_ab = QSlider(Qt.Horizontal)
        self.perf_ab.setRange(0, 100)
        self.perf_ab.setValue(0)
        self.perf_ab.setToolTip("A/B: 0 = the tape through only, 100 = the section "
                                "only, equal power in between; the picture "
                                "dissolves with it (Picture = follow). With the "
                                "second video input on B (W) the tape's sound stays "
                                "whole and only the section comes in, since the "
                                "webcam is silent. Keys [ and ] "
                                "step by 10; a MIDI expression pedal drives it "
                                "through the Controls tab. Goes to full B on its "
                                "own when the video input (Video tab) is off.")
        self.perf_ab.valueChanged.connect(self._perf_ab_changed)
        self.perf_ab_lbl = QLabel("A 100 %  ·  B 0 %")
        self.perf_ab.setValue(100)                    # video in starts off: full B
        self.perf_ab_lbl.setProperty("role", "sub")
        form.addRow("A/B", self._row([("", self.perf_ab), ("", self.perf_ab_lbl)]))
        # the section big, for the eyes on stage: what plays and the bar
        self.perf_meter = PerfMeter()
        self.perf_meter.setToolTip("What plays and where the loop is: the current bar lit, "
                                   "a sweep through it over the bar, a dot per beat, a mark "
                                   "after each 4-bar phrase (where a break lands). A held "
                                   "repeat draws its window in yellow over the cells, with "
                                   "its own playhead and a chevron for the direction; the "
                                   "sweep underneath is the loop running on, where letting "
                                   "go lands.")
        form.addRow(self.perf_meter)
        # the small status line: kept for the code that writes it, not shown;
        # the meter's big line carries what it said
        self.perf_status = QLabel("no section")
        self.perf_status.hide()
        # in a scroll area, like the FX tabs: the page is the tallest and
        # would otherwise set the window's minimum height, which at a high
        # DPI scale can top the screen's work area and stop the window
        # from maximizing in one go; at full size it shows whole
        area = self._scrolling(self._wrap(form))
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return area

    def _tab_video(self):
        form = QFormLayout()
        self.video_jump = QCheckBox("jump to a random spot on every section")
        self.video_jump.setChecked(True)
        self.video_jump.setToolTip("The backdrop video seeks to a random place when a "
                                   "song's section changes (every 8 bars of the live "
                                   "clock too), so the picture moves with the music "
                                   "instead of running through; unticked it plays on. "
                                   "Instant on transcoded videos.")
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
            "(Sections page) to 'all' if it was off, so cuts land anywhere in "
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
            "Pick which layers the picture follows. drums: the picture cuts "
            "on every kick and snare to the frame it was carved at and runs "
            "on from there until the next one (2 s at most), hats and glitches "
            "ignored, chops and textures filling a break. everything: a kick "
            "or snare wins over a hat, a hat over a chop, a chop over a "
            "texture (drums nearly always). varies: which layer leads is "
            "re-rolled at every section and every 4 bars, textures and chops "
            "a little more often than the drums, the hats rarely. Off: the "
            "backdrop behaves as usual.")
        form.addRow("", self._row([("picture follows", self.video_follow)]))
        # the picture under a section and how the A/B pedal blends it: the
        # Perform tab's rows once, here with the rest of the picture's choices
        self.perf_picture = QComboBox()
        self.perf_picture.addItems(["follow", "take", "live"])
        self.perf_picture.setToolTip(
            "What the screen shows while a section plays: follow = the picture "
            "dissolves from the live picture to the take frames with the A/B "
            "pedal, the way the sound does; take = the frames the sounds were "
            "cut from (needs takes with video); live = the live picture as it "
            "comes in. With the second video input on B (W), the webcam stands "
            "in for the take frames in all three.")
        self.perf_dissolve = QComboBox()
        self.perf_dissolve.addItems(BLEND_MODES)
        self.perf_dissolve.setToolTip(
            "How the tape and the B layer (the take frames, or the webcam with "
            "W) meet under the A/B pedal. All "
            "three are the tape alone at 0 and B alone at 100; mix is a "
            "plain dissolve, add and screen lay the two pictures over each "
            "other at full strength halfway (add clips bright on bright, "
            "screen never clips).")
        self.perf_dissolve.currentIndexChanged.connect(
            lambda i: setattr(self.backdrop, "override_blend", i))
        form.addRow("Picture", self._row([("", self.perf_picture),
                                          ("dissolve", self.perf_dissolve)]))
        self._refresh_through_devices()
        self.live_auto = QCheckBox("auto: tempo, beat and root note from the audio")
        self.live_auto.setChecked(True)
        self.live_auto.setToolTip(
            "Listens along: the tempo and beat phase from the onsets (kicks "
            "first), the root note from a running chroma for the tint. Needs "
            "about 4 s of music. Tapping T switches this off and follows "
            "your taps instead; X marks a drop by hand and restarts the bar. "
            "With Ableton Link on (Settings tab) the tempo is the session's "
            "and only the root note is taken from the audio.")
        self.live_status = QLabel("")
        self.live_status.setProperty("role", "sub")
        form.addRow("", self._row([("", self.live_auto), ("", self.live_status)]))
        self._refresh_live_devices()

        self.auto_hide = QCheckBox("hide the interface when a song starts, "
                                   "and again after")
        self.auto_hide.setChecked(True)
        self.auto_hide.setToolTip("The picture takes the whole window when a song "
                                  "starts; moving the mouse brings the interface back, "
                                  "and it goes again after the seconds next to it, for "
                                  "as long as the song plays.")
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
        self.shader_blend.setToolTip("How the shader layer meets the video at its "
                                     "opacity: mix is a plain crossfade, add and screen "
                                     "lay the two over each other (add clips bright on "
                                     "bright, screen never clips)")
        self.shader_blend.currentIndexChanged.connect(
            lambda i: setattr(self.backdrop, "shader_blend", i))
        form.addRow("Shader layer", self._row([("opacity", self.shader_mix),
                                               ("blend", self.shader_blend)]))
        self.video_plain = QCheckBox("video only: no effects, no shader layer (C)")
        self.video_plain.setToolTip(
            "Show the video as it is, for as long as this is ticked: every "
            "effect below and the shader layer are bypassed. The "
            "picture still follows the song (sample timeline, section jumps, "
            "clip switches, rewinds, the outro fade). Key C toggles it.")
        form.addRow("", self.video_plain)
        self.video_off = QCheckBox("video off: no picture, no decoding, no GPU work (V)")
        self.video_off.setToolTip(
            "For a night of sound only, or a weak machine: the clip and the "
            "capture devices are closed (free for other programs), the "
            "repaint stops, the effects, the shader layer and the section's "
            "frames cost nothing; the window shows its background. Songs, "
            "the live input, sections and the MIDI out run as usual; a take "
            "recorded meanwhile has sound only. Ticking it off brings back "
            "what was chosen. Remembered.")
        self.video_off.toggled.connect(self._toggle_video_off)
        form.addRow("", self.video_off)

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
                ("Grain", ("grain", "gran")),
                ("Motion", ("zoom", "trails", "glitch", "rgb", "reverse")),
                ("Frame", ("vign", "kaleido", "clean"))]
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

    def _tab_lfo(self):
        """Eight LFOs, every slot a row: on, destination (any knob of the
        Controls tab), shape, rate (a cycle length in bars, on the live
        clock), depth, phase, polarity, mode, and its wave right now. The
        rows are remembered as the stage setup, on states included."""
        self.lfo = LfoEngine(LFO_SLOTS, self._lfo_get, self._lfo_set,
                             lambda: (self.tempo.bpm, self.tempo.anchor))
        self.lfo_rows = []
        grid = QGridLayout()
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        heads = ["", "on", "destination", "shape", "rate", "depth", "phase", "pol.", "mode", ""]
        for c, head in enumerate(heads):
            lbl = QLabel(head)
            lbl.setProperty("role", "sub")
            grid.addWidget(lbl, 0, c)
        saved = self._lfo_saved()
        for i in range(LFO_SLOTS):
            r = i + 1
            st = saved[i] if i < len(saved) else {}
            num = QLabel(str(r))
            num.setProperty("role", "sub")
            on = QPushButton("on")
            on.setCheckable(True)
            on.setFixedWidth(36)
            on.setToolTip(f"LFO {r} in or out; out, its knob goes back where you set it. "
                          "Remembered. A MIDI button in the Controls tab's LFOs group.")
            dest = QComboBox()                # a knob always: off is the "on" button
            for ident, label in LFO_DESTS:
                dest.addItem(label, ident)
            dest.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            dest.setMinimumContentsLength(20)
            dest.setToolTip("The knob this LFO moves: any knob of the Controls tab. The "
                            "knob's own value is the centre; turn it, or its CC, while the "
                            "LFO runs and that is the new centre.")
            searchable(dest)
            shape = QComboBox()
            shape.addItems(LFO_SHAPES)
            shape.setToolTip("sine, triangle, saw up, saw down, square; random step (a new "
                             "value each cycle, held), random smooth (the same, glided), "
                             "random walk (a drift, each cycle a step from the last)")
            rate = QComboBox()
            rate.addItems([name for name, _b in LFO_RATES])
            rate.setToolTip("One cycle this long, on the live clock (a playing section's "
                            "bars, Link, the taps): every LFO is locked to the bar and in "
                            "phase with the others")
            depth = QSpinBox(minimum=-100, maximum=100, singleStep=5, suffix=" %")
            depth.setToolTip("How far the knob moves: ± half its range at 100 % (a centred "
                             "knob sweeps the whole range), the whole range upward in + "
                             "polarity; negative flips the wave. A MIDI knob in the LFOs "
                             "group sets 0 to 100 %, keeping the sign.")
            phase = QSpinBox(minimum=0, maximum=359, singleStep=45, suffix="°")
            phase.setToolTip("Where in the cycle the one lands: 180° puts two LFOs on "
                             "two knobs in opposition")
            pol = QComboBox()
            pol.addItems(["±", "+"])
            pol.setToolTip("± moves the knob either way around its value; + moves it from "
                           "its value upward only (a filter opening from where it sits)")
            mode = QComboBox()
            mode.addItems(LFO_MODES)
            mode.setToolTip("cycle: free on the clock. trig: restarts at a section start, "
                            "a drop or the retrigger button. one-shot: runs one cycle from "
                            "the trigger and holds its end, an envelope (its start before "
                            "any trigger).")
            bar = LfoBar()
            row = {"on": on, "dest": dest, "shape": shape, "rate": rate, "depth": depth,
                   "phase": phase, "pol": pol, "mode": mode, "bar": bar}
            self.lfo_rows.append(row)
            # the saved row, before the signals: nothing is written yet
            j = dest.findData(st.get("dest", ""))
            dest.setCurrentIndex(max(0, j))
            if st.get("shape") in LFO_SHAPES:
                shape.setCurrentText(st["shape"])
            rate.setCurrentText(st.get("rate", LFO_RATE_DEFAULT)
                                if rate.findText(str(st.get("rate", ""))) >= 0 else LFO_RATE_DEFAULT)
            depth.setValue(int(round(100 * float(st.get("depth", 0.5)))))
            phase.setValue(int(round(360 * float(st.get("phase", 0.0)))) % 360)
            pol.setCurrentIndex(1 if st.get("unipolar") else 0)
            if st.get("mode") in LFO_MODES:
                mode.setCurrentText(st["mode"])
            on.setChecked(bool(st.get("on", False)))
            self._lfo_push(i, save=False)
            on.toggled.connect(lambda _v, i=i: self._lfo_push(i))
            dest.currentIndexChanged.connect(lambda _v, i=i: self._lfo_push(i))
            for cb in (shape, rate, pol, mode):
                cb.currentTextChanged.connect(lambda _v, i=i: self._lfo_push(i))
            for sp in (depth, phase):
                sp.valueChanged.connect(lambda _v, i=i: self._lfo_push(i))
            for c, wdg in enumerate((num, on, dest, shape, rate, depth, phase, pol, mode, bar)):
                grid.addWidget(wdg, r, c)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 1)
        hint = self._info("An LFO moves a knob around where you set it, on the live clock, so "
                      "every LFO sits on the bar and in step with the others. The knob "
                      "wiggles on screen; its consumers (a rack, the picture, the A/B mix) "
                      "follow. Off writes the knob back. The depth knobs and the on "
                      "buttons are MIDI rows (Controls tab, LFOs group); Shift+L is the "
                      "panic: every LFO off.")
        off_btn = QPushButton("all off (Shift+L)")
        off_btn.setToolTip("Every LFO off, the knobs back where you set them")
        off_btn.clicked.connect(self._lfo_all_off)
        trig_btn = QPushButton("retrigger")
        trig_btn.setToolTip("Every trig and one-shot LFO restarts its cycle now")
        trig_btn.clicked.connect(lambda: self.lfo.trigger(time.monotonic()))
        self.lfo_status = QLabel("")
        self.lfo_status.setProperty("role", "sub")
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 8, 8)
        row.addWidget(self.lfo_status, stretch=1)
        row.addWidget(trig_btn)
        row.addWidget(off_btn)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(grid)
        outer.addWidget(hint)
        outer.addLayout(row)
        outer.addStretch(1)
        page = QWidget()
        page.setLayout(outer)
        self._lfo_ticks = 0
        return self._scrolling(page)

    def _lfo_saved(self):
        try:
            rows = json.loads(self.settings.value("lfo/slots", "[]"))
            return rows if isinstance(rows, list) else []
        except (ValueError, TypeError):
            return []

    def _lfo_row_settings(self, i):
        r = self.lfo_rows[i]
        return {"on": r["on"].isChecked(), "dest": r["dest"].currentData() or "",
                "shape": r["shape"].currentText(), "rate": r["rate"].currentText(),
                "depth": r["depth"].value() / 100.0, "phase": r["phase"].value() / 360.0,
                "unipolar": r["pol"].currentIndex() == 1, "mode": r["mode"].currentText()}

    def _lfo_push(self, i, save=True):
        """Row i's widgets into the engine; the rows into the settings."""
        self.lfo.configure(i, **self._lfo_row_settings(i))
        if save:
            self.settings.setValue("lfo/slots", json.dumps(
                [self._lfo_row_settings(j) for j in range(LFO_SLOTS)]))

    def _lfo_all_off(self):
        for r in self.lfo_rows:
            r["on"].setChecked(False)

    def _lfo_depth_frac(self, i, x):
        """A CC on a slot's depth: 0 to 100 %, the sign as set."""
        sp = self.lfo_rows[i]["depth"]
        sign = -1 if sp.value() < 0 else 1
        sp.setValue(sign * round(100 * max(0.0, min(1.0, x))))

    def _lfo_get(self, ident):
        """A knob as 0..1 of its range, or None when there is nothing there."""
        w = self.knob_widgets.get(ident)
        if w is not None:
            lo, hi = w.minimum(), w.maximum()
            return (w.value() - lo) / (hi - lo) if hi > lo else None
        if ident == "shader_pick":
            files = self._shader_pool()
            n = len(files) + 1
            name = self.shader.currentText()
            idx = next((k + 1 for k, f in enumerate(files) if f.stem == name), 0)
            return (idx + 0.5) / n
        if ident == "burst_len":
            lo, hi = self.burst_len.minimum(), self.burst_len.maximum()
            return math.log(self.burst_len.value() / lo) / math.log(hi / lo)
        if ident in self.frag_combos:
            c = self.frag_combos[ident]
            return (c.currentIndex() + 0.5) / max(1, c.count())
        if ident == "rep_div":
            n = self.rep_div.count()
            return (self.rep_div.currentIndex() + 0.5) / n
        if ident.startswith("vst_"):
            return self.inserts.knob_value(int(ident[4:]) - 1)
        if ident.startswith("lfo_") and ident.endswith("_depth"):   # a macro on an LFO's depth
            try:
                return abs(self.lfo_rows[int(ident[4:-6]) - 1]["depth"].value()) / 100.0
            except (ValueError, IndexError):
                return None
        return None

    def _lfo_set(self, ident, x):
        act = self.actions.get(ident)
        if act is not None:
            act(x)

    def _lfo_tick(self):
        """The engine, 100 times a second, a scene morph in flight first
        (what it writes is the LFOs' new centre); the bars and the status a
        third of that."""
        now = time.monotonic()
        self._scene_tick(now)
        self.lfo.tick(now)
        self._lfo_ticks += 1
        if self._lfo_ticks % 3:
            return
        self._scene_status()
        on = []
        for i, (s, r) in enumerate(zip(self.lfo.slots, self.lfo_rows)):
            r["bar"].set_value(s.value if s.on else 0.0, s.on and s.out is not None)
            if s.on and s.dest:
                on.append(str(i + 1))
        self.lfo_status.setText("LFOs on: " + ", ".join(on) if on else "no LFO on")

    # ---------- macros ----------
    def _tab_macros(self):
        """Eight macros, every slot a row: a name, the knob, its value, the
        two capture buttons and an edit button; under them the
        destinations of the macro picked in the combo, eight rows of a
        knob with its lo and its hi. Everything is remembered."""
        self.macros = MacroEngine(MACRO_SLOTS, self._lfo_set, MACRO_ROWS)
        self.macros.load_json(self.settings.value("macros/slots", ""))
        self.macro_rows = []
        self._macro_save_timer = QTimer(self)      # a knob under an LFO moves 100 times a
        self._macro_save_timer.setSingleShot(True)  # second: the settings a little later
        self._macro_save_timer.setInterval(400)
        self._macro_save_timer.timeout.connect(self._macro_save_settings)
        grid = QGridLayout()
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        heads = ["", "name", "knob", "", "", "", "", ""]
        for c, head in enumerate(heads):
            lbl = QLabel(head)
            lbl.setProperty("role", "sub")
            grid.addWidget(lbl, 0, c)
        for i, mc in enumerate(self.macros.slots):
            r = i + 1
            num = QLabel(str(r))
            num.setProperty("role", "sub")
            name = QLineEdit(mc.name)
            name.setPlaceholderText(f"macro {r}")
            name.setToolTip("A name for the row and the status; the slot is the number")
            name.textEdited.connect(lambda t, i=i: self._macro_name_edited(i, t))
            knob = QSlider(Qt.Horizontal)
            knob.setRange(0, 1000)
            knob.setValue(int(round(1000 * mc.value)))
            knob.setMinimumWidth(120)
            knob.setToolTip(f"Macro {r}: every knob in its list follows, from its lo at the "
                            "left to its hi at the right. A MIDI knob in the Controls tab's "
                            "macros group, and a destination for an LFO.")
            knob.valueChanged.connect(lambda v, i=i: self._macro_moved(i, v))
            val = QLabel(f"{round(100 * mc.value)} %")
            val.setProperty("role", "sub")
            val.setFixedWidth(40)
            lo_btn = QPushButton("set lo")
            lo_btn.setFixedWidth(56)
            lo_btn.setToolTip(f"Every knob macro {r} points at takes the value it sits at "
                              "now as its lo: dial the quiet state by hand, press this")
            lo_btn.clicked.connect(lambda _c, i=i: self._macro_capture(i, "lo"))
            hi_btn = QPushButton("set hi")
            hi_btn.setFixedWidth(56)
            hi_btn.setToolTip(f"Every knob macro {r} points at takes the value it sits at "
                              "now as its hi: dial the peak by hand, press this")
            hi_btn.clicked.connect(lambda _c, i=i: self._macro_capture(i, "hi"))
            edit = QPushButton("edit")
            edit.setFixedWidth(44)
            edit.setToolTip(f"Macro {r}'s destinations in the list below")
            edit.clicked.connect(lambda _c, i=i: self.macro_pick.setCurrentIndex(i))
            info = QLabel("")
            info.setProperty("role", "sub")
            row = {"name": name, "knob": knob, "val": val, "lo": lo_btn, "hi": hi_btn,
                   "edit": edit, "info": info}
            self.macro_rows.append(row)
            cells = [num, name, knob, val, lo_btn, hi_btn, edit, info]
            for c, wdg in enumerate(cells):
                grid.addWidget(wdg, r, c)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(len(heads) - 1, 1)
        # the destinations of one macro at a time
        pick_row = QHBoxLayout()
        pick_row.setContentsMargins(8, 4, 8, 0)
        pick_lbl = QLabel("destinations of")
        pick_lbl.setProperty("role", "sub")
        self.macro_pick = QComboBox()
        for mc in self.macros.slots:
            self.macro_pick.addItem(mc.label)
        self.macro_pick.setToolTip("Which macro the list below edits")
        self.macro_pick.currentIndexChanged.connect(self._macro_show)
        self.macro_status = QLabel("")
        self.macro_status.setProperty("role", "sub")
        pick_row.addWidget(pick_lbl)
        pick_row.addWidget(self.macro_pick)
        pick_row.addSpacing(12)
        pick_row.addWidget(self.macro_status, stretch=1)
        dgrid = QGridLayout()
        dgrid.setContentsMargins(8, 4, 8, 8)
        dgrid.setHorizontalSpacing(6)
        dgrid.setVerticalSpacing(4)
        for c, head in enumerate(["", "knob", "lo %", "hi %"]):
            lbl = QLabel(head)
            lbl.setProperty("role", "sub")
            dgrid.addWidget(lbl, 0, c)
        self.macro_dest_rows = []
        for j in range(MACRO_ROWS):
            num = QLabel(str(j + 1))
            num.setProperty("role", "sub")
            dest = QComboBox()                # empty = no knob on this row
            for ident, label in MACRO_DESTS:
                dest.addItem(label, ident)
            dest.setProperty("clearable", True)
            dest.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            dest.setMinimumContentsLength(20)
            dest.setToolTip("A knob this macro moves: any knob of the Controls tab but the "
                            "macros themselves. Delete the name and press Enter to drop "
                            "the knob from the row.")
            searchable(dest)
            dest.setCurrentIndex(-1)
            lo = QSpinBox()
            lo.setRange(0, 100)
            lo.setToolTip("Where the knob sits with the macro at 0, as a percentage of the "
                          "knob's range; above hi, the knob runs the other way. Changing it "
                          "writes the knob at the macro's value now, so you hear it.")
            hi = QSpinBox()
            hi.setRange(0, 100)
            hi.setValue(100)
            hi.setToolTip("Where the knob sits with the macro at 100 %, as a percentage of "
                          "the knob's range")
            dest.currentIndexChanged.connect(lambda _v, j=j: self._macro_dest_edited(j, False))
            lo.valueChanged.connect(lambda _v, j=j: self._macro_dest_edited(j, True))
            hi.valueChanged.connect(lambda _v, j=j: self._macro_dest_edited(j, True))
            self.macro_dest_rows.append({"dest": dest, "lo": lo, "hi": hi})
            for c, wdg in enumerate([num, dest, lo, hi]):
                dgrid.addWidget(wdg, j + 1, c)
        dgrid.setColumnStretch(1, 1)
        hint = self._info("A macro is one knob over many: each destination runs from its lo to "
                      "its hi as the macro goes from 0 to 100 %, set by the numbers or by "
                      "ear with set lo and set hi (every destination takes where it sits "
                      "now). The macro writes when it moves; a hand turn, a scene or an "
                      "LFO can move a destination in between, the next move takes it "
                      "back. An LFO on a macro wiggles the whole list. Every macro is a "
                      "MIDI knob (Controls tab, macros group).")
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(grid)
        outer.addLayout(pick_row)
        outer.addLayout(dgrid)
        outer.addWidget(hint)
        outer.addStretch(1)
        page = QWidget()
        page.setLayout(outer)
        self._macro_refresh_rows()
        self._macro_show(0)
        return self._scrolling(page)

    def _macro_save_settings(self):
        self.settings.setValue("macros/slots", self.macros.to_json())

    def _macro_refresh_rows(self):
        for mc, r in zip(self.macros.slots, self.macro_rows):
            n = len(mc.active)
            r["info"].setText(f"{n} knob{'s' if n != 1 else ''}" if n else "no knob")

    def _macro_name_edited(self, i, text):
        self.macros.slots[i].name = text
        self.macro_pick.setItemText(i, self.macros.slots[i].label)
        self._macro_save_settings()

    def _macro_moved(self, i, v):
        """The knob of macro i: every destination written, the row's
        value, the status, the settings a little later."""
        self.macros.move(i, v / 1000.0)
        self.macro_rows[i]["val"].setText(f"{round(v / 10)} %")
        self.macro_status.setText(self.macros.summary(i))
        self._macro_save_timer.start()

    def _macro_set_quiet(self, i, x):
        """Macro i's knob to x without the fan-out (a scene load, which
        carries the destinations itself)."""
        knob = self.macro_rows[i]["knob"]
        knob.blockSignals(True)
        try:
            knob.setValue(int(round(1000 * max(0.0, min(1.0, x)))))
        finally:
            knob.blockSignals(False)
        self.macros.move(i, knob.value() / 1000.0, write=False)
        self.macro_rows[i]["val"].setText(f"{round(knob.value() / 10)} %")
        self._macro_save_timer.start()

    def _macro_capture(self, i, which):
        """Every destination's lo or hi from where its knob sits now."""
        n = self.macros.capture(i, which, self._scene_knob)
        if self.macro_pick.currentIndex() == i:
            self._macro_show(i)
        self._macro_save_settings()
        label = self.macros.slots[i].label
        self.log.appendPlainText(f"macro: {label} {which} set from {n} knob{'s' if n != 1 else ''}"
                                 if n else f"macro: {label} has no knob to set")

    def _macro_show(self, i):
        """Macro i's destinations into the editor rows, quietly."""
        for d, r in zip(self.macros.slots[i].dests, self.macro_dest_rows):
            widgets = (r["dest"], r["lo"], r["hi"])
            for w in widgets:
                w.blockSignals(True)
            try:
                r["dest"].setCurrentIndex(r["dest"].findData(d.ident))   # -1: no knob
                r["lo"].setValue(int(round(100 * d.lo)))
                r["hi"].setValue(int(round(100 * d.hi)))
            finally:
                for w in widgets:
                    w.blockSignals(False)

    def _macro_dest_edited(self, j, write):
        """Row j of the editor into the macro it shows; a range change
        (`write`) sets that knob at the macro's value now, so it is heard;
        picking a knob leaves it where it sits until the macro moves."""
        i = self.macro_pick.currentIndex()
        r = self.macro_dest_rows[j]
        ident = r["dest"].currentData() or ""
        self.macros.assign(i, j, ident, r["lo"].value() / 100.0, r["hi"].value() / 100.0)
        if ident and write:
            d = self.macros.slots[i].dests[j]
            self._lfo_set(ident, max(0.0, min(1.0, d.at(self.macros.slots[i].value))))
        self._macro_refresh_rows()
        self._macro_save_settings()

    # ---------- scenes ----------
    def _tab_scenes(self):
        """Eight scenes, every slot a row: a name, what it carries (the four
        groups), save, load, and when it was saved. Below, the morph: off,
        a load sets every knob at once; on, the knobs glide over a time in
        bars at the live tempo. Everything is remembered."""
        self.scenes = SceneBank(SCENE_SLOTS)
        self.scenes.load_json(self.settings.value("scenes/slots", ""))
        self.scene_morph = None                  # a Morph in flight, or None
        self.scene_rows = []
        grid = QGridLayout()
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        heads = ["", "name", "carries", "", "", "", "", "", ""]
        for c, head in enumerate(heads):
            lbl = QLabel(head)
            lbl.setProperty("role", "sub")
            grid.addWidget(lbl, 0, c)
        for i, sc in enumerate(self.scenes.slots):
            r = i + 1
            num = QLabel(str(r))
            num.setProperty("role", "sub")
            name = QLineEdit(sc.name)
            name.setPlaceholderText(f"scene {r}")
            name.setToolTip("A name for the row and the status; the slot is the number")
            name.textEdited.connect(lambda t, i=i: self._scene_name_edited(i, t))
            ticks = {}
            for g in SCENE_GROUPS:
                cb = QCheckBox(SCENE_GROUP_LABELS[g])
                cb.setChecked(sc.ticks.get(g, True))
                cb.setToolTip({
                    "audio": "the twelve racks and their volumes, the main volume, the sync "
                             "delay, the A/B pedal, the burst length, the VST knobs and the "
                             "inserts bypass",
                    "voices": "the drum voices in or out, the solos, the stems, the quantize "
                              "grid",
                    "lfo": "the eight LFO rows, on states included",
                    "video": "the source grade, every effect strength, the shader, its "
                             "opacity and blend, video-only, the picture mode and the "
                             "dissolve",
                    "mc": "the Model:Cycles tab: every track's knobs, menus and switches, "
                          "the delay and reverb, the retrig's settings (sent to the box as "
                          "they land)"}[g] + "; unticked, a load leaves it alone (a save "
                                              "keeps everything)")
                cb.toggled.connect(lambda on, i=i, g=g: self._scene_tick_changed(i, g, on))
                ticks[g] = cb
            save = QPushButton("save")
            save.setFixedWidth(48)
            save.setToolTip(f"The whole state as it is now into scene {r} (knobs an LFO "
                            "is on at their centre). A MIDI button in the Controls tab's "
                            "scenes group.")
            save.clicked.connect(lambda _c, i=i: self._scene_save(i))
            load = QPushButton("load")
            load.setFixedWidth(48)
            load.setToolTip(f"Scene {r} back: the ticked groups, the switches at once, the "
                            "knobs at once or gliding (morph below). A MIDI button in the "
                            "scenes group.")
            load.clicked.connect(lambda _c, i=i: self._scene_load(i))
            info = QLabel("")
            info.setProperty("role", "sub")
            row = {"name": name, "ticks": ticks, "save": save, "load": load, "info": info}
            self.scene_rows.append(row)
            cells = [num, name] + [ticks[g] for g in SCENE_GROUPS] + [save, load, info]
            for c, wdg in enumerate(cells):
                grid.addWidget(wdg, r, c)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(len(heads) - 1, 1)
        self.scene_morph_on = QCheckBox("morph")
        self.scene_morph_on.setChecked(self.settings.value("scenes/morph", False, type=bool))
        self.scene_morph_on.setToolTip("On, a load glides every knob that differs from where "
                                       "it sits to the scene's value over the time next to "
                                       "it; the switches (voices, stems, the shader, the LFO "
                                       "rows) flip at the start. Off, everything lands at "
                                       "once. A MIDI button in the scenes group.")
        self.scene_morph_on.toggled.connect(
            lambda on: self.settings.setValue("scenes/morph", bool(on)))
        self.scene_morph_bars = QComboBox()
        self.scene_morph_bars.addItems([name for name, _b in MORPH_BARS])
        b = self.settings.value("scenes/morph_bars", MORPH_DEFAULT)
        self.scene_morph_bars.setCurrentText(
            str(b) if self.scene_morph_bars.findText(str(b)) >= 0 else MORPH_DEFAULT)
        self.scene_morph_bars.setToolTip("How long the glide takes, in bars at the live tempo "
                                         "(a playing section's, Link's or the taps')")
        self.scene_morph_bars.currentTextChanged.connect(
            lambda t: self.settings.setValue("scenes/morph_bars", t))
        self.scene_status = QLabel("no scene loaded")
        self.scene_status.setProperty("role", "sub")
        hint = self._info("A scene is the live state: the racks, the voices, the LFOs, the "
                      "picture. Save keeps all of it; load applies the groups the row "
                      "ticks, so a scene with only video ticked is a look that leaves the "
                      "sound alone. Loading another scene mid-morph glides on from where "
                      "the knobs are. Each slot's save and load are MIDI buttons (Controls "
                      "tab, scenes group).")
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 8, 8)
        row.addWidget(self.scene_morph_on)
        row.addWidget(self.scene_morph_bars)
        row.addSpacing(12)
        row.addWidget(self.scene_status, stretch=1)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(grid)
        outer.addLayout(row)
        outer.addWidget(hint)
        outer.addStretch(1)
        page = QWidget()
        page.setLayout(outer)
        self._scene_refresh_rows()
        return self._scrolling(page)

    def _scene_refresh_rows(self):
        for sc, r in zip(self.scenes.slots, self.scene_rows):
            if sc.empty:
                text = "empty"
            else:
                text = f"saved {sc.saved_at}" if sc.saved_at else "saved"
                if self.scenes.current == sc.i:
                    text = "loaded · " + text
            r["info"].setText(text)
            r["load"].setEnabled(not sc.empty)

    def _scene_save_settings(self):
        self.settings.setValue("scenes/slots", self.scenes.to_json())

    def _scene_name_edited(self, i, text):
        self.scenes.slots[i].name = text
        self._scene_save_settings()
        self._scene_status()

    def _scene_tick_changed(self, i, g, on):
        self.scenes.slots[i].ticks[g] = bool(on)
        self._scene_save_settings()

    # what a scene holds, group by group
    def _scene_knob_ids(self, g):
        if g == "audio":
            return ([f"afx_{ch}_{k}" for ch in ALL_CHANNELS for k, *_ in AUDIO_FX]
                    + [f"avol_{ch}" for ch in ALL_CHANNELS]
                    + ["volume", "sync", "ab", "burst_len"]
                    + list(self.frag_combos) + ["frag_motion", "frag_pitch", "frag_speed",
                                                "frag_repeats", "frag_shape", "frag_a",
                                                "frag_d", "frag_s", "frag_r",
                                                "freeze_size", "freeze_smear", "freeze_pitch",
                                                "freeze_a", "freeze_r"]
                    + [i for i in self.fxset if i not in self.frag_combos]
                    + [f"vst_{n}" for n in range(1, VST_KNOBS + 1)]
                    + [f"macro_{n}" for n in range(1, MACRO_SLOTS + 1)])
        if g == "video":
            return ([f"grade_{k}" for k, *_ in VIDEO_GRADE]
                    + [f"fx_{k}" for k, *_ in VIDEO_EFFECTS] + ["shader_mix"])
        if g == "mc":
            return ([self._mc_ident(t, key) for (t, key), w in self.mc.items()
                     if not isinstance(w, QCheckBox)] + list(MC_RETRIG_KNOBS))
        return []

    def _scene_knob(self, ident):
        """A knob as the LFOs read it, but at its centre when an LFO is on
        it: what you set, never the wiggle."""
        for s in self.lfo.slots:
            if s.on and s.dest == ident and s.base is not None:
                return s.base
        return self._lfo_get(ident)

    def _scene_switches(self, g):
        if g == "audio":
            return {"inserts": self.vst_bypass.isChecked()}
        if g == "voices":
            sw = {f"voice_{v}": not cb.isChecked() for v, cb in self.perf_voices.items()}
            sw.update({f"solo_{v}": b.isChecked() for v, b in self.perf_solos.items()})
            sw.update({f"stem_{st}": not cb.isChecked() for st, cb in self.perf_stems.items()})
            sw["quant"] = self.perf_quant.currentText()
            return sw
        if g == "lfo":
            return {"lfo_rows": [self._lfo_row_settings(j) for j in range(LFO_SLOTS)]}
        if g == "mc":
            return {ident: cb.isChecked() for ident, cb in self.mc_toggles.items()}
        if g == "video":
            return {"shader": self.shader.currentText(),
                    "shader_blend": self.shader_blend.currentText(),
                    "video_plain": self.video_plain.isChecked(),
                    "picture": self.perf_picture.currentText(),
                    "dissolve": self.perf_dissolve.currentText()}
        return {}

    def _scene_capture(self):
        """The whole state now: {group: {"knobs": {id: 0..1}, "switches": {}}}."""
        state = {}
        for g in SCENE_GROUPS:
            knobs = {}
            for ident in self._scene_knob_ids(g):
                x = self._scene_knob(ident)
                if x is not None:
                    knobs[ident] = round(float(x), 6)
            state[g] = {"knobs": knobs, "switches": self._scene_switches(g)}
        return state

    def _scene_apply_switch(self, ident, val):
        """One switch of a scene back onto its widget."""
        if ident == "inserts":
            self.vst_bypass.setChecked(bool(val))
        elif ident.startswith("voice_") and ident[6:] in self.perf_voices:
            self.perf_voices[ident[6:]].setChecked(not bool(val))   # val: in; the M: out
        elif ident.startswith("solo_") and ident[5:] in self.perf_solos:
            self.perf_solos[ident[5:]].setChecked(bool(val))
        elif ident.startswith("stem_") and ident[5:] in self.perf_stems:
            self.perf_stems[ident[5:]].setChecked(not bool(val))   # val: in; the M: out
        elif ident == "quant":
            if self.perf_quant.findText(str(val)) >= 0:
                self.perf_quant.setCurrentText(str(val))
        elif ident == "lfo_rows" and isinstance(val, list):
            self._scene_apply_lfo_rows(val)
        elif ident == "shader":
            name = str(val)
            if name == "off":
                self.select_shader_key(0)
            elif self.shader.findText(name) >= 0:
                self.shader.setCurrentText(name)
                self._shader_user_off = False
        elif ident == "shader_blend":
            if self.shader_blend.findText(str(val)) >= 0:
                self.shader_blend.setCurrentText(str(val))
        elif ident == "video_plain":
            self.video_plain.setChecked(bool(val))
        elif ident == "picture":
            if self.perf_picture.findText(str(val)) >= 0:
                self.perf_picture.setCurrentText(str(val))
        elif ident == "dissolve":
            if self.perf_dissolve.findText(str(val)) >= 0:
                self.perf_dissolve.setCurrentText(str(val))
        elif ident in self.mc_toggles:
            self.mc_toggles[ident].setChecked(bool(val))

    def _scene_apply_lfo_rows(self, rows):
        """The LFO rows of a scene onto the tab: the widgets quietly, then
        each row into the engine (a slot going off writes its centre back,
        a new destination too)."""
        for i, st in enumerate(rows[:LFO_SLOTS]):
            if not isinstance(st, dict):
                continue
            r = self.lfo_rows[i]
            widgets = [r[k] for k in ("on", "dest", "shape", "rate", "depth", "phase", "pol",
                                      "mode")]
            for w in widgets:
                w.blockSignals(True)
            try:
                j = r["dest"].findData(st.get("dest", ""))
                r["dest"].setCurrentIndex(max(0, j))
                if st.get("shape") in LFO_SHAPES:
                    r["shape"].setCurrentText(st["shape"])
                if r["rate"].findText(str(st.get("rate", ""))) >= 0:
                    r["rate"].setCurrentText(str(st["rate"]))
                r["depth"].setValue(int(round(100 * float(st.get("depth", 0.5)))))
                r["phase"].setValue(int(round(360 * float(st.get("phase", 0.0)))) % 360)
                r["pol"].setCurrentIndex(1 if st.get("unipolar") else 0)
                if st.get("mode") in LFO_MODES:
                    r["mode"].setCurrentText(st["mode"])
                r["on"].setChecked(bool(st.get("on", False)))
            finally:
                for w in widgets:
                    w.blockSignals(False)
            self._lfo_push(i)

    def _scene_save(self, i):
        """The state now into slot i (the name stays)."""
        sc = self.scenes.slots[i]
        sc.save(self._scene_capture())
        self.scenes.current = i
        self._scene_save_settings()
        self._scene_refresh_rows()
        self._scene_status()
        self.log.appendPlainText(f"scene: {sc.label} saved")

    def _scene_morph_secs(self):
        bars = dict(MORPH_BARS).get(self.scene_morph_bars.currentText(), 1.0)
        return bars * 4 * 60.0 / max(1e-6, self.tempo.bpm)

    def _scene_load(self, i):
        """Slot i back: the switches of its ticked groups at once, the knobs
        at once or as a morph from where they sit."""
        sc = self.scenes.slots[i]
        if sc.empty:
            self.log.appendPlainText(f"scene: {sc.label} is empty")
            return
        switches, knobs = sc.plan()
        for ident, val in switches.items():
            self._scene_apply_switch(ident, val)
        self.scenes.current = i
        if self.scene_morph_on.isChecked() and knobs:
            start = {ident: self._scene_knob(ident) for ident in knobs}
            self.scene_morph = Morph(time.monotonic(), self._scene_morph_secs(), start, knobs)
            if not self.scene_morph.pairs:
                self.scene_morph = None
        else:
            self.scene_morph = None
            for ident, x in knobs.items():
                self._scene_set(ident, x)
        self._scene_refresh_rows()
        self._scene_status()
        self.log.appendPlainText(f"scene: {sc.label} loaded"
                                 + (f", morph {self.scene_morph_bars.currentText()}"
                                    if self.scene_morph else ""))

    def _scene_tick(self, now):
        m = self.scene_morph
        if m is None:
            return
        for ident, x in m.step(now).items():
            self._scene_set(ident, x)
        if m.done:
            self.scene_morph = None

    def _scene_set(self, ident, x):
        """A knob of a scene onto its widget; a macro's own knob without
        its fan-out, the scene carrying the destinations itself."""
        if ident.startswith("macro_") and ident in self.knob_widgets:
            self._macro_set_quiet(int(ident[6:]) - 1, x)
        else:
            self._lfo_set(ident, x)

    def _scene_status(self):
        cur = self.scenes.current
        if cur is None:
            text = "no scene loaded"
        else:
            text = self.scenes.slots[cur].label
            if self.scene_morph is not None:
                text += (f" · morph {self.scene_morph_bars.currentText()}, "
                         f"{round(100 * self.scene_morph.frac)} %")
        if self.scene_status.text() != text:
            self.scene_status.setText(text)

    def _tab_video_fx(self):
        """The pool: which video effects and which shaders are in the show at
        all, for whoever cannot stand one of them. An unticked effect counts
        as strength 0 whatever its knob in the Video tab says (the knob greys
        out) and Randomize leaves it alone; an unticked shader never comes up
        by random, Space, the digit keys or the shader knob, only by name in
        the Video tab's combo. Every tick is a MIDI button (Controls tab,
        group 'video fx: the pool'), and the ticks are remembered."""
        try:
            pool = json.loads(self.settings.value("videofx/pool", "") or "{}")
        except ValueError:
            pool = {}
        fx_on, sh_off = set(pool.get("effects_on", [])), set(pool.get("shaders_off", []))
        form = QFormLayout()
        self.vfx_on = {}
        for key, label, _d, tip in VIDEO_EFFECTS:
            cb = QCheckBox(label)
            cb.setProperty("role", "pool")           # big: ticked by a finger on stage
            cb.setToolTip(tip)
            cb.setChecked(key in fx_on)              # all off until ticked
            cb.toggled.connect(lambda on, key=key: self._vfx_on_changed(key, on))
            self.vfx_on[key] = cb
        form.addRow("Effects", self._pool_box(self.vfx_on))
        self.shader_on = {}
        for p in shader_files():
            cb = QCheckBox(p.stem)
            cb.setProperty("role", "pool")
            cb.setToolTip(f"{p.name}: in the pool the random choice, Space, the digit "
                          "keys and the shader knob draw from")
            cb.setChecked(p.stem not in sh_off)
            cb.toggled.connect(lambda on, name=p.stem: self._shader_on_changed(name, on))
            self.shader_on[p.stem] = cb
        form.addRow("Shaders", self._pool_box(self.shader_on))
        hint = self._info("what is unticked is out of the show: an effect's strength counts "
                      "as 0 and its knob greys out in the Video tab; a shader is left "
                      "out of random, Space, the digit keys and the shader knob, and "
                      "only comes up when picked by name. With nothing playing, a "
                      "ticked effect runs on the clip at its strength, on the tap "
                      "tempo's clock. Remembered; every tick is also a MIDI button "
                      "in the Controls tab")
        form.addRow("", hint)
        for key, cb in self.vfx_on.items():
            self.vfx[key].setEnabled(cb.isChecked())
        return self._scrolling(self._wrap(form))     # the big ticks may outgrow the pane

    def _pool_box(self, boxes):
        """A grid of tick boxes, seven per column, with all / none below."""
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(4)
        for i, cb in enumerate(boxes.values()):
            grid.addWidget(cb, i % 7, i // 7)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for text, on in (("all", True), ("none", False)):
            b = QPushButton(text)
            b.setToolTip("Every box ticked: the whole pool in the show" if on
                         else "Every box unticked: nothing of this kind in the show")
            b.setObjectName("small")
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            b.clicked.connect(lambda _c=False, on=on: [cb.setChecked(on) for cb in boxes.values()])
            row.addWidget(b)
        row.addStretch(1)
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addLayout(grid)
        box.addLayout(row)
        w = QWidget()
        w.setLayout(box)
        return w

    def _vfx_on_changed(self, key, on):
        self.vfx[key].setEnabled(on)
        self._vfx_save()

    def _shader_on_changed(self, name, on):
        """A shader's tick (the tab or its MIDI pad) shows on screen: ticking
        one while the layer is off switches the layer on, in random mode over
        the pool, so with that one tick alone it is the one that shows;
        unticking the one on screen (or the one the combo names) draws
        another from the pool, and the last tick to go takes the layer off."""
        self._vfx_save()
        if not hasattr(self, "backdrop"):
            return
        choice = self.shader.currentText()
        cur = getattr(self.backdrop, "shader_path", None)
        if on:
            if choice == "off":
                self._shader_user_off = False
                self.shader.setCurrentText("random")   # draws from the pool
                self.log.appendPlainText(f"video: shader layer on, {name} joins the pool")
            elif choice == "random" and (cur is None or len(self._shader_pool()) == 1):
                self._pick_section_shader()             # nothing was showing
            return
        if not self._shader_pool():
            if choice in ("random", name):
                self._shader_user_off = True
                self.shader.setCurrentText("off")
                self.log.appendPlainText("video: the shader pool is empty, layer off")
            return
        if choice == name:                              # named on the combo: to the pool
            self.shader.setCurrentText("random")
        if (cur is not None and cur.stem == name) or choice == name:
            self._pick_section_shader()                 # the one on screen left the pool

    def _vfx_save(self):
        self.settings.setValue("videofx/pool", json.dumps({
            "effects_on": [k for k, cb in self.vfx_on.items() if cb.isChecked()],
            "shaders_off": [n for n, cb in self.shader_on.items() if not cb.isChecked()]}))

    # ---------- the clip by hand (MIDI) ----------
    def _show_clip(self, path):
        """Put `path` on screen now; the automatic switching starts its count
        from here. Nothing while a video input is on."""
        if self.backdrop.set_video(path, random_start=self.video_jump.isChecked()) is None:
            return
        self._live_switch_t = time.monotonic()
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self._last_switch_ms = self._playhead_ms()
        self.log.appendPlainText(f"video: {Path(path).name}")

    def _video_step(self, delta):
        """The previous / next ticked video, in the Videos tab's order."""
        pool = self._song_videos()
        if not pool:
            return
        cur = self.backdrop.current
        i = pool.index(cur) if cur in pool else -1
        self._show_clip(pool[(i + delta) % len(pool)])

    def _video_random(self):
        """A random ticked video, never the same twice in a row and none
        again before every other one has had its turn: the pool is shuffled
        into a deck once, dealt from the top, and shuffled again when it
        runs out or when the ticks change."""
        pool = self._song_videos()
        if not pool:
            return
        if not self._video_deck or set(self._video_deck_pool) != set(pool):
            deck = list(pool)
            random.shuffle(deck)
            if len(deck) > 1 and deck[0] == self.backdrop.current:
                deck.append(deck.pop(0))           # not the one on screen first
            self._video_deck, self._video_deck_pool = deck, list(pool)
        self._show_clip(self._video_deck.pop(0))

    def _shader_pool(self):
        """The shader files ticked in the Video FX tab, in list order (all of
        them before that tab exists)."""
        on = getattr(self, "shader_on", {})
        return [p for p in shader_files() if p.stem not in on or on[p.stem].isChecked()]

    def _randomize_video(self):
        self._shuffle(*[sp for k, sp in self.vfx.items() if self.vfx_on[k].isChecked()],
                      self.shader_blend, self.reverse_len)
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
            files = self._shader_pool()
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
        files = self._shader_pool()
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
        files = self._shader_pool()
        if not files:                          # an empty pool: no layer
            self.backdrop.set_shader(None)
            return
        cur = getattr(self.backdrop, "shader_path", None)
        pool = [p for p in files if p != cur] or files
        self.backdrop.set_shader(random.choice(pool))

    def toggle_fullscreen(self):
        """Fullscreen on/off. Own flag rather than isFullScreen(): on Windows
        the window is kept a pixel taller than the screen while fullscreen
        (see _fullscreen_size), and Qt then no longer reports the state."""
        if self._fullscreen:
            self._fullscreen = False
            self._fullscreen_size(0)    # exact again, and once Qt has seen
            QApplication.processEvents()  # that it is fullscreen in its own
            self.showNormal()           # eyes again, it restores the frame
            if self._was_maximized:     # back to the maximized window it came from
                self.showMaximized()
            self._show_ui()             # back to the desktop: interface back
            self._arm_hide()
        else:
            self._fullscreen = True
            self._was_maximized = self.isMaximized()
            self.showFullScreen()
            self._fullscreen_size(1)
            self._hide_ui(force=True)   # straight to the picture

    def _fullscreen_size(self, extra):
        """Windows: a window exactly the size of its screen is handed to the
        display directly (the shell reports a D3D fullscreen application)
        and nothing else is composited over it any more: a combo's list
        opened out of sight behind the picture, working only by keyboard.
        One extra pixel of height keeps the window an ordinary one that
        happens to cover the screen (the taskbar still steps aside). Qt
        drops its fullscreen state when the size differs, so extra=0 puts
        the exact size back before showNormal(); Qt notices on the next
        event round and then restores the frame and the old geometry as
        usual. Nothing to do elsewhere."""
        if sys.platform != "win32":
            return
        import ctypes
        import ctypes.wintypes as wt

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wt.DWORD), ("rcMonitor", wt.RECT),
                        ("rcWork", wt.RECT), ("dwFlags", wt.DWORD)]
        u = ctypes.windll.user32
        h = wt.HWND(int(self.winId()))
        mon = u.MonitorFromWindow(h, 2)                     # MONITOR_DEFAULTTONEAREST
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if not (mon and u.GetMonitorInfoW(mon, ctypes.byref(mi))):
            return
        r = mi.rcMonitor
        u.SetWindowPos(h, None, r.left, r.top, r.right - r.left,
                       r.bottom - r.top + extra, 0x0004 | 0x0010)   # NOZORDER | NOACTIVATE
        # Windows 11 rounds the corners of an ordinary window, which the
        # fullscreen one now is: square them (DWMWA_WINDOW_CORNER_PREFERENCE,
        # DONOTROUND), back to the default on the way out; Windows 10 has no
        # such attribute and the call just fails
        pref = ctypes.c_int(1 if extra else 0)
        try:
            ctypes.windll.dwmapi.DwmSetWindowAttribute(h, 33, ctypes.byref(pref), 4)
        except OSError:
            pass

    def _live_io_rows(self, form):
        """The live I/O, below the song parameters in the Settings tab: the app's audio output
        (the stream every sound goes through), the live input, the video
        input and the audio analysis."""
        self.through_on = QCheckBox("live input: plays through its rack to the "
                                    "output (A)")
        self.through_on.setToolTip(
            "The live path of the show: the chosen input (the VHS grabber) "
            "joins the app's audio stream and reaches the output through the "
            "live rack (Audio FX), alongside the section engine and songs. "
            "Two 256-frame buffers make about 11 ms plus the devices' own "
            "latency. With audio analysis on, the analysis listens to this "
            "input. Key A toggles it; the stream reopens with or without the "
            "input, which stops a playing song.")
        self.through_on.toggled.connect(self._toggle_through)
        self.through_in = QComboBox()
        self.through_in.setToolTip("Hardware input: the grabber's audio")
        self.through_out = QComboBox()
        self.through_out.setToolTip(
            "The app's audio output: everything it plays (songs, the section "
            "engine, the live input) goes through this device and the racks. "
            "PortAudio devices: on Linux the (hw:x,y) entries are the cards "
            "themselves, exclusive and lowest latency, unavailable while "
            "PipeWire or another app plays through them; 'pipewire' and "
            "'default' share the desktop's routing at about 11 ms. Windows "
            "lists the WASAPI devices. When it cannot be opened, songs fall "
            "back to Qt's player, without racks.")
        self.through_block = QComboBox()
        self.through_block.addItems(["64", "128", "256", "480", "512", "1024", "2048"])
        self.through_block.setToolTip(
            "Frames per block, the usual driver sizes: at 48 kHz 64 = 1.3 ms, 128 = "
            "2.7 ms, 256 = 5.3 ms, 512 = 10.7 ms, 1024 = 21 ms, 2048 = 43 ms each "
            "way. 480 is WASAPI shared mode's 10 ms period. Python processes one "
            "block per round, so 64 and 128 want ASIO or an exclusive device and "
            "a quiet machine; raise the size if xruns appear. On ASIO match the "
            "driver panel's buffer.")
        self.through_sr = QComboBox()
        self.through_sr.addItems(["44100", "48000", "88200", "96000"])
        self.through_sr.setCurrentText("48000")
        self.through_sr.setToolTip(
            "Sample rate of the stream, in Hz. 48 kHz is the app's own: songs "
            "and stems are rendered at it and play without resampling, and the "
            "block sizes' times above are at it. At another rate the sections, "
            "songs and drum samples are resampled on their way in, and a block "
            "lasts proportionally less (256 frames at 96 kHz = 2.7 ms). Pick the "
            "rate the interface runs at (an ASIO driver's panel, or Windows' "
            "device format in shared mode); a device that refuses the rate falls "
            "back to its own, and the status line says which is in use. Remembered.")
        self.through_ring = QComboBox()
        self.through_ring.addItem("ring off", 0)
        for n in (1, 2, 3, 4, 5, 6, 8):
            self.through_ring.addItem(f"ring {n}", n)
        self.through_ring.setToolTip(
            "Python off the driver's thread: with a ring of N, the stream's callback "
            "is C (the rtmixer package) and only copies between the device and two "
            "ring buffers, while a thread of ours renders N blocks ahead into one and "
            "reads the input from the other. A wait for the engine's interpreter "
            "lock then costs nothing until it outlasts the whole lead: a margin on "
            "top of the engine running in its own process, which is what cured the "
            "crackle at 128; the price is N blocks of added output latency (2.7 ms "
            "per block at 128 frames, 48 kHz), which the Link bars and the MIDI "
            "out's lead already account for. Off = the usual way, Python in the "
            "driver's callback. The lead must cover what the driver "
            "takes per call: on ASIO that is one block, so start with 2; WASAPI "
            "shared takes 10 ms at a time, 4 blocks at 128 (ring 5) or 2 at 256 "
            "(ring 3), and a lead short of that underruns every period. Needs "
            "pip install rtmixer; without it the stream opens the usual way and the "
            "status line says so. Remembered.")
        self.through_excl = QCheckBox("exclusive")
        self.through_excl.setChecked(True)
        self.through_excl.setToolTip(
            "Windows: open the devices in WASAPI exclusive mode, straight to the "
            "driver with no system mixer, the lowest latency Windows offers "
            "without ASIO; other apps lose the device while the stream is open. "
            "Falls back to shared mode (Windows converts) when a device refuses. "
            "No effect on Linux or macOS.")
        self.through_excl.setVisible(sys.platform == "win32")
        self.through_excl.toggled.connect(self._io_changed)
        self.through_api = QComboBox()
        self.through_api.setToolTip(
            "The audio driver model (PortAudio host API). ASIO, when the "
            "interface has a driver, is the one for the stage: one duplex "
            "stream at the buffer set in the driver's control panel, the "
            "lowest latency there is; input and output must then be the same "
            "device. WASAPI is Windows' own path. The device lists follow "
            "this choice.")
        self.through_api.setVisible(sys.platform == "win32")
        self.through_api.currentIndexChanged.connect(self._api_changed)
        self.through_sync = QSpinBox(minimum=0, maximum=2000, singleStep=5,
                                     suffix=" ms")
        self.through_sync.setValue(0)
        self.through_sync.setToolTip(
            "Lip sync: hold the sound back by this much so it meets the "
            "picture, which arrives late through the grabber, the decoder "
            "and the display. Adjust while watching a tape: mouths, cuts, "
            "hits. Changes live without a click. Typical: 60 to 150 ms.")
        self.through_sync.valueChanged.connect(self._through_sync_changed)
        self.through_status = QLabel("")
        self.through_status.setProperty("role", "sub")
        for combo in (self.through_out, self.through_in, self.through_block, self.through_sr,
                      self.through_ring):
            combo.currentIndexChanged.connect(self._io_changed)
        self.video_in = QComboBox()
        self.video_in.setToolTip(
            "Live video in: a capture device (a webcam, or a USB composite "
            "grabber with a VHS deck on it; V4L2 on Linux, DirectShow on "
            "Windows) replaces the clips as "
            "the backdrop. Everything else runs on it as usual: the effects, "
            "the shader layer, and rewinds jog the live picture "
            "through the last seconds captured. Clip changes are held off "
            "until this is back to off.")
        self.video_in.currentIndexChanged.connect(self._video_in_changed)
        self.video_in_mode = QComboBox()
        self.video_in_mode.setToolTip(
            "Capture mode of the device: frame size and pixel format or codec, "
            "as the driver offers them (largest first). 'default' leaves the "
            "choice to the driver. A VHS grabber wants its native 720x576 "
            "(PAL) or 720x480 (NTSC) raw mode; a webcam's mjpeg modes are "
            "cheaper to decode than h264 at the same size. Remembered.")
        self.video_in_mode.addItem("default", None)
        self.video_in_mode.currentIndexChanged.connect(self._video_mode_changed)
        self.video_in2 = QComboBox()
        self.video_in2.setToolTip(
            "A second capture device, a webcam next to the VHS grabber. Both "
            "are kept open at once (two USB devices, two decoder threads); "
            "W puts this one on the B side of the A/B pedal, where the "
            "section's take frames go otherwise, so the pedal dissolves the "
            "tape into the webcam. Must be another device than the first. Two devices on one "
            "USB 2.0 hub share its bandwidth: a grabber in its raw yuyv mode "
            "takes nearly all of it and the webcam then fails to open (I/O "
            "error in the log). Put them on different ports of the machine, "
            "or run the grabber in its 720x576 mjpeg mode.")
        self.video_in2.currentIndexChanged.connect(
            lambda *_: self._video_in_changed(second=True))
        self.video_in2_mode = QComboBox()
        self.video_in2_mode.setToolTip("Capture mode of the second device, as for "
                                       "the first. Remembered.")
        self.video_in2_mode.addItem("default", None)
        self.video_in2_mode.currentIndexChanged.connect(
            lambda *_: self._video_mode_changed(second=True))
        self.video_second = QCheckBox("second on B (W)")
        self.video_second.setToolTip(
            "Ticked: the second device's picture is the B side of the A/B "
            "pedal, in place of the section's take frames (a burst still "
            "shows its frame); the pedal at 0 is the tape, at 100 the webcam, "
            "with the Picture row's dissolve, and it works with no section "
            "playing too. The tape's sound is left whole by the pedal while "
            "this is on (the webcam is silent); the section still comes in. "
            "Off: B is the take frames again. Also a MIDI button "
            "in the picture group. Takes record the first input either way.")
        self.video_second.toggled.connect(self._toggle_video_second)
        self.live_on = QCheckBox("live: the audio drives the visuals "
                                 "instead of a song (L)")
        self.live_on.setToolTip(
            "VJ mode. Loudness comes from the chosen input (a mic, a line "
            "in, or on PipeWire/Pulse a monitor of what the machine plays) "
            "and the beat from tap tempo: press T on the beat four times, "
            "starting on the one. D restarts the bar on the current tempo. "
            "Every 8 bars count as a section for the look, shader and clip "
            "changes. Turning a song on switches live off.")
        self.live_on.toggled.connect(self._toggle_live)
        self.live_device = QComboBox()
        self.live_device.setToolTip("Capture device; the list follows what "
                                    "the system offers")
        self.live_level = QProgressBar()
        self.live_level.setRange(0, 100)
        self.live_level.setTextVisible(False)
        self.live_level.setFixedHeight(10)
        self.live_level.setToolTip("Normalised loudness, what the effects see")
        form.addRow("Audio output", self._row([("driver", self.through_api),
                                              ("device", self.through_out),
                                              ("block", self.through_block),
                                              ("rate", self.through_sr),
                                              ("", self.through_excl),
                                              ("", self.through_ring)]))
        form.addRow("", self.through_status)
        # the scheduling DAWs give themselves, each a tick, remembered,
        # applied live to the engine process (gacha_audioproc: its
        # set_process_priority and gc, gacha_audio.thread_pro_audio)
        self.audio_pro = QCheckBox("Pro Audio (MMCSS) on the audio thread")
        self.audio_pro.setToolTip(
            "Windows: the engine's audio thread (the driver's, or with a ring on, "
            "its render thread) joins the MMCSS \"Pro Audio\" class, the scheduling "
            "DAWs give their audio threads, so the desktop's work waits for it and "
            "not the other way round. Takes effect on the next block. Remembered.")
        self.audio_high = QCheckBox("high process priority")
        self.audio_high.setToolTip(
            "Windows: the audio engine's process in HIGH_PRIORITY_CLASS, as DAWs "
            "run: the scheduler takes its threads before the desktop's, this "
            "interface included. At once. Remembered.")
        self.audio_gc = QCheckBox("GC off")
        self.audio_gc.setToolTip(
            "Python's garbage collector off in the audio engine's process: a "
            "collection there holds its interpreter lock while it walks the heap, "
            "the audio thread waiting. Memory of reference cycles is then never "
            "reclaimed in the engine, a little growth over a show. At once. "
            "Remembered.")
        for cb in (self.audio_pro, self.audio_high):
            cb.setVisible(sys.platform == "win32")
        for cb in (self.audio_pro, self.audio_high, self.audio_gc):
            cb.toggled.connect(lambda _v: self._priority_changed())
        form.addRow("Priority", self._row([("", self.audio_pro), ("", self.audio_high),
                                           ("", self.audio_gc)]))
        self.rt_btn = QPushButton("test round-trip")
        self.rt_btn.setToolTip(
            "Measure the real in-to-out latency: plug a cable from an output of "
            "the interface into the live input, turn the input level up, stop "
            "any song, then press. A short 1 kHz burst goes out and the input "
            "is watched for it to come back; the result is the round trip "
            "through the device, its buffers and one processing block, what "
            "a guitar through the app experiences. The live input is muted "
            "during the test so the cable cannot feed back.")
        self.rt_btn.clicked.connect(self._ping_start)
        # a small button: hugs its text instead of stretching across the row
        self.rt_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.rt_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        self.rt_lbl = QLabel("")
        self.rt_lbl.setProperty("role", "sub")
        form.addRow("Audio input", self._row([("", self.through_on), ("", self.rt_btn),
                                              ("", self.rt_lbl)]))
        form.addRow("", self._row([("device", self.through_in),
                                   ("sync", self.through_sync)]))
        form.addRow("Video input", self._row([("device", self.video_in),
                                              ("mode", self.video_in_mode)]))
        form.addRow("", self._row([("second", self.video_in2),
                                   ("mode", self.video_in2_mode),
                                   ("", self.video_second)]))
        self.keep_takes = QCheckBox("keep the takes: written to videos/takes_<date>/ "
                                    "when the app quits")
        self.keep_takes.setToolTip(
            "The takes recorded with R stay in RAM during the show (encoding "
            "them to MJPEG while playing glitches the sound). On quit, with this "
            "ticked, every take not yet on disk is written by ffmpeg as a mov "
            "with its sound, a progress window showing; unticked they are "
            "forgotten. The Perform tab's save takes button writes them earlier, "
            "between pieces. Remembered.")
        self.keep_takes.setChecked(self.settings.value("takes/keep", True, type=bool))
        self.keep_takes.toggled.connect(lambda on: self.settings.setValue("takes/keep", on))
        form.addRow("Takes", self.keep_takes)
        form.addRow("Audio analysis", self.live_on)
        form.addRow("", self._row([("device", self.live_device),
                                   ("level", self.live_level)]))
        self.video_stats = QCheckBox("stats on screen: fps, cpu, memory, audio")
        self.video_stats.setToolTip(
            "A small readout in the top right corner of the picture, kept "
            "with the interface hidden: paint rate and GPU time of the "
            "backdrop, the video's frame rate, the machine's CPU and this "
            "app's share, memory used by the app (the takes live there) and "
            "left on the machine, and with the audio stream open its DSP load "
            "and xruns.")
        self.video_stats.toggled.connect(self._toggle_stats)
        form.addRow("Stats", self.video_stats)

    def _tab_general(self):
        form = QFormLayout()
        self.duration = QSpinBox(minimum=10, maximum=600, value=180,
                                 suffix=" s")
        self.duration.setToolTip("Songs only: the length Generate aims for; sections are "
                                 "added until it is reached, so it lands within a "
                                 "section of it")
        form.addRow("Song duration", self.duration)

        self.seed_random = QCheckBox("random")
        self.seed_random.setChecked(True)
        self.seed_random.setToolTip("Songs only: every Generate rolls a new seed; untick "
                                    "to set one and get the same song again from the "
                                    "same samples and settings")
        self.seed = QSpinBox(minimum=0, maximum=999999, value=42)
        self.seed.setToolTip("The seed of the song's every random choice: the same "
                             "number with the same samples and settings renders the "
                             "same song, so a good one can be made again after a "
                             "tweak.")
        self.seed.setEnabled(False)
        self.seed_random.toggled.connect(lambda on: self.seed.setEnabled(not on))
        form.addRow("Seed", self._pair(self.seed, self.seed_random))

        self.all_samples = QCheckBox("use all")
        self.all_samples.setChecked(True)
        self.all_samples.setToolTip("Songs only: every ticked sample goes into the "
                                    "render; untick to draw a number of them at random "
                                    "per song, for variety between renders")
        self.num_samples = QSpinBox(minimum=1, maximum=9999, value=10)
        self.num_samples.setToolTip("How many of the ticked samples a song draws, at "
                                    "random, when 'use all' is off")
        self.num_samples.setEnabled(False)
        self.all_samples.toggled.connect(
            lambda on: self.num_samples.setEnabled(not on))
        form.addRow("Samples", self._pair(self.num_samples, self.all_samples))
        self._live_clock_rows(form)             # the BPM row, under the song rows
        gap = QWidget()
        gap.setFixedHeight(10)                  # a breath before the live I/O rows
        form.addRow("", gap)
        self._live_io_rows(form)
        # every remembered value forgotten, after a yes: the app restarts
        # with its defaults (the files, takes, renders and plugins stay)
        reset_btn = QPushButton("forget every saved setting...")
        reset_btn.setObjectName("small")
        reset_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        reset_btn.setToolTip("Every value Gacha remembers between launches (ports, the MIDI "
                             "map, the knobs, the pools, the scenes, the macros, the M:C "
                             "tab, the window) is forgotten and the app restarts with its "
                             "defaults; asks first. Samples, videos, takes, renders and "
                             "plugins on disk are untouched.")
        reset_btn.clicked.connect(self._reset_settings)
        # the same without the hardware and without a restart: the values,
        # the LFOs, the macros, the scenes, the M:C tab and the racks go
        # live, the sound card, the MIDI map and ports, the Erae, Link, the
        # video inputs and the pools stay
        values_btn = QPushButton("forget the values only...")
        values_btn.setObjectName("small")
        values_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        values_btn.setToolTip("A clean slate without touching the hardware: every knob, "
                              "tick and menu, the effect settings, the LFOs, the macros, "
                              "the scenes, the M:C tab, the VST inserts and the takes' "
                              "settings are forgotten, every knob lands on its default and the "
                              "MIDI map goes back to the stock one, here and now; the sound "
                              "card, the MIDI ports, the MIDI "
                              "out, the Erae, Link, the video inputs and the ticked samples "
                              "and videos stay; asks first.")
        values_btn.clicked.connect(lambda: self._reset_settings(values_only=True))
        # a set: the same values and the pools in a JSON file, for another
        # day or another machine; loading one lands it live
        save_btn = QPushButton("save set...")
        save_btn.setObjectName("small")
        save_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        save_btn.setToolTip("Write everything the values-only reset would forget, plus the "
                            "ticked samples and videos, to a JSON file (the sets folder by "
                            "default): the knobs, ticks and menus, the effect settings, the "
                            "LFOs, the macros, the scenes, the M:C tab, the VST inserts, the "
                            "takes' settings. The sound card, MIDI, the Erae, Link and the "
                            "video inputs are not in it, so a set loads on any machine.")
        save_btn.clicked.connect(self._save_set)
        load_btn = QPushButton("load set...")
        load_btn.setObjectName("small")
        load_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        load_btn.setToolTip("Pick a set file: its values and pools replace the current ones "
                            "at once, every tab following, and are remembered for the next "
                            "launch. The hardware settings stay as they are.")
        load_btn.clicked.connect(self._load_set)
        gap2 = QWidget()
        gap2.setFixedHeight(10)
        form.addRow("", gap2)
        form.addRow("", self._row([("", save_btn), ("", load_btn),
                                   ("", values_btn), ("", reset_btn)]))
        self._mode_forms.append(form)
        return self._wrap(form)

    # the settings groups that are the hardware (the sound card, MIDI, the
    # Erae, Link, the video inputs): a set file never carries them, and the
    # values-only reset keeps them along with the pools
    HARDWARE_GROUPS = ("audio", "midi", "erae", "clock", "video")
    KEEP_GROUPS = HARDWARE_GROUPS + ("samples", "videos")
    SET_VERSION = 1

    # the live state remembered between launches: the knobs and the
    # switches of the audio and video scene groups (the racks, the volumes,
    # the A/B, the grade, the video effects, the shader), not the voices (in
    # at launch, always) nor the VST knobs (the plugins keep their own state)
    LIVE_GROUPS = ("audio", "video")

    def _live_state(self):
        state = self._scene_capture()
        return {g: {"knobs": {k: v for k, v in state[g]["knobs"].items()
                              if not k.startswith("vst_")},
                    "switches": state[g]["switches"]} for g in self.LIVE_GROUPS}

    def _live_save(self, force=False):
        """The live state into the settings when it changed: the timer every
        few seconds, the close, a set saved."""
        if getattr(self, "_live_restoring", False):
            return
        text = json.dumps(self._live_state(), sort_keys=True)
        if force or text != getattr(self, "_live_saved", None):
            self._live_saved = text
            self.settings.setValue("live/state", text)

    def _live_restore(self, state=None):
        """The remembered live state (or `state`) onto the widgets, the way
        a scene lands: at launch, when a set is loaded, on a values reset."""
        if state is None:
            try:
                state = json.loads(self.settings.value("live/state", "") or "{}")
            except (ValueError, TypeError):
                state = {}
        if not isinstance(state, dict):
            return
        self._live_restoring = True
        try:
            for g in self.LIVE_GROUPS:
                part = state.get(g)
                if not isinstance(part, dict):
                    continue
                for ident, val in (part.get("switches") or {}).items():
                    self._scene_apply_switch(ident, val)
                for ident, x in (part.get("knobs") or {}).items():
                    if ident.startswith("vst_") or self._lfo_get(ident) is None:
                        continue
                    try:
                        self._scene_set(ident, float(x))
                    except (TypeError, ValueError):
                        pass
        finally:
            self._live_restoring = False
        self._live_saved = None

    def _flush_settings(self):
        """Every write still waiting on a timer, into the settings now."""
        for t in (self._mc_save_timer, self._macro_save_timer):
            if t.isActive():
                t.stop()
                t.timeout.emit()
        self._live_save(force=True)
        self._vst_save()
        self.settings.sync()

    def _sets_folder(self):
        folder = Path.cwd() / "sets"
        folder.mkdir(exist_ok=True)
        return folder

    def _save_set(self):
        """The values and the pools to a JSON file: every settings key that
        is not hardware, as the store holds it."""
        self._flush_settings()
        name = time.strftime("set %Y-%m-%d %H.%M.json")
        path, _f = QFileDialog.getSaveFileName(self, "Save the set",
                                               str(self._sets_folder() / name),
                                               "Gacha set (*.json)")
        if not path:
            return
        values = {k: self.settings.value(k) for k in self.settings.allKeys()
                  if k.split("/")[0] not in self.HARDWARE_GROUPS}
        data = {"gacha_set": self.SET_VERSION, "saved": time.strftime("%Y-%m-%d %H:%M"),
                "values": values}
        try:
            Path(path).write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        except (OSError, TypeError, ValueError) as e:
            self.log.appendPlainText(f"set: not saved: {e}")
            return
        self.log.appendPlainText(f"set: {len(values)} values saved to {Path(path).name}")

    def _load_set(self):
        """A set file picked, its values and pools replace the current ones
        at once: into the settings, then into every tab. The hardware keys
        stay, the file's hardware keys, if any, are ignored."""
        path, _f = QFileDialog.getOpenFileName(self, "Load a set", str(self._sets_folder()),
                                               "Gacha set (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            values = data["values"]
            if not isinstance(values, dict) or "gacha_set" not in data:
                raise ValueError("not a Gacha set")
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.log.appendPlainText(f"set: not loaded: {e}")
            return
        values = {k: v for k, v in values.items()
                  if isinstance(k, str) and k.split("/")[0] not in self.HARDWARE_GROUPS}
        self._apply_values(values)
        self.log.appendPlainText(f"set: {Path(path).name} loaded, {len(values)} values "
                                 f"(saved {data.get('saved', '?')})")

    def _apply_values(self, values):
        """A set's values into the running app: the settings first (every
        value key gone, the set's written), then every widget that reads a
        remembered value is set from them again, as at build, so the
        engines follow through the widgets' own handlers and the settings
        hold the set for the next launch."""
        for key in self.settings.allKeys():
            if key.split("/")[0] not in self.HARDWARE_GROUPS:
                self.settings.remove(key)
        for key, val in values.items():
            self.settings.setValue(key, val)
        self.settings.sync()
        self._restore_values()

    @staticmethod
    def _slot_rows(text, n):
        """A remembered JSON list of slot records, one per slot, empty
        records for the missing ones, as JSON again."""
        try:
            rows = json.loads(text or "[]")
        except (ValueError, TypeError):
            rows = []
        if not isinstance(rows, list):
            rows = []
        rows = [r if isinstance(r, dict) else {} for r in rows[:n]]
        return rows + [{}] * (n - len(rows))

    def _restore_values(self):
        """Every remembered value from the settings into its widget, the
        way the tabs read them at build: the mirror of those reads, one
        place, for a set loaded live. A new remembered widget needs its
        line here."""
        sv = self.settings.value

        def combo(w, key, default):
            v = str(sv(key, default))
            w.setCurrentText(v if w.findText(v) >= 0 else str(default))

        # the effect settings
        for ident, _fx, _label, kind, _lo, _hi, default, _suffix, _what in FX_SETTINGS:
            w = self.fxset[ident]
            if kind == "list":
                combo(w, f"afx/{ident}", default)
            else:
                try:
                    v = float(sv(f"afx/{ident}", default))
                except (TypeError, ValueError):
                    v = float(default)
                w.setValue(int(round(v)) if kind == "int" else v)
        # the Perform tab and the takes
        self.take_pre.setValue(sv("takes/pre_roll", 0.0, type=float))
        self.keep_takes.setChecked(sv("takes/keep", True, type=bool))
        self.perf_gap.setChecked(sv("perform/gap", False, type=bool))
        combo(self.rep_div, "perform/rep_div", REPEAT_DEFAULT)
        combo(self.speed_sync, "perform/speed_sync", SPEED_SYNC_DEFAULT)
        combo(self.pitch_mode, "perform/pitch", PITCH_MODES[0])
        combo(self.brake_len, "perform/brake_len", BRAKE_DEFAULT)
        self.nudge_pct.setValue(sv("perform/nudge_pct", NUDGE_DEFAULT, type=int))
        combo(self.frag_div, "perform/frag_div", FRAG_DEFAULTS["div"])
        self._frag_fill_slices(str(sv("perform/frag_slice", "follow")))
        combo(self.frag_size, "perform/frag_size", FRAG_DEFAULTS["size"])
        combo(self.frag_rate, "perform/frag_rate", FRAG_DEFAULTS["rate"])
        for name in ("motion", "pitch", "speed", "repeats", "shape", "a", "d", "s", "r"):
            getattr(self, f"frag_{name}").setValue(
                sv(f"perform/frag_{name}", FRAG_DEFAULTS[name], type=int))
        combo(self.frag_dir, "perform/frag_dir", FRAG_DEFAULTS["dir"])
        combo(self.frag_play, "perform/frag_play", FRAG_DEFAULTS["play"])
        self._frag_changed()
        for name, d in (("size", 80), ("smear", 0), ("pitch", 0), ("a", 10), ("r", 250)):
            getattr(self, f"freeze_{name}").setValue(sv(f"perform/freeze_{name}", d, type=int))
        self.burst_len.setValue(sv("perform/burst_ms", 100.0, type=float))
        combo(self.perf_quant, "perform/quant", QUANT_DEFAULT)
        # the M:C tab: the cells, the modes, the FX channel, the retrig
        try:
            saved = json.loads(sv("mc/values", "") or "{}")
            if not isinstance(saved, dict):
                saved = {}
        except (ValueError, TypeError):
            saved = {}
        for (t, key), w in self.mc.items():
            v = saved.get(("fx_" if t == "fx" else f"{t}_") + key)
            default = MC_PARAM[key][3]
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v) if v is not None else bool(default))
                continue
            try:
                n = int(v) if v is not None else int(default)
            except (TypeError, ValueError):
                n = int(default)
            if isinstance(w, QComboBox):
                w.setCurrentIndex(max(0, min(w.count() - 1, n)))
            else:
                w.setValue(n)
        for t, m in self.mc_mode.items():
            try:
                n = int(saved.get(f"{t}_mode", 0))
            except (TypeError, ValueError):
                n = 0
            m.setCurrentIndex(max(0, min(m.count() - 1, n)))
        self.mc_fx_channel.setValue(sv("mc/fx_channel", 9, type=int))
        combo(self.mc_retrig_rate, "mc/retrig_rate", "1/16")
        combo(self.mc_retrig_len, "mc/retrig_len", "1 step")
        self.mc_retrig_fade.setValue(sv("mc/retrig_fade", 0, type=int))
        self.mc_retrig_vel.setValue(sv("mc/retrig_vel", 100, type=int))
        # the LFOs: the rows onto the tab and into the engine, the missing
        # ones off
        rows = self._lfo_saved()[:LFO_SLOTS]
        rows += [{"on": False, "dest": ""}] * (LFO_SLOTS - len(rows))
        self._scene_apply_lfo_rows(rows)
        # the macros: the engine, then the rows (the knobs through their
        # handler, so the destinations follow) and the editor
        self.macros.load_json(json.dumps(self._slot_rows(sv("macros/slots", ""), MACRO_SLOTS)))
        for mc, r in zip(self.macros.slots, self.macro_rows):
            r["name"].setText(mc.name)
            r["knob"].setValue(int(round(1000 * mc.value)))
            r["val"].setText(f"{round(100 * mc.value)} %")
        self._macro_refresh_rows()
        self._macro_show(self.macro_pick.currentIndex())
        self._macro_save_settings()
        # the scenes: the bank, the rows, the morph
        self.scenes.load_json(json.dumps(self._slot_rows(sv("scenes/slots", ""), SCENE_SLOTS)))
        self.scenes.current = None
        for sc, r in zip(self.scenes.slots, self.scene_rows):
            r["name"].setText(sc.name)
            for g, cb in r["ticks"].items():
                cb.setChecked(sc.ticks.get(g, True))
        self._scene_refresh_rows()
        self._scene_save_settings()
        self.scene_morph_on.setChecked(sv("scenes/morph", False, type=bool))
        combo(self.scene_morph_bars, "scenes/morph_bars", MORPH_DEFAULT)
        # the video FX pool: the effects through their handler (the knobs
        # grey out), the shaders quietly (no layer switching on a load)
        try:
            pool = json.loads(sv("videofx/pool", "") or "{}")
            if not isinstance(pool, dict):
                pool = {}
        except (ValueError, TypeError):
            pool = {}
        fx_on, sh_off = set(pool.get("effects_on", [])), set(pool.get("shaders_off", []))
        for key, cb in self.vfx_on.items():
            cb.setChecked(key in fx_on)
        for name, cb in self.shader_on.items():
            cb.blockSignals(True)
            cb.setChecked(name not in sh_off)
            cb.blockSignals(False)
        self._vfx_save()
        # the VST inserts: the chain as the set has it (the text first:
        # removing writes the empty chain back)
        text = sv("vst/inserts", "")
        for ins in list(self.inserts.items):
            self._vst_remove(ins)
        self.settings.setValue("vst/inserts", text)
        self._vst_restore()
        # the pools
        self.samples_tree.restore()
        self.videos_tree.restore()
        # the live knobs and switches last, over everything above
        self._live_restore()
        self._flush_settings()

    def _reset_settings(self, values_only=False):
        """A confirmation, then: every setting cleared on the way out and
        the app started again (closing writes settings, so the wipe is
        last), or with values_only the values alone, live, the groups in
        KEEP_GROUPS (the sound card, MIDI, the Erae, Link, the video inputs,
        the pools) staying."""
        ask = QMessageBox(self)
        ask.setIcon(QMessageBox.Warning)
        if values_only:
            ask.setWindowTitle("Forget the values?")
            ask.setText("Forget every saved value and put every knob at its default?")
            ask.setInformativeText("Every knob, tick and menu, the effect settings, the "
                                   "LFOs, the macros, the scenes, the M:C tab, the VST "
                                   "inserts and the takes' settings are forgotten, and the "
                                   "MIDI map goes back to the stock one. The "
                                   "sound card, the MIDI ports, the MIDI out, the "
                                   "Erae, Link, the video inputs and the ticked samples and "
                                   "videos stay, and so do the files on disk.")
        else:
            ask.setWindowTitle("Forget every saved setting?")
            ask.setText("Forget every saved setting and restart Gacha with its defaults?")
            ask.setInformativeText("Ports, the MIDI map, every knob and tick, the pools, the "
                                   "scenes, the macros, the M:C tab and the window are forgotten. "
                                   "Files on disk (samples, videos, takes, renders, plugins) stay.")
        ask.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        ask.setDefaultButton(QMessageBox.Cancel)
        if ask.exec() != QMessageBox.Yes:
            return
        if values_only:
            self._reset_values()
            return
        self._wipe_settings = "all"
        self.log.appendPlainText("settings: everything forgotten on the way out, restarting")
        self.close()

    def _reset_values(self):
        """The values-only reset, live: every key outside KEEP_GROUPS goes,
        every remembered widget reads its default again (as a set loads),
        the live knobs and switches land on what they were at build, and
        the MIDI map goes back to the stock one (the ports stay)."""
        for key in self.settings.allKeys():
            if key.split("/")[0] not in self.KEEP_GROUPS:
                self.settings.remove(key)
        self.settings.sync()
        self._restore_values()
        self._live_restore(self._live_defaults)
        self._midi_defaults()                   # the map too; the ports stay
        self._live_save(force=True)
        self.settings.sync()
        self.log.appendPlainText("settings: the values forgotten, every knob at its default, "
                                 "the stock MIDI map")

    def _live_clock_rows(self, form):
        """The one BPM: the live clock. It runs the visuals, the racks' delays
        and the section engine whenever no song plays, a generated song takes
        it unless random is ticked, and a song that plays sets it to its own;
        from the box, from taps, or shared with an Ableton Link session."""
        self.live_bpm = QDoubleSpinBox(minimum=40.0, maximum=240.0,
                                       singleStep=1.0, decimals=1, suffix=" bpm")
        self.live_bpm.setValue(self.tempo.bpm)
        self.live_bpm.setToolTip(
            "The tempo: of the visuals, the racks' delays and the section "
            "engine whenever no song plays, and of the next generated song "
            "unless random is ticked. Type it, tap it (T), or share it with "
            "Ableton Link (the button next to it); a song that starts playing "
            "sets it to its own.")
        self.live_bpm.valueChanged.connect(self._live_bpm_edited)
        self.bpm_random = QCheckBox("random")
        self.bpm_random.setChecked(True)
        self.bpm_random.setToolTip(
            "Generate picks a tempo between 60 and 180 for each song. Unticked, "
            "the song is rendered at the BPM shown here (the live clock, or the "
            "Link session's).")
        tap_btn = QPushButton("Tap (T)")
        tap_btn.setToolTip("Tap on the beat, four or more times, the first "
                           "one on the downbeat: the intervals set the tempo")
        tap_btn.clicked.connect(self.tap_tempo)
        down_btn = QPushButton("Downbeat (D)")
        down_btn.setToolTip(
            "Mark the one of the bar: the bar restarts now, tempo kept, so the "
            "bar count (the 8-bar sections, the shaders' iBar, the tint's walk) "
            "lines up with the music. Not sent over Link.")
        down_btn.clicked.connect(self.live_downbeat)
        # one row for the clock: the bpm, random, Link on/off, tap, downbeat
        # and Link's status, small buttons so it all fits
        self.link_on = QPushButton("Link off")
        self.link_on.setCheckable(True)
        self.link_on.setToolTip(
            "Ableton Link: on, the live clock joins the Link session on the "
            "network (Live, Traktor, drum machines...) and takes its bpm; "
            "typing in the box, tapping T or a playing section then changes "
            "the bpm for everyone. Start/stop is shared with the Perform "
            "tab's section: a peer's play starts the current section again, "
            "its stop stops it on the bar, and N / Shift+N here start and "
            "stop the peers. A section started from silence waits for the "
            "session's bar (the beat and meter here show it); alone in the "
            "session, the bar is ours. Remembered.")
        self.link_on.toggled.connect(self._link_toggled)
        self.link_play = QPushButton("▶")      # shows what a press sends
        self.link_play.setToolTip(
            "Play or stop, sent to the Link session: ▶ when the session is "
            "stopped, Live and the other peers start from their bar and, as "
            "for any peer's play, the current section starts again here (none "
            "yet: the log says so); ■ while it plays, the peers stop and the "
            "section here stops on the bar as with Shift+N. Needs Link on.")
        self.link_play.clicked.connect(self._link_press)
        self.link_play.setFixedWidth(28)              # just the icon
        self.link_play.setStyleSheet("padding: 2px 0px;")
        self.link_status = QLabel("off")
        self.link_status.setProperty("role", "sub")
        # the status takes what is left of the row and never widens the pane
        self.link_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        # (the peers come first in its text, so a narrow screen loses the
        # transport glyph at the end, never the peers; the beat itself shows
        # on the Link button while it is on)
        for b in (self.link_on, self.link_play, tap_btn, down_btn):
            b.setObjectName("small")
        form.addRow("BPM", self._row([("", self.live_bpm), ("", self.bpm_random),
                                      ("", self.link_on), ("", self.link_play),
                                      ("", tap_btn), ("", down_btn), ("", self.link_status)]))
        if self.link is None:
            self.link_on.setEnabled(False)
            self.link_play.setEnabled(False)
            self.link_status.setText("no aalink")
            self.link_status.setToolTip(f"aalink not installed ({LINK_INSTALL_HINT})")
        else:
            self.link_on.setChecked(self.settings.value("clock/link", False, type=bool))

    def _tab_drums(self):
        form = QFormLayout()
        self.drum_style = QComboBox()
        self.drum_style.addItems(["random", "four-floor", "breakbeat",
                                  "boom-bap", "halftime", "dnb", "minimal",
                                  "ukg", "dembow", "one-drop", "footwork",
                                  "clave", "idm"])
        self.drum_style.setToolTip(
            "The kit's pattern family, for the songs and the live sections alike: "
            "four-floor (a kick on every beat, house), breakbeat (syncopated, the "
            "snare off the grid, lots of ghosts), boom-bap (hip hop's lazy "
            "backbeat), halftime (the snare on beat 3 only, trap hats), dnb "
            "(two-step, the kick on 1 and the and of 3), minimal (a few hits), ukg "
            "(2-step garage, no kick on beat 3, skippy hats), dembow (reggaeton's "
            "boom-ch-boom), one-drop (dub and reggae, everything on beat 3), "
            "footwork (juke, cells of kicks and a sparse clap), clave (afro-latin, "
            "the son clave 3-2 over a bombo kick), idm (irregular). random picks "
            "one per song, or per live section.")
        form.addRow("Drum style", self.drum_style)

        self.swing_random = QCheckBox("random (0–0.06)")
        self.swing_random.setChecked(True)
        self.swing_random.setToolTip("A swing drawn per song (or per live section) "
                                     "between none and 0.06 of a step, a light shuffle; "
                                     "untick to set one.")
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
        for role, what in (("kick", "the kick"), ("snare", "the snare"),
                           ("hihat", "the closed hihat"), ("ride", "the ride")):
            self.gain[role].setToolTip(
                f"Level of {what} carved out of the samples (each hit is a slice of "
                "a sample shaped into that drum), 1 = as carved, 0 = the voice left "
                "out of the pattern. Live, the Perform tab's M buttons mute a voice "
                "without a render.")
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
        return form

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
        self.intro_style.addItems(["random", "ambient", "sparse", "none", "build",
                                   "drums-first", "reverse-swell", "collage",
                                   "filtered"])
        self.intro_style.setToolTip(
            "How a song opens, and what a live intro section is shaped like: "
            "ambient (textures, the drums often left out), sparse (one voice, one "
            "texture), none (a plain groove from the start), build (the voices stack "
            "up bar by bar), drums-first (the kit alone, no textures), reverse-swell "
            "(a long reversed sample crescendos into the drop), collage (one-shots "
            "scattered like a field recording, no beat), filtered (a lowpass opens "
            "bar by bar, the club door). random picks one.")
        form.addRow("Intro style", self.intro_style)

        self.intro_cap = QCheckBox("cap at")
        self.intro_cap.setToolTip("Songs only: tick to cap the intro at the bars next to "
                                  "it; unticked the intro takes a section length of its "
                                  "own. A live intro is always 8 bars.")
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
        self.outro_tail_auto.setToolTip("The ring-out's length from the outro style "
                                        "(2.5 to 7 s); untick to set it. A live outro "
                                        "keeps it inside its 8 bars, 4 bars at most.")
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
        self.fade_in.setToolTip("Songs only: a fade on the whole song's start")
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
        return form

    def _tab_sections_layers(self):
        """The sections, the drums and the layers & effects on one page, a
        rule between them, one Randomize for all three."""
        forms = (self._tab_sections(), self._tab_drums(), self._tab_layers())
        self._mode_forms.extend(forms)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 8, 8)
        for i, form in enumerate(forms):
            form.setContentsMargins(8, 8, 8, 8)
            form.setVerticalSpacing(6)
            if i:
                hr = QFrame()
                hr.setFrameShape(QFrame.HLine)
                hr.setStyleSheet("color: rgba(255, 255, 255, 50);")
                outer.addWidget(hr)
            outer.addLayout(form)
        outer.addStretch(1)
        btn = QPushButton("Randomize")
        btn.setToolTip("Random value for every control on this page, within its range")
        btn.clicked.connect(self._randomize_sections)
        btn.clicked.connect(self._randomize_drums)
        btn.clicked.connect(self._randomize_layers)
        legend = QLabel(f"{MODE_SONG} the generated songs only; the rest the live render "
                        "reads at N as well")
        legend.setProperty("role", "sub")
        legend.setToolTip("The mark after a row's name, or after a box's own label when "
                          "the row mixes both.")
        row = QHBoxLayout()
        row.addWidget(legend)
        row.addStretch(1)
        row.addWidget(btn)
        outer.addLayout(row)
        w = QWidget()
        w.setLayout(outer)
        return self._scrolling(w)      # three forms outgrow the pane

    def _mode_mark(self, label):
        """The faint note after a label's text, no taller than the text, and
        what it means as the tip."""
        if label.property("mode_marked"):
            return
        label.setProperty("mode_marked", True)
        label.setTextFormat(Qt.RichText)
        label.setText(f'{html.escape(label.text())}&nbsp;<span style="font-size: 10px; '
                      f'color: rgba(242,233,229,0.3)">{MODE_SONG}</span>')
        label.setToolTip(MODE_TIP)

    @staticmethod
    def _inside_box(w, top, kinds):
        """Whether `w` sits inside another box (the line edit of a spinbox or
        a combo), looking up to `top`."""
        par = w.parentWidget()
        while par is not None:
            if isinstance(par, kinds):            # top itself may be the box
                return True
            if par is top:
                break
            par = par.parentWidget()
        return False

    def _mark_modes(self, forms):
        """The marks on the Sections and Settings pages: the controls only a
        song uses (its structure, its length, its seed, its sample count),
        against what gacha_section reads off the knobs (and the pans and the
        styles the live render takes), which is the rest and goes unmarked.
        A row of song-only boxes is marked once at its name; a row that
        mixes both marks the song boxes' own labels and leaves the name
        bare, a song box with no label of its own getting the tip alone."""
        song = {self.intro_bars, self.intro_cap, self.section_bars, self.fade_in,
                self.fade_out, self.cymbal_chance, self.repeat_chance, self.gap_chance,
                self.break_chance, *self.synth.values(), self.synth_style,
                self.duration, self.seed, self.seed_random, self.num_samples,
                self.all_samples, self.bpm_random}
        kinds = (QAbstractSpinBox, QComboBox, QCheckBox, QLineEdit)
        for form in forms:
            for i in range(form.rowCount()):
                li = form.itemAt(i, QFormLayout.LabelRole)
                fi = form.itemAt(i, QFormLayout.FieldRole)
                if li is None or fi is None or li.widget() is None or fi.widget() is None:
                    continue
                label, field = li.widget(), fi.widget()
                if not isinstance(label, QLabel) or not label.text():
                    continue
                boxes = [w for w in [field, *field.findChildren(QWidget)]
                         if isinstance(w, kinds) and not self._inside_box(w, field, kinds)]
                if not boxes or not any(w in song for w in boxes):
                    continue
                if all(w in song for w in boxes):
                    self._mode_mark(label)
                    continue
                lay = field.layout()
                prev = None
                for j in range(lay.count() if lay is not None else 0):
                    w = lay.itemAt(j).widget()
                    if w in song and isinstance(prev, QLabel):
                        self._mode_mark(prev)
                    elif w in song:
                        w.setToolTip(f"{w.toolTip()}\n\n{MODE_SONG} {MODE_TIP}".strip())
                    if w is not None:
                        prev = w

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
        for w in (self.layers_min, self.layers_max):
            w.setToolTip(
                "A texture is a long cut of a sample looped under the drums for the "
                "whole section (the layers stem): maybe reversed, maybe stretched to "
                "whole bars, through a random effect chain, panned, at a level from "
                "the row below. A groove section gets between min and max of them, "
                "drawn at random; a break more, a sparse or build section one, some "
                "intros none. 0 and 0 = no textures at all.")
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
        self.chop_gain.setToolTip("Level of the chops: a sample cut into eight slices "
                                  "and replayed on a 16th-note pattern, through one "
                                  "effect chain (the chops stem)")
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
        for w in (self.fx_min, self.fx_max):
            w.setToolTip(
                "Every texture, chop layer and one-shot goes through a chain of "
                "effects drawn at random from a pool (a tempo-synced delay, reverb, "
                "chorus, phaser, the saturator, bitcrush, a resonant filter, a pitch "
                "shift): between min and max of them, in a random order. 1 to 1 is "
                "one effect, cleaner; 3 and up is a wash.")
        form.addRow("Effects / chain", self._row([("min", self.fx_min),
                                                  ("max", self.fx_max)]))

        self.pan_drums = self._pan_spin(0.3)
        self.pan_layers = self._pan_spin(0.6)
        self.pan_events = self._pan_spin(0.5)
        self.pan_drums.setToolTip("How far from the centre a drum hit may land, 0 to 1 "
                                  "(the kick stays in the middle; the hat stutter "
                                  "sweeps across): each hit drawn at random within it")
        self.pan_layers.setToolTip("How far from the centre a texture may sit, 0 to 1, "
                                   "drawn at random per texture")
        self.pan_events.setToolTip("How far from the centre the one-shots, the chops "
                                   "and the collage hits may land, 0 to 1")
        form.addRow("Pan", self._row([("drums", self.pan_drums),
                                      ("layers", self.pan_layers),
                                      ("events", self.pan_events)]))
        return form

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
        add_btn.setToolTip("The selected section of the render on the left onto the "
                           "end of the final arrangement; sections from different "
                           "renders can be mixed, and one added as often as wanted")
        add_btn.clicked.connect(self._arr_add)
        up_btn = QPushButton("↑")
        up_btn.setToolTip("Move the selected section of the final one place earlier")
        up_btn.clicked.connect(lambda: self._arr_move(-1))
        down_btn = QPushButton("↓")
        down_btn.setToolTip("Move the selected section of the final one place later")
        down_btn.clicked.connect(lambda: self._arr_move(1))
        del_btn = QPushButton("Remove")
        del_btn.setToolTip("Take the selected section out of the final arrangement")
        del_btn.clicked.connect(self._arr_remove)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Empty the final arrangement")
        clear_btn.clicked.connect(self.arrangement.clear)
        render_btn = QPushButton("Render final")
        render_btn.setToolTip("Write the arrangement as one stitched track, the "
                              "sections joined with seamless crossfades, listed with "
                              "the renders")
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
            if i:
                box.addSpacing(6)
            if len(pairs) >= 3 and isinstance(widget, QAbstractSpinBox):
                # crowded rows: drop the arrows, keep typing / wheel / keys
                widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
            if name:                     # no empty label, it would pad the widget
                lbl = QLabel(name)
                lbl.setProperty("role", "sub")
                box.addWidget(lbl)
            box.addWidget(widget, stretch=1)
        w = QWidget()
        w.setLayout(box)
        return w

    # a key named at the end of a button's or a tick's text: "(N)", "(Shift+G)",
    # "(F11)", "(/)"; a range like "(0-0.06)" is not one
    KEY_IN_TEXT = re.compile(r"^(.*\S) \(((?:Shift\+|Ctrl\+|Alt\+)*(?:F\d{1,2}|[A-Z]|[-=,./;'\[\]\\`]))\)$")

    def _keys_into_tips(self):
        """The keyboard shortcut off every button's and tick's text and into
        its tooltip, as a last line, so the faces stay short. The texts set
        later (record / stop, next section) are written without one."""
        for w in [*self.findChildren(QPushButton), *self.findChildren(QCheckBox)]:
            m = self.KEY_IN_TEXT.match(w.text())
            if m is None:
                continue
            text, key = m.groups()
            w.setText(text)
            tip = w.toolTip().rstrip()
            w.setToolTip(f"{tip}\n\nKey: {key}" if tip else f"Key: {key}")

    @staticmethod
    def _info(text):
        """A hint as an info panel: a faint circled i, the text wrapped
        beside it, in a soft box with a little padding."""
        panel = QFrame()
        panel.setObjectName("info")
        box = QHBoxLayout(panel)
        box.setContentsMargins(10, 6, 10, 6)
        box.setSpacing(8)
        icon = QLabel("\u24d8")
        icon.setObjectName("info_icon")
        body = QLabel(text)
        body.setProperty("role", "sub")
        body.setWordWrap(True)
        box.addWidget(icon, alignment=Qt.AlignTop)
        box.addWidget(body, stretch=1)
        return panel

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
    def _knobs_dict(self):
        """The Generate tab's knobs as the engine's dict (songs and live
        sections read the same values)."""
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
        return knobs

    def _knobs(self):
        """Knobs object from the Generate tab, for gacha_section."""
        return E.Knobs(**{k: v for k, v in self._knobs_dict().items()
                          if k in E.Knobs.__dataclass_fields__}).sanitized()

    def _section_kw(self):
        """The Generate tab's compose settings a live section takes as well:
        the pans and the drum style. Read on the GUI thread."""
        return {"pan_drums": min(1.0, abs(self.pan_drums.value())),
                "pan_layers": min(1.0, abs(self.pan_layers.value())),
                "pan_events": min(1.0, abs(self.pan_events.value())),
                "drum_style": self.drum_style.currentText(),
                "intro_style": self.perf_intro.currentText(),      # the Perform tab's copies:
                "outro_style": self.perf_outro.currentText()}      # never none

    def generate(self):
        self._show_log()                        # watch the render as it goes
        samples = self.checked_samples()
        video_audio = [str(p) for p in self._song_videos()] \
            if self.video_audio.isChecked() else []
        if not samples and not video_audio:
            # nothing ticked: the ticked videos' own sound, if they have any
            with_audio = [p for p in self._song_videos() if video_has_audio(p)]
            if with_audio:
                self.log.appendPlainText(
                    f"no samples ticked: using the audio of {len(with_audio)} video"
                    f"{'s' if len(with_audio) != 1 else ''} as the material "
                    "(video audio switched on, Video tab)")
                self.video_audio.setChecked(True)
                video_audio = [str(p) for p in self._song_videos()]
            else:
                self.log.appendPlainText(
                    "✗ nothing to build from: no samples ticked, and the ticked "
                    "videos have no audio track. Tick samples in the Samples tab, "
                    "or tick a video with sound in the Videos tab")
                return
        job = {"count": 1,                      # one song a Generate
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
               "bpm": None if self.bpm_random.isChecked() else self._song_bpm()}
        if not self.all_samples.isChecked():
            job["num_samples"] = self.num_samples.value()
        if self.intro_cap.isChecked():
            job["intro_bars"] = self.intro_bars.value()
        knobs = self._knobs_dict()
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
        self.generate_btn.setText("Generate song")
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

    def _build_prep_row(self):
        """Under the Samples tree: the sample prep tool (gacha_prep.py),
        which cuts a recording of single notes into a note-named set."""
        self.prep_window = None
        btn = QPushButton("Prepare samples…")
        btn.setObjectName("small")
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.setToolTip("Cut a recording (or a video's sound) into one-shot samples: "
                       "every note found, cropped, pitched, brought to one loudness "
                       "and written as a new set here, G2 as G2 in the filename")
        btn.clicked.connect(self._open_prep)
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.addWidget(btn)
        row.addStretch(1)
        return row

    def _open_prep(self):
        if self.prep_window is None:
            from gacha_prep import PrepWindow
            self.prep_window = PrepWindow()
            self.prep_window.exported.connect(lambda _folder: self.refresh_lists())
        self.prep_window.show()
        self.prep_window.raise_()
        self.prep_window.activateWindow()

    def _build_fetch_row(self):
        """Under the Videos tree: a URL and a Fetch button. yt-dlp pulls
        the clip into videos/web/ and gacha_transcode.py makes it seek
        fast, both on a worker thread (see gacha_fetch.py)."""
        self.fetch_worker = None
        self.fetch_url = QLineEdit()
        self.fetch_url.setPlaceholderText("paste a video URL (YouTube, Vimeo, ...)")
        self.fetch_url.setClearButtonEnabled(True)
        self.fetch_url.returnPressed.connect(self._fetch_video)
        self.fetch_btn = QPushButton("Fetch")
        self.fetch_btn.setObjectName("small")
        self.fetch_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.fetch_btn.clicked.connect(self._fetch_video)
        tip = ("Download the video at this URL with yt-dlp into videos/web/ "
               "(best picture up to 1080p, one video, never a whole playlist), "
               "then transcode it for instant seeking when ffprobe is installed. "
               "Progress in the Log tab; the clip shows up here when done")
        if not fetch_available():
            tip += f"  (yt-dlp is not installed: {FETCH_INSTALL_HINT})"
            self.fetch_url.setPlaceholderText(f"yt-dlp missing: {FETCH_INSTALL_HINT}")
            self.fetch_url.setEnabled(False)
            self.fetch_btn.setEnabled(False)
        self.fetch_url.setToolTip(tip)
        self.fetch_btn.setToolTip(tip)
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.fetch_url, stretch=1)
        row.addWidget(self.fetch_btn)
        return row

    def _fetch_video(self):
        url = self.fetch_url.text().strip()
        if self.fetch_worker is not None and self.fetch_worker.isRunning():
            return
        if not looks_like_url(url):
            self.videos_status.setText("that is not a http(s) URL")
            return
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("Fetching…")
        self.videos_status.setText("fetching…")
        self.fetch_worker = FetchWorker(url)
        self.fetch_worker.line.connect(self.log.appendPlainText)
        self.fetch_worker.progress.connect(self._fetch_progress)
        self.fetch_worker.done.connect(self._fetch_done)
        self.fetch_worker.start()

    def _fetch_progress(self, pct):
        self.videos_status.setText(f"fetching {pct:.0f} %" if pct >= 0
                                   else "downloaded, transcoding…")

    def _fetch_done(self, ok, path):
        self.fetch_btn.setEnabled(True)
        self.fetch_btn.setText("Fetch")
        if ok:
            self.fetch_url.clear()
            self.videos_tree.refresh()
            self.log.appendPlainText(f"✔ in the Videos tab: {self.videos_tree.name(path)}")
        else:
            self.log.appendPlainText("✗ fetch failed, see the lines above")
        self._on_videos_changed()

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
        cams = video_inputs()
        gone = []
        for second, combo in ((False, self.video_in), (True, self.video_in2)):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("off", None)
            for name, dev in cams:
                combo.addItem(name, dev)
            if cur:
                i = combo.findData(cur)
                if i >= 0:
                    combo.setCurrentIndex(i)
            combo.blockSignals(False)
            if combo.currentData() != cur:
                gone.append(second)
        if hasattr(self, "through_in"):
            self._refresh_through_devices()
        for second in gone:
            self._video_in_changed(second=second)    # the device went away

    def _video_audio_toggled(self, on):
        """Video audio on: a soundtrack is one long take, so cuts should
        start anywhere in it, not always at its first seconds."""
        if on and self.random_start.currentText() == "off":
            self.random_start.setCurrentText("all")
            self.log.appendPlainText("video audio: random start set to 'all' "
                                     "so cuts land anywhere in the film")

    def _video_widgets(self, second):
        """(device combo, mode combo, settings key, backdrop setter, name)
        of the first or the second video input."""
        bd = getattr(self, "backdrop", None)
        if second:
            return (self.video_in2, self.video_in2_mode, "video/in2_mode",
                    bd.set_second if bd is not None else None, "video input 2")
        return (self.video_in, self.video_in_mode, "video/in_mode",
                bd.set_input if bd is not None else None, "video input")

    def _video_in_changed(self, *_, second=False):
        """A video input combo -> backdrop: a capture device, or off (back
        to clips for the first; the second just closes). The mode list
        follows the device; the remembered mode is re-picked. The two inputs
        must be different devices: DirectShow and V4L2 open a device once."""
        combo, modes, key, apply, name = self._video_widgets(second)
        dev = combo.currentData()
        other = (self.video_in if second else self.video_in2).currentData()
        if dev and dev == other:
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
            dev = None
            if hasattr(self, "log"):
                self.log.appendPlainText(f"{name}: that device is already the other "
                                         "input; pick another one")
        want = self.settings.value(key, "")
        modes.blockSignals(True)
        modes.clear()
        modes.addItem("default", None)
        if dev:
            for label, opts in video_input_modes(dev):
                modes.addItem(label, opts)
            i = modes.findText(want) if want else -1
            modes.setCurrentIndex(max(0, i))
        modes.blockSignals(False)
        modes.setEnabled(bool(dev))
        if apply is not None:
            apply(dev, modes.currentData())
        if not second and not dev and hasattr(self, "perf_ab")                 and self.perf_ab.value() != 100:
            self.perf_ab.setValue(100)                # no live picture: full B
            if hasattr(self, "log"):
                self.log.appendPlainText(f"{name}: off, A/B to full B")
        if second and hasattr(self, "through"):
            self._sync_hold_tape()
        if second and hasattr(self, "log"):
            if dev:
                self.log.appendPlainText(
                    f"{name}: {combo.currentText()}, W puts it on the B side"
                    if not self.video_second.isChecked() else
                    f"{name}: {combo.currentText()}, on the B side of the pedal")
            elif self.video_second.isChecked():
                self.log.appendPlainText(f"{name}: off, B is the take frames again")

    def _video_mode_changed(self, *_, second=False):
        """Another capture mode: the device is reopened with it."""
        combo, modes, key, apply, name = self._video_widgets(second)
        self.settings.setValue(key, modes.currentText() if modes.currentData() else "")
        dev = combo.currentData()
        if dev and apply is not None:
            apply(dev, modes.currentData())
            self.log.appendPlainText(f"{name}: {combo.currentText()} at "
                                     f"{modes.currentText()}")

    def _toggle_video_second(self, on):
        """W / the tick / a pad: the second input's picture as the B side of
        the pedal, in place of the take frames, or the take frames again."""
        if not hasattr(self, "backdrop"):
            return
        self.backdrop.set_second_on_b(on)
        self._sync_hold_tape()
        if on and not self.video_in2.currentData():
            self.log.appendPlainText("video input: no second device chosen, B stays "
                                     "the take frames (pick one next to 'second')")
        elif on:
            self.log.appendPlainText(f"video input: {self.video_in2.currentText()} on "
                                     "the B side of the pedal (W back to the take frames)")
        else:
            self.log.appendPlainText("video input: B is the take frames again")

    def _toggle_video_off(self, on):
        """V / the tick / a pad: the whole picture off (decoders closed, no
        repaint) or back on as it was."""
        self.settings.setValue("video/off", on)
        if not hasattr(self, "backdrop"):
            return
        self.backdrop.set_enabled(not on)
        if on:
            self.log.appendPlainText("video: off, decoders closed, no GPU work (V brings it back)")
        else:
            self.log.appendPlainText("video: on again")

    def _webcam_b(self):
        """The second video input stands in for the take frames on B."""
        return self.video_second.isChecked() and bool(self.video_in2.currentData())

    def _sync_hold_tape(self):
        """Webcam on B: the pedal moves the picture only, the tape's sound
        stays whole (the webcam brings none in); the section still comes in
        with the pedal. Off: equal power tape against section as usual."""
        if self.through is not None:
            self.through.hold_tape = self._webcam_b()

    def _toggle_live(self, on):
        """Live mode on: the song stops and its analysis is dropped, the
        input opens, the clock restarts on a downbeat and the clip is
        re-picked as for a new song. Off: capture closes, songs work again."""
        self._sec_idx = -1
        if on:
            self.player.stop()
            self._sections, self._sec_bounds, self._env = [], [], None
            self._outro = None
            if self.through is not None and self.through.in_name:   # the stream owns the input
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
        # only what is ticked: the samples, and the ticked videos' soundtracks
        # when Video audio (Video tab) says so or no sample is ticked, as for
        # a song; nothing ticked anywhere is no material, never the whole folder
        files = tuple(str(p) for p in self.samples_tree.checked())
        videos = ()
        if self.video_audio.isChecked() or not files:
            videos = tuple(str(p) for p in self.videos_tree.checked())
        return ("samples", files, videos) if files or videos else None

    def _perf_bank(self, key):
        """Material for `key`, built if needed (call on the render thread)."""
        if self._perf_material and self._perf_material[0] == key:
            return self._perf_material[1]
        if key[0] == "takes":
            bank = self.takes.bank()
        else:
            import contextlib, io
            files = list(key[1])
            with contextlib.redirect_stdout(io.StringIO()):
                if key[2]:                       # the videos' soundtracks, extracted
                    tracks = E.extract_video_audio(key[2])   # once and cached (ffmpeg)
                    files += [E.VIDEO_AUDIO_DIR / n for n in tracks]
                bank = E.load_samples(files)
        mat = Material(bank, seed=random.randrange(1 << 30))
        self._perf_material = (key, mat)
        return mat

    def _perf_key(self):
        """What a section is rendered from: the material, the kind, the
        tempo, and the Generate tab's knobs as of now, so a section rendered
        ahead with other knobs (a sub chance turned down, say) is stale and
        gets rendered again rather than played."""
        kind = self.perf_kind.currentText()
        stamp = (tuple(sorted(self._knobs_dict().items())),
                 tuple(sorted(self._section_kw().items())))
        return (self._perf_material_key(), kind, round(self.tempo.bpm, 1), stamp)

    PERF_RESTAMP_S = 0.5       # knobs quiet this long before the next section is redone

    def _perf_refresh_ready(self):
        """Every so often from the poll: the section rendered ahead no longer
        matches the knobs (or the kind, the tempo, the material) and the
        knobs have settled: render it again, so N still lands at once."""
        now = time.monotonic()
        key = self._perf_key()
        if key != self._perf_key_last:
            self._perf_key_last, self._perf_key_t = key, now
            return
        if (self._perf_ready is not None and self._perf_ready[0] != key
                and self._perf_busy is None and key[0] is not None
                and now - self._perf_key_t >= self.PERF_RESTAMP_S
                and self.through is not None and self.through.running):
            if self._perf_start("section", key):
                self.log.appendPlainText("perform: settings changed, the next section "
                                         "is rendered again")

    def _perf_render(self, key, kind_choice, knobs=None, kw=None):
        """Render thread: a section for `key`, stems resampled to the audio
        thread's rate; result lands in _perf_result. `knobs` and `kw` are the
        Generate tab as of the press (the sub, the gains, the chances, the
        pans, the drum style...)."""
        try:
            mat = self._perf_bank(key[0])
            if knobs is not None:
                mat.k = knobs                         # carving is cached, the knobs are live
            if not mat.ok:
                self._perf_result = ("error", "no material with sound (record a take, "
                                              "or tick samples or videos with sound)")
                return
            kind = random.choice(KINDS) if kind_choice == "random" else kind_choice
            sec = render_section(mat, seed=random.randrange(1 << 30), bpm=key[2],
                                 bars=8, kind=kind, **(kw or {}))
            sec.knobs, sec.kw, sec.mat_key = mat.k, dict(kw or {}), key[0]   # for a shuffle
            sr = self.through.sr if self.through is not None else 48000
            stems = {}
            for name, data in sec.stems.items():
                d = data if sr == E.SR else resample_poly(data, sr // 300, E.SR // 300, axis=0)
                stems[name] = np.ascontiguousarray(d.T.astype(np.float32))
            sec.swell = None                          # the break's cymbal swell, a bar
            try:
                sw = render_swell(mat, random.randrange(1 << 30), sec.bpm)
                if sw is not None:
                    if sr != E.SR:
                        sw = resample_poly(sw, sr // 300, E.SR // 300, axis=0)
                    sec.swell = np.ascontiguousarray(sw.T.astype(np.float32))
            except Exception as e:                    # noqa: BLE001 - a break without it
                print(f"perform: no swell for this section ({e})")
            self._perf_result = ("section", key, sec, stems)
        except Exception as e:                        # noqa: BLE001
            self._perf_result = ("error", f"render failed: {e}")

    def _perf_render_shot(self, key, knobs=None, kw=None):
        try:
            mat = self._perf_bank(key[0])
            if knobs is not None:
                mat.k = knobs
            if not mat.ok:
                self._perf_result = ("error", "no material with sound")
                return
            shot = render_oneshot(mat, random.randrange(1 << 30), 60.0 / key[2],
                                  (kw or {}).get("pan_events", 0.5))
            sr = self.through.sr if self.through is not None else 48000
            clip = shot["clip"] * shot["gain"]
            if sr != E.SR:
                clip = resample_poly(clip, sr // 300, E.SR // 300, axis=0)
            self._perf_result = ("shot", np.ascontiguousarray(clip.T.astype(np.float32)), shot)
        except Exception as e:                        # noqa: BLE001
            self._perf_result = ("error", f"one-shot failed: {e}")

    def perform_shuffle(self, part):
        """Randomize, F5-F8 (or the button; the code says shuffle): `part`
        (drums, layers, chops, events)
        of the current section rendered again from the material,
        a new sound in that part, the other three untouched, the loop
        where it is: on the next bar. A voice (KIT_PARTS, the R buttons):
        another sample carved into that drum, the hits as they are. The
        render takes a second, after whatever renders now; several
        presses queue up, one of each part (pressed again before it
        lands, the press is the same one)."""
        if self.through is None or not self.through.running:
            self.log.appendPlainText("perform: the audio output is not open (Settings tab)")
            self._shuffle_faces()
            return
        if self._perf_current is None:
            self.log.appendPlainText("randomize: no section yet, press N for one")
            self._shuffle_faces()
            return
        busy = self._perf_busy
        if part in self._shuffle_queue or part in self._shuffle_landing \
                or (busy is not None and busy[0] == "shuffle" and busy[1] == part):
            self._shuffle_faces()                     # pressed twice: one new sound
            return
        self._shuffle_queue.append(part)
        self._shuffle_kick()
        self._shuffle_faces()

    def _shuffle_faces(self):
        """The four buttons say where their shuffle is: queued, rendering,
        on its way to the bar, or nothing (plain, unlit)."""
        busy = self._perf_busy
        for part, btn in self.perf_shuffles.items():
            if busy is not None and busy[0] == "shuffle" and busy[1] == part:
                face = "rendering..."
            elif part in self._shuffle_landing:
                face = "on the bar..."
            elif part in self._shuffle_queue:
                face = "queued..."
            else:
                face = "randomize"
            if btn.text() != face:
                btn.setText(face)
            btn.setChecked(face != "randomize")
        for part, btn in self.perf_randoms.items():
            btn.setChecked(part in self._shuffle_queue or part in self._shuffle_landing
                           or (busy is not None and busy[0] == "shuffle" and busy[1] == part))

    def _shuffle_kick(self):
        """The next shuffle asked, once the render thread is free."""
        if not self._shuffle_queue or self._perf_busy is not None:
            return
        cur = self._perf_current
        if cur is None:
            self._shuffle_queue.clear()
            return
        part = self._shuffle_queue.pop(0)
        sec, stems, _sr = cur
        salts, force = dict(sec.salts), set(sec.force)
        kit_salts = dict(getattr(sec, "kit_salts", {}))
        if part in KIT_PARTS:                         # the sound only
            kit_salts[part] = kit_salts.get(part, 0) + 1
        else:
            salts[part] = salts.get(part, 0) + 1
            force.add(part)
        mat_key = self._perf_material_key() or getattr(sec, "mat_key", None)
        if mat_key is None:
            self.log.appendPlainText("randomize: no material (record a take, or tick samples or videos)")
            self._shuffle_faces()
            return
        self._perf_busy = ("shuffle", part)
        threading.Thread(target=self._perf_render_shuffle,
                         args=(mat_key, sec, stems, part, salts, force, kit_salts),
                         daemon=True, name="gacha-render").start()
        self.log.appendPlainText(f"randomize {part}: rendering...")

    def _perf_render_shuffle(self, mat_key, sec, stems, part, salts, force, kit_salts):
        """Render thread: `sec` again with the same seed, kind, tempo,
        knobs and shape, `part`'s salt moved on (and that part forced in),
        then only that part's stems (SHUFFLE_STEMS: the sub with the drums,
        the layers and the chops) into a copy of `stems` at the player's
        rate, its events in place
        of the old part's. Lands in _perf_result."""
        try:
            mat = self._perf_bank(mat_key)
            mat.k = getattr(sec, "knobs", mat.k)
            if not mat.ok:
                self._perf_result = ("error", "no material with sound")
                return
            kw = dict(getattr(sec, "kw", {}))
            if sec.kind == "intro" and sec.style:        # the shape pinned, not drawn again
                kw["intro_style"] = sec.style
            elif sec.kind == "outro" and sec.style:
                kw["outro_style"] = sec.style
            new = render_section(mat, seed=sec.seed, bpm=sec.bpm, bars=sec.bars, kind=sec.kind,
                                 salts=salts, force=force, kit_salts=kit_salts, **kw)
            sr = self.through.sr if self.through is not None else 48000
            out = dict(stems)
            for name in SHUFFLE_STEMS[part]:
                d = new.stems[name]
                d = d if sr == E.SR else resample_poly(d, sr // 300, E.SR // 300, axis=0)
                out[name] = np.ascontiguousarray(d.T.astype(np.float32))
            roles = set(SHUFFLE_ROLES[part])
            events = sorted([e for e in sec.events if e[5] not in roles]
                            + [e for e in new.events if e[5] in roles], key=lambda e: e[0])
            self._perf_result = ("shuffle", part, sec, new, out, events, salts, force, kit_salts)
        except Exception as e:                        # noqa: BLE001
            self._perf_result = ("error", f"randomize failed: {e}")

    SHUFFLE_NOTES = {"drums": ("drums", "sub"), "layers": ("layer", "sub"),
                     "chops": ("chops",), "events": ("event", "swell", "collage")}

    def _perf_shuffled(self, part, sec, new, out, events, salts, force, kit_salts):
        """A shuffle's render landed: into the section it was for, if that
        is still the current one, and onto the player on the bar if it plays."""
        cur = self._perf_current
        if cur is None or cur[0] is not sec:
            self.log.appendPlainText(f"randomize {part}: the section moved on, dropped")
            self._shuffle_faces()
            return
        heads = self.SHUFFLE_NOTES.get(part, ())
        own = lambda note: note.split(" ", 1)[0] in heads   # noqa: E731
        sec.salts, sec.force, sec.events, sec.kit_salts = salts, force, events, kit_salts
        sec.notes = [n for n in sec.notes if not own(n)] + [n for n in new.notes if own(n)]
        for name in SHUFFLE_STEMS[part]:
            sec.stems[name] = new.stems[name]
        self._perf_current = (sec, out, cur[2])
        self._pad_cache.clear()                       # the pads follow the section's kit
        if part in KIT_PARTS:
            srcs = sorted({e[1] for e in new.events if e[5] == part})
            what = f"{part} <- {', '.join(srcs)}" if srcs else "not in this section"
        else:
            what = "; ".join(n for n in new.notes if own(n)) or "nothing in it"
        la = self.through
        pl = la.player if la is not None and la.running else None
        if self._perf_playing is sec and pl is not None and pl.playing:
            pl.replace(out, events)
            self._shuffle_landing.append(part)
            self.log.appendPlainText(f"randomize {part}: {what}; on the bar")
        else:
            self.log.appendPlainText(f"randomize {part}: {what}; with the next start")
        self._shuffle_faces()

    def _perf_start(self, what, key):
        if self._perf_busy is not None:
            return False
        self._perf_busy = (what, key)
        target = self._perf_render if what == "section" else self._perf_render_shot
        # the Generate tab as of this press, read here: Qt widgets stay on
        # this thread; so a kick gain or a pan moved between two N lands
        knobs, kw = self._knobs(), self._section_kw()
        args = (key, self.perf_kind.currentText(), knobs, kw) if what == "section" \
            else (key, knobs, kw)
        threading.Thread(target=target, args=args, daemon=True, name="gacha-render").start()
        return True

    def perform_next(self):
        """N: the next section, on the bar. Uses the pre-rendered one when it
        matches the material, kind and tempo; else renders and plays when done."""
        if self.through is None or not self.through.running:
            self.log.appendPlainText("perform: the audio output is not open (Settings tab)")
            return
        key = self._perf_key()
        if key[0] is None:
            self.log.appendPlainText("perform: no material (record a take, or tick samples or videos)")
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
        pl = self.through.player
        bar_t0 = self._link_bar_t0() if not pl.playing else None   # from silence: Link's bar
        if not pl.playing:                  # from silence: a new song for the picture too,
            self._pick_song_video()         # a ticked video instead of whatever was on
            self._live_switch_t = time.monotonic()
        # an outro is entered at its second half: four bars of groove into
        # the ending effect and the ring-out, then the stop
        start_bar = sec.bars // 2 if sec.kind == "outro" else 0
        pl.queue(stems, sec.events, sec.bpm, sec.bars, bar_t0, start_bar)
        if self._shuffle_landing:                     # the queued section replaces everything
            self._shuffle_landing = []
            self._shuffle_faces()
        self._perf_playing = sec
        self._perf_current = (sec, stems, self.through.sr)   # what a start brings back
        self.perf_next_btn.setText("next section")           # from here on N moves on
        wait = "" if bar_t0 is None or bar_t0 <= time.monotonic() \
            else f", on the Link bar in {bar_t0 - time.monotonic():.1f} s"
        self.log.appendPlainText(f"section: {sec.kind} {sec.bpm:.0f} bpm, "
                                 f"{'; '.join(sec.notes)}{wait}")

    LINK_GRACE = 0.25          # beats: this far past the session's bar, start late into it

    def _link_bar_t0(self):
        """Starting a section from silence with Link on: the monotonic time
        of the session's bar to start it on, the next one, or the one that
        began less than LINK_GRACE beats ago (a peer's play is seen a poll
        late), the loop then entered that far in. Alone in the session the
        session's bar is moved to now instead and the start is immediate
        (None), so the peers that join later share our bar."""
        if not self._link_active():
            return None
        if self.link.peers == 0:
            self.link.restart_bar()
            return None
        bpm, _beat, phase = self.link.snapshot()
        now = time.monotonic()
        beat_s = 60.0 / max(20.0, bpm)
        if phase < self.LINK_GRACE:
            return now - phase * beat_s
        return now + (LinkClock.QUANTUM - phase) * beat_s

    def perform_stop(self):
        """Shift+N: stop the section on the bar; with none playing, start
        the current one again (a Link peer's stop and play do the same)."""
        la = self.through
        pl = la.player if la is not None and la.running else None
        if pl is not None and not pl.playing and pl.pending is None:
            self.perform_start()
            return
        self._perf_want = False                        # a render on its way stays quiet
        if la is not None:
            la.player.clear()
        if self._shuffle_landing:                      # it is in the section, for the next start
            self._shuffle_landing = []
            self._shuffle_faces()
        self._perf_playing = None
        self._perf_hold = None
        self.backdrop.override = None
        self.perf_status.setText("no section")

    def perform_start(self, source="stop"):
        """Start the current section again, as it was: the one that played
        last (stopped by Shift+N or a Link peer), nothing new is rendered.
        Nothing to bring back: says so and waits for N."""
        if self.through is None or not self.through.running:
            self.log.appendPlainText("perform: the audio output is not open (Settings tab)")
            return False
        pl = self.through.player
        if pl.playing or pl.pending is not None:
            return True                                # already on, or on its way
        cur = self._perf_current
        if cur is None or cur[2] != self.through.sr:
            self.log.appendPlainText(f"{source}: no section to start yet, press N for one")
            return False
        sec, stems, _sr = cur
        self._perf_play(sec, stems)
        self.log.appendPlainText(f"{source}: the section starts again")
        return True

    PAD_ROLL = 4                   # a hat stutter pad: this many hits, a 1/32 apart

    def perform_hit(self, role, vel=1.0):
        """A drum pad: `role`'s hit from the material (the kit the sections
        are carved from), at `vel` 0..1 (the velocity layer and the gain),
        into its voice's rack on the quantize grid, over the tape whatever
        the pedal says, and its note to the MIDI out at the same instant.
        No material: the note alone (a Model:Cycles plays it)."""
        la = self.through
        if la is None or not la.running:
            self.log.appendPlainText("perform: the audio output is not open (Settings tab)")
            return
        vel = max(0.0, min(1.0, float(vel)))
        clip = None
        mat = self._perf_material[1] if self._perf_material else None
        src = "hihat" if role == "glitch" else role
        sec = self._perf_current[0] if self._perf_current else None
        salt = getattr(sec, "kit_salts", {}).get(role, 0) if sec is not None else 0
        variants = mat.kit_for(src, salt)[0] if mat is not None else []   # the section's sound
        if variants:
            rr = self._pad_rr.get(src, 0)
            self._pad_rr[src] = rr + 1
            layer = 0 if vel < 0.5 else 1 if vel < 0.85 else 2
            key = (id(mat), src, salt, rr % len(variants), layer, la.sr)
            clip = self._pad_cache.get(key)
            if clip is None:
                hit = variants[rr % len(variants)][layer]
                if la.sr != E.SR:
                    hit = resample_poly(hit, la.sr // 300, E.SR // 300, axis=0)
                clip = np.ascontiguousarray(hit.T.astype(np.float32))
                if role == "glitch":                   # the stutter: a short roll of the hat
                    step = int(la.sr * 60.0 / max(1e-6, self.tempo.bpm) / 8)
                    n = clip.shape[1] + step * (self.PAD_ROLL - 1)
                    roll = np.zeros((2, n), np.float32)
                    for k in range(self.PAD_ROLL):
                        roll[:, k * step:k * step + clip.shape[1]] += clip * (0.9 ** k)
                    clip = roll
                if len(self._pad_cache) > 64:
                    self._pad_cache.clear()
                self._pad_cache[key] = clip
            clip = clip * (0.4 + 0.6 * vel)
        la.pad_hit(role, clip, vel)
        self.perf_status.setText(f"{VOICE_LABELS.get(role, role)} hit, {vel:.2f}")

    def perform_event(self):
        """E: a one-shot from the material, on the quantize grid."""
        if self.through is None or not self.through.running:
            self.log.appendPlainText("perform: the audio output is not open (Settings tab)")
            return
        key = self._perf_key()
        if key[0] is None:
            return
        self._perf_start("shot", key)

    def perform_break(self, kind="glitch"):
        """G / Shift+G: a break of `kind` on the 4-bar phrase's last bar, then
        the next section on the one (the pre-rendered one when it fits)."""
        if self.through is None or not self.through.running:
            self.log.appendPlainText("perform: the audio output is not open (Settings tab)")
            return
        pl = self.through.player
        if pl.playing and not pl.filling:
            swell = getattr(self._perf_playing, "swell", None) if kind == "break" else None
            pl.fill(kind, swell)
            self.log.appendPlainText(f"{kind}: on the phrase's last bar, then next"
                                     + ("" if kind != "break" or swell is not None
                                        else " (no swell for this section)"))
        self.perform_next()                            # queued: lands when the break ends

    BURST_MIN_SHOW = 0.08          # s: a 10 ms burst still flashes two screen frames

    def perform_burst(self):
        """B: `burst_len` ms of the material from a random spot, on the
        quantize grid, over the tape; the frame it was cut from takes the
        picture while it sounds."""
        if self.through is None or not self.through.running:
            self.log.appendPlainText("perform: the audio output is not open (Settings tab)")
            return
        key = self._perf_key()
        if key[0] is None:
            self.log.appendPlainText("perform: no material (record a take, or tick samples or videos)")
            return
        length = self.burst_len.value() / 1000.0
        take = None
        if key[0][0] == "takes":
            takes = [t for t in self.takes.takes if len(t.audio)]
            pool = [t for t in takes if t.frames] or takes
            take = random.choice(pool)
            audio, src = take.engine_audio(), take.name
        else:                                          # rehearsal: the sample bank
            if not (self._perf_material and self._perf_material[0] == key[0]):
                self.log.appendPlainText("burst: material not loaded yet, fire an event "
                                         "(E) or a section (N) first")
                return
            mat = self._perf_material[1]
            src = random.choice(mat.names)
            audio = mat.bank[src]
        off = random.uniform(0.0, max(0.0, len(audio) / E.SR - length))
        clip = cut_burst(audio, E.SR, off, length, self.through.sr)
        num = self.through.player.fire(clip, now=True)
        frame = take is not None and bool(take.frames)
        self._burst = (num, max(length, self.BURST_MIN_SHOW), take, off) if frame else None
        self.log.appendPlainText(f"burst <- {src} @ {off:.1f} s, {length * 1000:.0f} ms"
                                 + ("" if frame else " (no picture)"))

    def _burst_len_frac(self, x):
        """A knob over 10 ms .. 1 s on a log scale: halfway is 100 ms."""
        lo, hi = self.burst_len.minimum(), self.burst_len.maximum()
        self.burst_len.setValue(lo * (hi / lo) ** max(0.0, min(1.0, x)))

    def _burst_frame(self):
        """The burst's picture while it sounds: the take's frames running
        from the cut point, in step with the sound, from the moment the
        player started it (on the grid). None before and once it is over."""
        if self._burst is not None:
            num, show, take, off = self._burst
            la = self.through
            fired = la.player.burst_fired if la is not None else None
            if fired is None or fired[0] < num:
                return None                        # waiting for the grid
            t0 = fired[1]
            now = time.monotonic()
            if now < t0 + show:
                return take.frame_at(off + max(0.0, now - t0))
            self._burst = None
        return None

    def _perf_stem(self, stem, on):
        """F1-F4: a stem in or out, for the section and a song alike, on
        the quantize grid."""
        if self.through is not None:
            self.through.switch("stem", stem, on)

    def _perf_voice(self, v, on):
        """A drum voice in or out, on the grid; its MIDI notes with it."""
        if self.through is not None:
            self.through.switch("voice", v, on)

    def _perf_solo(self, v, on):
        if self.through is not None:
            self.through.switch("solo", v, on)

    def _voices_reset(self):
        """Shift+F8: every voice in, no solo, the voice racks clean."""
        for v in VOICES:
            self.perf_solos[v].setChecked(False)
            self.perf_voices[v].setChecked(False)     # the M off: the voice in
            for key, *_ in AUDIO_FX:
                self.afx[(v, key)].setValue(0.0)
            self.afx[(v, "vol")].setValue(1.0)

    def _perf_gap_changed(self, on):
        self.settings.setValue("perform/gap", bool(on))
        if self.through is not None:
            self.through.player.gap = bool(on)

    # ---- the repeat
    def _rep_hold(self, source, on):
        """The repeat held (`on`) or let go by `source`: "key", "midi" or
        "ui" (the latching button). It sounds while any of them holds it."""
        if on:
            self._rep_sources.add(source)
        else:
            self._rep_sources.discard(source)
        self._rep_apply()

    def _rep_apply(self):
        """Push the repeat (division, position, direction) to the player
        while it is held, or let it go; the button shows the hold."""
        held = bool(self._rep_sources)
        if self.rep_btn.isChecked() != held:
            self.rep_btn.blockSignals(True)
            self.rep_btn.setChecked(held)
            self.rep_btn.blockSignals(False)
        la = self.through
        if la is None or not la.running:
            return
        pl = la.player
        if held and pl.playing:
            pl.set_repeat(dict(REPEAT_DIVS).get(self.rep_div.currentText(), 1.0),
                          self.rep_pos.value(), self.rep_rev.isChecked())
        else:
            # let go on the next bar, so the cells roll out in time (at
            # once with the quantize grid off)
            pl.clear_repeat(1.0 if la.quant > 0 else 0.0)

    def _rep_div_changed(self, name):
        self.settings.setValue("perform/rep_div", name)
        self._rep_apply()

    def _rep_div_step(self, d):
        """- and =: the division one finer (d = 1) or coarser, no wrap."""
        i = self.rep_div.currentIndex() + d
        self.rep_div.setCurrentIndex(max(0, min(self.rep_div.count() - 1, i)))

    def _rep_div_frac(self, x):
        """A knob over the divisions: the bar at the bottom, 1/64 at the top."""
        n = self.rep_div.count()
        self.rep_div.setCurrentIndex(min(n - 1, int(max(0.0, min(1.0, x)) * n)))

    def _rep_pos_changed(self, v):
        self.rep_pos_lbl.setText(f"{v:+d}" if v else "0")
        self._rep_apply()

    # ---------- the speed: gears, tape stop, nudge, pitch ----------
    def _player(self):
        la = self.through
        return la.player if la is not None and la.running else None

    def _speed_toggled(self, gear, on):
        """Half or double time pressed: the other one off, the player told
        (it lands on the grid)."""
        other = self.double_btn if gear == "half" else self.half_btn
        if on and other.isChecked():
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        self._speed_apply()

    def _speed_apply(self):
        pl = self._player()
        if pl is None:
            return
        gear = "half" if self.half_btn.isChecked() else \
            "double" if self.double_btn.isChecked() else None
        pl.speed_sync = dict(SPEED_SYNCS).get(self.speed_sync.currentText(), 1.0)
        pl.locked = self.pitch_mode.currentText() == "locked"
        pl.brake_bars = dict(BRAKE_LENS).get(self.brake_len.currentText(), 1.0)
        pl.set_speed(gear)

    def _speed_sync_changed(self, name):
        self.settings.setValue("perform/speed_sync", name)
        self._speed_apply()

    def _pitch_changed(self, name):
        self.settings.setValue("perform/pitch", name)
        self._speed_apply()

    def _brake_len_changed(self, name):
        self.settings.setValue("perform/brake_len", name)
        self._speed_apply()

    def _brake_hold(self, source, on):
        """The tape stop held (`on`) or let go by `source`: "key", "midi" or
        "ui" (the latching button). It brakes while any of them holds it."""
        if on:
            self._brake_sources.add(source)
        else:
            self._brake_sources.discard(source)
        held = bool(self._brake_sources)
        if self.brake_btn.isChecked() != held:
            self.brake_btn.blockSignals(True)
            self.brake_btn.setChecked(held)
            self.brake_btn.blockSignals(False)
        pl = self._player()
        if pl is not None:
            pl.brake_bars = dict(BRAKE_LENS).get(self.brake_len.currentText(), 1.0)
            pl.set_brake(held and pl.playing)

    def _nudge_hold(self, sign, source, on):
        """A nudge held or let go; the rate follows at the next poll."""
        if on:
            self._nudge_sources[sign].add(source)
        else:
            self._nudge_sources[sign].discard(source)
        if on and self._link_active():
            self.log.appendPlainText("nudge: not under Link, the session owns the beat")

    def _nudge_factor(self):
        """The rate's nudge: 1 ± the amount while held (none under Link)."""
        if self._link_active():
            return 1.0
        f = 1.0
        pct = self.nudge_pct.value() / 100.0
        if self._nudge_sources[1]:
            f *= 1.0 + pct
        if self._nudge_sources[-1]:
            f *= 1.0 - pct
        return f

    def _frag_fill_slices(self, want=None):
        """The slice pick: follow, then one entry per slice of the division."""
        cur = want if want is not None else self.frag_slice.currentText()
        div = int(self.frag_div.currentText() or 16)
        self.frag_slice.blockSignals(True)
        self.frag_slice.clear()
        self.frag_slice.addItems(["follow"] + [str(k) for k in range(1, div + 1)])
        self.frag_slice.setCurrentText(cur if self.frag_slice.findText(cur) >= 0 else "follow")
        self.frag_slice.blockSignals(False)

    def _frag_div_changed(self, _text):
        self._frag_fill_slices()
        self._frag_changed()

    def _frag_changed(self, *_a):
        """The Fragments rows into the engine, and remembered."""
        f = FRAGMENTS
        f["div"] = int(self.frag_div.currentText() or 16)
        f["size"] = dict(FRAG_UNITS).get(self.frag_size.currentText(), 1 / 16)
        f["rate"] = dict(FRAG_RATES).get(self.frag_rate.currentText(), 0.0)
        f["motion"] = self.frag_motion.value() / 100.0
        f["pitch"] = float(self.frag_pitch.value())
        f["speed"] = self.frag_speed.value() / 100.0
        f["repeats"] = int(self.frag_repeats.value())
        f["shape"] = self.frag_shape.value() / 100.0
        f["dir"] = self.frag_dir.currentText()
        f["play"] = self.frag_play.currentText()
        f["attack"] = self.frag_a.value() / 1000.0
        f["decay"] = self.frag_d.value() / 1000.0
        f["sustain"] = self.frag_s.value() / 100.0
        f["release"] = self.frag_r.value() / 1000.0
        st = self.settings.setValue
        st("perform/frag_div", f["div"])
        st("perform/frag_slice", self.frag_slice.currentText())
        for name, w in (("size", self.frag_size), ("rate", self.frag_rate),
                        ("dir", self.frag_dir), ("play", self.frag_play)):
            st(f"perform/frag_{name}", w.currentText())
        for name, w in (("motion", self.frag_motion), ("pitch", self.frag_pitch),
                        ("speed", self.frag_speed), ("repeats", self.frag_repeats),
                        ("shape", self.frag_shape), ("a", self.frag_a), ("d", self.frag_d),
                        ("s", self.frag_s), ("r", self.frag_r)):
            st(f"perform/frag_{name}", int(w.value()))

    def _freeze_changed(self, *_a):
        """The Freeze row into the master freeze, and remembered."""
        FREEZE["size_ms"] = float(self.freeze_size.value())
        FREEZE["smear"] = self.freeze_smear.value() / 100.0
        FREEZE["pitch"] = float(self.freeze_pitch.value())
        FREEZE["attack"] = self.freeze_a.value() / 1000.0
        FREEZE["release"] = self.freeze_r.value() / 1000.0
        st = self.settings.setValue
        for name, w in (("size", self.freeze_size), ("smear", self.freeze_smear),
                        ("pitch", self.freeze_pitch), ("a", self.freeze_a),
                        ("r", self.freeze_r)):
            st(f"perform/freeze_{name}", int(w.value()))

    def _freeze_hold(self, source, on):
        """The freeze held (`on`) or let go by `source`: "key", "midi" or
        "ui" (the latching button). It holds while any of them holds it;
        the audio stream's master freeze and the picture both read it."""
        if on:
            self._freeze_sources.add(source)
        else:
            self._freeze_sources.discard(source)
        held = bool(self._freeze_sources)
        if self.freeze_btn.isChecked() != held:
            self.freeze_btn.blockSignals(True)
            self.freeze_btn.setChecked(held)
            self.freeze_btn.blockSignals(False)
        la = self.through
        if la is not None:
            la.freeze.held = held

    def _freeze_state(self):
        """The master freeze sounding, for the picture (see
        gacha_freeze.Freeze.state), or None."""
        la = self.through
        if la is None or not la.running or not la.freeze.active:
            return None
        return la.freeze.state()

    def _frag_hold(self, key, source, on):
        """Slice `key` (1..32, or 0 for the slice under the playhead) held
        (`on`) or let go by `source`: "key", "midi" or "ui". The voice
        plays while any source holds that key."""
        srcs = self._frag_sources.setdefault(key, set())
        was = bool(srcs)
        if on:
            srcs.add(source)
        else:
            srcs.discard(source)
        held = bool(srcs)
        pl = self._player()
        if pl is not None and held != was:
            if held:
                if pl.playing:
                    pl.frag.press(key)
            else:
                pl.frag.release(key)
        if key == self._frag_pick_key and self.frag_btn.isChecked() != held:
            self.frag_btn.blockSignals(True)
            self.frag_btn.setChecked(held)
            self.frag_btn.blockSignals(False)

    def _frag_hold_pick(self, source, on):
        """The F key and the button: the slice picked in the Perform tab
        (follow = the one under the playhead), the key fixed at the press
        so the let-go releases the same voice."""
        if on and not any(self._frag_sources.get(self._frag_pick_key, ())):
            pick = self.frag_slice.currentText()
            self._frag_pick_key = int(pick) if pick.isdigit() else 0
        self._frag_hold(self._frag_pick_key, source, on)

    def _frag_release_all(self):
        """Every held slice let go (a stop): the sources and the voices."""
        for key, srcs in list(self._frag_sources.items()):
            if srcs:
                srcs.clear()
                self._frag_hold(key, "ui", False)
        pl = self._player()
        if pl is not None:
            pl.frag.clear()

    def _frag_note(self):
        """The fragments for the status: 'fragments 3/16 · 1/16 · dense ·
        motion 100 %' or ''."""
        pl = self._player()
        fs = pl.frag_state() if pl is not None else None
        if fs is None:
            return ""
        parts = [f"fragments {fs['slice']}/{fs['div']}", self.frag_size.currentText(),
                 self.frag_rate.currentText(), f"motion {self.frag_motion.value()} %"]
        if self.frag_speed.value() != 100:
            parts.append("frozen" if self.frag_speed.value() == 0
                         else f"speed {self.frag_speed.value()} %")
        if self.frag_pitch.value():
            parts.append(f"{self.frag_pitch.value():+d} st")
        if fs["voices"] > 1:
            parts.append(f"{fs['voices']} voices")
        return " · ".join(parts)

    def _gran_state(self):
        """The fragments sounding, for the picture, or None."""
        pl = self._player()
        return pl.frag_state() if pl is not None else None

    def _gran_heads(self, gs, size):
        """The bands of the granular picture: (left edge, width, playhead,
        lit), all as fractions of the frame's width. A fragment's own
        grains when one sounds, the frame being its slice; else a few
        made up on the Fragments rows' settings, cycling on the grain
        clock over GRAN_VIEW_S seconds, so the picture shows what the rows
        would do."""
        heads = []
        if gs is not None:
            for x0, w, ph, rev in gs["heads"][:GRAN_HEADS]:
                x0 = min(x0, 1.0 - w)
                hx = x0 + (1.0 - ph if rev else ph) * w
                lit = 0.5 - 0.5 * math.cos(2 * math.pi * ph)
                heads.append((x0, w, hx, lit))
            return heads
        spread = min(1.0, self.frag_motion.value() / 400.0)
        for i in range(5):
            life = self._gran_clock + ((i * 0.618) % 1.0)
            gen = math.floor(life)
            ph = life - gen
            h = lambda k: (math.sin(i * 127.1 + gen * 7.13 + k * 311.7) * 43758.5453) % 1.0
            pos = size + 0.005 + h(1) * spread
            rev = spread > 0.5 and h(2) < spread - 0.5
            x0 = 1.0 - pos / GRAN_VIEW_S
            w = size / GRAN_VIEW_S
            hx = x0 + (1.0 - ph if rev else ph) * w
            heads.append((x0, w, hx, 0.5 - 0.5 * math.cos(2 * math.pi * ph)))
        return heads

    def _freeze_heads(self, fz):
        """The granular picture's bands for the master freeze: its grains
        over the last GRAN_VIEW_S seconds of the mix, the frozen spot near
        the right edge (see _gran_heads)."""
        heads = []
        for back, length, ph, rev in fz["heads"][:GRAN_HEADS]:
            w = length / GRAN_VIEW_S
            x0 = min(1.0 - w, max(0.0, 1.0 - back / GRAN_VIEW_S))
            hx = x0 + (1.0 - ph if rev else ph) * w
            heads.append((x0, w, hx, 0.5 - 0.5 * math.cos(2 * math.pi * ph)))
        return heads

    def _speed_note(self, pl):
        """The speed for the status: '' or ' · half time', ' · tape stop
        42 %', ' · nudge +4 %', ' · 126/120 bpm'."""
        parts = []
        hs = pl.head_state()
        if hs["mode"] == "brake":
            parts.append(f"tape stop {round(100 * hs['rate'])} %")
        elif hs["mode"] == "half":
            parts.append("half time")
        elif hs["mode"] == "double":
            parts.append("double time")
        nf = self._nudge_factor()
        if nf != 1.0:
            parts.append(f"nudge {'+' if nf > 1 else '-'}{self.nudge_pct.value()} %")
        if self._freeze_sources:
            parts.append("frozen")
        if abs(self.tempo.bpm - pl.bpm) > 0.05:
            parts.append(f"{self.tempo.bpm:.0f}/{pl.bpm:.0f} bpm")
        if parts and pl.locked and hs["mode"] != "brake":
            parts.append("at pitch")
        return "".join(" · " + p for p in parts)

    def _head_phase(self):
        """(beat phase, bar phase) of what sounds under a gear or a tape
        stop, for the shaders, or None when the head is the transport."""
        pl = self._player()
        if pl is None or not pl.playing or pl.head_state()["mode"] is None \
                or not pl.beat_frames:
            return None
        h = pl.heard_s() * pl.sr
        bf = pl.beat_frames
        return (h % bf) / bf, (h % (4 * bf)) / (4 * bf)

    def _quant_frac(self):
        """The quantize grid as a fraction of a bar, 0 = at once."""
        return dict(QUANT_GRIDS).get(self.perf_quant.currentText(), 0.0)

    def _quant_changed(self, name):
        self.settings.setValue("perform/quant", name)
        if self.through is not None:
            self.through.quant = self._quant_frac()

    def _voices_audible(self):
        """The drum voices in, as the switches ask (the sound follows on
        the grid): in, and not soloed out."""
        solo = {v for v, b in self.perf_solos.items() if b.isChecked()}
        return {v for v, cb in self.perf_voices.items()
                if not cb.isChecked() and (not solo or v in solo)}

    def _push_voices(self, la):
        """The voices' switches, the solos and the grid into a fresh stream."""
        for v, cb in self.perf_voices.items():
            la.voice_on[v] = not cb.isChecked()       # the M button: lit while out
        la.voice_solo = {v for v, b in self.perf_solos.items() if b.isChecked()}
        la.quant = self._quant_frac()
        la.player.gap = self.perf_gap.isChecked()

    def _perf_ab_changed(self, v):
        self.perf_ab_lbl.setText(f"A {100 - v} %  ·  B {v} %")
        if self.through is not None:
            self.through.ab = v / 100.0

    def _poll_perform(self):
        """20 ms: collect render results, keep the live clock on the section,
        and put the take's frames on screen for the sound that plays."""
        r, self._perf_result = self._perf_result, None
        self._perf_poll_n += 1
        if self._perf_poll_n % 10 == 0:                # 5 times a second: the knobs
            self._perf_refresh_ready()
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
            elif what == "shuffle":
                self._perf_shuffled(*r[1:])
            self._shuffle_kick()                       # a shuffle waiting its turn
            self._shuffle_faces()
        la = self.through
        pl = la.player if la is not None and la.running else None
        burst = self._burst_frame()
        if burst is not None:                          # a burst: its frame, whole
            self.backdrop.override = burst
            self.backdrop.override_ab = 1.0
        if pl is None or not pl.playing:
            if self._perf_sounding:                    # our section just stopped (on the bar,
                self._perf_sounding = False            # or the output closed): the peers stop
                self._link_transport(False)            # with it; idling alone sends nothing
                if "ui" in self._rep_sources:          # the latched repeat ends with it
                    self._rep_hold("ui", False)
                if "ui" in self._brake_sources:        # and the latched tape stop
                    self._brake_hold("ui", False)
                if any(self._frag_sources.values()):   # and the fragments held
                    self._frag_release_all()
                for btn in (self.half_btn, self.double_btn):   # the gears start over
                    if btn.isChecked():
                        btn.setChecked(False)
            if self.backdrop.override is not None and burst is None:
                self.backdrop.override = None
            if burst is None and self._webcam_b():     # webcam on B: the pedal works
                self.backdrop.override_ab = float(la.ab) if la is not None \
                    else self.perf_ab.value() / 100.0  # without a stream, the slider
            if self._perf_playing is not None and pl is None:
                self._perf_playing = None
            wait = pl.waiting_for() if pl is not None else None
            if wait is not None:                       # a timed start: Link's bar
                self.perf_status.setText(
                    f"starts on the Link bar in {max(0.0, wait - time.monotonic()):.1f} s...")
                self.perf_meter.set_state("waiting for the Link bar...")
            elif self._perf_busy:
                self.perf_status.setText("rendering...")
                self.perf_meter.set_state("rendering...")
            else:
                self.perf_meter.set_state("no section")
            return
        self._perf_sounding = True
        if pl.swapped:                                 # the live clock follows the loop
            pl.swapped = False
            if self._rep_sources:                      # still held: on the new section too
                self._rep_apply()
            self.lfo.trigger(time.monotonic())         # trig and one-shot LFOs restart
            self.tempo.set_bpm(pl.bpm)
            self.tempo.anchor = pl.loop_t0 if pl.loop_t0 else time.monotonic()
            self._show_live_bpm()
            self._link_push(pl.bpm)                    # and so does the Link session
            self._link_transport(True)                 # which plays now, if it did not
            self._speed_apply()                        # the gears, the pitch, the sync
            if self._brake_sources:                    # still held: on the new one too
                self._brake_hold("ui", "ui" in self._brake_sources)
        # the transport's rate: the clock's tempo over the section's (a
        # tempo change, Link's or the box's, is followed without a render)
        # times a held nudge; and the clock's bar on the transport, so the
        # shaders, the LFOs and the meter slide with a nudge
        pl.set_rate(self.tempo.bpm / max(1e-6, pl.bpm) * self._nudge_factor())
        if pl.loop_t0 and not pl.filling:
            bar_s = 4 * 60.0 / self.tempo.bpm
            off = (pl.loop_t0 - self.tempo.anchor) % bar_s
            if off > bar_s / 2:
                off -= bar_s
            if abs(off) > 0.002:
                self.tempo.anchor += off
        if pl.replaced:                                # the shuffles handed over landed, on the bar
            pl.replaced = False
            if self._shuffle_landing:                  # (the last carries the ones before it)
                self.log.appendPlainText(f"randomize {', '.join(self._shuffle_landing)}: in")
                self._shuffle_landing = []
                self._shuffle_faces()
        if pl.dropped:                                 # a break ended here: the drop
            pl.dropped = False
            self._perf_drop_t = time.monotonic()
            self.lfo.trigger(self._perf_drop_t)
            if self._live_active():
                self._live_drop(self._perf_drop_t)     # flash, iDrop, maybe a new clip
            else:
                self._drop_clip(self._perf_drop_t)     # the clock is the loop's: clip only
        sec = self._perf_playing
        pos = pl.pos_s()
        bar_pos = pos * pl.bpm / 60.0 / 4                # bars into the loop
        bar = int(bar_pos) + 1
        busy = self._perf_busy
        ready = "next ready" if self._perf_ready else (
            "" if not busy else f"randomizing {busy[1]}..." if busy[0] == "shuffle" else "rendering next...")
        if self._shuffle_landing:
            landing = ", ".join(self._shuffle_landing) + " on the bar..."
            ready = f"{landing}  ·  {ready}" if ready else landing
        if sec is None:                                # stop asked: it lands on the bar
            self.perf_status.setText("stopping on the bar...")
            big = "stopping on the bar..."
        elif pl.filling:
            self.perf_status.setText(f"{pl.fill_state()['kind']}...  ·  {ready}")
            big = f"{pl.fill_state()['kind']}..."
        elif pl.fill_armed:                            # waiting for the phrase's last bar
            on = min(pl.bars, ((bar - 1) // PHRASE_BARS + 1) * PHRASE_BARS)
            self.perf_status.setText(
                f"{pl.fill_armed[0]} on bar {on}  ·  bar {bar}/{pl.bars}  ·  {ready}")
            big = f"{pl.fill_armed[0]} on bar {on}"
        elif pl.frag_state() is not None:              # a slice as grains over the loop
            big = self._frag_note() + self._speed_note(pl)
            self.perf_status.setText(f"{big}  ·  bar {bar}/{pl.bars}  ·  {ready}")
        elif pl.repeat is not None:                    # a held repeat over the loop
            _div, off, rev = pl.repeat
            big = (f"repeat {self.rep_div.currentText()}" + (f" {off:+d}" if off else "")
                   + (" reversed" if rev else "") + self._speed_note(pl))
            self.perf_status.setText(f"{big}  ·  bar {bar}/{pl.bars}  ·  {ready}")
        else:
            root = f" in {sec.root}" if getattr(sec, "root", None) else ""
            spd = self._speed_note(pl)
            self.perf_status.setText(
                f"playing {sec.kind}{root} {pl.bpm:.0f} bpm{spd}  ·  bar {bar}/{pl.bars}"
                f"  ·  {ready}")
            kind = sec.kind + (f" {sec.style}" if getattr(sec, "style", None) else "")
            big = f"{kind}{root}  ·  {pl.bpm:.0f} bpm{spd}"
            if getattr(sec, "drum_style", None):
                big += f"  ·  {sec.drum_style}"
            if ready:
                big += f"  ·  {ready}"
            # an outro: silent by its last bar, the sections stop there
            if sec.kind == "outro" and bar >= pl.bars and pl.pending is None:
                self.perform_stop()                    # _perf_playing is None from here
                big = f"{kind}{root}  ·  ending"
        win = pl.repeat_window()
        if win is not None:                            # the window, in bars
            bar_frames = 4 * pl.beat_frames
            win = (win[0] / bar_frames, win[1] / bar_frames, win[2],
                   pl.heard_s() * pl.bpm / 60.0 / 4)
        if pl.head_state()["mode"] is not None and pl.repeat is None:
            bar_pos = (pl.heard_s() * pl.bpm / 60.0 / 4) % max(1, pl.bars)   # what sounds
        self.perf_meter.set_state(big, pl.bars, bar_pos, win)
        # picture: the frame the sounding sample was cut from, dissolved into
        # the live picture by the A/B pedal (follow), or pinned to either side
        mode = self.perf_picture.currentText()
        ab = 1.0 if mode == "take" else (0.0 if mode == "live" else float(la.ab))
        if burst is not None:
            return
        if self._webcam_b():                           # the webcam stands in for the take frames
            self._perf_hold = None
            self.backdrop.override = None
            self.backdrop.override_ab = ab
            return
        heard = pl.heard_s()                           # a repeat: its cell's frame
        fr = self._perf_frame(heard) if (ab > 0.0 and sec is not None) else None
        if fr is not None:
            self._perf_hold = fr
        elif ab > 0.0 and sec is not None:            # a gap between events, the
            fr = self._perf_hold or self._perf_any_frame(heard)   # loop's seam: hold, no tape
        else:
            self._perf_hold = None
        self.backdrop.override = fr
        self.backdrop.override_ab = ab if fr is not None else 0.0

    def _perf_frame(self, pos):
        """Frame of the take the loudest-priority event sounding at `pos`
        (seconds into the loop) was cut from, or None (no video in it)."""
        sec = self._perf_playing
        if sec is None:                              # gone in this very tick (an outro's end)
            return None
        takes = {t.name: t for t in self.takes.takes}
        order = {"kick": 0, "snare": 0, "hihat": 0, "ohat": 0, "ride": 0, "glitch": 0,
                 "chop": 1, "oneshot": 1, "texture": 2}
        aud = self._voices_audible()                 # a voice out does not cut the picture
        best = None
        for t, src, off, dur, rate, role, *_vel in sec.events:
            if role in VOICE_LABELS and role not in aud:
                continue
            if t <= pos < t + dur and src in takes and takes[src].frames:
                pri = order.get(role, 3)
                if best is None or pri < best[0]:
                    best = (pri, t, src, off, rate)
        if best is None:
            return None
        _, t, src, off, rate = best
        return takes[src].frame_at(off + (pos - t) * rate)

    def _perf_any_frame(self, pos):
        """Before any event with a picture has sounded: a frame from the
        first take with video, at the loop position, so the pedal on B never
        shows the tape. None when no take has video."""
        for t in self.takes.takes:
            if t.frames:
                return t.frame_at(pos % max(t.seconds, 1e-3))
        return None

    # ---------- takes ----------
    def toggle_take(self):
        """R / the button / a footswitch later: start or stop a take."""
        audio = self.through if (self.through is not None and self.through.running
                                and self.through.in_name) else None
        video = self.backdrop.live_source()       # the capture device on screen
        if not self.takes.recording and audio is None and video is None:
            self.log.appendPlainText("take: nothing to record, switch the live input "
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

    def discard_take(self):
        """Shift+R / the button / a pad: the take recording is dropped, nothing
        kept; otherwise the last take is forgotten (RAM only)."""
        gone = self.takes.discard_last()
        if gone is None:
            self.log.appendPlainText("takes: nothing to forget")
        elif gone == "recording":
            self.log.appendPlainText("take: recording dropped, nothing kept")
        else:
            n = len(self.takes.takes)
            self.log.appendPlainText(
                f"{gone.name}: forgotten ({gone.seconds:.1f} s)  ·  {n} take{'s' if n != 1 else ''} left"
                + ("  ·  its file in videos/ stays" if gone.saved else ""))

    def clear_takes(self):
        """Ctrl+Shift+R / the button / a pad: every take forgotten, RAM only."""
        was_rec = self.takes.recording
        n = self.takes.clear()
        if not n and not was_rec:
            self.log.appendPlainText("takes: nothing to forget")
        else:
            self.log.appendPlainText(
                f"takes: {n} take{'s' if n != 1 else ''} forgotten"
                + (", the recording dropped" if was_rec else "")
                + "  ·  files already in videos/ stay")

    def _log_take(self, take):
        self.log.appendPlainText(
            f"{take.name}: {take.seconds:.1f} s, {len(take.frames)} frames, "
            f"{take.nbytes / 1e9:.2f} GB  ·  {len(self.takes.takes)} takes, "
            f"{self.takes.nbytes / 1e9:.1f} GB in RAM"
            + ("  (limit reached)" if self.takes.limit_hit else ""))

    def _takes_changed(self):
        rec = self.takes.recording
        self.rec_btn.setChecked(rec)
        self.rec_btn.setText("■ stop" if rec else "● record")
        if rec:
            self._take_timer.start(100)
        else:
            self._take_timer.stop()
            n = len(self.takes.takes)
            self.take_status.setText(
                "no takes" if not n else
                f"{n} take{'s' if n > 1 else ''}, {sum(t.seconds for t in self.takes.takes):.0f} s, "
                f"{self.takes.nbytes / 1e9:.1f} GB in RAM"
                + (f"  ·  writing {self.takes.pending} to videos/" if self.takes.pending
                   else (f"  ·  {len(self.takes.unsaved())} not on disk yet"
                         if self.takes.unsaved() else "")))

    def _poll_take(self):
        was = self.takes.recording
        self.takes.tick()
        if was and not self.takes.recording:           # the limit stopped it
            self._log_take(self.takes.takes[-1])
            return
        self.take_status.setText(f"● recording {self.takes.elapsed:.1f} s / "
                                 f"{self.takes.limit:.0f} s")

    def _save_takes(self):
        todo = self.takes.unsaved()
        if not todo:
            self.log.appendPlainText("takes: nothing to save" if not self.takes.takes
                                     else "takes: all on disk already")
            return
        self.takes.save(self.takes.archive_dir, takes=todo)
        self.log.appendPlainText(f"takes: writing {len(todo)} take(s) to "
                                 f"{self._rel(self.takes.archive_dir)}/ in the background")

    def _flush_takes(self):
        """Quit: the takes still only in RAM are encoded now (the MJPEG mov by
        ffmpeg, on the archive worker) when the Settings tab says to keep
        them, a progress window counting them down; cancel quits without
        the rest. This is what keeps the encode out of the show."""
        todo = self.takes.unsaved() if self.keep_takes.isChecked() else []
        if todo:
            self.takes.save(self.takes.archive_dir, takes=todo)
        total = self.takes.pending
        if not total:
            return
        folder = self._rel(self.takes.archive_dir)
        dlg = QProgressDialog(f"writing {total} take(s) to {folder}/ ...",
                              "quit without the rest", 0, total, self)
        dlg.setWindowTitle("gacha: keeping the takes")
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        while self.takes.pending and not dlg.wasCanceled():
            dlg.setValue(total - self.takes.pending)
            QApplication.processEvents()
            time.sleep(0.05)
        if self.takes.pending:
            print(f"takes: quit with {self.takes.pending} take(s) not written", file=sys.stderr)
        dlg.close()

    def _take_saved(self, take, paths, err):
        """The archive worker finished a take (queued to the GUI thread)."""
        if err:
            self.log.appendPlainText(f"{take.name}: not saved: {err}")
        else:
            names = ", ".join(self._rel(p) for p in paths) or "nothing"
            self.log.appendPlainText(f"{take.name}: saved {names}")
        if any(str(p).lower().endswith(VIDEO_EXTS) for p in paths):
            self.videos_tree.refresh()             # the set folder shows up in Videos
        if self.takes.pending == 0 and not self.takes.recording:
            self._takes_changed()                  # back to the plain status line

    @staticmethod
    def _rel(p):
        try:
            return Path(p).relative_to(BASE).as_posix()
        except ValueError:
            return str(p)

    # ---------- the audio stream ----------
    def _set_volume(self, v):
        self.audio_out.setVolume(v / 100)
        if self.through is not None:
            self.through.gain = v / 100

    def _through_sync_changed(self, ms):
        if self.through is not None:
            self.through.delay_ms = ms

    def _refresh_through_devices(self):
        """Hardware devices pedalboard can open, the grabber preselected."""
        apis = audio_apis()
        if [self.through_api.itemText(i) for i in range(self.through_api.count())] != apis:
            want = self.through_api.currentText() or self.settings.value("audio/api", "") or (apis[0] if apis else "")
            self.through_api.blockSignals(True)
            self.through_api.clear()
            self.through_api.addItems(apis)
            if self.through_api.findText(want) >= 0:
                self.through_api.setCurrentText(want)
            self.through_api.blockSignals(False)
        api = self.through_api.currentText() or None
        self.through_excl.setVisible(sys.platform == "win32" and api != "ASIO")
        ins, outs = audio_devices(api)
        saved = {self.through_in: self.settings.value("audio/in", None),
                 self.through_out: self.settings.value("audio/out", None)}
        for combo, items, prefer in ((self.through_in, ins, ("MS210x", "USB Video", "Grabber")),
                                     (self.through_out, outs, ("pipewire", "default"))):
            cur = combo.currentData()
            if cur is None and saved[combo]:
                cur = saved[combo]
            combo.blockSignals(True)
            combo.clear()
            if combo is self.through_in:
                combo.addItem("(none: output only)", "")   # songs and sections only
                if cur == "":
                    cur = None
            for short, full in items:
                combo.addItem(short, full)
            i = combo.findData(cur) if cur else -1
            if i < 0:
                for k, (short, _) in enumerate(items):
                    if any(p in short for p in prefer):
                        i = k + (1 if combo is self.through_in else 0)
                        break
            if i < 0 and combo is self.through_in and items:
                i = 1                                  # a real input if there is one
            combo.setCurrentIndex(max(0, i))
            combo.blockSignals(False)

    def _ping_start(self):
        """The round-trip test: needs the stream open with a live input."""
        la = self.through
        if la is None or not la.running:
            self.rt_lbl.setText("the audio output is not open")
            return
        if not la.in_name:
            self.rt_lbl.setText("tick the live input first, with a cable from an output "
                                "into it")
            return
        self.rt_lbl.setText("measuring...")
        self.rt_btn.setEnabled(False)
        la.ping()
        self._ping_polls = 0
        QTimer.singleShot(200, self._ping_poll)

    def _ping_poll(self):
        la = self.through
        r = la.ping_result if la is not None else None
        self._ping_polls += 1
        if r is None and self._ping_polls < 20 and la is not None and la.running:
            QTimer.singleShot(200, self._ping_poll)
            return
        self.rt_btn.setEnabled(True)
        if r is None:
            self.rt_lbl.setText("no result (stream stopped?)")
            return
        self.rt_lbl.setText(r["msg"])
        self.log.appendPlainText(f"audio: round-trip test: {r['msg']}")

    def _toggle_through(self, on):
        """A: the live input joins the stream, or leaves it."""
        self._reopen_stream()

    def _api_changed(self, *_):
        """Another driver model: the device lists change with it."""
        self.settings.setValue("audio/api", self.through_api.currentText())
        self.through_in.blockSignals(True)
        self.through_out.blockSignals(True)
        self.through_in.clear()
        self.through_out.clear()
        self.through_in.blockSignals(False)
        self.through_out.blockSignals(False)
        self._refresh_through_devices()
        self._io_changed()

    def _io_changed(self, *_):
        if getattr(self, "_io_ready", False):
            self._reopen_stream()

    def _reopen_stream(self):
        """Open, or reopen, the app's audio stream on the chosen output, with
        the live input when ticked. Every sound the app makes goes through
        it (racks, stems, F1-F4); while it is not running, songs fall back to
        Qt's player. The choices are remembered."""
        out_name = self.through_out.currentData()
        in_name = self.through_in.currentData() if self.through_on.isChecked() else None
        self.settings.setValue("audio/out", out_name or "")
        self.settings.setValue("audio/in", self.through_in.currentData() or "")
        self.settings.setValue("audio/block", self.through_block.currentText())
        self.settings.setValue("audio/sr", self.through_sr.currentText())
        self.settings.setValue("audio/ring", int(self.through_ring.currentData() or 0))
        self.settings.setValue("audio/live_in", self.through_on.isChecked())
        self.settings.setValue("audio/exclusive", self.through_excl.isChecked())
        if self.through is not None:
            if self.player.streaming and self.player.playbackState() != QMediaPlayer.StoppedState:
                self.player.stop()
                self.log.appendPlainText("audio: stream reopening, the song stopped")
            self._through_timer.stop()
            self.through.stop()
            self.through = None
        if not out_name:
            why = audio_status() or "no output device chosen"
            self.through_status.setText(f"not open: {why}. Songs play through Qt, no racks")
            if why != getattr(self, "_audio_why_logged", None):
                self._audio_why_logged = why
                self.log.appendPlainText(f"audio: {why}")
            self._update_afx_status()
            return
        self.through = AudioProxy(in_name or None, out_name, int(self.through_block.currentText()),
                                 sr=int(self.through_sr.currentText()),
                                 exclusive=self.through_excl.isChecked(),
                                 api=self.through_api.currentText() or None,
                                 ring=int(self.through_ring.currentData() or 0))
        self.through.gain = self.vol.value() / 100
        self.through.freeze.held = bool(self._freeze_sources)
        self.through.delay_ms = self.through_sync.value()
        self.through.ab = self.perf_ab.value() / 100.0
        self.through.hold_tape = self._webcam_b()
        for stem, cb in self.perf_stems.items():
            self.through.on[stem] = not cb.isChecked()   # the M: lit while out
        self._push_voices(self.through)
        self._push_afx()
        self.through.inserts = self.inserts
        self.through.pro_audio = self.audio_pro.isChecked()
        self._midi_out_apply()
        self.through.start()
        self.through_status.setText("opening...")
        self._through_ticks = 0
        self._through_error = None
        self._through_timer.start(20)
        src = self.through_in.currentText() if in_name else "output only"
        ring = f", ring of {self.through.ring}" if self.through.ring else ""
        self.log.appendPlainText(f"audio: {src} -> {self.through_out.currentText()}, "
                                 f"{self.through.block} frames at {self.through.sr} Hz{ring}")
        if self.live_on.isChecked():                 # the analysis follows the input
            if in_name:
                self.live.start_external(self.through_in.currentText(), self.through.sr)
            elif self.live.external:
                err = self.live.start(self.live_device.currentData())
                if err:
                    self.log.appendPlainText(f"live: {err}")
        QTimer.singleShot(2500, self._update_afx_status)

    def _poll_through(self):
        la = self.through
        if la is None:
            return
        if not la.running:
            if la.error and la.error != self._through_error:
                self._through_error = la.error
                self.log.appendPlainText(f"audio: {la.error}; songs play through Qt, "
                                         "no racks, until the output can be opened")
                self.through_status.setText(f"not open: {la.error}")
                self._through_timer.stop()
                self._update_afx_status()
            return
        if self._live_active() and self.live.external:
            for block, t in la.drain():
                self.live.feed(block, t)
        # the racks' delays follow the playing song, else the live clock
        song_on = self.player.streaming and self.player.playbackState() == QMediaPlayer.PlayingState
        la.beat_s = (self._beat_ms if song_on else self.tempo.beat_ms) / 1000.0
        la.metro.beat_s = self.tempo.beat_ms / 1000.0     # the metronome: the live clock
        la.metro.anchor = self.tempo.anchor
        self._through_ticks += 1
        if self._through_ticks % 10 == 0:
            s = la.stats()
            db = lambda p: f"{20 * math.log10(max(p, 1e-5)):.0f} dB"
            src = f"in {db(s['in_peak'])}" if la.in_name else "output only"
            mode = f"{s['mode']}  ·  " if s.get("mode") else ""
            text = (f"{mode}{src}  out {db(s['out_peak'])}  ·  dsp {s['load']:.0%} "
                    f"peak {s['peak']:.0%}  ·  "
                    f"buffers {s['buffer_ms']:.1f} ms  ·  sync {s['delay_ms']:.0f} ms  ·  "
                    f"overflows {s['dropped']}  underruns {s['underruns']}  late {s['late']}  "
                    f"gap {s['gap_ms']:.1f} ms")
            src = self.backdrop.live_source()
            lat = src.capture_latency_ms if src is not None else None
            if lat is not None:
                # what the picture is already late by before paint and display
                text += f"  ·  picture {lat:.0f} ms + paint + display"
            self.through_status.setText(text)

    def _update_live_meter(self):
        self.live_level.setValue(int(100 * self.live.loud))
        an = self.live.analyzer
        if self._link_active():
            root = NOTE_NAMES[an.root] if an is not None and an.root >= 0 else "-"
            self.live_status.setText(f"{self.tempo.bpm:.1f} bpm from link, root {root}")
        elif an is not None and self.live_auto.isChecked():
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
                and an.beat_time != self._last_beat_sync \
                and not self._link_active():          # Link owns tempo and beat then
            self._last_beat_sync = an.beat_time
            self.tempo.align(an.bpm, an.beat_time)
            self._show_live_bpm()
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
        self._drop_clip(t)
        self.lfo.trigger(t)

    def _drop_clip(self, t):
        """The picture's part of a drop at monotonic time t: a new clip once
        4 bars have passed since the last switch, else a jump."""
        bars_since = (t - self._live_switch_t) / (4 * self.tempo.beat_ms / 1000)
        if self.video_switch.isChecked() and len(self._song_videos()) > 1 \
                and bars_since >= 4:
            self.backdrop.pick_random()
            self._live_switch_t = t
        elif self.video_jump.isChecked():
            self.backdrop.jump()

    def _live_bpm_edited(self, bpm):
        self.tempo.set_bpm(bpm)
        self._link_push(bpm)

    def _song_bpm(self):
        """The clock's tempo for a render: whole when it is (clean file
        names), else to a tenth."""
        bpm = round(self.tempo.bpm, 1)
        return int(bpm) if bpm == int(bpm) else bpm

    def _show_live_bpm(self):
        """The bpm box shows the clock, without that counting as an edit."""
        for box in (self.live_bpm, self.perf_bpm):
            box.blockSignals(True)
            box.setValue(self.tempo.bpm)
            box.blockSignals(False)

    def tap_tempo(self):
        """T: one tap on the beat. The first tap of a series is the downbeat,
        the intervals set the tempo (shown in the spinbox)."""
        if self.live_auto.isChecked():
            self.live_auto.setChecked(False)
            self.log.appendPlainText("live: auto off, following your taps")
        n = self.tempo.tap()
        if n >= 2:
            self._show_live_bpm()
            self._link_push(self.tempo.bpm)

    def live_downbeat(self):
        """D: the bar starts now, tempo unchanged; the Erae's loop with it."""
        self.tempo.downbeat()
        self._erae_start(at_next_bar=False)

    # ------------------------------------------------------------ Ableton Link
    def _link_active(self):
        return self.link is not None and self.link_on.isChecked()

    def _link_press(self, *_):
        """The BPM row's play / stop button (and its MIDI row): the session's
        transport is flipped, and the poll then treats it exactly as a
        peer's press, so the section here follows the same way."""
        if not self._link_active():
            self.log.appendPlainText("link: off, switch it on first (the Link button)")
            return
        on = not self.link.playing
        self.link.playing = on
        self.link_play.setText("■" if on else "▶")
        self.log.appendPlainText(f"link: {'play' if on else 'stop'} sent to the session")

    def _link_toggled(self, on):
        """The Settings tab's Link button: join or leave the Link session. Alone
        our tempo is the session's; joining peers, theirs comes in (Link's
        rule: the session that is already there wins) and shows in the bpm
        box within a poll."""
        if self.link is None:
            return
        self.settings.setValue("clock/link", on)
        self.link_on.setText("Link on" if on else "Link off")
        if on:
            self.link.tempo = self.tempo.bpm
            self.link.enabled = True
            self._link_ticks = 0
            self._link_playing = self.link.playing    # take the session's transport as is
            self._link_timer.start(50)
            self.link_status.setText("joining...")
            self.log.appendPlainText(f"link: on, {self.tempo.bpm:.1f} bpm, looking for peers")
        else:
            self._link_timer.stop()
            self.link.enabled = False
            self.link_status.setText("off")
            self.link_play.setText("▶")
            self.log.appendPlainText("link: off")

    def _link_push(self, bpm):
        """Our tempo changed (a tap, the box, a section): the session follows."""
        if self._link_active() and abs(self.link.tempo - bpm) > 0.01:
            self.link.tempo = bpm

    def _link_transport(self, on):
        """A section started or stopped here: the session plays or stops
        with it (Live from its own bar). Remembered as seen, so the poll
        does not take our own change for a peer's."""
        if self._link_active() and on != self._link_playing:
            self._link_playing = on
            self.link.playing = on
            self.log.appendPlainText(f"link: {'play' if on else 'stop'} sent to the peers")

    def _poll_link(self):
        """Every 50 ms while Link is on: the session's tempo sets the live
        clock (the bar position is kept: tempo only), unless a section
        plays, whose loop is the clock's master then (its tempo is pushed
        out on every swap instead). A peer's play starts the Perform tab's
        current section again, its stop stops it on the bar, as Shift+N
        does; songs are not touched. The status shows the peers."""
        if not self._link_active():
            return
        bpm, _beat, _phase = self.link.snapshot()
        la = self.through
        pl = la.player if la is not None and la.running else None
        if abs(bpm - self.tempo.bpm) > 0.01:            # a playing section follows (its rate)
            self.tempo.set_bpm(bpm)
            self._show_live_bpm()
        playing = self.link.playing
        if playing != self._link_playing:               # a peer pressed play or stop
            self._link_playing = playing
            if playing:
                self.perform_start("link")
            elif pl is not None and (pl.playing or pl.pending is not None):
                self.log.appendPlainText("link: stop, the section stops on the bar")
                self.perform_stop()
        self._link_ticks += 1
        if self._link_ticks % 4 == 0:
            peers = self.link.peers
            if peers != self._link_peers:
                self._link_peers = peers
                self.log.appendPlainText(f"link: {peers} peer{'s' if peers != 1 else ''}, {bpm:.1f} bpm")
        peers = max(0, self._link_peers)
        who = "no peers" if peers == 0 else f"{peers} peer{'s' if peers != 1 else ''}"
        # the session's bar, every poll: the beat 1..4 and a 4-cell meter (a
        # label's text, 20 times a second: nothing the machine notices)
        b = min(4, int(_phase) + 1)
        self.link_on.setText(f"Link {b}")               # the session's beat, on the button
        self.link_status.setText(f"{who} · {bpm:.1f} bpm · {'▶' if playing else '■'}")
        self.link_status.setToolTip(f"{who} in the Link session · {bpm:.1f} bpm · "
                                    f"{'playing' if playing else 'stopped'} · beat {b}")
        self.link_play.setText("■" if playing else "▶")   # what a press would send

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
        if source is None or self.backdrop.live_source() is not None \
                or not self._following():
            self._follow_cur = None
            return
        groups = self._follow_groups(phrase_key)
        # the drums' hold: in drums mode, and in varies while the drums lead;
        # everything keeps a hit as long as its sound, the layers between
        hold = FOLLOW_DRUM_HOLD_MS if groups and groups[0] == "drums"             and self.video_follow.currentText() != "everything" else 0.0
        ev = self._follow.pick(pos, groups, hold)
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
        self._song_visuals_pending = True    # clip pick + shader: when the sound starts
        self._outro = None
        if self.through is not None:
            self.through.song.set_events([])
        try:
            meta = json.loads(Path(wav).with_suffix(".json").read_text())
            if self.through is not None:     # its drum hits, for the MIDI out
                self.through.song.set_events(meta.get("events") or [])
            self._sections = meta["sections"]
            self._sec_bounds = [int(s["start_sec"] * 1000)
                                for s in self._sections]
            self._beat_ms = 60000.0 / meta["bpm"]
            self.tempo.set_bpm(meta["bpm"])            # the BPM box shows what plays
            self._show_live_bpm()
            self._link_push(self.tempo.bpm)
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

    def _roll_look(self, idx, phrase, shuffle=True):
        """Random look: which stylistic effects are on and how hard. Re-rolled
        every 4-bar phrase so long sections keep moving; section-level
        choices (kaleidoscope, mono tones, shader) change per section.
        shuffle=False (idle) keeps every effect at its knob's strength."""
        key = (idx, phrase)
        if key == self._look_idx:
            return
        section_changed = self._look_idx is None or self._look_idx[0] != idx
        self._look_idx = key
        if shuffle and self.shuffle_look.isChecked():
            self._look = {k: (0.0 if random.random() < 0.5
                              else random.uniform(0.4, 1.3))
                          for k, *_ in VIDEO_EFFECTS if k not in LOOK_EXEMPT}
        else:
            self._look = {}
        if not section_changed:
            return
        # kaleidoscope: strength is the chance a section gets it
        kal = self.vfx["kaleido"].value() if self.vfx_on["kaleido"].isChecked() else 0.0
        self._kaleido = random.choice([4, 6, 8]) if random.random() < kal else 0
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
        song's, or in live mode the tap-tempo clock with the input's loudness.
        With nothing playing (idle) the ticked effects of the Video FX tab
        still run, on the tap-tempo clock at a steady mid loudness: no random
        look, so what is ticked is what shows, and the clip is left alone.
        A perform section sounding without the live input counts like live
        for the clip: it swaps on the 8-bar pseudo sections the way a song
        does on quiet boundaries."""
        live = self._live_active()
        idle = not live and self.player.playbackState() != QMediaPlayer.PlayingState
        section = idle and self._perf_sounding
        if idle and not section and not self._freeze_sources                 and not any(cb.isChecked() for cb in self.vfx_on.values()):
            return None                         # the clip as it is (a freeze holds it)
        head_speed = None                       # a file tape's speed under the section's head
        if live or idle:
            now = time.monotonic()
            if live and self.live_auto.isChecked():
                self._follow_analyzer(now)
            pos = self.tempo.pos_ms(now)
            fade = 1.0
            beat_ms = self.tempo.beat_ms
            if live:
                env = self.live.loud
            else:                               # idle: a pulse on every beat
                env = 0.35 + 0.65 * (1.0 - (pos % beat_ms) / beat_ms) ** 2
            # the section's head (a gear, a tape stop) takes the picture
            # with it: a file on the tape side runs at its speed (the live
            # input cannot), the effects' clock slows and stops with it
            pl = self._player()
            hr = 1.0
            if pl is not None and pl.playing and pl.head_state()["mode"] is not None:
                hr = float(pl.head_state()["rate"])
            self.backdrop.time_rate = hr
            src = getattr(self.backdrop, "source", None)
            if src is not None and isinstance(getattr(src, "current", None), Path):
                head_on = hr != 1.0 or self._head_phase() is not None
                if head_on != self._head_tape:
                    self._head_tape = head_on
                    src.catch_up = not head_on      # slowed on purpose: no rewind after
                if head_on:
                    head_speed = hr
        else:
            pos = self._playhead_ms()
            fade = self._outro_fade(pos)
            beat_ms = self._beat_ms
            env = 0.5
            if self._env is not None and len(self._env):
                env = float(self._env[min(len(self._env) - 1, int(pos // 50))])
        v = {k: (sp.value() if self.vfx_on[k].isChecked() else 0.0)
             for k, sp in self.vfx.items()}
        idx, sec, kind = 0, None, "groove"
        if self._perf_playing is not None and getattr(self._perf_playing, "kind", "") \
                in EXTRA_KINDS:
            kind = self._perf_playing.kind              # the shaders' iSection: intro, outro
        sec_start_ms = 0
        if live or idle:
            # no section map: every 8 bars count as a section, so the look,
            # the shader and the clip keep changing the way they do in a song
            sec_len = self.LIVE_SECTION_BARS * 4 * beat_ms
            idx = int(pos // sec_len)
            sec_start_ms = idx * sec_len
            if idx != self._sec_idx:
                self._sec_idx = idx
                if live or section:             # idle: the clip is yours
                    self._on_live_section(idx, pos)
        elif self._sections and self._sec_bounds:
            idx = max(0, bisect.bisect_right(self._sec_bounds, pos) - 1)
            sec = self._sections[idx]
            kind = sec["kind"]
            sec_start_ms = self._sec_bounds[idx]
        # bars counted from the section's own first bar, so phrases line up
        bar_i = int(max(0, pos - sec_start_ms) // (4 * beat_ms))
        self._roll_look(idx, bar_i // 4, shuffle=not idle)
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
        elif idle:
            root = NOTE_NAMES[(idx * 7) % 12]   # no music: round the circle of fifths
        elif sec is not None and sec.get("transition"):
            drop = max(0.0, 1 - (pos - self._sec_bounds[idx]) / 400.0)
        # a section playing with the pedal on its side: its root (the sub
        # sits on it) rather than the tape's
        ps = self._perf_playing
        if ps is not None and getattr(ps, "root", None) and self.through is not None                 and self.through.ab >= 0.5:
            root = ps.root
        st = {"music": {
            "loud": env, "beat": (pos % beat_ms) / beat_ms,
            "bar": (pos % (4 * beat_ms)) / (4 * beat_ms),
            "bpm": 60000.0 / beat_ms, "section": sec_id,
            "root": int(round(NOTE_HUES[root] * 12)) if root in NOTE_HUES else -1,
            "drop": drop,
            # live: no song to be a fraction of; a slow 5-minute loop keeps
            # the shaders that age with it (vhs) moving
            "song_pos": (pos / 300000.0) % 1.0 if live or idle
            else pos / total_ms if total_ms else 0.0}}
        ph = self._head_phase()                 # a gear or a tape stop: the beat is the head's
        if ph is not None:
            st["music"]["beat"], st["music"]["bar"] = ph
        if head_speed is not None:
            st["speed"] = head_speed
        # clean video: rolled once per bar, on its downbeat, with probability
        # = strength; the first 1/16 note of the bar then flashes the video as
        # it is (no effects, no shader layer), only the outro fade
        # still applies
        bar_key = (idx, bar_i)
        if bar_key != self._clean_bar:
            self._clean_bar = bar_key
            self._clean_on = random.random() < v["clean"]
        in_bar = (pos - sec_start_ms) - bar_i * 4 * beat_ms
        if not (live or idle):
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
        if v["gran"]:
            # the granular picture: the frame in grains the size of the
            # sound's, on the row's settings, lit by the loudest cloud
            # sounding (alone at the knob's strength with none), its clock
            # one grain per grain length, the frame held while frozen
            gs = self._gran_state()
            fz = self._freeze_state() if gs is None else None   # the master freeze's grains
            size = max(0.005, FRAGMENTS["size"] * 4 * beat_ms / 1000.0)
            tnow = time.monotonic()
            if self._gran_t is not None:
                self._gran_clock += max(0.0, min(0.2, tnow - self._gran_t)) / size
            self._gran_t = tnow
            loud = gs or fz
            amt = v["gran"] * ((0.6 + 0.4 * loud["level"]) if loud else 0.7)
            frozen = gs is not None and gs["speed"] == 0.0
            st["gran"] = {"amt": min(1.0, amt), "freeze": frozen or fz is not None,
                          "heads": self._freeze_heads(fz) if fz else self._gran_heads(gs, size)}
            if frozen:
                st["freeze"] = True
        if self._freeze_sources:                     # the freeze: the picture on its frame
            st["freeze"] = True
        if self.video_plain.isChecked() or (self._clean_on and in_bar < beat_ms / 4):
            st["clean"] = True
            if fade < 1.0:
                st["brightness"] = fade
            st["music"]["swell"] = 0.0
            return st
        if v["pump"]:
            st["brightness"] = 0.7 + 0.7 * v["pump"] * env
        if live or idle:
            # the song-side colour moves, on the live clock's pseudo-sections
            if v["color"]:
                st["hue"] = v["color"] * ((idx * 0.37) % 1.0) * 0.5
                st["saturation"] = 1 + 0.5 * v["color"]
            if lk("tint") and root in NOTE_HUES:
                st["tint"] = (NOTE_HUES[root], 0.6 * lk("tint"))
            if v["flash"] and drop_age is not None and drop_age < 120 * v["flash"]:
                st["invert"] = True            # the drop hits
            if v["pixel"]:                     # no breaks here: the break's rhythm
                r = random.Random(hash((idx, bar_i // 2, self._pix_seed)))   # on every section
                if r.random() < 0.66:
                    size = (2 + 10 * v["pixel"]) * r.uniform(0.5, 1.5)
                    st["pixelate"] = max(2, int(size * (0.7 + 0.6 * env)))
        if (live or idle) and v["flash"]:
            # a live break: the song's drop visuals, timed by the section
            # player itself (beat-repeat stutter, frozen gap, swell to white,
            # inverted on the drop)
            pl = self.through.player if (self.through is not None
                                         and self.through.running) else None
            fs = pl.fill_state() if pl is not None else None
            if fs is not None:
                if fs["gap"]:
                    st["freeze"] = True
                else:
                    st["repeat"] = (fs["k"], fs["frac"])
                if fs["swell"] > 0.0:
                    st["swell"] = fs["swell"]
                    st["zoom"] = st.get("zoom", 0.0) + 0.25 * fs["swell"]
            else:
                rs = pl.repeat_state() if pl is not None else None
                if rs is not None:                  # a held repeat: the same stutter
                    st["repeat"] = (rs["k"], rs["frac"])
            if self._perf_drop_t is not None \
                    and (now - self._perf_drop_t) * 1000.0 < 120 * v["flash"]:
                st["invert"] = True                 # the drop hits
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
        if state == QMediaPlayer.PlayingState and getattr(self, "_song_visuals_pending", False):
            # the sound has started: now the picture changes with it (the
            # stream reports playing only once its stems are loaded)
            self._song_visuals_pending = False
            self._pick_song_video()
            self._shader_on_for_song()
        if hasattr(self, "afx_status"):
            self._update_afx_status()
        if state == QMediaPlayer.PlayingState and not self.player.streaming \
                and any(sp.value() != (1.0 if k == "vol" else 0.0) for (_c, k), sp in self.afx.items()):
            self.log.appendPlainText("audio fx: the song plays through Qt, the racks do not "
                                     "apply; the audio output is not open (Settings tab)")
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
            if t == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                mods = event.modifiers()
                if (mods == Qt.ControlModifier and self._learn_click(obj)) \
                        or (mods == Qt.AltModifier and self._learn_clear_click(obj)):
                    return True             # the control keeps its value
            if t == QEvent.KeyPress and event.key() == Qt.Key_Escape \
                    and self._fullscreen and not event.isAutoRepeat():
                self.toggle_fullscreen()
            if t == QEvent.KeyPress:
                self._hold_key(event, True)
            if not self.ui.isVisible():
                self._show_ui()
            self._arm_hide()
        elif t == QEvent.KeyRelease and isinstance(obj, QWidget):
            self._hold_key(event, False)
        return super().eventFilter(obj, event)

    def _hold_key(self, event, down):
        """A held action's key (HOLD) going down or up: the press and the
        release both reach its handler, auto-repeats and typing in a text
        field (where the shortcuts do not fire either) do not."""
        if event.isAutoRepeat() or not self._hold_keys:
            return
        ident = self._hold_keys.get(QKeySequence(event.keyCombination().toCombined()).toString())
        if ident is None:
            return
        if down and isinstance(QApplication.focusWidget(),
                               (QLineEdit, QAbstractSpinBox, QPlainTextEdit)):
            return
        self.holds[ident]("key", down)

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
        self._closing = True
        self._lfo_timer.stop()
        self.lfo.restore_all()                  # the knobs as you set them, into the settings
        self._live_save(force=True)
        self.inserts.close_editor()             # a plugin editor still open goes too
        self._show_ui()                         # never leave the cursor hidden
        self.player.stop()
        self.live.stop()
        if self.takes.recording:
            self.takes.stop()
        self._flush_takes()                     # the takes reach the disk now, not during the show
        if self.through is not None:
            self.through.stop()
        self._vst_save()
        self._vst_watch_stop.set()
        audio_host().shutdown()                 # the engine process, once the chain is saved
        if getattr(self, "midi", None) is not None:
            self.midi.close_all()
        if getattr(self, "midi_out", None) is not None:
            self.midi_out.close()
        if getattr(self, "erae", None) is not None:
            self.erae.close()
        if self.link is not None:
            self.link.close()
        for src in (getattr(self.backdrop, "source", None),
                    getattr(self.backdrop, "source2", None)):
            if src is not None:
                src.stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        if getattr(self, "fetch_worker", None) is not None:
            self.fetch_worker.stop()
        event.accept()
        if getattr(self, "_wipe_settings", False):
            # once the event loop is done: the timers that write settings
            # (the port scans, the audio status) have fired their last by then
            QApplication.instance().aboutToQuit.connect(self._wipe_and_restart)

    def _wipe_and_restart(self):
        self.settings.clear()
        self.settings.sync()
        subprocess.Popen([sys.executable, *sys.argv], cwd=str(Path.cwd()),
                         close_fds=True)                  # a fresh Gacha with the defaults


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    win.showMaximized()                         # the whole screen from the start
    sys.exit(app.exec())
