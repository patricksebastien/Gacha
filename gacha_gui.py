#!/usr/bin/env python3
"""
gacha_gui.py — Qt front-end for gacha.py.

Left side: all generation parameters + live render log.
Right side: two tabs — past renders (output/) and the raw samples
(samples/) — double-click anything to listen. New renders start
playing automatically when generation finishes.

Usage:  python3 gacha_gui.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from PySide6.QtCore import Qt, QRectF, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QPlainTextEdit, QPushButton, QSlider, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

BASE = Path(__file__).parent
GACHA = BASE / "gacha.py"
SAMPLES_DIR = BASE / "samples"
OUT_DIR = BASE / "output"
BG_VIDEO = BASE / "gacha.mp4"

STYLE = """
QWidget { color: #f2e9e5; font-size: 13px;
          font-family: "Noto Sans Mono", "Ubuntu Mono", monospace; }
QGroupBox {
    background: rgba(28, 11, 9, 130); border: 1px solid rgba(255,255,255,40);
    border-radius: 10px; margin-top: 0px; padding-top: 0px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QListWidget, QPlainTextEdit {
    background: rgba(28, 11, 9, 130); border: 1px solid rgba(255,255,255,35);
    border-radius: 8px;
}
QTabWidget::pane {
    background: rgba(28, 11, 9, 100); border: 1px solid rgba(255,255,255,35);
    border-radius: 8px;
}
QTabBar::tab {
    background: rgba(28, 11, 9, 110); padding: 6px 16px;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}
QTabBar::tab:selected { background: rgba(188, 38, 26, 190); }
QPushButton {
    background: rgba(134, 28, 20, 160); border: 1px solid rgba(255,255,255,50);
    border-radius: 8px; padding: 6px 14px;
}
QPushButton:hover { background: rgba(212, 48, 30, 210); }
QPushButton:disabled { color: rgba(242,233,229,110); }
QSpinBox, QDoubleSpinBox, QComboBox {
    background: rgba(28, 11, 9, 130); border: 1px solid rgba(255,255,255,40);
    border-radius: 6px; padding: 2px 8px;
}
QComboBox QAbstractItemView { background: rgb(42, 18, 14); }
QLabel, QCheckBox { background: transparent; }
QSplitter::handle { background: transparent; }
"""


class VideoBackdrop(QWidget):
    """Paints a muted, looping video behind its child widget, under a scrim."""

    def __init__(self, child):
        super().__init__()
        self._image = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(child)
        if BG_VIDEO.exists():
            self._sink = QVideoSink()
            self._sink.videoFrameChanged.connect(self._on_frame)
            self._vplayer = QMediaPlayer()      # no audio output = silent
            self._vplayer.setVideoSink(self._sink)
            self._vplayer.setSource(QUrl.fromLocalFile(str(BG_VIDEO)))
            self._vplayer.setLoops(QMediaPlayer.Loops.Infinite)
            self._vplayer.play()

    def _on_frame(self, frame):
        if frame.isValid():
            self._image = frame.toImage()
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
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
            p.drawImage(QRectF(self.rect()), img, src)
        p.fillRect(self.rect(), QColor(12, 7, 5, 55))   # legibility scrim


class RenderWorker(QThread):
    """Runs gacha.py in a subprocess and streams its log lines."""
    line = Signal(str)
    finished_ok = Signal(bool)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(GACHA)] + self.args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for ln in proc.stdout:
                self.line.emit(ln.rstrip())
            self.finished_ok.emit(proc.wait() == 0)
        except Exception as e:
            self.line.emit(f"error: {e}")
            self.finished_ok.emit(False)


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

        # ---------- parameters panel ----------
        form = QFormLayout()

        self.count = QSpinBox(minimum=1, maximum=50, value=1)
        form.addRow("Number of outputs", self.count)

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

        self.drum_style = QComboBox()
        self.drum_style.addItems(["random", "four-floor", "breakbeat",
                                  "boom-bap", "halftime", "dnb", "minimal",
                                  "ukg", "dembow", "one-drop", "footwork",
                                  "clave", "idm"])
        form.addRow("Drum style", self.drum_style)

        self.intro_style = QComboBox()
        self.intro_style.addItems(["ambient", "sparse", "none", "build",
                                   "drums-first", "reverse-swell", "collage",
                                   "filtered"])
        form.addRow("Intro style", self.intro_style)

        self.intro_cap = QCheckBox("cap at")
        self.intro_bars = QSpinBox(minimum=1, maximum=32, value=4,
                                   suffix=" bars")
        self.intro_bars.setEnabled(False)
        self.intro_cap.toggled.connect(
            lambda on: self.intro_bars.setEnabled(on))
        form.addRow("Intro length", self._pair(self.intro_bars, self.intro_cap))

        self.pan_drums = self._pan_spin(0.3)
        self.pan_layers = self._pan_spin(0.6)
        self.pan_events = self._pan_spin(0.5)
        form.addRow("Pan drums", self.pan_drums)
        form.addRow("Pan layers", self.pan_layers)
        form.addRow("Pan events", self.pan_events)

        params_box = QGroupBox()
        params_box.setLayout(form)

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
        self.samples_list = QListWidget()
        self.samples_list.itemDoubleClicked.connect(self._play_item)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_lists)

        tabs = QTabWidget()
        tabs.addTab(self.outputs_list, "Outputs")
        tabs.addTab(self.samples_list, "Samples")
        tabs.addTab(self._build_mixer(), "Mixer")

        # ---------- player bar ----------
        self.now_playing = QLabel("—")
        self.now_playing.setStyleSheet("font-weight: bold;")
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.player.stop)
        self.pos_slider = QSlider(Qt.Horizontal)
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
        split.setSizes([450, 600])
        self.setStyleSheet(STYLE)
        self.setCentralWidget(VideoBackdrop(split))

        self.refresh_lists()

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
    def _pan_spin(val):
        s = QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.1)
        s.setValue(val)
        return s

    # ---------- generation ----------
    def generate(self):
        args = ["--count", str(self.count.value()),
                "--duration", str(self.duration.value()),
                "--intro-style", self.intro_style.currentText(),
                "--drum-style", self.drum_style.currentText(),
                "--pan-drums", str(self.pan_drums.value()),
                "--pan-layers", str(self.pan_layers.value()),
                "--pan-events", str(self.pan_events.value())]
        if not self.bpm_random.isChecked():
            args += ["--bpm", str(self.bpm.value())]
        if not self.seed_random.isChecked():
            args += ["--seed", str(self.seed.value())]
        if not self.all_samples.isChecked():
            args += ["--num-samples", str(self.num_samples.value())]
        if self.intro_cap.isChecked():
            args += ["--intro-bars", str(self.intro_bars.value())]

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Rendering…")
        self.log.appendPlainText(f"$ gacha.py {' '.join(args)}")
        self.worker = RenderWorker(args)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.finished_ok.connect(self._render_done)
        self.worker.start()

    def _render_done(self, ok):
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate")
        self.refresh_lists()
        if ok and self.outputs_list.count():
            newest = self.outputs_list.item(0)
            self.outputs_list.setCurrentItem(newest)
            self._play_item(newest)          # listen right away
        elif not ok:
            self.log.appendPlainText("✗ render failed")

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

        self.samples_list.clear()
        exts = ("*.wav", "*.flac", "*.aif", "*.aiff", "*.mp3")
        sfiles = sorted(p for ext in exts for p in SAMPLES_DIR.glob(ext))
        for f in sfiles:
            item = QListWidgetItem(f.name)
            item.setData(Qt.UserRole, str(f))
            self.samples_list.addItem(item)

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

    def _play_item(self, item):
        path = item.data(Qt.UserRole)
        self._segment = None
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
                        or self.samples_list.currentItem()
                        or (self.outputs_list.count()
                            and self.outputs_list.item(0)))
                if item:
                    self._play_item(item)
                    return
            self.player.play()

    def _on_state(self, state):
        self.play_btn.setText(
            "Pause" if state == QMediaPlayer.PlayingState else "Play")

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
        self.player.stop()
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec())
