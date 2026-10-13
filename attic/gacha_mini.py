#!/usr/bin/env python3
"""
attic/gacha_mini.py — the audio stream alone, in a small Qt window: does a
bare callback crackle on this machine, and what makes it start? Kept for
reference (2026-10-13): the hunt it served is over, the crackle was the
GUI holding the interpreter lock and the engine in its own process
(gacha_audioproc.py) settled it. Still runs from here: python attic/gacha_mini.py

The main app is 10k lines of Qt, video, MIDI and DSP round one ASIO
callback; when it crackles, this is the control: the same stream opened
the same way (gacha_audio's device lookup, host API, block size, switch
interval, power hints), with nothing else in the process. The callback
plays silence, a tone, the input through, or runs the real engine's block
(gacha_audio.LiveAudio._dsp_block, nothing playing, no racks), and each
of the things suspected of crackling can be switched on one at a time:

  fake load      the callback spins for a share of the block, as if the
                 DSP cost that much (many short numpy calls, the GIL let
                 go between them, like the real block)
  GUI burn       a thread running pure Python, like the main window at
                 its busiest
  paint burn     a widget repainting a fresh 640x480 image 60 times a
                 second, like the video
  switch         the interpreter's switch interval (5 ms default, the
                 engine runs at 0.5 ms)
  gc off         gc.disable(), in case a collection is the stall
  Pro Audio      the driver's thread in the MMCSS "Pro Audio" class
  high priority  the whole process in HIGH_PRIORITY_CLASS, as DAWs run
  core           the driver's thread pinned: any core, the P-cores only,
                 or one P-core (a hybrid CPU parks threads on E-cores)

The stats also say which CPUs the callback actually ran on, P or E.

What it shows, ten times a second: the DSP time per block (smoothed and
the worst lately), the gap between callbacks (the one number that is a
glitch on ASIO: a block more than two blocks after the last one came
late, and the driver has already played the hole), the driver's own
underrun flags, PortAudio's CPU load, the GIL watcher's stalls
(gacha_gilwatch: which thread held it), and a trace of the last 600
blocks' gaps and durations. Late blocks and GIL stalls go to the log
with the time, so the two can be lined up.

    python gacha_mini.py
"""
import ctypes
import gc
import math
import os
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo, for gacha_audio
import gacha_audio as ga                                   # noqa: E402
from gacha_gilwatch import GilWatch                        # noqa: E402

from PySide6.QtCore import Qt, QSettings, QTimer           # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen   # noqa: E402
from PySide6.QtWidgets import (                            # noqa: E402
    QApplication, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QPlainTextEdit, QPushButton, QSlider, QVBoxLayout, QWidget)

sd = ga.sd
SOURCES = ("silence", "tone 220 Hz", "input through", "engine idle (gacha_audio block)")
SWITCHES = (("5 ms (Python default)", 0.005), ("1 ms", 0.001),
            ("0.5 ms (the engine)", 0.0005), ("0.1 ms", 0.0001))
BLOCKS = ("64", "128", "256", "512", "driver's own")
CORES = ("any core", "P-cores only", "one P-core")
RING = 4096
_K32 = ctypes.windll.kernel32 if sys.platform == "win32" else None


def cpu_topology():
    """Windows: {efficiency class: mask of logical CPUs} from
    GetLogicalProcessorInformationEx; the highest class is the P-cores.
    {} when it cannot be read (one processor group assumed)."""
    if _K32 is None:
        return {}
    try:
        n = ctypes.c_ulong(0)
        _K32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(n))   # RelationProcessorCore
        buf = ctypes.create_string_buffer(n.value)
        if not _K32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(n)):
            return {}
        raw, off, classes = buf.raw, 0, {}
        while off < n.value:
            rel, size = struct.unpack_from("<II", raw, off)
            if rel == 0:
                eff = raw[off + 9]                                 # EfficiencyClass
                groups, = struct.unpack_from("<H", raw, off + 30)
                for g in range(groups):
                    mask, grp = struct.unpack_from("<QH", raw, off + 32 + 16 * g)
                    if grp == 0:
                        classes[eff] = classes.get(eff, 0) | mask
            off += size
        return classes
    except Exception:
        return {}


def _mask_text(mask):
    cpus = [i for i in range(64) if mask >> i & 1]
    if not cpus:
        return "-"
    runs, start = [], cpus[0]
    for a, b in zip(cpus, cpus[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}-{a}" if a != start else str(start))
            start = b
    return ",".join(runs)


def set_process_priority(high):
    """HIGH_PRIORITY_CLASS for the process (what DAWs run at), or normal."""
    if _K32 is None:
        return False
    try:
        _K32.GetCurrentProcess.restype = ctypes.c_void_p
        _K32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        return bool(_K32.SetPriorityClass(_K32.GetCurrentProcess(), 0x80 if high else 0x20))
    except Exception:
        return False


class Probe:
    """One stream and its callback; every attribute is read from the GUI
    thread and written here or on the driver's thread (plain numbers)."""

    def __init__(self, in_name, out_name, block, api, exclusive, source):
        self.in_name, self.out_name = (in_name or None), out_name
        self.block = int(block)                 # 0: the driver's size
        self.api, self.exclusive = api, bool(exclusive)
        self.sr = ga.SR
        self.source = source                    # index into SOURCES, changed live
        self.fake_load = 0.0                    # share of the block to spin, 0..1.5
        self.pro_audio = False                  # MMCSS on the driver's thread, once
        self.affinity = 0                       # CPU mask for the driver's thread, 0 = any
        self._aff_done = None
        self.cpu_hist = np.zeros(64, np.int64)  # which CPU each callback ran on
        self.mode = ""
        self.error = None
        self.running = False
        self.stream = None
        self.engine = None                      # LiveAudio, for the engine source
        self._phase = 0.0
        self._pro_done = False
        self._old_switch = None
        # stats
        self.frames = 0                         # the driver's buffer, as seen
        self.rounds = 0
        self.load = 0.0                         # dsp time / block time, smoothed
        self.peak = 0.0                         # the worst lately (decays)
        self.worst_dsp = 0.0                    # the worst ever, as a ratio
        self.late = 0                           # gaps over 2 blocks
        self.near = 0                           # gaps over 1.5 blocks
        self.gap_ms = 0.0                       # the longest gap lately (decays)
        self.worst_gap = 0.0                    # ratio, ever
        self.underruns = self.overflows = 0
        self.out_peak = self.in_peak = 0.0
        self.out_latency_ms = self.buffer_ms = 0.0
        self.gaps = np.zeros(RING, np.float32)  # gap / block, per callback
        self.durs = np.zeros(RING, np.float32)  # dsp / block, per callback
        self.idx = 0
        self.events = []                        # (monotonic, text) for the log
        self._cb_last = None

    # ---- open / close
    def start(self):
        try:
            self.stream = self._open()
            if self.source == 3:
                self._make_engine(self.block or 128)
            self._old_switch = sys.getswitchinterval()
            hints = ga.power_hints(True)
            self.stream.start()
            lat = self.stream.latency
            if isinstance(lat, tuple):
                self.out_latency_ms = 1000.0 * float(lat[1])
                self.buffer_ms = 1000.0 * float(sum(lat))
            else:
                self.out_latency_ms = self.buffer_ms = 1000.0 * float(lat)
            if hints:
                self.mode += f", {hints}"
            self.running = True
        except Exception as e:
            self.error = str(e)

    def _open(self):
        if sd is None:
            raise RuntimeError(f"sounddevice missing: {ga.SD_ERROR}")
        api_name = sd.query_hostapis(ga._hostapi(self.api))["name"]
        extra = None
        if api_name == "Windows WASAPI":
            extra = sd.WasapiSettings(exclusive=self.exclusive, auto_convert=not self.exclusive)
        duplex = api_name == "ASIO"
        bs = self.block
        lat = (1 if duplex else 2) * (bs or 128) / self.sr
        j, _d = ga._device_index(self.out_name, "output", self.api)
        self.mode = api_name + (" exclusive" if extra is not None and self.exclusive else "")
        if self.in_name:
            i, d = ga._device_index(self.in_name, "input", self.api)
            return sd.Stream(device=(i, j), samplerate=self.sr, blocksize=bs,
                             channels=(min(2, int(d["max_input_channels"])), 2),
                             dtype="float32", latency=lat, extra_settings=extra,
                             callback=self._cb_duplex)
        return sd.OutputStream(device=j, samplerate=self.sr, blocksize=bs, channels=2,
                               dtype="float32", latency=lat, extra_settings=extra,
                               callback=self._cb_out)

    def _make_engine(self, block):
        la = ga.LiveAudio(None, self.out_name, block=block, sr=self.sr, api=self.api)
        la._dsp_init()
        self.engine = la

    def stop(self):
        self.running = False
        st, self.stream = self.stream, None
        if st is not None:
            try:
                st.stop()
                st.close()
            except Exception:
                pass
        ga.power_hints(False)
        if self._old_switch is not None:
            sys.setswitchinterval(self._old_switch)

    # ---- the callback
    def _cb_duplex(self, indata, outdata, frames, _time, status):
        self._cb(indata, outdata, frames, status)

    def _cb_out(self, outdata, frames, _time, status):
        self._cb(None, outdata, frames, status)

    def _cb(self, indata, outdata, frames, status):
        try:
            t0 = time.perf_counter()
            if self._cb_last is None:
                threading.current_thread().name = "mini-asio"
                ga.thread_no_throttle()
            if self.pro_audio and not self._pro_done:
                self._pro_done = True
                self._set_pro_audio()
            if self.affinity != self._aff_done:        # pin (or free) the driver's thread
                self._aff_done = self.affinity
                self._set_affinity(self.affinity)
            if _K32 is not None:
                self.cpu_hist[_K32.GetCurrentProcessorNumber() & 63] += 1
            if status:
                if status.output_underflow:
                    self.underruns += 1
                    self.events.append((time.monotonic(), "driver: output underflow"))
                if status.input_overflow:
                    self.overflows += 1
            self.frames = frames
            block_s = frames / self.sr
            last, self._cb_last = self._cb_last, t0
            if last is not None:
                gap = (t0 - last) / block_s
                self.gaps[self.idx % RING] = gap
                if gap > self.worst_gap:
                    self.worst_gap = gap
                if gap > 2.0:
                    self.late += 1
                    self.events.append((time.monotonic(), f"late block: {1000 * gap * block_s:.1f} ms "
                                        f"after the last ({gap:.1f} blocks)"))
                elif gap > 1.5:
                    self.near += 1
                self.gap_ms = max(1000.0 * gap * block_s, self.gap_ms * (1.0 - 2.0 * block_s))
            src = self.source
            if src == 0:
                outdata.fill(0)
            elif src == 1:
                step = 2 * math.pi * 220.0 / self.sr
                ph = self._phase + np.arange(frames, dtype=np.float64) * step
                self._phase = float((ph[-1] + step) % (2 * math.pi))
                tone = (0.25 * np.sin(ph)).astype(np.float32)
                outdata[:, 0] = tone
                outdata[:, 1] = tone
            elif src == 2:
                if indata is None:
                    outdata.fill(0)
                elif indata.shape[1] >= 2:
                    outdata[:] = indata[:, :2]
                else:
                    outdata[:, 0] = indata[:, 0]
                    outdata[:, 1] = indata[:, 0]
            else:
                la = self.engine
                if la is None:
                    self._make_engine(frames)          # once, on the first block
                    la = self.engine
                if indata is None:
                    x = np.zeros((2, frames), np.float32)
                else:
                    x = indata.T if indata.shape[1] == 2 else np.repeat(indata.T, 2, axis=0)
                    x = np.ascontiguousarray(x, dtype=np.float32)
                if frames == la.block:
                    outdata[:] = la._dsp_block(x, time.monotonic()).T
                else:                                  # a driver handing odd sizes, as the engine does
                    n = min(frames, la.block)
                    xb = np.zeros((2, la.block), np.float32)
                    xb[:, :n] = x[:, :n]
                    outdata.fill(0)
                    outdata[:n] = la._dsp_block(xb, time.monotonic())[:, :n].T
            if self.fake_load > 0:                     # spin, in short numpy calls
                until = t0 + self.fake_load * block_s
                a = np.ones(64, np.float32)
                while time.perf_counter() < until:
                    a = a * 1.0001
            if indata is not None:
                self.in_peak = float(np.abs(indata).max())
            self.out_peak = float(np.abs(outdata).max())
            r = (time.perf_counter() - t0) / block_s
            self.durs[self.idx % RING] = r
            self.idx += 1
            self.load = 0.9 * self.load + 0.1 * r
            self.peak = max(r, self.peak * (1.0 - 4.0 * block_s))
            if r > self.worst_dsp:
                self.worst_dsp = r
            self.rounds += 1
        except Exception as e:
            self.error = f"callback: {e}"
            outdata.fill(0)

    @staticmethod
    def _set_pro_audio():
        if sys.platform != "win32":
            return
        try:
            task = ctypes.c_ulong(0)
            ctypes.windll.avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task))
        except Exception:
            pass

    @staticmethod
    def _set_affinity(mask):
        """This thread on the CPUs of `mask`; 0 = every CPU the process may use."""
        if _K32 is None:
            return
        try:
            if not mask:
                pm, sm = ctypes.c_size_t(0), ctypes.c_size_t(0)
                _K32.GetCurrentProcess.restype = ctypes.c_void_p
                _K32.GetProcessAffinityMask(_K32.GetCurrentProcess(), ctypes.byref(pm), ctypes.byref(sm))
                mask = pm.value or (1 << os.cpu_count()) - 1
            _K32.GetCurrentThread.restype = ctypes.c_void_p
            _K32.SetThreadAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
            _K32.SetThreadAffinityMask(_K32.GetCurrentThread(), ctypes.c_size_t(mask))
        except Exception:
            pass

    def cpu_text(self, classes):
        """'cpu 2 (93 %) P, 14 (7 %) E' for the CPUs the callback ran on."""
        tot = int(self.cpu_hist.sum())
        if not tot:
            return "-"
        p_mask = classes[max(classes)] if classes else 0
        parts = []
        for i in np.argsort(-self.cpu_hist)[:4]:
            c = int(self.cpu_hist[i])
            if not c:
                break
            kind = ("P" if p_mask >> int(i) & 1 else "E") if classes else ""
            parts.append(f"{int(i)} ({100 * c / tot:.0f} %){kind}")
        return ", ".join(parts)

    def cpu_load(self):
        try:
            return float(self.stream.cpu_load) if self.stream is not None else 0.0
        except Exception:
            return 0.0

    def drain(self):
        ev, self.events = self.events, []
        return ev


class Trace(QWidget):
    """The last 600 callbacks: a bar per block, its gap in blocks (grey,
    yellow past 1.5, red past two), the DSP share as blue dots; 1x and 2x
    marked."""

    def __init__(self):
        super().__init__()
        self.probe = None
        self.setMinimumHeight(120)

    def paintEvent(self, _e):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(24, 24, 28))
        pr = self.probe
        n = min(600, w)
        scale = h / 3.0                           # 3 blocks tall
        for k, col in ((1.0, QColor(60, 140, 60)), (2.0, QColor(160, 60, 60))):
            y = int(h - k * scale)
            p.setPen(QPen(col, 1, Qt.DashLine))
            p.drawLine(0, y, w, y)
        if pr is None or pr.idx < 2:
            p.end()
            return
        i1 = pr.idx
        i0 = max(0, i1 - n)
        ids = np.arange(i0, i1) % RING
        gaps = pr.gaps[ids]
        durs = pr.durs[ids]
        x0 = w - (i1 - i0)
        for k in range(i1 - i0):
            g = min(3.0, float(gaps[k]))
            d = min(3.0, float(durs[k]))
            if g > 2.0:
                col = QColor(200, 70, 70)
            elif g > 1.5:
                col = QColor(220, 180, 70)
            else:
                col = QColor(110, 110, 125)
            p.setPen(col)
            p.drawLine(x0 + k, h, x0 + k, int(h - g * scale))
            p.setPen(QColor(90, 170, 230))
            p.drawPoint(x0 + k, int(h - d * scale))
        p.end()


class Burner(QWidget):
    """The paint burn: a fresh 640x480 image drawn every tick."""

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(1)
        self.on = False
        self.setMinimumHeight(60)

    def paintEvent(self, _e):
        p = QPainter(self)
        if not self.on:
            p.fillRect(self.rect(), QColor(30, 30, 34))
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter, "paint burn off")
            p.end()
            return
        a = self.rng.integers(0, 255, (480, 640, 4), dtype=np.uint8)
        img = QImage(a.data, 640, 480, 640 * 4, QImage.Format_RGBA8888)
        p.drawImage(self.rect(), img)
        p.end()


class Mini(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gacha mini: the audio stream alone")
        self.settings = QSettings("gacha", "gacha_mini")
        main = QSettings("gacha", "gacha")             # the main app's devices as defaults
        self.probe = None
        self.gilwatch = None
        self._burn_stop = threading.Event()
        self._burn_thread = None
        self._t0 = time.monotonic()
        self._cpu = None
        try:
            import psutil
            self._cpu = psutil.Process()
            self._cpu.cpu_percent()
        except Exception:
            pass

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        g = QGridLayout()
        lay.addLayout(g)
        self.api = QComboBox()
        self.api.addItems(ga.audio_apis())
        api = self.settings.value("api", main.value("audio/api", "")) or ""
        if api and self.api.findText(api) >= 0:
            self.api.setCurrentText(api)
        self.api.currentTextChanged.connect(self._fill_devices)
        self.inp, self.out = QComboBox(), QComboBox()
        self.blk = QComboBox()
        self.blk.addItems(BLOCKS)
        self.blk.setCurrentText(str(self.settings.value("block", main.value("audio/block", "128"))))
        self.excl = QCheckBox("WASAPI exclusive")
        self.excl.setChecked(self.settings.value("exclusive", True, type=bool))
        self.src = QComboBox()
        self.src.addItems(SOURCES)
        self.src.setCurrentIndex(int(self.settings.value("source", 1)))
        self.src.currentIndexChanged.connect(self._source)
        self.go = QPushButton("Start")
        self.go.clicked.connect(self._toggle)
        self.load = QSlider(Qt.Horizontal)
        self.load.setRange(0, 150)
        self.load.valueChanged.connect(self._load)
        self.load_lbl = QLabel("fake load 0 %")
        self.sw = QComboBox()
        for name, _v in SWITCHES:
            self.sw.addItem(name)
        self.sw.setCurrentIndex(int(self.settings.value("switch", 2)))
        self.sw.currentIndexChanged.connect(self._switch)
        self.burn = QCheckBox("GUI burn (a busy Python thread)")
        self.burn.toggled.connect(self._burn)
        self.paint = QCheckBox("paint burn (640x480 image, 60/s)")
        self.paint.toggled.connect(self._paint)
        self.gcoff = QCheckBox("gc off")
        self.gcoff.toggled.connect(lambda on: gc.disable() if on else gc.enable())
        self.pro = QCheckBox("Pro Audio (MMCSS) on the driver's thread")
        self.pro.setChecked(self.settings.value("pro", False, type=bool))
        self.classes = cpu_topology()                  # {efficiency class: cpu mask}
        self.hiprio = QCheckBox("process high priority (as DAWs run)")
        self.hiprio.toggled.connect(self._hiprio)
        self.core = QComboBox()
        self.core.addItems(CORES)
        self.core.currentIndexChanged.connect(self._core)
        if not self.classes:
            self.core.setEnabled(False)
        r = 0
        for label, w in (("host API", self.api), ("input", self.inp), ("output", self.out),
                         ("block", self.blk), ("source", self.src)):
            g.addWidget(QLabel(label), r, 0)
            g.addWidget(w, r, 1)
            r += 1
        g.addWidget(self.excl, 3, 2)
        g.addWidget(self.go, 4, 2)
        g.addWidget(self.load_lbl, r, 0)
        g.addWidget(self.load, r, 1, 1, 2)
        r += 1
        g.addWidget(QLabel("switch interval"), r, 0)
        g.addWidget(self.sw, r, 1)
        g.addWidget(self.pro, r, 2)
        r += 1
        g.addWidget(self.burn, r, 0, 1, 2)
        g.addWidget(self.gcoff, r, 2)
        r += 1
        g.addWidget(self.paint, r, 0, 1, 2)
        g.addWidget(self.hiprio, r, 2)
        r += 1
        g.addWidget(QLabel("driver's thread on"), r, 0)
        g.addWidget(self.core, r, 1)
        self.stats = QLabel("not open")
        self.stats.setFont(QFont("Consolas", 10))
        self.stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.stats)
        self.trace = Trace()
        lay.addWidget(self.trace)
        self.burner = Burner()
        lay.addWidget(self.burner)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFont(QFont("Consolas", 9))
        lay.addWidget(self.log, 1)
        row = QHBoxLayout()
        clear = QPushButton("reset counters")
        clear.clicked.connect(self._reset)
        row.addWidget(clear)
        row.addStretch(1)
        lay.addLayout(row)
        self._fill_devices()
        saved_in = self.settings.value("in", main.value("audio/in", "")) or ""
        saved_out = self.settings.value("out", main.value("audio/out", "")) or ""
        if self.inp.findText(saved_in) >= 0:
            self.inp.setCurrentText(saved_in)
        if self.out.findText(saved_out) >= 0:
            self.out.setCurrentText(saved_out)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)
        self.paint_timer = QTimer(self)
        self.paint_timer.timeout.connect(self.burner.update)
        self.resize(900, 760)
        why = ga.audio_status()
        if why:
            self._log(why)
        self._log(f"python {sys.version.split()[0]}  sounddevice {sd.__version__ if sd else '-'}  "
                  f"switch interval now {1000 * sys.getswitchinterval():.2f} ms")
        if self.classes:
            top = max(self.classes)
            self._log("cpus: " + "  ".join(
                f"{'P' if k == top else 'E'}-cores {_mask_text(m)}" for k, m in
                sorted(self.classes.items(), reverse=True)))
        else:
            self._log("cpus: topology not readable, the core choice is off")

    # ---- controls
    def _fill_devices(self):
        api = self.api.currentText() or None
        ins, outs = ga.audio_devices(api)
        cur_in, cur_out = self.inp.currentText(), self.out.currentText()
        self.inp.clear()
        self.inp.addItem("(none: output only)")
        self.inp.addItems([n for n, _ in ins])
        self.out.clear()
        self.out.addItems([n for n, _ in outs])
        if self.inp.findText(cur_in) >= 0:
            self.inp.setCurrentText(cur_in)
        if self.out.findText(cur_out) >= 0:
            self.out.setCurrentText(cur_out)

    def _toggle(self):
        if self.probe is not None:
            self._stop()
            return
        out = self.out.currentText()
        if not out:
            self._log("no output device")
            return
        inp = self.inp.currentText() if self.inp.currentIndex() > 0 else ""
        blk = self.blk.currentText()
        block = 0 if not blk.isdigit() else int(blk)
        for k, v in (("api", self.api.currentText()), ("in", inp), ("out", out), ("block", blk),
                     ("exclusive", self.excl.isChecked()), ("source", self.src.currentIndex()),
                     ("switch", self.sw.currentIndex()), ("pro", self.pro.isChecked())):
            self.settings.setValue(k, v)
        self.probe = Probe(inp, out, block, self.api.currentText() or None,
                           self.excl.isChecked(), self.src.currentIndex())
        self.probe.fake_load = self.load.value() / 100.0
        self.probe.pro_audio = self.pro.isChecked()
        self.probe.affinity = self._core_mask(self.core.currentIndex())
        self.trace.probe = self.probe
        self._switch(self.sw.currentIndex())
        self.probe.start()
        if self.probe.error:
            self._log(f"not open: {self.probe.error}")
            self.probe = None
            self.trace.probe = None
            return
        self._t0 = time.monotonic()
        self._log(f"open: {inp or 'output only'} -> {out}, block {blk}, {self.probe.mode}, "
                  f"switch {1000 * sys.getswitchinterval():.2f} ms, "
                  f"source '{SOURCES[self.src.currentIndex()]}'")
        # under 2 ms the watcher's own 1 ms sleep waking late (Windows'
        # scheduler) reads as holds of the idle main thread: noise
        thr = max(2.0, 0.75 * 1000.0 * (block or 128) / ga.SR)
        self.gilwatch = GilWatch(threshold_ms=thr)
        self.gilwatch.start()
        self._log(f"gil watch: a hold over {thr:.1f} ms is a stall")
        self.go.setText("Stop")
        for w in (self.api, self.inp, self.out, self.blk, self.excl):
            w.setEnabled(False)

    def _stop(self):
        pr, self.probe = self.probe, None
        if pr is not None:
            pr.stop()
            self._log(f"closed after {pr.rounds} blocks: late {pr.late}, near {pr.near}, "
                      f"underruns {pr.underruns}, worst gap {pr.worst_gap:.2f} blocks, "
                      f"worst dsp {100 * pr.worst_dsp:.0f} %")
        if self.gilwatch is not None:
            self.gilwatch.stop()
            self.gilwatch = None
        self.trace.probe = None
        self.go.setText("Start")
        for w in (self.api, self.inp, self.out, self.blk, self.excl):
            w.setEnabled(True)

    def _source(self, i):
        if self.probe is not None:
            self.probe.source = i
            self._log(f"source: {SOURCES[i]}")

    def _load(self, v):
        self.load_lbl.setText(f"fake load {v} %")
        if self.probe is not None:
            self.probe.fake_load = v / 100.0

    def _switch(self, i):
        v = SWITCHES[i][1]
        sys.setswitchinterval(v)
        if self.probe is not None:
            self._log(f"switch interval {1000 * v:.2f} ms")

    def _burn(self, on):
        if on and self._burn_thread is None:
            self._burn_stop.clear()

            def burn():
                d = {}
                while not self._burn_stop.is_set():
                    for i in range(2000):
                        d[i] = i * 2
            self._burn_thread = threading.Thread(target=burn, name="mini-burn", daemon=True)
            self._burn_thread.start()
            self._log("GUI burn on")
        elif not on and self._burn_thread is not None:
            self._burn_stop.set()
            self._burn_thread.join(1.0)
            self._burn_thread = None
            self._log("GUI burn off")

    def _hiprio(self, on):
        ok = set_process_priority(on)
        self._log(f"process priority {'high' if on else 'normal'}{'' if ok else ' (refused)'}")

    def _core_mask(self, i):
        if not self.classes or i == 0:
            return 0
        p_mask = self.classes[max(self.classes)]
        if i == 1:
            return p_mask
        cpus = [c for c in range(64) if p_mask >> c & 1]
        return 1 << cpus[min(2, len(cpus) - 1)]       # not CPU 0, where the interrupts land

    def _core(self, i):
        mask = self._core_mask(i)
        if self.probe is not None:
            self.probe.affinity = mask
        self._log(f"driver's thread on {CORES[i]}" + (f" (cpus {_mask_text(mask)})" if mask else ""))

    def _paint(self, on):
        self.burner.on = on
        if on:
            self.paint_timer.start(16)
        else:
            self.paint_timer.stop()
            self.burner.update()
        self._log(f"paint burn {'on' if on else 'off'}")

    def _reset(self):
        pr = self.probe
        if pr is None:
            return
        pr.late = pr.near = pr.underruns = pr.overflows = 0
        pr.worst_gap = pr.worst_dsp = 0.0
        pr.rounds = 0
        if self.gilwatch is not None:
            self.gilwatch.stalls = 0
            self.gilwatch.worst_ms = 0.0
        self._t0 = time.monotonic()

    # ---- the tick
    def _log(self, text):
        self.log.appendPlainText(f"{time.monotonic() - self._t0:8.2f}  {text}")

    def _tick(self):
        pr = self.probe
        if pr is None:
            return
        for _t, text in pr.drain():
            self._log(text)
        w = self.gilwatch
        if w is not None:
            for _t, ms, what in w.drain():
                self._log(f"gil: held {ms:.1f} ms  {what}")
        if pr.error:
            self._log(pr.error)
            pr.error = None
        db = lambda p: f"{20 * math.log10(max(p, 1e-5)):4.0f} dB"
        up = time.monotonic() - self._t0
        cpu = f"  process cpu {self._cpu.cpu_percent():.0f} %" if self._cpu else ""
        cadence = ""
        if pr.idx > 50:                                  # the median gap of the last 200 blocks
            ids = np.arange(max(0, pr.idx - 200), pr.idx) % RING
            med = float(np.median(pr.gaps[ids]))
            if med < 0.5:
                cadence = (f"\ncadence: bursts (median gap {med:.2f} blocks): the host API hands "
                           "several blocks per period, so gaps are not glitches here; use ASIO or exclusive")
            else:
                cadence = f"\ncadence: one block per call (median gap {med:.2f} blocks)"
        self.stats.setText(
            f"{pr.mode}   block {pr.frames} frames = {1000 * pr.frames / pr.sr:.2f} ms   "
            f"out latency {pr.out_latency_ms:.1f} ms   buffers {pr.buffer_ms:.1f} ms   up {up:.0f} s\n"
            f"dsp {100 * pr.load:5.1f} %   peak {100 * pr.peak:4.0f} %   worst {100 * pr.worst_dsp:4.0f} %   "
            f"portaudio load {100 * pr.cpu_load():4.1f} %{cpu}\n"
            f"gap {pr.gap_ms:5.2f} ms   worst {pr.worst_gap:5.2f} blocks   "
            f"late (>2) {pr.late}   near (>1.5) {pr.near}   blocks {pr.rounds}\n"
            f"driver underruns {pr.underruns}   overflows {pr.overflows}   "
            f"in {db(pr.in_peak)}   out {db(pr.out_peak)}   "
            f"gil stalls {w.stalls if w else 0}   worst {w.worst_ms if w else 0:.1f} ms   "
            f"switch {1000 * sys.getswitchinterval():.2f} ms   gc {'off' if not gc.isenabled() else 'on'}\n"
            f"callback ran on cpu {pr.cpu_text(self.classes)}"
            f"{cadence}")
        self.trace.update()

    def closeEvent(self, e):
        self._stop()
        self._burn(False)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    win = Mini()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
