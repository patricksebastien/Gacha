#!/usr/bin/env python3
"""
gacha_vst.py — VST3 inserts on the master output.

Third-party effect plugins (plugdata-fx, a compressor, a reverb...) loaded
through pedalboard and run in the audio thread, in the app's own process:
no extra buffer, no added latency beyond what a plugin reports (pedalboard
compensates that by itself). The chain sits after the five racks are summed
and before the master gain and the limiter, so the limiter still protects
the PA. The price of the in-process choice: a plugin's editor window blocks
the Qt interface while it is open (the audio keeps running), and a plugin
that crashes takes the app down with it. On Windows the editor window has
to be moved on screen by hand (place_editor_window), see there.

    chain = InsertChain()
    ins = chain.add("vst/plugdata-fx.vst3")   # GUI thread, ~1-2 s for plugdata
    y = chain.process(x, sr)                  # audio thread, (2, bs) float32
    chain.to_json() / InsertChain.from_json(text)   # settings

Plugins are looked for in ./vst first, then the platform's usual VST3
folders. Loading is slow (a second or two for a JUCE plugin) and must not
happen on the audio thread; the audio thread only reads `chain.items`, a
tuple that the GUI thread swaps whole.
"""
import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    from pedalboard import load_plugin, VST3Plugin
    VST_AVAILABLE = True
except ImportError:                            # an old pedalboard
    VST_AVAILABLE = False

VST_DIR = Path(__file__).parent / "vst"
KNOBS = 8                                      # MIDI-mappable parameter slots
_stderr_muted = False


def mute_plugin_stderr():
    """Send the process's file descriptor 2 to /dev/null, keeping Python's
    own sys.stderr on the terminal. plugdata-fx prints some thirty JUCE
    assertion lines per processed block (a bus-layout check that fails in
    its Linux build); written from the audio thread to a terminal, those
    are a real-time hazard, and 5000 lines a second bury everything else.
    Done once, at the first plugin load, so a run without plugins keeps
    its stderr. Qt's own warnings, which also go to fd 2, are lost too."""
    global _stderr_muted
    if _stderr_muted:
        return False
    try:
        keep = os.dup(2)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 2)
        os.close(null)
        sys.stderr = os.fdopen(keep, "w", buffering=1)
        _stderr_muted = True
        return True
    except OSError:
        return False


def vst_dirs():
    """Where plugins live: the app's own folder, then the system's."""
    dirs = [VST_DIR]
    home = Path.home()
    if sys.platform == "win32":
        dirs += [Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Common Files" / "VST3",
                 Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
                 / "Programs" / "Common" / "VST3"]
    elif sys.platform == "darwin":
        dirs += [home / "Library" / "Audio" / "Plug-Ins" / "VST3",
                 Path("/Library/Audio/Plug-Ins/VST3")]
    else:
        dirs += [home / ".vst3", Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3")]
    return dirs


def scan_plugins():
    """[(label, path)] of every .vst3 found, the app's folder first, then
    the rest alphabetically; vendor subfolders (Common Files/VST3/Xfer/...)
    one level down are looked into too. Nothing is loaded: a JUCE plugin
    takes a second or two to load, so that waits for the user's choice."""
    found, seen = [], set()
    for d in vst_dirs():
        if not d.is_dir():
            continue
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for p in entries:
            if p.suffix.lower() != ".vst3" and p.is_dir():      # a vendor folder
                try:
                    subs = sorted(p.iterdir(), key=lambda q: q.name.lower())
                except OSError:
                    continue
                for q in subs:
                    if q.suffix.lower() == ".vst3" and q.name not in seen:
                        seen.add(q.name)
                        found.append((f"{q.stem}  ({p.name})", str(q)))
                continue
            if p.suffix.lower() != ".vst3" or p.name in seen:
                continue
            seen.add(p.name)
            label = p.stem if d == VST_DIR else f"{p.stem}  ({d})"
            found.append((label, str(p)))
    return found


def bundle_binary(path):
    """The binary inside a .vst3 bundle folder, or None if `path` is not
    such a folder or holds none for this platform. Windows bundles carry
    Contents/x86_64-win/<name>.vst3 (a DLL), Linux ones
    Contents/x86_64-linux/<name>.so, macOS ones Contents/MacOS/<name>."""
    p = Path(path)
    contents = p / "Contents"
    if not (p.is_dir() and contents.is_dir()):
        return None
    if sys.platform == "win32":
        archs, exts = ("x86_64-win", "arm64-win", "x86-win"), (".vst3",)
    elif sys.platform == "darwin":
        archs, exts = ("MacOS",), ("",)
    else:
        archs, exts = ("x86_64-linux", "aarch64-linux", "arm64-linux", "i386-linux"), (".so",)
    for arch in archs:
        d = contents / arch
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                return f
    return None


def load_paths(path):
    """The paths to try for `path`, best first. JUCE on Windows only opens
    a .vst3 *file*, so a bundle folder (what Common Files/VST3 holds) is
    handed over by its inner DLL; Linux and macOS take the folder as it
    is, the inner binary is the fallback. A file is tried as given."""
    p = Path(path)
    inner = bundle_binary(p)
    if inner is None:
        return [p]
    return [inner, p] if sys.platform == "win32" else [p, inner]


def place_editor_window(owner_hwnd=None, timeout=5.0):
    """Windows only: bring the editor window pedalboard is about to open
    onto the screen. Call right before plugin.show_editor(). pedalboard's
    window is a JUCE DocumentWindow it never positions, so its client
    area lands at (0, 0) and the native title bar, which JUCE adds around
    it, sits above the screen edge: the plugin shows in the top-left
    corner with nothing to grab, move or close it by. Window managers on
    X11 and macOS push a window back on screen; Windows leaves it there.
    So a thread waits for the window (this process's top-level "Pedalboard",
    up to `timeout` s), centres it on the monitor that holds `owner_hwnd`
    (the app's window; the primary one without) clamped into the work area,
    and brings it to the front. SetWindowPos from another thread is fine:
    it is a message the window's own thread handles in pedalboard's loop,
    which releases the GIL while it runs. No-op off Windows."""
    if sys.platform != "win32":
        return False
    import ctypes
    import ctypes.wintypes as wt
    u = ctypes.windll.user32
    ENUM = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    u.EnumWindows.argtypes = [ENUM, wt.LPARAM]
    pid = os.getpid()

    def find():
        found = []

        @ENUM
        def cb(h, _):
            owner_pid = wt.DWORD()
            u.GetWindowThreadProcessId(h, ctypes.byref(owner_pid))
            if owner_pid.value == pid and u.IsWindowVisible(h):
                buf = ctypes.create_unicode_buffer(64)
                u.GetWindowTextW(h, buf, 64)
                if buf.value == "Pedalboard":
                    found.append(h)
            return True
        u.EnumWindows(cb, 0)
        return found[0] if found else None

    def work_area(h):
        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wt.DWORD), ("rcMonitor", wt.RECT),
                        ("rcWork", wt.RECT), ("dwFlags", wt.DWORD)]
        mon = u.MonitorFromWindow(wt.HWND(h), 2)          # MONITOR_DEFAULTTONEAREST
        mi = MONITORINFO(); mi.cbSize = ctypes.sizeof(MONITORINFO)
        if mon and u.GetMonitorInfoW(mon, ctypes.byref(mi)):
            return mi.rcWork
        r = wt.RECT()
        u.SystemParametersInfoW(48, 0, ctypes.byref(r), 0)  # SPI_GETWORKAREA
        return r

    def run():
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            h = find()
            if h:
                break
            time.sleep(0.05)
        else:
            return
        r = wt.RECT()
        u.GetWindowRect(h, ctypes.byref(r))
        w, ht = r.right - r.left, r.bottom - r.top
        wa = work_area(owner_hwnd or h)
        x = wa.left + max(0, (wa.right - wa.left - w) // 2)
        y = wa.top + max(0, (wa.bottom - wa.top - ht) // 2)
        u.SetWindowPos(wt.HWND(h), None, x, y, 0, 0, 0x0001 | 0x0004)  # NOSIZE | NOZORDER
        u.SetForegroundWindow(wt.HWND(h))

    threading.Thread(target=run, name="vst-editor-place", daemon=True).start()
    return True


class Insert:
    """One loaded plugin: on/off with a one-block fade, parameters by name,
    state as bytes. Built with Insert.load() on the GUI thread."""

    def __init__(self, path, plugin):
        self.path = str(path)
        self.plugin = plugin
        self.name = getattr(plugin, "name", None) or Path(path).stem
        self.on = True                          # set from any thread
        self.error = None                       # a processing error, then bypassed
        self._on_now = 1.0                      # fade state, audio thread
        self._fifo = None                       # pending output, if the plugin
                                                # hands back short blocks

    @classmethod
    def load(cls, path):
        """Load the plugin at `path`; raises on failure. Slow (1-2 s)."""
        if not VST_AVAILABLE:
            raise RuntimeError("this pedalboard has no VST3 support")
        mute_plugin_stderr()
        plugin, errors = None, []
        for candidate in load_paths(path):
            try:
                plugin = load_plugin(str(candidate))
                break
            except ImportError as e:              # not a plugin here: try the next form
                errors.append(f"{candidate}: {e}")
        if plugin is None:
            raise ImportError("; ".join(errors))
        if not getattr(plugin, "is_effect", True):
            raise RuntimeError(f"{Path(path).stem} is an instrument, not an effect")
        return cls(path, plugin)

    # ---- parameters ----
    def params(self):
        """{name: parameter} of what the plugin exposes, without plugdata's
        placeholder slots (disabled_paramN) that no [param] object claims."""
        try:
            items = self.plugin.parameters.items()
        except Exception:
            return {}
        return {n: p for n, p in items if not n.startswith("disabled_")}

    def set_frac(self, name, x):
        """Parameter `name` to x in 0..1 over its whole range."""
        try:
            self.plugin.parameters[name].raw_value = max(0.0, min(1.0, float(x)))
            return True
        except Exception:
            return False

    def get_frac(self, name):
        try:
            return float(self.plugin.parameters[name].raw_value)
        except Exception:
            return None

    # ---- state ----
    @property
    def state(self):
        try:
            return bytes(self.plugin.raw_state)
        except Exception:
            return b""

    @state.setter
    def state(self, data):
        if data:
            self.plugin.raw_state = bytes(data)

    def latency_samples(self):
        return int(getattr(self.plugin, "reported_latency_samples", 0) or 0)

    def raw_names(self):
        """The plugin's parameter names as it reports them now, in its own
        order, disabled slots included: a cheap fingerprint of its parameter
        set (7 ms for plugdata-fx's 2595 slots, against 20 ms for params(),
        which builds pedalboard's wrappers), safe from any thread."""
        try:
            return tuple(p.name for p in self.plugin._parameters)
        except Exception:
            return ()

    # ---- audio thread ----
    def _fit(self, y, x):
        """Exactly x.shape[1] frames out, whatever the plugin returned."""
        n = x.shape[1]
        if y.shape[1] == n and self._fifo is None:
            return y
        q = y if self._fifo is None or self._fifo.shape[1] == 0 \
            else np.concatenate([self._fifo, y], axis=1)
        if q.shape[1] >= n:
            out, self._fifo = q[:, :n], q[:, n:]
            return np.ascontiguousarray(out)
        self._fifo = q
        return np.zeros_like(x)

    def process(self, x, sr):
        """(2, bs) -> (2, bs); bypass fades over one block, and a bypassed
        plugin costs nothing (its tail freezes, which is fine)."""
        want = 1.0 if (self.on and self.error is None) else 0.0
        now = self._on_now
        if want == 0.0 and now == 0.0:
            return x
        try:
            y = self._fit(self.plugin.process(x, sr, reset=False), x)
        except Exception as e:                  # a Python-level failure: out of the chain
            self.error = str(e) or type(e).__name__
            self._on_now = 0.0
            return x
        if want != now:
            w = np.linspace(now, want, x.shape[1], dtype=np.float32)
            y = x * (1.0 - w) + y * w
            self._on_now = want
        return y


class InsertChain:
    """The inserts in order. `items` is a tuple the audio thread reads and
    the GUI thread replaces whole; `bypass` mutes the whole chain (a pedal
    switch on stage)."""

    def __init__(self):
        self.items = ()
        self.bypass = False
        self._byp_now = 0.0
        # KNOBS MIDI slots: each (insert index, parameter name) or None
        self.knobs = [None] * KNOBS

    # ---- GUI thread ----
    def add(self, path):
        ins = Insert.load(path)
        self.items = self.items + (ins,)
        return ins

    def remove(self, ins):
        self.items = tuple(i for i in self.items if i is not ins)
        self.knobs = [k if k is not None and k[0] < len(self.items) else None
                      for k in self.knobs]

    def move(self, ins, delta):
        lst = list(self.items)
        i = lst.index(ins)
        j = max(0, min(len(lst) - 1, i + delta))
        if i != j:
            lst[i], lst[j] = lst[j], lst[i]
            self.items = tuple(lst)
            for n, k in enumerate(self.knobs):      # the slots follow the plugin
                if k is not None and k[0] in (i, j):
                    self.knobs[n] = (j if k[0] == i else i, k[1])

    def clear(self):
        self.items = ()
        self.knobs = [None] * KNOBS

    def knob(self, n, x):
        """MIDI slot n (0-based) to x in 0..1."""
        k = self.knobs[n] if 0 <= n < KNOBS else None
        if k is None or k[0] >= len(self.items):
            return False
        return self.items[k[0]].set_frac(k[1], x)

    def fingerprint(self):
        """The parameter names of the whole chain, for spotting a plugin
        that renamed or added parameters (plugdata when a [param] object
        is put in the patch, Surge XT Effects when its type is switched)."""
        return tuple(ins.raw_names() for ins in self.items)

    def watch(self, on_change, period=1.0):
        """A daemon thread that calls on_change() (from that thread) when
        fingerprint() differs from the last look, checked every `period` s.
        Returns the threading.Event that stops it. Nothing is read while the
        chain is empty."""
        stop = threading.Event()

        def run():
            last = None
            while not stop.wait(period):
                if not self.items:
                    last = None
                    continue
                fp = self.fingerprint()
                if last is not None and fp != last:
                    on_change()
                last = fp
        threading.Thread(target=run, name="vst-params-watch", daemon=True).start()
        return stop

    def to_json(self):
        return json.dumps({
            "inserts": [{"path": i.path, "on": bool(i.on),
                         "state": base64.b64encode(i.state).decode("ascii")}
                        for i in self.items],
            "knobs": [list(k) if k is not None else None for k in self.knobs],
            "bypass": bool(self.bypass)})

    def load_json(self, text):
        """Rebuild from to_json() output; returns [(path, error)] for the
        plugins that could not come back. Slow: loads every plugin."""
        errors = []
        try:
            d = json.loads(text) if text else {}
        except ValueError:
            return [("settings", "unreadable inserts entry")]
        items = []
        for e in d.get("inserts", []):
            try:
                ins = Insert.load(e["path"])
                st = e.get("state", "")
                if st:
                    ins.state = base64.b64decode(st)
                ins.on = bool(e.get("on", True))
                items.append(ins)
            except Exception as ex:
                errors.append((e.get("path", "?"), str(ex) or type(ex).__name__))
        self.items = tuple(items)
        knobs = d.get("knobs", [])
        self.knobs = [(int(k[0]), str(k[1])) if isinstance(k, list) and len(k) == 2
                      and int(k[0]) < len(items) else None
                      for k in (knobs + [None] * KNOBS)[:KNOBS]]
        self.bypass = bool(d.get("bypass", False))
        return errors

    # ---- audio thread ----
    def process(self, x, sr):
        items = self.items
        if not items:
            return x
        want = 0.0 if self.bypass else 1.0
        now = self._byp_now
        if want == 0.0 and now == 0.0:
            return x
        y = np.ascontiguousarray(x, dtype=np.float32)
        for ins in items:
            y = ins.process(y, sr)
        if want != now:
            w = np.linspace(now, want, x.shape[1], dtype=np.float32)
            y = x * (1.0 - w) + y * w
            self._byp_now = want
        return np.ascontiguousarray(y, dtype=np.float32)


if __name__ == "__main__":                      # quick check from a terminal
    import time
    print("plugins found:")
    for label, path in scan_plugins():
        print("  ", label, "->", path)
    if len(sys.argv) > 1:
        chain = InsertChain()
        t = time.perf_counter()
        ins = chain.add(sys.argv[1])
        print(f"loaded {ins.name} in {time.perf_counter() - t:.2f} s, "
              f"latency {ins.latency_samples()} samples, "
              f"{len(ins.params())} parameters: {list(ins.params())[:8]}")
        sr, bs = 48000, 256
        x = (np.random.default_rng(1).standard_normal((2, bs)) * 0.1).astype(np.float32)
        for _ in range(20):
            chain.process(x, sr)
        t = time.perf_counter()
        for _ in range(200):
            chain.process(x, sr)
        print(f"{(time.perf_counter() - t) / 200 * 1000:.3f} ms per {bs}-frame block")
        text = chain.to_json()
        print(f"settings entry: {len(text)} bytes")
