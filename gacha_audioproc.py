#!/usr/bin/env python3
"""
gacha_audioproc.py — the audio engine in a process of its own.

Why: the audio thread needs the interpreter's lock (the GIL) for every
block, and in the app's process the GUI holds it for 10 to 40 ms at a time
(a section change, a video opening, a busy repaint), which at 256 frames
is a hole in the sound, heard as crackle. A process has a GIL of its own,
so the engine (gacha_audio.
LiveAudio with its players, racks, freeze, metronome and the VST inserts)
runs here, in a bare process with nothing else in it, and the GUI keeps
holding its own lock for as long as it likes. This is how DAWs are built.

How the GUI sees it: AudioProxy has LiveAudio's names (player, song,
racks, freeze, metro, inserts, switch(), pad_hit(), stats(), drain(),
record_start() and so on) and `audio_host().inserts` is the VST chain, so
gacha.py reads and writes them as before. Underneath, a proxy is a path
into the engine's object tree:

    la.player.gap = True            -> ("set", path, value)      no answer
    la.racks["live"].amounts[k] = v -> the same, an item step in the path
    la.switch("stem", "drums", on)  -> ("call", path, args)      no answer
    la.player.fire(clip, now=True)  -> a call with an answer (SYNC below)
    la.player.playing, pl.pos_s()   -> read from the last SNAPSHOT
    la.player.queue(stems, ...)     -> the stems go through shared memory

The engine sends a snapshot of the state the GUI reads (SNAP_LA and the
inserts) fifty times a second and with every answer, so reads cost
nothing and are at most 20 ms old; a value the GUI itself wrote is
answered from the GUI's own copy until the engine confirms it applied
(each command carries a sequence number, each snapshot the last applied),
so "if pl.swapped: pl.swapped = False" behaves. The live analyzer's
blocks (drain()) stream back when first asked for. The MIDI out stays in
the GUI (the Erae mirror and the M:C control need its port there): the
engine's notes come back as ("midi", method, args) and are replayed on
it, with their perf_counter due times, which both processes share.

The engine process: the main thread opens, starts and stops the stream
(ASIO wants the thread that made it) and shows plugin editors (a window
of this process); a reader thread applies commands as they come, as the
GUI thread did in-process; a snapshot thread sends the state. It is
started by audio_host() the first time it is needed and quits with the
app (shutdown()).

    python gacha_audioproc.py      # the engine, started by the app with
                                   # GACHA_AUDIO_ADDR and GACHA_AUDIO_KEY set
"""
import gc
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from multiprocessing import shared_memory
from multiprocessing.connection import Client, Listener

import numpy as np

ATTR, ITEM, CALL = "a", "i", "c"          # the steps of a path: attribute, item, call()
SNAP_PERIOD = 0.02                        # s between snapshots
SHM_MIN = 1 << 20                         # a dict of arrays this big goes through shared memory
LA = ((ATTR, "la"),)                      # the engine's root paths
INSERTS = ((ATTR, "inserts"),)


class Ref:
    """A path into the engine's tree, standing for the object in a message."""
    __slots__ = ("path",)

    def __init__(self, path):
        self.path = path


class ShmArrays:
    """A dict of arrays, laid out in a shared memory segment."""
    __slots__ = ("name", "layout")

    def __init__(self, name, layout):
        self.name, self.layout = name, layout     # [(key, shape, dtype str, offset, nbytes)]


class NodeList:
    """In a snapshot: a sequence of engine objects (the inserts), by path."""
    __slots__ = ("paths",)

    def __init__(self, paths):
        self.paths = tuple(paths)


# ----------------------------------------------------------------- the GUI side
# the state the GUI reads, relative to the LiveAudio (CALL = the method's answer)
SNAP_LA = [
    ((ATTR, "running"),), ((ATTR, "error"),), ((ATTR, "mode"),), ((ATTR, "sr"),),
    ((ATTR, "block"),), ((ATTR, "in_name"),), ((ATTR, "out_latency_ms"),),
    ((ATTR, "buffer_ms"),), ((ATTR, "ping_result"),), ((ATTR, "midi_sent"),),
    ((ATTR, "in_peak"),), ((ATTR, "out_peak"),),
    ((ATTR, "stats"), (CALL, None)),
    ((ATTR, "player"), (ATTR, "playing")), ((ATTR, "player"), (ATTR, "bpm")),
    ((ATTR, "player"), (ATTR, "bars")), ((ATTR, "player"), (ATTR, "n")),
    ((ATTR, "player"), (ATTR, "sr")), ((ATTR, "player"), (ATTR, "pending")),
    ((ATTR, "player"), (ATTR, "repeat")), ((ATTR, "player"), (ATTR, "filling")),
    ((ATTR, "player"), (ATTR, "beat_frames")), ((ATTR, "player"), (ATTR, "fill_armed")),
    ((ATTR, "player"), (ATTR, "brake_bars")), ((ATTR, "player"), (ATTR, "swapped")),
    ((ATTR, "player"), (ATTR, "dropped")), ((ATTR, "player"), (ATTR, "locked")),
    ((ATTR, "player"), (ATTR, "replaced")),
    ((ATTR, "player"), (ATTR, "speed_sync")), ((ATTR, "player"), (ATTR, "burst_fired")),
    ((ATTR, "player"), (ATTR, "gap")),
    ((ATTR, "player"), (ATTR, "head_state"), (CALL, None)),
    ((ATTR, "player"), (ATTR, "heard_s"), (CALL, None)),
    ((ATTR, "player"), (ATTR, "frag_state"), (CALL, None)),
    ((ATTR, "player"), (ATTR, "fill_state"), (CALL, None)),
    ((ATTR, "player"), (ATTR, "repeat_window"), (CALL, None)),
    ((ATTR, "player"), (ATTR, "repeat_state"), (CALL, None)),
    ((ATTR, "player"), (ATTR, "waiting_for"), (CALL, None)),
    ((ATTR, "player"), (ATTR, "pos_s"), (CALL, None)),
    ((ATTR, "song"), (ATTR, "loaded")), ((ATTR, "song"), (ATTR, "error")),
    ((ATTR, "song"), (ATTR, "state")), ((ATTR, "song"), (ATTR, "stems")),
    ((ATTR, "song"), (ATTR, "ended")),
    ((ATTR, "song"), (ATTR, "position_ms"), (CALL, None)),
    ((ATTR, "song"), (ATTR, "duration_ms"), (CALL, None)),
    ((ATTR, "freeze"), (ATTR, "active")), ((ATTR, "freeze"), (ATTR, "held")),
    ((ATTR, "freeze"), (ATTR, "state"), (CALL, None)),
]
# what an insert shows the GUI
SNAP_INSERT = [((ATTR, "name"),), ((ATTR, "path"),), ((ATTR, "on"),), ((ATTR, "error"),),
               ((ATTR, "params"), (CALL, None)), ((ATTR, "latency_samples"), (CALL, None))]
# calls that wait for their answer (the rest return at once, with None)
SYNC = {
    LA + ((ATTR, "player"), (ATTR, "fire")): 5.0,
    LA + ((ATTR, "player"), (ATTR, "queue")): 20.0,
    LA + ((ATTR, "player"), (ATTR, "replace")): 20.0,
    LA + ((ATTR, "record_stop"),): 20.0,
    INSERTS + ((ATTR, "add"),): 120.0,
    INSERTS + ((ATTR, "remove"),): 10.0,
    INSERTS + ((ATTR, "move"),): 10.0,
    INSERTS + ((ATTR, "clear"),): 10.0,
    INSERTS + ((ATTR, "to_json"),): 10.0,
    INSERTS + ((ATTR, "load_json"),): 300.0,
    ((ATTR, "new"),): 20.0,
    ((ATTR, "close"),): 15.0,
}
_MISSING = object()


class RemoteNode:
    """A path into the engine: attributes and items are paths below it,
    calling it is a call, reads come from the snapshot or the GUI's own
    writes, writes go to the engine."""
    __slots__ = ("_h", "_p")

    def __init__(self, host, path):
        object.__setattr__(self, "_h", host)
        object.__setattr__(self, "_p", path)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._h.read(self._p + ((ATTR, name),))

    def __setattr__(self, name, value):
        self._h.write(self._p + ((ATTR, name),), value)

    def __getitem__(self, key):
        return self._h.read(self._p + ((ITEM, key),))

    def __setitem__(self, key, value):
        self._h.write(self._p + ((ITEM, key),), value)

    def __call__(self, *args, **kwargs):
        return self._h.call(self._p, args, kwargs)

    def get(self, key, default=None):
        """dict.get on a dict the GUI fills (on, voice_on)."""
        v = self._h.read(self._p + ((ITEM, key),))
        return default if isinstance(v, RemoteNode) else v

    def __bool__(self):                   # an unknown value reads as nothing
        return False

    def __float__(self):                  # a number the GUI asked for that the engine
        raise TypeError(f"{self!r} is not in the snapshot: add it to SNAP_LA")   # never sends

    __int__ = __float__

    def __eq__(self, other):
        return isinstance(other, RemoteNode) and other._p == self._p

    def __hash__(self):
        return hash(self._p)

    def __repr__(self):
        return "<engine " + ".".join(str(k) for _s, k in self._p if k is not None) + ">"


class AudioProxy(RemoteNode):
    """LiveAudio's stand-in: the same constructor, start(), stop() and
    names; the engine process does the work."""

    def __init__(self, in_name, out_name, block=256, sr=48000, exclusive=True, api=None):
        host = audio_host()
        super().__init__(host, LA)
        kw = dict(in_name=(in_name or None), out_name=out_name, block=int(block), sr=int(sr),
                  exclusive=bool(exclusive), api=api)
        host.new_engine(kw)                   # built there, the stream not yet open

    def __setattr__(self, name, value):
        if name == "midi_out":                # the GUI's port: the notes come back to it
            self._h.set_midi(value)
            return
        RemoteNode.__setattr__(self, name, value)

    def start(self):
        self._h.call(((ATTR, "open"),), (), {})

    def stop(self):
        self._h.call(((ATTR, "close"),), (), {})


class InsertsProxy(RemoteNode):
    """The VST chain in the engine, with the GUI-side extras."""

    def __init__(self, host):
        super().__init__(host, INSERTS)

    def watch(self, on_change, period=1.0):
        """on_change() (from the host's reader thread) when the chain's
        parameter names change; returns the Event that stops it."""
        return self._h.watch_inserts(on_change)

    def edit(self, ins, owner_hwnd=0):
        """The plugin's own window, from the engine process; the event
        "editor_closed" comes back when it closes."""
        self._h.call(((ATTR, "edit"),), (ins, int(owner_hwnd or 0)), {})

    def close_editor(self):
        self._h.call(((ATTR, "close_editor"),), (), {})


class AudioHost:
    """The engine process and the pipe to it, one per app."""

    def __init__(self):
        self._proc = None
        self._listener = None
        self._conn = None
        self._ready = threading.Event()
        self._lock = threading.RLock()            # the pipe's send side, the caches
        self._seq = 0                            # commands sent
        self._applied = 0                        # ...the engine has applied, per the last snapshot
        self._snap = {}                          # path -> value
        self._snap_t = 0.0                       # monotonic of the last snapshot
        self._cache = {}                         # path -> (seq, value): the GUI's own writes
        self._replies = {}                       # reply id -> [Event, ok, value]
        self._blocks = []                        # the analyzer's blocks, drained by drain()
        self._want_blocks = False
        self._midi = None                        # the GUI's MidiOut, the engine's notes land on it
        self._midi_state = None
        self._watchers = []                      # [(Event stop, callback)] on the inserts' names
        self._fp = None
        self._events = {}                        # name -> [callbacks]
        self.log = None                          # callback(text) for the engine's log lines
        self.error = None                        # set when the engine is gone
        self.inserts = InsertsProxy(self)

    # ---- the process
    def launch(self):
        key = secrets.token_bytes(16)
        if sys.platform == "win32":
            family, address = "AF_PIPE", rf"\\.\pipe\gacha-audio-{os.getpid()}-{secrets.token_hex(4)}"
        else:
            import tempfile
            family = "AF_UNIX"
            address = os.path.join(tempfile.gettempdir(), f"gacha-audio-{os.getpid()}-{secrets.token_hex(4)}")
        self._listener = Listener(address, family=family, authkey=key)
        env = dict(os.environ, GACHA_AUDIO_ADDR=address, GACHA_AUDIO_KEY=key.hex(),
                   PYTHONUNBUFFERED="1")
        here = os.path.dirname(os.path.abspath(__file__))
        self._proc = subprocess.Popen([sys.executable, os.path.join(here, "gacha_audioproc.py")],
                                      env=env, cwd=here)
        threading.Thread(target=self._accept, name="gacha-host-accept", daemon=True).start()
        threading.Thread(target=self._watch_proc, name="gacha-host-watch", daemon=True).start()

    def _accept(self):
        try:
            conn = self._listener.accept()
        except Exception as e:
            self._dead(f"audio engine did not connect: {e}")
            return
        self._conn = conn
        self._ready.set()
        threading.Thread(target=self._reader, name="gacha-host-reader", daemon=True).start()

    def _watch_proc(self):
        code = self._proc.wait()
        self._dead(f"audio engine exited (code {code})")
        try:
            self._listener.close()
        except Exception:
            pass

    def _dead(self, msg):
        with self._lock:
            if self.error is None:
                self.error = msg
            self._snap = dict(self._snap)
            self._snap[LA + ((ATTR, "running"),)] = False
            self._snap[LA + ((ATTR, "error"),)] = msg
            for rep in self._replies.values():
                rep[1], rep[2] = False, msg
                rep[0].set()
            self._replies.clear()
        self._ready.set()

    def shutdown(self):
        """The engine quits; called when the app closes."""
        if self._proc is None:
            return
        try:
            self._send(("quit",))
        except Exception:
            pass
        try:
            self._proc.wait(5.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    # ---- the pipe
    def _send(self, msg):
        if self.error:
            raise RuntimeError(self.error)
        if not self._ready.wait(30.0) or self._conn is None:
            self._dead("audio engine did not start in 30 s")
            raise RuntimeError(self.error)
        with self._lock:
            self._conn.send(msg)

    def _reader(self):
        conn = self._conn
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                self._dead("audio engine exited")
                return
            except Exception as e:
                self._dead(f"audio engine pipe: {e}")
                return
            try:
                self._handle(msg)
            except Exception as e:
                print(f"gacha_audioproc: host: {msg[0]}: {e}", file=sys.stderr)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "snap":
            self._apply_snap(msg[1], msg[2], msg[3])
        elif kind == "reply":
            _k, rid, ok, value, applied, t, snap = msg
            self._apply_snap(applied, t, snap)
            with self._lock:
                rep = self._replies.pop(rid, None)
            if rep is not None:
                rep[1], rep[2] = ok, self._decode(value)
                rep[0].set()
        elif kind == "midi":
            mo = self._midi
            if mo is not None:
                getattr(mo, msg[1])(*msg[2])
        elif kind == "blocks":
            with self._lock:
                self._blocks.extend(msg[1])
        elif kind == "event":
            for cb in self._events.get(msg[1], ()):
                cb(*msg[2])
        elif kind == "log":
            if self.log is not None:
                self.log(msg[1])
            else:
                print("engine:", msg[1])

    def _apply_snap(self, applied, t, snap):
        d = {p: self._decode(v) for p, v in snap.items()}
        with self._lock:
            self._snap, self._snap_t = d, t
            self._applied = max(self._applied, applied)
        fp = d.get(INSERTS + ((ATTR, "fingerprint"),))
        if fp is not None and self._fp is not None and fp != self._fp:
            for stop, cb in list(self._watchers):
                if not stop.is_set():
                    cb()
        if fp is not None:
            self._fp = fp
        mo = self._midi                        # the port's state, for the engine's `midi` test
        if mo is not None:
            st = (bool(mo.is_open), bool(mo.mirrored))
            if st != self._midi_state:
                self._midi_state = st
                try:
                    self._send(("set", 0, LA + ((ATTR, "midi_out"), (ATTR, "is_open")), st[0]))
                    self._send(("set", 0, LA + ((ATTR, "midi_out"), (ATTR, "mirrored")), st[1]))
                except Exception:
                    pass

    def _decode(self, v):
        if isinstance(v, Ref):
            return RemoteNode(self, v.path)
        if isinstance(v, NodeList):
            return tuple(RemoteNode(self, path) for path in v.paths)
        if isinstance(v, tuple):
            return tuple(self._decode(x) for x in v)
        if isinstance(v, list):
            return [self._decode(x) for x in v]
        return v

    def _encode(self, v, keep):
        """A value for the pipe: nodes as paths, a big dict of arrays as a
        shared memory segment (kept open in `keep` until answered)."""
        if isinstance(v, RemoteNode):
            return Ref(v._p)
        if isinstance(v, dict):
            if v and all(isinstance(x, np.ndarray) for x in v.values()) \
                    and sum(x.nbytes for x in v.values()) >= SHM_MIN:
                total = sum(x.nbytes for x in v.values())
                shm = shared_memory.SharedMemory(create=True, size=total)
                layout, off = [], 0
                for key, arr in v.items():
                    arr = np.ascontiguousarray(arr)
                    dst = np.ndarray(arr.shape, arr.dtype, buffer=shm.buf[off:off + arr.nbytes])
                    dst[...] = arr
                    del dst
                    layout.append((key, arr.shape, arr.dtype.str, off, arr.nbytes))
                    off += arr.nbytes
                keep.append(shm)
                return ShmArrays(shm.name, layout)
            return {k: self._encode(x, keep) for k, x in v.items()}
        if isinstance(v, tuple):
            return tuple(self._encode(x, keep) for x in v)
        if isinstance(v, list):
            return [self._encode(x, keep) for x in v]
        return v

    # ---- what the proxies do
    def read(self, path):
        with self._lock:
            c = self._cache.get(path)
            if c is not None and (c[0] > self._applied or path not in self._snap):
                return c[1]
            if path in self._snap:
                return self._snap[path]
        return RemoteNode(self, path)

    def write(self, path, value):
        keep = []
        enc = self._encode(value, keep)
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._cache[path] = (seq, value)
        self._send(("set", seq, path, enc))

    def call(self, path, args, kwargs):
        key = path + ((CALL, None),)
        if not args and not kwargs:
            with self._lock:
                if key in self._snap:
                    v = self._snap[key]
                    if path == LA + ((ATTR, "song"), (ATTR, "position_ms")) and v is not None \
                            and self._snap.get(LA + ((ATTR, "song"), (ATTR, "state"))) == "playing":
                        v = v + 1000.0 * (time.monotonic() - self._snap_t)   # since the snapshot
                    return v
        if path == INSERTS + ((ATTR, "knob_value"),) and len(args) == 1:
            with self._lock:
                vals = self._snap.get(INSERTS + ((ATTR, "knob_values"),)) or ()
            n = int(args[0])
            return vals[n] if 0 <= n < len(vals) else None
        if path == LA + ((ATTR, "drain"),):
            with self._lock:
                blocks, self._blocks = self._blocks, []
            if not self._want_blocks:
                self._want_blocks = True
                self._send(("want_blocks", True))
            return blocks
        keep = []
        enc_args = tuple(self._encode(a, keep) for a in args)
        enc_kwargs = {k: self._encode(v, keep) for k, v in kwargs.items()}
        timeout = SYNC.get(path)
        sync = timeout is not None or bool(keep)
        with self._lock:
            self._seq += 1
            seq = self._seq
            rep = None
            if sync:
                rep = [threading.Event(), False, None]
                self._replies[seq] = rep
        try:
            self._send(("call", seq, path, enc_args, enc_kwargs, seq if sync else None))
            if not sync:
                return None
            if not rep[0].wait(timeout or 10.0):
                with self._lock:
                    self._replies.pop(seq, None)
                raise RuntimeError(f"audio engine: no answer to {_pstr(path)} in {timeout or 10.0:.0f} s")
            if not rep[1]:
                raise RuntimeError(str(rep[2]))
            return rep[2]
        finally:
            for shm in keep:                     # the engine has mapped it, or failed
                try:
                    shm.close()
                except Exception:
                    pass

    def new_engine(self, kw):
        """A fresh LiveAudio in the engine (not started); the proxy answers
        its constructor's values until the engine says otherwise."""
        with self._lock:
            for k, v in kw.items():
                self._cache[LA + ((ATTR, k),)] = (0, v)
        self.call(((ATTR, "new"),), (), kw)

    def set_midi(self, mo):
        self._midi = mo
        self._midi_state = None

    def watch_inserts(self, cb):
        stop = threading.Event()
        self._watchers.append((stop, cb))
        return stop

    def on_event(self, name, cb):
        self._events.setdefault(name, []).append(cb)

    def priority(self, high, gc_off):
        """The engine process's class and its GC, from the Settings ticks."""
        try:
            self.call(((ATTR, "priority"),), (bool(high), bool(gc_off)), {})
        except Exception:
            pass


_HOST = None


def audio_host():
    """The one engine process of this app, started on first use."""
    global _HOST
    if _HOST is None:
        _HOST = AudioHost()
        _HOST.launch()
    return _HOST


def _pstr(path):
    return ".".join(str(k) for _s, k in path if k is not None)


# -------------------------------------------------------------- the engine side
def _resolve(root, path):
    obj = root
    for kind, key in path:
        if kind == ATTR:
            obj = getattr(obj, key)
        elif kind == ITEM:
            obj = obj[key]
        else:
            obj = obj()
    return obj


def _plain(v, depth=0):
    """A snapshot value: numbers, strings, bools, None, and tuples, lists
    and dicts of them; numpy scalars as numbers, small arrays as lists."""
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 64 else None
    if depth > 4:
        return None
    if isinstance(v, dict):
        return {k: _plain(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (tuple, list, set, frozenset)):
        return type(v)(_plain(x, depth + 1) for x in v) if not isinstance(v, (set, frozenset)) \
            else [_plain(x, depth + 1) for x in v]
    return None


class _MidiRelay:
    """The engine's MIDI out: what the audio thread hands it goes back to
    the GUI's port, from a thread of its own (no pipe write on the audio
    thread). `is_open` and `mirrored` mirror the GUI's port."""

    def __init__(self, send):
        self._send = send
        self.is_open = False
        self.mirrored = False
        self._q = deque()
        self._ev = threading.Event()
        threading.Thread(target=self._run, name="gacha-midi-relay", daemon=True).start()

    def _push(self, name, args):
        self._q.append((name, args))
        self._ev.set()

    def note(self, channel, note, velocity, due, gate=0.05):
        self._push("note", (int(channel), int(note), int(velocity), float(due), float(gate)))

    def grid(self, beat_t, beat_s):
        self._push("grid", (None if beat_t is None else float(beat_t), float(beat_s)))

    def start(self, due):
        self._push("start", (float(due),))

    def stop(self):
        self._push("stop", ())

    def _run(self):
        while True:
            self._ev.wait()
            self._ev.clear()
            while self._q:
                name, args = self._q.popleft()
                try:
                    self._send(("midi", name, args))
                except Exception:
                    return


class _Root:
    """The engine's object tree: `la` the LiveAudio (None before "new"),
    `inserts` the VST chain (it outlives the stream)."""

    def __init__(self):
        self.la = None
        self.inserts = None
        self.midi = None


class Engine:
    def __init__(self, conn):
        import gacha_audio
        from gacha_vst import InsertChain
        self.ga = gacha_audio
        self.conn = conn
        self.send_lock = threading.Lock()
        self.root = _Root()
        self.root.inserts = InsertChain()
        self.root.midi = _MidiRelay(self.send)
        self.applied = 0
        self.want_blocks = False
        self.mainq = queue.Queue()
        self.editor_close = threading.Event()
        self.shms = deque()                       # the sections' segments, the last few
        self.keep = set()                         # every Insert ever loaded (quit is a hard exit)
        self._ins_cache = {}                      # Insert -> its slow answers (params, latency)
        self._ins_lock = threading.Lock()         # one asker at a time (the answer and the snapshot)
        self._fp = None                           # the chain's parameter names, hashed
        self._fp_at = 0.0                         # ...last asked
        self.editor_open = False                  # a plugin's window is up (main thread)
        self.stop = threading.Event()

    # ---- the pipe
    def send(self, msg):
        with self.send_lock:
            self.conn.send(msg)

    def log(self, text):
        print("engine:", text, flush=True)
        try:
            self.send(("log", text))
        except Exception:
            pass

    def reply(self, rid, ok, value):
        self.send(("reply", rid, ok, self._encode(value), self.applied, time.monotonic(),
                   self.snapshot(fresh=True)))

    def _encode(self, v):
        try:
            from gacha_vst import Insert
        except Exception:
            Insert = ()
        if Insert and isinstance(v, Insert):
            items = self.root.inserts.items
            for i, ins in enumerate(items):
                if ins is v:
                    return Ref(INSERTS + ((ATTR, "items"), (ITEM, i)))
            return None
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, tuple) and len(x) == 2
                                                       and isinstance(x[0], np.ndarray) for x in v):
            return list(v)                        # record_stop's blocks: arrays travel as they are
        if isinstance(v, (str, bytes, bool, int, float)) or v is None:
            return v
        if isinstance(v, np.ndarray):
            return v
        return _plain(v)

    def _decode(self, v):
        if isinstance(v, Ref):
            return _resolve(self.root, v.path)
        if isinstance(v, ShmArrays):
            shm = shared_memory.SharedMemory(name=v.name)
            self.shms.append(shm)
            while len(self.shms) > 4:             # an old section's segment: let it go once
                old = self.shms.popleft()         # nothing looks at it any more
                try:
                    old.close()
                except BufferError:
                    self.shms.append(old)         # still viewed: next time
                    break
            out = {}
            for key, shape, dtype, off, nbytes in v.layout:
                out[key] = np.ndarray(shape, np.dtype(dtype), buffer=shm.buf[off:off + nbytes])
            return out
        if isinstance(v, tuple):
            return tuple(self._decode(x) for x in v)
        if isinstance(v, list):
            return [self._decode(x) for x in v]
        if isinstance(v, dict):
            return {k: self._decode(x) for k, x in v.items()}
        return v

    # ---- the snapshot
    def snapshot(self, fresh=False):
        """The state the GUI reads. An insert's parameter list and latency
        are slow to ask (plugdata enumerates its patch), so they are asked
        once a second and after an answered call (`fresh`), not per snapshot."""
        d = {}
        la = self.root.la
        if la is not None:
            for p in SNAP_LA:
                try:
                    v = _resolve(la, p)
                except Exception:
                    v = None
                d[LA + p] = _plain(self._shape(p, v))
        ch = self.root.inserts
        d[INSERTS + ((ATTR, "bypass"),)] = bool(ch.bypass)
        d[INSERTS + ((ATTR, "knobs"),)] = list(ch.knobs)
        try:
            d[INSERTS + ((ATTR, "knob_values"),)] = [ch.knob_value(n) for n in range(len(ch.knobs))]
        except Exception:
            d[INSERTS + ((ATTR, "knob_values"),)] = [None] * len(ch.knobs)
        items = ch.items
        self.keep.update(items)                   # a removed plugin is kept, never destroyed
        d[INSERTS + ((ATTR, "items"),)] = NodeList(INSERTS + ((ATTR, "items"), (ITEM, i))
                                                   for i in range(len(items)))
        now = time.monotonic()
        # the chain's parameter names: a plugin renames or adds some while
        # its window is open (plugdata, a [param] put in the patch), so they
        # are asked once a second while one is, and after an answered call;
        # a change refreshes the slow answers below
        changed = False
        if fresh or (self.editor_open and now - self._fp_at > 1.0):
            self._fp_at = now
            try:
                fp = hash(ch.fingerprint()) if items else None
            except Exception:
                fp = self._fp
            changed = fp != self._fp
            self._fp = fp
        d[INSERTS + ((ATTR, "fingerprint"),)] = self._fp
        force = fresh or changed
        for i, ins in enumerate(items):
            base = INSERTS + ((ATTR, "items"), (ITEM, i))
            for p in SNAP_INSERT:
                if p[-1][0] != CALL:              # a plain attribute, cheap
                    try:
                        d[base + p] = _plain(_resolve(ins, p))
                    except Exception:
                        d[base + p] = None
            # the slow answers, under the lock: an asker that waited for
            # another finds them in the cache and does not ask again
            with self._ins_lock:
                at, c = self._ins_cache.get(ins, (0.0, None))
                if c is None or (force and at < now):   # not asked since this snapshot began
                    c = {}
                    for p in SNAP_INSERT:
                        if p[-1][0] == CALL:
                            try:
                                c[p] = _plain(_resolve(ins, p))
                            except Exception:
                                c[p] = None
                    self._ins_cache[ins] = (time.monotonic(), c)
                for p, v in c.items():
                    d[base + p] = v
        return d

    @staticmethod
    def _shape(p, v):
        """The few values the GUI reads only the shape of."""
        name = p[-1][1] if p[-1][0] == ATTR else p[-2][1]
        if name == "pending":
            return None if v is None else True
        if name == "stems" and isinstance(v, dict):
            return list(v.keys())
        if name == "fill_armed" and v is not None:
            return (v[0], None)
        return v

    def _snapshots(self):
        while not self.stop.is_set():
            time.sleep(SNAP_PERIOD)
            try:
                d = self.snapshot()
                self.send(("snap", self.applied, time.monotonic(), d))
                la = self.root.la
                if self.want_blocks and la is not None and la.running:
                    blocks = la.drain()
                    if blocks:
                        self.send(("blocks", blocks))
            except (OSError, EOFError, BrokenPipeError):
                self.stop.set()
                return
            except Exception as e:
                print("engine: snapshot:", e, file=sys.stderr, flush=True)

    # ---- the commands
    def _reader(self):
        while not self.stop.is_set():
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                self.stop.set()
                self.mainq.put(("quit", None, None))
                return
            kind = msg[0]
            try:
                if kind == "set":
                    _k, seq, path, value = msg
                    value = self._decode(value)
                    parent = _resolve(self.root, path[:-1])
                    step, key = path[-1]
                    if step == ATTR:
                        setattr(parent, key, value)
                    else:
                        parent[key] = value
                    if seq:
                        self.applied = max(self.applied, seq)
                elif kind == "call":
                    _k, seq, path, args, kwargs, rid = msg
                    self._call(seq, path, args, kwargs, rid)
                elif kind == "want_blocks":
                    self.want_blocks = bool(msg[1])
                elif kind == "quit":
                    self.stop.set()
                    self.mainq.put(("quit", None, None))
                    return
            except Exception as e:
                self.log(f"{kind} {_pstr(msg[2]) if len(msg) > 2 and isinstance(msg[2], tuple) else ''}: {e}")
                if kind == "call" and msg[5] is not None:
                    try:
                        self.reply(msg[5], False, f"{e}")
                    except Exception:
                        pass

    def _call(self, seq, path, args, kwargs, rid):
        args = self._decode(args)
        kwargs = self._decode(kwargs)
        if len(path) == 1 and path[0][0] == ATTR and path[0][1] in (
                "new", "open", "close", "edit", "close_editor", "priority", "quit"):
            op = path[0][1]
            if op == "new":
                la = self.ga.LiveAudio(**kwargs)
                la.inserts = self.root.inserts
                la.midi_out = self.root.midi
                self.root.la = la
                self.applied = max(self.applied, seq)
                self.reply(rid, True, None)
                return
            if op == "priority":
                self.ga.set_process_priority(bool(args[0]))
                (gc.disable if args[1] else gc.enable)()
                self.applied = max(self.applied, seq)
                return
            if op == "close_editor":
                self.editor_close.set()
                self.applied = max(self.applied, seq)
                return
            self.mainq.put((op, (seq, args, kwargs), rid))   # the stream and the windows: main thread
            return
        if path[:1] == INSERTS:
            self.keep.update(self.root.inserts.items)   # before a remove could drop one
            if path[1][1] in ("add", "load_json"):
                # pedalboard binds a plugin to the thread that loaded it (its
                # editor opens only from there): the main thread, like the editors
                self.mainq.put(("inserts_call", (seq, path, (args, kwargs)), rid))
                return
        fn = _resolve(self.root, path)
        result = fn(*args, **kwargs)
        self.applied = max(self.applied, seq)
        if rid is not None:
            self.reply(rid, True, result)

    # ---- the main thread: the stream and the editors
    def run(self):
        threading.Thread(target=self._reader, name="gacha-engine-reader", daemon=True).start()
        threading.Thread(target=self._snapshots, name="gacha-engine-snap", daemon=True).start()
        while True:
            op, payload, rid = self.mainq.get()
            la = self.root.la
            try:
                if op == "quit":
                    if la is not None:
                        la.stop()
                    # a hard exit: a plugin's teardown (its DLL, JUCE's
                    # threads) hangs or crashes the interpreter's own exit,
                    # so no plugin is ever destroyed here (see `keep`)
                    sys.stdout.flush()
                    os._exit(0)
                seq, args, kwargs = payload
                if op == "inserts_call":          # add, load_json: a plugin loads here
                    path = args
                    t0 = time.perf_counter()
                    result = _resolve(self.root, path)(*kwargs[0], **kwargs[1])
                    t1 = time.perf_counter()
                    self.keep.update(self.root.inserts.items)
                    self.applied = max(self.applied, seq)
                    if rid is not None:
                        self.reply(rid, True, result)
                    self.log(f"vst: {_pstr(path[1:])}: loaded in {t1 - t0:.1f} s, its parameters "
                             f"asked in {time.perf_counter() - t1:.1f} s")
                    continue
                if op == "open":
                    la.start()
                    la.ready.wait(15.0)           # other hosts than ASIO open on a thread
                    self.log(f"stream: {la.error or la.mode}")
                elif op == "close":
                    la.stop()
                elif op == "edit":
                    self._edit(args[0], args[1] if len(args) > 1 else 0)
                self.applied = max(self.applied, seq)
                if rid is not None:
                    self.reply(rid, True, None)
            except Exception as e:
                self.log(f"{op}: {e}")
                if rid is not None:
                    try:
                        self.reply(rid, False, f"{e}")
                    except Exception:
                        pass

    def _edit(self, ins, owner_hwnd):
        from gacha_vst import place_editor_window
        self.log(f"vst: {ins.name} editor open")
        place_editor_window(int(owner_hwnd) if owner_hwnd else None)
        self.editor_close.clear()
        self.editor_open = True
        try:
            ins.plugin.show_editor(self.editor_close)
        except Exception as e:
            self.log(f"vst: {ins.name} editor: {e}")
        finally:
            self.editor_open = False
        self.send(("event", "editor_closed", (ins.name,)))


def serve(address, key):
    conn = None
    for _ in range(50):
        try:
            conn = Client(address, authkey=key)
            break
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            time.sleep(0.1)
    if conn is None:
        print("gacha_audioproc: no listener at", address, file=sys.stderr)
        return 1
    eng = Engine(conn)
    eng.run()
    return 0


if __name__ == "__main__":
    addr = os.environ.get("GACHA_AUDIO_ADDR")
    key = os.environ.get("GACHA_AUDIO_KEY")
    if not addr or not key:
        print(__doc__)
        sys.exit(2)
    # the engine runs the module's code, not this __main__ copy of it, so
    # that Ref and the others unpickle as the classes it tests against
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gacha_audioproc as _mod
    sys.exit(_mod.serve(addr, bytes.fromhex(key)))
