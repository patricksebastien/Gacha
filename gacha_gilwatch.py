#!/usr/bin/env python3
"""
gacha_gilwatch.py — who holds the GIL when the audio callback stalls.

The audio block is a tenth of a 128-frame buffer; what crackles is the
driver's thread waiting for the GIL. Pure-Python work on the other
threads yields every SWITCH_INTERVAL (gacha_audio), but a C call that
keeps the GIL (a texture upload, a USB write, a garbage collection) can
not be interrupted, and from the callback's side it is invisible: it just
arrives late. This watcher is a thread that sleeps a millisecond at a
time; sleeping lets the GIL go and waking takes it back, so a wake that
comes `threshold` late means the GIL was held that long (or the machine
did not schedule the thread, which a busy CPU would show). On every such
stall it reads every thread's stack at that instant (the culprit is
still in, or just out of, the call that held it) and whether a garbage
collection ran meanwhile, and keeps a short list for the GUI to log.

    w = GilWatch(threshold_ms=4.0)
    w.start()
    ...
    for t, held_ms, what in w.drain():     # from any thread, now and then
        log(f"gil: {held_ms:.1f} ms  {what}")
    w.stop()
"""
import gc
import sys
import threading
import time


class GilWatch(threading.Thread):
    KEEP = 40                      # stalls kept until drained

    def __init__(self, threshold_ms=4.0, period_s=0.001):
        super().__init__(name="gacha-gilwatch", daemon=True)
        self.threshold = threshold_ms / 1000.0
        self.period = period_s
        self.stalls = 0            # in all
        self.worst_ms = 0.0
        self._events = []          # (monotonic, held ms, description)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._gc = None            # (generation, start) of a collection under way
        self._gc_last = None       # (generation, start, end) of the last one
        self._skip = {self.ident}

    # ---- the GC: when and how long
    def _gc_cb(self, phase, info):
        now = time.perf_counter()
        if phase == "start":
            self._gc = (info.get("generation", -1), now)
        elif self._gc is not None:
            self._gc_last = (self._gc[0], self._gc[1], now)
            self._gc = None

    def run(self):
        gc.callbacks.append(self._gc_cb)
        self._skip = {self.ident}
        try:
            last = time.perf_counter()
            while not self._stop.is_set():
                time.sleep(self.period)                 # the GIL let go...
                now = time.perf_counter()               # ...and taken back: this long later
                held = now - last
                last = now
                if held > self.threshold:
                    self._stall(now, held)
        finally:
            try:
                gc.callbacks.remove(self._gc_cb)
            except ValueError:
                pass

    def _stall(self, now, held):
        ms = held * 1000.0
        self.stalls += 1
        self.worst_ms = max(self.worst_ms, ms)
        what = self._describe(now, held)
        with self._lock:
            self._events.append((time.monotonic(), ms, what))
            del self._events[:-self.KEEP]

    def _describe(self, now, held):
        parts = []
        g = self._gc_last
        if g is not None and g[2] > now - held and g[2] - g[1] > 0.25 * held:
            parts.append(f"gc gen{g[0]} {1000.0 * (g[2] - g[1]):.1f} ms")   # a collection, most of it
        g = self._gc
        if g is not None:
            parts.append(f"gc gen{g[0]} still running")
        names = {t.ident: t.name for t in threading.enumerate()}
        main_id = threading.main_thread().ident
        try:
            frames = sys._current_frames()
        except Exception:
            frames = {}
        for tid, frame in frames.items():
            if tid in self._skip:
                continue
            name = names.get(tid, str(tid))
            depth = 3 if tid == main_id else 1
            chain = []
            f = frame
            while f is not None and len(chain) < depth:
                code = f.f_code
                mod = code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
                chain.append(f"{mod}:{code.co_name}:{f.f_lineno}")
                f = f.f_back
            if not chain:
                continue
            top = chain[0]
            # idle threads: waiting in the usual places; not the culprit
            if tid != main_id and any(k in top for k in
                                      ("threading.py:wait", "threading.py:_wait",
                                       "queue.py:get", "selectors.py", "socket.py")):
                continue
            parts.append(f"{'main' if tid == main_id else name}: {' < '.join(chain)}")
        return "  |  ".join(parts) if parts else "no thread in Python code (a C call, or not scheduled)"

    def drain(self):
        """The stalls since the last call, oldest first: (monotonic, held ms, what)."""
        with self._lock:
            ev, self._events = self._events, []
        return ev

    def stop(self):
        self._stop.set()


if __name__ == "__main__":                      # a check: three kinds of hold, seen
    import json
    sys.setswitchinterval(0.0005)               # as the app runs with the stream open
    w = GilWatch(threshold_ms=4.0)
    w.start()
    time.sleep(0.2)
    big = {str(i): list(range(50)) for i in range(60000)}
    json.dumps(big)                             # a C call holding the GIL on the main thread
    time.sleep(0.05)
    cyc = []
    for i in range(400000):
        a = []
        a.append(a)
        cyc.append(a)
    gc.collect()                                # a collection
    time.sleep(0.05)

    def burn():
        s = 0
        for i in range(3_000_000):              # pure Python: yields, should not show
            s += i
    t = threading.Thread(target=burn, name="burner")
    t.start()
    t.join()
    time.sleep(0.05)
    for _t, ms, what in w.drain():
        print(f"{ms:6.1f} ms  {what}")
    print(f"{w.stalls} stalls, worst {w.worst_ms:.1f} ms")
    w.stop()
