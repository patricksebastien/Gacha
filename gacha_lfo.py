#!/usr/bin/env python3
"""
gacha_lfo.py — the LFOs: eight modulators over any knob of the Controls tab.

An LFO slot has a destination (a knob id, the same ids the keyboard and the
MIDI map use), a shape, a rate as a cycle length in bars, a depth, a phase,
a polarity and a mode. The knob's own value is the centre: the engine reads
it as a fraction of the knob's range (the getter), adds depth times the
wave, clips to the range and writes it back through the knob's setter, so
the knob visibly wiggles on screen and every consumer under it (a rack,
the picture, the A/B mix) gets the modulated value. The centre (`base`) is
remembered: turning the knob or moving its CC while the LFO runs is seen
as a value the engine did not write and becomes the new centre, and
switching the slot off, or pointing it elsewhere, writes the centre back.

Phase comes from the live clock (bpm and the monotonic time of a
downbeat), so every LFO is locked to the bar and in phase with the others:
a 1-bar and a 1/4-bar LFO meet on the one. Modes: "cycle" runs on the
clock; "trig" restarts at a trigger (a section start, a drop, the MIDI
button); "one" runs a single cycle from the trigger and holds its end, an
envelope (before any trigger it holds its start).

Shapes, each -1..1 over a cycle: sine (0 rising), triangle (-1 up to 1
and back), saw up (-1 to 1), saw down, square (1 then -1), random step (a
new value each cycle, sample and hold), random smooth (the same, cosine
interpolated), random walk (a drift, each cycle a step from the last, also
interpolated). The random shapes are seeded by the slot and the cycle
number, so the GUI's bar and the sound agree and a test can predict them.

Polarity: bipolar ("±") moves the knob depth/2 of its range either way, so
a centred knob at full depth sweeps the whole range; unipolar ("+") moves
it from the knob upward by depth of the range (an Elektron-style filter
opening). A negative depth flips the wave.

    eng = LfoEngine(8, getter, setter, clock)   # getter(id) -> 0..1 or None,
    eng.configure(0, on=True, dest="afx_drums_lowpass", shape="sine",  # setter(id, x),
                  rate="1 bar", depth=0.6, phase=0.0, unipolar=False, mode="cycle")
    eng.tick(time.monotonic())                 # from a timer, 100 times a second
    eng.trigger(time.monotonic())              # a drop: trig and one-shot slots restart
"""
import math
import random

SHAPES = ["sine", "triangle", "saw up", "saw down", "square",
          "random step", "random smooth", "random walk"]
RATES = [("8 bars", 8.0), ("4 bars", 4.0), ("2 bars", 2.0), ("1 bar", 1.0), ("1/2", 0.5),
         ("1/4", 0.25), ("1/8", 0.125), ("1/16", 1 / 16), ("1/32", 1 / 32)]
RATE_DEFAULT = "1 bar"
MODES = ["cycle", "trig", "one-shot"]
N_SLOTS = 8
EPS = 1e-6


def _rand(seed, k):
    """A reproducible value in -1..1 for cycle `k` of a slot seeded `seed`."""
    return random.Random((seed + 1) * 1000003 + k).uniform(-1.0, 1.0)


def _lerp_cos(a, b, t):
    return a + (b - a) * (1.0 - math.cos(math.pi * t)) / 2.0


class Lfo:
    """One slot: its settings and what the engine keeps between ticks."""

    def __init__(self, seed):
        self.seed = seed
        self.on = False
        self.dest = ""
        self.shape = "sine"
        self.rate = RATE_DEFAULT
        self.depth = 0.5                 # -1..1
        self.phase = 0.0                 # 0..1 of a cycle
        self.unipolar = False
        self.mode = "cycle"
        # runtime
        self.base = None                 # the centre, the knob as you set it
        self.written = None              # what the engine wrote last (re-read)
        self.trig_t = None               # monotonic time of the last trigger
        self.walk = {}                   # cycle k -> the walk's value there
        self.value = 0.0                 # the wave, -1..1, for the display
        self.out = None                  # the fraction written, for the display

    @property
    def bars(self):
        return dict(RATES).get(self.rate, 1.0)

    def settings(self):
        return {"on": self.on, "dest": self.dest, "shape": self.shape, "rate": self.rate,
                "depth": self.depth, "phase": self.phase, "unipolar": self.unipolar,
                "mode": self.mode}

    def wave(self, p, k):
        """The shape at phase p (0..1) in cycle k, -1..1."""
        s = self.shape
        if s == "sine":
            return math.sin(2 * math.pi * p)
        if s == "triangle":
            return 1.0 - 4.0 * abs(p - 0.5)
        if s == "saw up":
            return 2.0 * p - 1.0
        if s == "saw down":
            return 1.0 - 2.0 * p
        if s == "square":
            return 1.0 if p < 0.5 else -1.0
        if s == "random step":
            return _rand(self.seed, k)
        if s == "random smooth":
            return _lerp_cos(_rand(self.seed, k), _rand(self.seed, k + 1), p)
        if s == "random walk":
            return _lerp_cos(self._walk(k), self._walk(k + 1), p)
        return 0.0

    def _walk(self, k):
        """The walk's value at cycle k: a step of up to half the range from
        the cycle before, kept inside -1..1; the walk starts at 0 on cycle 0
        (or at the first cycle asked for after a trigger)."""
        w = self.walk
        if k in w:
            return w[k]
        if not w or k < min(w):
            v = 0.0
        else:
            prev = max(x for x in w if x < k)
            v = w[prev]
            for j in range(prev + 1, k + 1):
                v = max(-1.0, min(1.0, v + 0.5 * _rand(self.seed, j)))
                w[j] = v
            return w[k]
        w[k] = v
        return v


class LfoEngine:
    """The slots and the clock; `tick()` writes every slot that is on."""

    def __init__(self, n=N_SLOTS, getter=None, setter=None, clock=None):
        self.slots = [Lfo(i) for i in range(n)]
        self.getter = getter or (lambda ident: None)
        self.setter = setter or (lambda ident, x: None)
        self.clock = clock or (lambda: (120.0, 0.0))      # (bpm, anchor: a downbeat's time)

    # ---- settings
    def configure(self, i, **kw):
        """Change a slot's settings; off, or a new destination, writes the
        old destination's centre back first."""
        s = self.slots[i]
        new_dest = kw.get("dest", s.dest)
        new_on = kw.get("on", s.on)
        if s.base is not None and (not new_on or new_dest != s.dest):
            self._restore(s)
        for key, val in kw.items():
            if hasattr(s, key):
                setattr(s, key, val)
        if "mode" in kw or "shape" in kw:
            s.walk = {}

    def _restore(self, s):
        """The centre back on the knob, the runtime cleared."""
        if s.base is not None and s.dest:
            self.setter(s.dest, s.base)
        s.base = s.written = s.out = None
        s.value = 0.0

    def set_on(self, i, on):
        self.configure(i, on=bool(on))

    def all_off(self):
        for i in range(len(self.slots)):
            self.set_on(i, False)

    def restore_all(self):
        """Every centre back on its knob without changing the on states:
        what to do at quit, so the settings keep what you set."""
        for s in self.slots:
            self._restore(s)

    def trigger(self, now):
        """A trigger (a section start, a drop, the button): trig and
        one-shot slots restart their cycle here."""
        for s in self.slots:
            if s.mode in ("trig", "one-shot"):
                s.trig_t = now
                s.walk = {}

    # ---- the clock
    def phase(self, s, now, bpm, anchor):
        """(phase 0..1, cycle number) of slot s at `now`."""
        cycle_s = s.bars * 4 * 60.0 / max(1e-6, bpm)
        if s.mode == "cycle" or s.trig_t is None and s.mode == "trig":
            t = (now - anchor) / cycle_s + s.phase
        elif s.mode == "trig":
            t = (now - s.trig_t) / cycle_s + s.phase
        else:                                            # one-shot: holds its end
            t = s.phase if s.trig_t is None else min((now - s.trig_t) / cycle_s + s.phase,
                                                      1.0 - EPS)
        k = math.floor(t)
        return t - k, int(k)

    def tick(self, now):
        """Every slot that is on: read the knob, adopt a value the engine
        did not write as the new centre, compute, write, re-read."""
        bpm, anchor = self.clock()
        for s in self.slots:
            if not s.on or not s.dest:
                continue
            cur = self.getter(s.dest)
            if cur is None:                              # nothing there (an empty VST slot)
                continue
            if s.base is None or s.written is None or abs(cur - s.written) > EPS:
                s.base = cur                             # yours, not ours: the new centre
            p, k = self.phase(s, now, bpm, anchor)
            w = s.wave(p, k)
            if s.unipolar:
                out = s.base + s.depth * (w + 1.0) / 2.0
            else:
                out = s.base + 0.5 * s.depth * w
            out = max(0.0, min(1.0, out))
            self.setter(s.dest, out)
            back = self.getter(s.dest)
            s.written = out if back is None else back    # as the knob quantised it
            s.value, s.out = w, out

    def summary(self, i):
        """One line for a status: '3 drums: lowpass ~ 1/4 60%'."""
        s = self.slots[i]
        if not s.on or not s.dest:
            return ""
        return f"{i + 1} {s.dest} {s.shape} {s.rate} {round(s.depth * 100):+d}%"
