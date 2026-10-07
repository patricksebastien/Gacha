#!/usr/bin/env python3
"""
gacha_macro.py — the macros: eight knobs that each move many knobs at once.

A macro is one knob, 0..1, and up to eight destinations: a knob id (the
ids the keyboard, the MIDI map and the LFOs use) with a `lo` and a `hi`,
fractions of that knob's range. Moving the macro sets every destination
to lo + (hi - lo) * macro, so at 0 every knob sits at its lo, at 1 at its
hi, and a lo above hi runs the knob the other way. That is the Ableton
model: absolute, not an offset around the knob's own value. The macro
writes only when it moves, through the same setter the scene morph uses,
so the destination knobs move on screen and whatever sits under them
(a rack, the picture, the mix) follows; nothing runs on a timer. Last
writer wins: a hand turn, a scene load or an LFO can move a destination
in between, and the next macro move takes it back. An LFO on a macro
sees the macro's own knob, so one LFO wiggles eight knobs through their
ranges at once.

`capture(i, which, read)` is how a macro is set up live: with `read` a
getter over the knob ids, every destination's lo (which="lo") or hi
(which="hi") takes the value its knob sits at now. Dial the quiet state,
set lo; dial the peak, set hi; the macro now morphs between them.

    eng = MacroEngine(8, setter)               # setter(ident, x)
    eng.assign(0, 0, "afx_drums_lowpass", 0.2, 0.9)
    eng.assign(0, 1, "fx_strobe", 0.0, 1.0)
    eng.move(0, 0.5)                           # lowpass -> 0.55, strobe -> 0.5
    text = eng.to_json(); eng.load_json(text)
"""
import json

N_SLOTS = 8
N_DESTS = 8
EPS = 1e-9


def _frac(x, default=0.0):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default


class Dest:
    """One destination of a macro: a knob id and its range."""

    def __init__(self, ident="", lo=0.0, hi=1.0):
        self.ident = ident
        self.lo = lo
        self.hi = hi

    def at(self, x):
        """The knob's fraction for macro value x."""
        return self.lo + (self.hi - self.lo) * x

    def settings(self):
        return {"ident": self.ident, "lo": round(self.lo, 6), "hi": round(self.hi, 6)}

    @classmethod
    def restore(cls, d):
        if not isinstance(d, dict):
            return cls()
        return cls(str(d.get("ident", "") or ""), _frac(d.get("lo", 0.0), 0.0),
                   _frac(d.get("hi", 1.0), 1.0))


class Macro:
    """One slot: a name, its value, its destinations."""

    def __init__(self, i, n_dests=N_DESTS):
        self.i = i
        self.name = ""
        self.value = 0.0
        self.dests = [Dest() for _ in range(n_dests)]

    @property
    def label(self):
        return self.name.strip() or f"macro {self.i + 1}"

    @property
    def active(self):
        """The destinations that point somewhere."""
        return [d for d in self.dests if d.ident]

    def plan(self, x):
        """{ident: fraction} for macro value x; a knob named twice takes
        the last row."""
        return {d.ident: max(0.0, min(1.0, d.at(x))) for d in self.active}

    def settings(self):
        return {"name": self.name, "value": round(self.value, 6),
                "dests": [d.settings() for d in self.dests]}

    def restore(self, d):
        if not isinstance(d, dict):
            return
        self.name = str(d.get("name", "") or "")
        self.value = _frac(d.get("value", 0.0))
        rows = d.get("dests", [])
        if isinstance(rows, list):
            for j, row in enumerate(rows[:len(self.dests)]):
                self.dests[j] = Dest.restore(row)


class MacroEngine:
    """The slots and the setter; `move()` writes a slot's destinations."""

    def __init__(self, n=N_SLOTS, setter=None, n_dests=N_DESTS):
        self.slots = [Macro(i, n_dests) for i in range(n)]
        self.setter = setter or (lambda ident, x: None)

    def assign(self, i, j, ident, lo=None, hi=None):
        """Destination j of macro i: the knob, and its range when given.
        A new knob keeps the row's range; an empty id clears the row."""
        d = self.slots[i].dests[j]
        d.ident = ident or ""
        if lo is not None:
            d.lo = _frac(lo)
        if hi is not None:
            d.hi = _frac(hi, 1.0)

    def move(self, i, x, write=True):
        """Macro i to x: its value kept, every destination written (unless
        `write` is off: a scene load that carries the destinations
        itself). Returns what was planned."""
        s = self.slots[i]
        s.value = _frac(x)
        plan = s.plan(s.value)
        if write:
            for ident, v in plan.items():
                self.setter(ident, v)
        return plan

    def capture(self, i, which, read):
        """Every destination's lo or hi from where its knob sits now:
        `read(ident)` -> 0..1 or None (left alone). Returns the rows set."""
        n = 0
        for d in self.slots[i].active:
            cur = read(d.ident)
            if cur is None:
                continue
            setattr(d, which, _frac(cur))
            n += 1
        return n

    def dests_of(self, ident):
        """The macros pointing at a knob, as slot indexes."""
        return [s.i for s in self.slots if any(d.ident == ident for d in s.active)]

    def to_json(self):
        return json.dumps([s.settings() for s in self.slots])

    def load_json(self, text):
        try:
            rows = json.loads(text) if text else []
        except (ValueError, TypeError):
            rows = []
        if not isinstance(rows, list):
            return
        for s, d in zip(self.slots, rows):
            s.restore(d)

    def summary(self, i):
        """One line for a status: 'drop: 3 knobs, 40 %'."""
        s = self.slots[i]
        n = len(s.active)
        if not n:
            return ""
        return f"{s.label}: {n} knob{'s' if n != 1 else ''}, {round(100 * s.value)} %"
