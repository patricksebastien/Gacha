#!/usr/bin/env python3
"""
gacha_scene.py — the scenes: eight snapshots of the whole live state, each
recalled at once or morphed into over a few bars.

A scene is what the Perform, Audio FX, LFO, Video and M:C tabs hold at the
moment you press save, in five groups a scene can carry or leave alone:
"audio" (the twelve racks and their volumes, the main volume, the sync
delay, the A/B pedal, the burst length, the eight VST knobs and the inserts
bypass), "voices" (the drum voices in or out, the solos, the stems, the
quantize grid), "lfo" (the eight LFO rows) and "video" (the source grade,
every effect strength, the shader, its opacity and blend, video-only, the
picture mode and the dissolve) and "mc" (the Model:Cycles tab: every
track's cells, the delay and reverb, the retrig's settings). Saving
captures every group; loading
applies the ticked ones only, so a scene with only video ticked is a look
that leaves the sound alone.

Each group has knobs (a fraction 0..1 of the knob's range, the ids the
Controls tab uses) and switches (anything else: on/off, a name, the LFO
rows). A load flips the switches at once, then either sets the knobs at
once or starts a Morph: a linear glide from where each knob sits to the
scene's value over a time in bars at the live tempo, 100 steps a second,
only the knobs that differ. A knob an LFO is on is captured at its centre,
not mid-wiggle, and the LFO follows the glide (what the morph writes is a
value the LFO did not write: its new centre).

    bank = SceneBank()                        # eight empty slots
    bank.slots[2].save(name="drop", state=captured)   # captured: {group: {"knobs": {}, "switches": {}}}
    sw, knobs = bank.slots[2].plan()          # what a load applies, ticked groups only
    m = Morph(now, 4.0, current, knobs)       # 4 s from the knobs now to the scene's
    m.step(now + 2.0)                         # -> {ident: halfway} ; m.done afterwards
    text = bank.to_json(); bank.load_json(text)
"""
import json
import time

GROUPS = ["audio", "voices", "lfo", "video", "mc"]
GROUP_LABELS = {"audio": "audio", "voices": "voices", "lfo": "LFOs", "video": "video",
                "mc": "M:C"}
MORPH_BARS = [("1/4", 0.25), ("1/2", 0.5), ("1 bar", 1.0), ("2 bars", 2.0),
              ("4 bars", 4.0), ("8 bars", 8.0), ("16 bars", 16.0), ("32 bars", 32.0)]
MORPH_DEFAULT = "1 bar"
N_SLOTS = 8
EPS = 1e-6


class Scene:
    """One slot: a name, the four ticks, and the captured state or None."""

    def __init__(self, i):
        self.i = i
        self.name = ""
        self.ticks = {g: True for g in GROUPS}
        self.state = None                      # {group: {"knobs": {}, "switches": {}}}
        self.saved_at = ""                     # "21:04", for the row

    @property
    def empty(self):
        return not self.state

    @property
    def label(self):
        return self.name.strip() or f"scene {self.i + 1}"

    def save(self, state, name=None, when=None):
        """Keep `state` (every group, whatever the ticks say)."""
        self.state = {g: {"knobs": dict(state.get(g, {}).get("knobs", {})),
                          "switches": dict(state.get(g, {}).get("switches", {}))}
                      for g in GROUPS}
        if name is not None:
            self.name = name
        self.saved_at = when or time.strftime("%H:%M")

    def plan(self):
        """What a load applies: (switches {id: value}, knobs {id: fraction})
        of the ticked groups; both empty for an empty slot."""
        switches, knobs = {}, {}
        if self.state:
            for g in GROUPS:
                if self.ticks.get(g, True) and g in self.state:
                    switches.update(self.state[g].get("switches", {}))
                    knobs.update(self.state[g].get("knobs", {}))
        return switches, knobs

    def settings(self):
        return {"name": self.name, "ticks": dict(self.ticks), "state": self.state,
                "saved_at": self.saved_at}

    def restore(self, d):
        if not isinstance(d, dict):
            return
        self.name = str(d.get("name", "") or "")
        ticks = d.get("ticks", {})
        self.ticks = {g: bool(ticks.get(g, True)) if isinstance(ticks, dict) else True
                      for g in GROUPS}
        st = d.get("state")
        self.state = st if isinstance(st, dict) and st else None
        self.saved_at = str(d.get("saved_at", "") or "")


class SceneBank:
    def __init__(self, n=N_SLOTS):
        self.slots = [Scene(i) for i in range(n)]
        self.current = None                    # the slot loaded last, or None

    def to_json(self):
        return json.dumps([s.settings() for s in self.slots])

    def load_json(self, text):
        try:
            rows = json.loads(text or "[]")
        except (ValueError, TypeError):
            return
        if not isinstance(rows, list):
            return
        for s, d in zip(self.slots, rows):
            s.restore(d)


class Morph:
    """A glide of knobs from `start` to `target` over `dur` seconds from
    `t0`; `step(now)` gives what to write, `done` once the end was given
    out. Knobs already at their target are left alone (yours, or an
    LFO's), and a knob missing on either side is skipped."""

    def __init__(self, t0, dur, start, target):
        self.t0 = t0
        self.dur = max(0.0, float(dur))
        self.pairs = {k: (float(start[k]), float(v)) for k, v in target.items()
                      if k in start and start[k] is not None and v is not None
                      and abs(float(start[k]) - float(v)) > EPS}
        self.done = False
        self.frac = 0.0

    def step(self, now):
        """{ident: value} for `now`; the targets exactly at the end."""
        if self.done:
            return {}
        f = 1.0 if self.dur <= 0 else min(1.0, max(0.0, (now - self.t0) / self.dur))
        self.frac = f
        if f >= 1.0:
            self.done = True
            return {k: b for k, (_a, b) in self.pairs.items()}
        return {k: a + (b - a) * f for k, (a, b) in self.pairs.items()}
