#!/usr/bin/env python3
"""
gacha_autofx.py — auto fx: the stems' racks set for the style playing.

With the Perform tab's *auto fx* ticked, every section that starts (and a
song that starts through the racks) has the four stems' racks (gacha_fx:
drums, layers, chops, events) reset and a few of their knobs set from the
style's feel: a reverb on the layers and a reverse delay on the chops for
drone, dub echoes on the chops and the events for dub, a crusher on the
chops for glitch. AUTO_FX says what each style asks for, per stem and
knob, with a range the amount is drawn from and the chance the knob is
touched at all, so two sections in one style do not get the same racks;
plan() draws one. A style without a row takes its feel's favoured chain
effects (gacha_engine.Feel.favour, the names of the texture chains'
effects) mapped onto the rack's knobs, on the layers and the chops. The
racks' knobs run 0..1, 1 fully wet, so the amounts stay under 0.7.

Beyond the settings, the racks move: AUTO_GESTURES (below) gives each
style moves in time, a delay thrown on an 8th now and then, a crusher
for a beat, a reverb tail on the loop's last bar, the ladder closing over
the last two, several at once; plan_gestures() draws a section's and
Gestures runs them on the loop clock from the Perform poll.

Some styles move a knob rather than set it: AUTO_LFO gives each a few
LFOs (gacha_lfo: a knob, a shape, a cycle in bars, a depth, the
polarity), the ladder rising over 4 bars for house, the chops' pitch
stepping at random on the 16ths for glitch, the events' delay drifting
for dub; plan_lfos() draws them. The GUI puts them in LFO slots that are
off, never in one the player turned on, and takes its own back before
the next section. Every LFO is locked to the live clock's bar, and the
racks' delays follow the beat, so what auto fx sets lands in time.

    plan("drone", random.Random(seed))   # {("layers", "reverb"): 0.52, ("chops", "reverse"): 0.41}
    plan_lfos("acid", rng)               # [{"dest": "afx_chops_lowpass", "shape": "saw down", ...}]
"""
import random

import gacha_engine as E
from gacha_fx import AUDIO_FX, STEM_CHANNELS

FX_KEYS = {k for k, *_ in AUDIO_FX}

# a feel's favoured chain effects (pedalboard class names, gacha_engine's
# Saturator, the codec, the ladder) onto the rack's knobs
CHAIN_TO_RACK = {"Reverb": "reverb", "Delay": "delay", "Chorus": "chorus", "Phaser": "phaser",
                 "Saturator": "drive", "Bitcrush": "crush", "GSM": "lofi", "Ladder": "lowpass",
                 "PitchShift": None}            # the pitch knob is bipolar and never set here

# style -> ((stem, knob, (lo, hi), chance), ...)
AUTO_FX = {
    "four-floor": (("layers", "chorus", (0.2, 0.35), 0.8), ("chops", "delay", (0.2, 0.35), 0.7),
                   ("events", "reverb", (0.2, 0.4), 0.8)),
    "breakbeat": (("drums", "drive", (0.15, 0.3), 0.8), ("chops", "crush", (0.15, 0.3), 0.6),
                  ("events", "reverse", (0.15, 0.3), 0.5)),
    "boom-bap": (("layers", "lofi", (0.2, 0.4), 0.8), ("drums", "drive", (0.1, 0.25), 0.7),
                 ("chops", "lowpass", (0.1, 0.3), 0.6)),
    "halftime": (("events", "reverb", (0.3, 0.5), 0.9), ("layers", "lowpass", (0.2, 0.4), 0.7),
                 ("chops", "delay", (0.15, 0.3), 0.6)),
    "dnb": (("drums", "drive", (0.15, 0.3), 0.8), ("layers", "lowpass", (0.2, 0.35), 0.6),
            ("chops", "reverse", (0.15, 0.3), 0.6)),
    "minimal": (("layers", "delay", (0.2, 0.35), 0.8), ("chops", "phaser", (0.15, 0.3), 0.7),
                ("events", "delay", (0.15, 0.3), 0.5)),
    "ukg": (("chops", "chorus", (0.2, 0.35), 0.8), ("events", "delay", (0.2, 0.35), 0.7),
            ("layers", "reverb", (0.15, 0.3), 0.5)),
    "dembow": (("events", "reverb", (0.2, 0.35), 0.8), ("chops", "delay", (0.15, 0.3), 0.7)),
    "one-drop": (("chops", "delay", (0.35, 0.55), 0.9), ("events", "delay", (0.3, 0.5), 0.8),
                 ("events", "reverb", (0.25, 0.4), 0.7), ("drums", "reverb", (0.1, 0.2), 0.6)),
    "footwork": (("chops", "delay", (0.15, 0.3), 0.7), ("events", "crush", (0.15, 0.3), 0.6),
                 ("layers", "phaser", (0.15, 0.3), 0.5)),
    "clave": (("events", "reverb", (0.2, 0.35), 0.8), ("layers", "chorus", (0.15, 0.3), 0.7),
              ("chops", "delay", (0.1, 0.25), 0.5)),
    "idm": (("chops", "crush", (0.2, 0.4), 0.8), ("events", "reverse", (0.2, 0.4), 0.7),
            ("layers", "lofi", (0.15, 0.3), 0.6), ("drums", "crush", (0.1, 0.2), 0.4)),
    "hypnotic": (("layers", "phaser", (0.3, 0.5), 0.9), ("chops", "delay", (0.25, 0.4), 0.8),
                 ("drums", "reverb", (0.1, 0.2), 0.6), ("events", "reverb", (0.3, 0.5), 0.7)),
    "drone": (("layers", "reverb", (0.4, 0.65), 0.95), ("chops", "reverse", (0.3, 0.5), 0.9),
              ("events", "reverb", (0.3, 0.5), 0.8), ("layers", "lowpass", (0.1, 0.3), 0.5)),
    "glitch": (("chops", "crush", (0.3, 0.5), 0.9), ("events", "lofi", (0.3, 0.5), 0.8),
               ("chops", "reverse", (0.2, 0.4), 0.7), ("drums", "crush", (0.1, 0.25), 0.5)),
    "dub": (("chops", "delay", (0.35, 0.55), 0.95), ("events", "delay", (0.3, 0.5), 0.85),
            ("events", "reverb", (0.2, 0.4), 0.7), ("drums", "reverb", (0.1, 0.25), 0.7),
            ("layers", "phaser", (0.2, 0.35), 0.6)),
    "polyrhythm": (("chops", "delay", (0.25, 0.45), 0.9), ("layers", "phaser", (0.2, 0.4), 0.7),
                   ("events", "delay", (0.2, 0.35), 0.6)),
    "electro": (("layers", "chorus", (0.25, 0.4), 0.9), ("chops", "lofi", (0.15, 0.3), 0.7),
                ("events", "phaser", (0.15, 0.3), 0.6)),
    "acid": (("chops", "lowpass", (0.3, 0.5), 0.9), ("layers", "phaser", (0.2, 0.35), 0.7),
             ("events", "delay", (0.2, 0.35), 0.6), ("chops", "delay", (0.15, 0.3), 0.5)),
    "jungle": (("drums", "drive", (0.15, 0.3), 0.8), ("chops", "lowpass", (0.15, 0.3), 0.6),
               ("events", "reverse", (0.2, 0.35), 0.6)),
    "gabber": (("drums", "drive", (0.3, 0.5), 0.95), ("layers", "drive", (0.2, 0.4), 0.8),
               ("chops", "crush", (0.15, 0.3), 0.6)),
    "jersey": (("chops", "reverb", (0.15, 0.3), 0.8), ("events", "delay", (0.15, 0.3), 0.7),
               ("layers", "chorus", (0.15, 0.3), 0.5)),
    "gqom": (("layers", "lowpass", (0.3, 0.45), 0.9), ("chops", "reverb", (0.3, 0.45), 0.8),
             ("events", "reverb", (0.3, 0.5), 0.8), ("drums", "reverb", (0.1, 0.2), 0.5)),
    "amapiano": (("chops", "reverb", (0.2, 0.35), 0.8), ("layers", "chorus", (0.2, 0.35), 0.8),
                 ("events", "delay", (0.15, 0.3), 0.6)),
    "jazz": (("layers", "reverb", (0.25, 0.4), 0.9), ("drums", "reverb", (0.1, 0.2), 0.7),
             ("chops", "chorus", (0.15, 0.3), 0.6)),
    "industrial": (("drums", "drive", (0.2, 0.4), 0.9), ("chops", "crush", (0.3, 0.5), 0.9),
                   ("layers", "lofi", (0.2, 0.4), 0.7), ("events", "drive", (0.15, 0.3), 0.5)),
    "bruk": (("chops", "phaser", (0.2, 0.35), 0.8), ("layers", "chorus", (0.15, 0.3), 0.7),
             ("events", "reverb", (0.15, 0.3), 0.5)),
    "ambient": (("layers", "reverb", (0.45, 0.7), 0.95), ("events", "reverb", (0.4, 0.6), 0.9),
                ("events", "reverse", (0.2, 0.4), 0.6), ("layers", "lowpass", (0.1, 0.3), 0.5)),
    "motorik": (("layers", "delay", (0.25, 0.4), 0.9), ("chops", "phaser", (0.2, 0.35), 0.7),
                ("events", "reverb", (0.15, 0.3), 0.5)),
    "afrobeat": (("layers", "chorus", (0.2, 0.35), 0.8), ("chops", "delay", (0.15, 0.3), 0.7),
                 ("events", "reverb", (0.15, 0.3), 0.6)),
    "cumbia": (("chops", "reverb", (0.15, 0.3), 0.8), ("layers", "chorus", (0.2, 0.35), 0.8),
               ("events", "delay", (0.15, 0.3), 0.6)),
}


# style -> ((stem, knob, shape, cycle, depth, unipolar, chance), ...): the knob
# moved on the live clock; depth is the LFO's (± half the knob's range at
# 1, the whole range upward with unipolar); the pitch knob is bipolar, so a
# depth of 0.5 either way is three semitones with the range setting at 12
AUTO_LFO = {
    "four-floor": (("layers", "lowpass", "saw up", "4 bars", 0.6, True, 0.6),),
    "breakbeat": (("chops", "lowpass", "saw down", "1 bar", 0.4, True, 0.5),),
    "boom-bap": (("layers", "lofi", "sine", "8 bars", 0.3, True, 0.5),),
    "halftime": (("layers", "lowpass", "sine", "2 bars", 0.5, True, 0.6),),
    "dnb": (("layers", "lowpass", "saw up", "4 bars", 0.5, True, 0.6),),
    "minimal": (("layers", "phaser", "sine", "4 bars", 0.4, True, 0.7),),
    "ukg": (("chops", "pitch", "random step", "1/4", 0.3, False, 0.5),),
    "one-drop": (("events", "delay", "random smooth", "2 bars", 0.4, True, 0.6),),
    "footwork": (("chops", "pitch", "random step", "1/4", 0.4, False, 0.6),),
    "clave": (("layers", "chorus", "sine", "4 bars", 0.3, True, 0.5),),
    "idm": (("events", "pitch", "random step", "1/8", 0.7, False, 0.8),
            ("chops", "crush", "square", "1/4", 0.4, True, 0.5)),
    "hypnotic": (("layers", "phaser", "sine", "8 bars", 0.5, True, 0.8),
                 ("chops", "pitch", "triangle", "4 bars", 0.3, False, 0.5)),
    "drone": (("layers", "lowpass", "sine", "8 bars", 0.5, True, 0.8),
              ("events", "pitch", "random walk", "4 bars", 0.3, False, 0.5)),
    "glitch": (("chops", "pitch", "random step", "1/16", 0.6, False, 0.8),
               ("chops", "crush", "square", "1/8", 0.5, True, 0.6)),
    "dub": (("events", "delay", "random smooth", "2 bars", 0.4, True, 0.7),
            ("layers", "phaser", "sine", "4 bars", 0.4, True, 0.5)),
    "polyrhythm": (("chops", "delay", "triangle", "2 bars", 0.3, True, 0.6),),
    "electro": (("layers", "chorus", "triangle", "2 bars", 0.3, True, 0.6),),
    "acid": (("chops", "lowpass", "saw down", "1/2", 0.6, True, 0.9),
             ("layers", "phaser", "sine", "2 bars", 0.4, True, 0.5)),
    "jungle": (("chops", "pitch", "random step", "1/2", 0.3, False, 0.5),),
    "gabber": (("drums", "drive", "square", "1/4", 0.3, True, 0.4),),
    "jersey": (("chops", "pitch", "random step", "1/8", 0.4, False, 0.6),),
    "gqom": (("layers", "lowpass", "sine", "8 bars", 0.4, True, 0.7),),
    "amapiano": (("chops", "delay", "sine", "4 bars", 0.3, True, 0.5),),
    "jazz": (),
    "industrial": (("chops", "lofi", "square", "1/2", 0.5, True, 0.6),),
    "bruk": (("layers", "chorus", "sine", "4 bars", 0.3, True, 0.5),),
    "ambient": (("layers", "reverb", "sine", "8 bars", 0.3, True, 0.7),
                ("layers", "lowpass", "sine", "8 bars", 0.4, True, 0.7)),
    "motorik": (("layers", "phaser", "sine", "8 bars", 0.5, True, 0.8),),
    "afrobeat": (("layers", "chorus", "sine", "4 bars", 0.3, True, 0.5),),
    "cumbia": (("layers", "chorus", "sine", "4 bars", 0.3, True, 0.5),),
    "dembow": (),
}


def plan_lfos(style, rng=None):
    """The LFOs for `style` (AUTO_LFO), each drawn with its chance: a list
    of slot settings as the GUI's LFO rows take them, {dest, shape, rate,
    depth, unipolar, mode}; [] for a style without any."""
    rng = rng or random.Random()
    out = []
    for stem, knob, shape, rate, depth, unipolar, chance in AUTO_LFO.get(style, ()):
        if stem in STEM_CHANNELS and knob in FX_KEYS and rng.random() < chance:
            out.append({"dest": f"afx_{stem}_{knob}", "shape": shape, "rate": rate,
                        "depth": depth, "unipolar": unipolar, "mode": "cycle"})
    return out


def plan(style, rng=None):
    """The racks for `style` (a feel's name, gacha_engine.STYLE_FEEL):
    {(stem, knob): amount} drawn from AUTO_FX, or from the feel's favoured
    chain effects for a style without a row (the first on the layers, the
    second on the chops, 0.2 to 0.35); {} for the default feel and for
    nothing to say."""
    rng = rng or random.Random()
    out = {}
    rows = AUTO_FX.get(style)
    if rows is None:
        feel = E.STYLE_FEEL.get(style)
        if feel is None:
            return out
        knobs = [CHAIN_TO_RACK.get(w) for w in feel.favour]
        knobs = [k for k in knobs if k]
        rows = tuple((stem, knob, (0.2, 0.35), 0.8)
                     for stem, knob in zip(("layers", "chops"), knobs[:2]))
    for stem, knob, (lo, hi), chance in rows:
        if stem in STEM_CHANNELS and knob in FX_KEYS and rng.random() < chance:
            out[(stem, knob)] = round(rng.uniform(lo, hi), 2)
    return out


def describe(p, lfos=()):
    """A plan as the log says it: 'layers reverb 0.52, chops reverse 0.41;
    LFO chops pitch random step 1/16 ±0.6'."""
    knobs = ", ".join(f"{stem} {knob} {v:.2f}" for (stem, knob), v in sorted(p.items()))
    moves = ", ".join(f"LFO {l['dest'][4:].replace('_', ' ')} {l['shape']} {l['rate']} "
                      f"{'+' if l['unipolar'] else '±'}{l['depth']:g}" for l in lfos)
    return "; ".join(x for x in (knobs, moves) if x) or "nothing"


# ---------------------------------------------------------------- gestures
# The racks as moves in time rather than settings: a delay thrown on an 8th
# now and then, a crusher for a beat, a reverb tail on the loop's last bar,
# the ladder closing over the last two, several at once. A gesture is
# (dest, start, length, amount, shape) in beats from the loop's start:
# "hold" keeps the amount, "in" rises to it, "out" falls from it; the knob
# is the largest gesture on it at the moment, the static plan's value when
# none is on. AUTO_GESTURES holds a palette per style, each line a kind of
# move with its chance; plan_gestures() draws a section's from the palette
# and Gestures runs them on the loop clock (the Perform poll, 50 times a
# second, the player's position in beats). The pitch knob takes signed
# amounts (down is negative). A knob an LFO holds is left to the LFO.

def throw(stem, knob, amount, length, grid, count, chance, shape="hold"):
    """`count` (lo, hi) moves of `length` beats at random places on a grid
    of `grid` beats: the delay throw, the crusher burst."""
    return {"kind": "throw", "stem": stem, "knob": knob, "amount": amount, "length": length,
            "grid": grid, "count": count, "chance": chance, "shape": shape}


def onbar(stem, knob, amount, length, bars, beat, chance, shape="hold"):
    """A move of `length` beats at `beat` (0..3) of the bars `bars` says:
    "last", "first", "even", "odd", "all", a list of 1-based bars."""
    return {"kind": "onbar", "stem": stem, "knob": knob, "amount": amount, "length": length,
            "bars": bars, "beat": beat, "chance": chance, "shape": shape}


def ramp(stem, knob, amount, bars, shape, chance):
    """A rise ("in") or a fall ("out") across the bars `bars` says: (from,
    to) 1-based inclusive, "last", "last2", "last4", "first4"."""
    return {"kind": "ramp", "stem": stem, "knob": knob, "amount": amount, "bars": bars,
            "chance": chance, "shape": shape}


def count(stem, knob, amount, periods, bars, chance, length=0.25):
    """A 16th-long move every so many 16ths (one of `periods`) across the
    bars `bars` says: Tenney's pulse on a knob."""
    return {"kind": "count", "stem": stem, "knob": knob, "amount": amount, "periods": periods,
            "bars": bars, "chance": chance, "length": length}


DEFAULT_GESTURES = (
    onbar("chops", "delay", (0.5, 0.7), 1.0, [4, 8], 3, 0.5),
    onbar("events", "reverb", (0.6, 0.8), 1.0, "last", 3, 0.4),
    ramp("layers", "lowpass", (0.5, 0.8), "last", "in", 0.3),
)

AUTO_GESTURES = {
    "four-floor": (ramp("layers", "lowpass", (0.6, 0.8), "last4", "out", 0.6),
                   onbar("events", "reverb", (0.6, 0.8), 1.0, "last", 3, 0.6),
                   onbar("chops", "delay", (0.5, 0.7), 1.0, [4, 8], 3, 0.7)),
    "breakbeat": (throw("drums", "crush", (0.5, 0.7), 2.0, 1.0, (1, 2), 0.6),
                  onbar("drums", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6),
                  throw("events", "reverse", (0.7, 0.9), 2.0, 1.0, (1, 2), 0.5)),
    "boom-bap": (throw("chops", "lofi", (0.5, 0.7), 2.0, 1.0, (1, 2), 0.5),
                 onbar("drums", "delay", (0.4, 0.6), 1.0, [4, 8], 3, 0.5)),
    "halftime": (ramp("layers", "lowpass", (0.6, 0.8), "last2", "in", 0.6),
                 onbar("drums", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.5),
                 throw("chops", "delay", (0.5, 0.7), 1.0, 1.0, (1, 2), 0.5)),
    "dnb": (throw("drums", "crush", (0.4, 0.6), 2.0, 1.0, (1, 2), 0.6),
            onbar("drums", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6),
            throw("events", "reverse", (0.7, 0.9), 2.0, 1.0, (1, 2), 0.5)),
    "minimal": (throw("chops", "delay", (0.4, 0.6), 1.0, 1.0, (1, 3), 0.7),
                ramp("layers", "lowpass", (0.4, 0.6), "last", "in", 0.4)),
    "ukg": (throw("chops", "delay", (0.5, 0.7), 0.5, 0.25, (3, 6), 0.7),
            throw("chops", "pitch", (0.4, 0.7), 0.5, 0.25, (2, 4), 0.5)),
    "dembow": (onbar("drums", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6),
               onbar("events", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.4)),
    "one-drop": (onbar("drums", "delay", (0.6, 0.8), 1.0, "even", 3, 0.8),
                 ramp("events", "reverb", (0.6, 0.8), "last", "in", 0.6),
                 ramp("layers", "lowpass", (0.6, 0.8), "last", "in", 0.5),
                 throw("chops", "reverse", (0.6, 0.8), 2.0, 1.0, (1, 2), 0.4)),
    "footwork": (throw("chops", "delay", (0.5, 0.7), 0.5, 0.25, (3, 6), 0.7),
                 throw("chops", "pitch", (0.4, 0.7), 0.5, 0.25, (2, 4), 0.5),
                 throw("drums", "lofi", (0.7, 0.9), 0.5, 0.5, (1, 3), 0.4)),
    "clave": (onbar("events", "delay", (0.5, 0.7), 1.0, "even", 3, 0.5),
              onbar("events", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.4)),
    "idm": (throw("chops", "delay", (0.6, 0.8), 0.5, 0.25, (3, 6), 0.8),
            throw("chops", "crush", (0.6, 0.8), 1.0, 1.0, (1, 3), 0.7),
            throw("events", "reverse", (0.7, 0.9), 2.0, 1.0, (1, 2), 0.6),
            throw("chops", "pitch", (-0.9, -0.5), 1.0, 0.5, (1, 3), 0.5, "out"),
            throw("drums", "lofi", (0.8, 1.0), 0.5, 0.5, (1, 3), 0.5)),
    "hypnotic": (count("chops", "delay", (0.4, 0.6), (5, 7), "last4", 0.7),
                 ramp("layers", "lowpass", (0.3, 0.5), "last4", "in", 0.4),
                 onbar("events", "reverb", (0.6, 0.8), 2.0, "last", 2, 0.4)),
    "drone": (ramp("layers", "reverb", (0.6, 0.8), "first4", "in", 0.7),
              ramp("events", "pitch", (-0.7, -0.4), "last2", "in", 0.5),
              onbar("chops", "reverse", (0.7, 0.9), 4.0, [3, 7], 0, 0.6),
              ramp("layers", "lowpass", (0.4, 0.6), "last2", "in", 0.5)),
    "glitch": (throw("chops", "delay", (0.6, 0.8), 0.5, 0.25, (4, 8), 0.9),
               throw("chops", "crush", (0.6, 0.8), 1.0, 1.0, (2, 4), 0.8),
               throw("events", "reverse", (0.7, 0.9), 2.0, 1.0, (1, 2), 0.6),
               throw("drums", "lofi", (0.8, 1.0), 0.5, 0.5, (2, 4), 0.6),
               throw("chops", "pitch", (-0.9, -0.5), 1.0, 0.5, (1, 3), 0.6, "out"),
               throw("layers", "crush", (0.5, 0.7), 0.5, 0.25, (2, 4), 0.5)),
    "dub": (onbar("drums", "delay", (0.7, 0.9), 1.0, "even", 3, 0.9),
            ramp("events", "reverb", (0.7, 0.9), "last", "in", 0.7),
            ramp("layers", "lowpass", (0.6, 0.8), "last", "in", 0.6),
            throw("chops", "reverse", (0.6, 0.8), 2.0, 1.0, (1, 2), 0.5),
            throw("drums", "reverb", (0.6, 0.8), 1.0, 1.0, (1, 2), 0.5)),
    "polyrhythm": (count("chops", "delay", (0.5, 0.7), (5, 7), "last4", 0.7),
                   count("events", "reverb", (0.5, 0.7), (3, 5), "last2", 0.4)),
    "electro": (onbar("drums", "delay", (0.4, 0.6), 1.0, "even", 3, 0.6),
                throw("chops", "lofi", (0.5, 0.7), 1.0, 1.0, (1, 3), 0.5),
                throw("layers", "pitch", (0.3, 0.5), 1.0, 1.0, (1, 2), 0.4, "out")),
    "acid": (throw("chops", "lowpass", (0.8, 1.0), 0.25, 0.25, (6, 12), 0.8),
             onbar("chops", "drive", (0.5, 0.7), 4.0, "last", 0, 0.5),
             onbar("chops", "delay", (0.5, 0.7), 1.0, [4, 8], 3, 0.5)),
    "jungle": (throw("drums", "crush", (0.5, 0.7), 2.0, 1.0, (1, 2), 0.6),
               throw("chops", "pitch", (-0.8, -0.5), 1.0, 0.5, (2, 4), 0.6, "out"),
               onbar("drums", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6),
               throw("events", "reverse", (0.7, 0.9), 2.0, 1.0, (1, 2), 0.5)),
    "gabber": (onbar("drums", "drive", (0.6, 0.8), 4.0, "last", 0, 0.7),
               ramp("layers", "pitch", (0.4, 0.7), "last2", "in", 0.5),
               throw("drums", "crush", (0.5, 0.7), 0.5, 0.5, (2, 4), 0.5)),
    "jersey": (throw("chops", "delay", (0.5, 0.7), 0.5, 0.25, (3, 6), 0.7),
               throw("chops", "pitch", (0.4, 0.7), 0.5, 0.25, (2, 4), 0.6),
               onbar("events", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.4)),
    "gqom": (onbar("drums", "reverb", (0.5, 0.7), 1.0, "even", 3, 0.6),
             ramp("layers", "lowpass", (0.7, 0.9), "last", "in", 0.6),
             throw("chops", "reverse", (0.6, 0.8), 2.0, 1.0, (1, 2), 0.5)),
    "amapiano": (onbar("chops", "delay", (0.5, 0.7), 1.0, [4, 8], 3, 0.7),
                 throw("events", "reverb", (0.6, 0.8), 2.0, 1.0, (1, 2), 0.5)),
    "jazz": (onbar("events", "reverb", (0.6, 0.8), 2.0, [4, 8], 2, 0.6),
             throw("chops", "chorus", (0.4, 0.6), 2.0, 1.0, (1, 2), 0.4)),
    "industrial": (throw("drums", "crush", (0.6, 0.8), 1.0, 1.0, (3, 6), 0.8),
                   throw("chops", "drive", (0.5, 0.7), 2.0, 1.0, (1, 3), 0.6),
                   onbar("layers", "lofi", (0.8, 1.0), 4.0, "last", 0, 0.5),
                   throw("drums", "lofi", (0.8, 1.0), 0.5, 0.5, (2, 4), 0.5)),
    "bruk": (throw("chops", "phaser", (0.5, 0.7), 2.0, 1.0, (1, 2), 0.5),
             onbar("drums", "delay", (0.4, 0.6), 1.0, "even", 3, 0.5)),
    "ambient": (ramp("layers", "reverb", (0.6, 0.8), "first4", "in", 0.8),
                ramp("layers", "lowpass", (0.5, 0.7), "last4", "out", 0.6),
                onbar("events", "reverse", (0.6, 0.8), 8.0, [5], 0, 0.4)),
    "motorik": (onbar("events", "reverb", (0.5, 0.7), 2.0, "last", 2, 0.4),
                onbar("chops", "delay", (0.4, 0.6), 1.0, [8], 3, 0.5)),
    "afrobeat": (throw("chops", "delay", (0.4, 0.6), 1.0, 1.0, (1, 3), 0.6),
                 onbar("events", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.4)),
    "cumbia": (throw("chops", "delay", (0.4, 0.6), 1.0, 1.0, (1, 3), 0.6),
               onbar("events", "reverb", (0.5, 0.7), 2.0, [8], 2, 0.5)),
}


def _bars_of(spec, bars):
    """The 1-based bars a spec names, inside `bars`."""
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and spec[0] <= spec[1] and not isinstance(spec, list):
            return [b for b in range(spec[0], spec[1] + 1) if 1 <= b <= bars]
        return [b for b in spec if 1 <= b <= bars]
    return {"last": [bars], "first": [1], "even": list(range(2, bars + 1, 2)),
            "odd": list(range(1, bars + 1, 2)), "all": list(range(1, bars + 1)),
            "last2": list(range(max(1, bars - 1), bars + 1)),
            "last4": list(range(max(1, bars - 3), bars + 1)),
            "first4": list(range(1, min(4, bars) + 1))}.get(spec, [])


def _amount(rng, a):
    return round(rng.uniform(*a), 2) if isinstance(a, (list, tuple)) else float(a)


def plan_gestures(style, rng=None, bars=8, exclude=()):
    """A section's gestures from the style's palette (AUTO_GESTURES, the
    default palette without one, none for the default feel): a list of
    (dest, start_beat, length_beats, amount, shape). Knobs in `exclude`
    (ids an LFO holds) are left out."""
    rng = rng or random.Random()
    if style == "default":
        return []
    out = []
    loop = bars * 4
    for g in AUTO_GESTURES.get(style, DEFAULT_GESTURES):
        dest = f"afx_{g['stem']}_{g['knob']}"
        if g["stem"] not in STEM_CHANNELS or g["knob"] not in FX_KEYS or dest in exclude:
            continue
        if rng.random() >= g["chance"]:
            continue
        kind = g["kind"]
        if kind == "throw":
            n = rng.randint(*g["count"])
            slots = int(loop / g["grid"])
            for i in sorted(rng.sample(range(slots), min(n, slots))):
                out.append((dest, i * g["grid"], g["length"], _amount(rng, g["amount"]), g["shape"]))
        elif kind == "onbar":
            for b in _bars_of(g["bars"], bars):
                out.append((dest, (b - 1) * 4 + g["beat"], g["length"], _amount(rng, g["amount"]),
                            g["shape"]))
        elif kind == "ramp":
            bs = _bars_of(g["bars"], bars)
            if bs:
                start, end = (bs[0] - 1) * 4, bs[-1] * 4
                out.append((dest, start, end - start, _amount(rng, g["amount"]), g["shape"]))
        elif kind == "count":
            bs = _bars_of(g["bars"], bars)
            if bs:
                period = rng.choice(g["periods"])
                start, end = (bs[0] - 1) * 16, bs[-1] * 16
                amt = _amount(rng, g["amount"])
                for st in range(start + rng.randrange(period), end, period):
                    out.append((dest, st / 4.0, g["length"], amt, "hold"))
    return out


class Gestures:
    """A section's gestures on its loop clock: values(beat) gives every
    knob with a gesture its value now, the static plan's (`base`) when no
    gesture is on it, the largest gesture when several are."""

    def __init__(self, gestures, bars, base=None):
        self.gestures = list(gestures)
        self.loop = bars * 4.0
        self.base = dict(base or {})
        self.dests = sorted({g[0] for g in self.gestures})

    def values(self, beat):
        beat = beat % self.loop if self.loop > 0 else 0.0
        out = {d: self.base.get(d, 0.0) for d in self.dests}
        for dest, start, length, amount, shape in self.gestures:
            t = beat - start
            if t < 0 or t >= length or length <= 0:
                continue
            f = t / length
            v = amount * (f if shape == "in" else (1.0 - f) if shape == "out" else 1.0)
            if abs(v) > abs(out[dest]):
                out[dest] = round(v, 3)
        return out

    def rest(self):
        """Every knob back to its base: the section stopped."""
        return {d: self.base.get(d, 0.0) for d in self.dests}
