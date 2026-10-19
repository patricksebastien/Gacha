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
from gacha_fx import AUDIO_FX, STEM_CHANNELS, VOICES, Gestures  # noqa: F401 (Gestures re-exported)

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


def plan_lfos(style, rng=None, table=None, channels=STEM_CHANNELS):
    """The LFOs for `style` (`table`: AUTO_LFO, the stems', or
    AUTO_LFO_VOICES), each drawn with its chance: a list of slot settings
    as the GUI's LFO rows take them, {dest, shape, rate, depth, unipolar,
    mode}; [] for a style without any."""
    rng = rng or random.Random()
    table = AUTO_LFO if table is None else table
    if style == "default":
        return []
    out, taken = [], set()
    rows = tuple(table.get(style, ())) + GENERIC.get(id(table), ())   # the style's, then the palette
    for ch, knob, shape, rate, depth, unipolar, chance in rows:
        dest = f"afx_{ch}_{knob}"
        if ch in channels and knob in FX_KEYS and dest not in taken and rng.random() < chance:
            taken.add(dest)                           # one LFO a knob
            out.append({"dest": dest, "shape": shape, "rate": rate,
                        "depth": depth, "unipolar": unipolar, "mode": "cycle"})
    return out


def plan(style, rng=None, table=None, channels=STEM_CHANNELS):
    """The racks for `style` (a feel's name, gacha_engine.STYLE_FEEL):
    {(channel, knob): amount} drawn from `table` (AUTO_FX, the stems', or
    AUTO_FX_VOICES, the drum voices'), or for the stems of a style without
    a row from the feel's favoured chain effects (the first on the layers,
    the second on the chops, 0.2 to 0.35); {} for the default feel and for
    nothing to say."""
    rng = rng or random.Random()
    table = AUTO_FX if table is None else table
    out = {}
    rows = table.get(style)
    if rows is None and table is AUTO_FX:
        feel = E.STYLE_FEEL.get(style)
        if feel is None:
            return out
        knobs = [CHAIN_TO_RACK.get(w) for w in feel.favour]
        knobs = [k for k in knobs if k]
        rows = tuple((stem, knob, (0.2, 0.35), 0.8)
                     for stem, knob in zip(("layers", "chops"), knobs[:2]))
    for ch, knob, (lo, hi), chance in rows or ():
        if ch in channels and knob in FX_KEYS and rng.random() < chance:
            out[(ch, knob)] = round(rng.uniform(lo, hi), 2)
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
                 throw("drums", "lofi", (0.7, 0.9), 0.5, 0.5, (1, 3), 0.4),
                 throw("chops", "lofi", (0.9, 1.0), 0.25, 0.25, (2, 4), 0.5)),
    "clave": (onbar("events", "delay", (0.5, 0.7), 1.0, "even", 3, 0.5),
              onbar("events", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.4)),
    "idm": (throw("chops", "delay", (0.6, 0.8), 0.5, 0.25, (3, 6), 0.8),
            throw("chops", "crush", (0.6, 0.8), 1.0, 1.0, (1, 3), 0.7),
            throw("events", "reverse", (0.7, 0.9), 2.0, 1.0, (1, 2), 0.6),
            throw("chops", "pitch", (-0.9, -0.5), 1.0, 0.5, (1, 3), 0.5, "out"),
            throw("drums", "lofi", (0.8, 1.0), 0.5, 0.5, (1, 3), 0.5),
            throw("chops", "lofi", (0.9, 1.0), 0.25, 0.25, (3, 6), 0.7)),
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
               throw("layers", "crush", (0.5, 0.7), 0.5, 0.25, (2, 4), 0.5),
               throw("chops", "lofi", (0.9, 1.0), 0.25, 0.25, (4, 8), 0.8),
               throw("events", "crush", (0.9, 1.0), 0.25, 0.25, (2, 4), 0.6)),
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
                   throw("drums", "lofi", (0.8, 1.0), 0.5, 0.5, (2, 4), 0.5),
                   throw("chops", "crush", (0.9, 1.0), 0.25, 0.25, (3, 6), 0.6)),
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


def plan_gestures(style, rng=None, bars=8, exclude=(), table=None, default=None,
                  channels=STEM_CHANNELS):
    """A section's gestures from the style's palette (`table`:
    AUTO_GESTURES, the stems', or AUTO_GESTURES_VOICES; `default` the
    palette of a style without one; none for the default feel): a list of
    (dest, start_beat, length_beats, amount, shape). Knobs in `exclude`
    (ids an LFO holds) are left out."""
    rng = rng or random.Random()
    if style == "default":
        return []
    table = AUTO_GESTURES if table is None else table
    default = DEFAULT_GESTURES if default is None else default
    out = []
    loop = bars * 4
    for g in tuple(table.get(style, default)) + GENERIC.get(id(table), ()):
        dest = f"afx_{g['stem']}_{g['knob']}"
        if g["stem"] not in channels or g["knob"] not in FX_KEYS or dest in exclude:
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
        elif kind == "seq":                           # a few values in a row, then back
            if g["grid"]:
                lo, hi = g["count"]
                slots = int(loop / g["grid"])
                places = [i * g["grid"] for i in
                          sorted(rng.sample(range(slots), min(rng.randint(lo, hi), slots)))]
            else:
                places = [(b - 1) * 4 + g["beat"] for b in _bars_of(g["bars"], bars)]
            for st in places:
                sign = -1.0 if g["flip"] and rng.random() < 0.5 else 1.0
                for i, a in enumerate(g["amounts"]):
                    v = round(_amount(rng, a) * sign, 2)
                    if v and st + i * g["step"] < loop:
                        out.append((dest, st + i * g["step"], g["step"], v, "hold"))
    return out


# ---------------------------------------------------------------- the drum voices
# The same three tables for the voices' racks (gacha_fx.VOICES: kick,
# snare, hihat, ohat, ride, glitch (the hat stutter), sub), behind the
# Perform tab's second tick, "auto fx drums", keyed on the drum style (the
# kit's own; the stems' tables above go by the feel): a reverb on the snare, the
# snare thrown into the echo on the fourth beat of the even bars (the dub
# classic), the codec on the snare for a 16th for glitch, drive on the
# kick for gabber, the hats' highpass rising over the last four bars for
# house. The voices' racks sit before the drums bus rack, so a stem move
# on the drums channel and a voice move stack.

AUTO_FX_VOICES = {
    "four-floor": (("snare", "reverb", (0.2, 0.35), 0.8), ("ohat", "delay", (0.15, 0.3), 0.6)),
    "breakbeat": (("snare", "drive", (0.15, 0.3), 0.7), ("hihat", "lofi", (0.15, 0.3), 0.5)),
    "boom-bap": (("kick", "drive", (0.15, 0.3), 0.7), ("snare", "lofi", (0.2, 0.4), 0.7),
                 ("hihat", "lowpass", (0.1, 0.25), 0.5)),
    "halftime": (("snare", "reverb", (0.3, 0.5), 0.8), ("hihat", "delay", (0.1, 0.25), 0.5),
                 ("sub", "drive", (0.1, 0.25), 0.5)),
    "dnb": (("snare", "drive", (0.15, 0.3), 0.7), ("kick", "drive", (0.1, 0.2), 0.5),
            ("hihat", "lofi", (0.15, 0.3), 0.5)),
    "minimal": (("hihat", "delay", (0.2, 0.35), 0.7), ("snare", "phaser", (0.15, 0.3), 0.5)),
    "ukg": (("snare", "reverb", (0.2, 0.35), 0.7), ("hihat", "chorus", (0.15, 0.3), 0.6)),
    "dembow": (("snare", "reverb", (0.2, 0.35), 0.8), ("snare", "delay", (0.1, 0.25), 0.5)),
    "one-drop": (("snare", "delay", (0.3, 0.5), 0.9), ("snare", "reverb", (0.2, 0.35), 0.7),
                 ("hihat", "delay", (0.15, 0.3), 0.5)),
    "footwork": (("snare", "delay", (0.15, 0.3), 0.6), ("glitch", "crush", (0.2, 0.4), 0.6)),
    "clave": (("snare", "reverb", (0.2, 0.35), 0.8), ("ride", "delay", (0.15, 0.3), 0.5)),
    "idm": (("snare", "crush", (0.2, 0.4), 0.7), ("glitch", "lofi", (0.3, 0.5), 0.7),
            ("kick", "crush", (0.1, 0.2), 0.4)),
    "hypnotic": (("snare", "delay", (0.3, 0.45), 0.8), ("hihat", "phaser", (0.2, 0.4), 0.7),
                 ("ride", "delay", (0.2, 0.35), 0.5)),
    "drone": (("kick", "reverb", (0.3, 0.5), 0.8), ("snare", "reverb", (0.4, 0.6), 0.7),
              ("hihat", "delay", (0.2, 0.4), 0.5)),
    "glitch": (("snare", "lofi", (0.2, 0.4), 0.7), ("glitch", "crush", (0.3, 0.5), 0.8),
               ("hihat", "reverse", (0.15, 0.3), 0.5), ("kick", "crush", (0.1, 0.25), 0.5)),
    "dub": (("snare", "delay", (0.35, 0.55), 0.95), ("snare", "reverb", (0.2, 0.35), 0.8),
            ("hihat", "delay", (0.15, 0.3), 0.6), ("kick", "reverb", (0.05, 0.15), 0.4)),
    "polyrhythm": (("snare", "delay", (0.25, 0.4), 0.8), ("hihat", "phaser", (0.15, 0.3), 0.5)),
    "electro": (("snare", "reverb", (0.2, 0.35), 0.7), ("ride", "delay", (0.15, 0.3), 0.6),
                ("kick", "drive", (0.1, 0.2), 0.4)),
    "acid": (("kick", "drive", (0.15, 0.3), 0.7), ("snare", "delay", (0.15, 0.3), 0.5),
             ("hihat", "highpass", (0.2, 0.35), 0.5)),
    "jungle": (("snare", "drive", (0.15, 0.3), 0.7), ("snare", "reverse", (0.1, 0.25), 0.4),
               ("hihat", "lofi", (0.15, 0.3), 0.5)),
    "gabber": (("kick", "drive", (0.3, 0.5), 0.9), ("kick", "crush", (0.1, 0.25), 0.5),
               ("snare", "reverb", (0.2, 0.35), 0.6)),
    "jersey": (("snare", "reverb", (0.2, 0.35), 0.7), ("kick", "drive", (0.1, 0.2), 0.5)),
    "gqom": (("kick", "reverb", (0.1, 0.25), 0.6), ("ride", "reverb", (0.3, 0.5), 0.7),
             ("hihat", "lowpass", (0.2, 0.35), 0.5)),
    "amapiano": (("snare", "reverb", (0.25, 0.4), 0.8), ("hihat", "delay", (0.15, 0.3), 0.6)),
    "jazz": (("ride", "reverb", (0.25, 0.4), 0.9), ("snare", "reverb", (0.2, 0.35), 0.8),
             ("kick", "lowpass", (0.1, 0.25), 0.5)),
    "industrial": (("kick", "drive", (0.3, 0.5), 0.9), ("snare", "crush", (0.3, 0.5), 0.8),
                   ("hihat", "lofi", (0.2, 0.4), 0.7), ("ride", "drive", (0.2, 0.4), 0.5)),
    "bruk": (("snare", "reverb", (0.2, 0.35), 0.7), ("hihat", "phaser", (0.15, 0.3), 0.5)),
    "ambient": (("kick", "reverb", (0.4, 0.6), 0.9), ("kick", "lowpass", (0.2, 0.4), 0.6)),
    "motorik": (("snare", "delay", (0.2, 0.35), 0.7), ("hihat", "phaser", (0.2, 0.35), 0.6)),
    "afrobeat": (("snare", "reverb", (0.15, 0.3), 0.7), ("hihat", "chorus", (0.1, 0.25), 0.5)),
    "cumbia": (("snare", "reverb", (0.2, 0.35), 0.8), ("ride", "delay", (0.15, 0.3), 0.5)),
}

AUTO_LFO_VOICES = {
    "four-floor": (("hihat", "highpass", "saw up", "4 bars", 0.5, True, 0.5),),
    "hypnotic": (("hihat", "phaser", "sine", "8 bars", 0.5, True, 0.7),),
    "drone": (("kick", "lowpass", "sine", "8 bars", 0.4, True, 0.6),),
    "glitch": (("glitch", "pitch", "random step", "1/16", 0.6, False, 0.7),
               ("snare", "crush", "square", "1/8", 0.5, True, 0.5)),
    "idm": (("snare", "pitch", "random step", "1/8", 0.5, False, 0.6),),
    "acid": (("hihat", "highpass", "saw down", "1/2", 0.4, True, 0.5),),
    "minimal": (("hihat", "phaser", "sine", "4 bars", 0.4, True, 0.6),),
    "motorik": (("hihat", "phaser", "sine", "8 bars", 0.5, True, 0.7),),
    "ambient": (("kick", "lowpass", "sine", "8 bars", 0.4, True, 0.6),),
    "footwork": (("glitch", "pitch", "random step", "1/4", 0.5, False, 0.5),),
    "jersey": (("snare", "pitch", "random step", "1/8", 0.4, False, 0.5),),
    "industrial": (("snare", "lofi", "square", "1/2", 0.5, True, 0.5),),
}

DEFAULT_GESTURES_VOICES = (
    onbar("snare", "delay", (0.5, 0.7), 1.0, [4, 8], 3, 0.5),
    onbar("snare", "reverb", (0.6, 0.8), 1.0, "last", 3, 0.4),
)

AUTO_GESTURES_VOICES = {
    "four-floor": (ramp("hihat", "highpass", (0.5, 0.7), "last4", "in", 0.5),
                   onbar("snare", "reverb", (0.6, 0.8), 1.0, "last", 3, 0.6),
                   onbar("snare", "delay", (0.5, 0.7), 1.0, [4, 8], 3, 0.6)),
    "breakbeat": (throw("snare", "reverse", (0.7, 0.9), 1.0, 0.5, (1, 3), 0.6),
                  throw("hihat", "crush", (0.5, 0.7), 1.0, 1.0, (1, 2), 0.5),
                  onbar("snare", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6)),
    "boom-bap": (onbar("snare", "delay", (0.4, 0.6), 1.0, [4, 8], 3, 0.6),
                 throw("hihat", "lofi", (0.6, 0.8), 2.0, 1.0, (1, 2), 0.4)),
    "halftime": (onbar("snare", "reverb", (0.7, 0.9), 1.0, "last", 2, 0.6),
                 throw("hihat", "delay", (0.5, 0.7), 0.5, 0.25, (2, 4), 0.5),
                 throw("sub", "drive", (0.4, 0.6), 2.0, 1.0, (1, 2), 0.3)),
    "dnb": (throw("snare", "reverse", (0.7, 0.9), 1.0, 0.5, (1, 3), 0.6),
            onbar("snare", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6),
            throw("hihat", "crush", (0.5, 0.7), 1.0, 1.0, (1, 2), 0.4)),
    "minimal": (throw("hihat", "delay", (0.4, 0.6), 1.0, 1.0, (1, 3), 0.6),
                onbar("snare", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.3)),
    "ukg": (throw("snare", "delay", (0.5, 0.7), 0.5, 0.25, (2, 4), 0.6),
            throw("hihat", "pitch", (0.4, 0.7), 0.5, 0.25, (2, 4), 0.4)),
    "dembow": (onbar("snare", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6),),
    "one-drop": (onbar("snare", "delay", (0.7, 0.9), 1.0, "even", 3, 0.9),
                 throw("hihat", "reverse", (0.5, 0.7), 2.0, 1.0, (1, 2), 0.4),
                 ramp("kick", "lowpass", (0.5, 0.7), "last", "in", 0.4)),
    "footwork": (throw("snare", "delay", (0.5, 0.7), 0.5, 0.25, (3, 6), 0.6),
                 throw("glitch", "crush", (0.6, 0.8), 0.5, 0.25, (2, 4), 0.5)),
    "clave": (onbar("snare", "delay", (0.5, 0.7), 1.0, "even", 3, 0.5),
              onbar("ride", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.4)),
    "idm": (throw("snare", "lofi", (0.9, 1.0), 0.25, 0.25, (4, 8), 0.8),
            throw("kick", "crush", (0.6, 0.8), 0.5, 0.5, (2, 4), 0.6),
            throw("hihat", "reverse", (0.6, 0.8), 1.0, 0.5, (1, 3), 0.5),
            throw("snare", "pitch", (-0.9, -0.5), 0.5, 0.25, (2, 4), 0.5, "out")),
    "hypnotic": (count("snare", "delay", (0.5, 0.7), (5, 7), "last4", 0.6),
                 ramp("hihat", "highpass", (0.3, 0.5), "last4", "in", 0.4)),
    "drone": (ramp("kick", "reverb", (0.6, 0.8), "first4", "in", 0.6),
              onbar("snare", "reverse", (0.7, 0.9), 4.0, [3, 7], 0, 0.5)),
    "glitch": (throw("snare", "lofi", (0.9, 1.0), 0.25, 0.25, (4, 8), 0.9),
               throw("kick", "crush", (0.6, 0.8), 0.5, 0.5, (2, 4), 0.7),
               throw("hihat", "reverse", (0.6, 0.8), 1.0, 0.5, (1, 3), 0.6),
               throw("glitch", "pitch", (-0.9, -0.5), 0.5, 0.25, (2, 4), 0.6, "out"),
               throw("ohat", "crush", (0.7, 0.9), 0.25, 0.25, (2, 4), 0.5),
               throw("sub", "lofi", (0.8, 1.0), 0.25, 0.25, (1, 3), 0.4)),
    "dub": (onbar("snare", "delay", (0.8, 1.0), 1.0, "even", 3, 0.95),
            onbar("snare", "reverb", (0.6, 0.8), 1.0, "last", 3, 0.6),
            throw("hihat", "reverse", (0.5, 0.7), 2.0, 1.0, (1, 2), 0.4),
            ramp("kick", "lowpass", (0.5, 0.7), "last", "in", 0.4)),
    "polyrhythm": (count("snare", "delay", (0.5, 0.7), (5, 7), "last4", 0.6),
                   count("hihat", "reverb", (0.4, 0.6), (3, 5), "last2", 0.4)),
    "electro": (onbar("snare", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6),
                throw("ride", "lofi", (0.6, 0.8), 1.0, 1.0, (1, 2), 0.4)),
    "acid": (throw("hihat", "highpass", (0.7, 0.9), 0.25, 0.25, (4, 8), 0.6),
             onbar("kick", "drive", (0.5, 0.7), 4.0, "last", 0, 0.5),
             onbar("snare", "delay", (0.5, 0.7), 1.0, [4, 8], 3, 0.5)),
    "jungle": (throw("snare", "reverse", (0.7, 0.9), 1.0, 0.5, (2, 4), 0.7),
               throw("snare", "pitch", (-0.8, -0.5), 0.5, 0.25, (2, 4), 0.6, "out"),
               onbar("snare", "delay", (0.5, 0.7), 1.0, "even", 3, 0.6),
               throw("hihat", "crush", (0.5, 0.7), 1.0, 1.0, (1, 2), 0.4)),
    "gabber": (onbar("kick", "drive", (0.7, 0.9), 4.0, "last", 0, 0.7),
               throw("kick", "crush", (0.5, 0.7), 0.5, 0.5, (2, 4), 0.5),
               onbar("snare", "reverb", (0.7, 0.9), 1.0, "last", 3, 0.5)),
    "jersey": (throw("snare", "delay", (0.5, 0.7), 0.5, 0.25, (2, 4), 0.6),
               throw("kick", "pitch", (0.4, 0.7), 0.5, 0.25, (2, 4), 0.5),
               onbar("snare", "reverb", (0.6, 0.8), 1.0, "last", 3, 0.4)),
    "gqom": (onbar("ride", "reverb", (0.6, 0.8), 1.0, "even", 3, 0.6),
             ramp("kick", "lowpass", (0.6, 0.8), "last", "in", 0.5),
             throw("hihat", "reverse", (0.6, 0.8), 2.0, 1.0, (1, 2), 0.4)),
    "amapiano": (onbar("snare", "delay", (0.5, 0.7), 1.0, [4, 8], 3, 0.6),
                 throw("hihat", "reverb", (0.5, 0.7), 2.0, 1.0, (1, 2), 0.4)),
    "jazz": (onbar("ride", "reverb", (0.6, 0.8), 2.0, [4, 8], 2, 0.5),
             throw("snare", "reverb", (0.6, 0.8), 1.0, 1.0, (1, 2), 0.4)),
    "industrial": (throw("kick", "crush", (0.6, 0.8), 1.0, 1.0, (3, 6), 0.8),
                   throw("snare", "lofi", (0.9, 1.0), 0.25, 0.25, (3, 6), 0.7),
                   throw("hihat", "drive", (0.5, 0.7), 2.0, 1.0, (1, 3), 0.5),
                   onbar("ride", "lofi", (0.8, 1.0), 4.0, "last", 0, 0.4)),
    "bruk": (throw("snare", "phaser", (0.5, 0.7), 2.0, 1.0, (1, 2), 0.5),
             onbar("snare", "delay", (0.4, 0.6), 1.0, "even", 3, 0.5)),
    "ambient": (ramp("kick", "reverb", (0.6, 0.8), "first4", "in", 0.7),
                ramp("kick", "lowpass", (0.5, 0.7), "last4", "out", 0.5)),
    "motorik": (onbar("snare", "delay", (0.5, 0.7), 1.0, [8], 3, 0.5),
                onbar("hihat", "reverb", (0.5, 0.7), 2.0, "last", 2, 0.4)),
    "afrobeat": (throw("snare", "delay", (0.4, 0.6), 1.0, 1.0, (1, 3), 0.6),
                 onbar("ride", "reverb", (0.5, 0.7), 1.0, "last", 3, 0.4)),
    "cumbia": (throw("snare", "delay", (0.4, 0.6), 1.0, 1.0, (1, 3), 0.6),
               onbar("ride", "reverb", (0.5, 0.7), 2.0, [8], 2, 0.5)),
}


# ---------------------------------------------------------------- pan and the subtle LFOs
# The racks' pan (gacha_fx: after the effects, -1 hard left .. 1 hard right)
# moved for the style, mostly as moves in time: seq() steps a knob through
# a few values, each `step` beats long, then lets it back to its setting
# (1/8 left, 1/8 right, back to the centre), mirrored at random so it goes
# either way first; and, where a slow drift suits the style, an LFO. With
# sixteen LFO slots the styles also lean on many subtle LFOs (a pan
# wandering a little, a filter breathing, a delay send drifting): every
# style takes the palettes below (GENERIC_*), at low odds, on top of its own
# rows, which the *_EXTRA tables add to AUTO_LFO, AUTO_LFO_VOICES,
# AUTO_GESTURES and AUTO_GESTURES_VOICES. A knob gets one LFO at most, and a
# move on a knob an LFO holds is left out (the LFO has it).

def seq(stem, knob, amounts, step, chance, grid=None, count=None, bars=None, beat=0,
        flip=True):
    """The knob stepped through `amounts` (each a value or a (lo, hi) range,
    the sign kept), `step` beats each, then back to its setting: thrown at
    `count` (lo, hi) places on a grid of `grid` beats, or on `beat` of the
    bars `bars` says; `flip` mirrors the whole run at random (left first or
    right first)."""
    return {"kind": "seq", "stem": stem, "knob": knob, "amounts": amounts, "step": step,
            "chance": chance, "grid": grid, "count": count, "bars": bars, "beat": beat,
            "flip": flip, "amount": amounts[0], "shape": "hold"}


# every style's, at low odds
GENERIC_LFO = (
    ("layers", "pan", "sine", "8 bars", 0.15, False, 0.5),
    ("events", "pan", "random smooth", "2 bars", 0.3, False, 0.5),
    ("chops", "pan", "triangle", "4 bars", 0.2, False, 0.35),
    ("layers", "chorus", "sine", "8 bars", 0.1, True, 0.3),
    ("events", "delay", "random smooth", "4 bars", 0.1, True, 0.3),
)
GENERIC_LFO_VOICES = (
    ("hihat", "pan", "sine", "2 bars", 0.15, False, 0.45),
    ("ride", "pan", "sine", "4 bars", 0.2, False, 0.4),
    ("ohat", "pan", "triangle", "1 bar", 0.15, False, 0.3),
    ("glitch", "pan", "random step", "1/8", 0.5, False, 0.5),
)
GENERIC_GESTURES = (
    seq("events", "pan", (-0.7, 0.7), 0.5, 0.4, grid=2.0, count=(1, 2)),
    seq("chops", "pan", ((-0.6, -0.4),), 1.0, 0.3, grid=1.0, count=(1, 2)),
)
GENERIC_GESTURES_VOICES = (
    seq("snare", "pan", (-0.6, 0.6), 0.5, 0.4, grid=2.0, count=(1, 2)),
    seq("ohat", "pan", (-0.6, 0.6), 0.25, 0.3, grid=1.0, count=(1, 2)),
)

AUTO_LFO_EXTRA = {
    "four-floor": (("events", "delay", "sine", "4 bars", 0.15, True, 0.4),
                   ("chops", "phaser", "sine", "2 bars", 0.2, True, 0.4)),
    "breakbeat": (("layers", "lowpass", "random smooth", "4 bars", 0.2, True, 0.4),),
    "boom-bap": (("chops", "lowpass", "sine", "4 bars", 0.15, True, 0.4),),
    "halftime": (("events", "pan", "sine", "2 bars", 0.4, False, 0.5),),
    "dnb": (("chops", "pan", "random step", "1/4", 0.3, False, 0.5),
            ("events", "phaser", "sine", "2 bars", 0.2, True, 0.4)),
    "minimal": (("chops", "pan", "sine", "2 bars", 0.3, False, 0.6),
                ("events", "delay", "random smooth", "4 bars", 0.2, True, 0.5)),
    "ukg": (("layers", "chorus", "sine", "4 bars", 0.2, True, 0.5),),
    "dembow": (("events", "pan", "sine", "1 bar", 0.3, False, 0.4),),
    "one-drop": (("chops", "pan", "sine", "2 bars", 0.4, False, 0.6),
                 ("layers", "lowpass", "sine", "8 bars", 0.15, True, 0.4)),
    "footwork": (("events", "pan", "random step", "1/8", 0.4, False, 0.5),),
    "clave": (("events", "pan", "sine", "2 bars", 0.3, False, 0.5),),
    "idm": (("chops", "pan", "random step", "1/16", 0.6, False, 0.7),
            ("layers", "pan", "random smooth", "1 bar", 0.4, False, 0.5),
            ("events", "crush", "random step", "1/4", 0.3, True, 0.5)),
    "hypnotic": (("events", "pan", "sine", "8 bars", 0.3, False, 0.6),
                 ("chops", "delay", "sine", "8 bars", 0.15, True, 0.5)),
    "drone": (("layers", "pan", "sine", "8 bars", 0.35, False, 0.8),
              ("events", "reverb", "sine", "8 bars", 0.15, True, 0.6),
              ("chops", "pan", "random smooth", "4 bars", 0.4, False, 0.6)),
    "glitch": (("chops", "pan", "random step", "1/16", 0.7, False, 0.8),
               ("events", "pan", "random step", "1/8", 0.6, False, 0.7),
               ("layers", "lofi", "square", "1/4", 0.2, True, 0.4)),
    "dub": (("chops", "pan", "sine", "2 bars", 0.5, False, 0.7),
            ("events", "pan", "sine", "4 bars", 0.4, False, 0.6),
            ("layers", "lowpass", "random smooth", "8 bars", 0.2, True, 0.5)),
    "polyrhythm": (("events", "pan", "triangle", "1 bar", 0.4, False, 0.6),),
    "electro": (("chops", "pan", "square", "1/2", 0.3, False, 0.5),),
    "acid": (("chops", "pan", "sine", "1 bar", 0.3, False, 0.5),
             ("events", "lowpass", "saw down", "1/4", 0.3, True, 0.5)),
    "jungle": (("layers", "pan", "sine", "4 bars", 0.3, False, 0.5),),
    "gabber": (("layers", "pan", "square", "1/4", 0.2, False, 0.4),),
    "jersey": (("events", "pan", "random step", "1/4", 0.4, False, 0.5),),
    "gqom": (("events", "pan", "sine", "4 bars", 0.4, False, 0.6),
             ("chops", "pan", "sine", "2 bars", 0.3, False, 0.5)),
    "amapiano": (("chops", "pan", "sine", "2 bars", 0.3, False, 0.5),
                 ("layers", "lowpass", "sine", "8 bars", 0.15, True, 0.4)),
    "jazz": (("layers", "pan", "sine", "8 bars", 0.2, False, 0.5),),
    "industrial": (("chops", "pan", "random step", "1/4", 0.5, False, 0.6),
                   ("events", "drive", "square", "1/2", 0.2, True, 0.4)),
    "bruk": (("events", "pan", "sine", "2 bars", 0.3, False, 0.5),),
    "ambient": (("layers", "pan", "sine", "8 bars", 0.4, False, 0.8),
                ("events", "pan", "random smooth", "4 bars", 0.5, False, 0.7),
                ("events", "delay", "sine", "8 bars", 0.2, True, 0.5)),
    "motorik": (("chops", "pan", "sine", "4 bars", 0.3, False, 0.5),
                ("events", "pan", "sine", "8 bars", 0.3, False, 0.5)),
    "afrobeat": (("events", "pan", "sine", "2 bars", 0.3, False, 0.5),),
    "cumbia": (("events", "pan", "sine", "2 bars", 0.3, False, 0.5),),
}

AUTO_LFO_VOICES_EXTRA = {
    "four-floor": (("ohat", "pan", "sine", "1 bar", 0.25, False, 0.6),
                   ("ride", "highpass", "sine", "8 bars", 0.15, True, 0.3)),
    "breakbeat": (("hihat", "pan", "random step", "1/4", 0.25, False, 0.5),),
    "boom-bap": (("hihat", "lofi", "sine", "8 bars", 0.15, True, 0.4),),
    "halftime": (("hihat", "pan", "random step", "1/8", 0.35, False, 0.6),
                 ("snare", "reverb", "sine", "4 bars", 0.1, True, 0.4)),
    "dnb": (("hihat", "highpass", "sine", "2 bars", 0.2, True, 0.4),),
    "minimal": (("hihat", "pan", "sine", "1 bar", 0.3, False, 0.7),
                ("ride", "delay", "sine", "8 bars", 0.15, True, 0.4)),
    "ukg": (("hihat", "pan", "square", "1/4", 0.2, False, 0.5),),
    "dembow": (("hihat", "pan", "sine", "1 bar", 0.2, False, 0.5),),
    "one-drop": (("hihat", "delay", "random smooth", "4 bars", 0.2, True, 0.5),),
    "footwork": (("hihat", "pan", "random step", "1/8", 0.4, False, 0.6),),
    "clave": (("ride", "pan", "sine", "2 bars", 0.3, False, 0.6),),
    "idm": (("hihat", "pitch", "random step", "1/16", 0.3, False, 0.5),
            ("ride", "pan", "random step", "1/8", 0.5, False, 0.5)),
    "hypnotic": (("hihat", "pan", "sine", "4 bars", 0.25, False, 0.7),
                 ("ride", "phaser", "sine", "8 bars", 0.2, True, 0.4)),
    "drone": (("hihat", "pan", "sine", "8 bars", 0.3, False, 0.5),
              ("ride", "reverb", "sine", "8 bars", 0.15, True, 0.5)),
    "glitch": (("hihat", "pan", "random step", "1/16", 0.6, False, 0.7),
               ("ohat", "pan", "random step", "1/8", 0.5, False, 0.5)),
    "dub": (("hihat", "pan", "sine", "2 bars", 0.3, False, 0.6),
            ("ohat", "delay", "sine", "4 bars", 0.2, True, 0.5)),
    "polyrhythm": (("hihat", "pan", "triangle", "1/2", 0.3, False, 0.6),
                   ("ride", "pan", "triangle", "1 bar", 0.3, False, 0.5)),
    "electro": (("hihat", "pan", "square", "1/4", 0.2, False, 0.5),
                ("snare", "phaser", "sine", "4 bars", 0.15, True, 0.3)),
    "acid": (("ohat", "pan", "sine", "1/2", 0.2, False, 0.5),),
    "jungle": (("hihat", "pan", "random step", "1/8", 0.3, False, 0.5),),
    "gabber": (("hihat", "pan", "square", "1/8", 0.2, False, 0.4),),
    "jersey": (("hihat", "pan", "random step", "1/8", 0.3, False, 0.5),),
    "gqom": (("ride", "pan", "sine", "2 bars", 0.35, False, 0.6),),
    "amapiano": (("hihat", "pan", "sine", "1 bar", 0.3, False, 0.6),),
    "jazz": (("ride", "pan", "sine", "4 bars", 0.15, False, 0.5),
             ("snare", "pan", "random smooth", "2 bars", 0.2, False, 0.4)),
    "industrial": (("hihat", "pan", "random step", "1/8", 0.4, False, 0.5),),
    "bruk": (("hihat", "pan", "sine", "1 bar", 0.25, False, 0.5),),
    "ambient": (("ride", "reverb", "sine", "8 bars", 0.15, True, 0.5),
                ("hihat", "pan", "sine", "8 bars", 0.3, False, 0.5)),
    "motorik": (("hihat", "pan", "sine", "8 bars", 0.2, False, 0.5),),
    "afrobeat": (("hihat", "pan", "sine", "1 bar", 0.25, False, 0.5),),
    "cumbia": (("hihat", "pan", "sine", "1 bar", 0.3, False, 0.6),),
}

AUTO_GESTURES_EXTRA = {
    "glitch": (seq("chops", "pan", (-1.0, 1.0, -1.0, 1.0), 0.125, 0.8, grid=0.5, count=(2, 4)),),
    "idm": (seq("events", "pan", (-0.9, 0.9), 0.25, 0.7, grid=0.5, count=(2, 4)),),
    "drone": (seq("events", "pan", ((-0.8, -0.5),), 4.0, 0.5, bars=[3, 7]),),
    "ambient": (seq("events", "pan", ((-0.8, -0.5),), 4.0, 0.5, bars=[3, 7]),),
    "polyrhythm": (seq("chops", "pan", (-0.6, 0.6), 0.75, 0.5, grid=3.0, count=(1, 3)),),
    "electro": (seq("chops", "pan", (-0.6, 0.6), 0.5, 0.4, bars="even", beat=2),),
    "acid": (seq("chops", "pan", (-0.5, 0.5, -0.5, 0.5), 0.25, 0.5, grid=2.0, count=(1, 2)),),
    "dub": (seq("events", "pan", (-0.8, 0.8), 0.5, 0.6, bars="even", beat=3),),
    "footwork": (seq("events", "pan", (-0.7, 0.7), 0.25, 0.5, grid=1.0, count=(1, 3)),),
}

AUTO_GESTURES_VOICES_EXTRA = {
    "dub": (seq("snare", "pan", (-0.8, 0.8), 0.5, 0.7, bars="even", beat=3),),
    "one-drop": (seq("snare", "pan", (-0.7, 0.7), 0.5, 0.5, bars="even", beat=2),),
    "glitch": (seq("snare", "pan", (-0.9, 0.9), 0.25, 0.7, grid=0.5, count=(2, 5)),),
    "halftime": (seq("hihat", "pan", (-0.6, 0.6, -0.6, 0.6), 0.25, 0.6, grid=1.0,
                     count=(1, 3)),),
    "footwork": (seq("snare", "pan", (-0.7, 0.7), 0.25, 0.5, grid=1.0, count=(1, 3)),),
    "breakbeat": (seq("snare", "pan", (-0.5, 0.5), 0.5, 0.5, grid=1.0, count=(1, 2)),),
    "dnb": (seq("snare", "pan", (-0.5, 0.5), 0.5, 0.5, grid=1.0, count=(1, 2)),),
    "jungle": (seq("snare", "pan", (-0.6, 0.6), 0.5, 0.6, grid=1.0, count=(1, 3)),),
    "industrial": (seq("snare", "pan", (-1.0, 1.0), 0.25, 0.5, grid=1.0, count=(1, 3)),),
    "gabber": (seq("snare", "pan", (-1.0, 1.0), 0.25, 0.4, grid=1.0, count=(1, 2)),),
    "four-floor": (seq("ohat", "pan", (-0.4, 0.4), 0.5, 0.4, bars="even", beat=1),),
    "hypnotic": (seq("snare", "pan", ((-0.5, -0.3),), 0.25, 0.4, grid=1.0, count=(2, 4)),),
    "jazz": (seq("snare", "pan", ((-0.3, -0.15),), 1.0, 0.3, grid=2.0, count=(1, 2)),),
}

for _table, _extra in ((AUTO_LFO, AUTO_LFO_EXTRA), (AUTO_LFO_VOICES, AUTO_LFO_VOICES_EXTRA),
                       (AUTO_GESTURES, AUTO_GESTURES_EXTRA),
                       (AUTO_GESTURES_VOICES, AUTO_GESTURES_VOICES_EXTRA)):
    for _s, _rows in _extra.items():
        _table[_s] = tuple(_table.get(_s, ())) + tuple(_rows)
GENERIC = {id(AUTO_LFO): GENERIC_LFO, id(AUTO_LFO_VOICES): GENERIC_LFO_VOICES,
           id(AUTO_GESTURES): GENERIC_GESTURES, id(AUTO_GESTURES_VOICES): GENERIC_GESTURES_VOICES}


# ---------------------------------------------------------------- the Model:Cycles
# The box's own sound moved for the drum style (the kit's own), behind the Perform tab's third
# tick, "auto fx m:c": its tracks' pitch, decay, color, shape, sweep,
# contour, pan, delay and reverb sends, chance, punch and gate, its delay
# (time, feedback) and reverb (size, tone), and the retrig done from here,
# each with its rate, its curve's length, its fade and its velocity: the
# hats rolled in 32nds for a beat, the snare's build-up roll fading in over
# the last bar, the snare thrown into the box's echo on the fourth beat of
# the even bars for dub. A track is named by the drum it plays
# (gacha_midi.CYCLES_CHANNELS: kick T1, snare T2, hihat T3, ohat T4, ride
# T5, glitch T6), "fx" the delay and reverb. An amount is ("rel", lo, hi),
# that far from the M:C tab's own value (the centre the moves come back
# to), ("abs", lo, hi), ("on", True / False) for a switch, or ("pick",
# values). The palette (MC_GENERIC) is every style's, at low odds; AUTO_MC
# adds each style's own; the drum kind playing scales the lot
# (MC_INTENSITY: a sparse beat moves the box a little, an overload one a
# lot). plan_mc() draws a section's, compile_mc() makes it the rows the
# audio engine sends on the loop (LiveAudio.set_mc_moves).

from gacha_midi import (CYCLES_CHANNELS, CYCLES_FX_PARAMS, CYCLES_PARAMS,  # noqa: E402
                        CYCLES_RANGES, CYCLES_RETRIG_LENS, CYCLES_RETRIG_RATES,
                        cycles_cc_value)

MC_KEYS = ("pitch", "decay", "color", "shape", "sweep", "contour", "pan", "dsend", "rsend",
           "chance", "punch", "gate")
MC_FX_KEYS = ("delay_time", "delay_fb", "rev_size", "rev_tone")
MC_PARAMS = {key: (cc, kind) for key, cc, _l, kind, _d, _w in CYCLES_PARAMS + CYCLES_FX_PARAMS
             if key in MC_KEYS + MC_FX_KEYS}
MC_RATES = dict(CYCLES_RETRIG_RATES)      # "1/32" -> a bar's fraction
MC_LENS = dict(CYCLES_RETRIG_LENS)        # "1 step" -> 16ths (0 = a flat curve)
# the drum kind playing (the section's own, or the drum kind box's): how much
MC_INTENSITY = {"intro": 0.5, "sparse": 0.5, "break": 0.7, "build": 0.8, "outro": 0.8,
                "groove": 1.0, "dense": 1.3, "heavy": 1.6, "insane": 2.0, "overload": 2.6}


def hold(track, key, amount, chance):
    """The knob set for the whole section."""
    return {"kind": "hold", "stem": track, "knob": key, "amount": amount, "chance": chance}


def retrig(track, length, rates, lens, fades, vels, chance, grid=None, count=None,
           bars=None, beat=0):
    """A retrig of `length` beats: thrown at `count` (lo, hi) places on a grid
    of `grid` beats, or on `beat` of the bars `bars` says; its rate, its
    curve's length (CYCLES_RETRIG_RATES, CYCLES_RETRIG_LENS names), its fade
    (-128..127) and its velocity drawn from what is given (a fade or a
    velocity as (lo, hi))."""
    return {"kind": "retrig", "stem": track, "knob": "retrig", "length": length,
            "rates": rates, "lens": lens, "fades": fades, "vels": vels, "chance": chance,
            "grid": grid, "count": count, "bars": bars, "beat": beat}


HATS = ("hihat", "ohat")
TONES = ("ride", "glitch")                 # T5 and T6: a tone and a chord machine by default
ANY = ("kick", "snare", "hihat", "ohat", "ride", "glitch")

# every style's, at low odds: the box breathing
MC_GENERIC = (
    hold(ANY, "decay", ("rel", -20, 20), 0.4),
    hold("fx", "delay_time", ("pick", (12, 16, 24, 32, 48)), 0.5),
    hold("fx", "rev_size", ("abs", 50, 100), 0.4),
    hold("fx", "rev_tone", ("rel", -20, 20), 0.3),
    throw("snare", "dsend", ("abs", 70, 110), 0.5, 1.0, (1, 2), 0.4),
    onbar("snare", "rsend", ("abs", 60, 100), 1.0, "last", 3, 0.3),
    throw(ANY, "color", ("rel", -35, 35), 1.0, 1.0, (1, 2), 0.3),
    throw(ANY, "pan", ("rel", -40, 40), 0.5, 0.5, (1, 3), 0.25),
    ramp("kick", "sweep", ("rel", 15, 40), "last4", "in", 0.2),
    ramp("fx", "delay_fb", ("abs", 90, 115), "last", "in", 0.25),
    retrig("snare", 1.0, ("1/16", "1/32"), ("1 step", "2 steps", "4 steps"), (40, 127),
           (80, 110), 0.35, bars="last", beat=3),
    retrig(HATS, 0.5, ("1/32", "1/24"), ("1 step", "2 steps"), (-60, 60), (60, 100), 0.3,
           grid=1.0, count=(1, 2)),
)

AUTO_MC = {
    "four-floor": (ramp("hihat", "decay", ("rel", 10, 35), "last4", "in", 0.6),
                   onbar("ohat", "dsend", ("abs", 60, 90), 1.0, "even", 2, 0.5),
                   retrig("snare", 2.0, ("1/16", "1/32"), ("8 steps",), (60, 127), (70, 100),
                          0.5, bars="last", beat=2)),
    "breakbeat": (retrig("snare", 0.5, ("1/32", "1/16"), ("1 step",), (-40, 40), (70, 110),
                         0.5, grid=1.0, count=(1, 2)),
                  throw("snare", "pitch", ("rel", -4, 4), 0.5, 1.0, (1, 2), 0.4)),
    "boom-bap": (hold("snare", "punch", ("on", True), 0.6),
                 hold(HATS, "decay", ("rel", -25, -10), 0.5),
                 throw("snare", "dsend", ("abs", 50, 80), 1.0, 2.0, (1, 1), 0.4)),
    "halftime": (retrig("hihat", 1.0, ("1/24", "1/32", "1/48"), ("2 steps", "4 steps"),
                        (-90, 90), (60, 100), 0.85, grid=1.0, count=(2, 4)),
                 throw("kick", "pitch", ("rel", -7, -3), 1.0, 2.0, (1, 2), 0.5),
                 onbar("snare", "rsend", ("abs", 70, 110), 1.0, "last", 2, 0.5)),
    "dnb": (retrig("snare", 0.5, ("1/32",), ("1 step", "2 steps"), (-40, 60), (70, 110),
                   0.6, grid=1.0, count=(1, 3)),
            throw(HATS, "decay", ("rel", -30, -10), 1.0, 1.0, (1, 3), 0.4)),
    "minimal": (count("hihat", "decay", ("rel", 15, 35), (3, 5), "all", 0.5),
                count("ohat", "pan", ("rel", -45, 45), (5, 7), "last4", 0.5),
                throw("ride", "dsend", ("abs", 60, 95), 1.0, 1.0, (1, 2), 0.4)),
    "ukg": (retrig("hihat", 0.5, ("1/32",), ("1 step",), (-30, 30), (50, 90), 0.7,
                   grid=0.5, count=(1, 3)),
            throw("snare", "rsend", ("abs", 60, 95), 0.5, 1.0, (1, 2), 0.4)),
    "dembow": (onbar("snare", "dsend", ("abs", 60, 90), 1.0, "even", 3, 0.5),
               retrig("hihat", 0.5, ("1/16", "1/32"), ("1 step",), (0, 0), (60, 90), 0.4,
                      grid=1.0, count=(1, 2))),
    "one-drop": (onbar("snare", "dsend", ("abs", 100, 127), 1.0, "even", 2, 0.9),
                 hold("fx", "delay_time", ("pick", (24, 48)), 0.9),
                 ramp("fx", "delay_fb", ("abs", 95, 118), "last", "in", 0.6),
                 hold("snare", "rsend", ("abs", 30, 60), 0.6)),
    "footwork": (retrig(HATS, 0.5, ("1/24", "1/12"), ("1 step", "2 steps"), (-60, 60),
                        (60, 100), 0.8, grid=0.5, count=(2, 5)),
                 retrig("kick", 0.5, ("1/12", "1/24"), ("1 step",), (-80, 0), (80, 110), 0.5,
                        grid=1.0, count=(1, 2)),
                 throw("kick", "pitch", ("rel", 3, 7), 0.5, 1.0, (1, 3), 0.4)),
    "clave": (throw("ride", "pitch", ("rel", 3, 7), 0.5, 1.0, (1, 2), 0.4),
              onbar("snare", "rsend", ("abs", 50, 80), 1.0, "last", 3, 0.4)),
    "idm": (retrig(ANY, 0.25, ("1/48", "1/64", "1/32", "1/20"), ("1/2 step", "1 step"),
                   (-128, 127), (50, 110), 0.85, grid=0.25, count=(3, 7)),
            throw(ANY, "pitch", ("rel", -12, 12), 0.25, 0.25, (2, 5), 0.6),
            throw(ANY, "decay", ("rel", -40, 30), 0.5, 0.5, (2, 4), 0.6),
            throw(ANY, "shape", ("rel", -40, 40), 0.5, 0.5, (1, 3), 0.5)),
    "hypnotic": (count("hihat", "decay", ("rel", 15, 40), (5, 7), "all", 0.7),
                 count("snare", "dsend", ("abs", 60, 95), (5, 7), "last4", 0.5),
                 ramp("ride", "sweep", ("rel", 10, 30), "last4", "in", 0.4)),
    "drone": (hold(ANY, "rsend", ("abs", 60, 100), 0.8),
              hold("fx", "rev_size", ("abs", 110, 126), 0.9),
              hold(ANY, "decay", ("rel", 20, 45), 0.6),
              ramp(TONES, "sweep", ("rel", 15, 40), "last4", "in", 0.5)),
    "glitch": (retrig(ANY, 0.25, ("1/32", "1/48", "1/64"), ("1/2 step", "1 step"),
                      (-128, 127), (50, 110), 0.9, grid=0.25, count=(4, 8)),
               throw(ANY, "pitch", ("rel", -12, 12), 0.25, 0.25, (2, 5), 0.7),
               throw(ANY, "decay", ("rel", -50, -20), 0.25, 0.25, (2, 5), 0.6),
               throw(ANY, "contour", ("rel", -40, 40), 0.5, 0.5, (1, 3), 0.5),
               hold(HATS, "chance", ("abs", 50, 85), 0.4)),
    "dub": (onbar("snare", "dsend", ("abs", 105, 127), 1.0, "even", 3, 0.95),
            hold("fx", "delay_time", ("pick", (24, 48, 32)), 0.95),
            ramp("fx", "delay_fb", ("abs", 100, 120), "last", "in", 0.7),
            hold("snare", "rsend", ("abs", 40, 70), 0.7),
            throw(HATS, "dsend", ("abs", 70, 100), 1.0, 2.0, (1, 2), 0.5),
            hold("fx", "rev_tone", ("rel", -30, -10), 0.5)),
    "polyrhythm": (count(HATS, "pan", ("rel", -45, 45), (5, 7), "all", 0.6),
                   count("ride", "color", ("rel", -30, 30), (3, 5), "last4", 0.5)),
    "electro": (onbar("snare", "dsend", ("abs", 60, 90), 1.0, "even", 3, 0.5),
                throw(TONES, "sweep", ("rel", -30, 30), 1.0, 1.0, (1, 2), 0.5),
                hold("kick", "punch", ("on", True), 0.5)),
    "acid": (throw(TONES, "sweep", ("rel", 20, 50), 0.25, 0.25, (4, 10), 0.8),
             ramp(TONES, "contour", ("rel", 15, 40), "last2", "in", 0.6),
             hold(TONES, "gate", ("on", True), 0.4),
             retrig("ride", 0.5, ("1/16",), ("1 step",), (0, 40), (70, 100), 0.4,
                    grid=1.0, count=(1, 2))),
    "jungle": (retrig("snare", 0.5, ("1/32", "1/24"), ("1 step", "2 steps"), (-60, 60),
                      (70, 110), 0.7, grid=0.5, count=(1, 3)),
               throw("snare", "pitch", ("rel", -7, 7), 0.5, 1.0, (1, 3), 0.6),
               onbar("snare", "dsend", ("abs", 70, 100), 1.0, "even", 3, 0.5)),
    "gabber": (hold("kick", "punch", ("on", True), 0.95),
               retrig("kick", 2.0, ("1/16", "1/32"), ("8 steps",), (60, 127), (90, 120), 0.6,
                      bars="last", beat=2),
               throw("kick", "contour", ("rel", 15, 40), 1.0, 1.0, (1, 3), 0.5)),
    "jersey": (retrig("kick", 0.5, ("1/12", "1/24"), ("1 step",), (-40, 0), (80, 110), 0.7,
                      grid=1.0, count=(1, 3)),
               throw("snare", "pitch", ("rel", 3, 7), 0.5, 1.0, (1, 2), 0.4)),
    "gqom": (hold("kick", "rsend", ("abs", 30, 60), 0.6),
             throw("kick", "pitch", ("rel", -5, -2), 1.0, 2.0, (1, 2), 0.5),
             onbar("ride", "rsend", ("abs", 70, 100), 1.0, "even", 3, 0.5)),
    "amapiano": (throw("kick", "pitch", ("rel", -12, -5), 0.5, 1.0, (2, 4), 0.7),
                 hold("kick", "decay", ("rel", 20, 40), 0.6),
                 retrig("hihat", 0.5, ("1/16", "1/32"), ("1 step",), (0, 30), (50, 80), 0.4,
                        grid=1.0, count=(1, 2))),
    "jazz": (hold("ride", "rsend", ("abs", 40, 70), 0.8),
             retrig("snare", 1.0, ("1/12", "1/16"), ("4 steps",), (-90, -30), (40, 70), 0.5,
                    grid=2.0, count=(1, 2)),
             hold("ride", "decay", ("rel", 10, 30), 0.5)),
    "industrial": (hold(("kick", "snare"), "punch", ("on", True), 0.8),
                   retrig(("kick", "snare"), 0.5, ("1/16", "1/32"), ("1 step",), (-60, 60),
                          (90, 127), 0.7, grid=0.5, count=(2, 4)),
                   throw(ANY, "contour", ("rel", -40, 40), 1.0, 1.0, (2, 4), 0.6),
                   hold("fx", "rev_tone", ("rel", 10, 30), 0.4)),
    "bruk": (throw("snare", "pitch", ("rel", -3, 3), 0.5, 1.0, (1, 2), 0.4),
             retrig("hihat", 0.5, ("1/24",), ("1 step",), (-40, 40), (60, 90), 0.4,
                    grid=1.0, count=(1, 2))),
    "ambient": (hold(ANY, "rsend", ("abs", 70, 110), 0.9),
                hold("fx", "rev_size", ("abs", 115, 126), 0.9),
                hold(ANY, "decay", ("rel", 25, 50), 0.7),
                ramp(TONES, "sweep", ("rel", 10, 30), "first4", "in", 0.5)),
    "motorik": (count("hihat", "decay", ("rel", 10, 30), (4, 8), "all", 0.4),
                onbar("snare", "dsend", ("abs", 60, 90), 1.0, [8], 3, 0.5)),
    "afrobeat": (throw("ride", "pitch", ("rel", 2, 5), 0.5, 1.0, (1, 2), 0.4),
                 retrig("hihat", 0.5, ("1/16",), ("1 step",), (0, 30), (50, 80), 0.3,
                        grid=1.0, count=(1, 2))),
    "cumbia": (throw("ride", "pitch", ("rel", 2, 5), 0.5, 1.0, (1, 2), 0.4),
               onbar("snare", "rsend", ("abs", 50, 80), 1.0, [8], 2, 0.4)),
}


def _mc_draw(rng, a):
    """An amount drawn: ("rel", d), ("abs", v) or ("on", b)."""
    if a[0] == "on":
        return ("on", bool(a[1]))
    if a[0] == "pick":
        return ("abs", rng.choice(a[1]))
    return (a[0], int(round(rng.uniform(a[1], a[2]))))


def _mc_num(rng, x):
    return int(round(rng.uniform(*x))) if isinstance(x, (tuple, list)) else int(x)


def _mc_starts(rng, g, bars, k):
    """Where a throw or a retrig goes, in beats: `count` places on its grid
    (the count scaled by `k`), or `beat` of its bars."""
    if g.get("grid"):
        lo, hi = g["count"]
        n = max(1, int(round(rng.randint(lo, hi) * k)))
        slots = int(bars * 4 / g["grid"])
        return [i * g["grid"] for i in sorted(rng.sample(range(slots), min(n, slots)))]
    return [(b - 1) * 4 + g["beat"] for b in _bars_of(g["bars"], bars)]


def plan_mc(style, rng=None, bars=8, rung="groove", exclude=(), sampler=()):
    """A section's moves for the Model:Cycles: the palette (MC_GENERIC)
    and the style's own (AUTO_MC), each line with its chance and the
    throws' counts scaled by the drum kind (`rung`, MC_INTENSITY); a build
    adds the snare's roll into the drop. `exclude`: (track, key) left
    alone (an LFO holds the cell); `sampler`: the tracks on the Model-TG
    sampler (no pitch: the CC would pin it). {"held": {(track, key):
    amount}, "moves": [(track, key, start, length, amount, shape)],
    "retrigs": [(track, start, length, rate, curve, fade, velocity)]}, in
    beats from the loop's start; all empty for the default feel."""
    rng = rng or random.Random()
    out = {"held": {}, "moves": [], "retrigs": []}
    if style == "default":
        return out
    k = MC_INTENSITY.get(rung, 1.0)
    rows = MC_GENERIC + AUTO_MC.get(style, ())
    if rung == "build":
        rows += (retrig("snare", 2.0, ("1/16", "1/32"), ("8 steps",), (90, 127), (60, 100),
                        2.0, bars="last", beat=2),)   # whatever the build's own scale: always
    for g in rows:
        if rng.random() >= min(1.0, g["chance"] * k):
            continue
        track = g["stem"] if isinstance(g["stem"], str) else rng.choice(g["stem"])
        key, kind = g["knob"], g["kind"]
        if kind == "retrig":
            if track not in CYCLES_CHANNELS:
                continue
            for st in _mc_starts(rng, g, bars, k):
                out["retrigs"].append((track, st, g["length"], rng.choice(g["rates"]),
                                       rng.choice(g["lens"]), _mc_num(rng, g["fades"]),
                                       max(1, min(127, _mc_num(rng, g["vels"])))))
            continue
        if key not in MC_PARAMS or (track, key) in exclude \
                or (key == "pitch" and track in sampler) \
                or (track == "fx") != (key in MC_FX_KEYS) \
                or (track != "fx" and track not in CYCLES_CHANNELS):
            continue
        if kind == "hold":
            out["held"][(track, key)] = _mc_draw(rng, g["amount"])
        elif kind in ("throw", "onbar"):
            for st in _mc_starts(rng, g, bars, k if kind == "throw" else 1.0):
                out["moves"].append((track, key, st, g["length"], _mc_draw(rng, g["amount"]),
                                     g["shape"]))
        elif kind == "ramp":
            bs = _bars_of(g["bars"], bars)
            if bs:
                start, end = (bs[0] - 1) * 4, bs[-1] * 4
                out["moves"].append((track, key, start, end - start,
                                     _mc_draw(rng, g["amount"]), g["shape"]))
        elif kind == "count":
            bs = _bars_of(g["bars"], bars)
            if bs:
                period = rng.choice(g["periods"])
                start, end = (bs[0] - 1) * 16, bs[-1] * 16
                amt = _mc_draw(rng, g["amount"])
                for st in range(start + rng.randrange(period), end, period):
                    out["moves"].append((track, key, st / 4.0, g["length"], amt, "hold"))
    return out


def _mc_value(kind, base, amount):
    """An amount on a knob of `kind` whose value is `base` now, clamped to
    the knob's range (a switch: the amount's state)."""
    what, v = amount
    if kind == "on":
        return bool(v) if what == "on" else bool(base)
    lo, hi = CYCLES_RANGES[kind]
    return max(lo, min(hi, int(round(base + v if what == "rel" else v))))


def compile_mc(plan, base, bars=8, fx_channel=9, notes=None):
    """A plan_mc() plan as the rows the audio engine runs
    (LiveAudio.set_mc_moves): (beat, order, kind, args) sorted, "cc"
    (channel, cc, value 0..127, the value as the M:C tab shows it, for
    its cells) wherever a knob's value changes and on
    the loop's start for every knob touched (so each loop starts right),
    "rt" (channel, track, note, velocity, rate in bars, curve in 16ths,
    fade) and "rx" (channel,). `base`: {(track, key): the M:C tab's value}
    the moves come back to (a knob missing from it is left alone);
    `notes`: {track: the note its retrig plays}, none without one.
    Returns (rows, the (channel, cc) touched)."""
    notes = notes or {}
    loop = bars * 4.0
    held = plan.get("held", {})
    segs = {key: [] for key in held}                  # (track, key) -> [(start, end, shape, amount)]
    for track, key, start, length, amount, shape in plan.get("moves", ()):
        if length > 0 and start < loop:
            segs.setdefault((track, key), []).append(
                (float(start), min(loop, float(start) + float(length)), shape, amount))
    rows, touched = [], set()
    for (track, key), ss in segs.items():
        if key not in MC_PARAMS or (track, key) not in base:
            continue
        cc, kind = MC_PARAMS[key]
        ch = fx_channel if track == "fx" else CYCLES_CHANNELS.get(track)
        if ch is None:
            continue
        centre = base[(track, key)]
        if (track, key) in held:
            centre = _mc_value(kind, centre, held[(track, key)])
        ss = sorted((s, e, "hold" if kind == "on" else sh, _mc_value(kind, centre, a))
                    for s, e, sh, a in ss)

        def at(x, ss=ss, centre=centre):
            v = centre
            for start, end, shape, target in ss:      # the latest one begun wins
                if start <= x < end:
                    f = (x - start) / (end - start)
                    v = (centre + (target - centre) * f if shape == "in"
                         else target + (centre - target) * f if shape == "out" else target)
            return v

        points = {0.0}
        for start, end, shape, _t in ss:
            points.add(start)
            if end < loop:
                points.add(end)
            if shape in ("in", "out"):                # a ramp: a step every 16th
                x = start + 0.25
                while x < end - 1e-9:
                    points.add(round(x, 4))
                    x += 0.25
        last = None
        for x in sorted(points):
            tab = at(x)
            v = int(cycles_cc_value(kind, tab))
            if x == 0.0 or v != last:
                rows.append((x, 1, "cc", (ch, cc, v, bool(tab) if kind == "on"
                                          else int(round(tab)))))
                last = v
        touched.add((ch, cc))
    by_track = {}
    for r in plan.get("retrigs", ()):
        by_track.setdefault(r[0], []).append(r)
    for track, rts in by_track.items():
        ch, note = CYCLES_CHANNELS.get(track), notes.get(track)
        if ch is None or note is None or not 0 <= note <= 127:
            continue
        rts.sort(key=lambda r: r[1])
        for i, (_t, start, length, rate, curve, fade, vel) in enumerate(rts):
            end = min(loop, start + length)
            if i + 1 < len(rts):
                end = min(end, rts[i + 1][1])         # the next one takes over
            if end <= start or start >= loop or rate not in MC_RATES:
                continue
            rows.append((float(start), 2, "rt", (ch, track, int(note), int(vel), MC_RATES[rate],
                                                 MC_LENS.get(curve, 1.0), int(fade))))
            rows.append((float(end) % loop, 0, "rx", (ch,)))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, touched


def describe_mc(plan):
    """A plan as the log says it: 'held fx delay_time 24, snare punch on;
    12 moves; 3 retrigs (hihat 1/32, snare 1/16)'."""
    def amt(a):
        return ("on" if a[1] else "off") if a[0] == "on" else (f"{a[1]:+d}" if a[0] == "rel"
                                                               else str(a[1]))
    held = ", ".join(f"{t} {k} {amt(a)}" for (t, k), a in sorted(plan.get("held", {}).items()))
    moves, rts = plan.get("moves", ()), plan.get("retrigs", ())
    parts = [f"held {held}" if held else "",
             f"{len(moves)} moves" if moves else "",
             (f"{len(rts)} retrigs (" + ", ".join(sorted({f"{r[0]} {r[3]}" for r in rts})) + ")")
             if rts else ""]
    return "; ".join(p for p in parts if p) or "nothing"
