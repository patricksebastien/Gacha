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
