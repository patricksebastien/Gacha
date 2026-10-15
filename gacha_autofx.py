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

    plan("drone", random.Random(seed))   # {("layers", "reverb"): 0.52, ("chops", "reverse"): 0.41}
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


def describe(p):
    """A plan as the log says it: 'layers reverb 0.52, chops reverse 0.41'."""
    return ", ".join(f"{stem} {knob} {v:.2f}" for (stem, knob), v in sorted(p.items())) or "nothing"
