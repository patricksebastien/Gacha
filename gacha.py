#!/usr/bin/env python3
"""
gacha.py — generative sample chopper.

Loads every sample in ./samples, carves drum hits (kick / snare / hihat)
out of randomly chosen samples, sequences a steady-but-experimental
pattern, layers looped / reversed / chopped background textures, runs
everything through randomized pedalboard effect chains (reverb, delay,
distortion, chorus, phaser, bitcrush...), and renders a ~3 minute song.

Every run is different. Every sample is used at least once.

Usage:
    python3 gacha.py                 # random seed
    python3 gacha.py --seed 42       # reproducible run
    python3 gacha.py --duration 180  # length in seconds
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from pedalboard import (
    Pedalboard, Reverb, Delay, Distortion, Chorus, Phaser, Compressor,
    Limiter, Gain, HighpassFilter, LowpassFilter, PitchShift, Bitcrush,
    LadderFilter, GSMFullRateCompressor,
)

SR = 44100
SAMPLES_DIR = Path(__file__).parent / "samples"
OUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------- loading

def load_samples():
    """Load every audio file as float32 stereo (frames, 2) at 44.1 kHz."""
    bank = {}
    files = sorted(SAMPLES_DIR.glob("*.wav")) + sorted(SAMPLES_DIR.glob("*.flac")) \
        + sorted(SAMPLES_DIR.glob("*.aif*")) + sorted(SAMPLES_DIR.glob("*.mp3"))
    for f in files:
        try:
            data, sr = sf.read(f, dtype="float32", always_2d=True)
        except Exception as e:
            print(f"  ! skipping {f.name}: {e}")
            continue
        if sr != SR:
            data = librosa.resample(data.T, orig_sr=sr, target_sr=SR).T
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak * 0.9
        bank[f.name] = np.ascontiguousarray(data[:, :2])
    return bank


# ---------------------------------------------------------------- helpers

def mono(x):
    return x.mean(axis=1)


def apply_fx(clip, board, tail=2.0):
    """Run a clip through a pedalboard, padded so reverb/delay tails ring out."""
    pad = np.zeros((int(tail * SR), 2), dtype="float32")
    padded = np.vstack([clip, pad])
    out = board(padded.T, SR).T  # pedalboard wants (channels, frames)
    peak = np.abs(out).max()
    if peak > 1.0:
        out = out / peak
    return out.astype("float32")


def envelope(n, attack=0.002, decay_curve=4.0):
    """Percussive amp envelope: fast attack, exponential decay."""
    env = np.exp(-decay_curve * np.linspace(0, 1, n))
    a = max(1, int(attack * SR))
    env[:a] *= np.linspace(0, 1, a)
    env[-min(64, n):] *= np.linspace(1, 0, min(64, n))
    return env[:, None]


def strongest_onset(clip):
    """Sample index of the loudest transient in a clip."""
    m = np.abs(mono(clip))
    if len(m) < 512:
        return 0
    env = np.convolve(m, np.ones(256) / 256, mode="same")
    return max(0, int(np.argmax(env)) - 128)


def panned(clip, p):
    """Equal-power stereo balance. p in [-1, 1]: -1 = hard left, 0 = center."""
    if p == 0:
        return clip
    out = clip.copy()
    out[:, 0] *= np.sqrt(1 - p)
    out[:, 1] *= np.sqrt(1 + p)
    return out


def place(buf, clip, start, gain=1.0):
    """Mix a clip into the master buffer at a sample offset."""
    if start >= len(buf):
        return
    end = min(len(buf), start + len(clip))
    buf[start:end] += clip[: end - start] * gain


def crossfade_loop(clip, target_len, fade=0.05):
    """Tile a clip to target_len with equal-power crossfades at the seams."""
    n_fade = min(int(fade * SR), len(clip) // 4)
    out = np.zeros((target_len, 2), dtype="float32")
    pos = 0
    step = len(clip) - n_fade
    fade_in = np.sqrt(np.linspace(0, 1, n_fade))[:, None]
    while pos < target_len:
        c = clip.copy()
        if pos > 0 and n_fade > 0:
            c[:n_fade] *= fade_in
        end = min(target_len, pos + len(c))
        out[pos:end] += c[: end - pos]
        pos += step if step > 0 else len(clip)
    return out


# ---------------------------------------------------------------- drum kit

def carve_drum(rng, clip, role):
    """Cut a short chunk from any sample and sculpt it into a drum hit."""
    start = strongest_onset(clip)
    if role == "kick":
        length = rng.uniform(0.12, 0.28)
        board = Pedalboard([
            LowpassFilter(cutoff_frequency_hz=rng.uniform(120, 300)),
            Distortion(drive_db=rng.uniform(6, 18)),
            Gain(gain_db=6),
        ])
    elif role == "snare":
        length = rng.uniform(0.12, 0.3)
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=rng.uniform(150, 350)),
            LowpassFilter(cutoff_frequency_hz=rng.uniform(6000, 12000)),
            Distortion(drive_db=rng.uniform(0, 10)),
            Reverb(room_size=0.15, wet_level=rng.uniform(0.02, 0.12), dry_level=0.9),
        ])
    elif role == "ride":
        length = rng.uniform(0.4, 1.0)
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=rng.uniform(2500, 5000)),
            Reverb(room_size=0.3, wet_level=0.15, dry_level=0.85),
            Gain(gain_db=2),
        ])
    else:  # hihat
        length = rng.uniform(0.03, 0.11)
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=rng.uniform(4000, 8000)),
            Gain(gain_db=3),
        ])
    n = int(length * SR)
    chunk = clip[start:start + n]
    if len(chunk) < n:
        chunk = np.vstack([chunk, np.zeros((n - len(chunk), 2), dtype="float32")])
    decay = {"kick": 3.5, "snare": 5.0, "hihat": 5.0, "ride": 2.0}[role]
    chunk = chunk * envelope(len(chunk), decay_curve=decay)
    hit = apply_fx(chunk, board, tail=0.3)
    peak = np.abs(hit).max()
    if peak > 0:
        hit = hit / peak * 0.9
    return hit


# ---------------------------------------------------------------- patterns

DRUM_STYLES = ["four-floor", "breakbeat", "boom-bap", "halftime", "dnb",
               "minimal", "ukg", "dembow", "one-drop", "footwork", "clave",
               "idm"]


def make_pattern(rng, role, style, steps=16):
    """A one-bar step pattern: list of velocities (0 = rest), 16th-note grid.
    The style decides the rhythmic skeleton for kick/snare/hihat."""
    pat = [0.0] * steps
    if role == "ride":
        offset = rng.choice([0, 0, 2])              # on-beat or offbeat ride
        for s in range(offset, steps, 4):
            pat[s] = 1.0 if s % 8 == 0 else rng.uniform(0.6, 0.9)
        return pat

    if style == "four-floor":
        if role == "kick":
            for s in (0, 4, 8, 12):
                pat[s] = 1.0
        elif role == "snare":
            pat[4], pat[12] = 1.0, 1.0              # clap on 2 & 4
            if rng.random() < 0.2:
                pat[15] = 0.3
        else:                                       # offbeat open hats
            for s in (2, 6, 10, 14):
                pat[s] = rng.uniform(0.85, 1.0)
            for s in range(0, steps, 2):
                if pat[s] == 0 and rng.random() < 0.35:
                    pat[s] = rng.uniform(0.3, 0.5)
    elif style == "breakbeat":
        if role == "kick":
            pat[0] = 1.0
            for s in (3, 6, 10, 11, 14):
                if rng.random() < 0.35:
                    pat[s] = rng.uniform(0.7, 1.0)
        elif role == "snare":
            pat[4], pat[12] = 1.0, 1.0
            for s in range(steps):
                if pat[s] == 0 and rng.random() < 0.15:
                    pat[s] = rng.uniform(0.2, 0.5)  # lots of ghosts
        else:
            div = rng.choice([1, 1, 2])             # busy 16th hats
            for s in range(0, steps, div):
                if rng.random() < 0.85:
                    pat[s] = 1.0 if s % 4 == 0 else rng.uniform(0.4, 0.7)
    elif style == "boom-bap":
        if role == "kick":
            pat[0] = 1.0
            for s in rng.sample([3, 6, 7, 10, 11], rng.randint(1, 2)):
                pat[s] = rng.uniform(0.8, 1.0)
        elif role == "snare":
            pat[4], pat[12] = 1.0, 1.0
            if rng.random() < 0.25:
                pat[rng.choice([7, 15])] = 0.3      # rare ghost
        else:                                       # lazy 8th hats
            for s in range(0, steps, 2):
                if rng.random() < 0.9:
                    pat[s] = 0.9 if s % 4 == 0 else rng.uniform(0.35, 0.55)
    elif style == "halftime":
        if role == "kick":
            pat[0] = 1.0
            if rng.random() < 0.5:
                pat[rng.choice([6, 7, 10])] = rng.uniform(0.8, 1.0)
        elif role == "snare":
            pat[8] = 1.0                            # snare on beat 3 only
        else:                                       # trap-style 16th hats
            for s in range(steps):
                if rng.random() < 0.85:
                    pat[s] = 0.9 if s % 4 == 0 else rng.uniform(0.3, 0.6)
    elif style == "dnb":
        if role == "kick":
            pat[0] = 1.0                            # two-step: 1 + "and" of 3
            pat[10] = rng.uniform(0.85, 1.0)
            if rng.random() < 0.3:
                pat[6] = 0.7
        elif role == "snare":
            pat[4], pat[12] = 1.0, 1.0
            for s in (7, 11, 15):
                if rng.random() < 0.25:
                    pat[s] = rng.uniform(0.25, 0.45)
        else:
            for s in range(0, steps, 2):
                pat[s] = 0.8 if s % 4 == 0 else rng.uniform(0.35, 0.6)
    elif style == "ukg":                            # 2-step garage: shuffly
        if role == "kick":
            pat[0] = 1.0                            # no kick on beat 3
            pat[rng.choice([10, 11])] = rng.uniform(0.8, 1.0)
            if rng.random() < 0.3:
                pat[7] = 0.7
        elif role == "snare":
            pat[4], pat[12] = 1.0, 1.0
            if rng.random() < 0.4:
                pat[rng.choice([9, 15])] = 0.35
        else:                                       # skippy offbeat hats
            for s in (2, 6, 10, 14):
                if rng.random() < 0.8:
                    pat[s] = rng.uniform(0.6, 1.0)
            for s in (3, 7, 11, 15):
                if rng.random() < 0.3:
                    pat[s] = rng.uniform(0.3, 0.5)
    elif style == "dembow":                         # reggaeton boom-ch-boom
        if role == "kick":
            for s in (0, 4, 8, 12):
                pat[s] = 1.0
        elif role == "snare":
            for s in (3, 6, 11, 14):                # the dembow snare figure
                pat[s] = 0.9 if s in (3, 11) else 1.0
        else:
            for s in range(0, steps, 2):
                if rng.random() < 0.7:
                    pat[s] = 0.8 if s % 4 == 0 else rng.uniform(0.3, 0.5)
    elif style == "one-drop":                       # dub/reggae: all on beat 3
        if role == "kick":
            pat[8] = 1.0
        elif role == "snare":
            pat[8] = 0.9                            # rim together with kick
            if rng.random() < 0.3:
                pat[14] = 0.3
        else:                                       # steady closed 8th hats
            for s in range(0, steps, 2):
                pat[s] = 0.7 if s % 4 == 0 else rng.uniform(0.35, 0.5)
    elif style == "footwork":                       # juke: tom-like kick cells
        if role == "kick":
            pat[0] = 1.0
            for s in rng.choice([(3, 6), (5, 10), (6, 12), (3, 6, 12)]):
                pat[s] = rng.uniform(0.8, 1.0)
        elif role == "snare":
            if rng.random() < 0.5:
                pat[rng.choice([4, 12])] = 0.9      # sparse clap
        else:                                       # minimal hats, off-grid feel
            for s in (2, 7, 10, 15):
                if rng.random() < 0.4:
                    pat[s] = rng.uniform(0.4, 0.6)
    elif style == "clave":                          # afro-latin, son clave 3-2
        if role == "kick":
            pat[0], pat[6] = 1.0, rng.uniform(0.8, 1.0)   # bombo feel
        elif role == "snare":
            for s in (0, 3, 6, 10, 12):             # clave hits, played light
                pat[s] = rng.uniform(0.4, 0.7)
        else:
            for s in range(0, steps, 2):
                if rng.random() < 0.85:
                    pat[s] = 0.8 if s % 4 == 0 else rng.uniform(0.4, 0.6)
    elif style == "idm":                            # structured chaos
        if role == "kick":
            pat[0] = 1.0
            for s in rng.sample(range(1, steps), rng.randint(2, 4)):
                pat[s] = rng.uniform(0.6, 1.0)
        elif role == "snare":
            for s in rng.sample(range(steps), rng.randint(1, 3)):
                pat[s] = rng.uniform(0.5, 1.0)
        else:
            for s in range(steps):
                if rng.random() < 0.5:
                    pat[s] = rng.uniform(0.2, 1.0)
    else:  # minimal
        if role == "kick":
            pat[0], pat[8] = 1.0, rng.uniform(0.8, 1.0)
            if rng.random() < 0.3:
                pat[4] = 0.9
        elif role == "snare":
            if rng.random() < 0.6:
                pat[12] = 0.8                       # one hit per bar, if any
        else:                                       # sparse quarter hats
            for s in range(0, steps, 4):
                if rng.random() < 0.7:
                    pat[s] = rng.uniform(0.4, 0.7)
    return pat


def mutate(rng, pat, amount=0.15):
    pat = list(pat)
    for s in range(len(pat)):
        if rng.random() < amount:
            pat[s] = 0.0 if pat[s] > 0 and rng.random() < 0.5 else rng.uniform(0.4, 1.0)
    return pat


# ---------------------------------------------------------------- fx chains

def synced_delay(rng, beat):
    """Tempo-synced delay: very short / medium / long, all on the 16th grid.
    Feedback and mix scale down as the delay time grows."""
    tier = rng.choice(["short", "med", "med", "long"])
    if tier == "short":
        t = beat * rng.choice([0.125, 0.25])       # 1/32, 1/16: slapback
        fb, mix = rng.uniform(0.15, 0.35), rng.uniform(0.25, 0.5)
    elif tier == "med":
        t = beat * rng.choice([0.5, 0.75])         # 1/8, dotted 1/8
        fb, mix = rng.uniform(0.25, 0.45), rng.uniform(0.2, 0.4)
    else:
        t = beat * rng.choice([1, 2, 4])           # 1/4, 1/2, one bar
        fb, mix = rng.uniform(0.2, 0.35), rng.uniform(0.15, 0.3)
    # how long the echoes stay audible (feedback^n decays below ~5%)
    ring = t * max(1.0, np.log(0.05) / np.log(max(fb, 0.05)))
    return Delay(delay_seconds=t, feedback=fb, mix=mix), min(ring, 8.0)


def random_texture_board(rng, beat):
    """A randomized effect chain for background layers.
    Returns (board, tail_seconds) so delay echoes are given room to ring."""
    delay_fx, delay_tail = synced_delay(rng, beat)
    pool = [
        Reverb(room_size=rng.uniform(0.4, 0.95), wet_level=rng.uniform(0.2, 0.5),
               dry_level=rng.uniform(0.4, 0.8), width=1.0),
        delay_fx,
        Chorus(rate_hz=rng.uniform(0.3, 2.0), depth=rng.uniform(0.2, 0.7),
               mix=rng.uniform(0.2, 0.5)),
        Phaser(rate_hz=rng.uniform(0.1, 1.5), mix=rng.uniform(0.2, 0.5)),
        Distortion(drive_db=rng.uniform(3, 15)),
        Bitcrush(bit_depth=rng.uniform(6, 12)),
        GSMFullRateCompressor(),                      # gnarly lo-fi codec
        LadderFilter(mode=LadderFilter.Mode.LPF12,
                     cutoff_hz=rng.uniform(400, 4000),
                     resonance=rng.uniform(0.1, 0.6)),
        PitchShift(semitones=rng.choice([-12, -7, -5, 0, 5, 7, 12])),
    ]
    rng.shuffle(pool)
    chain = pool[: rng.randint(1, 3)]
    chain.append(Compressor(threshold_db=-18, ratio=3))
    tail = delay_tail if delay_fx in chain else 2.0
    return Pedalboard(chain), max(2.0, tail)


# ---------------------------------------------------------------- composer

def compose(seed, duration, intro_bars=None, intro_style="ambient", bpm=None,
            pan_drums=0.3, pan_layers=0.6, pan_events=0.5, num_samples=None,
            drum_style="random"):
    rng = random.Random(seed)
    bank = load_samples()
    if not bank:
        sys.exit(f"No samples found in {SAMPLES_DIR}")
    if num_samples is not None and 0 < num_samples < len(bank):
        keep = rng.sample(sorted(bank.keys()), num_samples)
        bank = {n: bank[n] for n in keep}
    names = list(bank.keys())
    used = set()
    print(f"Loaded {len(bank)} samples | seed={seed}")

    if drum_style == "random" or drum_style not in DRUM_STYLES:
        drum_style = rng.choice(DRUM_STYLES)
    bpm = bpm if bpm is not None else rng.randint(84, 128)
    beat = 60.0 / bpm
    bar = 4 * beat
    step_len = beat / 4                               # 16th notes
    n_bars = max(8, round(duration / bar))
    total = int(n_bars * bar * SR) + int(4 * SR)      # room for tails
    buf = np.zeros((total, 2), dtype="float32")
    swing = rng.uniform(0.0, 0.06) * step_len * SR    # subtle 16th swing

    print(f"BPM {bpm} | {n_bars} bars | ~{n_bars * bar:.0f}s | style {drum_style}")

    # ---- drum kit carved from random samples --------------------------
    # each role draws from samples no other role has claimed, so kick,
    # snare, hihat and ride never share a source (unless the bank is tiny)
    kit = {}
    kit_sources = set()
    for role in ("kick", "snare", "hihat", "ride"):
        variants = []
        for _ in range(1 if role == "ride" else rng.randint(1, 3)):
            pool = [n for n in names if n not in kit_sources] or names
            src = rng.choice(pool)
            kit_sources.add(src)
            used.add(src)
            variants.append(carve_drum(rng, bank[src], role))
            print(f"  {role:6s} <- {src}")
        kit[role] = variants

    # ---- song structure ------------------------------------------------
    # split bars into sections; each section gets its own patterns/layers
    sections = []
    b = 0
    while b < n_bars:
        if b == 0 and intro_bars is not None:
            length = min(intro_bars, n_bars)
        else:
            length = min(rng.choice([4, 4, 8, 8, 16]), n_bars - b)
        sections.append((b, length))
        b += length

    gains = {"kick": 0.95, "snare": 0.8, "hihat": 0.45, "ride": 0.3}
    long_names = [n for n in names if len(bank[n]) > SR]  # >1s = texture material

    sections_meta = []
    for si, (start_bar, length) in enumerate(sections):
        sec_start = int(start_bar * bar * SR)
        is_intro = si == 0 and intro_style != "none"
        is_outro = si == len(sections) - 1
        is_break = (not is_intro and not is_outro and rng.random() < 0.25)
        kind = ("intro" if is_intro else "outro" if is_outro
                else "break" if is_break else "groove")
        sections_meta.append({
            "i": si, "kind": kind, "start_bar": start_bar, "bars": length,
            "start_sec": round(start_bar * bar, 3),
            "end_sec": round((start_bar + length) * bar, 3),
        })

        pats = {r: make_pattern(rng, r, drum_style)
                for r in ("kick", "snare", "hihat", "ride")}
        intro_kind = intro_style if is_intro else None
        sec_len = int(length * bar * SR)
        drop_drums = (intro_kind == "ambient" and rng.random() < 0.6) \
            or intro_kind in ("reverse-swell", "collage")
        build_from = {}                       # build intro: bar each role enters

        # ---- drums ----
        if not drop_drums:
            active = ["kick", "snare", "hihat"]
            if intro_kind == "sparse":
                active = [rng.choice(["hihat", "kick"])]
            elif is_break:
                active = rng.sample(active, rng.randint(1, 2))
            if intro_kind == "build":         # roles stack up bar by bar
                build_from = {"hihat": 0, "kick": max(1, length // 3),
                              "snare": max(2, (2 * length) // 3)}
            # ride shows up in some sections only
            if not is_intro and rng.random() < 0.35:
                active.append("ride")
            for bi in range(length):
                bar_start = sec_start + int(bi * bar * SR)
                bar_pats = {r: (mutate(rng, pats[r], 0.1) if rng.random() < 0.3
                                else pats[r]) for r in active}
                # drum fill on the last bar of a section
                if bi == length - 1 and rng.random() < 0.6 and "snare" in active:
                    p = list(bar_pats["snare"])
                    for s in range(12, 16):
                        p[s] = rng.uniform(0.5, 1.0)
                    bar_pats["snare"] = p
                for role in active:
                    if bi < build_from.get(role, 0):
                        continue
                    for s, vel in enumerate(bar_pats[role]):
                        if vel <= 0:
                            continue
                        pos = bar_start + int(s * step_len * SR)
                        if s % 2 == 1:
                            pos += int(swing)
                        pos += rng.randint(-30, 30)   # human jitter (<1ms)
                        hit = rng.choice(kit[role])
                        if role != "kick" and pan_drums > 0:
                            hit = panned(hit, rng.uniform(-pan_drums, pan_drums))
                        place(buf, hit, max(0, pos), vel * gains[role])
                # hihat glitch: a burst of very fast retriggers (1/32-1/96)
                if "hihat" in active and rng.random() < 0.15:
                    s = rng.randrange(16)
                    n_rep = rng.choice([6, 8, 12, 16])
                    spacing = max(1, int(step_len * SR / rng.choice([2, 3, 4, 6])))
                    hit = rng.choice(kit["hihat"])
                    ramp = rng.choice([-1, 1])        # fade out or fade in
                    for i in range(n_rep):
                        t = i / max(1, n_rep - 1)
                        g = gains["hihat"] * (1.0 - 0.7 * (t if ramp < 0 else 1 - t))
                        h = hit
                        if pan_drums > 0:             # glitch sweeps the field
                            h = panned(hit, np.sin(t * np.pi * 2) * 0.7)
                        place(buf, h, bar_start + int(s * step_len * SR)
                              + i * spacing, g)

        # ---- special intro content ----
        if intro_kind == "reverse-swell":
            # a long reversed sample crescendos straight into the first beat
            src = rng.choice(long_names or names)
            used.add(src)
            clip = bank[src][::-1].copy()
            board = Pedalboard([
                Reverb(room_size=0.9, wet_level=0.5, dry_level=0.5, width=1.0),
                Compressor(threshold_db=-18, ratio=3),
            ])
            clip = apply_fx(clip, board, tail=0.5)
            if len(clip) >= sec_len:
                swell = clip[-sec_len:].copy()    # end lands on the drop
            else:
                swell = np.vstack([np.zeros((sec_len - len(clip), 2),
                                            dtype="float32"), clip])
            ramp = (np.linspace(0.15, 1.0, len(swell)) ** 2)[:, None]
            place(buf, swell * ramp, sec_start, 0.55)
            print(f"  sec{si} swell <- {src}")
        elif intro_kind == "collage":
            # scattered one-shots: a field-recording scene, no beat
            for _ in range(rng.randint(6, 12)):
                src = rng.choice(names)
                used.add(src)
                clip = bank[src][: int(rng.uniform(0.5, 3.0) * SR)].copy()
                clip = clip * envelope(len(clip), attack=0.02, decay_curve=1.5)
                board, tail = random_texture_board(rng, beat)
                clip = apply_fx(clip, board, tail=min(tail, 2.0))
                spread = max(pan_events, 0.6)
                clip = panned(clip, rng.uniform(-spread, spread))
                pos = sec_start + rng.randint(0, max(1, sec_len - len(clip)))
                place(buf, clip, pos, rng.uniform(0.3, 0.5))
            print(f"  sec{si} collage intro")

        # ---- background texture layers (loop / reverse / stretch) ----
        if intro_kind in ("drums-first", "collage", "reverse-swell"):
            n_layers = 0
        elif intro_kind == "build":
            n_layers = 1
        elif is_intro or is_break:
            n_layers = 2
        else:
            n_layers = rng.randint(1, 3)
        for _ in range(n_layers):
            src = rng.choice(long_names or names)
            used.add(src)
            clip = bank[src].copy()
            ops = []
            if rng.random() < 0.35:
                clip = clip[::-1].copy(); ops.append("rev")
            if rng.random() < 0.3 and len(clip) > SR:   # stretch a chunk to bars
                n_target_bars = rng.choice([1, 2, 4])
                target = int(n_target_bars * bar * SR)
                chunk = clip[: min(len(clip), target * 2)]
                rate = len(chunk) / target
                st = np.stack([librosa.effects.time_stretch(chunk[:, c], rate=rate)
                               for c in range(2)], axis=1)
                clip = st.astype("float32"); ops.append(f"stretch{n_target_bars}bar")
            board, tail = random_texture_board(rng, beat)
            clip = apply_fx(clip, board, tail=tail)
            loop = crossfade_loop(clip, sec_len)
            fade = min(int(0.5 * SR), len(loop) // 4)
            loop[:fade] *= np.linspace(0, 1, fade)[:, None]
            loop[-fade:] *= np.linspace(1, 0, fade)[:, None]
            lvl = rng.uniform(0.18, 0.4) * (1.4 if (is_intro or is_break) else 1.0)
            if pan_layers > 0:
                loop = panned(loop, rng.uniform(-pan_layers, pan_layers))
            place(buf, loop, sec_start, lvl)
            print(f"  sec{si} bg <- {src} [{','.join(ops) or 'loop'}]")

        # ---- rhythmic chop layer: slice a sample, replay on the grid ----
        if not is_intro and rng.random() < 0.6:
            src = rng.choice(names)
            used.add(src)
            clip = bank[src]
            n_slices = 8
            sl = max(1, len(clip) // n_slices)
            slices = [clip[i * sl:(i + 1) * sl] for i in range(n_slices)]
            board, tail = random_texture_board(rng, beat)
            tail = min(tail, 4 * beat)   # let echoes ring up to one bar
            chop_pat = [rng.randrange(n_slices) if rng.random() < 0.4 else None
                        for _ in range(16)]
            max_slice = int(step_len * SR * rng.choice([1, 2, 2, 4]))
            for bi in range(length):
                bar_start = sec_start + int(bi * bar * SR)
                for s, idx in enumerate(chop_pat):
                    if idx is None:
                        continue
                    piece = slices[idx][:max_slice]
                    if rng.random() < 0.2:
                        piece = piece[::-1]
                    piece = piece * envelope(len(piece), decay_curve=2.0)
                    piece = apply_fx(piece, board, tail=tail)
                    if pan_events > 0:
                        piece = panned(piece, rng.uniform(-pan_events, pan_events))
                    place(buf, piece, bar_start + int(s * step_len * SR), 0.4)
            print(f"  sec{si} chops <- {src}")

        # ---- filtered intro: a lowpass opens bar by bar (club-door) ----
        if intro_kind == "filtered":
            cutoffs = np.geomspace(250, 12000, length)
            for bi in range(length):
                a = sec_start + int(bi * bar * SR)
                b = min(len(buf), a + int(bar * SR))
                pre = max(0, a - 2048)        # context so the filter settles
                lp = Pedalboard(
                    [LowpassFilter(cutoff_frequency_hz=float(cutoffs[bi]))])
                seg = lp(buf[pre:b].T, SR).T
                buf[a:b] = seg[a - pre:]

    # ---- guarantee every sample appears at least once ------------------
    leftovers = [n for n in names if n not in used]
    print(f"Placing {len(leftovers)} unused samples as one-shot events...")
    for src in leftovers:
        clip = bank[src]
        if rng.random() < 0.4:
            clip = clip[::-1].copy()
        clip = clip[: int(min(len(clip), rng.uniform(1, 5) * SR))]
        clip = clip * envelope(len(clip), attack=0.05, decay_curve=1.5)
        board, tail = random_texture_board(rng, beat)
        clip = apply_fx(clip, board, tail=tail)
        pos = int(rng.uniform(0.05, 0.9) * n_bars * bar * SR)
        pos = int(round(pos / (beat * SR)) * beat * SR)   # snap to the beat
        if pan_events > 0:
            clip = panned(clip, rng.uniform(-pan_events, pan_events))
        place(buf, clip, pos, rng.uniform(0.25, 0.45))

    # ---- master bus -----------------------------------------------------
    print("Mastering...")
    master = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        Compressor(threshold_db=-14, ratio=2.5, attack_ms=10, release_ms=150),
        Limiter(threshold_db=-1.0),
    ])
    out = master(buf.T, SR).T
    peak = np.abs(out).max()
    if peak > 0:
        out = out / peak * 0.95
    # global fade in/out
    out = out[: int(n_bars * bar * SR)]      # cut at the last bar, no tail
    # keep edge fades to a click-guard only, so the outro loops seamlessly
    f_in, f_out = int(0.3 * SR), int(0.03 * SR)
    out[:f_in] *= np.linspace(0, 1, f_in)[:, None]
    out[-f_out:] *= np.linspace(1, 0, f_out)[:, None]
    meta = {"seed": seed, "bpm": bpm, "style": drum_style,
            "duration_sec": round(len(out) / SR, 3),
            "sections": sections_meta}
    return out.astype("float32"), bpm, meta


EXAMPLES = """\
examples:
  python3 gacha.py                                  random 3-minute song
  python3 gacha.py --seed 42                        reproduce a run you liked
  python3 gacha.py --bpm 140 --duration 120         fast 2-minute song
  python3 gacha.py --intro-style none               full drums from bar 1
  python3 gacha.py --intro-bars 4 --intro-style sparse   short intro with a beat
  python3 gacha.py --pan-drums 0 --pan-layers 0 --pan-events 0   all centered
  python3 gacha.py --pan-events 1.0                 one-shots hard left/right
"""


def main():
    ap = argparse.ArgumentParser(
        description="Generative sample chopper",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration", type=float, default=180.0, help="seconds")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--bpm", type=int, default=None,
                    help="tempo in beats per minute (default: random 84-128)")
    ap.add_argument("--drum-style", choices=DRUM_STYLES + ["random"],
                    default="random",
                    help="rhythmic skeleton for the drums (default: random)")
    ap.add_argument("--intro-bars", type=int, default=None,
                    help="cap the intro section at N bars (default: random 4/8/16)")
    ap.add_argument("--intro-style",
                    choices=["ambient", "sparse", "none", "build",
                             "drums-first", "reverse-swell", "collage",
                             "filtered"],
                    default="ambient",
                    help="ambient: 60%% chance the intro has no drums (default); "
                         "sparse: one drum voice only; none: full drums from "
                         "bar 1; build: drums stack up bar by bar; "
                         "drums-first: dry kit alone, textures join later; "
                         "reverse-swell: reversed sample crescendos into the "
                         "drop; collage: scattered one-shots, no beat; "
                         "filtered: lowpass opens up over the intro")
    ap.add_argument("--count", type=int, default=1, metavar="N",
                    help="number of songs to render in one go (default 1)")
    ap.add_argument("--num-samples", type=int, default=None, metavar="N",
                    help="randomly pick only N samples from the directory "
                         "(default: use all of them)")
    ap.add_argument("--pan-drums", type=float, default=0.3, metavar="0..1",
                    help="random pan spread per snare/hihat hit, kick stays "
                         "centered (default 0.3, 0 = off)")
    ap.add_argument("--pan-layers", type=float, default=0.6, metavar="0..1",
                    help="random static pan per background layer "
                         "(default 0.6, 0 = off)")
    ap.add_argument("--pan-events", type=float, default=0.5, metavar="0..1",
                    help="random pan per chop slice and one-shot event "
                         "(default 0.5, 0 = off)")
    args = ap.parse_args()

    if len(sys.argv) == 1:
        print(EXAMPLES)
        print("No options given — rendering with defaults.\n")

    for i in range(max(1, args.count)):
        seed = (args.seed + i) if args.seed is not None else random.randrange(10 ** 6)
        song, bpm, meta = compose(seed, args.duration, intro_bars=args.intro_bars,
                            intro_style=args.intro_style, bpm=args.bpm,
                            pan_drums=min(1, abs(args.pan_drums)),
                            pan_layers=min(1, abs(args.pan_layers)),
                            pan_events=min(1, abs(args.pan_events)),
                            num_samples=args.num_samples,
                            drum_style=args.drum_style)

        OUT_DIR.mkdir(exist_ok=True)
        if args.out:
            p = Path(args.out)
            out_path = p if args.count == 1 else p.with_stem(f"{p.stem}_{i+1}")
        else:
            out_path = OUT_DIR / (f"chop_{time.strftime('%Y%m%d_%H%M%S')}"
                                  f"_seed{seed}_bpm{bpm}.wav")
        sf.write(out_path, song, SR, subtype="PCM_16")
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(f"\n✔ wrote {out_path}  ({len(song)/SR:.1f}s, {bpm} BPM, seed {seed})")


if __name__ == "__main__":
    main()
