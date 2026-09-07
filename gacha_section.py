#!/usr/bin/env python3
"""
gacha_section.py — one section at a time, from material in memory, for the
live show. Where gacha_engine.compose() writes a whole song to disk, this
renders a single 8-bar loop out of the takes (or any bank of stereo arrays)
as separate stems the performer switches on stage: drums, layers, chops,
events. Every stem loops seamlessly (tails wrap around to the start), and
the same event timeline the songs carry is kept, so the picture can show
the frames each sound was cut from.

    mat = Material(bank, seed)                    # carves the drum kits once
    sec = render_section(mat, seed=1, bpm=124, bars=8, kind="groove")
    sec.stems["drums"]                            # (n, 2) float32 at 44.1 kHz
    sec.events                                    # [t, src, off, dur, rate, role]
    shot = render_oneshot(mat, seed)              # a single hit/chop for a footswitch

Everything musical is borrowed from the engine: patterns, humanising, the
drum carving, the random effect chains, the texture windows. Measured: a
groove of 8 bars renders in about a second on a 2-take bank once the
material is carved.
"""
import random

import numpy as np

import gacha_engine as E
from gacha_engine import SR

KINDS = ("groove", "break", "build", "sparse")
STEMS = ("drums", "layers", "chops", "events")


class Material:
    """A bank of samples with the drum kits already carved: the slow part,
    done once per set of takes. `bank` is {name: (n, 2) float32 @ 44.1 kHz}."""

    def __init__(self, bank, seed=0, knobs=None):
        # short or silent material is no material (a take of tape hiss is)
        self.bank = {n: c for n, c in bank.items()
                     if len(c) > SR // 4 and float(np.abs(c).max()) > 1e-3}
        self.names = list(self.bank.keys())
        self.long_names = [n for n in self.names if len(self.bank[n]) > SR]
        self.k = (knobs or E.Knobs()).sanitized()
        rng = random.Random(seed)
        self.kit, self.kit_src = {}, {}
        kit_sources, hat_sources = set(), []
        for role in ("kick", "snare", "hihat", "ride"):
            variants, srcs = [], []
            for _ in range(1 if role == "ride" else 2):
                if not self.names:
                    break
                pool = [n for n in self.names if n not in kit_sources] or self.names
                src = rng.choice(pool)
                kit_sources.add(src)
                variants.append(E.velocity_layers(E.carve_drum(rng, self.bank[src], role)))
                srcs.append((src, E.strongest_onset(self.bank[src])))
                if role == "hihat":
                    hat_sources.append(src)
            self.kit[role], self.kit_src[role] = variants, srcs
        self.kit["ohat"] = [E.velocity_layers(E.carve_drum(rng, self.bank[s], "ohat"))
                            for s in hat_sources]
        self.kit_src["ohat"] = [(s, E.strongest_onset(self.bank[s])) for s in hat_sources]

    @property
    def ok(self):
        return bool(self.names)


class Section:
    def __init__(self, kind, bpm, bars, seed, stems, events, notes):
        self.kind, self.bpm, self.bars, self.seed = kind, bpm, bars, seed
        self.stems = stems              # {stem: (n, 2) float32 @ SR}, all n equal
        self.events = events            # [[t, src, off, dur, rate, role]] t in s
        self.notes = notes              # human-readable what-went-in
        self.n = len(next(iter(stems.values())))

    @property
    def seconds(self):
        return self.n / SR


def _place_wrap(buf, clip, start, gain=1.0):
    """Mix clip into a loop buffer at `start`, tails wrapping to the front."""
    n = len(buf)
    start %= n
    i = 0
    while i < len(clip):
        room = n - start
        k = min(room, len(clip) - i)
        buf[start:start + k] += clip[i:i + k] * gain
        i += k
        start = 0


def _loopify(x, n, fade_s=0.05):
    """First n samples of x as a seamless loop: what spills past n is
    crossfaded onto the start, so a texture's last echo meets its first."""
    out = x[:n].copy()
    if len(out) < n:
        out = np.vstack([out, np.zeros((n - len(out), 2), np.float32)])
    fade = min(int(fade_s * SR), len(x) - n, n // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        out[:fade] = out[:fade] * ramp + x[n:n + fade] * (1 - ramp)
    return out


def render_section(mat, seed, bpm, bars=8, kind="groove", pan_drums=0.3,
                   pan_layers=0.6, pan_events=0.5, drum_style="random"):
    """One looping section as stems + events. `kind`: groove (full kit, one
    to three layers, maybe chops), break (one or two drum voices, more
    layers), build (voices stack up bar by bar), sparse (one voice, one
    layer)."""
    k = mat.k
    rng = random.Random(seed)
    if drum_style == "random" or drum_style not in E.DRUM_STYLES:
        drum_style = rng.choice(E.DRUM_STYLES)
    beat = 60.0 / bpm
    bar = 4 * beat
    step_len = beat / 4
    n = int(round(bars * bar * SR))
    stems = {s: np.zeros((n, 2), np.float32) for s in STEMS}
    events, notes = [], []

    def ev(t, src, off, dur, rate, role):
        events.append([round((t % n) / SR, 4), src, round(max(0, off) / SR, 4),
                       round(dur / SR, 4), round(rate, 4), role])

    gains = {"kick": k.gain_kick, "snare": k.gain_snare, "hihat": k.gain_hihat,
             "ohat": k.gain_ohat, "ride": k.gain_ride}
    swing = (k.swing if k.swing is not None else rng.uniform(0.0, 0.06)) * step_len * SR

    # ---- drums ----
    pats = {r: E.make_pattern(rng, r, drum_style) for r in ("kick", "snare", "hihat", "ride")}
    pats["hihat"], pats["ohat"] = E.open_hat_pattern(rng, pats["hihat"], drum_style)
    active = ["kick", "snare", "hihat"]
    build_from = {}
    if kind == "break":
        active = rng.sample(active, rng.randint(1, 2))
    elif kind == "sparse":
        active = [rng.choice(["hihat", "kick"])]
    elif kind == "build":
        build_from = {"hihat": 0, "kick": max(1, bars // 3), "snare": max(2, (2 * bars) // 3)}
    if kind == "groove" and rng.random() < 0.35:
        active.append("ride")
    if "hihat" in active and kind != "sparse":
        active.append("ohat")
        build_from.setdefault("ohat", build_from.get("hihat", 0))
    active = [r for r in active if gains[r] > 0 and mat.kit.get(r)]
    buf = stems["drums"]
    for bi in range(bars):
        bar_start = int(bi * bar * SR)
        bar_pats = {r: (E.mutate(rng, pats[r], k.mutation_amount)
                        if rng.random() < k.mutation_chance else pats[r]) for r in active}
        if bi == bars - 1 and rng.random() < 0.6 and "snare" in active:
            p = list(bar_pats["snare"])
            for s in range(12, 16):
                p[s] = rng.uniform(0.5, 1.0)
            bar_pats["snare"] = p
        ohat_pat = bar_pats.get("ohat", [0.0] * 16)
        closed_pat = bar_pats.get("hihat", [0.0] * 16)
        for role in active:
            if bi < build_from.get(role, 0):
                continue
            for s, vel in enumerate(bar_pats[role]):
                if vel <= 0 or (role == "hihat" and ohat_pat[s] > 0):
                    continue
                pos = bar_start + int(s * step_len * SR)
                if s % 2 == 1:
                    pos += int(swing)
                pos += E.humanize_time(rng, k.humanize)
                vel = E.humanize_vel(rng, vel, s, k.humanize)
                vi = rng.randrange(len(mat.kit[role]))
                hit = E.pick_layer(mat.kit[role][vi], vel)
                ksrc, onset = mat.kit_src[role][vi]
                ev(pos, ksrc, onset, len(mat.kit[role][vi][1]), 1.0, role)
                if role == "ohat":
                    nxt = next((t for t in range(s + 1, 16) if closed_pat[t] > 0), 16)
                    hit = E.choke(hit, int((nxt - s) * step_len * SR))
                if role != "kick" and pan_drums > 0:
                    hit = E.panned(hit, rng.uniform(-pan_drums, pan_drums))
                _place_wrap(buf, hit, max(0, pos), vel * gains[role])
        if "hihat" in active and rng.random() < k.glitch_chance:
            s = rng.randrange(16)
            n_rep = rng.choice([6, 8, 12, 16])
            spacing = max(1, int(step_len * SR / rng.choice([2, 3, 4, 6])))
            vi = rng.randrange(len(mat.kit["hihat"]))
            hit, gsrc = mat.kit["hihat"][vi][1], mat.kit_src["hihat"][vi]
            ramp = rng.choice([-1, 1])
            for i in range(n_rep):
                t = i / max(1, n_rep - 1)
                g = gains["hihat"] * (1.0 - 0.7 * (t if ramp < 0 else 1 - t))
                h = E.panned(hit, np.sin(t * np.pi * 2) * 0.7) if pan_drums > 0 else hit
                gpos = bar_start + int(s * step_len * SR) + i * spacing
                _place_wrap(buf, h, gpos, g)
                ev(gpos, gsrc[0], gsrc[1], spacing, 1.0, "glitch")
    notes.append(f"drums {drum_style}: {', '.join(active) or 'none'}")

    # ---- layers: looped / reversed / stretched textures ----
    if kind in ("break",):
        n_layers = min(max(2, k.layers_min), k.layers_max)
    elif kind in ("build", "sparse"):
        n_layers = 1
    else:
        n_layers = rng.randint(k.layers_min, k.layers_max)
    for _ in range(n_layers if mat.names else 0):
        src = rng.choice(mat.long_names or mat.names)
        clip = E.random_window(rng, mat.bank[src], n, k.random_start == "all").copy()
        w_off, w_len = len(mat.bank[src]) - len(clip), len(clip)
        ops, rev, rate = [], False, 1.0
        if rng.random() < k.reverse_chance:
            clip = clip[::-1].copy(); ops.append("rev"); rev = True
        if rng.random() < k.stretch_chance and len(clip) > SR:
            import librosa
            n_target_bars = rng.choice([1, 2, 4])
            target = int(n_target_bars * bar * SR)
            chunk = clip[: min(len(clip), target * 2)]
            rate = len(chunk) / target
            st = np.stack([librosa.effects.time_stretch(chunk[:, c], rate=rate)
                           for c in range(2)], axis=1)
            clip = st.astype("float32"); ops.append(f"stretch{n_target_bars}bar")
        mat_n = len(clip)
        board, tail = E.random_texture_board(rng, beat, k.fx_min, k.fx_max, k.saturation)
        clip = E.apply_fx(clip, board, tail=tail)
        loop = _loopify(E.crossfade_loop(clip, n + int(0.1 * SR)), n)
        n_fade = min(int(0.05 * SR), len(clip) // 4)
        step_n = (len(clip) - n_fade) or len(clip)
        lp = 0
        while lp < n:
            ev(lp, src, w_off + w_len if rev else w_off, min(mat_n, n - lp),
               -rate if rev else rate, "texture")
            lp += step_n
        lvl = rng.uniform(k.layer_level_min, k.layer_level_max) * (1.4 if kind == "break" else 1.0)
        if pan_layers > 0:
            loop = E.panned(loop, rng.uniform(-pan_layers, pan_layers))
        stems["layers"] += loop * lvl
        notes.append(f"layer <- {src} [{','.join(ops) or 'loop'}]")

    # ---- chops: slice a sample, replay on the grid ----
    if mat.names and (kind == "groove" and rng.random() < k.chop_chance or kind == "break"
                      and rng.random() < 0.5):
        src = rng.choice(mat.names)
        clip = mat.bank[src]
        n_slices = 8
        sl = max(1, len(clip) // n_slices)
        slices = [clip[i * sl:(i + 1) * sl] for i in range(n_slices)]
        board, tail = E.random_texture_board(rng, beat, k.fx_min, k.fx_max, k.saturation)
        tail = min(tail, 4 * beat)
        chop_pat = [rng.randrange(n_slices) if rng.random() < 0.4 else None for _ in range(16)]
        max_slice = int(step_len * SR * rng.choice([1, 2, 2, 4]))
        pieces = {}                                   # slice index -> processed piece
        for bi in range(bars):
            bar_start = int(bi * bar * SR)
            for s, idx in enumerate(chop_pat):
                if idx is None:
                    continue
                if idx not in pieces:
                    piece = slices[idx][:max_slice]
                    p_off, p_rate = idx * sl, 1.0
                    if rng.random() < 0.2:
                        piece = piece[::-1]
                        p_off, p_rate = idx * sl + len(piece), -1.0
                    p_dur = len(piece)
                    piece = piece * E.envelope(len(piece), decay_curve=2.0)
                    piece = E.apply_fx(piece, board, tail=tail)
                    if pan_events > 0:
                        piece = E.panned(piece, rng.uniform(-pan_events, pan_events))
                    pieces[idx] = (piece, p_off, p_dur, p_rate)
                piece, p_off, p_dur, p_rate = pieces[idx]
                cpos = bar_start + int(s * step_len * SR)
                _place_wrap(stems["chops"], piece, cpos, k.chop_gain)
                ev(cpos, src, p_off, p_dur, p_rate, "chop")
        notes.append(f"chops <- {src}")

    # ---- events: a few one-shots scattered on the beat ----
    for _ in range(rng.randint(1, 3) if mat.names else 0):
        shot = render_oneshot(mat, rng.randrange(1 << 30), beat, pan_events)
        pos = int(round(rng.uniform(0.0, bars * 4 - 1)) * beat * SR)
        _place_wrap(stems["events"], shot["clip"], pos, shot["gain"])
        ev(pos, shot["src"], shot["off"], shot["dur"], shot["rate"], "oneshot")
        notes.append(f"event <- {shot['src']}")

    for s in STEMS:                                  # headroom, no clipping
        peak = float(np.abs(stems[s]).max()) if stems[s].size else 0.0
        if peak > 0.98:
            stems[s] *= 0.98 / peak
    events.sort(key=lambda e: e[0])
    return Section(kind, bpm, bars, seed, stems, events, notes)


def render_oneshot(mat, seed, beat=0.5, pan_events=0.5):
    """A single event from the bank: a 1 to 4 s window, maybe reversed,
    through a random chain. {clip, gain, src, off, dur, rate}."""
    rng = random.Random(seed)
    k = mat.k
    src = rng.choice(mat.names)
    clip = mat.bank[src]
    L, rev = len(clip), False
    if rng.random() < 0.4:
        clip = clip[::-1].copy(); rev = True
    cut = int(min(len(clip), rng.uniform(1, 4) * SR))
    clip = E.random_window(rng, clip, cut, k.random_start != "off")
    o_off = L - len(clip)
    clip = clip[:cut]
    clip = clip * E.envelope(len(clip), attack=0.05, decay_curve=1.5)
    board, tail = E.random_texture_board(rng, beat, k.fx_min, k.fx_max, k.saturation)
    clip = E.apply_fx(clip, board, tail=min(tail, 3.0))
    if pan_events > 0:
        clip = E.panned(clip, rng.uniform(-pan_events, pan_events))
    return {"clip": clip.astype(np.float32), "gain": rng.uniform(0.35, 0.55), "src": src,
            "off": (L - o_off) / SR if rev else o_off / SR, "dur": cut / SR,
            "rate": -1.0 if rev else 1.0}
