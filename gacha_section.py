#!/usr/bin/env python3
"""
gacha_section.py — one section at a time, from material in memory, for the
live show. Where gacha_engine.compose() writes a whole song to disk, this
renders a single 8-bar loop out of the takes (or any bank of stereo arrays)
as separate stems the performer switches on stage: layers, chops, events,
and the drums as one stem per voice (gacha_fx.VOICES: kick, snare, closed
and open hat, ride, hat stutter, sub), each through its own rack and mute
on stage, summed into the drums channel there. The sub bass the songs
carry is here too, one note per bar on the first kick, on the root the
layers land on. Every stem loops seamlessly (tails wrap around to the start), and
the same event timeline the songs carry is kept, so the picture can show
the frames each sound was cut from.

    mat = Material(bank, seed)                    # carves the drum kits once
    mat.k = knobs.sanitized()                     # the Generate tab, as of this press
    sec = render_section(mat, seed=1, bpm=124, bars=8, kind="groove")
    sec = render_section(mat, seed=2, bpm=124, kind="intro", intro_style="filtered")
    sec = render_section(mat, seed=3, bpm=124, kind="outro", outro_style="tape_stop")
    sec.stems["kick"]                             # (n, 2) float32 at 44.1 kHz
    sec.events                                    # [t, src, off, dur, rate, role, vel]
    shot = render_oneshot(mat, seed)              # a single hit/chop for a footswitch

Everything musical is borrowed from the engine: patterns, humanising, the
drum carving, the random effect chains, the texture windows, and the songs'
intros and outros as two more kinds: "intro" shaped by the Generate tab's
intro style (ambient, sparse, build, drums-first, filtered, reverse-swell,
collage; none is a groove) and "outro", a groove whose ending effect
(the outro style, random picks one) blends in over the last bars and rings
out inside the loop, so bar 8 ends in silence; the GUI stops the sections
there. Measured: a
groove of 8 bars renders in about a second on a 2-take bank once the
material is carved.
"""
import random

import numpy as np
from scipy.signal import resample_poly

import gacha_engine as E
from gacha_engine import SR
from gacha_fx import VOICES

KINDS = ("groove", "break", "build", "sparse")      # what "random" picks from
EXTRA_KINDS = ("intro", "outro")                      # picked by name only
INTRO_STYLES = ("random", "ambient", "sparse", "none", "build", "drums-first",
                "reverse-swell", "collage", "filtered")
OUTRO_TAIL_MAX_BARS = 4        # the ring-out never takes more than half the loop
STEMS = ("drums", "layers", "chops", "events")      # the live mix's channels (F1-F4)
SECTION_STEMS = ("layers", "chops", "events") + VOICES   # what a section is rendered as
# a shuffle (render_section's `salts`): the rendered stems and the event
# roles each of the four parts owns; the sub goes with the drums (it sits
# on the kicks), the layers (it takes their root) and the chops (its level
# is set against their mix)
SHUFFLE_STEMS = {"drums": VOICES, "layers": ("layers", "sub"),
                 "chops": ("chops", "sub"), "events": ("events",)}
SHUFFLE_ROLES = {"drums": ("kick", "snare", "hihat", "ohat", "ride", "glitch"),
                 "layers": ("texture",), "chops": ("chop",),
                 "events": ("oneshot", "swell", "collage")}
# a voice's sound shuffle (render_section's `kit_salts`): the drum carved
# again from other samples, its pattern kept; the sub's level follows the
# kit's mix, so it goes with each
KIT_PARTS = ("kick", "snare", "hihat", "ohat", "ride", "glitch")
for _v in KIT_PARTS:
    SHUFFLE_STEMS[_v] = (_v, "sub")
    SHUFFLE_ROLES[_v] = (_v,)


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
        self.seed = seed
        self._kits = {}                 # (role, salt) -> (variants, sources), see kit_for
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

    def kit_for(self, role, salt=0):
        """`role`'s carved drum as (variants, sources): the kit's own at salt
        0, else carved again from other samples drawn by the salt, cached.
        The same number of variants, so a section's draws stay in step and
        only the sound changes: a voice's R in the Perform tab, the groove
        kept. "glitch" asks for a hat of its own, the stutter's."""
        base = self.kit.get(role) or []
        base_src = self.kit_src.get(role) or []
        salt = int(salt or 0)
        if salt <= 0:
            return base, base_src
        key = (role, salt)
        got = self._kits.get(key)
        if got is None:
            if not base or not self.names:
                got = ([], [])
            else:
                rng = random.Random(f"{self.seed}:{role}:{salt}")
                current = {src for src, _onset in base_src}
                pool = [n for n in self.names if n not in current] or self.names
                variants, srcs = [], []
                for _ in range(len(base)):
                    src = rng.choice(pool)
                    pool = [n for n in pool if n != src] or pool   # the variants apart, as the kit's
                    variants.append(E.velocity_layers(E.carve_drum(rng, self.bank[src], role)))
                    srcs.append((src, E.strongest_onset(self.bank[src])))
                got = (variants, srcs)
            self._kits[key] = got
        return got


class Section:
    def __init__(self, kind, bpm, bars, seed, stems, events, notes, root=None,
                 drum_style=None, style=None, salts=None, force=(), kit_salts=None):
        self.kind, self.bpm, self.bars, self.seed = kind, bpm, bars, seed
        self.salts = dict(salts or {})  # {stem: int}: each part's stream of chance (a shuffle)
        self.kit_salts = dict(kit_salts or {})   # {voice: int}: the drum carved again (its sound)
        self.force = set(force)         # the parts rendered whatever the kind (shuffled once)
        self.drum_style = drum_style    # the style the drums were patterned in
        self.style = style              # an intro's or an outro's style, else None
        self.stems = stems              # {stem: (n, 2) float32 @ SR}, all n equal (SECTION_STEMS)
        self.events = events            # [[t, src, off, dur, rate, role]] t in s
        self.notes = notes              # human-readable what-went-in
        self.root = root                # note name of the layers ("A#"), for the sub and the tint
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
                   pan_layers=0.6, pan_events=0.5, drum_style="random",
                   intro_style="ambient", outro_style="random", salts=None, force=(),
                   kit_salts=None):
    """One looping section as stems + events. `kind`: groove (full kit, one
    to three layers, maybe chops), break (one or two drum voices, more
    layers), build (voices stack up bar by bar), sparse (one voice, one
    layer), intro (the song intro of `intro_style`, see the module note),
    outro (a groove ending in `outro_style`, silent by the last bar).

    The four parts (STEMS: the drums, the layers, the chops, the events)
    are each drawn from their own stream of chance, seeded by `seed` and
    the part's salt (`salts`, {stem: int}, 0 where not given); the shape
    of the section (an intro's or an outro's style) from the seed alone.
    So the section rendered again with one salt changed has that part
    new and the other three the same: a shuffle. The sub follows the
    drums (it sits on the kicks), the layers (it takes their root) and
    the chops (its level is set against the mix of the three).
    `force`: parts rendered even where the kind would leave them out
    (chops outside a groove, layers or drums in an intro without them),
    so a shuffle of a part always gives one to hear. `kit_salts` ({voice:
    int}, KIT_PARTS) take a drum's sound from another carving of the
    material (Material.kit_for), the pattern and every draw the same: a
    voice's sound shuffled, the groove kept."""
    k = mat.k
    salts = {s: int((salts or {}).get(s, 0)) for s in STEMS}
    kit_salts = {v: int((kit_salts or {}).get(v, 0)) for v in KIT_PARTS}
    kits = {r: mat.kit_for(r, kit_salts[r]) for r in ("kick", "snare", "hihat", "ohat", "ride")}
    kits["glitch"] = mat.kit_for("hihat", kit_salts["glitch"])   # the stutter's own hat
    force = set(force)
    rng = random.Random(seed)                         # the section's shape
    rngs = {s: random.Random(f"{seed}:{s}:{salts[s]}") for s in STEMS}
    rd, rl, rc, rx = (rngs[s] for s in STEMS)        # the drums', the layers', the chops', the events'
    if drum_style == "random" or drum_style not in E.DRUM_STYLES:
        drum_style = rd.choice(E.DRUM_STYLES)
    intro_kind = intro_style if kind == "intro" and intro_style in INTRO_STYLES else None
    if intro_kind == "random":
        intro_kind = rng.choice(E.INTRO_STYLES)
    if intro_kind == "none":
        intro_kind = None                             # a plain groove to start on
    style = None
    if kind == "intro":
        style = intro_kind or "none"
    elif kind == "outro":
        style = rng.choice(E.OUTRO_STYLES) if outro_style == "random" else outro_style
        if style not in E.OUTRO_STYLES:
            style = "none"
    beat = 60.0 / bpm
    bar = 4 * beat
    step_len = beat / 4
    n = int(round(bars * bar * SR))
    stems = {s: np.zeros((n, 2), np.float32) for s in SECTION_STEMS}
    events, notes = [], []

    def drums_mix():
        return sum(stems[v] for v in VOICES)

    def ev(t, src, off, dur, rate, role, vel=1.0):
        # the seventh field, the hit's velocity 0..1, is what the MIDI out sends
        events.append([round((t % n) / SR, 4), src, round(max(0, off) / SR, 4),
                       round(dur / SR, 4), round(rate, 4), role, round(vel, 3)])

    gains = {"kick": k.gain_kick, "snare": k.gain_snare, "hihat": k.gain_hihat,
             "ohat": k.gain_ohat, "ride": k.gain_ride}
    swing = (k.swing if k.swing is not None else rd.uniform(0.0, 0.06)) * step_len * SR

    # ---- drums ----
    pats = {r: E.make_pattern(rd, r, drum_style) for r in ("kick", "snare", "hihat", "ride")}
    pats["hihat"], pats["ohat"] = E.open_hat_pattern(rd, pats["hihat"], drum_style)
    active = ["kick", "snare", "hihat"]
    build_from = {}
    # an intro the songs drop the drums from: ambient now and then, the
    # swell and the collage always
    drop_drums = (intro_kind == "ambient" and rd.random() < 0.6) \
        or intro_kind in ("reverse-swell", "collage")
    if drop_drums:
        active = []
    elif kind == "break":
        active = rd.sample(active, rd.randint(1, 2))
    elif kind == "sparse" or intro_kind == "sparse":
        active = [rd.choice(["hihat", "kick"])]
    elif kind == "build" or intro_kind == "build":
        build_from = {"hihat": 0, "kick": max(1, bars // 3), "snare": max(2, (2 * bars) // 3)}
    if not active and "drums" in force:               # shuffled into an intro without them
        active = ["kick", "snare", "hihat"]
    if kind in ("groove", "outro") and rd.random() < 0.35:
        active.append("ride")
    if "hihat" in active and kind != "sparse" and intro_kind != "sparse":
        active.append("ohat")
        build_from.setdefault("ohat", build_from.get("hihat", 0))
    active = [r for r in active if gains[r] > 0 and kits[r][0]]
    kick_hits = []                                    # (pos, bar, step) for the sub
    for bi in range(bars):
        bar_start = int(bi * bar * SR)
        bar_pats = {r: (E.mutate(rd, pats[r], k.mutation_amount)
                        if rd.random() < k.mutation_chance else pats[r]) for r in active}
        if bi == bars - 1 and rd.random() < 0.6 and "snare" in active:
            p = list(bar_pats["snare"])
            for s in range(12, 16):
                p[s] = rd.uniform(0.5, 1.0)
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
                pos += E.humanize_time(rd, k.humanize)
                vel = E.humanize_vel(rd, vel, s, k.humanize)
                if role == "kick":
                    kick_hits.append((max(0, pos), bi, s))
                kv, ksr = kits[role]
                vi = rd.randrange(len(kv))
                hit = E.pick_layer(kv[vi], vel)
                ksrc, onset = ksr[vi]
                ev(pos, ksrc, onset, len(kv[vi][1]), 1.0, role, vel)
                if role == "ohat":
                    nxt = next((t for t in range(s + 1, 16) if closed_pat[t] > 0), 16)
                    hit = E.choke(hit, int((nxt - s) * step_len * SR))
                if role != "kick" and pan_drums > 0:
                    hit = E.panned(hit, rd.uniform(-pan_drums, pan_drums))
                _place_wrap(stems[role], hit, max(0, pos), vel * gains[role])
        if "hihat" in active and rd.random() < k.glitch_chance:
            s = rd.randrange(16)
            n_rep = rd.choice([6, 8, 12, 16])
            spacing = max(1, int(step_len * SR / rd.choice([2, 3, 4, 6])))
            gv, gsr = kits["glitch"]
            vi = rd.randrange(len(gv))
            hit, gsrc = gv[vi][1], gsr[vi]
            ramp = rd.choice([-1, 1])
            for i in range(n_rep):
                t = i / max(1, n_rep - 1)
                gv = 1.0 - 0.7 * (t if ramp < 0 else 1 - t)
                g = gains["hihat"] * gv
                h = E.panned(hit, np.sin(t * np.pi * 2) * 0.7) if pan_drums > 0 else hit
                gpos = bar_start + int(s * step_len * SR) + i * spacing
                _place_wrap(stems["glitch"], h, gpos, g)
                ev(gpos, gsrc[0], gsrc[1], spacing, 1.0, "glitch", gv)
    notes.append(f"drums {drum_style}: {', '.join(active) or 'none'}")

    # ---- special intro content, as in the songs ----
    if intro_kind == "reverse-swell" and mat.names:
        # a long reversed sample crescendos straight into the loop's end:
        # the drop is the next section
        src = rx.choice(mat.long_names or mat.names)
        clip = mat.bank[src][::-1].copy()
        board = E.Pedalboard([
            E.Reverb(room_size=0.9, wet_level=0.5, dry_level=0.5, width=1.0),
            E.Compressor(threshold_db=-18, ratio=3),
        ])
        clip = E.apply_fx(clip, board, tail=0.5)
        L = len(mat.bank[src])
        j0 = len(clip) - n                            # first clip sample in the swell
        if j0 >= 0:                                   # reversed: index j <- source L-1-j
            ev(0, src, L - j0, min(n, L - j0), -1.0, "swell")
        else:
            ev(-j0, src, L, L, -1.0, "swell")
        if len(clip) >= n:
            swell = clip[-n:].copy()                  # the end lands on the loop's end
        else:
            swell = np.vstack([np.zeros((n - len(clip), 2), np.float32), clip])
        ramp = (np.linspace(0.15, 1.0, len(swell)) ** 2)[:, None]
        stems["events"] += swell * ramp * 0.55
        notes.append(f"swell <- {src}")
    elif intro_kind == "collage" and mat.names:
        # scattered one-shots: a field-recording scene, no beat
        for _ in range(rx.randint(6, 12)):
            src = rx.choice(mat.names)
            cut = int(rx.uniform(0.5, 3.0) * SR)
            clip = E.random_window(rx, mat.bank[src], cut, k.random_start == "all")
            c_off = len(mat.bank[src]) - len(clip)
            clip = clip[:cut].copy()
            c_dur = len(clip)
            clip = clip * E.envelope(len(clip), attack=0.02, decay_curve=1.5)
            board, tail = E.random_texture_board(rx, beat, k.fx_min, k.fx_max, k.saturation)
            clip = E.apply_fx(clip, board, tail=min(tail, 2.0))
            spread = max(pan_events, 0.6)
            clip = E.panned(clip, rx.uniform(-spread, spread))
            pos = rx.randint(0, max(1, n - len(clip)))
            _place_wrap(stems["events"], clip, pos, rx.uniform(0.3, 0.5))
            ev(pos, src, c_off, c_dur, 1.0, "collage")
        notes.append("collage intro")

    # ---- layers: looped / reversed / stretched textures ----
    if intro_kind in ("drums-first", "collage", "reverse-swell"):
        n_layers = 0
    elif intro_kind == "build":
        n_layers = min(1, k.layers_max)
    elif kind == "break" or intro_kind is not None:
        n_layers = min(max(2, k.layers_min), k.layers_max)
    elif kind in ("build", "sparse"):
        n_layers = 1
    else:
        n_layers = rl.randint(k.layers_min, k.layers_max)
    if "layers" in force:                             # shuffled into a section without them
        n_layers = max(1, n_layers)
    for _ in range(n_layers if mat.names else 0):
        src = rl.choice(mat.long_names or mat.names)
        clip = E.random_window(rl, mat.bank[src], n, k.random_start == "all").copy()
        w_off, w_len = len(mat.bank[src]) - len(clip), len(clip)
        ops, rev, rate = [], False, 1.0
        if rl.random() < k.reverse_chance:
            clip = clip[::-1].copy(); ops.append("rev"); rev = True
        if rl.random() < k.stretch_chance and len(clip) > SR:
            import librosa
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")   # librosa 1.0's own deprecations, inside phase_vocoder
            n_target_bars = rl.choice([1, 2, 4])
            target = int(n_target_bars * bar * SR)
            chunk = clip[: min(len(clip), target * 2)]
            rate = len(chunk) / target
            st = np.stack([librosa.effects.time_stretch(chunk[:, c], rate=rate)
                           for c in range(2)], axis=1)
            clip = st.astype("float32"); ops.append(f"stretch{n_target_bars}bar")
        mat_n = len(clip)
        board, tail = E.random_texture_board(rl, beat, k.fx_min, k.fx_max, k.saturation)
        clip = E.apply_fx(clip, board, tail=tail)
        loop = _loopify(E.crossfade_loop(clip, n + int(0.1 * SR)), n)
        n_fade = min(int(0.05 * SR), len(clip) // 4)
        step_n = (len(clip) - n_fade) or len(clip)
        lp = 0
        while lp < n:
            ev(lp, src, w_off + w_len if rev else w_off, min(mat_n, n - lp),
               -rate if rev else rate, "texture")
            lp += step_n
        lvl = rl.uniform(k.layer_level_min, k.layer_level_max) * (1.4 if kind == "break" else 1.0)
        if pan_layers > 0:
            loop = E.panned(loop, rl.uniform(-pan_layers, pan_layers))
        stems["layers"] += loop * lvl
        notes.append(f"layer <- {src} [{','.join(ops) or 'loop'}]")

    # ---- chops: slice a sample, replay on the grid (never in an intro) ----
    chop_roll = rc.random()                           # drawn either way: the stream stays in step
    chops_in = (kind in ("groove", "outro") and chop_roll < k.chop_chance
                or kind == "break" and chop_roll < 0.5)
    if mat.names and (chops_in or "chops" in force):
        src = rc.choice(mat.names)
        clip = mat.bank[src]
        n_slices = 8
        sl = max(1, len(clip) // n_slices)
        slices = [clip[i * sl:(i + 1) * sl] for i in range(n_slices)]
        board, tail = E.random_texture_board(rc, beat, k.fx_min, k.fx_max, k.saturation)
        tail = min(tail, 4 * beat)
        chop_pat = [rc.randrange(n_slices) if rc.random() < 0.4 else None for _ in range(16)]
        max_slice = int(step_len * SR * rc.choice([1, 2, 2, 4]))
        pieces = {}                                   # slice index -> processed piece
        for bi in range(bars):
            bar_start = int(bi * bar * SR)
            for s, idx in enumerate(chop_pat):
                if idx is None:
                    continue
                if idx not in pieces:
                    piece = slices[idx][:max_slice]
                    p_off, p_rate = idx * sl, 1.0
                    if rc.random() < 0.2:
                        piece = piece[::-1]
                        p_off, p_rate = idx * sl + len(piece), -1.0
                    p_dur = len(piece)
                    piece = piece * E.envelope(len(piece), decay_curve=2.0)
                    piece = E.apply_fx(piece, board, tail=tail)
                    if pan_events > 0:
                        piece = E.panned(piece, rc.uniform(-pan_events, pan_events))
                    pieces[idx] = (piece, p_off, p_dur, p_rate)
                piece, p_off, p_dur, p_rate = pieces[idx]
                cpos = bar_start + int(s * step_len * SR)
                _place_wrap(stems["chops"], piece, cpos, k.chop_gain)
                ev(cpos, src, p_off, p_dur, p_rate, "chop")
        notes.append(f"chops <- {src}")

    # ---- events: a few one-shots scattered on the beat ----
    for _ in range(rx.randint(1, 3) if mat.names else 0):
        shot = render_oneshot(mat, rx.randrange(1 << 30), beat, pan_events)
        pos = int(round(rx.uniform(0.0, bars * 4 - 1)) * beat * SR)
        _place_wrap(stems["events"], shot["clip"], pos, shot["gain"])
        ev(pos, shot["src"], shot["off"], shot["dur"], shot["rate"], "oneshot")
        notes.append(f"event <- {shot['src']}")

    # ---- root of the section (from the layers), then the sub bass: one
    #      note per bar on its first kick, as in a song. Its own rng, so
    #      toggling the sub leaves the rest of the section unchanged ----
    root_rng = random.Random(seed ^ 0x5B)
    pc = section_root(stems["layers"])
    if pc is None:
        pc = root_rng.randrange(12)
    root = E.NOTE_NAMES[pc]
    if k.sub_level > 0 and kick_hits:
        first_kick = {}                              # bar -> (pos, step)
        for pos, bar_i, s in kick_hits:
            if bar_i not in first_kick or s < first_kick[bar_i][1]:
                first_kick[bar_i] = (pos, s)
        starts = []
        for bar_i in sorted(first_kick):
            if root_rng.random() < k.sub_chance:     # weighted occurrence
                midi = E.sub_midi(pc)
                if root_rng.random() < 0.2:          # some bars sit on the fifth
                    midi = midi + 7 if midi + 7 <= E.SUB_HI else midi - 5
                starts.append((first_kick[bar_i][0], midi))
        hits = []
        note_len = int((beat + k.sub_release) * SR)  # one-beat hold + release
        for i, (pos, midi) in enumerate(starts):
            # the next note (the first one again, a loop later) cuts this one
            nxt = starts[i + 1][0] if i + 1 < len(starts) else starts[0][0] + n
            length = min(note_len, nxt - pos - int(0.01 * SR))
            if length > int(0.05 * SR):
                hits.append((pos, midi, length))
        if hits:
            # rendered a note longer than the loop, the tail folds onto the start
            sub = E.render_sub(hits, n + note_len, k.sub_phaser, k.sub_pan, k.sub_release)
            sub[:note_len] += sub[n:n + note_len]
            sub = sub[:n]
            active = np.abs(sub).sum(axis=1) > 1e-6
            if active.any():
                drums = drums_mix()
                mix = drums + stems["layers"] + stems["chops"]
                ref = E.rms(mix) or E.rms(drums)
                sub *= k.sub_level * ref / E.rms(sub[active])
                stems["sub"] += sub
                notes.append(f"sub {root}: {len(hits)} notes over {len(first_kick)} bars")

    # ---- filtered intro: a lowpass opens bar by bar (the club door) ----
    if intro_kind == "filtered":
        cutoffs = np.geomspace(250, 12000, bars)
        for bi in range(bars):
            a = int(bi * bar * SR)
            b = min(n, a + int(bar * SR))
            pre = max(0, a - 2048)                    # context so the filter settles
            lp = E.Pedalboard([E.LowpassFilter(cutoff_frequency_hz=float(cutoffs[bi]))])
            for s in SECTION_STEMS:
                if not np.any(stems[s][pre:b]):
                    continue
                seg = lp(stems[s][pre:b].T, SR).T
                stems[s][a:b] = seg[a - pre:]
        notes.append("filtered intro")
    if intro_kind is not None:
        notes.append(f"intro {intro_kind}")

    # ---- outro: the ending effect blends in over the last bars of a
    #      shorter body and rings out to silence inside the loop ----
    if kind == "outro" and style != "none":
        tail_s = k.outro_tail if k.outro_tail is not None else E.OUTRO_TAIL[style]
        tail_s = min(tail_s, OUTRO_TAIL_MAX_BARS * bar)
        tail_n = int(tail_s * SR)
        body_n = max(int(bar * SR), n - tail_n)
        tail_n = n - body_n
        onset_n = min(body_n, int(k.outro_bars * bar * SR))
        for s in SECTION_STEMS:
            if not np.any(stems[s]):
                continue
            out = E.apply_outro(stems[s][:body_n], style, beat, tail_sec=tail_n / SR,
                                onset_n=onset_n, wet=k.outro_wet, amount=k.outro_amount)
            if len(out) < n:
                out = np.vstack([out, np.zeros((n - len(out), 2), np.float32)])
            stems[s] = np.ascontiguousarray(out[:n], dtype=np.float32)
        events = [e for e in events if e[0] * SR < body_n]
        notes.append(f"outro {style}: {body_n / SR / bar:.1f} bars then {tail_n / SR:.1f} s out")

    kit = [v for v in VOICES if v != "sub"]
    for s in SECTION_STEMS:                          # headroom, no clipping
        if s in kit:
            continue
        peak = float(np.abs(stems[s]).max()) if stems[s].size else 0.0
        if peak > 0.98:
            stems[s] *= 0.98 / peak
    peak = float(np.abs(sum(stems[v] for v in kit)).max()) if n else 0.0   # the kit, one scale
    if peak > 0.98:                                  # (the sub apart: it follows the mix)
        for v in kit:
            stems[v] *= 0.98 / peak
    events.sort(key=lambda e: e[0])
    return Section(kind, bpm, bars, seed, stems, events, notes, root=root,
                   drum_style=drum_style, style=style, salts=salts, force=force,
                   kit_salts=kit_salts)


SECTION_ROOT_MAX_S = 8.0     # the chroma looks at this much of the layers at most


def section_root(layers):
    """Dominant pitch class of the layers stem (E.section_root, on no more
    than SECTION_ROOT_MAX_S seconds so a render stays quick), or None when
    the layers are silent."""
    return E.section_root(layers[: int(SECTION_ROOT_MAX_S * SR)])


def cut_burst(audio, sr, off_s, len_s, out_sr=None, fade_s=0.003):
    """`len_s` seconds of `audio` ((n, 2) float32 at `sr`) from `off_s`,
    short fades on both ends so it never clicks, as a (2, n) float32 block
    at `out_sr` (default `sr`). The tape at its own level: no gain, no fx."""
    n = max(2, int(round(len_s * sr)))
    i0 = int(min(max(0.0, off_s) * sr, max(0, len(audio) - n)))
    clip = np.array(audio[i0:i0 + n], dtype=np.float32, copy=True)
    f = int(min(fade_s * sr, len(clip) // 2))
    if f > 0:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)[:, None]
        clip[:f] *= ramp
        clip[-f:] *= ramp[::-1]
    if out_sr and out_sr != sr:
        from math import gcd
        g = gcd(int(out_sr), int(sr))
        clip = resample_poly(clip, int(out_sr) // g, int(sr) // g, axis=0).astype(np.float32)
    return np.ascontiguousarray(clip.T)


def render_swell(mat, seed, bpm, bars=1, gain=0.36):
    """The song's reverse cymbal swell for a live break: a bar long at
    `bpm`, from the kit's ride when there is one, else from half a second
    of a random bank clip; (n, 2) float32 at SR, or None with no material.
    `gain` sits under the stems (0.45 was too loud as the sweep opens up)."""
    rng = random.Random(seed)
    length = int(bars * 4 * 60.0 / bpm * SR)
    if mat.kit.get("ride"):
        hit = E.pick_layer(rng.choice(mat.kit["ride"]), 1.0)
    elif mat.names:
        clip = mat.bank[rng.choice(mat.names)]
        hit = E.random_window(rng, clip, min(len(clip), SR // 2), True)
    else:
        return None
    return (E.cymbal_swell(rng, hit, length) * gain).astype(np.float32)


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
