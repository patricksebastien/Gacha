#!/usr/bin/env python3
"""
gacha_frag.py — Fragments: the bar sliced, each slice played as grains.

Bitwig's Sampler in Sliced mode with the Fragments play mode, over the
section that loops: the bar playing is cut into `div` equal slices (8 or
16, the sampler's "Slice at: Divisions"), and a slice held (a MIDI note
per slice, or the key on the slice under the playhead) sounds as a cloud
of grains read from it, in place of the section (the transport runs on
underneath, ducked by the voice's envelope, and comes back as the release
fades), through the same stems, so the racks, the voice switches and
F1-F4 apply to it as to the section.

The voice: a playhead runs through the slice at `speed` (100 % = the
section's own, 0 = frozen on the spot, negative = backwards), looping
round the slice, bouncing at its ends or stopping there (`play`); grains
are spawned from it every `rate` (a beat unit, on the bar's grid, for the
hypnotic rhythms, or "dense": eight of them overlapping whatever the
size), each `size` long (a beat unit of the bar, a 16th by default, so a
tempo change keeps them in time), from the playhead moved by `motion` (a
random distance up to that many grain sizes either way, and the stereo
place with it), at `pitch` (semitones, the grain's own read rate, the
playhead's speed untouched), forwards, backwards, at random or in turn
(`dir`), each grain triggered `repeats` times over (the same spot, the
same place, the roll's density), under a grain envelope morphed by
`shape` from a pluck (a 2 % attack, the rest a fall) through a Hann to
an open window (10 % ramps). The voice's own envelope is an ADSR, gated:
attack on the press, decay to the sustain, release on the let-go, in
milliseconds. Up to MAX_VOICES slices at once, the newest stealing the
oldest.

What keeps a voice smooth when the playhead's speed and the grains' pitch
disagree (half speed at pitch, frozen, backwards): a new grain of the
same pitch and direction as the last is lined up with it by correlation
(WSOLA, the spot searched +-5 ms for the best match against where the
last grain reads now, over the stems summed), so overlapping grains add
in phase instead of combing at the grain rate; the motion's random
spread still lands each grain on a matching phase of the wave.

Everything is one dict, FRAGMENTS, set from the GUI thread (the Perform
tab's Fragments rows) and read per block; presses and releases go through
a deque. The player calls `block()` after its own render, with the stems,
and `state()` tells the picture and the status what sounds.

    fr = Fragments(48000)
    fr.press(3)                       # slice 3 of the bar playing; 0 = the slice under the playhead
    fr.block(stems, n, bar_f, t0, rate, locked, out, bs)   # audio thread, after the render
    fr.release(3)
"""
from collections import deque

import numpy as np

# a bar's fractions: Bitwig's beat units, a 16th the default grain
FRAG_UNITS = [("1/64", 1 / 64), ("1/32", 1 / 32), ("1/16", 1 / 16), ("1/8", 1 / 8),
              ("1/4", 1 / 4), ("1/2", 1 / 2), ("1 bar", 1.0)]
FRAG_RATES = [("dense", 0.0)] + FRAG_UNITS     # dense: DENSITY grains overlapping
FRAG_DIVS = [4, 8, 16, 32]
FRAG_DIRS = ["forward", "reverse", "random", "alternate"]
FRAG_PLAYS = ["loop", "ping-pong", "once"]
MAX_VOICES = 4
GRAIN_MAX = 48                                # grains per voice at once, at the most
ALIGN_SEARCH, ALIGN_TPL = 256, 256            # the alignment: +-5 ms searched, a 5 ms template
DENSITY = 8.0                                 # "dense": grains overlapping whatever the size
FRAGMENTS = {"div": 16,          # slices per bar
             "size": 1 / 16,     # a grain, as a fraction of the bar
             "rate": 0.0,        # a grain every this fraction of the bar; 0 = dense
             "motion": 1.0,      # the random spread round the playhead, in grain sizes (0..8)
             "pitch": 0.0,       # semitones
             "speed": 1.0,       # the playhead's speed, 1 = the section's; 0 frozen; < 0 backwards
             "repeats": 1,       # each grain that many times over
             "shape": 0.5,       # the grain envelope: 0 pluck, 0.5 Hann, 1 open
             "dir": "forward",   # FRAG_DIRS
             "play": "loop",     # FRAG_PLAYS: what the playhead does at the slice's ends
             "attack": 0.005, "decay": 0.2, "sustain": 1.0, "release": 0.25}   # seconds, 0..1


def _window_edges(shape):
    """(attack share, release share) of a grain's window from the shape."""
    s = max(0.0, min(1.0, float(shape)))
    if s < 0.5:
        a = 0.02 + 0.48 * (s / 0.5)
        return a, 1.0 - a
    e = 0.5 - 0.4 * ((s - 0.5) / 0.5)
    return e, e


class Voice:
    """One slice held: its geometry, playhead, envelope and grains."""

    def __init__(self, key, k, s0, L, sr):
        self.key = key                        # what was pressed (0 = follow), for the release
        self.k = k                            # the slice, 1-based
        self.s0, self.L = int(s0), int(L)     # loop frames: the slice's start and length
        self.sr = sr
        self.p = 0.0                          # the playhead, frames into the slice
        self.dirn = 1.0                       # ping-pong: which way
        self.stage = "A"                      # A, D, S, R, or "" when done
        self.lvl = 0.0                        # the envelope now
        self.gate = 0.0                       # the duck: rises with A, falls with R
        self.rel_from = 1.0                   # the level the release started from
        self.released = False
        self._next = 0.0                      # output frames to the next dense spawn
        self.rep_left = 0                     # repeats left of the last grain's spot
        self.rep_spot = (0.0, 0.0)            # (position in the slice, pan) repeated
        self.alt = False                      # "alternate": the next grain backwards
        self.t = 0                            # output frames since the press
        self.prev = None                      # (spot, t, ratio, rev) of the last grain, aligned
        self.spawned = 0
        self.clear_grains()

    def clear_grains(self):
        self.g_pos = np.zeros(0, np.float64)  # the grain's start, frames into the slice
        self.g_len = np.zeros(0, np.int64)    # output frames
        self.g_ratio = np.zeros(0, np.float64)
        self.g_rev = np.zeros(0, bool)
        self.g_age = np.zeros(0, np.int64)
        self.g_a = np.zeros(0, np.float64)    # the window's attack share
        self.g_r = np.zeros(0, np.float64)    # and release share
        self.g_gain = np.zeros((2, 0), np.float32)

    @property
    def done(self):
        return self.stage == ""


class Fragments:
    def __init__(self, sr):
        self.sr = int(sr)
        self.voices = []
        self._req = deque()                   # ("on", key) / ("off", key) / ("clear",)
        self.rng = np.random.default_rng()
        self._duck_end = 0.0                  # the duck at the last block's end
        self._mono = None                     # the stems summed, left, for the alignment
        self._mono_key = None

    # ---- GUI thread
    def press(self, key):
        """Slice `key` (1..div) held; 0: the slice under the playhead now."""
        self._req.append(("on", int(key)))

    def release(self, key):
        self._req.append(("off", int(key)))

    def clear(self):
        """Every voice off at once (a stop, a new section)."""
        self._req.append(("clear",))

    @property
    def sounding(self):
        return bool(self.voices)

    def state(self):
        """What sounds, for the picture and the status: the newest voice's
        slice and division, its playhead (0..1 into the slice), its level,
        the grains as (start 0..1 into the slice, width as a share of the
        slice, phase 0..1 through the window, backwards), or None."""
        if not self.voices:
            return None
        v = self.voices[-1]
        L = max(1, v.L)
        heads = []
        if len(v.g_len):
            order = np.argsort(v.g_age)[:16]
            for i in order:
                n, r, rev = int(v.g_len[i]), float(v.g_ratio[i]), bool(v.g_rev[i])
                heads.append((float(v.g_pos[i] / L) % 1.0, min(1.0, n * r / L),
                              float(min(1.0, max(0.0, v.g_age[i] / n))), rev))
        return {"slice": v.k, "div": int(FRAGMENTS.get("div", 16)), "playhead": v.p / L,
                "level": float(v.lvl), "heads": heads, "speed": float(FRAGMENTS.get("speed", 1.0)),
                "voices": len(self.voices)}

    # ---- audio thread
    def _settings(self):
        f = FRAGMENTS
        return (max(1, int(f.get("div", 16))), max(1e-4, float(f.get("size", 1 / 16))),
                max(0.0, float(f.get("rate", 0.0))), max(0.0, min(8.0, float(f.get("motion", 1.0)))),
                max(-48.0, min(48.0, float(f.get("pitch", 0.0)))),
                max(-4.0, min(4.0, float(f.get("speed", 1.0)))),
                max(1, min(16, int(f.get("repeats", 1)))), float(f.get("shape", 0.5)),
                str(f.get("dir", "forward")), str(f.get("play", "loop")),
                max(0.0, float(f.get("attack", 0.005))), max(0.0, float(f.get("decay", 0.2))),
                max(0.0, min(1.0, float(f.get("sustain", 1.0)))),
                max(0.0, float(f.get("release", 0.25))))

    def _requests(self, t0, bar_f, n, div):
        while self._req:
            req = self._req.popleft()
            if req[0] == "clear":
                self.voices = []
            elif req[0] == "off":
                for v in self.voices:
                    if v.key == req[1] and not v.released:
                        v.released = True
                        v.stage, v.rel_from = "R", v.lvl
            elif bar_f and n:
                key = req[1]
                L = max(1, bar_f // div)
                bar0 = (int(t0) % n) // bar_f * bar_f
                k = key if key >= 1 else (int(t0) % n % bar_f) // L + 1
                k = max(1, min(div, k))
                v = Voice(key, k, bar0 + (k - 1) * L, L, self.sr)
                self.voices.append(v)
                while len(self.voices) > MAX_VOICES:
                    self.voices.pop(0)

    def _env(self, v, bs, A, D, S, R):
        """The voice's envelope and its gate (sustain 1, no decay) over
        the block, each a (bs,) ramp, the stage machine advanced."""
        sr = self.sr
        env = np.empty(bs, np.float32)
        gate = np.empty(bs, np.float32)
        i = 0
        while i < bs:
            left = bs - i
            if v.stage == "A":
                step = 1.0 / max(1.0, A * sr)
                take = min(left, int(np.ceil((1.0 - v.lvl) / step)) if step > 0 else left)
                take = max(1, take)
                seg = v.lvl + step * np.arange(1, take + 1)
                env[i:i + take] = np.minimum(seg, 1.0)
                gate[i:i + take] = np.minimum(v.gate + step * np.arange(1, take + 1), 1.0)
                v.lvl, v.gate = float(env[i + take - 1]), float(gate[i + take - 1])
                if v.lvl >= 1.0:
                    v.stage = "D"
            elif v.stage == "D":
                step = (1.0 - S) / max(1.0, D * sr)
                take = min(left, int(np.ceil((v.lvl - S) / step)) if step > 0 else left)
                take = max(1, take)
                seg = v.lvl - step * np.arange(1, take + 1)
                env[i:i + take] = np.maximum(seg, S)
                gate[i:i + take] = 1.0
                v.lvl, v.gate = float(env[i + take - 1]), 1.0
                if v.lvl <= S + 1e-9:
                    v.stage = "S"
            elif v.stage == "S":
                env[i:], gate[i:] = S, 1.0
                v.lvl, v.gate = S, 1.0
                take = left
            elif v.stage == "R":
                step = max(v.rel_from, 1e-6) / max(1.0, R * sr)
                take = min(left, int(np.ceil(v.lvl / step)) if step > 0 else left)
                take = max(1, take)
                env[i:i + take] = np.maximum(v.lvl - step * np.arange(1, take + 1), 0.0)
                gstep = 1.0 / max(1.0, R * sr)
                gate[i:i + take] = np.maximum(v.gate - gstep * np.arange(1, take + 1), 0.0)
                v.lvl, v.gate = float(env[i + take - 1]), float(gate[i + take - 1])
                if v.lvl <= 0.0 and v.gate <= 0.0:
                    v.stage = ""
            else:
                env[i:], gate[i:] = 0.0, 0.0
                take = left
            i += take
        return env, gate

    def _align(self, v, spot, ratio, rev, t):
        """The spot moved within +-ALIGN_SEARCH frames (round the slice)
        to where the slice best matches the last grain's read head now,
        the same pitch and direction only."""
        pv = v.prev
        mono = self._mono
        if pv is None or mono is None or pv[2] != ratio or pv[3] != rev:
            return spot
        ps, pt, _r, _d = pv
        L = v.L
        S, W = min(ALIGN_SEARCH, L // 4), min(ALIGN_TPL, L // 4)
        if S < 8 or W < 8:
            return spot
        natural = ps + (-1.0 if rev else 1.0) * (t - pt) * ratio    # where it reads now
        tpl = mono[v.s0 + ((int(natural) + np.arange(W)) % L)]
        if not tpl.any():
            return spot
        lo = int(spot) - S
        span = mono[v.s0 + ((lo + np.arange(2 * S + W)) % L)]
        c = np.correlate(span, tpl, "valid")                        # 2 S + 1 offsets
        return float((lo + int(np.argmax(c)) + (spot - int(spot))) % L)

    def _spawn(self, v, j, pos, size_f, n_out, ratio, motion, dirn, repeats, shape):
        """One grain `j` output frames into the block from the playhead at
        `pos` (frames into the slice)."""
        rng = self.rng
        if dirn == "reverse":
            rev = True
        elif dirn == "random":
            rev = bool(rng.random() < 0.5)
        elif dirn == "alternate":
            rev, v.alt = v.alt, not v.alt
        else:
            rev = False
        t = v.t + int(j)
        if v.rep_left > 0:
            spot, pan = v.rep_spot
            v.rep_left -= 1
        else:
            spot = (pos + (rng.random() * 2.0 - 1.0) * motion * size_f) % v.L
            spot = self._align(v, spot, ratio, rev, t)
            pan = (rng.random() * 2.0 - 1.0) * min(1.0, motion) * 0.8
            v.rep_spot, v.rep_left = (spot, pan), repeats - 1
        v.prev = (spot, t, ratio, rev)
        a, r = _window_edges(shape)
        th = (pan + 1.0) * 0.25 * np.pi
        gl, gr = float(np.cos(th)) * np.sqrt(2.0), float(np.sin(th)) * np.sqrt(2.0)
        v.g_pos = np.append(v.g_pos, spot)
        v.g_len = np.append(v.g_len, n_out)
        v.g_ratio = np.append(v.g_ratio, ratio)
        v.g_rev = np.append(v.g_rev, rev)
        v.g_age = np.append(v.g_age, -int(j))
        v.g_a = np.append(v.g_a, a)
        v.g_r = np.append(v.g_r, r)
        v.g_gain = np.concatenate([v.g_gain, np.array([[gl], [gr]], np.float32)], axis=1)
        v.spawned += 1

    def block(self, stems, n, bar_f, t0, rate, locked, out, bs):
        """The voices into `out` ({stem: (2, bs)}, the section's block,
        every stem present): the section ducked by the voices' gates, the
        grains added, through the same stems. `t0` is the transport frame
        at the block's start, `rate` the transport's rate (frames per
        output frame), `locked` whether the section keeps its pitch under
        it (then so do the grains). Returns the duck at the block's end,
        0..1 (the hits go out under 0.5)."""
        div, size, rfrac, motion, pitch, speed, repeats, shape, dirn, play, A, D, S, R = \
            self._settings()
        self._requests(t0, bar_f, n, div)
        if not self.voices or stems is None or not n or not bar_f:
            self._duck_end = 0.0
            return 0.0
        sr = self.sr
        if self._mono_key != id(stems):                        # the stems summed, once a section
            self._mono_key = id(stems)
            self._mono = np.zeros(n, np.float32)
            for data in stems.values():
                self._mono += data[0]
        size_f = max(32.0, size * bar_f)                       # transport frames
        n_out = max(32, int(round(size_f / rate)))             # output frames per grain
        ratio = 2.0 ** (pitch / 12.0) * (1.0 if locked else rate)
        rate_f = rfrac * bar_f                                 # the grid's step, or 0
        interval = (rate_f if rate_f > 0 else size_f / DENSITY) / rate   # output frames
        overlap = max(1.0, n_out / max(1.0, interval))
        a_w, r_w = _window_edges(shape)
        wmean, w2mean = 1.0 - (a_w + r_w) / 2.0, 1.0 - (a_w + r_w) * 5.0 / 8.0
        c = min(1.0, motion)
        gain = (1.0 / (overlap * wmean)) * (1.0 - c) + c / np.sqrt(overlap * w2mean)
        k = np.arange(bs)
        sp = speed * rate                                      # slice frames per output frame
        duck = np.zeros(bs, np.float32)
        add = {name: np.zeros((2, bs), np.float32) for name in stems}
        for v in list(self.voices):
            env, gate = self._env(v, bs, A, D, S, R)
            duck = np.maximum(duck, gate)
            # the playhead over the block
            if play == "ping-pong":
                raw = v.p + v.dirn * sp * k
                per = 2.0 * v.L
                ph = np.mod(raw, per)
                pos = np.where(ph < v.L, ph, per - ph)
                end = v.p + v.dirn * sp * bs
                flips = int(abs(end) // v.L) != int(abs(v.p) // v.L) if sp else False
                v.p = float(np.mod(end, per))
                if v.p >= v.L:
                    v.p, v.dirn = per - v.p, -v.dirn
                    v.p = float(np.clip(v.p, 0.0, v.L - 1e-3))
            elif play == "once":
                raw = v.p + sp * k
                pos = np.clip(raw, 0.0, v.L - 1e-3)
                v.p = float(v.p + sp * bs)
                if (v.p >= v.L or v.p < 0.0) and not v.released:
                    v.released = True
                    v.stage, v.rel_from = "R", v.lvl
                v.p = float(np.clip(v.p, 0.0, v.L - 1e-3))
            else:
                pos = np.mod(v.p + sp * k, v.L)
                v.p = float(np.mod(v.p + sp * bs, v.L))
            # the spawns: on the bar's grid every rate_f transport frames,
            # or free-running, dense
            if not v.released or v.lvl > 0.0:
                if rate_f > 0:
                    m = int(np.ceil(t0 / rate_f))
                    j = (m * rate_f - t0) / rate
                    while j < bs:
                        if len(v.g_len) < GRAIN_MAX:
                            jj = int(j)
                            self._spawn(v, jj, float(pos[min(bs - 1, jj)]), size_f, n_out,
                                        ratio, motion, dirn, repeats, shape)
                        m += 1
                        j = (m * rate_f - t0) / rate
                else:
                    j = v._next
                    while j < bs:
                        if len(v.g_len) < GRAIN_MAX:
                            jj = int(j)
                            self._spawn(v, jj, float(pos[min(bs - 1, jj)]), size_f, n_out,
                                        ratio, motion, dirn, repeats, shape)
                        j += interval
                    v._next = j - bs
            G = len(v.g_len)
            if G:
                jg = v.g_age[:, None] + k[None, :]                      # (G, bs)
                ng = v.g_len[:, None]
                on = (jg >= 0) & (jg < ng)
                jj = np.where(on, jg, 0)
                u = jj / ng
                a, r = v.g_a[:, None], v.g_r[:, None]
                w = np.where(u < a, 0.5 - 0.5 * np.cos(np.pi * np.minimum(u, a) / a),
                             np.where(u > 1.0 - r,
                                      0.5 - 0.5 * np.cos(np.pi * np.minimum(1.0 - u, r) / r),
                                      1.0))
                off = np.where(v.g_rev[:, None], -jj, jj) * v.g_ratio[:, None] + v.g_pos[:, None]
                src = v.s0 + np.mod(off, v.L)                           # wraps in the slice
                i0 = np.floor(src)
                fr = (src - i0).astype(np.float32)
                i0 = i0.astype(np.int64) % n
                i1 = (i0 + 1) % n
                win = (w * on).astype(np.float32) * gain
                for name, data in stems.items():
                    blk = add[name]
                    for ch in range(2):
                        d = data[ch]
                        sv = d[i0] * (1.0 - fr) + d[i1] * fr
                        blk[ch] += np.sum(sv * win * v.g_gain[ch][:, None], axis=0) * env
                v.g_age = v.g_age + bs
                keep = v.g_age < v.g_len
                if not keep.all():
                    v.g_pos, v.g_len = v.g_pos[keep], v.g_len[keep]
                    v.g_ratio, v.g_rev = v.g_ratio[keep], v.g_rev[keep]
                    v.g_age, v.g_a, v.g_r = v.g_age[keep], v.g_a[keep], v.g_r[keep]
                    v.g_gain = v.g_gain[:, keep]
            v.t += bs
            if v.done:
                self.voices.remove(v)
        keep_gain = 1.0 - duck
        for name in stems:
            blk = out.get(name)
            if blk is None:
                out[name] = add[name]
            else:
                out[name] = blk * keep_gain + add[name]
        self._duck_end = float(duck[-1])
        return self._duck_end
