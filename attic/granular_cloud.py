#!/usr/bin/env python3
"""
attic/granular_cloud.py — the first granular, kept for reference (2026-10-06).

A granular cloud as a rack effect (gacha_fx.py's 11th effect, one knob per
channel: the amount, with a share of the cloud fed back past a third), its
size, pitch, spread and freeze one setting for every channel (the
GRANULAR dict, a Perform tab row), WSOLA-aligned grains under Tukey
windows. Replaced by gacha_frag.py (Fragments: the bar sliced, each slice
played as grains, Bitwig's Sampler way) after a listen; not wired in
anywhere. To bring it back: add ("granular", ...) to AUDIO_FX, build it in
Rack._get, call p.process(x, a) in Rack._wet.

    g = Granular(48000)
    GRANULAR.update(size_ms=80, pitch=0, spread=0.4, freeze=False)
    wet = g.process(x, amount)          # (2, bs) float32 in and out
"""
import numpy as np
from scipy.signal import lfilter


# the granular's settings, one set for every channel (the Perform tab's
# Granular row): set from any thread, read per block
GRANULAR = {"size_ms": 80.0,     # a grain's length, 10 to 250 ms
            "pitch": 0.0,        # semitones, -24 to +24
            "spread": 0.4,       # 0..1: position jitter, pitch scatter, pan, reverse
            "freeze": False}     # the buffer stops: the cloud reads the same spot
GRAIN_MAX = 64                   # grains sounding at once, at the most
GRAIN_SEARCH, GRAIN_TPL = 256, 256   # the alignment: +-5 ms searched over a 5 ms template
GRAIN_INTERVALS = (12.0, -12.0, 7.0, -5.0, 12.0, -12.0)   # the scatter's steps


class Granular:
    """A granular cloud over a ring buffer of the channel's last seconds.

    Grains are spawned at a steady rate (`overlap` of them sounding at
    once, 2 at a low amount up to 8 at full, whatever their size) from a
    jittered spot behind the write head, each with its own length, pitch
    ratio, direction and pan, and summed under Tukey windows (Hann for the
    short ones, flatter as they lengthen so a 250 ms grain keeps its
    transients). The spread knob is one knob for everything random: the
    position jitter (up to a second back), the pitch scatter (octaves and
    fifths, a few cents of detune), the stereo spray and, past the middle,
    reversed grains. A share of the cloud is fed back into the buffer at
    high amounts, so it blooms, darkened a little each pass and soft-clipped
    so it cannot run away. Frozen, the write head stops and the cloud
    plays the buffer as it was: with no spread one spot loops, a drone.

    What keeps it smooth: a new grain at the same pitch as the last one is
    lined up with it (WSOLA: the spot is searched +-5 ms for the best
    match against where the last grain's read head is now), so overlapping
    grains add in phase instead of combing the fundamental out, the warble
    of the cheap granular shifters. At no spread and no shift the cloud is
    then the signal itself; a shift is clean; the spread's jitter still
    lands each grain on a matching phase of the wave.

    Every grain that sounds is one row of a few arrays; a block is one
    gather over (grains x frames), so the cost does not grow with the
    grain count in Python. `state()` is what the picture follows."""

    def __init__(self, sr):
        self.sr = int(sr)
        self.n = 1 << int(4.0 * sr).bit_length()             # ring, >= 4 s
        self.ring = np.zeros((2, self.n), np.float32)
        self.wpos = 0                                          # next frame written
        self.rng = np.random.default_rng()
        self._next = 0.0                                       # frames to the next spawn
        self._fb = None                                        # last wet block, fed back
        self._lp = np.zeros((2, 1), np.float64)                # the feedback's lowpass state
        self.spawned = 0                                       # grains spawned in all
        self.level = 0.0                                       # wet peak, smoothed
        self._written = 0                                      # frames written in all
        self._t = 0                                            # output frames in all
        self._prev = None                                      # (start, t, ratio) of the last forward grain
        # the grains sounding: parallel arrays, compacted every block
        self._clear()

    def reset(self):
        self.ring[:] = 0.0
        self._next = 0.0
        self._fb = None
        self.level = 0.0
        self._prev = None
        self._lp = np.zeros((2, 1), np.float64)
        self._clear()

    def _clear(self):
        self.g_start = np.zeros(0, np.float64)                 # source frame (absolute)
        self.g_len = np.zeros(0, np.int64)                     # frames at the output
        self.g_ratio = np.zeros(0, np.float64)                 # source frames per frame
        self.g_rev = np.zeros(0, bool)
        self.g_age = np.zeros(0, np.int64)                     # frames already output
        self.g_alpha = np.zeros(0, np.float64)                 # the Tukey ramp share
        self.g_gain = np.zeros((2, 0), np.float32)             # per channel, with the pan

    @staticmethod
    def _settings():
        g = GRANULAR
        size = max(0.01, min(0.25, float(g.get("size_ms", 80.0)) / 1000.0))
        pitch = max(-24.0, min(24.0, float(g.get("pitch", 0.0))))
        spread = max(0.0, min(1.0, float(g.get("spread", 0.4))))
        return size, pitch, spread, bool(g.get("freeze", False))

    def _spawn(self, k, size, pitch, spread, a):
        """One grain starting `k` frames into the block (past the block's
        end: it starts later)."""
        rng = self.rng
        sr = self.sr
        n = max(32, int(size * sr * (1.0 + 0.3 * spread * (rng.random() - 0.5))))
        st = pitch
        if rng.random() < 0.6 * spread:                        # the scatter: a step
            st += GRAIN_INTERVALS[rng.integers(len(GRAIN_INTERVALS))]
        st += 0.3 * spread * (rng.random() - 0.5)               # a few cents of detune
        ratio = float(2.0 ** (max(-36.0, min(36.0, st)) / 12.0))
        rev = bool(spread > 0.5 and rng.random() < (spread - 0.5))
        # the spot: far enough behind the write head for the whole grain
        # to be written already, plus the jitter (up to a second back)
        need = n * ratio + 2 + GRAIN_SEARCH                     # room for the alignment too
        back = need + spread * sr * rng.random()
        start = self._written - back
        t = self._t + k
        if rev:
            start += n * ratio                                  # runs back to the spot
        else:
            start = self._align(start, n, ratio, t)
            self._prev = (start, t, ratio)
        # the level: the windows summing coherently (spread 0) or by power,
        # for this size's window (a Tukey of ramp share alpha has a mean of
        # 1 - alpha/2 and a mean square of 1 - 5 alpha/8; Hann: 1/2, 3/8)
        overlap = 2.0 + 6.0 * a
        alpha = 1.0 if size <= 0.04 else max(0.4, 1.0 - (size - 0.04) / 0.21 * 0.6)
        coherent = 1.0 / (overlap * (1.0 - alpha / 2.0))
        by_power = 1.0 / np.sqrt(overlap * (1.0 - 5.0 * alpha / 8.0))
        gain = coherent * (1.0 - spread) + by_power * spread
        pan = 0.8 * spread * (rng.random() * 2.0 - 1.0)        # -1..1
        th = (pan + 1.0) * 0.25 * np.pi
        gl, gr = float(np.cos(th)) * np.sqrt(2.0), float(np.sin(th)) * np.sqrt(2.0)
        self.g_start = np.append(self.g_start, start)
        self.g_len = np.append(self.g_len, n)
        self.g_ratio = np.append(self.g_ratio, ratio)
        self.g_rev = np.append(self.g_rev, rev)
        self.g_age = np.append(self.g_age, -int(k))
        self.g_alpha = np.append(self.g_alpha, alpha)
        self.g_gain = np.concatenate(
            [self.g_gain, np.array([[gl * gain], [gr * gain]], np.float32)], axis=1)
        self.spawned += 1

    def _align(self, start, n, ratio, t):
        """The spot moved within +-GRAIN_SEARCH frames to where the source
        best matches the last grain's read head now (same ratio only)."""
        pv = self._prev
        if pv is None or pv[2] != ratio:
            return start
        ps, pt, _r = pv
        natural = ps + (t - pt) * ratio                        # where it reads now
        W, S = GRAIN_TPL, GRAIN_SEARCH
        if natural < 0 or natural + W > self._written:
            return start
        lo = start - S
        if lo < 0 or start + S + n * ratio + 2 > self._written + 1:
            return start
        rn = self.n
        tpl_i = (int(natural) + np.arange(W)) % rn
        span_i = (int(lo) + np.arange(2 * S + W)) % rn
        mono = self.ring[0] + self.ring[1]
        tpl, span = mono[tpl_i], mono[span_i]
        if not tpl.any():
            return start
        c = np.correlate(span, tpl, "valid")                   # 2 * S + 1 offsets
        return lo + int(np.argmax(c)) + (start - int(start))

    def process(self, x, a):
        """(2, bs) in, the fully wet cloud (2, bs) out, at amount `a`."""
        bs = x.shape[1]
        sr = self.sr
        size, pitch, spread, freeze = self._settings()
        # the buffer: the input, plus a share of the last cloud (the bloom);
        # frozen, nothing is written and the write head stays
        if not freeze:
            blk = x
            fb = 0.5 * (a - 0.3) / 0.7 if a > 0.3 else 0.0
            if fb > 0.0 and self._fb is not None and self._fb.shape == x.shape:
                # the bloom, tamed: the cloud darkened a little each pass
                # (a one-pole lowpass, the tape's way) and soft-clipped, so
                # it can never run away however the grains pile up
                k = 0.35                                        # ~3.5 kHz at 48 kHz
                z, self._lp = lfilter([k], [1.0, k - 1.0], self._fb * fb, axis=1, zi=self._lp)
                blk = x + np.tanh(z).astype(np.float32)
            idx = (self.wpos + np.arange(bs)) % self.n
            self.ring[:, idx] = blk
            self.wpos = (self.wpos + bs) % self.n
            self._written += bs
        # the scheduler: `overlap` grains at once whatever the size
        overlap = 2.0 + 6.0 * a
        interval = max(32.0, size * sr / overlap)
        nxt = self._next
        while nxt < bs:
            if len(self.g_len) < GRAIN_MAX:
                self._spawn(nxt, size, pitch, spread, a)
            nxt += interval * (1.0 + 0.6 * spread * (self.rng.random() - 0.5))
        self._next = nxt - bs
        self._t += bs
        out = np.zeros((2, bs), np.float32)
        G = len(self.g_len)
        if G:
            k = np.arange(bs)
            j = self.g_age[:, None] + k[None, :]                        # (G, bs): the grain's frame
            n = self.g_len[:, None]
            on = (j >= 0) & (j < n)
            jj = np.where(on, j, 0)
            u = jj / n                                                  # 0..1 over the grain
            # Tukey: a Hann ramp over alpha/2 at each end, flat between
            al = self.g_alpha[:, None]
            edge = np.minimum(u, 1.0 - u)
            w = np.where(edge < al * 0.5,
                         0.5 - 0.5 * np.cos(2.0 * np.pi * np.minimum(edge, al * 0.5) / al), 1.0)
            src = np.where(self.g_rev[:, None], -jj, jj) * self.g_ratio[:, None] \
                + self.g_start[:, None]
            # absolute frames to the ring, linear interpolation
            i0 = np.floor(src)
            fr = (src - i0).astype(np.float32)
            i0 = i0.astype(np.int64) % self.n
            i1 = (i0 + 1) % self.n
            win = (w * on).astype(np.float32)                           # (G, bs)
            for c in range(2):
                r = self.ring[c]
                sv = r[i0] * (1.0 - fr) + r[i1] * fr                    # (G, bs)
                out[c] = np.sum(sv * win * self.g_gain[c][:, None], axis=0)
            # the grains run on; the finished ones go
            self.g_age = self.g_age + bs
            keep = self.g_age < self.g_len
            if not keep.all():
                self.g_start, self.g_len = self.g_start[keep], self.g_len[keep]
                self.g_ratio, self.g_rev = self.g_ratio[keep], self.g_rev[keep]
                self.g_age, self.g_alpha = self.g_age[keep], self.g_alpha[keep]
                self.g_gain = self.g_gain[:, keep]
        self._fb = out
        pk = float(np.abs(out).max()) if out.size else 0.0
        self.level = max(pk, self.level * 0.9)
        return out

    def state(self):
        """What sounds, for the picture: the grains at once, the level, and
        the newest grains as heads: (seconds behind the write head of the
        grain's far edge, its length in seconds, its phase 0..1 through its
        window, backwards)."""
        G = len(self.g_len)
        heads = []
        if G:
            sr = self.sr
            order = np.argsort(self.g_age)[:16]                         # the newest first
            for i in order:
                n, r, rev = int(self.g_len[i]), float(self.g_ratio[i]), bool(self.g_rev[i])
                span = n * r
                left = self.g_start[i] - span if rev else self.g_start[i]
                heads.append((float((self._written - left) / sr), n / sr,
                              float(min(1.0, max(0.0, self.g_age[i] / n))), rev))
        return {"grains": G, "heads": heads, "spawned": self.spawned,
                "level": min(1.0, self.level)}
