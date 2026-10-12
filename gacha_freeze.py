#!/usr/bin/env python3
"""
gacha_freeze.py — the freeze: the whole mix held on the instant of a press.

One button (held, or latched) on the master: the last seconds of what is
heard (the tape through, the section, the song and the bursts, after the
racks and the sync delay, so what freezes is what lines up with the
picture) sit in a ring; at the press the ring's write head stops and the
instant just before it is played as a cloud of grains read from that
spot, WSOLA-aligned so with no smear it is a clean drone of that moment
(the grains line up in phase, the attic cloud's trick), while the live
mix crossfades out under it over the attack; the transport runs on
underneath, so the let-go crossfades the live mix back over the release,
right where the loop is, and the tape through never has to catch up.

Settings, one dict (FREEZE), set from the GUI thread and read per block:
size (a grain, ms), smear (0..1: a position jitter round the spot of up
to half a second, a few cents of detune and the stereo spray; past the
middle, octave and fifth scatter and reversed grains), pitch (semitones),
attack and release (seconds). `held` is the button; `active` tells the
mix (and the MIDI out) whether the freeze sounds.

    fz = Freeze(48000)
    fz.held = True                       # any thread
    y = fz.process(y)                    # audio thread: (2, bs) in, (2, bs) out
"""
import numpy as np

FREEZE = {"size_ms": 80.0,     # a grain's length, 20 to 250 ms
          "smear": 0.0,        # 0..1: the spread round the frozen spot
          "pitch": 0.0,        # semitones, -24 to +24
          "attack": 0.01,      # s: the live mix out, the cloud in
          "release": 0.25}     # s: the cloud out, the live mix back
OVERLAP = 4.0                  # grains sounding at once, whatever the size
GRAIN_MAX = 48
SEARCH, TPL = 256, 256         # the alignment: +-5 ms searched over a 5 ms template
INTERVALS = (12.0, -12.0, 7.0, -5.0, 12.0, -12.0)   # the scatter's steps, past half smear


class Freeze:
    """The master freeze (module note). `process()` is the cheap path
    while nothing is held: a ring write and the block handed back."""

    def __init__(self, sr):
        self.sr = int(sr)
        self.n = 1 << int(4.0 * sr).bit_length()             # ring, >= 4 s
        self.ring = np.zeros((2, self.n), np.float32)
        self.rng = np.random.default_rng()
        self.held = False                                      # the button, any thread
        self.active = False                                    # the cloud sounds (audio thread)
        self.level = 0.0                                       # the cloud's peak, smoothed
        self._written = 0                                      # frames written in all
        self._t = 0                                            # output frames in all
        self._env = 0.0                                        # the cloud's share, 0..1
        self._on = False                                       # frozen: the write head stopped
        self._next = 0.0                                       # frames to the next spawn
        self._prev = None                                      # (start, t, ratio) of the last forward grain
        self._clear()

    def reset(self):
        self.ring[:] = 0.0
        self._env, self._on, self._next, self._prev = 0.0, False, 0.0, None
        self.active, self.level = False, 0.0
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
        f = FREEZE
        size = max(0.02, min(0.25, float(f.get("size_ms", 80.0)) / 1000.0))
        smear = max(0.0, min(1.0, float(f.get("smear", 0.0))))
        pitch = max(-24.0, min(24.0, float(f.get("pitch", 0.0))))
        att = max(0.001, min(2.0, float(f.get("attack", 0.01))))
        rel = max(0.001, min(4.0, float(f.get("release", 0.25))))
        return size, smear, pitch, att, rel

    def state(self):
        """What sounds, for the picture: the newest grains as (seconds
        behind the frozen spot of the grain's far edge, its length in
        seconds, its phase 0..1, backwards), the level, and the share."""
        heads = []
        # the audio thread replaces these arrays one by one as grains die:
        # read one snapshot of each and no further than the shortest
        g_len, g_age, g_ratio = self.g_len, self.g_age, self.g_ratio
        g_rev, g_start = self.g_rev, self.g_start
        G = min(len(g_len), len(g_age), len(g_ratio), len(g_rev), len(g_start))
        if G:
            sr = self.sr
            order = np.argsort(g_age[:G])[:16]
            for i in order:
                n, r, rev = int(g_len[i]), float(g_ratio[i]), bool(g_rev[i])
                span = n * r
                left = g_start[i] - span if rev else g_start[i]
                heads.append((float((self._written - left) / sr), n / sr,
                              float(min(1.0, max(0.0, g_age[i] / max(1, n)))), rev))
        return {"grains": G, "heads": heads, "level": min(1.0, self.level),
                "env": self._env}

    def _spawn(self, k, size, smear, pitch):
        """One grain starting `k` frames into the block, from the frozen
        spot (the instant before the write head stopped)."""
        rng = self.rng
        sr = self.sr
        n = max(32, int(size * sr * (1.0 + 0.3 * smear * (rng.random() - 0.5))))
        st = pitch
        scatter = max(0.0, smear - 0.5) * 2.0                   # past the middle only
        if scatter and rng.random() < 0.6 * scatter:
            st += INTERVALS[rng.integers(len(INTERVALS))]
        st += 0.3 * smear * (rng.random() - 0.5)                # a few cents of detune
        ratio = float(2.0 ** (max(-36.0, min(36.0, st)) / 12.0))
        rev = bool(scatter and rng.random() < 0.5 * scatter)
        # the spot: the whole grain written before the head stopped, plus
        # the smear's jitter (up to half a second back)
        need = n * ratio + 2 + SEARCH
        back = need + 0.5 * smear * sr * rng.random()
        start = self._written - back
        t = self._t + k
        if rev:
            start += n * ratio
        else:
            start = self._align(start, n, ratio, t)
            self._prev = (start, t, ratio)
        alpha = 1.0 if size <= 0.04 else max(0.4, 1.0 - (size - 0.04) / 0.21 * 0.6)
        coherent = 1.0 / (OVERLAP * (1.0 - alpha / 2.0))
        by_power = 1.0 / np.sqrt(OVERLAP * (1.0 - 5.0 * alpha / 8.0))
        gain = coherent * (1.0 - smear) + by_power * smear
        pan = 0.8 * smear * (rng.random() * 2.0 - 1.0)
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

    def _align(self, start, n, ratio, t):
        """The spot moved within +-SEARCH frames to where the source best
        matches the last grain's read head now (same ratio only)."""
        pv = self._prev
        if pv is None or pv[2] != ratio:
            return start
        ps, pt, _r = pv
        natural = ps + (t - pt) * ratio
        W, S = TPL, SEARCH
        if natural < 0 or natural + W > self._written:
            return start
        lo = start - S
        if lo < 0 or start + S + n * ratio + 2 > self._written + 1:
            return start
        rn = self.n
        tpl_i = (int(natural) + np.arange(W)) % rn
        span_i = (int(lo) + np.arange(2 * S + W)) % rn
        L, R = self.ring                                       # summed where read only,
        tpl, span = L[tpl_i] + R[tpl_i], L[span_i] + R[span_i]   # not the whole 4 s ring
        if not tpl.any():
            return start
        c = np.correlate(span, tpl, "valid")
        return lo + int(np.argmax(c)) + (start - int(start))

    def process(self, y):
        """(2, bs) of the mix in, the mix with the freeze over it out."""
        bs = y.shape[1]
        sr = self.sr
        held = bool(self.held)
        if not held and not self._on:                          # idle: the ring only
            idx = (self._written + np.arange(bs)) % self.n
            self.ring[:, idx] = y
            self._written += bs
            self.active = False
            return y
        size, smear, pitch, att, rel = self._settings()
        if held and not self._on:                              # the press: the head stops
            self._on = True
            self._prev = None
            self._next = 0.0
            self.active = True
        # the cloud's share: up over the attack, down over the release
        e0 = self._env
        step = bs / (sr * (att if held else rel))
        e1 = min(1.0, e0 + step) if held else max(0.0, e0 - step)
        env = np.linspace(e0, e1, bs, dtype=np.float32) if e1 != e0 else e0
        self._env = e1
        # the grains, from the frozen spot
        interval = max(32.0, size * sr / OVERLAP)
        nxt = self._next
        if held or e1 > 0.0:
            while nxt < bs:
                if len(self.g_len) < GRAIN_MAX:
                    self._spawn(nxt, size, smear, pitch)
                nxt += interval * (1.0 + 0.6 * smear * (self.rng.random() - 0.5))
        self._next = nxt - bs
        self._t += bs
        out = np.zeros((2, bs), np.float32)
        G = len(self.g_len)
        if G:
            k = np.arange(bs)
            j = self.g_age[:, None] + k[None, :]
            n = self.g_len[:, None]
            on = (j >= 0) & (j < n)
            jj = np.where(on, j, 0)
            u = jj / n
            al = self.g_alpha[:, None]
            edge = np.minimum(u, 1.0 - u)
            w = np.where(edge < al * 0.5,
                         0.5 - 0.5 * np.cos(2.0 * np.pi * np.minimum(edge, al * 0.5) / al), 1.0)
            src = np.where(self.g_rev[:, None], -jj, jj) * self.g_ratio[:, None] \
                + self.g_start[:, None]
            i0 = np.floor(src)
            fr = (src - i0).astype(np.float32)
            i0 = i0.astype(np.int64) % self.n
            i1 = (i0 + 1) % self.n
            win = (w * on).astype(np.float32)
            for c in range(2):
                r = self.ring[c]
                sv = r[i0] * (1.0 - fr) + r[i1] * fr
                out[c] = np.sum(sv * win * self.g_gain[c][:, None], axis=0)
            self.g_age = self.g_age + bs
            keep = self.g_age < self.g_len
            if not keep.all():
                self.g_start, self.g_len = self.g_start[keep], self.g_len[keep]
                self.g_ratio, self.g_rev = self.g_ratio[keep], self.g_rev[keep]
                self.g_age, self.g_alpha = self.g_age[keep], self.g_alpha[keep]
                self.g_gain = self.g_gain[:, keep]
        pk = float(np.abs(out).max()) if out.size else 0.0
        self.level = max(pk, self.level * 0.9)
        mixed = y * (1.0 - env) + out * env
        if not held and e1 <= 0.0:                             # let go and faded: the head runs again
            self._on = False
            self.active = False
            self._clear()
            self._prev = None
            idx = (self._written + np.arange(bs)) % self.n
            self.ring[:, idx] = y
            self._written += bs
        return mixed.astype(np.float32, copy=False)
