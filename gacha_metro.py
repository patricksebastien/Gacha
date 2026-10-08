#!/usr/bin/env python3
"""
gacha_metro.py — the metronome: a click on every beat, into the mix and
out as a MIDI note.

Driven from the audio thread, one call per block. The beat grid is the
playing section's own (its transport frames, through the gears and a tape
stop, the grid the MIDI clock follows) or, with no section, the live
clock's: the GUI hands the tap tempo's downbeat anchor (monotonic) and the
beat length, and the clock's downbeat is measured as a frame of the stream every block
and smoothed (the clock's time stamp is coarse on Windows, 16 ms; the
audio frames are not), so the clicks are steady and the one is the
clock's; a tap or a tempo change snaps it.

The click is a short sine burst, higher and louder on the one; it goes
into the mix before the inserts and the main volume, never into the round
trip test. The notes go out on the MIDI out port when one is open, under
the same lead as the drum hits: one note for the one (note_one) and
another for beats 2, 3 and 4 (note), either -1 for none; velocity 127 on
the one, 90 on the others, the gate the hits'.

    m = Metronome(48000)
    m.on, m.note_one, m.note = True, 76, 77                          # GUI thread
    m.anchor, m.beat_s = tempo.anchor, 0.5
    beats = m.block(bs, grid, now, lead_frames)  # audio thread: [(off, accent)] for the MIDI
    m.render(y)                                  # then the clicks into y (2, bs), in place
"""
import math

import numpy as np

CLICK_S = 0.025            # a click's length
CLICK_LEVEL = 0.4          # its peak on the one, the other beats a little under
CLICK_HZ = (1200.0, 1800.0)   # the beat, the one
EMA = 0.02                 # the free grid's smoothing: ~50 blocks to settle
VEL = (90, 127)            # the note's velocity: the beat, the one


class Metronome:
    def __init__(self, sr):
        self.sr = int(sr)
        self.on = False                  # the checkbox (GUI thread)
        self.note_one = 76               # the one's MIDI note (GUI thread), -1 = none
        self.note = 77                   # the other beats' note (GUI thread), -1 = none
        self.anchor = None               # the live clock's downbeat, monotonic (GUI thread)
        self.beat_s = 0.5                # the live clock's beat (GUI thread)
        self._frames = 0                 # output frames so far
        self._clicks = []                # (start frame, accent) sounding or due
        self._free = None                # the free grid: [downbeat frame, beat_s, last click, last note]
        n = int(CLICK_S * self.sr)
        t = np.arange(n) / self.sr
        env = np.minimum(1.0, t / 0.001) * np.exp(-t / (CLICK_S / 4.0))
        self._wave = tuple((CLICK_LEVEL * (1.0 if a else 0.8) * env
                            * np.sin(2.0 * np.pi * CLICK_HZ[a] * t)).astype(np.float32)
                           for a in (0, 1))

    def block(self, bs, grid, now, lead):
        """The beats of this block. `grid` is the section's (i0, bf, r,
        start): transport frame `i0` at output frame `start`, `bf` frames
        a beat, `r` transport frames per output frame; or None for the
        live clock. `now` is the block's monotonic time stamp, `lead` the
        MIDI out's lookahead in frames. Queues the clicks that fall in
        this block and returns [(off, accent)]: the beats the notes should
        leave for, `off` in output frames from this block's frame `lead`
        on (the hits' time base), each beat once."""
        out = []
        f0 = self._frames
        self._frames += bs
        if not self.on:
            self._free = None
            return out
        sr = self.sr
        if grid is not None and grid[1] > 0 and grid[2] > 0:
            i0, bf, r, start = grid
            self._free = None                              # the section owns the grid
            k = math.ceil(i0 / bf)
            beats = []
            while True:
                f = start + (k * bf - i0) / r
                if f >= bs + lead:
                    break
                beats.append((f, (k % 4) == 0))
                k += 1
        else:
            beat_s = float(self.beat_s)
            anchor = self.anchor
            if anchor is None or beat_s <= 0:
                return out
            beat_f = beat_s * sr
            # the clock's downbeat as a frame of ours: measured each block
            # (coarse, the time stamp's jitter), smoothed; snapped on a
            # tempo change or a jump of a beat or more
            a_meas = f0 - (now - anchor) * sr
            if self._free is None or self._free[1] != beat_s                     or abs(a_meas - self._free[0]) >= beat_f:
                self._free = [a_meas, beat_s, None, None]
            else:
                self._free[0] += EMA * (a_meas - self._free[0])
            A, _bs, last_c, last_m = self._free
            k_lo = math.ceil((f0 - A) / beat_f - 1e-9)   # the first beat at or after this block
            # the clicks: every beat once, the next after the last one
            # played (a beat the smoothing slid back over the edge plays now)
            k = k_lo if last_c is None else last_c + 1
            while A + k * beat_f < f0 + bs:
                self._clicks.append((max(f0, int(A + k * beat_f)), (k % 4) == 0))
                last_c = k
                k += 1
            # the notes: the same from `lead` on
            k = last_m + 1 if last_m is not None else math.ceil((f0 + lead - A) / beat_f - 1e-9)
            while A + k * beat_f < f0 + bs + lead:
                out.append((max(0.0, A + k * beat_f - f0 - lead), (k % 4) == 0))
                last_m = k
                k += 1
            self._free[2], self._free[3] = last_c, last_m
            return out
        for f, acc in beats:
            if 0 <= f < bs:
                self._clicks.append((f0 + int(f), acc))
            if lead <= f < bs + lead:
                out.append((f - lead, acc))
        return out

    def render(self, y):
        """The clicks sounding into `y` (2, bs), in place; after block()."""
        if not self._clicks:
            return
        bs = y.shape[1]
        f0 = self._frames - bs                             # this block's first frame
        keep = []
        for start, acc in self._clicks:
            w = self._wave[1 if acc else 0]
            a = max(start, f0)
            b = min(start + len(w), f0 + bs)
            if a < b:
                y[:, a - f0:b - f0] += w[a - start:b - start]
            if start + len(w) > f0 + bs:
                keep.append((start, acc))
        self._clicks = keep
