#!/usr/bin/env python3
"""
gacha_fx.py — the realtime audio racks, one per channel of the live mix.

Five channels: the live input through and the four generated stems (drums, layers,
chops, events; a section from the section engine and a generated song both
come as these stems). Each channel has ten effects and a volume, every one a
knob from 0 to 1 (a MIDI CC 0..127 spread over it):

    0          bypassed, no CPU spent
    small      a little of the effect on top of the dry sound
    1 (127)    fully wet: the dry sound is gone, only the effect remains
               (a reverb wash, only repeats, the filter fully closed)

Pitch is the one bipolar knob: -1..1, off in the middle, -12 to +12 semitones.

Built on pedalboard plugins streamed block by block (`reset=False`) plus the
engine's tanh saturator (gain-compensated, so drive gets denser, not louder)
and a two-head delay-line pitch shifter of our own (pedalboard's PitchShift
takes a second to start producing sound in streaming mode; this one is a
tape-style shifter with 25 ms of latency and a little warble), and a phone
codec imitation (pedalboard's GSM codec needs 40 ms blocks to produce any
sound when streamed: a 3.4 kHz lowpass, 8 kHz sample-and-hold and 8-bit
mu-law give the same flavour at zero latency). A plugin handing back blocks
of varying length goes through a FIFO. Every amount is ramped over the
block it changes in, so knobs never zip. Delay time follows the beat
(`Rack.beat_s`, set from the song or the live clock): a dotted eighth, the
dub classic.

    rack = Rack(48000)
    rack.amounts["reverb"] = 0.4      # from any thread
    rack.vol = 0.8
    y = rack.process(x)               # (2, bs) float32 in, same out; audio thread
"""
import numpy as np
from pedalboard import (Bitcrush, Chorus, Delay, LadderFilter, LowpassFilter,
                        Phaser, Reverb)

# key, label, default, tooltip. Order = the chain order: filters first, then
# the tone shapers, then the modulations, then time effects.
AUDIO_FX = [
    ("highpass", "highpass", 0.0, "Ladder highpass sweeping up from 20 Hz to 8 kHz; "
                                  "thins the sound to a whisper at full"),
    ("lowpass", "lowpass", 0.0, "Ladder lowpass sweeping down from 20 kHz to 150 Hz, "
                                "resonance rising with it; the club door closing"),
    ("drive", "drive", 0.0, "The engine's tanh saturator, gain-compensated: denser, "
                            "not louder; up to +30 dB into the shaper at full"),
    ("crush", "bitcrush", 0.0, "Bit depth from 16 down to 3 bits at full, crossfaded in"),
    ("lofi", "lo-fi codec", 0.0, "Phone codec: 3.4 kHz lowpass, 8 kHz sample-and-hold, "
                                 "8-bit mu-law, crossfaded in"),
    ("chorus", "chorus", 0.0, "Chorus, depth and mix rising together"),
    ("phaser", "phaser", 0.0, "Phaser, depth, feedback and mix rising together"),
    ("pitch", "pitch", 0.0, "Pitch shift: off in the middle, down to -12 semitones "
                            "at the bottom, up to +12 at the top; the shift fades "
                            "in over the first semitones"),
    ("delay", "delay", 0.0, "Tempo-synced delay, a dotted eighth, feedback rising "
                            "with the knob; only repeats at full"),
    ("reverb", "reverb", 0.0, "Reverb, the room growing with the knob; a wash at full"),
]
BIPOLAR = {"pitch"}                      # -1..1 knobs, off at 0 in the middle
CHANNELS = ("live", "drums", "layers", "chops", "events")   # live = the input through
STEM_CHANNELS = CHANNELS[1:]
PITCH_DEAD = 0.04                        # |pitch| below this = off (knob centre)


def _tanh_drive(x, amount, bias=0.15):
    """Soft saturation, RMS matched by a fixed curve so the level holds."""
    d = 10 ** (amount * 30 / 20)                              # up to +30 dB
    y = np.tanh(x * d + bias) - np.tanh(bias)
    return y * (0.3 / max(1e-6, np.tanh(0.3 * d)))            # compensate


class LoFi:
    """A phone line: the band above 3.4 kHz gone, 8 kHz sample-and-hold
    (the decimation phase carries across blocks), 8-bit mu-law companding."""

    def __init__(self, sr, rate=8000.0):
        self.sr = int(sr)
        self.step = self.sr / rate                           # 6 at 48 kHz
        self.lp = LowpassFilter(cutoff_frequency_hz=3400.0)
        self.phase = 0.0                                     # next sample to hold
        self.held = np.zeros((2, 1), np.float32)

    def reset(self):
        self.lp.reset()
        self.phase = 0.0
        self.held[:] = 0.0

    def process(self, x):
        bs = x.shape[1]
        y = self.lp.process(x, self.sr, reset=False)
        # hold points: every `step` frames from the carried phase
        k = np.arange(bs)
        holds = np.floor((k - self.phase) / self.step + 1e-9) >= 0
        # index of the most recent hold point at or before each frame
        n_new = int(np.floor((bs - 1 - self.phase) / self.step)) + 1 if self.phase <= bs - 1 else 0
        out = np.empty_like(y)
        if n_new <= 0:
            out[:] = self.held
        else:
            pts = (self.phase + self.step * np.arange(n_new)).astype(np.int64)
            pts = np.clip(pts, 0, bs - 1)
            vals = y[:, pts]
            idx = np.searchsorted(pts, k, side="right") - 1        # -1 before the first
            out[:] = np.where(idx[None, :] >= 0, vals[:, np.clip(idx, 0, None)], self.held)
            self.held = vals[:, -1:].copy()
            self.phase = self.phase + self.step * n_new - bs
        mu = 255.0
        c = np.sign(out) * np.log1p(mu * np.abs(out)) / np.log1p(mu)   # compress
        c = np.round(c * 127.0) / 127.0                                  # 8 bits
        return (np.sign(c) * (np.expm1(np.abs(c) * np.log1p(mu)) / mu)).astype(np.float32)


class PitchShifter:
    """Two read heads sweeping a ring buffer at a rate of 2^(semitones/12)
    relative to the write head, each fading in and out over a window and
    offset by half of it so one is always loud: the granular / doppler
    shifter of old rack units. Latency is half the window."""

    def __init__(self, sr, window_s=0.05):
        self.sr = int(sr)
        self.win = max(64, int(window_s * sr))               # frames
        self.n = 1 << (self.win * 4).bit_length()            # ring, power of 2
        self.ring = np.zeros((2, self.n), np.float32)
        self.wpos = 0
        self.phase = 0.0                                     # head A, 0..1

    def reset(self):
        self.ring[:] = 0.0
        self.phase = 0.0

    def process(self, x, semitones):
        bs = x.shape[1]
        idx = (self.wpos + np.arange(bs)) % self.n
        self.ring[:, idx] = x
        ratio = 2.0 ** (semitones / 12.0)
        # the heads move against the write head by (1 - ratio) frames per
        # frame; their delay runs over 0..win and wraps, cross-faded
        step = (1.0 - ratio) / self.win
        ph_a = (self.phase + step * np.arange(bs)) % 1.0
        self.phase = float((self.phase + step * bs) % 1.0)
        out = np.zeros_like(x)
        for ph in (ph_a, (ph_a + 0.5) % 1.0):
            delay = ph * self.win
            pos = (self.wpos + np.arange(bs) - delay) % self.n
            i0 = np.floor(pos).astype(np.int64)
            fr = (pos - i0).astype(np.float32)
            i1 = (i0 + 1) % self.n
            g = (0.5 - 0.5 * np.cos(2.0 * np.pi * ph)).astype(np.float32)   # Hann
            out += (self.ring[:, i0] * (1.0 - fr) + self.ring[:, i1] * fr) * g
        self.wpos = (self.wpos + bs) % self.n
        return out


class Rack:
    """One channel's effects and volume, streamed block by block."""

    def __init__(self, sr):
        self.sr = int(sr)
        self.amounts = {k: d for k, _l, d, _t in AUDIO_FX}   # set from any thread
        self.vol = 1.0                                        # 0..1, unity at 1
        self.beat_s = 0.5                                     # delay time base
        self._now = dict(self.amounts)                        # ramp state
        self._vol_now = 1.0
        self._plug = {}                                       # lazily built plugins
        self._fifo = {}                                       # key -> (2, n) pending output
        self._delay_s = None

    # ---- plugins, built on first use ----
    def _get(self, key):
        p = self._plug.get(key)
        if p is None:
            sr = self.sr
            if key == "highpass":
                p = LadderFilter(mode=LadderFilter.Mode.HPF12, cutoff_hz=20.0, resonance=0.1)
            elif key == "lowpass":
                p = LadderFilter(mode=LadderFilter.Mode.LPF24, cutoff_hz=20000.0, resonance=0.1)
            elif key == "drive":
                p = LowpassFilter(cutoff_frequency_hz=8000.0)     # tone on the wet path
            elif key == "crush":
                p = Bitcrush(bit_depth=16.0)
            elif key == "lofi":
                p = LoFi(sr)
            elif key == "chorus":
                p = Chorus(rate_hz=0.8, depth=0.5, centre_delay_ms=7.0, feedback=0.0, mix=1.0)
            elif key == "phaser":
                p = Phaser(rate_hz=0.6, depth=0.7, centre_frequency_hz=1300.0,
                           feedback=0.3, mix=1.0)
            elif key == "pitch":
                p = PitchShifter(sr)
            elif key == "delay":
                p = Delay(delay_seconds=0.75 * self.beat_s, feedback=0.3, mix=1.0)
                self._delay_s = 0.75 * self.beat_s
            elif key == "reverb":
                p = Reverb(room_size=0.6, damping=0.5, wet_level=1.0, dry_level=0.0, width=1.0)
            self._plug[key] = p
        return p

    def _wet(self, key, x, a):
        """The fully wet signal of effect `key` at amount `a` (0..1, or
        -1..1 for pitch), parameters following the knob."""
        sr = self.sr
        p = self._get(key)
        if key == "highpass":
            p.cutoff_hz = 20.0 * (8000.0 / 20.0) ** a
            p.resonance = 0.1 + 0.5 * a
            return p.process(x, sr, reset=False)
        if key == "lowpass":
            p.cutoff_hz = 20000.0 * (150.0 / 20000.0) ** a
            p.resonance = 0.1 + 0.6 * a
            return p.process(x, sr, reset=False)
        if key == "drive":
            return p.process(_tanh_drive(x, a), sr, reset=False)
        if key == "crush":
            p.bit_depth = 16.0 - 13.0 * a
            return p.process(x, sr, reset=False)
        if key == "lofi":
            return p.process(x)
        if key == "chorus":
            p.depth = 0.2 + 0.7 * a
            return p.process(x, sr, reset=False)
        if key == "phaser":
            p.depth = 0.3 + 0.7 * a
            p.feedback = 0.2 + 0.5 * a
            return p.process(x, sr, reset=False)
        if key == "pitch":
            return p.process(x, 12.0 * a)
        if key == "delay":
            want = 0.75 * max(0.1, min(2.0, self.beat_s))
            if self._delay_s is None or abs(want - self._delay_s) > 1e-3:
                p.delay_seconds = want
                self._delay_s = want
            p.feedback = 0.2 + 0.55 * a
            return p.process(x, sr, reset=False)
        if key == "reverb":
            p.room_size = 0.3 + 0.65 * a
            p.damping = 0.6 - 0.3 * a
            return p.process(x, sr, reset=False)
        return x

    def _fit(self, key, y, x):
        """Plugins with latency (the codec) hand back 0 to bs+ frames per
        block: queue what comes and emit exactly bs, zeros while filling."""
        n = x.shape[1]
        if y.shape[1] == n and key not in self._fifo:
            return y
        q = self._fifo.get(key)
        q = y if q is None or q.shape[1] == 0 else np.concatenate([q, y], axis=1)
        if q.shape[1] >= n:
            out, self._fifo[key] = q[:, :n], q[:, n:]
            return np.ascontiguousarray(out)
        self._fifo[key] = q
        return np.zeros_like(x)

    def process(self, x):
        """(2, bs) float32 in, (2, bs) float32 out. Bypassed effects cost
        nothing; an effect coming back from 0 starts from a clean state."""
        bs = x.shape[1]
        y = x
        for key, _l, _d, _t in AUDIO_FX:
            target = float(self.amounts.get(key, 0.0))
            bipolar = key in BIPOLAR
            if bipolar:
                target = max(-1.0, min(1.0, target))
                if abs(target) < PITCH_DEAD:
                    target = 0.0
            else:
                target = max(0.0, min(1.0, target))
            now = self._now[key]
            if target == 0.0 and now == 0.0:
                continue
            if now == 0.0 and key in self._plug:
                self._plug[key].reset()                    # fresh tail
                self._fifo.pop(key, None)
            a = target if target != 0.0 else now          # params while fading out
            wet = self._fit(key, self._wet(key, np.ascontiguousarray(y, dtype=np.float32), a), y)
            if bipolar:                                    # wet share fades in over ~3 st
                w_t, w_n = min(1.0, abs(target) * 4.0), min(1.0, abs(now) * 4.0)
            else:
                w_t, w_n = target, now
            if w_t != w_n:
                w = np.linspace(w_n, w_t, bs, dtype=np.float32)
            else:
                w = w_t
            y = y * (1.0 - w) + wet * w
            self._now[key] = target
        vol = max(0.0, min(1.0, float(self.vol))) ** 2      # a gentle taper
        if vol != self._vol_now:
            y = y * np.linspace(self._vol_now, vol, bs, dtype=np.float32)
            self._vol_now = vol
        elif vol != 1.0:
            y = y * vol
        return np.ascontiguousarray(y, dtype=np.float32)


if __name__ == "__main__":                       # cost check from a terminal
    import time
    sr, bs = 48000, 256
    rack = Rack(sr)
    x = (np.random.default_rng(1).standard_normal((2, bs)) * 0.1).astype(np.float32)
    for k, *_ in AUDIO_FX:
        rack.amounts[k] = 0.5
    for _ in range(20):
        rack.process(x)
    t = time.perf_counter()
    for _ in range(200):
        rack.process(x)
    ms = (time.perf_counter() - t) / 200 * 1000
    print(f"all ten on: {ms:.2f} ms per {bs}-frame block ({1000 * bs / sr:.1f} ms of audio)")
