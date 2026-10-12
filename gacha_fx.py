#!/usr/bin/env python3
"""
gacha_fx.py — the realtime audio racks, one per channel of the live mix.

Five channels: the live input through and the four generated stems (drums, layers,
chops, events; a section from the section engine and a generated song both
come as these stems), plus one per drum voice of a section (VOICES: kick,
snare, closed and open hat, ride, hat stutter, sub bass), summed into the
drums channel after their racks, so the drums channel is the kit's bus.
Each channel has eleven effects and a volume, every one a knob from 0 to 1 (a
MIDI CC 0..127 spread over it):

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
mu-law give the same flavour at zero latency), and a reverse delay of our
own (ReverseDelay below: the typical pedal, a beat at a time backwards,
its echo fed back). A plugin handing back blocks
of varying length goes through a FIFO. Every amount is ramped over the
block it changes in, so knobs never zip. Delay time follows the beat
(`Rack.beat_s`, set from the song or the live clock): a dotted eighth, the
dub classic. What a knob does not say (the delay's time and feedback, the
reverb's room, the chorus rate...) is a setting shared by every rack,
FX_SETTINGS / FX_SET below, set from the GUI thread and read per block.

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
                            "not louder; up to the gain setting (30 dB) into the shaper "
                            "at full, a lowpass (tone) on the driven sound"),
    ("crush", "bitcrush", 0.0, "Bit depth from 16 down to the bits setting (3) at full, "
                               "crossfaded in"),
    ("lofi", "lo-fi codec", 0.0, "Phone codec: 3.4 kHz lowpass, a sample-and-hold at the "
                                 "rate setting (8 kHz), mu-law at the bits setting (8), "
                                 "crossfaded in"),
    ("chorus", "chorus", 0.0, "Chorus, depth and mix rising together; rate, delay and "
                              "feedback in the settings"),
    ("phaser", "phaser", 0.0, "Phaser, depth, feedback and mix rising together; rate and "
                              "centre in the settings"),
    ("pitch", "pitch", 0.0, "Pitch shift: off in the middle, down by the range setting "
                            "(12 semitones) at the bottom, up by it at the top; the shift "
                            "fades in over the first semitones"),
    ("delay", "delay", 0.0, "Tempo-synced delay at the time setting (a dotted eighth), "
                            "the feedback setting its repeats; this knob the wet mix, only "
                            "repeats at full"),
    ("reverse", "reverse delay", 0.0, "Reverse delay: the time's worth of sound played "
                                      "backwards, that much late, the reversed echo fed back "
                                      "so the repeats come backwards, forwards, backwards; "
                                      "the time and the feedback in the settings, this knob "
                                      "the wet mix, only echoes at full"),
    ("reverb", "reverb", 0.0, "Reverb at the size, damping and width settings; this knob "
                              "the wet mix, gentle over its first half, a wash at full"),
]
BIPOLAR = {"pitch"}                      # -1..1 knobs, off at 0 in the middle
CHANNELS = ("live", "drums", "layers", "chops", "events")   # live = the input through
STEM_CHANNELS = CHANNELS[1:]
# the drum voices of a section, each a stem of its own with its own rack,
# summed into the drums channel (the bus) above; the first six are the MIDI
# out's roles (gacha_midi.DRUM_ROLES), the sub bass has no note
VOICES = ("kick", "snare", "hihat", "ohat", "ride", "glitch", "sub")
VOICE_LABELS = {"kick": "kick", "snare": "snare", "hihat": "closed hat", "ohat": "open hat",
                "ride": "ride", "glitch": "hat stutter", "sub": "sub bass"}
ALL_CHANNELS = CHANNELS + VOICES         # every rack there is
PITCH_DEAD = 0.04                        # |pitch| below this = off (knob centre)


def _tanh_drive(x, amount, db=30.0, bias=0.15):
    """Soft saturation, RMS matched by a fixed curve so the level holds;
    `db` the push into the shaper at amount 1."""
    d = 10 ** (amount * db / 20)                              # up to +db
    y = np.tanh(x * d + bias) - np.tanh(bias)
    return y * (0.3 / max(1e-6, np.tanh(0.3 * d)))            # compensate


class LoFi:
    """A phone line: the band above 3.4 kHz gone, 8 kHz sample-and-hold
    (the decimation phase carries across blocks), 8-bit mu-law companding."""

    def __init__(self, sr, rate=8000.0):
        self.sr = int(sr)
        self.step = self.sr / rate                           # 6 at 48 kHz, set per block
        self.lp = LowpassFilter(cutoff_frequency_hz=3400.0)
        self.phase = 0.0                                     # next sample to hold
        self.held = np.zeros((2, 1), np.float32)

    def reset(self):
        self.lp.reset()
        self.phase = 0.0
        self.held[:] = 0.0

    def process(self, x, rate=8000.0, bits=8):
        """The line at `rate` Hz of sample-and-hold and `bits` of mu-law."""
        bs = x.shape[1]
        self.step = self.sr / max(100.0, float(rate))
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
        q = float(2 ** (max(2, int(bits)) - 1) - 1)                      # 127 at 8 bits
        c = np.sign(out) * np.log1p(mu * np.abs(out)) / np.log1p(mu)   # compress
        c = np.round(c * q) / q                                          # the word
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


# the effects' settings, one for every channel's rack (the Audio FX tab's
# block under the racks): what the knob in the grid does not say. A row:
# (ident, effect key, label, kind, lo, hi, default, suffix, what); kind
# "int" a whole number, "dbl" one decimal, "list" a menu, lo the (name,
# value) pairs and default a name. FX_SET holds the values (a menu's as its
# number), set from the GUI thread and read per block.
DELAY_TIMES = [("1/32", 1 / 32), ("1/16", 1 / 16), ("1/12", 1 / 12), ("1/8", 1 / 8),
               ("3/16", 3 / 16), ("1/4", 1 / 4), ("3/8", 3 / 8), ("1/2", 1 / 2),
               ("1 bar", 1.0)]
REV_TIMES = [("1/64", 1 / 64), ("1/32", 1 / 32), ("1/16", 1 / 16), ("1/8", 1 / 8),
             ("1/4", 1 / 4), ("1/2", 1 / 2), ("1 bar", 1.0)]
FX_SETTINGS = [
    ("hp_res", "highpass", "resonance", "int", 0, 100, 60, " %",
     "how much the ladder rings at its cutoff with the knob up; 0 none, 100 a whistle"),
    ("hp_drive", "highpass", "drive", "dbl", 1.0, 10.0, 1.0, "",
     "the ladder's input gain: 1 clean, up to 10 for a filter that saturates"),
    ("lp_res", "lowpass", "resonance", "int", 0, 100, 75, " %",
     "how much the ladder rings at its cutoff with the knob up; 0 none, 100 a whistle"),
    ("lp_drive", "lowpass", "drive", "dbl", 1.0, 10.0, 1.0, "",
     "the ladder's input gain: 1 clean, up to 10 for a filter that saturates"),
    ("drv_gain", "drive", "gain", "int", 6, 40, 30, " dB",
     "the push into the shaper with the knob at full: 6 a warm edge, 40 a fuzz"),
    ("drv_tone", "drive", "tone", "dbl", 1.0, 16.0, 8.0, " kHz",
     "a lowpass on the driven sound: low dark and woolly, high open and fizzy"),
    ("crush_bits", "crush", "bits at full", "int", 1, 12, 3, " bits",
     "the bit depth the knob reaches at full, down from 16: 1 a square buzz, 8 a sampler"),
    ("lofi_rate", "lofi", "rate", "dbl", 2.0, 16.0, 8.0, " kHz",
     "the sample-and-hold rate of the phone line: 2 kHz a growl of aliasing, 8 a phone"),
    ("lofi_bits", "lofi", "bits", "int", 3, 12, 8, " bits",
     "the mu-law word: 3 crackles, 8 a phone, 12 nearly clean"),
    ("cho_rate", "chorus", "rate", "dbl", 0.1, 5.0, 0.8, " Hz",
     "the modulation's speed: slow a wide shimmer, fast a wobble"),
    ("cho_delay", "chorus", "delay", "dbl", 1.0, 20.0, 7.0, " ms",
     "the centre delay: short a flange, long a doubling"),
    ("cho_fb", "chorus", "feedback", "int", 0, 90, 0, " %",
     "the delayed sound fed back: 0 a chorus, high a flanger's metal"),
    ("pha_rate", "phaser", "rate", "dbl", 0.1, 5.0, 0.6, " Hz",
     "the sweep's speed: slow a slow swirl, fast a tremolo"),
    ("pha_centre", "phaser", "centre", "int", 200, 5000, 1300, " Hz",
     "the centre of the sweep: low in the body, high in the air"),
    ("pitch_range", "pitch", "range", "int", 1, 24, 12, " st",
     "the semitones at the knob's ends: 12 an octave each way, 1 a fine detune, 24 two"),
    ("dly_time", "delay", "time", "list", DELAY_TIMES, None, "3/16", "",
     "the delay's time as a fraction of the bar, following the tempo: 3/16 the dotted "
     "eighth, 1/12 a triplet, 1 bar a whole one"),
    ("dly_fb", "delay", "feedback", "int", 0, 95, 45, " %",
     "the echo fed back: 0 one repeat, 95 % a long tail"),
    ("rev_time", "reverse", "time", "list", REV_TIMES, None, "1/4", "",
     "the reverse delay's time as a fraction of the bar, following the tempo: that much "
     "sound is played backwards that much late (a beat by default; a 64th is a buzz at "
     "the grain of the tempo, a bar a whole phrase turned round)"),
    ("rev_fb", "reverse", "feedback", "int", 0, 95, 50, " %",
     "the reversed echo's share fed back into the line, so the repeats alternate "
     "backwards and forwards; 0 one echo, 95 % a long tail (soft-clipped, it cannot run "
     "away)"),
    ("rvb_size", "reverb", "size", "int", 0, 100, 65, " %",
     "the room: small a closet, 100 a hall that hardly decays"),
    ("rvb_damp", "reverb", "damping", "int", 0, 100, 50, " %",
     "how fast the highs die in the tail: 0 bright and metallic, 100 dark and soft"),
    ("rvb_width", "reverb", "width", "int", 0, 100, 100, " %",
     "the stereo spread of the tail: 0 mono, 100 wide"),
]
FX_SET = {ident: (dict(lo)[default] if kind == "list" else float(default))
          for ident, _fx, _l, kind, lo, _hi, default, _s, _w in FX_SETTINGS}


class ReverseDelay:
    """The typical reverse delay: the input is cut into chunks a delay
    time D long (FX_SET["rev_time"] of the bar, a beat by default, following
    the tempo) and each chunk is played
    back backwards during the next D, so what went in during a beat is
    heard a beat later, reversed, its tail first (a 10 ms fade at each
    chunk's ends keeps the seam clean). The wet output is fed back into
    the line, so the second echo is the reversed echo reversed again,
    forwards, two beats late, the third backwards once more, the pedal's
    alternating repeats; the fed-back share (FX_SET["rev_fb"], a percent) is
    soft-clipped so it cannot run away. A change of D (a tempo change, the
    row) crossfades over the block it lands in."""

    def __init__(self, sr):
        self.sr = int(sr)
        self.n = 1 << int(8.5 * sr).bit_length()             # ring, > 2 x 4 s + a block
        self.ring = np.zeros((2, self.n), np.float32)
        self.t = 0                                             # frames written in all
        self._D = None                                         # the chunk, frames
        self._fb = 0.0

    def reset(self):
        self.ring[:] = 0.0
        self._D = None

    def _read(self, t0, bs, D):
        """The reversed read for frames t0..t0+bs at chunk length D: each
        frame plays the previous chunk of the D grid backwards, faded at
        the chunk's ends."""
        t = t0 + np.arange(bs)
        c = t // D                                             # the window's chunk
        p = t - c * D                                          # 0..D-1 into it
        src = (2 * c * D - 1 - t) % self.n                     # the chunk before, backwards
        f = max(1, min(int(0.01 * self.sr), D // 8))
        g = (np.minimum(np.minimum(p, D - 1 - p), f) / f).astype(np.float32)
        return self.ring[:, src] * g

    def process(self, x, a, beat_s):
        """(2, bs) in, the fully wet echoes (2, bs) out (the amount `a` is
        the rack's mix), the chunk FX_SET["rev_time"] of a bar of `beat_s`."""
        bs = x.shape[1]
        secs = float(FX_SET.get("rev_time", 0.25)) * 4.0 * float(beat_s)
        D = int(round(max(0.01, min(4.0, secs)) * self.sr))   # 10 ms: a buzz, up to a bar
        t0 = self.t
        idx = (t0 + np.arange(bs)) % self.n
        self.ring[:, idx] = x                                  # written first: the seam reads
        if self._D is None:                                    # this block's own frames
            self._D = D
        D0 = self._D
        fb = max(0.0, min(0.95, float(FX_SET.get("rev_fb", 50.0)) / 100.0))
        xf = np.linspace(0.0, 1.0, bs, dtype=np.float32) if D != D0 else None
        # a chunk boundary inside the block: the frames after it read the
        # frames before it (the seam), so those get their echo written
        # first; the block goes in pieces at every boundary of either grid
        cuts = sorted({c for DD in {D0, D}
                       for c in range(-(-t0 // DD) * DD, t0 + bs, DD) if t0 < c < t0 + bs})
        y = np.empty((2, bs), np.float32)
        a0 = 0
        for cut in cuts + [t0 + bs]:
            a1 = cut - t0
            seg = self._read(t0 + a0, a1 - a0, D0)
            if xf is not None:
                seg = seg * (1.0 - xf[a0:a1]) + self._read(t0 + a0, a1 - a0, D) * xf[a0:a1]
            y[:, a0:a1] = seg
            self.ring[:, idx[a0:a1]] += np.tanh(seg * fb).astype(np.float32)   # the echo's echo
            a0 = a1
        self._D = D
        self.t = t0 + bs
        return y


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

    @property
    def idle(self):
        """No effect on and none fading out: zeros in would be zeros out,
        so a silent channel can skip the rack."""
        return not any(self._now.values()) and not any(self.amounts.values())

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
                self._delay_s = None                             # set on the first block
            elif key == "reverse":
                p = ReverseDelay(sr)
            elif key == "reverb":
                # wet_level 1 is Freeverb's +3.5 dB on noise and +8 dB on a kick
                # pattern (the tail fills the gaps): 0.3 puts the full wash
                # about 3 dB under the dry
                p = Reverb(room_size=0.6, damping=0.5, wet_level=0.3, dry_level=0.0, width=1.0)
            self._plug[key] = p
        return p

    def _wet(self, key, x, a):
        """The fully wet signal of effect `key` at amount `a` (0..1, or
        -1..1 for pitch), parameters following the knob."""
        sr = self.sr
        p = self._get(key)
        s = FX_SET
        if key == "highpass":
            p.cutoff_hz = 20.0 * (8000.0 / 20.0) ** a
            p.resonance = 0.1 + 0.8 * (s["hp_res"] / 100.0) * a
            p.drive = max(1.0, s["hp_drive"])
            return p.process(x, sr, reset=False)
        if key == "lowpass":
            p.cutoff_hz = 20000.0 * (150.0 / 20000.0) ** a
            p.resonance = 0.1 + 0.8 * (s["lp_res"] / 100.0) * a
            p.drive = max(1.0, s["lp_drive"])
            return p.process(x, sr, reset=False)
        if key == "drive":
            p.cutoff_frequency_hz = max(200.0, s["drv_tone"] * 1000.0)
            return p.process(_tanh_drive(x, a, s["drv_gain"]), sr, reset=False)
        if key == "crush":
            p.bit_depth = 16.0 - (16.0 - max(1.0, s["crush_bits"])) * a
            return p.process(x, sr, reset=False)
        if key == "lofi":
            return p.process(x, s["lofi_rate"] * 1000.0, int(s["lofi_bits"]))
        if key == "chorus":
            p.depth = 0.2 + 0.7 * a
            p.rate_hz = s["cho_rate"]
            p.centre_delay_ms = s["cho_delay"]
            p.feedback = s["cho_fb"] / 100.0
            return p.process(x, sr, reset=False)
        if key == "phaser":
            p.depth = 0.3 + 0.7 * a
            p.feedback = 0.2 + 0.5 * a
            p.rate_hz = s["pha_rate"]
            p.centre_frequency_hz = s["pha_centre"]
            return p.process(x, sr, reset=False)
        if key == "pitch":
            return p.process(x, s["pitch_range"] * a)
        if key == "delay":
            want = max(0.02, min(4.0, s["dly_time"] * 4.0 * self.beat_s))
            if self._delay_s is None or abs(want - self._delay_s) > 1e-3:
                p.delay_seconds = want
                self._delay_s = want
            p.feedback = max(0.0, min(0.95, s["dly_fb"] / 100.0))
            return p.process(x, sr, reset=False)
        if key == "reverse":
            return p.process(x, a, self.beat_s)
        if key == "reverb":
            p.room_size = s["rvb_size"] / 100.0
            p.damping = s["rvb_damp"] / 100.0
            p.width = s["rvb_width"] / 100.0
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
            elif key == "reverb":                          # squared: the first half of the
                w_t, w_n = target * target, now * now      # knob a room, not a wash
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
    print(f"all eleven on: {ms:.2f} ms per {bs}-frame block ({1000 * bs / sr:.1f} ms of audio)")
