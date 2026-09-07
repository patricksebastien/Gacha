#!/usr/bin/env python3
"""
gacha_engine.py — generative sample masher (render engine).

Loads the samples it is given, carves drum hits (kick / snare / closed
and open hihat / ride)
out of randomly chosen samples, sequences a steady-but-experimental
pattern, layers looped / reversed / chopped background textures, runs
everything through randomized pedalboard effect chains (reverb, delay,
distortion, chorus, phaser, bitcrush...), and renders a ~3 minute song.

Every run is different. Every sample is used at least once.

This module has no command line of its own: the GUI (gacha.py) writes
a JSON job file and runs ``python3 gacha_engine.py job.json`` in a subprocess,
streaming the printed log. See run_job() for the job format.
"""

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, resample_poly, sosfilt
from pedalboard import (
    Pedalboard, Reverb, Delay, Distortion, Chorus, Phaser, Compressor,
    Limiter, Gain, HighpassFilter, LowpassFilter, PitchShift, Bitcrush,
    LadderFilter, GSMFullRateCompressor, MP3Compressor,
)

# subprocesses: ffmpeg from PATH, else the static build imageio-ffmpeg
# installs with pip (handy on Windows); no console window flashes when the
# app runs under pythonw.exe
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"                 # let the call fail with a clear error


SR = 44100
SAMPLES_DIR = Path(__file__).parent / "samples"
OUT_DIR = Path(__file__).parent / "output"
VIDEO_AUDIO_DIR = Path(__file__).parent / "videos" / ".audio"   # extracted tracks

DEFAULT_SECTION_BARS = [4, 4, 8, 8, 16]
RANDOM_START_MODES = ["off", "one-shots", "all"]
SYNTH_STYLES = ["auto", "house", "techno", "dnb"]

# one of these goes into every filename, picked from the seed so a
# reproduced run gets the same name
WORDS = """
amber ash aster basalt bloom brass cinder cobalt comet coral crow dune dusk
ember fern flint fog frost garnet ghost glass gravel gust halo haze hollow
indigo iris ivory jade karma kelp kite lantern lava lilac lotus lunar maple
marble mist moss neon nettle nova oak ocean onyx opal orbit oxide paper pearl
pepper petal pine plum prism pulse quartz quill rain raven reed relic rust
saffron sage salt satin shadow silk slate smoke snow sonar spark static
storm sugar sulfur tape thistle thorn tide tin topaz totem tundra umber
velvet vapor violet wax willow wire wren zenith zinc
""".split()


@dataclass
class Knobs:
    """Every tunable that used to be a hard-coded number inside compose().
    Probabilities are 0..1; repeated values in section_bars act as weights."""
    # drums
    swing: float | None = None        # fraction of a 16th; None = random 0-0.06
    humanize: float = 0.5             # velocity + timing looseness, 0 = machine
    mutation_chance: float = 0.3      # per bar per role: pattern gets mutated
    mutation_amount: float = 0.1      # fraction of steps a mutation touches
    glitch_chance: float = 0.15       # per bar: hihat stutter burst
    gain_kick: float = 0.95
    gain_snare: float = 0.8
    gain_hihat: float = 0.45          # closed hat
    gain_ohat: float = 0.4            # open hat, carved from the same source
    gain_ride: float = 0.3
    synth_kick: float = 0.0           # synthesized kit, layered on the same
    synth_snare: float = 0.0          # patterns as the sample hits; 0 = off
    synth_hihat: float = 0.0
    synth_ohat: float = 0.0
    synth_ride: float = 0.0
    synth_style: str = "auto"         # house / techno / dnb, auto = from drum style
    sub_level: float = 0.6            # synth sub under the kick, 0 = off
    sub_phaser: float = 0.5           # phaser depth on the left oscillator
    sub_pan: float = 0.3              # dry osc to the right, phased to the left
    sub_chance: float = 0.6           # per bar: a sub note on its first kick
    sub_release: float = 1.5          # seconds of release after a one-beat hold
    # sections
    break_chance: float = 0.25        # a middle section becomes a break
    section_bars: list = field(
        default_factory=lambda: list(DEFAULT_SECTION_BARS))
    fade_in: float = 0.3              # seconds, on the whole song
    fade_out: float = 0.03            # seconds; default is just a click guard
    outro_tail: float | None = None   # seconds of ring-out; None = per-style default
    outro_bars: float = 2.0           # the effect blends in over the last N bars
    outro_wet: float = 1.0            # max wetness of the ending effect
    outro_amount: float = 0.5         # per-style primary knob (decay/feedback/crush)
    cymbal_chance: float = 0.3        # per drop: reverse cymbal swell into it
    repeat_chance: float = 0.25       # per drop: DJ beat-repeat on the last bar
    gap_chance: float = 0.0           # per drop: short silence right before it (off)
    # layers / fx
    layers_min: int = 1               # background textures per groove section
    layers_max: int = 3
    layer_level_min: float = 0.18     # texture gain range
    layer_level_max: float = 0.4
    chop_chance: float = 0.6          # per non-intro section: add a chop layer
    chop_gain: float = 0.4
    fx_min: int = 1                   # random effects per texture chain
    fx_max: int = 3
    reverse_chance: float = 0.35      # texture is played backwards
    stretch_chance: float = 0.3       # texture is time-stretched to bars
    saturation: float = 0.5           # drive of the texture saturator, 0 = off
    random_start: str = "off"         # cut from a random offset instead of 0:
                                      # "off", "one-shots", "all" (textures and
                                      # collage hits too)

    def sanitized(self):
        """Clamp probabilities, order ranges, drop nonsense."""
        k = Knobs(**asdict(self))
        for name in ("humanize", "mutation_chance", "mutation_amount", "glitch_chance",
                     "break_chance", "chop_chance", "reverse_chance",
                     "stretch_chance", "saturation", "cymbal_chance",
                     "repeat_chance", "gap_chance", "sub_level", "sub_phaser",
                     "sub_pan", "sub_chance", "outro_wet", "outro_amount"):
            setattr(k, name, min(1.0, max(0.0, getattr(k, name))))
        for name in ("gain_kick", "gain_snare", "gain_hihat", "gain_ohat",
                     "gain_ride", "synth_kick", "synth_snare", "synth_hihat",
                     "synth_ohat", "synth_ride",
                     "chop_gain", "layer_level_min", "layer_level_max",
                     "fade_in", "fade_out", "sub_release"):
            setattr(k, name, max(0.0, getattr(k, name)))
        if k.swing is not None:
            k.swing = min(0.5, max(0.0, k.swing))
        if k.outro_tail is not None:
            k.outro_tail = min(12.0, max(0.5, k.outro_tail))
        k.outro_bars = max(0.25, k.outro_bars)
        k.layers_min = max(0, k.layers_min)
        k.layers_max = max(k.layers_min, k.layers_max)
        k.layer_level_min, k.layer_level_max = sorted(
            (k.layer_level_min, k.layer_level_max))
        k.fx_min = max(1, k.fx_min)
        k.fx_max = max(k.fx_min, k.fx_max)
        if k.synth_style not in SYNTH_STYLES:
            k.synth_style = "auto"
        if k.random_start is True:                 # older job files used a bool
            k.random_start = "all"
        if k.random_start not in RANDOM_START_MODES:
            k.random_start = "off"
        k.section_bars = [int(b) for b in k.section_bars if int(b) >= 1] \
            or list(DEFAULT_SECTION_BARS)
        return k


# ---------------------------------------------------------------- loading

AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".mp3")


def sample_name(path):
    """Display / bank key for a sample: its path relative to samples/
    (so two sets can both hold a hit_bell.wav), or the bare filename
    for files living elsewhere."""
    path = Path(path)
    try:
        return path.relative_to(SAMPLES_DIR).as_posix()
    except ValueError:
        return path.name


def extract_video_audio(videos):
    """The audio tracks of video files as 44.1 kHz stereo wavs, cached under
    videos/.audio (keyed by path, size and mtime), so a video can be sample
    material. Returns {wav sample_name: video path}; videos without an
    audio track, or that ffmpeg cannot read, are skipped with a note."""
    out = {}
    for v in videos:
        v = Path(v)
        if not v.is_file():
            continue
        st = v.stat()
        key = hashlib.sha1(f"{v.resolve()}|{st.st_size}|{st.st_mtime_ns}"
                           .encode()).hexdigest()[:8]
        wav = VIDEO_AUDIO_DIR / f"{v.stem}_{key}.wav"
        if not wav.is_file():
            VIDEO_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            try:
                r = subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(v), "-vn",
                                    "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le",
                                    str(wav)], capture_output=True, text=True,
                                   errors="replace", **NO_WINDOW)
            except OSError as e:                            # no ffmpeg at all
                print(f"  ! ffmpeg not found ({e}): install ffmpeg or pip install imageio-ffmpeg")
                break
            if r.returncode or not wav.is_file() or wav.stat().st_size < 1024:
                wav.unlink(missing_ok=True)
                print(f"  ! no audio from {v.name}: "
                      f"{(r.stderr or 'no audio track').strip().splitlines()[-1]}")
                continue
        out[sample_name(wav)] = v
    return out


def load_samples(files):
    """Load the given audio files as float32 stereo (frames, 2) at
    44.1 kHz, keyed by sample_name(). Order is sorted by key so a seed
    reproduces the same song for the same selection."""
    bank = {}
    files = sorted((Path(f) for f in files), key=sample_name)
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
        bank[sample_name(f)] = np.ascontiguousarray(data[:, :2])
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


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def saturate(clip, amount, bias=0.15, tone_hz=8000.0):
    """Gain-compensated soft saturation.

    amount 0..1 sets the drive (up to +24 dB into the shaper). The wave-
    shaper runs 2x oversampled to keep aliasing down, `bias` skews it for
    even harmonics (tube/tape flavour), `tone_hz` rolls off the fizz, and
    the result is matched back to the input RMS so it never gets louder,
    only denser."""
    if amount <= 0 or len(clip) == 0:
        return clip
    in_rms = rms(clip)
    if in_rms == 0:
        return clip
    drive = 10 ** (amount * 24 / 20)
    up = resample_poly(clip, 2, 1, axis=0)
    shaped = np.tanh(up * drive + bias) - np.tanh(bias)
    down = resample_poly(shaped, 1, 2, axis=0)[: len(clip)].astype("float32")
    post = Pedalboard([HighpassFilter(cutoff_frequency_hz=20),
                       LowpassFilter(cutoff_frequency_hz=float(tone_hz))])
    down = post(down.T, SR).T
    out_rms = rms(down)
    if out_rms > 0:
        down *= in_rms / out_rms
    return down.astype("float32")


class Saturator:
    """A saturate() stage that can sit inside an FxChain."""

    def __init__(self, amount, bias, tone_hz):
        self.amount, self.bias, self.tone_hz = amount, bias, tone_hz

    def __call__(self, audio, sr=SR):          # (channels, frames) like pedalboard
        return saturate(audio.T, self.amount, self.bias, self.tone_hz).T


class FxChain:
    """Ordered stages, pedalboard plugins or Saturators, called like a
    Pedalboard: chain(audio_channels_first, sample_rate)."""

    def __init__(self, stages):
        self.stages = list(stages)

    def __contains__(self, stage):
        return any(s is stage for s in self.stages)

    def __call__(self, audio, sr):
        for stage in self.stages:
            audio = stage(audio, sr)
        return audio


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


def random_window(rng, clip, need, enabled):
    """The clip from a random start offset onwards, keeping at least `need`
    samples so the cut that follows still has material. With enabled=False
    the clip is returned untouched (cuts start at 0, the historic behaviour)."""
    if not enabled:
        return clip
    need = min(len(clip), int(need))
    off = rng.randrange(0, len(clip) - need + 1) if len(clip) > need else 0
    return clip[off:]


# ---------------------------------------------------------------- outro
# Ported from Marmelade's per-keeper "Ending FX" stage: the song body stays
# dry, the effect ramps in over the last bars, then rings out into an
# appended tail that ends in true silence.

OUTRO_STYLES = ["hall_wash", "dub_echo", "tape_stop", "filter_close",
                "shimmer_freeze", "bitcrush_collapse", "codec_rot",
                "glitch_stutter", "overdrive_bloom", "smear"]

OUTRO_TAIL = {                      # default ring-out seconds per style
    "hall_wash": 6.0, "dub_echo": 5.0, "tape_stop": 2.5, "filter_close": 3.0,
    "shimmer_freeze": 7.0, "bitcrush_collapse": 3.5, "codec_rot": 3.0,
    "glitch_stutter": 3.0, "overdrive_bloom": 4.0, "smear": 5.0,
}


def outro_board(style, amount, beat):
    """Pedalboard chain for an outro style, `amount` 0..1 = its primary knob.
    Delays are locked to the tempo grid, gacha-style."""
    p = min(1.0, max(0.0, amount))
    if style == "hall_wash":
        return [Reverb(room_size=0.85 + 0.14 * p, wet_level=0.6, dry_level=0.4,
                       width=1.0, damping=0.3)]
    if style == "dub_echo":
        return [Delay(delay_seconds=0.75 * beat, feedback=0.55 + 0.4 * p, mix=0.5),
                Reverb(room_size=0.5, wet_level=0.3, dry_level=0.7)]
    if style == "filter_close":
        return [LadderFilter(mode=LadderFilter.Mode.LPF24,
                             cutoff_hz=12000.0 * (1.0 - 0.9 * p) + 200.0,
                             resonance=0.3 + 0.4 * p, drive=1.0)]
    if style == "shimmer_freeze":
        return [PitchShift(semitones=12.0),
                Reverb(room_size=0.95, wet_level=0.7, dry_level=0.3, damping=0.1)]
    if style == "bitcrush_collapse":
        return [Bitcrush(bit_depth=max(1, int(round(8 - 6 * p)))),
                Reverb(room_size=0.6, wet_level=0.4, dry_level=0.6)]
    if style == "codec_rot":
        return [GSMFullRateCompressor(), MP3Compressor(vbr_quality=2.0 + 7.0 * p)]
    if style == "glitch_stutter":
        return [Delay(delay_seconds=beat / 8, feedback=0.7 + 0.25 * p, mix=0.6),
                Distortion(drive_db=6.0)]
    if style == "overdrive_bloom":
        return [Distortion(drive_db=12.0 + 18.0 * p),
                Reverb(room_size=0.7, wet_level=0.5, dry_level=0.5)]
    if style == "smear":
        return [Chorus(rate_hz=0.4, depth=0.9, mix=0.7),
                Reverb(room_size=0.8, wet_level=0.6, dry_level=0.4, damping=0.5)]
    return []


def tape_stop(src, out_len, amount):
    """A real tape stop: the last stretch of the song is replayed at a speed
    that decelerates to zero, so the pitch drops with it. `amount` bends the
    curve (0 = linear glide, 1 = quick slump). Returns out_len frames."""
    n_src = len(src)
    out = np.zeros((out_len, 2), dtype="float32")
    if n_src < 2 or out_len < 2:
        return out
    T = min(out_len, 2 * n_src)                 # frames until the reel halts
    t = np.arange(T, dtype="float64")
    speed = (1.0 - t / T) ** (0.6 + 1.6 * amount)
    pos = np.clip(np.cumsum(speed), 0, n_src - 1)
    for c in range(2):
        out[:T, c] = np.interp(pos, np.arange(n_src), src[:, c])
    out[:T] *= np.sqrt(speed)[:, None]          # dim as it slows
    return out


def apply_outro(song, style, beat, tail_sec=None, onset_n=None, wet=1.0,
                amount=0.5):
    """Append a ring-out tail to the song. Body stays dry, the effect ramps in
    over the last onset_n frames, the tail is fully wet, and a safety fade
    guarantees true silence at the very end."""
    if style not in OUTRO_STYLES:
        return song
    tail_n = int((tail_sec if tail_sec is not None else OUTRO_TAIL[style]) * SR)
    body_len = len(song)
    onset_n = min(body_len, onset_n or int(2 * 4 * beat * SR))
    padded = np.vstack([song, np.zeros((tail_n, 2), dtype="float32")])

    if style == "tape_stop":
        stopped = tape_stop(song[body_len - onset_n:], onset_n + tail_n, amount)
        wet_out = np.vstack([song[: body_len - onset_n], stopped])
        ramp_n = min(onset_n, int(0.05 * SR))   # hard cut into the slowdown
    else:
        board = Pedalboard(outro_board(style, amount, beat))
        wet_out = board(padded.T, SR).T.astype("float32")
        ramp_n = onset_n

    n = max(len(padded), len(wet_out))          # codecs may shift the length
    if len(padded) < n:
        padded = np.vstack([padded, np.zeros((n - len(padded), 2), "float32")])
    if len(wet_out) < n:
        wet_out = np.vstack([wet_out, np.zeros((n - len(wet_out), 2), "float32")])

    env = np.zeros(n, dtype="float32")
    if ramp_n > 0:
        env[body_len - ramp_n:body_len] = np.linspace(0, 1, ramp_n)
    env[body_len:] = 1.0
    e = (wet * env)[:, None]
    out = (1 - e) * padded + e * wet_out

    # safety fade: gentle 50 ms, then a steep 10 ms, last sample exactly 0
    for sec in (0.05, 0.01):
        f = min(int(sec * SR), n)
        out[-f:] *= np.linspace(1, 0, f)[:, None]
    out[-1] = 0.0
    return out.astype("float32")


# ---------------------------------------------------------------- sub bass

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def section_root(tonal_seg):
    """Dominant pitch class of a tonal segment via chroma, or None if the
    segment is essentially silent."""
    m = mono(tonal_seg)
    if len(m) < 4096 or rms(m) < 1e-4:
        return None
    chroma = librosa.feature.chroma_cqt(y=m.astype("float32"), sr=SR,
                                        hop_length=2048)
    return int(np.argmax(chroma.mean(axis=1)))


SUB_LO, SUB_HI = 24, 35     # C1 33 Hz .. B1 62 Hz: sub-bass, never a "note"


def sub_midi(pc):
    """Pitch class -> MIDI note inside the sub octave."""
    return SUB_LO + pc


def adsr(n, attack=0.02, decay=0.15, sustain=0.7, release=1.5):
    """Amplitude envelope, segment lengths in seconds, squeezed to fit n.
    The release is an exponential-ish tail so long values fade naturally."""
    a, d, r = (int(x * SR) for x in (attack, decay, release))
    if a + d + r > n:
        scale = n / (a + d + r)
        a, d, r = (max(1, int(x * scale)) for x in (a, d, r))
    env = np.full(n, sustain, dtype="float32")
    env[:a] = np.linspace(0, 1, a)
    env[a:a + d] = sustain + (1 - sustain) * np.exp(
        -5 * np.linspace(0, 1, d))
    env[n - r:] *= np.linspace(1, 0, r) ** 2
    return env


def render_sub(hits, n_total, phaser_amt, pan, release):
    """hits: (pos, midi, length_samples). Two sine oscillators at the same
    pitch, ADSR-shaped: the dry one panned right, the other through a slow
    phaser and panned left by `pan`. Returns a stereo buffer, not yet
    level-matched."""
    osc = np.zeros(n_total, dtype="float32")
    for pos, midi, n in hits:
        n = min(n, n_total - pos)
        if n <= 0:
            continue
        f = 440.0 * 2 ** ((midi - 69) / 12)
        t = np.arange(n) / SR
        osc[pos:pos + n] += np.sin(2 * np.pi * f * t) * adsr(n, release=release)
    phased = osc
    if phaser_amt > 0:
        phased = Pedalboard([Phaser(rate_hz=0.2, depth=phaser_amt,
                                    feedback=0.3 * phaser_amt,
                                    mix=0.5 * phaser_amt)])(osc[None, :], SR)[0]
    to_stereo = lambda x: np.repeat(x[:, None], 2, axis=1)
    out = panned(to_stereo(osc), pan) * 0.5 \
        + panned(to_stereo(phased), -pan) * 0.5
    return out.astype("float32")


# ---------------------------------------------------------------- transitions

def filter_sweep(clip, f_start, f_end, n_chunks=16, highpass=False):
    """Chunked filter whose cutoff glides geometrically from f_start to
    f_end over the clip. Each chunk is filtered with a little context so
    the filter has settled by the time the chunk starts."""
    n = len(clip)
    if n == 0:
        return clip
    out = clip.copy()
    step = max(1, n // n_chunks)
    cutoffs = np.geomspace(f_start, f_end, n_chunks)
    Filt = HighpassFilter if highpass else LowpassFilter
    for i in range(n_chunks):
        a = i * step
        b = n if i == n_chunks - 1 else min(n, a + step)
        if a >= b:
            break
        pre = max(0, a - 2048)
        seg = Pedalboard([Filt(cutoff_frequency_hz=float(cutoffs[i]))])(
            clip[pre:b].T, SR).T
        out[a:b] = seg[a - pre:]
    return out


def cymbal_swell(rng, hit, length):
    """Reverse-reverb cymbal: a huge reverb on the ride hit, played
    backwards so the wash builds and the hit lands on the drop, under a
    lowpass that opens up as it rises."""
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=rng.uniform(1500, 4000)),
        Reverb(room_size=0.95, damping=0.3, wet_level=0.8, dry_level=0.3,
               width=1.0),
    ])
    wet = apply_fx(hit, board, tail=length / SR + 0.5)[:length]
    if len(wet) < length:
        wet = np.vstack([wet, np.zeros((length - len(wet), 2), "float32")])
    swell = wet[::-1].copy()
    swell *= (np.linspace(0.0, 1.0, length) ** 1.5)[:, None]
    swell = filter_sweep(swell, rng.uniform(400, 900), 14000, n_chunks=24)
    peak = np.abs(swell).max()
    return swell / peak * 0.9 if peak > 0 else swell


def beat_repeat(rng, buf, start, bar_n, beat_n):
    """DJ-console beat repeat: capture the start of the bar at `start` and
    retrigger it over the bar, replacing what was there. Three flavours:
    accelerating roll, steady 16th stutter, or halving beat by beat."""
    seg = buf[start:start + bar_n].copy()
    if len(seg) < bar_n:
        return None
    flavour = rng.choice(["accelerate", "accelerate", "stutter", "halving"])
    if flavour == "accelerate":
        divs = [2, 4, 8, 16]                 # per beat: 1/8, 1/16, 1/32, 1/64
    elif flavour == "halving":
        divs = [1, 2, 4, 8]
    else:
        divs = [4, 4, 4, 4]
    fade_n = int(0.002 * SR)
    out = np.zeros_like(seg)
    pos = 0
    for div in divs:
        L = max(1, beat_n // div)
        piece = seg[:L].copy()
        f = min(fade_n, L // 4)
        if f > 0:
            piece[:f] *= np.linspace(0, 1, f)[:, None]
            piece[-f:] *= np.linspace(1, 0, f)[:, None]
        for _ in range(div):
            end = min(bar_n, pos + L)
            if end <= pos:
                break
            out[pos:end] = piece[: end - pos]
            pos = end
    if rng.random() < 0.5:                   # tension: highpass climbs
        out = filter_sweep(out, 40, rng.uniform(800, 2500), n_chunks=16,
                           highpass=True)
    buf[start:start + bar_n] = out
    return flavour


# ---------------------------------------------------------------- drum kit

def grain_cloud(rng, clip, n, grains, grain_ms=(2.0, 8.0)):
    """Dense cloud of tiny Hann-windowed grains picked from anywhere in the
    sample and scattered over n frames. Turns any material into a noisy
    wash that keeps the sample's colour: the trick that makes a flute or a
    voice pass for cymbal sizzle."""
    out = np.zeros((n, 2), dtype="float32")
    if len(clip) < 8:
        return out
    for _ in range(grains):
        g = min(len(clip), int(rng.uniform(*grain_ms) / 1000 * SR))
        pos = rng.randrange(0, len(clip) - g + 1)
        at = rng.randrange(0, max(1, n - g))
        grain = clip[pos:pos + g] * np.hanning(g)[:, None]
        end = min(n, at + g)
        out[at:end] += grain[: end - at]
    peak = np.abs(out).max()
    return out / peak if peak > 0 else out


def carve_drum(rng, clip, role):
    """Cut a short chunk from any sample and sculpt it into a drum hit."""
    start = strongest_onset(clip)
    if role in ("hihat", "ohat"):
        return carve_hat(rng, clip, start, open_=(role == "ohat"))
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
    n = int(length * SR)
    chunk = clip[start:start + n]
    if len(chunk) < n:
        chunk = np.vstack([chunk, np.zeros((n - len(chunk), 2), dtype="float32")])
    decay = {"kick": 3.5, "snare": 5.0, "ride": 2.0}[role]
    chunk = chunk * envelope(len(chunk), decay_curve=decay)
    hit = apply_fx(chunk, board, tail=0.3)
    peak = np.abs(hit).max()
    if peak > 0:
        hit = hit / peak * 0.9
    return hit


def carve_hat(rng, clip, start, open_):
    """Closed or open hihat out of any sample. Both mix the transient at the
    onset with a grain cloud from the whole sample, so the hit gets a noisy
    sizzle in the sample's own colour, then a steep highpass takes the body
    away. Closed: 30-70 ms, snappy decay, a touch of drive for edge.
    Open: 150-450 ms, slow decay, denser cloud, a resonant band around
    7-10 kHz for the ring of the cymbal edge, a little drive and a small
    room. The open hat is choked by the sequencer when a closed hat
    follows, like a pedal closing."""
    if open_:
        length = rng.uniform(0.15, 0.45)
        n = int(length * SR)
        body = grain_cloud(rng, clip, n, grains=rng.randint(40, 90),
                           grain_ms=(3.0, 12.0))
        cloud_mix, decay = rng.uniform(0.6, 0.9), rng.uniform(1.4, 2.4)
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=rng.uniform(3500, 7000)),
            LadderFilter(mode=LadderFilter.Mode.BPF12,
                         cutoff_hz=rng.uniform(7000, 10000),
                         resonance=rng.uniform(0.2, 0.5), drive=1.0),
            Distortion(drive_db=rng.uniform(2, 6)),
            Reverb(room_size=0.12, wet_level=rng.uniform(0.02, 0.08),
                   dry_level=0.9),
            Gain(gain_db=4),
        ])
    else:
        length = rng.uniform(0.03, 0.07)
        n = int(length * SR)
        body = grain_cloud(rng, clip, n, grains=rng.randint(10, 20))
        cloud_mix, decay = rng.uniform(0.3, 0.6), rng.uniform(5.0, 7.0)
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=rng.uniform(5000, 9000)),
            Distortion(drive_db=rng.uniform(3, 8)),
            Gain(gain_db=3),
        ])
    chunk = clip[start:start + n]
    if len(chunk) < n:
        chunk = np.vstack([chunk, np.zeros((n - len(chunk), 2), dtype="float32")])
    peak = np.abs(chunk).max()
    if peak > 0:
        chunk = chunk / peak
    hit = chunk * (1 - cloud_mix) + body * cloud_mix
    hit = hit * envelope(len(hit), attack=0.001, decay_curve=decay)
    hit = apply_fx(hit, board, tail=0.4 if open_ else 0.15)
    peak = np.abs(hit).max()
    if peak > 0:
        hit = hit / peak * 0.9
    return hit


def choke(hit, n, fade=0.005):
    """Cut a hit to n frames with a short fade: the hat pedal closing."""
    if n >= len(hit):
        return hit
    n = max(n, int(fade * SR) + 1)
    out = hit[:n].copy()
    f = int(fade * SR)
    out[-f:] *= np.linspace(1, 0, f)[:, None]
    return out


# ---------------------------------------------------------------- synth kit
# A drum machine in the 909/808 spirit, no samples involved: sine sweeps,
# square-wave metal, filtered noise. Three flavours, each a set of numbers
# below, jittered a little per song so no two kits are identical.

SYNTH_FLAVOUR = {                     # drum style -> synth flavour (auto)
    "four-floor": "house", "ukg": "house", "minimal": "house",
    "one-drop": "house", "boom-bap": "house",
    "halftime": "techno", "idm": "techno", "dembow": "techno", "clave": "techno",
    "dnb": "dnb", "breakbeat": "dnb", "footwork": "dnb",
}


def _sos(kind, freq, order=2):
    return butter(order, freq, btype=kind, fs=SR, output="sos")


def _filt(x, kind, freq, order=2):
    return sosfilt(_sos(kind, freq, order), x).astype("float32")


def _decay(n, seconds, attack=0.0005):
    t = np.arange(n) / SR
    env = np.exp(-t / max(1e-4, seconds))
    a = max(1, int(attack * SR))
    env[:a] *= np.linspace(0, 1, a)
    f = min(n // 4, int(0.01 * SR))            # guard the very end
    if f > 0:
        env[-f:] *= np.linspace(1, 0, f)
    return env.astype("float32")


def _finish(x, drive=1.0):
    """Soft clip, normalize, stereo."""
    if drive > 1.0:
        x = np.tanh(x * drive) / np.tanh(drive)
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak * 0.9
    return np.repeat(x.astype("float32")[:, None], 2, axis=1)


def _j(rng, v, pct=0.1):
    """Jitter a number by +-pct."""
    return v * rng.uniform(1 - pct, 1 + pct)


def synth_kick(rng, flavour):
    P = {"house": dict(f0=170, f1=48, tau=0.035, dec=0.38, drive=1.6, click=0.35),
         "techno": dict(f0=230, f1=44, tau=0.022, dec=0.55, drive=3.5, click=0.25),
         "dnb": dict(f0=160, f1=56, tau=0.016, dec=0.24, drive=2.2, click=0.6)}[flavour]
    f0, f1, tau, dec = (_j(rng, P[k]) for k in ("f0", "f1", "tau", "dec"))
    n = int(min(1.2, dec * 4.5) * SR)
    t = np.arange(n) / SR
    freq = f1 + (f0 - f1) * np.exp(-t / tau)          # exponential pitch drop
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * _decay(n, dec)
    noise = np.random.default_rng(rng.randrange(2 ** 31)).standard_normal(n)
    click = _filt(noise, "highpass", 2000) * _decay(n, 0.004) * P["click"]
    x = body + click
    if flavour == "techno":                            # a little boxy rumble
        x = _filt(x, "lowpass", 6000)
    return _finish(x, P["drive"])


def synth_snare(rng, flavour):
    P = {"house": dict(tones=(185, 330), tdec=0.14, ndec=0.22, hp=1200, lp=None,
                       nmix=0.8, drive=1.3),
         "techno": dict(tones=(170, 300), tdec=0.10, ndec=0.16, hp=900, lp=7000,
                        nmix=0.7, drive=2.5),
         "dnb": dict(tones=(210, 390), tdec=0.09, ndec=0.20, hp=1800, lp=None,
                     nmix=1.0, drive=1.8)}[flavour]
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    tone = np.zeros(n, dtype="float32")
    for f in P["tones"]:                               # 909: two tones that dip
        f = _j(rng, f, 0.06)
        freq = f * (1 + 0.3 * np.exp(-t / 0.01))
        tone += np.sin(2 * np.pi * np.cumsum(freq) / SR) / len(P["tones"])
    tone *= _decay(n, _j(rng, P["tdec"]))
    g = np.random.default_rng(rng.randrange(2 ** 31))
    noise = _filt(g.standard_normal(n), "highpass", _j(rng, P["hp"]))
    if P["lp"]:
        noise = _filt(noise, "lowpass", P["lp"])
    noise *= _decay(n, _j(rng, P["ndec"]))
    return _finish(tone * 0.6 + noise * P["nmix"] * 0.5, P["drive"])


def synth_hat(rng, flavour, open_):
    """808-style: six square waves at metallic ratios through a bandpass,
    plus a little noise; the open hat is the same metal ringing longer."""
    base = (263, 400, 421, 474, 587, 845)
    scale = {"house": 1.0, "techno": 0.8, "dnb": 1.1}[flavour] * rng.uniform(0.9, 1.15)
    dec = {"house": (0.06, 0.42), "techno": (0.05, 0.32),
           "dnb": (0.045, 0.28)}[flavour][1 if open_ else 0]
    dec = _j(rng, dec)
    n = int(min(1.0, dec * 5) * SR)
    t = np.arange(n) / SR
    metal = sum(np.sign(np.sin(2 * np.pi * f * scale * t + rng.random() * 6.28))
                for f in base) / len(base)
    metal = _filt(_filt(metal, "highpass", 5500, 4), "lowpass", 11000, 4)
    g = np.random.default_rng(rng.randrange(2 ** 31))
    noise = _filt(_filt(g.standard_normal(n), "highpass", 6000), "lowpass", 12000)
    x = (metal + 0.3 * noise) * _decay(n, dec)
    return _finish(x, 1.5)


def synth_ride(rng, flavour):
    """Inharmonic partials with their own decays, a noise shimmer and a
    stick click, bandpassed into the 3-10 kHz region."""
    dec = _j(rng, {"house": 1.2, "techno": 0.9, "dnb": 0.8}[flavour])
    n = int(min(2.0, dec * 3) * SR)
    t = np.arange(n) / SR
    base = 520 * rng.uniform(0.9, 1.1)
    x = np.zeros(n, dtype="float32")
    for r, d in zip((1.0, 1.47, 2.09, 2.56, 3.01, 3.72),
                    (1.0, 0.9, 0.8, 0.7, 0.55, 0.45)):
        x += np.sin(2 * np.pi * base * r * t + rng.random() * 6.28) \
            * _decay(n, dec * d) / 6
    g = np.random.default_rng(rng.randrange(2 ** 31))
    noise = g.standard_normal(n)
    x = x + 0.25 * _filt(noise, "highpass", 5000) * _decay(n, dec * 0.6)
    x = _filt(_filt(x, "highpass", 3000), "lowpass", 10000)
    x += 0.5 * _filt(noise, "highpass", 4000) * _decay(n, 0.008)   # stick
    return _finish(x, 1.2)


def synth_kit(rng, flavour):
    return {"kick": synth_kick(rng, flavour), "snare": synth_snare(rng, flavour),
            "hihat": synth_hat(rng, flavour, False),
            "ohat": synth_hat(rng, flavour, True), "ride": synth_ride(rng, flavour)}


# ---------------------------------------------------------------- feel
# Nobody hits a drum twice at the same level. Every hit (sample or synth)
# exists in three velocity layers, and the sequencer picks one from a
# humanized velocity: soft hits are duller and shorter, hard hits bite.

def velocity_layers(hit):
    """[soft, mid, hard] versions of one hit."""
    n = len(hit)
    soft = np.stack([_filt(hit[:, c], "lowpass", 6500) for c in range(2)], axis=1)
    soft = soft * _decay(n, max(0.02, n / SR * 0.6), attack=0.0005)[:, None]
    hard = np.tanh(hit * 1.6) / np.tanh(1.6)
    peak = np.abs(hard).max()
    if peak > 0:
        hard = hard / peak * 0.95
    return [soft.astype("float32"), hit, hard.astype("float32")]


def pick_layer(layers, vel):
    return layers[0 if vel < 0.5 else 1 if vel < 0.85 else 2]


def humanize_vel(rng, vel, step, amount):
    """A pattern velocity as a human would land it: downbeats lean in,
    offbeat 16ths sit back, and every hit wobbles a little."""
    if amount <= 0:
        return vel
    accent = 0.0
    if step == 0:
        accent = 0.10
    elif step % 4 == 0:
        accent = 0.04
    elif step % 2 == 1:
        accent = -0.12
    vel = vel * (1 + rng.gauss(0, 0.14 * amount)) + accent * amount
    return min(1.15, max(0.15, vel))


def humanize_time(rng, amount):
    """Timing jitter in frames: under a millisecond at 0, about +-6 ms at 1."""
    return int(rng.gauss(0, 30 + 220 * amount))


# ---------------------------------------------------------------- patterns

DRUM_STYLES = ["four-floor", "breakbeat", "boom-bap", "halftime", "dnb",
               "minimal", "ukg", "dembow", "one-drop", "footwork", "clave",
               "idm"]


# open hat placement per style: candidate steps and the chance each one
# becomes open; (2, 6, 10, 14) are the offbeat 8ths, the classic spot
OPEN_HAT = {
    "four-floor": ((2, 6, 10, 14), 0.75), "ukg": ((2, 6, 10, 14), 0.4),
    "one-drop": ((2, 6, 10, 14), 0.5), "dembow": ((6, 14), 0.5),
    "boom-bap": ((6, 14, 15), 0.35), "breakbeat": ((2, 10, 14), 0.3),
    "halftime": ((6, 14), 0.4), "dnb": ((6, 14), 0.25),
    "minimal": ((2, 6, 10, 14), 0.15), "footwork": ((14,), 0.3),
    "clave": ((6, 14), 0.3), "idm": (tuple(range(1, 16, 2)), 0.15),
}


def open_hat_pattern(rng, hihat, style):
    """Split a hihat pattern into (closed, open): some candidate steps
    become open hats and are removed from the closed pattern. About one
    bar in five stays fully closed for contrast."""
    closed = list(hihat)
    opened = [0.0] * len(closed)
    if rng.random() < 0.2:
        return closed, opened
    steps, chance = OPEN_HAT.get(style, ((2, 6, 10, 14), 0.25))
    for st in steps:
        if st < len(closed) and rng.random() < chance:
            opened[st] = max(closed[st], rng.uniform(0.7, 1.0))
            closed[st] = 0.0
    return closed, opened


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


def random_texture_board(rng, beat, fx_min=1, fx_max=3, saturation=0.5):
    """A randomized effect chain for background layers, fx_min..fx_max
    effects deep. Returns (chain, tail_seconds) so delay echoes ring out."""
    delay_fx, delay_tail = synced_delay(rng, beat)
    pool = [
        Reverb(room_size=rng.uniform(0.4, 0.95), wet_level=rng.uniform(0.2, 0.5),
               dry_level=rng.uniform(0.4, 0.8), width=1.0),
        delay_fx,
        Chorus(rate_hz=rng.uniform(0.3, 2.0), depth=rng.uniform(0.2, 0.7),
               mix=rng.uniform(0.2, 0.5)),
        Phaser(rate_hz=rng.uniform(0.1, 1.5), mix=rng.uniform(0.2, 0.5)),
        Saturator(amount=saturation * rng.uniform(0.5, 1.0),
                  bias=rng.uniform(0.0, 0.3),
                  tone_hz=rng.uniform(3000, 10000)),
        Bitcrush(bit_depth=rng.uniform(6, 12)),
        GSMFullRateCompressor(),                      # gnarly lo-fi codec
        LadderFilter(mode=LadderFilter.Mode.LPF12,
                     cutoff_hz=rng.uniform(400, 4000),
                     resonance=rng.uniform(0.1, 0.6)),
        PitchShift(semitones=rng.choice([-12, -7, -5, 0, 5, 7, 12])),
    ]
    if saturation <= 0:
        pool = [fx for fx in pool if not isinstance(fx, Saturator)]
    rng.shuffle(pool)
    depth = rng.randint(max(1, fx_min), min(max(fx_min, fx_max), len(pool)))
    chain = pool[:depth]
    chain.append(Compressor(threshold_db=-18, ratio=3))
    tail = delay_tail if any(fx is delay_fx for fx in chain) else 2.0
    return FxChain(chain), max(2.0, tail)


# ---------------------------------------------------------------- composer

def compose(seed, duration, sample_files, intro_bars=None,
            intro_style="ambient", bpm=None, pan_drums=0.3, pan_layers=0.6,
            pan_events=0.5, num_samples=None, drum_style="random", knobs=None,
            outro_style="random", video_sources=None):
    """Render one song. Besides the audio, the meta records `events`: every
    placed sound that came from a sample, as [t, src, off, dur, rate, role]
    (song seconds, sample name, source seconds at the start of the event,
    song seconds it lasts, source seconds per song second with the sign as
    direction, and kick/snare/hihat/ohat/ride/glitch/texture/chop/swell/
    collage/oneshot). With `video_sources` ({sample name: video path}) the
    GUI can then show the very frames the sound was cut from."""
    k = (knobs or Knobs()).sanitized()
    events = []

    def ev(t, src, off, dur, rate, role):
        events.append([round(t / SR, 4), src, round(max(0, off) / SR, 4),
                       round(dur / SR, 4), round(rate, 4), role])
    if intro_bars is not None and intro_bars < 1:   # 0 bars = no intro at all
        intro_bars, intro_style = None, "none"
    rng = random.Random(seed)
    bank = load_samples(sample_files)
    if not bank:
        sys.exit("No samples could be loaded")
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
    tonal = np.zeros((total, 2), dtype="float32")   # textures only, for pitch
    kick_hits = []                                  # (pos, section, bar, step)
    swing_frac = k.swing if k.swing is not None else rng.uniform(0.0, 0.06)
    swing = swing_frac * step_len * SR                # 16th-note swing

    print(f"BPM {bpm} | {n_bars} bars | ~{n_bars * bar:.0f}s | style {drum_style}")
    print(f"  swing {swing_frac:.3f} | sections {k.section_bars} | "
          f"layers {k.layers_min}-{k.layers_max} | fx {k.fx_min}-{k.fx_max} | "
          f"saturation {k.saturation}")

    # ---- drum kit carved from random samples --------------------------
    # each role draws from samples no other role has claimed, so kick,
    # snare, hihat and ride never share a source (unless the bank is tiny)
    gains = {"kick": k.gain_kick, "snare": k.gain_snare,
             "hihat": k.gain_hihat, "ohat": k.gain_ohat, "ride": k.gain_ride}
    sgains = {"kick": k.synth_kick, "snare": k.synth_snare,
              "hihat": k.synth_hihat, "ohat": k.synth_ohat, "ride": k.synth_ride}
    kit = {}
    kit_src = {}                  # role -> [(sample name, onset sample)] per variant
    kit_sources = set()
    hat_sources = []
    for role in ("kick", "snare", "hihat", "ride"):
        variants = []
        kit_src[role] = []
        # a role at sample volume 0 claims no samples (they stay free for
        # textures and one-shots); the hihat source also feeds the open hat
        want = gains[role] > 0 or (role == "hihat" and gains["ohat"] > 0)
        for _ in range((1 if role == "ride" else rng.randint(1, 3)) if want else 0):
            pool = [n for n in names if n not in kit_sources] or names
            src = rng.choice(pool)
            kit_sources.add(src)
            used.add(src)
            variants.append(velocity_layers(carve_drum(rng, bank[src], role)))
            kit_src[role].append((src, strongest_onset(bank[src])))
            if role == "hihat":
                hat_sources.append(src)
            print(f"  {role:6s} <- {src}")
        kit[role] = variants
    # the open hat is the same instrument as the closed one: same sources
    kit["ohat"] = [velocity_layers(carve_drum(rng, bank[src], "ohat"))
                   for src in hat_sources]
    kit_src["ohat"] = [(src, strongest_onset(bank[src])) for src in hat_sources]
    # synthesized kit, layered at its own levels (all 0 by default)
    synth = {}
    if any(v > 0 for v in sgains.values()):
        flavour = k.synth_style if k.synth_style != "auto" \
            else SYNTH_FLAVOUR.get(drum_style, "house")
        synth = {r: velocity_layers(h) for r, h in synth_kit(rng, flavour).items()}
        print(f"  synth kit: {flavour}  levels " + " ".join(
            f"{r}={v:g}" for r, v in sgains.items() if v > 0))

    # ---- song structure ------------------------------------------------
    # split bars into sections; each section gets its own patterns/layers
    sections = []
    b = 0
    while b < n_bars:
        if b == 0 and intro_bars is not None:
            length = min(intro_bars, n_bars)
        else:
            length = min(rng.choice(k.section_bars), n_bars - b)
        sections.append((b, length))
        b += length

    long_names = [n for n in names if len(bank[n]) > SR]  # >1s = texture material

    sections_meta = []
    for si, (start_bar, length) in enumerate(sections):
        sec_start = int(start_bar * bar * SR)
        is_intro = si == 0 and intro_style != "none"
        is_outro = si == len(sections) - 1
        is_break = (not is_intro and not is_outro
                    and rng.random() < k.break_chance)
        kind = ("intro" if is_intro else "outro" if is_outro
                else "break" if is_break else "groove")
        sections_meta.append({
            "i": si, "kind": kind, "start_bar": start_bar, "bars": length,
            "start_sec": round(start_bar * bar, 3),
            "end_sec": round((start_bar + length) * bar, 3),
        })

        pats = {r: make_pattern(rng, r, drum_style)
                for r in ("kick", "snare", "hihat", "ride")}
        pats["hihat"], pats["ohat"] = open_hat_pattern(rng, pats["hihat"],
                                                       drum_style)
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
            # the open hat comes with the closed one, except in sparse intros
            if "hihat" in active and intro_kind != "sparse":
                active.append("ohat")
                build_from.setdefault("ohat", build_from.get("hihat", 0))
            # a voice at volume 0 (samples and synth both) is not sequenced
            # at all, no glitch bursts or fills for it either
            active = [r for r in active
                      if (gains[r] > 0 and kit[r]) or (sgains[r] > 0 and synth)]
            for bi in range(length):
                bar_start = sec_start + int(bi * bar * SR)
                bar_pats = {r: (mutate(rng, pats[r], k.mutation_amount)
                                if rng.random() < k.mutation_chance
                                else pats[r]) for r in active}
                # drum fill on the last bar of a section
                if bi == length - 1 and rng.random() < 0.6 and "snare" in active:
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
                        if vel <= 0:
                            continue
                        if role == "hihat" and ohat_pat[s] > 0:
                            continue                  # the open hat owns this step
                        pos = bar_start + int(s * step_len * SR)
                        if s % 2 == 1:
                            pos += int(swing)
                        pos += humanize_time(rng, k.humanize)
                        vel = humanize_vel(rng, vel, s, k.humanize)
                        if role == "kick":
                            kick_hits.append((max(0, pos), si,
                                              start_bar + bi, s))
                        # sample hit and synth hit layer on the same step
                        layers = []
                        if gains[role] > 0 and kit[role]:
                            vi = rng.randrange(len(kit[role]))   # == rng.choice
                            layers.append((pick_layer(kit[role][vi], vel),
                                           gains[role]))
                            ksrc, onset = kit_src[role][vi]
                            ev(max(0, pos), ksrc, onset, len(kit[role][vi][1]),
                               1.0, role)
                        if sgains[role] > 0 and synth:
                            layers.append((pick_layer(synth[role], vel),
                                           sgains[role]))
                        pan = rng.uniform(-pan_drums, pan_drums) \
                            if (role != "kick" and pan_drums > 0) else 0.0
                        for hit, gain in layers:
                            if role == "ohat":
                                # choked by the next closed hat (or the next
                                # bar's downbeat), like the pedal closing
                                nxt = next((t for t in range(s + 1, 16)
                                            if closed_pat[t] > 0), 16)
                                hit = choke(hit, int((nxt - s) * step_len * SR))
                            if pan:
                                hit = panned(hit, pan)
                            place(buf, hit, max(0, pos), vel * gain)
                # hihat glitch: a burst of very fast retriggers (1/32-1/96)
                if "hihat" in active and rng.random() < k.glitch_chance:
                    s = rng.randrange(16)
                    n_rep = rng.choice([6, 8, 12, 16])
                    spacing = max(1, int(step_len * SR / rng.choice([2, 3, 4, 6])))
                    gsrc = None
                    if gains["hihat"] > 0 and kit["hihat"]:
                        vi = rng.randrange(len(kit["hihat"]))     # == rng.choice
                        hit, g_hat = kit["hihat"][vi][1], gains["hihat"]
                        gsrc = kit_src["hihat"][vi]
                    else:
                        hit, g_hat = synth["hihat"][1], sgains["hihat"]
                    ramp = rng.choice([-1, 1])        # fade out or fade in
                    for i in range(n_rep):
                        t = i / max(1, n_rep - 1)
                        g = g_hat * (1.0 - 0.7 * (t if ramp < 0 else 1 - t))
                        h = hit
                        if pan_drums > 0:             # glitch sweeps the field
                            h = panned(hit, np.sin(t * np.pi * 2) * 0.7)
                        gpos = bar_start + int(s * step_len * SR) + i * spacing
                        place(buf, h, gpos, g)
                        if gsrc:
                            ev(gpos, gsrc[0], gsrc[1], spacing, 1.0, "glitch")

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
            L = len(bank[src])
            j0 = len(clip) - sec_len          # first clip sample in the swell
            if j0 >= 0:                       # reversed: index j <- source L-1-j
                ev(sec_start, src, L - j0, min(sec_len, L - j0), -1.0, "swell")
            else:
                ev(sec_start - j0, src, L, L, -1.0, "swell")
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
                cut = int(rng.uniform(0.5, 3.0) * SR)
                clip = random_window(rng, bank[src], cut, k.random_start == "all")
                c_off = len(bank[src]) - len(clip)   # window = a suffix
                clip = clip[:cut].copy()
                c_dur = len(clip)
                clip = clip * envelope(len(clip), attack=0.02, decay_curve=1.5)
                board, tail = random_texture_board(rng, beat, k.fx_min, k.fx_max,
                                               k.saturation)
                clip = apply_fx(clip, board, tail=min(tail, 2.0))
                spread = max(pan_events, 0.6)
                clip = panned(clip, rng.uniform(-spread, spread))
                pos = sec_start + rng.randint(0, max(1, sec_len - len(clip)))
                place(buf, clip, pos, rng.uniform(0.3, 0.5))
                ev(pos, src, c_off, c_dur, 1.0, "collage")
            print(f"  sec{si} collage intro")

        # ---- background texture layers (loop / reverse / stretch) ----
        if intro_kind in ("drums-first", "collage", "reverse-swell"):
            n_layers = 0
        elif intro_kind == "build":
            n_layers = min(1, k.layers_max)
        elif is_intro or is_break:
            n_layers = min(max(2, k.layers_min), k.layers_max)
        else:
            n_layers = rng.randint(k.layers_min, k.layers_max)
        for _ in range(n_layers):
            src = rng.choice(long_names or names)
            used.add(src)
            clip = random_window(rng, bank[src], sec_len,
                                 k.random_start == "all").copy()
            w_off, w_len = len(bank[src]) - len(clip), len(clip)
            ops = []
            rev, rate = False, 1.0
            if rng.random() < k.reverse_chance:
                clip = clip[::-1].copy(); ops.append("rev"); rev = True
            if rng.random() < k.stretch_chance and len(clip) > SR:   # stretch to bars
                n_target_bars = rng.choice([1, 2, 4])
                target = int(n_target_bars * bar * SR)
                chunk = clip[: min(len(clip), target * 2)]
                rate = len(chunk) / target
                st = np.stack([librosa.effects.time_stretch(chunk[:, c], rate=rate)
                               for c in range(2)], axis=1)
                clip = st.astype("float32"); ops.append(f"stretch{n_target_bars}bar")
            mat_n = len(clip)                 # material before the fx tail
            board, tail = random_texture_board(rng, beat, k.fx_min, k.fx_max,
                                               k.saturation)
            clip = apply_fx(clip, board, tail=tail)
            loop = crossfade_loop(clip, sec_len)
            # one event per loop pass, the way crossfade_loop tiles the clip
            n_fade = min(int(0.05 * SR), len(clip) // 4)
            step_n = (len(clip) - n_fade) or len(clip)
            lp = 0
            while lp < sec_len:
                ev(sec_start + lp, src, w_off + w_len if rev else w_off,
                   min(mat_n, sec_len - lp), -rate if rev else rate, "texture")
                lp += step_n
            fade = min(int(0.5 * SR), len(loop) // 4)
            loop[:fade] *= np.linspace(0, 1, fade)[:, None]
            loop[-fade:] *= np.linspace(1, 0, fade)[:, None]
            lvl = rng.uniform(k.layer_level_min, k.layer_level_max) \
                * (1.4 if (is_intro or is_break) else 1.0)
            if pan_layers > 0:
                loop = panned(loop, rng.uniform(-pan_layers, pan_layers))
            place(buf, loop, sec_start, lvl)
            place(tonal, loop, sec_start, lvl)
            print(f"  sec{si} bg <- {src} [{','.join(ops) or 'loop'}]")

        # ---- rhythmic chop layer: slice a sample, replay on the grid ----
        if not is_intro and rng.random() < k.chop_chance:
            src = rng.choice(names)
            used.add(src)
            clip = bank[src]
            n_slices = 8
            sl = max(1, len(clip) // n_slices)
            slices = [clip[i * sl:(i + 1) * sl] for i in range(n_slices)]
            board, tail = random_texture_board(rng, beat, k.fx_min, k.fx_max,
                                               k.saturation)
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
                    p_off, p_rate = idx * sl, 1.0
                    if rng.random() < 0.2:
                        piece = piece[::-1]
                        p_off, p_rate = idx * sl + len(piece), -1.0
                    p_dur = len(piece)
                    piece = piece * envelope(len(piece), decay_curve=2.0)
                    piece = apply_fx(piece, board, tail=tail)
                    if pan_events > 0:
                        piece = panned(piece, rng.uniform(-pan_events, pan_events))
                    cpos = bar_start + int(s * step_len * SR)
                    place(buf, piece, cpos, k.chop_gain)
                    ev(cpos, src, p_off, p_dur, p_rate, "chop")
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
        L, rev = len(clip), False
        if rng.random() < 0.4:
            clip = clip[::-1].copy(); rev = True
        cut = int(min(len(clip), rng.uniform(1, 5) * SR))
        clip = random_window(rng, clip, cut, k.random_start != "off")
        o_off = L - len(clip)                 # offset within the (maybe reversed) clip
        clip = clip[:cut]
        clip = clip * envelope(len(clip), attack=0.05, decay_curve=1.5)
        board, tail = random_texture_board(rng, beat, k.fx_min, k.fx_max,
                                               k.saturation)
        clip = apply_fx(clip, board, tail=tail)
        pos = int(rng.uniform(0.05, 0.9) * n_bars * bar * SR)
        pos = int(round(pos / (beat * SR)) * beat * SR)   # snap to the beat
        if pan_events > 0:
            clip = panned(clip, rng.uniform(-pan_events, pan_events))
        place(buf, clip, pos, rng.uniform(0.25, 0.45))
        ev(pos, src, L - o_off if rev else o_off, cut, -1.0 if rev else 1.0,
           "oneshot")

    # ---- root note per section (from the textures), for the sub and the
    #      GUI's visuals; computed even when the sub is off -----------------
    root_rng = random.Random(seed ^ 0x5B)        # isolated: toggling the sub
    roots = {}                                   # leaves the song unchanged
    last = None
    for m in sections_meta:
        a, b = int(m["start_sec"] * SR), int(m["end_sec"] * SR)
        pc = section_root(tonal[a:b])
        if pc is None:
            pc = last if last is not None else root_rng.randrange(12)
        roots[m["i"]] = last = pc
        m["root"] = NOTE_NAMES[pc]
    print(f"  roots {' '.join(m['root'] for m in sections_meta)}")

    # ---- sub bass: one note per bar on its first kick ---------------------
    if k.sub_level > 0 and kick_hits:
        sub_rng = root_rng
        # one candidate per bar: its first kick (the start of the sequence)
        first_kick = {}
        for pos, si, bar_i, s in kick_hits:
            if sections_meta[si]["kind"] == "intro":
                continue                         # the sub enters at the drop
            if bar_i not in first_kick or s < first_kick[bar_i][2]:
                first_kick[bar_i] = (pos, si, s)
        starts = []
        for bar_i in sorted(first_kick):
            if sub_rng.random() < k.sub_chance:  # weighted occurrence
                pos, si, _ = first_kick[bar_i]
                midi = sub_midi(roots[si])
                if sub_rng.random() < 0.2:       # some bars sit on the fifth
                    midi = midi + 7 if midi + 7 <= SUB_HI else midi - 5
                starts.append((pos, midi))
        hits = []
        note_len = int((beat + k.sub_release) * SR)   # one-beat hold + release
        for i, (pos, midi) in enumerate(starts):
            nxt = starts[i + 1][0] if i + 1 < len(starts) else len(buf)
            length = min(note_len, nxt - pos - int(0.01 * SR))
            if length > int(0.05 * SR):
                hits.append((pos, midi, length))
        sub = render_sub(hits, len(buf), k.sub_phaser, k.sub_pan,
                         k.sub_release)
        active = np.abs(sub).sum(axis=1) > 1e-6
        if active.any():
            sub *= k.sub_level * rms(buf) / rms(sub[active])
            buf += sub
        print(f"  sub: {len(hits)} notes over {len(first_kick)} bars")

    # ---- transitions into drops: repeater, gap, reverse cymbal ---------
    # a "drop" is the start of any non-break section; after an intro or a
    # break it counts fully, groove-to-groove only half as often
    beat_n, bar_n = int(beat * SR), int(bar * SR)
    for si in range(1, len(sections_meta)):
        m = sections_meta[si]
        if m["kind"] == "break":
            continue
        scale = 1.0 if sections_meta[si - 1]["kind"] in ("intro", "break") \
            else 0.5
        boundary = int(m["start_bar"] * bar * SR)
        fx = []
        if rng.random() < k.repeat_chance * scale:
            flavour = beat_repeat(rng, buf, boundary - bar_n, bar_n, beat_n)
            if flavour:
                fx.append(f"repeat:{flavour}")
        if rng.random() < k.gap_chance * scale:
            gap = int(rng.choice([0.5, 1, 1, 2]) * beat_n)
            fade = min(int(0.005 * SR), boundary - gap)
            buf[boundary - gap - fade:boundary - gap] *= \
                np.linspace(1, 0, fade)[:, None]
            buf[boundary - gap:boundary] = 0.0
            fx.append(f"gap:{gap / beat_n:g}beat")
        ride_pool = kit["ride"] or ([synth["ride"]] if synth else [])
        if ride_pool and rng.random() < k.cymbal_chance * scale:
            length = min(rng.choice([1, 2]) * bar_n, boundary)
            swell = cymbal_swell(rng, rng.choice(ride_pool)[1], length)
            place(buf, swell, boundary - length, rng.uniform(0.35, 0.5))
            fx.append(f"cymbal:{length // bar_n}bar")
        if fx:
            m["transition"] = fx
            print(f"  drop into sec{si}: {', '.join(fx)}")

    # ---- outro: ending effect ringing out past the last bar --------------
    song = buf[: int(n_bars * bar * SR)]      # cut at the last bar
    if outro_style == "random":
        outro_style = rng.choice(OUTRO_STYLES)
    outro_meta = {"style": outro_style}
    if outro_style in OUTRO_STYLES:
        tail = k.outro_tail if k.outro_tail is not None else OUTRO_TAIL[outro_style]
        onset_n = int(k.outro_bars * bar * SR)
        song = apply_outro(song, outro_style, beat, tail_sec=tail,
                           onset_n=onset_n, wet=k.outro_wet, amount=k.outro_amount)
        outro_meta.update({"tail_sec": tail, "onset_bars": k.outro_bars,
                           "wet": k.outro_wet, "amount": k.outro_amount})
        sections_meta[-1]["end_sec"] = round(len(song) / SR, 3)
        sections_meta[-1]["outro_style"] = outro_style
        print(f"  outro {outro_style}: {k.outro_bars:g} bars in, {tail:g}s tail")

    # ---- master bus -----------------------------------------------------
    print("Mastering...")
    master = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30),
        Compressor(threshold_db=-14, ratio=2.5, attack_ms=10, release_ms=150),
        Limiter(threshold_db=-1.0),
    ])
    out = master(song.T, SR).T
    peak = np.abs(out).max()
    if peak > 0:
        out = out / peak * 0.95
    # edge fades; the default fade-out is a click guard only, so the outro
    # can be looped or stitched seamlessly in the mixer
    f_in = min(int(k.fade_in * SR), len(out) // 2)
    f_out = min(int(k.fade_out * SR), len(out) // 2)
    if f_in > 0:
        out[:f_in] *= np.linspace(0, 1, f_in)[:, None]
    if f_out > 0:
        out[-f_out:] *= np.linspace(1, 0, f_out)[:, None]
    meta = {"seed": seed, "bpm": bpm, "style": drum_style,
            "duration_sec": round(len(out) / SR, 3),
            "intro_style": intro_style, "outro": outro_meta,
            "swing": round(swing_frac, 4),
            "pan": {"drums": pan_drums, "layers": pan_layers,
                    "events": pan_events},
            "knobs": asdict(k),
            "samples": names,
            "sections": sections_meta,
            "events": sorted(events, key=lambda e: e[0]),
            "video_sources": {n: str(v) for n, v in (video_sources or {}).items()
                              if n in bank}}
    return out.astype("float32"), bpm, meta


COMPOSE_KEYS = ("intro_bars", "intro_style", "bpm", "pan_drums",
                "pan_layers", "pan_events", "num_samples", "drum_style",
                "outro_style")


def run_job(job):
    """Render one job dict (what the GUI writes as JSON):

        seed        int or None (None = a fresh random seed per song)
        count       number of songs; seeds increment from `seed`
        duration    target seconds
        samples     list of audio file paths to build the song from
        video_audio list of video files whose audio tracks join the samples
        knobs       dict of Knobs fields (missing ones keep their default)
        + any compose() keyword in COMPOSE_KEYS

    Files land in output/ and the paths are printed as they are written.
    """
    files = [Path(f) for f in job.get("samples", [])]
    files = [f for f in files if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
    video_sources = {}
    if job.get("video_audio"):
        print("Extracting audio from videos...")
        video_sources = extract_video_audio(job["video_audio"])
        files += [VIDEO_AUDIO_DIR / n for n in video_sources]
        for n, v in video_sources.items():
            print(f"  {v.name} -> {n}")
    if not files:
        sys.exit("No samples selected")
    knobs = Knobs(**{k: v for k, v in (job.get("knobs") or {}).items()
                     if k in Knobs.__dataclass_fields__})
    kw = {k: job[k] for k in COMPOSE_KEYS if k in job}
    for k in ("pan_drums", "pan_layers", "pan_events"):
        if k in kw:
            kw[k] = min(1.0, abs(float(kw[k])))
    base_seed = job.get("seed")
    count = max(1, int(job.get("count", 1)))
    duration = float(job.get("duration", 180.0))

    for i in range(count):
        seed = (base_seed + i) if base_seed is not None \
            else random.randrange(10 ** 6)
        song, bpm, meta = compose(seed, duration, files, knobs=knobs,
                                  video_sources=video_sources, **kw)
        OUT_DIR.mkdir(exist_ok=True)
        word = random.Random(seed).choice(WORDS)
        out_path = OUT_DIR / (f"gacha_{word}_"
                              f"{time.strftime('%Y%m%d_%H%M%S')}"
                              f"_seed{seed}_bpm{bpm}.wav")
        sf.write(out_path, song, SR, subtype="PCM_16")
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(f"\n✔ wrote {out_path}  ({len(song)/SR:.1f}s, {bpm} BPM, seed {seed})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):      # Windows consoles: no crash on ✔
        sys.stdout.reconfigure(errors="replace")
    if len(sys.argv) != 2:
        sys.exit("usage: gacha_engine.py job.json  (normally launched by gacha.py)")
    run_job(json.loads(Path(sys.argv[1]).read_text()))
