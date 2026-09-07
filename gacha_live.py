#!/usr/bin/env python3
"""
gacha_live.py — live audio input for gacha.py: VJ mode.

Instead of a rendered song and its section map, an audio input (a mic, a
line in, or on PipeWire/Pulse a monitor of whatever the machine plays) drives
the video effects and the generative shaders. Two pieces:

  LiveInput  captures the device through Qt's QAudioSource and keeps a
             loudness figure, 0..1, normalised against a slowly decaying
             running peak so quiet and loud sources both use the full range.
  TapTempo   a tap-tempo clock: tap on the beat a few times and it yields a
             BPM and a downbeat anchor, so beat and bar phases exist without
             any beat tracking. The first tap of a series is the downbeat.
  Analyzer   listens along and estimates tempo and beat phase (spectral
             flux onsets, autocorrelation with a tempo prior, comb-filter
             phase), the root note (dominant pitch class of a running
             chroma, as the engine does for songs) and drops (a loud hit
             after a quiet stretch). The GUI feeds its results into the
             TapTempo clock when auto mode is on; a tap takes over.

Everything is polled from the GUI's paint loop; nothing here runs a thread.
The analysis costs about a hundred small FFTs a second plus one larger one
every half second, well inside a frame.
"""

import math
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

# (numpy dtype, offset, scale) to turn Qt's sample formats into -1..1 floats
_SAMPLE_FORMATS = {
    QAudioFormat.SampleFormat.UInt8: (np.uint8, 128.0, 128.0),
    QAudioFormat.SampleFormat.Int16: (np.int16, 0.0, 32768.0),
    QAudioFormat.SampleFormat.Int32: (np.int32, 0.0, 2147483648.0),
    QAudioFormat.SampleFormat.Float: (np.float32, 0.0, 1.0),
}

PEAK_HALF_LIFE = 15.0     # s for the auto-gain reference to fall by half
PEAK_FLOOR = 0.01         # -40 dBFS: below this, silence stays quiet
RELEASE = 0.12            # s, loudness release time (attack is instant)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def audio_inputs():
    """[(description, QAudioDevice)] of the capture devices Qt sees."""
    return [(d.description(), d) for d in QMediaDevices.audioInputs()]


class LiveInput(QObject):
    """Audio capture with a normalised loudness reading (`loud`, 0..1)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._io = None
        self._fmt = None
        self._external = False    # fed by feed() from gacha_audio, no capture
        self.device_name = ""
        self.loud = 0.0           # normalised, smoothed: what the visuals use
        self.rms = 0.0            # raw RMS of the last block, for a meter
        self._env = 0.0
        self._peak = PEAK_FLOOR
        self._last_t = None
        self.analyzer = None      # an Analyzer while running

    @property
    def running(self):
        return self._source is not None or self._external

    @property
    def external(self):
        """True while the blocks come from outside (the audio-through path)."""
        return self._external

    def start_external(self, name, sr):
        """Listen to blocks handed in through feed() instead of capturing:
        the audio-through path already reads the device and owns it."""
        self.stop()
        self._external = True
        self.device_name = name
        self._env, self._peak, self._last_t = 0.0, PEAK_FLOOR, None
        self.loud = self.rms = 0.0
        self.analyzer = Analyzer(sr)
        return None

    def start(self, device=None):
        """Open `device` (QAudioDevice; None = default input). Returns an
        error string, or None when capture is running."""
        self.stop()
        if device is None:
            device = QMediaDevices.defaultAudioInput()
        if device is None or device.isNull():
            return "no audio input device"
        fmt = QAudioFormat()
        fmt.setSampleRate(48000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
        if not device.isFormatSupported(fmt):
            fmt = device.preferredFormat()
        if fmt.sampleFormat() not in _SAMPLE_FORMATS:
            return f"unsupported sample format {fmt.sampleFormat()}"
        src = QAudioSource(device, fmt, self)
        src.setBufferSize(fmt.bytesForDuration(20000))      # ~20 ms blocks
        io = src.start()
        err = src.error()          # compared by name: QAudio became QtAudio in Qt 6.7
        if io is None or err.name != "NoError":
            src.stop()
            return f"cannot open {device.description()}: {err.name}"
        io.readyRead.connect(self._read)
        self._source, self._io, self._fmt = src, io, fmt
        self.device_name = device.description()
        self._env, self._peak, self._last_t = 0.0, PEAK_FLOOR, None
        self.loud = self.rms = 0.0
        self.analyzer = Analyzer(fmt.sampleRate())
        return None

    def stop(self):
        if self._source is not None:
            try:
                self._io.readyRead.disconnect(self._read)
            except (RuntimeError, TypeError):
                pass
            self._source.stop()
            self._source.deleteLater()
        self._source = self._io = self._fmt = None
        self._external = False
        self.loud = self.rms = 0.0

    def _read(self):
        raw = self._io.readAll().data()
        dtype, off, scale = _SAMPLE_FORMATS[self._fmt.sampleFormat()]
        ch = self._fmt.channelCount()
        frame_bytes = self._fmt.bytesPerFrame()
        n = len(raw) // frame_bytes
        if n == 0:
            return
        x = np.frombuffer(raw[: n * frame_bytes], dtype=dtype).astype(np.float32)
        x = (x - off) / scale
        if ch > 1:
            x = x.reshape(n, ch).mean(axis=1)
        self.feed(x, time.monotonic())

    def feed(self, x, now):
        """A block of mono float samples that ended at monotonic time `now`:
        loudness, auto-gain reference and the analyzer."""
        if len(x) == 0:
            return
        rms = float(np.sqrt(np.mean(x * x)))
        self.rms = rms
        dt = 0.02 if self._last_t is None else max(1e-3, now - self._last_t)
        self._last_t = now
        # auto-gain: the reference is the recent peak, falling slowly so a
        # quieter passage opens the range up again after a few seconds
        self._peak = max(PEAK_FLOOR, rms, self._peak * 0.5 ** (dt / PEAK_HALF_LIFE))
        # instant attack, short release: kicks read as kicks
        if rms >= self._env:
            self._env = rms
        else:
            self._env += (rms - self._env) * (1 - math.exp(-dt / RELEASE))
        self.loud = min(1.0, self._env / self._peak)
        self.analyzer.feed(x, now, self._peak)


class Analyzer:
    """Tempo, beat phase, root note and drops from the live signal.

    Results, all read by polling:
      bpm         tempo estimate or None until 4 s have been heard
      beat_time   monotonic time of the most recent estimated beat
      root        dominant pitch class 0..11, or -1 when too quiet / unsure
      drop_time   monotonic time of the last detected drop, or None
    """

    FPS = 100                 # onset frames per second (10 ms hop)
    HISTORY = 8.0             # s of onsets kept for the tempo estimate
    BPM_MIN, BPM_MAX = 60.0, 200.0
    PRIOR_BPM, PRIOR_OCT = 125.0, 0.9        # log-normal tempo prior
    CHROMA_WIN = 3.0          # s of audio behind each chroma reading

    def __init__(self, sr):
        self.sr = int(sr)
        self.hop = self.sr // self.FPS
        self.win = 2 * self.hop
        self._hann = np.hanning(self.win).astype(np.float32)
        self._pending = np.zeros(0, np.float32)     # samples not yet framed
        self._prev_mag = None
        self.onset = deque(maxlen=int(self.HISTORY * self.FPS))
        self.onset_low = deque(maxlen=int(self.HISTORY * self.FPS))   # < 150 Hz
        self._low_bins = int(150 * self.win / self.sr) + 1
        self._frame_time = None    # monotonic time of the last frame's end
        self._last_tempo_t = 0.0
        self._bpm_votes = deque(maxlen=3)
        self.bpm = None
        self.beat_time = None
        # chroma: a 3 s ring of audio, read every half second
        self._ring = np.zeros(int(self.CHROMA_WIN * self.sr), np.float32)
        self._ring_pos = 0
        self._ring_full = False
        self._last_chroma_t = 0.0
        self._chroma = np.zeros(12)
        self.root = -1
        self._root_votes = deque(maxlen=3)
        # drops: RMS history at the block rate, ~50 Hz
        self._rms_hist = deque(maxlen=300)         # (time, rms)
        self.drop_time = None

    # ------------------------------------------------------------ feeding
    def feed(self, x, now, peak):
        """A block of mono float samples that ended at monotonic time now."""
        self._frames(x, now)
        n = len(x)
        end = self._ring_pos + n
        if end <= len(self._ring):
            self._ring[self._ring_pos:end] = x
            self._ring_full = self._ring_full or end == len(self._ring)
        else:                                        # wrap
            k = len(self._ring) - self._ring_pos
            self._ring[self._ring_pos:] = x[:k]
            self._ring[: n - k] = x[k:]
            self._ring_full = True
        self._ring_pos = end % len(self._ring)
        rms = float(np.sqrt(np.mean(x * x))) if n else 0.0
        self._rms_hist.append((now, rms))
        self._drops(now, peak)
        if now - self._last_tempo_t >= 0.5 and len(self.onset) >= 4 * self.FPS:
            self._last_tempo_t = now
            self._tempo()
        if now - self._last_chroma_t >= 0.5 and self._ring_full:
            self._last_chroma_t = now
            self._key(peak)

    def _frames(self, x, now):
        """Spectral-flux onset strength, one value per 10 ms hop."""
        buf = np.concatenate([self._pending, x]) if len(self._pending) else x
        if len(buf) < self.win:
            self._pending = buf
            return
        n_hops = (len(buf) - self.win) // self.hop + 1
        idx = np.arange(self.win)[None, :] + self.hop * np.arange(n_hops)[:, None]
        mag = np.abs(np.fft.rfft(buf[idx] * self._hann, axis=1))
        mag = np.log1p(20.0 * mag)                   # compress: hats count too
        prev = self._prev_mag if self._prev_mag is not None else mag[:1]
        d = np.maximum(np.diff(np.vstack([prev, mag]), axis=0), 0)
        self.onset.extend(d.sum(axis=1).tolist())
        # kicks live down here: the beat phase follows them, not the hats
        self.onset_low.extend(d[:, : self._low_bins].sum(axis=1).tolist())
        self._prev_mag = mag[-1:]
        self._pending = buf[n_hops * self.hop:]
        # the pending tail comes after the last frame
        self._frame_time = now - len(self._pending) / self.sr

    # ------------------------------------------------------------ tempo
    def _tempo(self):
        o = np.asarray(self.onset, dtype=np.float64)
        n = len(o)
        o = o - np.convolve(o, np.ones(self.FPS) / self.FPS, mode="same")   # detrend
        if o.std() < 1e-6:
            return
        f = np.fft.rfft(o, 2 * n)
        ac = np.fft.irfft(f * np.conj(f))[:n]
        ac /= max(ac[0], 1e-9)
        lag_min = int(60.0 * self.FPS / self.BPM_MAX)
        lag_max = int(60.0 * self.FPS / self.BPM_MIN)
        lags = np.arange(lag_min, lag_max + 1)
        bpms = 60.0 * self.FPS / lags
        prior = np.exp(-0.5 * (np.log2(bpms / self.PRIOR_BPM) / self.PRIOR_OCT) ** 2)
        score = ac[lags] * prior
        i = int(np.argmax(score))
        lag = lags[i]
        if 0 < i < len(lags) - 1:                    # parabolic refinement
            a, b, c = score[i - 1], score[i], score[i + 1]
            den = a - 2 * b + c
            if den < 0:
                lag = lag + 0.5 * (a - c) / den
        bpm = 60.0 * self.FPS / lag
        self._bpm_votes.append(bpm)
        if len(self._bpm_votes) < 2:
            return
        votes = np.asarray(self._bpm_votes)
        med = float(np.median(votes))
        if np.abs(votes / med - 1).max() > 0.04:      # not settled yet
            return
        self.bpm = med
        # beat phase: which offset from the last frame lines up best with the
        # bass onsets (kicks, not offbeat hats), recent frames counting more
        period = 60.0 * self.FPS / self.bpm
        span = min(n, int(4 * period))
        lo = np.asarray(self.onset_low, dtype=np.float64)
        lo = lo - np.convolve(lo, np.ones(self.FPS) / self.FPS, mode="same")
        if lo.std() > 1e-6:
            o = lo
        w = np.exp(-np.arange(n) / (2 * period))[::-1]        # recent = 1
        ow = o * w
        phis = np.arange(0, period, 0.5)
        m = np.arange(int(span / period) + 1)
        pos = (n - 1) - phis[:, None] - m[None, :] * period
        valid = pos >= 0
        pos = np.clip(np.rint(pos).astype(int), 0, n - 1)
        comb = (ow[pos] * valid).sum(axis=1)
        phi = float(phis[int(np.argmax(comb))])
        self.beat_time = self._frame_time - phi / self.FPS

    # ------------------------------------------------------------ key
    def _key(self, peak):
        y = np.concatenate([self._ring[self._ring_pos:], self._ring[: self._ring_pos]])
        if float(np.sqrt(np.mean(y * y))) < 0.3 * PEAK_FLOOR:
            self.root = -1                           # silence: no key
            self._root_votes.clear()
            return
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        freqs = np.fft.rfftfreq(len(y), 1.0 / self.sr)
        # from 30 Hz: the sub bass sits on the root, and it is what a live
        # mix has most of (matches the engine's section root 6 times in 10
        # on rendered songs; from 55 Hz it was under 3)
        band = (freqs >= 30.0) & (freqs <= 1800.0)
        f, a = freqs[band], spec[band]
        a = a * a / np.sqrt(f / 30.0)                # power, bass counts most
        pc = (np.rint(12 * np.log2(f / 440.0)) + 69).astype(int) % 12
        chroma = np.bincount(pc, weights=a, minlength=12)
        if chroma.sum() <= 0:
            return
        chroma /= chroma.sum()
        self._chroma = 0.6 * self._chroma + 0.4 * chroma      # ~1.5 s memory
        best = int(np.argmax(self._chroma))
        self._root_votes.append(best)
        if len(self._root_votes) == self._root_votes.maxlen \
                and len(set(self._root_votes)) == 1:
            self.root = best                         # steady for 1.5 s: take it

    # ------------------------------------------------------------ drops
    def _drops(self, now, peak):
        """A drop: the last 150 ms are loud, and somewhere in the 3 s before
        them there was a quiet half second (a break, a gap, the dip under a
        swell) far below that hit. At most one every 8 s."""
        if self.drop_time is not None and now - self.drop_time < 8.0:
            return
        if len(self._rms_hist) < 100:
            return
        t = np.fromiter((h[0] for h in self._rms_hist), float)
        r = np.fromiter((h[1] for h in self._rms_hist), float)
        recent = r[t > now - 0.15]
        before = r[(t < now - 0.15) & (t > now - 3.5)]
        if len(recent) < 2 or len(before) < 40:
            return
        hit = recent.mean()
        k = max(1, len(before) // 6)                  # ~half-second windows
        quiet = np.convolve(before, np.ones(k) / k, mode="valid").min()
        if hit > 0.5 * peak and hit > 2.5 * quiet and quiet < 0.35 * peak:
            self.drop_time = now


class TapTempo:
    """Tap-tempo clock. Taps within 2 s of each other form a series; the
    median interval sets the BPM and the first tap is the downbeat (the one
    of a bar), so tapping 1-2-3-4 on the count lines the bars up too."""

    def __init__(self, bpm=120.0):
        self.bpm = float(bpm)
        self.anchor = time.monotonic()      # time of a downbeat
        self._taps = []

    @property
    def beat_ms(self):
        return 60000.0 / self.bpm

    def tap(self, now=None):
        """Register a tap; returns how many taps the series has."""
        now = time.monotonic() if now is None else now
        if self._taps and now - self._taps[-1] > 2.0:
            self._taps = []
        if self._taps and now - self._taps[-1] < 0.2:
            return len(self._taps)      # a bounce, faster than 300 bpm
        self._taps.append(now)
        self._taps = self._taps[-8:]
        if len(self._taps) >= 2:
            bpm = 60.0 / float(np.median(np.diff(self._taps)))
            self.bpm = min(240.0, max(40.0, bpm))
        # this tap is beat n of the series; the first tap stays the downbeat
        n = len(self._taps) - 1
        self.anchor = now - n * 60.0 / self.bpm
        return len(self._taps)

    def downbeat(self, now=None):
        """Restart the bar here, keeping the tempo."""
        self.anchor = time.monotonic() if now is None else now
        self._taps = []

    def set_bpm(self, bpm, now=None):
        """Change the tempo while keeping the current position in the bar."""
        now = time.monotonic() if now is None else now
        bars = (now - self.anchor) / (4 * 60.0 / self.bpm)
        self.bpm = min(240.0, max(40.0, float(bpm)))
        self.anchor = now - bars * 4 * 60.0 / self.bpm

    def align(self, bpm, beat_time, rate=0.5):
        """Follow the analyzer: take its tempo (bar position kept) and pull
        the anchor part of the way toward having a beat at beat_time, by the
        smallest shift, so the bar count never jumps."""
        if abs(bpm - self.bpm) > 0.05:
            self.set_bpm(bpm, now=beat_time)
        b = 60.0 / self.bpm
        off = (beat_time - self.anchor) % b
        if off > b / 2:
            off -= b
        self.anchor += off * rate

    def pos_ms(self, now=None):
        """Milliseconds since the downbeat anchor: a playhead for the visuals."""
        now = time.monotonic() if now is None else now
        return max(0.0, (now - self.anchor) * 1000.0)
