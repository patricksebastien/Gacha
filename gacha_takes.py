#!/usr/bin/env python3
"""
gacha_takes.py — moments of the live feed recorded into RAM during a show.

A take is a stretch of the tape's own sound and picture, caught while
performing: press to start, press to stop. Every take starts with a
pre-roll, the seconds before the press, because you decide something was
interesting after it happened. A safety limit stops a forgotten recording;
after that, the next press starts a fresh take. Together the takes are the
sample bank the engine builds sections from at the end of the piece: short
takes are fine, all of them are used.

Audio comes from gacha_audio.LiveAudio (the dry input, before the rack),
frames from gacha_video.VideoSource's live capture; both keep the pre-roll
themselves and hand their blocks and frames over while recording. Time is
the monotonic clock on both sides, so an audio offset finds its frame:
frame time = audio_t0 + offset + video_lag, video_lag being how far behind
the world the grabber's frames are (measured, about 40 ms).

Memory: a minute of 720x576 frames is about 2.5 GB; `max_bytes` caps the
store and the oldest takes go first. save() writes wav + MJPEG mov copies
for rehearsal and offline work.
"""
import subprocess
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal
from scipy.signal import resample_poly

ENGINE_SR = 44100


class Take:
    def __init__(self, idx, audio, sr, audio_t0, frames, video_lag):
        self.idx = idx
        self.audio = audio                  # (n, 2) float32 at sr, or empty
        self.sr = sr
        self.audio_t0 = audio_t0            # monotonic time of sample 0 (None if no audio)
        self.frames = frames                # [Frame] with .arr (h, w, 4) and .t
        self.video_lag = video_lag          # s: frame .t minus the world time it shows
        self._bank = None

    @property
    def seconds(self):
        if len(self.audio):
            return len(self.audio) / self.sr
        if len(self.frames) > 1:
            return self.frames[-1].t - self.frames[0].t
        return 0.0

    @property
    def nbytes(self):
        return self.audio.nbytes + sum(f.arr.nbytes for f in self.frames)

    @property
    def name(self):
        return f"take{self.idx:02d}"

    def engine_audio(self):
        """The audio as the engine wants it: stereo float32 at 44.1 kHz."""
        if self._bank is None:
            if self.sr == ENGINE_SR or not len(self.audio):
                self._bank = self.audio
            else:
                from math import gcd
                g = gcd(ENGINE_SR, self.sr)
                self._bank = resample_poly(self.audio, ENGINE_SR // g, self.sr // g,
                                           axis=0).astype(np.float32)
        return self._bank

    def frame_at(self, offset_s):
        """The frame showing the world at `offset_s` into the audio."""
        if not self.frames:
            return None
        if self.audio_t0 is None:
            t = self.frames[0].t + offset_s
        else:
            t = self.audio_t0 + offset_s + self.video_lag
        times = self._times()
        i = int(np.searchsorted(times, t, side="right")) - 1
        return self.frames[min(max(i, 0), len(self.frames) - 1)]

    def _times(self):
        if getattr(self, "_t", None) is None or len(self._t) != len(self.frames):
            self._t = np.array([f.t for f in self.frames])
        return self._t


class TakeStore(QObject):
    """Record toggle, pre-roll, limit, RAM cap. `changed` fires on start,
    stop and drop; poll `recording` / `elapsed` for the status line."""

    changed = Signal()

    def __init__(self, pre_roll=10.0, limit=60.0, max_bytes=12e9, parent=None):
        super().__init__(parent)
        self.pre_roll = float(pre_roll)
        self.limit = float(limit)
        self.max_bytes = float(max_bytes)
        self.takes = []
        self.recording = False
        self.limit_hit = False
        self.started = None
        self._audio = self._video = None
        self._frames = []
        self._n = 0

    # ------------------------------------------------------------ control
    @property
    def elapsed(self):
        return time.monotonic() - self.started if self.recording else 0.0

    @property
    def nbytes(self):
        return sum(t.nbytes for t in self.takes)

    def toggle(self, audio, video):
        """Press: start a take (with the pre-roll) or stop the one running.
        `audio` is a running LiveAudio or None, `video` a live VideoSource
        or None. Returns the Take just finished, or None."""
        if self.recording:
            return self.stop()
        self.start(audio, video)
        return None

    def start(self, audio, video):
        if self.recording or (audio is None and video is None):
            return False
        self._audio, self._video = audio, video
        self._frames = []
        self.started = time.monotonic()
        self.recording = True
        self.limit_hit = False
        if audio is not None:
            audio.record_start(self.pre_roll)
        if video is not None:
            video.record_start(self.pre_roll, self._frames.append)
        self.changed.emit()
        return True

    def stop(self):
        if not self.recording:
            return None
        blocks = self._audio.record_stop() if self._audio is not None else []
        lag = 0.0
        if self._video is not None:
            self._video.record_stop()
            lag = (self._video.capture_latency_ms or 40.0) / 1000.0
        frames = list(self._frames)
        self._frames = []
        self.recording = False
        if blocks:
            sr = self._audio.sr
            audio = np.concatenate([b for b, _ in blocks], axis=1).T.astype(np.float32)
            audio = np.ascontiguousarray(audio)
            t0 = blocks[0][1] - blocks[0][0].shape[1] / sr
        else:
            sr, audio, t0 = ENGINE_SR, np.zeros((0, 2), np.float32), None
        self._n += 1
        take = Take(self._n, audio, sr, t0, frames, lag)
        self.takes.append(take)
        self._enforce_cap()
        self.changed.emit()
        return take

    def tick(self):
        """Call regularly while recording: stops at the limit."""
        if self.recording and self.elapsed >= self.limit:
            self.stop()
            self.limit_hit = True

    def _enforce_cap(self):
        while len(self.takes) > 1 and self.nbytes > self.max_bytes:
            self.takes.pop(0)

    def clear(self):
        self.takes = []
        self.changed.emit()

    # ------------------------------------------------------------ material
    def bank(self):
        """{take name: stereo float32 at 44.1 kHz} of every take with sound:
        the sample bank the engine works from."""
        return {t.name: t.engine_audio() for t in self.takes if len(t.audio)}

    def save(self, folder):
        """Write every take as wav + MJPEG mov (frames at their own rate)
        into `folder`; returns the paths written."""
        import soundfile as sf
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for t in self.takes:
            if len(t.audio):
                p = folder / f"{t.name}.wav"
                sf.write(str(p), t.audio, t.sr)
                out.append(p)
            if len(t.frames) > 1:
                h, w = t.frames[0].arr.shape[:2]
                dts = np.diff([f.t for f in t.frames])
                fps = 1.0 / max(1e-3, float(np.median(dts)))
                p = folder / f"{t.name}.mov"
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                       "-r", f"{fps:.3f}", "-i", "-"]
                if len(t.audio):
                    cmd += ["-itsoffset", f"{t.video_lag:.3f}", "-i", str(folder / f"{t.name}.wav"),
                            "-c:a", "pcm_s16le", "-shortest"]
                cmd += ["-c:v", "mjpeg", "-q:v", "4", "-pix_fmt", "yuvj420p", str(p)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                for f in t.frames:
                    proc.stdin.write(f.arr.tobytes())
                proc.stdin.close()
                proc.wait()
                out.append(p)
        return out
