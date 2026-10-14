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
store and the oldest takes go first. To disk, a take becomes an MJPEG mov
with the sound muxed in (the spec the Videos tab seeks instantly), written
by one background worker feeding ffmpeg. That encode is heavy enough to
glitch the live sound, so it does not happen during the show: the takes
stay in RAM and save() writes the ones not yet on disk when asked, which
the app does when it quits (and on its save button). `archive_on_stop`
brings back writing each take the moment it stops, for a machine that
can afford it.
"""
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal
from scipy.signal import resample_poly

from gacha_engine import NO_WINDOW, ffmpeg_exe

ENGINE_SR = 44100
LEVEL_LUFS = -14.0          # a levelled take's loudness: what the sample sets are cut at
LEVEL_PEAK_DB = -0.1        # ...unless that would push its peak past this: the peak wins


class Take:
    def __init__(self, idx, audio, sr, audio_t0, frames, video_lag):
        self.idx = idx
        self.audio = audio                  # (n, 2) float32 at sr, or empty
        self.sr = sr
        self.audio_t0 = audio_t0            # monotonic time of sample 0 (None if no audio)
        self.frames = frames                # [Frame] with .arr (h, w, 4) and .t
        self.video_lag = video_lag          # s: frame .t minus the world time it shows
        self.saved = False                  # written to disk (by the archive worker)
        self.level = False                  # engine_audio() levelled to LEVEL_LUFS
        self.level_db = 0.0                 # the gain that took, for the log
        self._bank = None                   # (levelled?, audio) as the engine wants it

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
        """The audio as the engine wants it: stereo float32 at 44.1 kHz, and
        with `level` brought to LEVEL_LUFS integrated (ITU-R BS.1770, the
        tape's silence gated out) the way the sample sets are, so a quiet
        tape and a loud one carve drums at one level; the gain is capped
        where the peak would pass LEVEL_PEAK_DB, so a take of hiss with one
        burst is not pushed into clipping. Once per take, cached; the file
        on disk stays as recorded."""
        if self._bank is None or self._bank[0] != self.level:
            if self.sr == ENGINE_SR or not len(self.audio):
                data = self.audio
            else:
                from math import gcd
                g = gcd(ENGINE_SR, self.sr)
                data = resample_poly(self.audio, ENGINE_SR // g, self.sr // g,
                                     axis=0).astype(np.float32)
            self.level_db = 0.0
            if self.level and len(data):
                from gacha_bank import integrated_lufs
                peak = float(np.abs(data).max())
                if peak > 0:
                    gain_db = LEVEL_LUFS - integrated_lufs(data, ENGINE_SR)
                    ceiling = LEVEL_PEAK_DB - 20 * np.log10(peak)
                    self.level_db = min(gain_db, ceiling)
                    data = (data * (10 ** (self.level_db / 20))).astype(np.float32)
            self._bank = (self.level, data)
        return self._bank[1]

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

    def save(self, folder, keep_wav=False):
        """Write this take into `folder`: an MJPEG mov (frames at their own
        rate) with the sound muxed in, pcm, offset by the video lag; a wav
        alone when there is no picture (or with `keep_wav`). Returns the
        paths written; raises on an ffmpeg failure. Safe to call from a
        thread: it only reads the take."""
        import soundfile as sf
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        wav = None
        if len(self.audio):
            wav = folder / f"{self.name}.wav"
            sf.write(str(wav), self.audio, self.sr)
            out.append(wav)
        if len(self.frames) > 1:
            h, w = self.frames[0].arr.shape[:2]
            dts = np.diff([f.t for f in self.frames])
            fps = 1.0 / max(1e-3, float(np.median(dts)))
            mov = folder / f"{self.name}.mov"
            tmp = folder / f"{self.name}.part.mov"
            cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                   "-r", f"{fps:.3f}", "-i", "-"]
            if wav is not None:
                cmd += ["-itsoffset", f"{self.video_lag:.3f}", "-i", str(wav),
                        "-c:a", "pcm_s16le", "-shortest"]
            cmd += ["-c:v", "mjpeg", "-q:v", "4", "-pix_fmt", "yuvj420p", "-f", "mov", str(tmp)]
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, **NO_WINDOW)
            except OSError as e:                     # no ffmpeg: the wav is saved
                raise RuntimeError(f"ffmpeg not found ({e}): frames of {self.name} not saved")
            try:
                for f in self.frames:
                    proc.stdin.write(f.arr.tobytes())
                proc.stdin.close()
            except (BrokenPipeError, OSError):      # ffmpeg died: its code says why
                pass
            rc = proc.wait()
            if rc != 0:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg failed ({rc}) on {self.name}")
            tmp.replace(mov)
            out.append(mov)
            if wav is not None and not keep_wav:     # the mov holds the sound
                wav.unlink(missing_ok=True)
                out.remove(wav)
        return out


class TakeStore(QObject):
    """Record toggle, pre-roll, limit, RAM cap. `changed` fires on start,
    stop and drop; poll `recording` / `elapsed` for the status line.

    `archive_dir`: the folder (or None) takes are written into, by a
    background worker, one take at a time: on save(), and with
    `archive_on_stop` also the moment each take stops. `saved(take, paths,
    error)` fires from that worker (error is "" when it went well)."""

    changed = Signal()
    saved = Signal(object, list, str)

    def __init__(self, pre_roll=0.0, limit=60.0, max_bytes=12e9, parent=None,
                 archive_dir=None, archive_on_stop=False):
        super().__init__(parent)
        self.pre_roll = float(pre_roll)
        self.limit = float(limit)
        self.max_bytes = float(max_bytes)
        self.archive_dir = Path(archive_dir) if archive_dir else None
        self.archive_on_stop = bool(archive_on_stop)
        self.takes = []
        self.level = False                       # new takes levelled (see Take.engine_audio)
        self.recording = False
        self.limit_hit = False
        self.started = None
        self._audio = self._video = None
        self._frames = []
        self._n = 0
        self._jobs = queue.Queue()               # (take, folder) for the worker
        self._worker = None
        self.pending = 0                         # takes queued or being written
        if self.archive_dir is not None:         # a second session today: count on
            self._n = self._on_disk(self.archive_dir)

    @staticmethod
    def _on_disk(folder):
        """The highest takeNN index already written into `folder` (0 if none),
        so a new session never overwrites an earlier one's files."""
        n = 0
        if folder.is_dir():
            for p in folder.iterdir():
                m = re.fullmatch(r"take(\d+)(?:\.part)?", p.stem)
                if m:
                    n = max(n, int(m.group(1)))
        return n

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
        take.level = self.level
        self.takes.append(take)
        self._enforce_cap()
        if self.archive_dir is not None and self.archive_on_stop:
            self.archive(take, self.archive_dir)  # queued first: `changed` sees it pending
        self.changed.emit()
        return take

    # ------------------------------------------------------------ to disk
    def archive(self, take, folder):
        """Queue `take` for the background worker to write into `folder`."""
        self.pending += 1
        self._jobs.put((take, Path(folder)))
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="take-archive",
                                            daemon=True)
            self._worker.start()

    def _work(self):
        while True:
            try:
                take, folder = self._jobs.get(timeout=2.0)
            except queue.Empty:
                return                           # idle: the next archive() restarts us
            paths, err = [], ""
            try:
                paths = take.save(folder)
                take.saved = True
            except Exception as e:               # noqa: BLE001 - report, keep going
                err = str(e)
            self.pending -= 1
            self.saved.emit(take, [str(p) for p in paths], err)

    def wait(self, timeout=None):
        """Block until every queued take is written (or `timeout` s pass).
        Returns True when the queue drained."""
        end = None if timeout is None else time.monotonic() + timeout
        while self.pending > 0:
            if end is not None and time.monotonic() >= end:
                return False
            time.sleep(0.05)
        return True

    def discard_last(self):
        """Forget the take being recorded (stopped, nothing kept) or, when
        none is running, the last one recorded, from RAM only: a file it
        already became stays. Returns "recording", the Take dropped, or
        None when there was nothing to forget."""
        if self.recording:
            if self._audio is not None:
                self._audio.record_stop()
            if self._video is not None:
                self._video.record_stop()
            self._frames = []
            self.recording = False
            self.limit_hit = False
            self.changed.emit()
            return "recording"
        if not self.takes:
            return None
        take = self.takes.pop()
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
        """Forget every take, the one recording included (nothing kept); the
        files already written stay. Returns how many were in RAM."""
        n = len(self.takes)
        if self.recording:
            self.discard_last()
        self.takes = []
        self.changed.emit()
        return n

    # ------------------------------------------------------------ material
    def set_level(self, on):
        """Every take levelled (or not) from now on, the ones in RAM too:
        their engine audio is made again on the next bank()."""
        self.level = bool(on)
        for t in self.takes:
            t.level = self.level

    def bank(self):
        """{take name: stereo float32 at 44.1 kHz} of every take with sound:
        the sample bank the engine works from."""
        return {t.name: t.engine_audio() for t in self.takes if len(t.audio)}

    def unsaved(self):
        """The takes not on disk yet (a failed encode counts as not saved)."""
        return [t for t in self.takes if not t.saved]

    def save(self, folder, background=True, takes=None):
        """Write `takes` (default: every take not yet on disk) into `folder`
        (mov with sound, or wav alone). `background`: through the worker,
        `saved` reports each one; else right here, returning the paths
        written (errors are printed)."""
        takes = self.unsaved() if takes is None else list(takes)
        if background:
            for t in takes:
                self.archive(t, folder)
            return []
        out = []
        for t in takes:
            try:
                out += t.save(folder)
            except RuntimeError as e:
                print(f"takes: {e}")
        return out
