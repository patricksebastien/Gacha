#!/usr/bin/env python3
"""
gacha_video.py — frame-accurate video source for the backdrops.

QMediaPlayer only plays forward at 1x and seeks with a lag, so gacha.py and
gacha_gl.py used to keep a few seconds of half-size frames in RAM to fake a
rewind. This module decodes with PyAV (ffmpeg) in its own thread and owns a
clock instead: a signed `speed` (1.0 normal, -1.0 backwards, 0 frozen, 3.0
fast), seeks that land on the exact frame, and an endless loop in both
directions. The same class opens a V4L2 capture device (`/dev/video0`) for
live video: the newest frame is shown, and the last seconds of captured
frames are kept in a ring so a rewind works there too. It scrubs back
through the ring at the asked speed and, once the speed is positive again,
fast-forwards through what it missed until it is live again, so every
rewind is a backwards-then-forwards jog. Files behave the same way: after
a rewind or a hold, playback runs at CATCHUP until it is back where
uninterrupted play would have been, then settles to the asked speed.

Forward playback decodes sequentially and drops frames when running faster
than real time. Backwards playback decodes a whole group of pictures
(keyframe to keyframe) into a cache and steps through it: for the all-intra
MJPEG files gacha_transcode.py writes, a group is one frame, so any speed
in either direction costs one seek plus one decode per shown frame. Long-GOP
files (camera originals) work too, at the price of decoding each group once
and, for very long groups, caching it at reduced size.

Frames are delivered as RGBA QImages wrapping a numpy buffer, scaled down to
`max_height` when the source is larger (the effects run at window size, so
4K sources gain nothing above 1080p and cost 4x the upload).
"""

import bisect
import os
import random
import struct
import sys
import threading
import time
try:
    import fcntl                        # V4L2 device queries; Linux only
except ImportError:                     # pragma: no cover - Windows, macOS
    fcntl = None
from collections import deque, namedtuple
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

try:
    import av
except ImportError as e:                       # pragma: no cover
    raise ImportError("gacha needs PyAV for video: pip install av") from e

# a capture device is a string: "/dev/videoN" on Linux (V4L2),
# "dshow:video=Name" on Windows (DirectShow), "avfoundation:N" on macOS
DEVICE_PREFIXES = ("/dev/", "dshow:", "avfoundation:")


def is_device(path):
    return str(path).startswith(DEVICE_PREFIXES)


def _device_open(path):
    """av.open() for a device string; (container, input format name)."""
    if path.startswith("dshow:"):
        return av.open(path[len("dshow:"):], format="dshow"), "dshow"
    if path.startswith("avfoundation:"):
        return av.open(path[len("avfoundation:"):], format="avfoundation"), "avfoundation"
    return av.open(path, format="v4l2"), "v4l2"

# a decoded frame ready to show: the QImage wraps `arr`, which must stay
# alive as long as the image is used; `t` is media time in seconds
Frame = namedtuple("Frame", "img arr t")

GOP_CACHE_BYTES = 384 * 1024 * 1024     # reverse cache budget per group
GOP_CACHE_GROUPS = 2                    # groups kept for backwards runs
SEEK_AHEAD_S = 2.0                      # forward gaps larger than this seek
                                        # instead of decoding through
LIVE_BUFFER_BYTES = 512 * 1024 * 1024   # ring of captured frames for rewinds
CATCHUP = 3.0                           # forward rate after a rewind or hold,
                                        # until back where play would have been

# a group of pictures decoded for backwards playback: frames sorted by time,
# covering media time [t0, t1)
_Gop = namedtuple("_Gop", "t0 t1 times frames")


def video_inputs():
    """[(name, device string)] of the video capture devices: a webcam, or a
    USB composite/VHS grabber. Linux asks V4L2 directly (metadata nodes,
    a webcam's second /dev/video, and devices we cannot open are left out);
    Windows and macOS take the cameras Qt sees and hand them to ffmpeg's
    DirectShow / AVFoundation inputs by name / index."""
    if sys.platform.startswith("linux") and fcntl is not None:
        return _v4l2_inputs()
    try:
        from PySide6.QtMultimedia import QMediaDevices
        cams = QMediaDevices.videoInputs()
    except Exception:
        return []
    out = []
    for i, cam in enumerate(cams):
        name = cam.description() or f"camera {i}"
        if sys.platform == "win32":
            out.append((name, f"dshow:video={name}"))
        elif sys.platform == "darwin":
            out.append((name, f"avfoundation:{i}:none"))
    return out


def _v4l2_inputs():
    VIDIOC_QUERYCAP, CAPTURE, META = 0x80685600, 0x1, 0x00800000
    out = []
    root = "/sys/class/video4linux"
    for n in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        dev = "/dev/" + n
        buf = bytearray(104)
        try:
            fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
            try:
                fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
            finally:
                os.close(fd)
        except OSError:
            continue
        _drv, card, _bus, _ver, _caps, dcaps = struct.unpack("16s32s32sIII", bytes(buf[:92]))
        if dcaps & CAPTURE and not dcaps & META:
            name = card.split(b"\0")[0].decode(errors="replace").strip() or n
            out.append((f"{name} ({n})", dev))
    return out


def _display_size(w, h, max_height):
    if max_height and h > max_height:
        s = max_height / h
        return max(2, int(round(w * s / 2)) * 2), max_height
    return w, h


class VideoSource(QObject):
    """A looping, speed-controllable, seekable video decoder on its own
    thread. `latest()` is the frame to show right now; `frameChanged` fires
    (queued, in the GUI thread) whenever it changes."""

    frameChanged = Signal()

    def __init__(self, max_height=1080, parent=None):
        super().__init__(parent)
        self.max_height = max_height
        self.current = None              # Path (or device) now open
        self.duration = None             # seconds; None until opened / live
        self.is_live = False
        self._live_fmt = None            # "v4l2", "dshow", "avfoundation" while live
        self.error = None                # last open failure, for the log
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._cmds = []                  # ("open", path, random) | ("seek", t | None)
        self._speed = 1.0                # asked for
        self._rate = 1.0                 # actually running (catch-up differs)
        self._debt = 0.0                 # seconds behind uninterrupted play
        self.catch_up = True             # False: slow/backward is deliberate,
                                         # no fast-forward afterwards
        self._last_tick = None
        self._pos0, self._t0 = 0.0, time.monotonic()   # clock: pos0 at t0
        self._latest = None
        self._quit = False
        # decoder state (worker thread only)
        self._ctr = None
        self._stream = None
        self._start = 0.0                # stream start time, subtracted
        self._tb = 1.0
        self._size = (0, 0)              # display size of decoded frames
        self._gen = None                 # forward decode generator
        self._next = None                # first decoded frame not yet due
        self._eof = False                # forward decode ran off the end
        self._shown_t = None             # media time of the frame shown
        self._gops = []                  # reverse cache, oldest first
        self._graphs = {}                # (in w, h, fmt, out w, h) -> filter graph
        self._ring_t = deque()           # live: capture wall times, oldest first
        self.capture_latency_ms = None   # live: kernel capture -> decoded, smoothed
        self._rec_sink = None            # live: callable taking each captured Frame
        self._rec_pre = None             # s of ring to hand over first, once
        self._ring = deque()             # live: the Frames for those times
        self._live_pos = None            # live: wall time shown when behind live
        self._live_tick = None           # live: wall time of the last tick
        self._thread = threading.Thread(target=self._run, name="gacha-video",
                                        daemon=True)
        self._thread.start()

    # ---- control (any thread) ----
    def open(self, path, random_start=False):
        """Start looping `path` (a file, or a capture device string, see
        video_inputs()), from the beginning or from a random position."""
        self.current = str(path) if is_device(path) else Path(path)
        with self._lock:
            self._cmds.append(("open", str(path), bool(random_start)))
        self._wake.set()

    def seek(self, t):
        """Jump to media time `t` in seconds (wraps around the duration)."""
        with self._lock:
            self._cmds.append(("seek", float(t)))
        self._wake.set()

    def jump(self):
        """Jump to a random position (no-op for live sources)."""
        with self._lock:
            self._cmds.append(("seek", None))
        self._wake.set()

    @property
    def speed(self):
        return self._speed

    def set_speed(self, v):
        """Playback rate: 1 normal, negative backwards, 0 holds the frame.
        Cheap when unchanged, so it can be called every paint."""
        v = float(v)
        if v == self._speed:
            return
        now = time.monotonic()
        with self._lock:
            self._pos0 = self._clock(now)
            self._t0 = now
            self._speed = v
        self._wake.set()

    def position(self):
        """Media time now, in seconds (0 for live sources)."""
        with self._lock:
            return self._clock(time.monotonic())

    def record_start(self, pre_roll_s, sink):
        """Hand every captured frame to `sink(frame)` from now on, starting
        with the last `pre_roll_s` seconds already in the ring. Live sources
        only; the decoder thread does the handing over."""
        self._rec_pre = float(pre_roll_s)
        self._rec_sink = sink

    def record_stop(self):
        self._rec_sink = None
        self._rec_pre = None

    def latest(self):
        """The Frame to show now, or None before the first decode."""
        return self._latest

    def stop(self):
        self._quit = True
        self._wake.set()
        self._thread.join(1.0)

    # ---- clock (call with the lock held) ----
    def _clock(self, now):
        t = self._pos0 + (now - self._t0) * self._rate
        d = self.duration
        if d:
            t %= d
        return t

    def _set_clock(self, t, now=None):
        """Jump the clock: a new file or a seek, so nothing to catch up."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._pos0, self._t0 = t, now
            self._debt, self._last_tick = 0.0, None
            self._rate = self._speed

    def _advance(self, now):
        """Book the time since the last tick against the catch-up debt and
        pick the rate to run at: the asked speed, or CATCHUP while behind.
        Returns (rate, media time now)."""
        with self._lock:
            speed = self._speed
            last, self._last_tick = self._last_tick, now
            if last is not None and self.catch_up:
                self._debt = max(0.0, self._debt + (now - last) * (1.0 - self._rate))
            elif not self.catch_up:
                self._debt = 0.0
            want = speed if speed < 1.0 else \
                (max(speed, CATCHUP) if self._debt > 0.0 else speed)
            if want != self._rate:
                self._pos0, self._t0 = self._clock(now), now
                self._rate = want
            return self._rate, self._clock(now)

    # ---- worker thread ----
    def _run(self):
        while not self._quit:
            try:
                self._tick()
            except Exception as e:           # never let a bad file kill the loop
                self.error = f"{self.current}: {e}"
                self._close()
                self._wake.wait(0.5)
            self._wake.clear()

    def _tick(self):
        with self._lock:
            cmds, self._cmds = self._cmds, []
        for cmd in cmds:
            if cmd[0] == "open":
                self._open(cmd[1], cmd[2])
            elif cmd[0] == "seek" and self._ctr is not None and not self.is_live:
                t = cmd[1]
                if t is None:
                    t = random.uniform(0.0, self.duration or 0.0)
                self._set_clock(t % (self.duration or 1.0))
                self._show(self._frame_at(self._clock_now(), forward=True))
        if self._ctr is None:
            self._wake.wait(0.05)
            return
        if self.is_live:
            self._show(self._live_step())
            return
        rate, target = self._advance(time.monotonic())
        if rate == 0.0:
            self._wake.wait(0.02)
            return
        fr = self._frame_at(target, forward=rate > 0)
        self._show(fr)
        # sleep until the next frame boundary is due, but stay responsive
        bound = self._next_boundary(target, rate > 0)
        wait = 0.005 if bound is None else (bound - target) / rate
        self._wake.wait(min(0.05, max(0.001, wait)))

    def _clock_now(self):
        with self._lock:
            return self._clock(time.monotonic())

    def _show(self, fr):
        if fr is None:
            return
        self._latest = fr
        self.frameChanged.emit()

    def _close(self):
        self._gen = self._next = None
        self._eof = False
        self._shown_t = None
        self._gops = []
        self._graphs = {}
        self._ring_t.clear()
        self._ring.clear()
        self._live_pos = self._live_tick = None
        if self._ctr is not None:
            try:
                self._ctr.close()
            except Exception:
                pass
        self._ctr = self._stream = None

    def _open(self, path, random_start):
        self._close()
        self.error = None
        live = is_device(path)
        self._live_fmt = None
        if live:
            ctr, self._live_fmt = _device_open(path)
        else:
            ctr = av.open(path)
        s = ctr.streams.video[0]
        s.thread_type = "AUTO"
        self._ctr, self._stream = ctr, s
        self._tb = float(s.time_base)
        self._start = (s.start_time or 0) * self._tb
        self._size = _display_size(s.codec_context.width, s.codec_context.height,
                                   self.max_height)
        self.is_live = live
        dur = None
        if not live:
            if s.duration:
                dur = s.duration * self._tb
            elif ctr.duration:
                dur = ctr.duration / av.time_base
        self.duration = dur
        t = random.uniform(0.0, dur) if (random_start and dur) else 0.0
        self._set_clock(t)
        if not live:
            self._show(self._frame_at(t, forward=True))
            if self._shown_t is not None:      # start the clock on the frame found
                self._set_clock(self._shown_t)

    # ---- decoding ----
    def _time(self, f):
        return f.pts * self._tb - self._start

    def _converter(self, f, w, h):
        """A libavfilter graph turning frames like `f` into w x h RGBA. Not
        VideoFrame.reformat / to_ndarray: PyAV allocates that output with 32
        byte rows and libswscale's SIMD tail writes past the end of the last
        row when a row is not a multiple of 64 bytes, which corrupts the heap
        on 808 or 810 pixel wide (portrait) clips. libavfilter pads its
        buffers, so the same conversion is safe here, at the same speed."""
        key = (f.width, f.height, f.format.name, w, h)
        g = self._graphs.get(key)
        if g is None:
            g = av.filter.Graph()
            last = g.add_buffer(width=f.width, height=f.height,
                                format=f.format.name,
                                time_base=self._stream.time_base)
            if (f.width, f.height) != (w, h):
                sc = g.add("scale", f"{w}:{h}:flags=bilinear")
                last.link_to(sc)
                last = sc
            fmt = g.add("format", "rgba")
            last.link_to(fmt)
            sink = g.add("buffersink")
            fmt.link_to(sink)
            g.configure()
            self._graphs[key] = g
        return g

    def _to_frame(self, f, scale=1.0):
        w, h = self._size
        if scale != 1.0:
            w, h = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
        g = self._converter(f, w, h)
        g.push(f)
        out = g.pull()
        arr = np.ascontiguousarray(out.to_ndarray())    # tight w*4 rows
        img = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
        return Frame(img, arr, self._time(f))

    def _seek(self, t):
        """Position the demuxer at the keyframe at or before media time t."""
        self._gen = self._next = None
        self._eof = False
        pts = int(round((max(0.0, t) + self._start) / self._tb))
        self._ctr.seek(pts, stream=self._stream, backward=True, any_frame=False)

    def _pull(self):
        """Next decoded frame with a timestamp, or None at end of file."""
        if self._gen is None:
            return None
        for f in self._gen:
            if f.pts is not None:
                return f
        self._gen = None
        self._eof = True
        return None

    def _frame_at(self, target, forward):
        """The frame covering media time `target`, converted, or None if it
        is the one already shown."""
        if forward:
            return self._forward_to(target)
        return self._backward_to(target)

    def _forward_to(self, target):
        shown_t, nxt = self._shown_t, self._next
        if shown_t is None or target < shown_t - 1e-6 \
                or (nxt is not None and target > self._time(nxt) + SEEK_AHEAD_S) \
                or (nxt is None and not self._eof):
            self._seek(target)
            self._gen = self._ctr.decode(self._stream)
            nxt = self._pull()
            if nxt is None:                  # past the last frame: wrap
                self._seek(0.0)
                self._gen = self._ctr.decode(self._stream)
                nxt = self._pull()
                if nxt is None:
                    return None
            if self._time(nxt) > target:     # seek overshot (start of file)
                self._next = self._pull()
                return self._take(nxt)
        new = None
        while nxt is not None and self._time(nxt) <= target:
            new, nxt = nxt, self._pull()
        self._next = nxt
        if new is None:
            return None
        return self._take(new)

    def _take(self, f):
        t = self._time(f)
        if self._shown_t is not None and abs(t - self._shown_t) < 1e-9:
            return None                      # same frame: nothing to convert
        fr = self._to_frame(f)
        self._shown_t = fr.t
        return fr

    def _backward_to(self, target):
        gop = self._gop_for(target)
        if gop is None:
            gop = self._load_gop(target)
            if gop is None:
                return None
        i = bisect.bisect_right(gop.times, target + 1e-9) - 1
        if i < 0:
            i = 0
        fr = gop.frames[i]
        if self._shown_t is not None and abs(fr.t - self._shown_t) < 1e-9:
            return None
        self._shown_t = fr.t
        return fr

    def _gop_for(self, t):
        for g in self._gops:
            if g.t0 - 1e-9 <= t < g.t1:
                return g
        return None

    def _load_gop(self, target):
        """Decode the group of pictures containing `target` into the cache."""
        self._seek(target)
        packets, end_pts = [], None
        for p in self._ctr.demux(self._stream):
            if p.size == 0:                  # ffmpeg's end-of-stream flush packet
                break
            if p.is_keyframe and packets:
                end_pts = p.pts
                break
            packets.append(p)
        if not packets:
            return None
        cc = self._stream.codec_context
        raw = []
        for p in packets:
            raw += [f for f in cc.decode(p) if f.pts is not None]
        raw += [f for f in cc.decode(None) if f.pts is not None]
        cc.flush_buffers()
        self._gen = self._next = None
        self._eof = False
        if not raw:
            return None
        raw.sort(key=lambda f: f.pts)
        w, h = self._size
        need = len(raw) * w * h * 4
        scale = 1.0
        while need * scale * scale > GOP_CACHE_BYTES and scale > 0.25:
            scale *= 0.5
        frames = [self._to_frame(f, scale) for f in raw]
        t0 = frames[0].t
        if end_pts is not None:
            t1 = end_pts * self._tb - self._start
        else:
            t1 = max(self.duration or 0.0, frames[-1].t + 1e-3)
        gop = _Gop(t0, t1, [f.t for f in frames], frames)
        self._gops.append(gop)
        del self._gops[:-GOP_CACHE_GROUPS]
        return gop

    def _next_boundary(self, target, forward):
        """Media time at which the shown frame changes next, if known."""
        if forward:
            if self._next is not None:
                return self._time(self._next)
            return self.duration if self._eof else None   # tail gap: wait for the wrap
        if self._shown_t is not None and self._shown_t <= target:
            return self._shown_t - 1e-4
        return None

    def _next_live(self):
        """Block for the next captured frame; its `t` is the capture wall time."""
        if self._gen is None:
            self._gen = self._ctr.decode(self._stream)
        f = next(self._gen, None)
        if f is None:
            raise EOFError("capture device stopped")
        fr = self._to_frame(f)
        now = time.monotonic()
        if f.pts is not None and self._live_fmt == "v4l2":
            # v4l2 stamps frames with the kernel's monotonic clock, so this is
            # how far behind the world the decoded picture already is (one
            # frame period on the MS210x grabber); paint and display add more
            lat = (now - f.pts * self._tb) * 1000.0
            if 0.0 <= lat < 2000.0:
                self.capture_latency_ms = lat if self.capture_latency_ms is None \
                    else 0.9 * self.capture_latency_ms + 0.1 * lat
        return Frame(fr.img, fr.arr, now)

    def _live_step(self):
        """Capture one frame into the ring, then pick the frame to show: the
        newest when live, or a ring frame while rewinding or catching up."""
        fr = self._next_live()
        self._ring_t.append(fr.t)
        self._ring.append(fr)
        sink = self._rec_sink
        if sink is not None:
            if self._rec_pre is not None:            # first frame: the pre-roll
                since = fr.t - self._rec_pre
                i = bisect.bisect_left(self._ring_t, since)
                for old in list(self._ring)[i:-1]:
                    sink(old)
                self._rec_pre = None
            sink(fr)
        w, h = self._size
        keep = max(2, LIVE_BUFFER_BYTES // max(1, w * h * 4))
        while len(self._ring) > keep:
            self._ring_t.popleft()
            self._ring.popleft()
        now = fr.t
        last, self._live_tick = self._live_tick, now
        speed = self._speed
        if self._live_pos is None:
            if speed >= 1.0:
                return fr                    # live, as it comes in
            self._live_pos = self._ring_t[-1] if len(self._ring_t) < 2 \
                else self._ring_t[-2]        # start scrubbing from what was shown
            last = now
        rate = speed if speed < 1.0 else max(speed, CATCHUP)
        pos = self._live_pos + (now - last) * rate
        if pos >= self._ring_t[-1]:
            self._live_pos = None            # caught up: live again
            return fr
        pos = max(pos, self._ring_t[0])
        self._live_pos = pos
        i = bisect.bisect_right(self._ring_t, pos) - 1
        return self._ring[max(0, i)]
