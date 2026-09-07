#!/usr/bin/env python3
"""
gacha_audio.py — live audio through: an input device plays out of an output
device, through a rack of effects, in realtime. Act one of the VHS show: the
deck's sound goes to the PA through the app, clean, so that later the same
path can carry effects and, later still, the generated material.

Built on pedalboard's AudioStream (JUCE/ALSA) with Python in the loop:

    input device  --read-->  [analysis tap]  -> rack -> sync delay --+
                                                                     A/B mix -> gain -->write-->  output
    section player (looping stems) + one-shots ----------------------+

The sync delay holds the sound back by a settable number of milliseconds so
it lines up with the picture, which arrives late through the capture
device, the decoder and the display (typically 50 to 150 ms). It changes
live without a click: a block is crossfaded from the old to the new delay.

Two streams, not one: a single AudioStream with both an input and an output
copies input to output inside its C++ callback and ignores write(), so
nothing from Python (the generator, later) could ever join the mix. With an
input-only and an output-only stream this loop owns every sample. The cost is
one Python round per block; measured on 256-frame blocks (5.3 ms) at 48 kHz
the loop runs at 1.00x with a three-plugin rack, all pedalboard plugins take
well under a millisecond per block, and it survived a busy GUI thread with no
drops (the interpreter's switch interval is shortened too, see _run()).

Devices: only ALSA "direct hardware" devices work (PipeWire's ALSA plugin
devices report no channels to JUCE). A device PipeWire is actively using
cannot be opened; PipeWire lets go of idle (suspended) devices by itself.
Songs keep playing through Qt as before; this path is for the live input.

    LiveAudio(in_name, out_name).start()   # opens in a thread (~2 s)
    la.board = Pedalboard([...])           # the rack, swap any time (crossfaded)
    la.gain, la.mute                       # ramped, click-free
    la.delay_ms = 80                       # audio held back to meet the picture
    la.drain()                             # mono blocks for the analyzer
    la.record_start(10.0); la.record_stop()   # a take: dry input, with pre-roll
    la.player                              # SectionPlayer: stems, next, one-shots
    la.ab = 0.0 .. 1.0                     # A tape through <-> B the section
    la.stats()                             # peaks, load, drops, latency
"""
import sys
import threading
import time
from collections import deque

import numpy as np
from pedalboard import Pedalboard
from pedalboard.io import AudioStream

SR = 48000
HW_SUFFIX = "; Direct hardware device without any conversions"
SWITCH_INTERVAL = 0.001          # s; Python's default 5 ms lets the GUI hold
                                 # the GIL for one whole block
MAX_DELAY_MS = 2000              # the sync delay's ring


def audio_devices():
    """([(short name, device name)], [(short name, device name)]) of the
    inputs and outputs pedalboard can open, short names for combos
    ("MS210x, USB Audio"), full names for AudioStream. On Linux only the
    ALSA hardware devices (PipeWire's plugin devices report no channels);
    elsewhere (WASAPI / DirectSound on Windows, CoreAudio on macOS) every
    device the library lists."""
    def hw(names):
        if not sys.platform.startswith("linux"):
            return [(n, n) for n in names]
        return [(n[: -len(HW_SUFFIX)] if n.endswith(HW_SUFFIX) else n, n)
                for n in names if HW_SUFFIX in n]
    try:
        return hw(AudioStream.input_device_names), hw(AudioStream.output_device_names)
    except Exception:                                    # no audio system at all
        return [], []


class SectionPlayer:
    """Looping stems at the stream's rate, swapped on a bar, plus one-shots.
    Everything here is set from the GUI thread and read in the audio
    thread; swaps go through `pending` and happen on a bar boundary."""

    def __init__(self, sr):
        self.sr = sr
        self.stems = None                    # {name: (2, n) float32 at sr}
        self.events = []                     # the section's timeline (seconds)
        self.bpm = 120.0
        self.bars = 8
        self.n = 0                           # loop length in frames
        self.pos = 0                         # play position in frames
        self.pending = None                  # (stems, events, bpm, bars) for the next bar
        self.on = {"drums": True, "layers": True, "chops": True, "events": True}
        self._gain_now = {}                  # stem -> gain reached (ramps)
        self.shots = []                      # [(clip (2, n), frames played)]
        self.shot_queue = []                 # clips waiting for the next beat
        self.loop_t0 = None                  # monotonic time the loop last wrapped
        self.swapped = False                 # a pending section just went live
        self.beat_frames = 0

    @property
    def playing(self):
        return self.stems is not None

    def queue(self, stems, events, bpm, bars):
        self.pending = (stems, events, bpm, bars)

    def clear(self):
        self.pending = ("clear",)

    def fire(self, clip):
        """A one-shot (2, n) float32 at sr, on the next beat (or now if no
        section plays)."""
        self.shot_queue.append(clip)

    def pos_s(self):
        return self.pos / self.sr

    def _swap(self):
        p, self.pending = self.pending, None
        if p[0] == "clear":
            self.stems, self.events, self.n, self.pos = None, [], 0, 0
            return
        stems, events, bpm, bars = p
        self.stems, self.events, self.bpm, self.bars = stems, events, bpm, bars
        self.n = next(iter(stems.values())).shape[1]
        self.pos = 0
        self.beat_frames = int(round(60.0 / bpm * self.sr))
        self.loop_t0 = time.monotonic()
        self.swapped = True

    def block(self, bs, now, ramp):
        """The next bs frames of section + one-shots, (2, bs) float32."""
        out = np.zeros((2, bs), np.float32)
        bar_f = 4 * self.beat_frames
        if self.pending is not None and (self.stems is None or self.pos == 0):
            self._swap()
        if self.stems is not None:
            i0 = self.pos
            idx = (np.arange(bs) + i0) % self.n
            for name, data in self.stems.items():
                target = 1.0 if self.on.get(name, True) else 0.0
                cur = self._gain_now.get(name, target)
                if cur != target:
                    out += data[:, idx] * np.linspace(cur, target, bs, dtype=np.float32)
                    self._gain_now[name] = target
                elif target:
                    out += data[:, idx]
            self.pos = (i0 + bs) % self.n
            if i0 + bs >= self.n:                     # wrapped inside this block
                self.loop_t0 = now + (self.n - i0) / self.sr
            # a bar boundary inside this block: a pending section comes in
            # at the next block (5 ms off at most), its position on the bar
            if self.pending is not None and bar_f and (i0 // bar_f) != ((i0 + bs) // bar_f):
                self.pos = 0                          # so the swap lands on the bar
            on_beat = (i0 // self.beat_frames) != ((i0 + bs) // self.beat_frames) \
                if self.beat_frames else True
        else:
            on_beat = True
        if self.shot_queue and on_beat:
            self.shots.extend((c, 0) for c in self.shot_queue)
            self.shot_queue = []
        keep = []
        for clip, k in self.shots:
            n = min(bs, clip.shape[1] - k)
            out[:, :n] += clip[:, k:k + n]
            if k + n < clip.shape[1]:
                keep.append((clip, k + n))
        self.shots = keep
        return out


class LiveAudio:
    """The through path, on its own thread. Attributes read from any thread:
    running, error, and the stats. Set from any thread: board, gain, mute."""

    def __init__(self, in_name, out_name, block=256, sr=SR):
        self.in_name, self.out_name = in_name, out_name
        self.block, self.sr = int(block), int(sr)
        self.board = Pedalboard([])          # the rack; assign a new one to swap
        self.gain = 1.0
        self.mute = False
        self.delay_ms = 0.0                  # sync: hold the sound back this much
        self.ab = 0.0                        # 0 = tape through only, 1 = section only
        self.player = SectionPlayer(self.sr)
        self.running = False
        self.error = None
        self.ready = threading.Event()       # set once open (or failed)
        self._stop = threading.Event()
        self._thread = None
        self._blocks = deque(maxlen=64)      # (mono float32 block, end time)
        self._lock = threading.Lock()
        self.pre_roll = 12.0                 # s of dry input always kept for takes
        self._pre = deque(maxlen=int(self.pre_roll * self.sr / self.block) + 1)
        self._rec = None                     # [(stereo block, end time)] while recording
        self._rec_req = None                 # pre-roll seconds asked for, once
        # stats
        self.in_peak = self.out_peak = 0.0   # last block's peaks, 0..1
        self.load = 0.0                      # DSP time / block time, smoothed
        self.dropped = 0                     # input frames lost (late reads)
        self.late = 0                        # loop rounds slower than 2 blocks
        self.rounds = 0

    # ------------------------------------------------------------ control
    def start(self):
        self._thread = threading.Thread(target=self._run, name="gacha-audio",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3.0)
        self.running = False

    def drain(self):
        """[(mono float32 block, monotonic end time)] captured since the
        last call, oldest first, for the analyzer."""
        with self._lock:
            items = list(self._blocks)
            self._blocks.clear()
        return items

    def record_start(self, pre_roll_s):
        """Start collecting the dry input, beginning `pre_roll_s` seconds
        ago (as far as kept). The audio thread picks it up on its next block."""
        self._rec_req = min(float(pre_roll_s), self.pre_roll)

    def record_stop(self):
        """Stop and return [(stereo (2, bs) float32 block, monotonic end
        time)] oldest first; [] if nothing was recording."""
        with self._lock:
            blocks, self._rec = self._rec, None
            self._rec_req = None
        return blocks or []

    def stats(self):
        return {"in_peak": self.in_peak, "out_peak": self.out_peak,
                "load": self.load, "dropped": self.dropped, "late": self.late,
                "buffer_ms": 2000.0 * self.block / self.sr,
                "delay_ms": float(self.delay_ms)}

    # ------------------------------------------------------------ thread
    def _delay_frames(self):
        ms = min(MAX_DELAY_MS, max(0.0, float(self.delay_ms)))
        return int(round(ms / 1000.0 * self.sr))

    def _open(self):
        src = AudioStream(input_device_name=self.in_name, output_device_name=None,
                          sample_rate=self.sr, buffer_size=self.block,
                          num_input_channels=2)
        src.ignore_dropped_input = True
        try:
            snk = AudioStream(input_device_name=None, output_device_name=self.out_name,
                              sample_rate=self.sr, buffer_size=self.block,
                              num_output_channels=2)
        except Exception:
            src.close()
            raise
        if not snk.sample_rate:                    # JUCE's way of saying busy
            src.close()
            snk.close()
            raise RuntimeError(f"{self.out_name}: cannot open (in use by "
                               "PipeWire or another app?)")
        return src, snk

    def _run(self):
        old_switch = sys.getswitchinterval()
        try:
            src, snk = self._open()
        except Exception as e:
            msg = str(e)
            if "no channels" in msg:
                msg = "no channels: the device is in use (PipeWire or another app)"
            self.error = msg
            self.ready.set()
            return
        sys.setswitchinterval(SWITCH_INTERVAL)
        bs, sr = self.block, self.sr
        block_s = bs / sr
        self._dsp_init()
        try:
            with src, snk:
                self.running = True
                self.ready.set()
                src.read()                        # drop what piled up while opening
                last = time.monotonic()
                d0 = src.dropped_input_frame_count
                while not self._stop.is_set():
                    x = src.read(bs)              # (2, bs) float32
                    now = time.monotonic()
                    t0 = time.perf_counter()
                    y = self._dsp_block(x, now)
                    dsp = time.perf_counter() - t0
                    self.load = 0.9 * self.load + 0.1 * (dsp / block_s)
                    snk.write(y, sr)
                    self.rounds += 1
                    if now - last > 2 * block_s:
                        self.late += 1
                    last = now
                    self.dropped = src.dropped_input_frame_count - d0
        except Exception as e:
            self.error = f"audio through stopped: {e}"
        finally:
            self.running = False
            self.ready.set()
            sys.setswitchinterval(old_switch)
            for s in (src, snk):
                try:
                    s.close()
                except Exception:
                    pass

    # ------------------------------------------------------------ DSP
    def _dsp_init(self):
        """State of the per-block processing: the rack in use, the gain
        reached, the sync ring (processed stereo audio, read `delay` frames
        behind the write position) and the block ramp for crossfades."""
        bs, sr = self.block, self.sr
        self._board_now = self.board
        self._gain_now = 0.0                      # fade in from silence
        self._ring_n = int(MAX_DELAY_MS / 1000.0 * sr) + bs
        self._ring = np.zeros((2, self._ring_n), dtype=np.float32)
        self._wpos = 0
        self._delay_now = self._delay_frames()
        self._ramp = np.linspace(0.0, 1.0, bs, dtype=np.float32)
        self._ar = np.arange(bs)
        self._ab_now = float(self.ab)

    def _ring_read(self, delay):
        idx = (self._ar + self._wpos - self.block - delay) % self._ring_n
        return self._ring[:, idx]

    def _dsp_block(self, x, now):
        """One block (2, bs) in, one block out: analysis tap, rack (swapped
        with a one-block crossfade), sync delay (changed with a one-block
        crossfade), gain (ramped), clip."""
        bs, sr, ramp = self.block, self.sr, self._ramp
        self.in_peak = float(np.abs(x).max()) if x.size else 0.0
        mono = x.mean(axis=0).astype(np.float32, copy=False)
        dry = x.copy()
        with self._lock:
            self._blocks.append((mono, now))
            self._pre.append((dry, now))
            if self._rec_req is not None:         # take starts: pre-roll first
                since = now - self._rec_req
                self._rec = [b for b in self._pre if b[1] >= since]
                self._rec_req = None
            elif self._rec is not None:
                self._rec.append((dry, now))
        new = self.board
        if new is not self._board_now:
            a = self._board_now.process(x, sr, reset=False)
            b = new.process(x, sr, reset=False)
            y = a * (1 - ramp) + b * ramp
            self._board_now = new
        else:
            y = self._board_now.process(x, sr, reset=False)
        idx = (self._ar + self._wpos) % self._ring_n
        self._ring[:, idx] = y
        self._wpos = (self._wpos + bs) % self._ring_n
        delay = self._delay_frames()
        if delay != self._delay_now:
            y = self._ring_read(self._delay_now) * (1 - ramp) + self._ring_read(delay) * ramp
            self._delay_now = delay
        elif delay:
            y = self._ring_read(delay)
        # A/B: the tape through against the section, equal power, ramped
        sec = self.player.block(bs, now, ramp)
        ab = min(1.0, max(0.0, float(self.ab)))
        if ab != self._ab_now:
            a = np.linspace(self._ab_now, ab, bs, dtype=np.float32)
            self._ab_now = ab
        else:
            a = ab
        y = y * np.sqrt(1.0 - a) + sec * np.sqrt(a)
        target = 0.0 if self.mute else float(self.gain)
        if target != self._gain_now:
            y = y * np.linspace(self._gain_now, target, bs, dtype=np.float32)
            self._gain_now = target
        elif self._gain_now != 1.0:
            y = y * self._gain_now
        y = np.ascontiguousarray(y, dtype=np.float32)
        np.clip(y, -1.0, 1.0, out=y)
        self.out_peak = float(np.abs(y).max()) if y.size else 0.0
        return y


if __name__ == "__main__":                      # quick check from a terminal
    ins, outs = audio_devices()
    print("inputs:", [s for s, _ in ins])
    print("outputs:", [s for s, _ in outs])
    if len(sys.argv) >= 3:
        la = LiveAudio(sys.argv[1], sys.argv[2],
                       int(sys.argv[3]) if len(sys.argv) > 3 else 256)
        la.start()
        la.ready.wait(10)
        print("error:", la.error, "running:", la.running)
        try:
            while la.running:
                time.sleep(1.0)
                s = la.stats()
                print(f"in {s['in_peak']:.3f} out {s['out_peak']:.3f} load {s['load']:.2f} "
                      f"dropped {s['dropped']} late {s['late']} buffers {s['buffer_ms']:.1f} ms")
        except KeyboardInterrupt:
            pass
        la.stop()
