#!/usr/bin/env python3
"""
gacha_audio.py — live audio through: an input device plays out of an output
device, through a rack of effects, in realtime. Act one of the VHS show: the
deck's sound goes to the PA through the app, clean, so that later the same
path can carry effects and, later still, the generated material.

Built on PortAudio (the sounddevice package) with Python in the loop:

    input device --read--> [analysis tap] -> live rack -> sync delay ----+
    section player (looping stems, one-shots) x sqrt(A/B) --+            |
    song player (a render's stems) -------------------------+-> 4 stem   +-> VST3 inserts -> gain -> limiter -->write--> output
                                                               racks, F1-F4    (gacha_vst)

The sync delay holds the sound back by a settable number of milliseconds so
it lines up with the picture, which arrives late through the capture
device, the decoder and the display (typically 50 to 150 ms). It changes
live without a click: a block is crossfaded from the old to the new delay.

Two blocking streams, an input and an output, with this loop owning every
sample between them: read a block, process, write it. PortAudio's blocking
calls sleep while they wait (pedalboard's AudioStream, used before, busy-
looped inside read() and write() and burned a whole core doing nothing).
The cost is one Python round per block; measured on 256-frame blocks (5.3 ms)
at 48 kHz the loop runs at 1.00x with a three-plugin rack, all pedalboard
plugins take well under a millisecond per block, and it survived a busy GUI
thread with no drops (the interpreter's switch interval is shortened too,
see _run()). Without an input the output's write paces the loop.

Devices: everything PortAudio lists for the platform's host API (ALSA on
Linux, WASAPI on Windows, CoreAudio on macOS). On Windows ASIO comes
first (the sounddevice wheel ships an ASIO-enabled PortAudio, loaded when
SD_ENABLE_ASIO is set before the import, which this module does): an ASIO
driver takes one client and one full-duplex stream, so on ASIO the input and
the output must be the same device and run in a single blocking duplex
stream, at the buffer the driver's control panel sets. Without ASIO the
stream asks for WASAPI exclusive mode (straight to the driver, no system
mixer) and falls back to shared mode with Windows' converter; the audio
thread joins the "Pro Audio" MMCSS class. WASAPI's engine runs 10 ms
periods, so 480 frames at 48 kHz is the drop-free block there; 256 works in
exclusive mode on good interfaces, 128 or less on ASIO. On Linux the "(hw:x,y)"
entries are the cards themselves, exclusive and lowest latency, and cannot
be opened while PipeWire or another app plays through them; "pipewire" and
"default" go through the desktop's routing, shared, about 11 ms of latency.
Input and output may be different cards; their clocks drift a little, so an
occasional over- or underrun over long runs is normal.

    LiveAudio(in_name, out_name).start()   # opens in a thread (~2 s); in_name None = output only
    la.racks["live"].amounts["reverb"]     # gacha_fx.Rack per channel: live in + 4 stems
    la.on["drums"] = False                 # F1-F4: stems in and out (section and song)
    la.song.load(path); la.song.play()     # a generated song, its stems through the racks
    la.beat_s = 0.5                        # the delays' beat
    la.gain, la.mute                       # ramped, click-free
    la.delay_ms = 80                       # audio held back to meet the picture
    la.drain()                             # mono blocks for the analyzer
    la.record_start(10.0); la.record_stop()   # a take: dry input, with pre-roll
    la.player                              # SectionPlayer: stems, next, one-shots
    la.ab = 0.0 .. 1.0                     # A tape through <-> B the section
    la.stats()                             # peaks, load, drops, latency
"""
import sys
import time
from collections import deque

import contextlib
import math
import os
import random
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Limiter
from scipy.signal import resample_poly
if sys.platform == "win32":                # the wheel ships an ASIO-enabled
    os.environ.setdefault("SD_ENABLE_ASIO", "1")   # PortAudio too: load that one
try:
    import sounddevice as sd
    SD_ERROR = ""
except (ImportError, OSError) as e:        # no package, or no libportaudio
    sd = None
    SD_ERROR = str(e)

from gacha_fx import CHANNELS, Rack, STEM_CHANNELS

SR = 48000
PREFERRED_APIS = ("ASIO", "Windows WASAPI", "ALSA", "Core Audio",
                  "Windows DirectSound", "MME")
SWITCH_INTERVAL = 0.001          # s; Python's default 5 ms lets the GUI hold
                                 # the GIL for one whole block
MAX_DELAY_MS = 2000              # the sync delay's ring


def _api_has_devices(idx, devs):
    return any(d["hostapi"] == idx and (d["max_input_channels"] > 0 or
                                        d["max_output_channels"] > 0) for d in devs)


def audio_apis():
    """Names of the host APIs present that have at least one device, best
    first (ASIO, WASAPI, ...). The ASIO build lists an ASIO host API even
    with no ASIO driver installed; empty, it is left out."""
    if sd is None:
        return []
    try:
        apis = sd.query_hostapis()
        devs = sd.query_devices()
    except Exception:
        return []
    names = [a["name"] for i, a in enumerate(apis) if _api_has_devices(i, devs)]
    return [n for n in PREFERRED_APIS if n in names] + \
        [n for n in names if n not in PREFERRED_APIS]


def _hostapi(api=None):
    """Index of the host API to use: `api` by name, else the best present
    that has devices."""
    apis = sd.query_hostapis()
    for i, a in enumerate(apis):
        if api and a["name"] == api:
            return i
    devs = sd.query_devices()
    for want in PREFERRED_APIS:
        for i, a in enumerate(apis):
            if a["name"] == want and _api_has_devices(i, devs):
                return i
    for i, a in enumerate(apis):
        if _api_has_devices(i, devs):
            return i
    return sd.default.hostapi if sd.default.hostapi >= 0 else 0


def audio_status():
    """Why there may be no devices: '' when all is well, else a sentence."""
    if sd is None:
        return ("sounddevice/PortAudio not available: pip install sounddevice "
                f"(Linux: also apt install libportaudio2). [{SD_ERROR}]")
    try:
        if not audio_apis():
            return "PortAudio lists no audio devices on this machine"
    except Exception as e:
        return f"PortAudio: {e}"
    return ""


def audio_devices(api=None):
    """([(short name, device name)], [(short name, device name)]) of the
    inputs and outputs PortAudio can open on host API `api` (the best one
    when None). Short names for combos, full names to open with; on one
    host API the names are unique."""
    if sd is None:
        return [], []
    try:
        idx = _hostapi(api)
        devs = sd.query_devices()
    except Exception:
        return [], []
    ins, outs = [], []
    for d in devs:
        if d["hostapi"] != idx:
            continue
        name = d["name"]
        if d["max_input_channels"] >= 1:
            ins.append((name, name))
        if d["max_output_channels"] >= 2:
            outs.append((name, name))
    return ins, outs


def _device_index(name, kind, api=None):
    """PortAudio device index for an exact name on the host API."""
    idx = _hostapi(api)
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] == idx and d["name"] == name and d[f"max_{kind}_channels"] >= 1:
            return i, d
    raise RuntimeError(f"{name}: not found (unplugged?)")


class SectionPlayer:
    """Looping stems at the stream's rate, swapped on a bar, plus one-shots.
    Everything here is set from the GUI thread and read in the audio
    thread; swaps go through `pending` and happen on a bar boundary.

    A break (`fill()`): from the next beat to the end of the bar, at least
    two beats, the loop is taken over, and a section queued meanwhile drops
    in on that bar. Two kinds, as in a generated song's run-up to a drop:
    "glitch" throws sixteenths of the bar that was playing about, some
    backwards, a stutter roll on the last beat and a gap right before the
    bar; "break" is the song's own transition, a beat-repeat of the bar's
    first slice (accelerating, steady or halving), a gap of silence before
    the drop, and a reverse cymbal swell handed in by the GUI rising under
    it. `fill_state()` tells the picture what is happening."""

    def __init__(self, sr):
        self.sr = sr
        self.stems = None                    # {name: (2, n) float32 at sr}
        self.events = []                     # the section's timeline (seconds)
        self.bpm = 120.0
        self.bars = 8
        self.n = 0                           # loop length in frames
        self.pos = 0                         # play position in frames
        self.pending = None                  # (stems, events, bpm, bars, bar_t0) for the next bar
        self._lead = 0                       # frames of silence before a timed start
        self.on = {"drums": True, "layers": True, "chops": True, "events": True}
        self._gain_now = {}                  # stem -> gain reached (ramps)
        self.shots = []                      # [(clip (2, n), frames played)]
        self.shot_queue = []                 # clips waiting for the next beat
        self.bursts = []                     # [(clip, played)]: now, and not under A/B
        self.loop_t0 = None                  # monotonic time the loop last wrapped
        self.swapped = False                 # a pending section just went live
        self.beat_frames = 0
        self.fill_armed = None               # (kind, swell): a break asked, next beat
        self._fill = None                    # dict: the break playing, see _fill_start
        self.dropped = False                 # a break just ended in a swap (GUI resets)

    @property
    def playing(self):
        return self.stems is not None

    @property
    def filling(self):
        return self._fill is not None

    def fill(self, kind="glitch", swell=None):
        """A break of `kind` ("glitch" or "break"), from the next beat: see
        the class note. `swell`: a (2, n) float32 clip at sr to rise into the
        drop (the break kind), its end on the bar. No-op with no section."""
        if self.playing:
            self.fill_armed = (kind, swell)

    def fill_state(self):
        """What the break is doing now, for the picture: {kind, k (retrigger
        count so far), frac (0..1 into the slice), gap, swell (0..1 up to the
        drop)}, or None."""
        f = self._fill
        if f is None:
            return None
        k = min(f["k"], len(f["idx"]) - 1)
        sw = 0.0
        if f["swell"] is not None and k >= f["swell_k0"]:
            sw = (k - f["swell_k0"]) / max(1, f["swell"].shape[1])
        return {"kind": f["kind"], "k": int(f["retrig"][k]), "frac": float(f["frac"][k]),
                "gap": bool(f["retrig"][k] < 0), "swell": min(1.0, sw)}

    # ---- the break's index map, built when it starts (audio thread)
    def _fill_map(self, start, beats, rng, kind):
        """(idx, gain, retrig, frac) for `beats` beats from `start`, drawn
        from the bar `start` sits in. glitch: random sixteenths, some
        reversed, repeats, the last beat a stutter roll, a sixteenth of
        silence. break: a beat-repeat of the bar's first slice, per beat
        1/8 1/16 1/32 1/64 (accelerate), 1/16 throughout (stutter) or 1/4
        1/8 1/16 1/32 (halving), then half a beat or a beat of silence.
        retrig counts the slices (-1 in the gap), frac runs 0..1 in each."""
        bf, n = self.beat_frames, self.n
        sx = max(1, bf // 4)                            # a sixteenth
        bar0 = (start // (4 * bf)) * 4 * bf
        idx, gain, retrig, frac = [], [], [], []
        count = [0]

        def piece(src, m, rev=False):
            a = np.arange(src, src + m) if not rev else np.arange(src + m - 1, src - 1, -1)
            idx.append(a % n)
            g = np.ones(m, np.float32)
            f = min(int(0.002 * self.sr), m // 2)      # 2 ms fades: no clicks
            if f:
                g[:f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
                g[-f:] = np.linspace(1.0, 0.0, f, dtype=np.float32)
            gain.append(g)
            retrig.append(np.full(m, count[0], np.int32))
            frac.append(np.linspace(0.0, 1.0, m, endpoint=False, dtype=np.float32))
            count[0] += 1

        def silence(m):
            idx.append(np.zeros(m, np.int64))
            gain.append(np.zeros(m, np.float32))
            retrig.append(np.full(m, -1, np.int32))
            frac.append(np.zeros(m, np.float32))

        if kind == "break":
            flavour = rng.choice(["accelerate", "accelerate", "stutter", "halving"])
            full = {"accelerate": [1, 2, 4, 8, 16], "halving": [1, 1, 2, 4, 8],
                    "stutter": [4, 4, 4, 4, 4]}[flavour]
            divs = full[-beats:] if beats <= 4 or flavour != "accelerate" else full
            if beats == 2 and flavour == "accelerate":
                divs = [4, 8]                           # two beats: 1/16 then 1/32
            gap = int(bf * (0.5 if beats <= 2 else rng.choice([0.5, 1.0])))
            total = beats * bf - gap
            for div in divs:
                m = max(1, bf // div)
                for _ in range(div):
                    if sum(len(a) for a in idx) + m > total:
                        break
                    piece(bar0, m)
            silence(gap)
        else:
            prev = None
            for _ in range(4 * (beats - 1)):            # every beat but the last
                if prev is not None and rng.random() < 0.35:
                    src = prev                          # stutter: the same slice again
                else:
                    src = bar0 + rng.randrange(16) * sx
                piece(src, sx, rev=rng.random() < 0.3)
                prev = src
            piece(bar0, sx)                             # last beat: the roll
            for _ in range(2):
                piece(bar0, sx // 2)
            for _ in range(4):
                piece(bar0, sx // 4)
            silence(sx)                                 # the gap before the drop
        idx, gain = np.concatenate(idx), np.concatenate(gain)
        retrig, frac = np.concatenate(retrig), np.concatenate(frac)
        want = beats * bf                               # exactly to the bar
        if len(idx) < want:
            pad = want - len(idx)
            idx = np.concatenate([idx, np.zeros(pad, np.int64)])
            gain = np.concatenate([gain, np.zeros(pad, np.float32)])
            retrig = np.concatenate([retrig, np.full(pad, -1, np.int32)])
            frac = np.concatenate([frac, np.zeros(pad, np.float32)])
        return idx[:want], gain[:want], retrig[:want], frac[:want]

    def _fill_start(self, i0):
        """The next beat has just passed at `i0`: the break runs from here to
        the bar boundary at least two beats away."""
        kind, swell = self.fill_armed
        bf = self.beat_frames
        beat = i0 // bf                                 # the beat we are in
        left = 4 - (beat % 4)                           # beats to the bar, this one included
        beats = left if left >= 2 else left + 4
        end = (beat + beats) * bf                       # the bar boundary, in loop frames
        idx, gain, retrig, frac = self._fill_map(beat * bf, beats, random.Random(), kind)
        late = i0 - beat * bf                           # the block boundary: a few ms in
        idx, gain, retrig, frac = idx[late:], gain[late:], retrig[late:], frac[late:]
        k0 = 0
        if swell is not None:                           # its end on the drop
            if swell.shape[1] > len(idx):
                swell = swell[:, swell.shape[1] - len(idx):]
            k0 = len(idx) - swell.shape[1]
        self._fill = {"idx": idx, "gain": gain, "retrig": retrig, "frac": frac, "k": 0,
                      "end": end, "kind": kind, "swell": swell, "swell_k0": k0}
        self.fill_armed = None

    def queue(self, stems, events, bpm, bars, bar_t0=None):
        """The next section: on the loop's next bar while one plays, else
        from the next block, or, `bar_t0` given (monotonic seconds, the
        Link session's bar), exactly then: silence up to it inside the
        block, or, when it has just passed, the loop entered that far in."""
        self.pending = (stems, events, bpm, bars, bar_t0)

    def waiting_for(self):
        """The bar time a queued section waits for, or None."""
        p = self.pending
        return p[4] if p is not None and p[0] != "clear" and self.stems is None else None

    def clear(self):
        self.pending = ("clear",)

    def fire(self, clip, now=False):
        """A one-shot (2, n) float32 at sr, on the next beat (or now if no
        section plays). `now`: a burst, played from the next block on and
        mixed outside the A/B crossfade, so it is heard over the tape too."""
        if now:
            self.bursts.append((clip, 0))
        else:
            self.shot_queue.append(clip)

    def pos_s(self):
        return self.pos / self.sr

    def _swap(self, now=None, bs=0):
        p, self.pending = self.pending, None
        if p[0] == "clear":
            self.stems, self.events, self.n, self.pos = None, [], 0, 0
            return
        stems, events, bpm, bars, bar_t0 = p
        from_silence = self.stems is None
        self.stems, self.events, self.bpm, self.bars = stems, events, bpm, bars
        self.n = next(iter(stems.values())).shape[1]
        self.pos, self._lead = 0, 0
        self.beat_frames = int(round(60.0 / bpm * self.sr))
        self.loop_t0 = time.monotonic() if now is None else now
        if bar_t0 is not None and from_silence and now is not None:
            if bar_t0 > now:                          # inside this block: silence first
                self._lead = min(bs, int(round((bar_t0 - now) * self.sr)))
            else:                                     # the bar just passed: enter late
                self.pos = int((now - bar_t0) * self.sr) % self.n
            self.loop_t0 = now + (self._lead - self.pos) / self.sr
        self.swapped = True

    def block(self, bs, now, ramp):
        """The next bs frames of the section, {stem: (2, bs) float32}, the
        one-shots in "events". Stem on/off is applied by LiveAudio after the
        racks. Empty dict when nothing plays."""
        out = {}
        bar_f = 4 * self.beat_frames
        breaking = self.fill_armed or self._fill is not None
        if self.pending is not None and (self.stems is None or (self.pos == 0 and not breaking)):
            wait = self.waiting_for()
            if wait is None or now + bs / self.sr >= wait:   # a timed start: not yet
                self._swap(now, bs)
        if self._fill is not None:
            on_beat = self._fill_block(out, bs)
        elif self.stems is not None:
            lead = min(self._lead, bs)                # a timed start: silence up to it
            self._lead -= lead
            m = bs - lead
            i0 = self.pos
            idx = (np.arange(m) + i0) % self.n
            for name, data in self.stems.items():
                blk = data[:, idx]
                if lead:
                    blk = np.concatenate([np.zeros((2, lead), np.float32), blk], axis=1)
                out[name] = blk
            self.pos = (i0 + m) % self.n
            if i0 + m >= self.n:                      # wrapped inside this block
                self.loop_t0 = now + (self.n - i0 + lead) / self.sr
            # a bar boundary inside this block: a pending section comes in
            # at the next block (5 ms off at most), its position on the bar
            if self.pending is not None and bar_f and not breaking \
                    and (i0 // bar_f) != ((i0 + m) // bar_f):
                self.pos = 0                          # so the swap lands on the bar
            on_beat = (i0 // self.beat_frames) != ((i0 + m) // self.beat_frames) \
                if self.beat_frames else True
            if self.fill_armed and on_beat and self.beat_frames:
                self._fill_start(self.pos)            # the break, from the next block
        else:
            on_beat = True
        if self.shot_queue and on_beat:
            self.shots.extend((c, 0) for c in self.shot_queue)
            self.shot_queue = []
        keep = []
        if self.shots:
            ev = out.get("events")
            ev = np.zeros((2, bs), np.float32) if ev is None else ev.copy()
            for clip, k in self.shots:
                n = min(bs, clip.shape[1] - k)
                ev[:, :n] += clip[:, k:k + n]
                if k + n < clip.shape[1]:
                    keep.append((clip, k + n))
            out["events"] = ev
        self.shots = keep
        if self.bursts:
            out["burst"], self.bursts = self._sum_clips(self.bursts, bs)
        return out

    def _fill_block(self, out, bs):
        """One block of the break into `out`; `pos` follows the slice that
        sounds, so the picture glitches with it. At the end: the pending
        section, or the loop on from the bar. Returns whether a beat passed."""
        f = self._fill
        idx, gain, k, end = f["idx"], f["gain"], f["k"], f["end"]
        m = min(bs, len(idx) - k)
        sl, g = idx[k:k + m], gain[k:k + m]
        for name, data in self.stems.items():
            blk = np.zeros((2, bs), np.float32)
            blk[:, :m] = data[:, sl] * g
            out[name] = blk
        swell = f["swell"]
        if swell is not None and k + m > f["swell_k0"]:   # the swell, under A/B
            a = max(0, k - f["swell_k0"])                  # like the stems
            b = min(swell.shape[1], k + m - f["swell_k0"])
            if b > a:
                ev = out.get("events")
                ev = np.zeros((2, bs), np.float32) if ev is None else ev
                o = max(0, f["swell_k0"] - k)
                ev[:, o:o + b - a] += swell[:, a:b]
                out["events"] = ev
        on_beat = (k // self.beat_frames) != ((k + m) // self.beat_frames)
        if k + m < len(idx):
            f["k"] = k + m
            self.pos = int(sl[-1]) if g[-1] > 0 else self.pos
        else:                                         # done: the drop
            self._fill = None
            self.pos = end % self.n
            if self.pending is not None:
                self._swap()
                self.dropped = True
            elif self.pos == 0:
                self.loop_t0 = time.monotonic()
            on_beat = True
        return on_beat

    @staticmethod
    def _sum_clips(clips, bs):
        """Mix the next `bs` frames of every (clip, played) into one (2, bs)
        block; returns it and the clips still sounding."""
        blk = np.zeros((2, bs), np.float32)
        keep = []
        for clip, k in clips:
            n = min(bs, clip.shape[1] - k)
            blk[:, :n] += clip[:, k:k + n]
            if k + n < clip.shape[1]:
                keep.append((clip, k + n))
        return blk, keep


class SongPlayer:
    """A generated song into the live mix: its stems file (8 channels, next
    to the wav) or, failing that, the mix as the "layers" stem. Loaded and
    resampled on a thread; play/pause/stop/seek from the GUI thread, block()
    from the audio thread."""

    STEMS = STEM_CHANNELS

    def __init__(self, sr, on):
        self.sr = int(sr)
        self.on = on                          # shared with SectionPlayer
        self.stems = None                     # {stem: (2, n) float32 at sr}
        self.n = 0
        self.pos = 0
        self.state = "stopped"                # loading / playing / paused / stopped
        self.path = None
        self.error = None
        self.ended = False
        self.has_stems = False
        self._gen = 0
        self._want_play = False

    def load(self, path, play=True):
        self._gen += 1
        self.state, self.stems, self.pos, self.n = "loading", None, 0, 0
        self.ended, self.error, self.path = False, None, str(path)
        self._want_play = play
        threading.Thread(target=self._load, args=(str(path), self._gen),
                         daemon=True, name="gacha-song").start()

    def _load(self, path, gen):
        try:
            p = Path(path)
            stems_file = p.with_name(p.stem + "_stems.flac")
            if stems_file.is_file():
                data, fsr = sf.read(str(stems_file), dtype="float32", always_2d=True)
                raw = {name: data[:, 2 * i:2 * i + 2] for i, name in enumerate(self.STEMS)
                       if data.shape[1] >= 2 * i + 2}
                has = True
            else:
                data, fsr = sf.read(str(p), dtype="float32", always_2d=True)
                if data.shape[1] == 1:
                    data = np.repeat(data, 2, axis=1)
                raw, has = {"layers": data[:, :2]}, False
            stems = {}
            for name, d in raw.items():
                if fsr != self.sr:
                    g = math.gcd(self.sr, int(fsr))
                    d = resample_poly(d, self.sr // g, int(fsr) // g, axis=0)
                stems[name] = np.ascontiguousarray(d.T.astype(np.float32))
        except Exception as e:
            if gen == self._gen:
                self.error, self.state = str(e), "stopped"
            return
        if gen != self._gen:                  # another load came in meanwhile
            return
        self.n = next(iter(stems.values())).shape[1]
        self.pos = 0
        self.has_stems = has
        self.stems = stems
        self.state = "playing" if self._want_play else "paused"

    @property
    def loaded(self):
        return self.stems is not None

    def play(self):
        if self.stems is None:
            self._want_play = True
            return
        if self.ended or self.pos >= self.n:
            self.pos, self.ended = 0, False
        self.state = "playing"

    def pause(self):
        if self.state == "playing":
            self.state = "paused"
        else:
            self._want_play = False

    def stop(self):
        self.state = "stopped" if self.stems is not None else self.state
        self._want_play = False
        self.pos = 0

    def seek_ms(self, ms):
        self.pos = max(0, min(self.n, int(ms / 1000.0 * self.sr)))
        self.ended = False

    def position_ms(self):
        return self.pos / self.sr * 1000.0

    def duration_ms(self):
        return self.n / self.sr * 1000.0

    def block(self, bs):
        """{stem: (2, bs)} for the next block, or None when not playing."""
        if self.state != "playing" or self.stems is None:
            return None
        i0 = self.pos
        end = min(self.n, i0 + bs)
        out = {}
        for name, data in self.stems.items():
            blk = np.zeros((2, bs), np.float32)
            blk[:, : end - i0] = data[:, i0:end]
            out[name] = blk
        self.pos = end
        if end >= self.n:
            self.state, self.ended = "stopped", True
        return out


class LiveAudio:
    """The through path, on its own thread. Attributes read from any thread:
    running, error, and the stats. Set from any thread: the racks' amounts,
    on, gain, mute, ab, beat_s, delay_ms; song and player have their own
    control methods."""

    def __init__(self, in_name, out_name, block=256, sr=SR, exclusive=True, api=None):
        self.in_name, self.out_name = (in_name or None), out_name   # no input: output only
        self.block, self.sr = int(block), int(sr)
        self.exclusive = bool(exclusive)     # Windows: WASAPI exclusive first
        self.api = api                       # host API name, None = the best present
        self.mode = ""                       # how the devices were opened, for the status
        self.racks = {ch: Rack(self.sr) for ch in CHANNELS}   # live in + the 4 stems
        self.beat_s = 0.5                    # the delays follow this
        self.gain = 1.0
        self.mute = False
        self.delay_ms = 0.0                  # sync: hold the sound back this much
        self.ab = 0.0                        # 0 = tape through only, 1 = section only
        self.hold_tape = False               # the pedal leaves the tape's sound whole
                                             # (the webcam on B has none to bring in)
        self.player = SectionPlayer(self.sr)
        self.on = self.player.on             # stems in/out, for section and song alike
        self.song = SongPlayer(self.sr, self.on)
        self._limiter = Limiter(threshold_db=-1.0, release_ms=100.0)
        self.inserts = None                  # gacha_vst.InsertChain on the master, or None
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
        self.dropped = 0                     # input overflows (blocks lost to late reads)
        self.underruns = 0                   # output underflows (the device starved)
        self.late = 0                        # loop rounds slower than 2 blocks
        self.rounds = 0
        self.buffer_ms = 2000.0 * self.block / self.sr
        self.out_latency_ms = 1000.0 * self.block / self.sr   # what is written but not yet heard

    # ------------------------------------------------------------ control
    def start(self):
        """Open the devices. On ASIO the driver drives us: the stream is
        opened and started here, on the calling thread (ASIO drivers are COM
        objects wanting the thread that made them, and its message loop),
        and every buffer arrives in a callback on the driver's thread; the
        blocking read/write layer, which timed out on real drivers (-9987),
        is not used. Other host APIs keep the blocking loop on a thread."""
        self._cb_stream = None
        if sd is not None and self._api_name() == "ASIO":
            try:
                src, snk = self._open(callback=True)
                self._dsp_init()
                snk.start()
                lat = snk.latency
                self.out_latency_ms = 1000.0 * float(lat[1] if isinstance(lat, tuple) else lat)
                self.buffer_ms = 1000.0 * float(sum(lat) if isinstance(lat, tuple) else lat)
                self._cb_stream = snk
                self.mode += " callback"
                self.running = True
            except Exception as e:
                self.error = self._explain(e)
            self.ready.set()
            return
        self._thread = threading.Thread(target=self._run, name="gacha-audio",
                                        daemon=True)
        self._thread.start()

    def _cb_duplex(self, indata, outdata, frames, _time, status):
        self._cb_process(indata, outdata, frames, status)

    def _cb_out(self, outdata, frames, _time, status):
        self._cb_process(None, outdata, frames, status)

    def _cb_process(self, indata, outdata, frames, status):
        """One driver buffer in, one out, on the driver's thread."""
        try:
            if status:
                if status.input_overflow:
                    self.dropped += 1
                if status.output_underflow:
                    self.underruns += 1
            now = time.monotonic()
            t0 = time.perf_counter()
            if indata is None:
                x = np.zeros((2, frames), np.float32)
            else:
                x = indata.T if indata.shape[1] == 2 else np.repeat(indata.T, 2, axis=0)
                x = np.ascontiguousarray(x, dtype=np.float32)
            if frames != self.block:                 # a driver handing odd sizes
                y = np.zeros((2, frames), np.float32)
                n = min(frames, self.block)
                xb = np.zeros((2, self.block), np.float32)
                xb[:, :n] = x[:, :n]
                y[:, :n] = self._dsp_block(xb, now)[:, :n]
            else:
                y = self._dsp_block(x, now)
            outdata[:] = y.T
            dsp = time.perf_counter() - t0
            self.load = 0.9 * self.load + 0.1 * (dsp / (frames / self.sr))
            self.rounds += 1
        except Exception as e:                       # never let the driver see one
            self.error = f"audio callback: {e}"
            outdata.fill(0)

    def _api_name(self):
        try:
            return sd.query_hostapis(_hostapi(self.api))["name"]
        except Exception:
            return ""

    @staticmethod
    def _explain(e):
        msg = str(e)
        low = msg.lower()
        if "unavailable" in low or "busy" in low:
            msg += (" (in use by PipeWire or another app? the 'pipewire' or "
                    "'default' device is shared)")
        if "-9999" in msg or "unanticipated" in low:
            msg += (" (the ASIO driver refused: is it open in another app or in its "
                    "control panel? set the interface to 48 kHz there, or pick the "
                    "WASAPI driver)")
        return msg

    def _set_sr(self, sr):
        """Run at another rate than asked (an ASIO interface locked to 44.1
        kHz): the racks, the players and the limiter are rebuilt for it, the
        knob values kept. Sections and songs are resampled to `self.sr`."""
        sr = int(sr)
        if sr == self.sr:
            return
        old_racks = self.racks
        self.sr = sr
        self.racks = {ch: Rack(sr) for ch in CHANNELS}
        for ch, r in self.racks.items():
            r.amounts.update(old_racks[ch].amounts)
            r.vol = old_racks[ch].vol
        self.player = SectionPlayer(sr)
        self.player.on = self.on
        self.song = SongPlayer(sr, self.on)
        self._limiter = Limiter(threshold_db=-1.0, release_ms=100.0)
        self.buffer_ms = 2000.0 * self.block / sr
        self.out_latency_ms = 1000.0 * self.block / sr

    def stop(self):
        self._stop.set()
        st = getattr(self, "_cb_stream", None)
        if st is not None:                           # ASIO: same thread as start()
            self._cb_stream = None
            for op in (st.stop, st.close):
                try:
                    op()
                except Exception:
                    pass
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

    def ping(self):
        """Arm the round-trip test: the next block puts a short 1 kHz burst
        on the output, and the input is watched for a second for it to come
        back through a cable (output -> input). The result lands in
        `ping_result`: {"ok", "ms", "frames", "msg"}."""
        self.ping_result = None
        self._ping = {"state": "armed", "pos": 0, "buf": []}

    def _ping_finish(self, pg):
        """Where did the burst come back? Cross-correlation against the
        template, per channel, the stronger one wins. The capture began one
        block after the block that carried the burst's start."""
        bs, sr = self.block, self.sr
        cap = np.concatenate(pg["buf"], axis=1)             # (2, n)
        tpl = self._ping_tpl
        best = None
        for ch in range(cap.shape[0]):
            corr = np.correlate(cap[ch], tpl, "valid")
            if not corr.size:
                continue
            a = np.abs(corr)
            idx = int(a.argmax())          # the one copy: the input is muted meanwhile
            peak = float(a[idx])
            floor = float(np.median(a)) + 1e-9
            if best is None or peak / floor > best[0]:
                best = (peak / floor, idx, peak)
        if best is None or best[0] < 12.0 or best[2] < 1e-3:
            self.ping_result = {"ok": False, "ms": None, "frames": None,
                                "msg": "no burst came back: loop an output into the "
                                       "live input with a cable, and turn its level up"}
            return
        frames = bs + best[1]
        self.ping_result = {"ok": True, "frames": frames, "ms": 1000.0 * frames / sr,
                            "msg": f"round trip {1000.0 * frames / sr:.1f} ms "
                                   f"({frames} frames at {sr} Hz, block {bs})"}

    def stats(self):
        return {"in_peak": self.in_peak, "out_peak": self.out_peak,
                "load": self.load, "dropped": self.dropped, "late": self.late,
                "underruns": self.underruns, "buffer_ms": self.buffer_ms,
                "delay_ms": float(self.delay_ms), "mode": self.mode}

    # ------------------------------------------------------------ thread
    def _delay_frames(self):
        ms = min(MAX_DELAY_MS, max(0.0, float(self.delay_ms)))
        return int(round(ms / 1000.0 * self.sr))

    def _open(self, callback=False):
        """(src, snk): two blocking streams, or one duplex stream on ASIO
        (src is snk then), with our callbacks attached when `callback`.
        Raises with every mode's error when none opens."""
        if sd is None:
            raise RuntimeError("sounddevice/PortAudio missing: pip install sounddevice "
                               f"(Linux: apt install libportaudio2) [{SD_ERROR}]")
        api_name = sd.query_hostapis(_hostapi(self.api))["name"]
        if api_name == "ASIO":
            if self.in_name and self.in_name != self.out_name:
                raise RuntimeError("ASIO: the input and the output must be the same "
                                   "device (one driver, one stream)")
            modes = [("ASIO", None)]
        elif api_name == "Windows WASAPI":
            modes = []
            if self.exclusive:
                modes.append(("WASAPI exclusive", sd.WasapiSettings(exclusive=True)))
            modes.append(("WASAPI shared", sd.WasapiSettings(exclusive=False, auto_convert=True)))
        else:
            modes = [(api_name, None)]
        errors = []
        rates = [self.sr]
        try:                                  # the device's own rate as a fallback
            _j, d = _device_index(self.out_name, "output", self.api)
            native = int(d["default_samplerate"])
            if native and native != self.sr:
                rates.append(native)
        except Exception:
            pass
        for sr in rates:
            for label, extra in modes:
                try:
                    if sr != self.sr:
                        self._set_sr(sr)
                    src, snk = self._open_streams(extra, duplex=(api_name == "ASIO"),
                                                  callback=callback)
                    self.mode = label if sr == SR else f"{label} @ {sr} Hz"
                    if errors:
                        self.mode += f" ({errors[-1]})"
                    return src, snk
                except Exception as e:
                    errors.append(f"{label} {sr} Hz: {str(e).strip()[:90]}")
        raise RuntimeError("; ".join(errors))

    def _open_streams(self, extra, duplex=False, callback=False):
        # two blocks of device buffer for the blocking loop; on ASIO the
        # suggested latency becomes the driver's buffer size, so one block
        # there, or the control panel shows twice what was asked
        lat = (1 if duplex else 2) * self.block / self.sr
        # in callback mode PortAudio adapts the driver's buffer to our block
        # size; in blocking mode on ASIO the driver's size is taken as is
        bs = 0 if (duplex and not callback) else self.block
        j, _d = _device_index(self.out_name, "output", self.api)
        if duplex and self.in_name:                    # ASIO: one stream, in and out
            i, d = _device_index(self.in_name, "input", self.api)
            st = sd.Stream(device=(i, j), samplerate=self.sr, blocksize=bs,
                           channels=(min(2, int(d["max_input_channels"])), 2),
                           dtype="float32", latency=lat, extra_settings=extra,
                           callback=self._cb_duplex if callback else None)
            return st, st
        src = None
        if self.in_name:
            i, d = _device_index(self.in_name, "input", self.api)
            src = sd.InputStream(device=i, samplerate=self.sr, blocksize=bs,
                                 channels=min(2, int(d["max_input_channels"])),
                                 dtype="float32", latency=lat, extra_settings=extra)
        try:
            snk = sd.OutputStream(device=j, samplerate=self.sr, blocksize=bs,
                                  channels=2, dtype="float32", latency=lat,
                                  extra_settings=extra,
                                  callback=self._cb_out if callback else None)
        except Exception:
            if src is not None:
                src.close()
            raise
        return src, snk

    @staticmethod
    def _boost_priority():
        """Best effort: the audio thread above the rest. Windows: the MMCSS
        "Pro Audio" class (what DAWs use) and time-critical priority; Linux:
        SCHED_FIFO if the user may (rtkit / limits.conf), else nothing."""
        try:
            if sys.platform == "win32":
                import ctypes
                task_index = ctypes.c_ulong(0)
                avrt = ctypes.windll.avrt
                if not avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(task_index)):
                    k32 = ctypes.windll.kernel32
                    k32.SetThreadPriority(k32.GetCurrentThread(), 15)   # TIME_CRITICAL
            elif hasattr(os, "sched_setscheduler"):
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
        except Exception:
            pass

    def _run(self):
        old_switch = sys.getswitchinterval()
        try:
            src, snk = self._open()
        except Exception as e:
            self.error = self._explain(e)
            self.ready.set()
            return
        sys.setswitchinterval(SWITCH_INTERVAL)
        self._boost_priority()
        bs, sr = self.block, self.sr
        block_s = bs / sr
        self._dsp_init()
        silence = np.zeros((2, bs), np.float32)
        try:
            with contextlib.ExitStack() as stack:
                if src is not None and src is not snk:
                    stack.enter_context(src)
                stack.enter_context(snk)
                if src is snk:                        # duplex: (in, out) latencies
                    self.out_latency_ms = 1000.0 * float(snk.latency[1])
                    self.buffer_ms = 1000.0 * float(sum(snk.latency))
                else:
                    self.out_latency_ms = 1000.0 * float(snk.latency)
                    self.buffer_ms = 1000.0 * ((src.latency if src is not None else 0.0) + snk.latency)
                self.running = True
                self.ready.set()
                last = time.monotonic()
                while not self._stop.is_set():
                    # with an input the read paces the loop; without one the
                    # output's write does. Both sleep while they wait.
                    if src is not None:
                        data, overflowed = src.read(bs)     # (bs, ch) float32
                        if overflowed:
                            self.dropped += 1
                        x = data.T if data.shape[1] == 2 else np.repeat(data.T, 2, axis=0)
                    else:
                        x = silence
                    now = time.monotonic()
                    t0 = time.perf_counter()
                    y = self._dsp_block(x, now)
                    dsp = time.perf_counter() - t0
                    self.load = 0.9 * self.load + 0.1 * (dsp / block_s)
                    if snk.write(np.ascontiguousarray(y.T)):
                        self.underruns += 1
                    self.rounds += 1
                    # `late` only means something when the input paces the
                    # loop; a write-paced loop returns in device-period
                    # bursts, which are not lateness (underruns are)
                    if src is not None and now - last > 2 * block_s:
                        self.late += 1
                    last = now
        except Exception as e:
            self.error = f"audio stream stopped: {e}"
        finally:
            self.running = False
            self.ready.set()
            sys.setswitchinterval(old_switch)
            for st in ({id(x): x for x in (src, snk) if x is not None}.values()):
                try:
                    st.close()
                except Exception:
                    pass

    # ------------------------------------------------------------ DSP
    def _dsp_init(self):
        """State of the per-block processing: the rack in use, the gain
        reached, the sync ring (processed stereo audio, read `delay` frames
        behind the write position) and the block ramp for crossfades."""
        bs, sr = self.block, self.sr
        self._stem_gain = {}                      # stem -> on/off gain reached
        self._gain_now = 0.0                      # fade in from silence
        # the round-trip test: a 4 ms 1 kHz burst at -10 dBFS, Hann-windowed
        n = int(0.004 * self.sr)
        t = np.arange(n) / self.sr
        self._ping_tpl = (0.3 * np.sin(2 * np.pi * 1000.0 * t) * np.hanning(n)).astype(np.float32)
        self._ping = None
        self._ring_n = int(MAX_DELAY_MS / 1000.0 * sr) + bs
        self._ring = np.zeros((2, self._ring_n), dtype=np.float32)
        self._wpos = 0
        self._delay_now = self._delay_frames()
        self._ramp = np.linspace(0.0, 1.0, bs, dtype=np.float32)
        self._ar = np.arange(bs)
        self._ab_now = float(self.ab)
        self._tw_now = self._tape_weight(self._ab_now)

    def _tape_weight(self, ab):
        """The tape's gain under the pedal: equal power against the section,
        or 1 while hold_tape keeps its sound whole."""
        return 1.0 if self.hold_tape else float(np.sqrt(1.0 - ab))

    def _ring_read(self, delay):
        idx = (self._ar + self._wpos - self.block - delay) % self._ring_n
        return self._ring[:, idx]

    def _dsp_block(self, x, now):
        """One block (2, bs) in, one block out: analysis tap, the live rack,
        sync delay (changed with a one-block crossfade), the section and the
        song through the stem racks with F1-F4 gates, gain (ramped), limiter."""
        bs, sr, ramp = self.block, self.sr, self._ramp
        self.in_peak = float(np.abs(x).max()) if x.size else 0.0
        mono = x.mean(axis=0).astype(np.float32, copy=False)
        dry = x.copy()
        pg = self._ping
        if pg is not None and pg["state"] in ("sending", "listen"):   # the echo, dry input
            pg["buf"].append(dry)          # from the block after the burst's first one
            if len(pg["buf"]) * bs >= sr:                 # a second is plenty
                self._ping = None
                self._ping_finish(pg)
        with self._lock:
            self._blocks.append((mono, now))
            self._pre.append((dry, now))
            if self._rec_req is not None:         # take starts: pre-roll first
                since = now - self._rec_req
                self._rec = [b for b in self._pre if b[1] >= since]
                self._rec_req = None
            elif self._rec is not None:
                self._rec.append((dry, now))
        for r in self.racks.values():
            r.beat_s = self.beat_s
        y = self.racks["live"].process(x)
        idx = (self._ar + self._wpos) % self._ring_n
        self._ring[:, idx] = y
        self._wpos = (self._wpos + bs) % self._ring_n
        delay = self._delay_frames()
        if delay != self._delay_now:
            y = self._ring_read(self._delay_now) * (1 - ramp) + self._ring_read(delay) * ramp
            self._delay_now = delay
        elif delay:
            y = self._ring_read(delay)
        # A/B: the tape through against the section, equal power, ramped;
        # the section's share is applied before its racks, the song (a
        # render playing through the app) is not under A/B at all
        ab = min(1.0, max(0.0, float(self.ab)))
        if ab != self._ab_now:
            a = np.linspace(self._ab_now, ab, bs, dtype=np.float32)
            self._ab_now = ab
        else:
            a = ab
        # the tape's share: the pedal's, or whole while the pedal only moves
        # the picture (hold_tape: the webcam on B); ramped over the block
        tw = self._tape_weight(ab)
        if tw != self._tw_now:
            y = y * np.linspace(self._tw_now, tw, bs, dtype=np.float32)
            self._tw_now = tw
        else:
            y = y * tw
        if pg is not None:
            y = y * 0.0          # round-trip test: no input to the output, or the
                                 # cable feeds the burst back round and round
        sec = self.player.block(bs, now, ramp)
        song = self.song.block(bs)
        sec_w = np.sqrt(a)
        burst = sec.get("burst")                    # the bursts: into the events
        for stem in STEM_CHANNELS:                  # rack, but not under A/B
            blk = sec.get(stem)
            blk = None if blk is None else blk * sec_w
            if burst is not None and stem == "events":
                blk = burst if blk is None else blk + burst
            if song is not None and stem in song:
                blk = song[stem] if blk is None else blk + song[stem]
            if blk is None:
                blk = np.zeros((2, bs), np.float32)     # tails keep ringing
            blk = self.racks[stem].process(blk)
            on = 1.0 if self.on.get(stem, True) else 0.0
            cur = self._stem_gain.get(stem, on)
            if cur != on:                                # F1-F4: a ramp, no click
                blk = blk * np.linspace(cur, on, bs, dtype=np.float32)
                self._stem_gain[stem] = on
            elif not on:
                continue
            y = y + blk
        # master inserts (VST3 plugins): after the racks, before the gain and
        # the limiter, kept out of the round-trip test so its burst stays clean;
        # nothing at all is done here while no plugin is loaded
        ins = self.inserts
        if ins is not None and ins.items and pg is None:
            y = ins.process(y, sr)
        target = 0.0 if self.mute else float(self.gain)
        if target != self._gain_now:
            y = y * np.linspace(self._gain_now, target, bs, dtype=np.float32)
            self._gain_now = target
        elif self._gain_now != 1.0:
            y = y * self._gain_now
        y = np.ascontiguousarray(y, dtype=np.float32)
        if pg is not None and pg["state"] in ("armed", "sending"):
            # the burst, spread over as many blocks as it needs; listening
            # starts with the block after the one that carried its start
            tpl, p0 = self._ping_tpl, pg["pos"]
            n = min(bs, len(tpl) - p0)
            y[:, :n] += tpl[p0:p0 + n]
            pg["pos"] = p0 + n
            pg["state"] = "sending" if pg["pos"] < len(tpl) else "listen"
        y = self._limiter.process(y, sr, reset=False)      # the racks can get loud
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
