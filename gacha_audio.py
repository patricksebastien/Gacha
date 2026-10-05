#!/usr/bin/env python3
"""
gacha_audio.py — live audio through: an input device plays out of an output
device, through a rack of effects, in realtime. Act one of the VHS show: the
deck's sound goes to the PA through the app, clean, so that later the same
path can carry effects and, later still, the generated material.

Built on PortAudio (the sounddevice package) with Python in the loop:

    input device --read--> [analysis tap] -> live rack -> sync delay ----+
    section player: 7 drum voices x sqrt(A/B) -> mute -> voice racks -+  |
                    layers, chops, one-shots x sqrt(A/B) -------------+  |
    song player (a render's stems) -------------------------+-> 4 stem   +-> VST3 inserts -> gain -> limiter -->write--> output
                                                               racks, F1-F4    (gacha_vst)

The drum voices (gacha_fx.VOICES: kick, snare, closed and open hat, ride,
hat stutter, sub) each have a rack of their own and a mute and a solo; the
voices' sum is the drums stem, through the drums rack (the kit's bus) and
F1. A mute gates the voice before its rack, so a delay or a reverb on it
rings out after the mute. Mutes, solos, F1-F4, the one-shots and the bursts
land on the quantize grid (`quant`, a fraction of a bar: a sixteenth, an
eighth, a beat, half a bar, the bar) while a section plays, at once with
the grid off or nothing playing; with `midi_follow_mutes` a muted voice's
hits leave the MIDI out too (off by default: the receiver does its own
muting).

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
    la.switch("stem", "drums", False)      # F1-F4: stems in and out (section and song)
    la.switch("voice", "snare", False)     # a drum voice out (its MIDI notes too)
    la.switch("solo", "kick", True)        # soloed: every other voice out
    la.quant = 1 / 8                       # the switches, events and bursts land on eighths
    la.racks["snare"].amounts["delay"]     # each voice has its own rack
    la.song.load(path); la.song.play()     # a generated song, its stems through the racks
    la.beat_s = 0.5                        # the delays' beat
    la.gain, la.mute                       # ramped, click-free
    la.delay_ms = 80                       # audio held back to meet the picture
    la.drain()                             # mono blocks for the analyzer
    la.record_start(10.0); la.record_stop()   # a take: dry input, with pre-roll
    la.player                              # SectionPlayer: stems, next, one-shots
    la.ab = 0.0 .. 1.0                     # A tape through <-> B the section
    la.stats()                             # peaks, load, drops, latency
    la.midi_out = MidiOut()                # the drum hits as notes (gacha_midi)
    la.midi_lead_ms = 25.0                 # sent this early: the receiver's latency

MIDI out: every drum hit of the section and the song is in their event
timelines (role, time, velocity), rendered before it sounds, so the block
that will carry a hit is known ahead. Each block the players list the hits
of a window `lookahead` frames ahead of what they render, and _dsp_block
hands them to the MidiOut scheduler with the moment they should leave: the
moment the hit will be heard (block start + the driver's output latency +
its offset) minus the lead, the receiver's own latency, so its drum lands
on ours. Section hits are under the A/B pedal like the section (velocity by
sqrt(ab), none at 0); with midi_follow_mutes they follow the drums stem's
F1 and each voice's mute and solo too, from the very frame a switch lands
on (the hits already handed over for the lead's window, a few ms, are the
one exception), otherwise every hit goes out and the receiver does its own
muting; a break sends the retriggered slices' hits as they sound.
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

from gacha_fx import ALL_CHANNELS, Rack, STEM_CHANNELS, VOICES
from gacha_midi import DRUM_NOTES, DRUM_ROLES

assert VOICES[:len(DRUM_ROLES)] == DRUM_ROLES      # one voice per MIDI role, then the sub

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


def drum_table(events, sr):
    """The drum hits of an event timeline ([t, src, off, dur, rate, role,
    (vel)] with t in seconds) as (frames int64 sorted, roles, velocities
    0..1), for the MIDI out; None when there is none."""
    rows = []
    for e in events or ():
        if len(e) < 6 or e[5] not in DRUM_ROLES:
            continue
        vel = float(e[6]) if len(e) > 6 else 1.0
        rows.append((int(round(float(e[0]) * sr)), e[5], min(1.0, max(0.0, vel))))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    return (np.array([r[0] for r in rows], dtype=np.int64),
            [r[1] for r in rows], np.array([r[2] for r in rows], dtype=np.float32))


def table_hits(table, lo, hi, shift, out):
    """Append (frame + shift, role, vel) for the table's hits with lo <= frame < hi."""
    if table is None or hi <= lo:
        return
    t, roles, vels = table
    i0, i1 = np.searchsorted(t, lo, "left"), np.searchsorted(t, hi, "left")
    for i in range(int(i0), int(i1)):
        out.append((int(t[i]) + shift, roles[i], float(vels[i])))


PHRASE_BARS = 4        # a live break lands on these bars' one (the loop's end too)


class SectionPlayer:
    """Looping stems at the stream's rate, swapped on a bar, plus one-shots.
    Everything here is set from the GUI thread and read in the audio
    thread; swaps go through `pending` and happen on a bar boundary.

    A break (`fill()`): on the phrase (PHRASE_BARS bars, or the loop when
    it is shorter), the loop is taken over for the phrase's last bar, and
    a section queued meanwhile drops in on the phrase's one, the loop's
    middle or its end. Asked earlier it waits, armed, for that bar; asked
    inside it, it runs from the next beat to the end, a beat at the least.
    Two kinds, as in a generated song's run-up to a drop:
    "glitch" throws sixteenths of the bar that was playing about, some
    backwards, a stutter roll on the last beat and a gap right before the
    bar; "break" is the song's own transition, a beat-repeat of the bar's
    first slice (accelerating, steady or halving), a gap of silence before
    the drop, and a reverse cymbal swell handed in by the GUI rising under
    it. With `gap` off neither leaves silence: the roll and the repeat run
    right into the one. `fill_state()` tells the picture what is happening.

    A repeat (`repeat`, set from the GUI while a key or a note is held): a
    cell of the bar, a division of it (bar, 1/2, ... 1/64), is read in place
    of the transport, the cell playing at the press (the anchor), moved
    some cells earlier or later round the loop, forwards or backwards. The
    transport runs on underneath, so the bar, the clock and a queued
    section are untouched and letting go lands where the loop would have
    been; the cell edges sit on the bar's own grid, so a change of division
    or offset is in time at once. A break wins over it; a stop ends it.
    `heard_s()` and `repeat_state()` tell the picture what sounds."""

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
        self.quant = 0.0                     # the grid, a fraction of a bar; 0 = at once
        self.gap = True                      # a break's silence before the drop (else it runs into it)
        self.grid_off = 0                    # block offset of this block's grid line, or None
        self.shots = []                      # [(clip (2, n), frames played; < 0: starts later)]
        self.shot_queue = []                 # clips waiting for the grid
        self.bursts = []                     # [(clip, played)]: not under A/B
        self.burst_queue = []                # [(clip, number)] waiting for the grid
        self.burst_n = 0                     # bursts asked so far
        self.burst_fired = None              # (number, monotonic start) of the last one out
        self.loop_t0 = None                  # monotonic time the loop last wrapped
        self.swapped = False                 # a pending section just went live
        self.beat_frames = 0
        self.fill_armed = None               # (kind, swell): a break asked, next beat
        self._fill = None                    # dict: the break playing, see _fill_start
        self.dropped = False                 # a break just ended in a swap (GUI resets)
        self._ev = None                      # drum_table of the section: the MIDI out
        self.lookahead = 0                   # frames ahead of the block the hits are listed
        self.hits = []                       # [(block offset, role, vel)] listed by block()
        self._covered = 0                    # loop frame up to which hits were listed
        self._pend_covered = 0               # same for the pending section, from its 0
        # the MIDI clock: the loop frame at this block's start (after `grid_lead`
        # frames of silence), None when nothing plays; the transport: the
        # (lead, pos) a section started from silence with, and a stop, both
        # consumed by LiveAudio in the same audio callback
        self.grid_i0 = None
        self.grid_lead = 0
        self.started = None
        self.stopped = False
        # the repeat: (division of a bar, offset in cells, backwards) from
        # the GUI, None = off; the anchor (the cell captured at the press,
        # a loop frame) and what is heard belong to the audio thread
        self.repeat = None
        self._rep_anchor = None
        self._heard = None                   # loop frame sounding at the block's end, or None
        self._rep_info = (0, 0.0)            # (cell count, 0..1 into the cell)
        self._resume = False                 # fade the transport back in after a repeat

    @property
    def playing(self):
        return self.stems is not None

    @property
    def filling(self):
        return self._fill is not None

    def phrase_end(self, pos):
        """The loop frame the phrase around `pos` ends on: the next
        PHRASE_BARS boundary, or the loop's end when that comes first."""
        ph = PHRASE_BARS * 4 * self.beat_frames
        return min(self.n, (pos // ph + 1) * ph) if ph else self.n

    def fill(self, kind="glitch", swell=None):
        """A break of `kind` ("glitch" or "break"), on the phrase's last bar:
        see the class note. `swell`: a (2, n) float32 clip at sr to rise into
        the drop (the break kind), its end on the bar. No-op with no section."""
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

    def set_repeat(self, div, offset=0, rev=False):
        """Hold: repeat `div` of a bar (1.0, 0.5, ... 1/64) from the cell
        playing now, `offset` cells earlier (< 0) or later round the loop,
        backwards with `rev`; call again to change any of them while held,
        the anchor stays. See the class note. No-op with no section."""
        if self.playing:
            self.repeat = (float(div), int(offset), bool(rev))

    def clear_repeat(self):
        """Let go: the transport, which ran on underneath, is heard again."""
        self.repeat = None

    def heard_s(self):
        """Seconds into the loop of what sounds now: the transport, or the
        repeated cell's frame while a repeat is held (a break has `pos`
        follow its slices already)."""
        return (self.pos if self._heard is None else self._heard) / self.sr

    def repeat_state(self):
        """What the repeat is doing, for the picture: {k (cells so far),
        frac (0..1 into the cell), div, rev}, or None while none sounds."""
        if self._heard is None or self.repeat is None:
            return None
        k, frac = self._rep_info
        return {"k": k, "frac": frac, "div": self.repeat[0], "rev": self.repeat[2]}

    def repeat_window(self):
        """The repeat's window for the display: (start, length) in loop
        frames and whether it runs backwards, or None before it sounds."""
        rep, a = self.repeat, self._rep_anchor
        if rep is None or a is None or not self.beat_frames or not self.n:
            return None
        ln, start, rev = self._rep_geom(a, rep)
        return start, ln, rev

    # ---- the repeat (audio thread)
    def _rep_geom(self, a, rep):
        """(cell length, window start, backwards) of the repeat `rep` from
        the anchor `a`: a coarser division snaps the anchor down within its
        bar, a finer one keeps it, so the roll tightens from the same one;
        the offset moves the window by cells, round the loop."""
        div, off, rev = rep
        bar_f = 4 * self.beat_frames
        ln = max(1, min(bar_f, int(round(bar_f * div))))
        start = ((a // bar_f) * bar_f + (a % bar_f // ln) * ln + off * ln) % self.n
        return ln, start, rev

    def _rep_cells(self, i0, rep):
        """The repeat's geometry at transport frame `i0`; the anchor, set
        at the first block, is the cell playing then."""
        if self._rep_anchor is None:
            bar_f = 4 * self.beat_frames
            ln = max(1, min(bar_f, int(round(bar_f * rep[0]))))
            self._rep_anchor = (i0 // bar_f) * bar_f + (i0 % bar_f // ln) * ln
        return self._rep_geom(self._rep_anchor, rep)

    def _repeat_idx(self, i0, m, rep):
        """The loop frames read for transport frames [i0, i0+m) under the
        repeat, and a gain with 2 ms fades at the cell edges (None when
        the cells are too short for one). The phase inside the cell is the
        transport's own, within its bar, so the hits land on the grid."""
        ln, start, rev = self._rep_cells(i0, rep)
        bar_f, n = 4 * self.beat_frames, self.n
        t = np.arange(i0, i0 + m)
        ph = (t % bar_f) % ln
        src = start + (ln - 1 - ph if rev else ph)
        idx = src % n
        f = int(0.002 * self.sr)
        g = None
        if 0 < f <= ln // 2:
            g = np.minimum(np.minimum(ph, ln - 1 - ph) / f, 1.0).astype(np.float32)
        self._heard = int(idx[-1])
        self._rep_info = (int(t[-1] // ln), float(ph[-1]) / ln)
        return idx, g

    def _src_hits(self, lo, cnt, off, rev):
        """The hits of loop frames [lo, lo+cnt) (round the loop's end) to
        the MIDI window, the first frame at block offset `off`, the range
        played backwards with `rev`."""
        n = self.n
        lo %= n
        done = 0
        while done < cnt:
            take = min(cnt - done, n - lo)
            tmp = []
            table_hits(self._ev, lo, lo + take, 0, tmp)
            for s, role, v in tmp:
                j = (s - lo) + done
                self.hits.append((off + (cnt - 1 - j if rev else j), role, v))
            done += take
            lo = 0

    def _repeat_hits(self, i0, m, lead, rep):
        """The MIDI out's window under a repeat: cell by cell, the repeated
        slice's hits where they sound (backwards when reversed)."""
        ln, start, rev = self._rep_cells(i0, rep)
        bar_f, n, L = 4 * self.beat_frames, self.n, self.lookahead
        a0, end = i0 + L, i0 + L + m
        t = max(i0, min(a0, self._covered))
        while t < end:
            tb = t % bar_f
            ph = tb % ln
            cnt = min(end - t, ln - ph, bar_f - tb)
            off = lead + (t - a0)                     # block offset of transport frame t
            if rev:
                self._src_hits(start + ln - ph - cnt, cnt, off, True)
            else:
                self._src_hits(start + ph, cnt, off, False)
            t += cnt
        self._covered = end - (n if i0 + m >= n else 0)

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
            gap = int(bf * (0.5 if beats <= 2 else rng.choice([0.5, 1.0]))) if self.gap else 0
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
            if self.gap:
                silence(sx)                             # the gap before the drop
            else:                                       # or the roll on into it
                for _ in range(8):
                    piece(bar0, sx // 8)
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
        """A beat has just passed at `i0`, inside a phrase's last bar: the
        break runs from here to the phrase's end."""
        kind, swell = self.fill_armed
        bf = self.beat_frames
        beat = i0 // bf                                 # the beat we are in
        end = self.phrase_end(i0)                       # in loop frames
        beats = max(1, -(-(end - beat * bf) // bf))     # beats to it, this one included
        idx, gain, retrig, frac = self._fill_map(beat * bf, beats, random.Random(), kind)
        late = i0 - beat * bf                           # the block boundary: a few ms in
        idx, gain, retrig, frac = idx[late:], gain[late:], retrig[late:], frac[late:]
        k0 = 0
        if swell is not None:                           # its end on the drop
            if swell.shape[1] > len(idx):
                swell = swell[:, swell.shape[1] - len(idx):]
            k0 = len(idx) - swell.shape[1]
        self._fill = {"idx": idx, "gain": gain, "retrig": retrig, "frac": frac, "k": 0,
                      "end": end, "kind": kind, "swell": swell, "swell_k0": k0,
                      "covered": 0, "after": 0}     # the MIDI out's window, see _fill_hits
        self.fill_armed = None

    def queue(self, stems, events, bpm, bars, bar_t0=None):
        """The next section: on the loop's next bar while one plays, else
        from the next block, or, `bar_t0` given (monotonic seconds, the
        Link session's bar), exactly then: silence up to it inside the
        block, or, when it has just passed, the loop entered that far in."""
        self.pending = (stems, events, bpm, bars, bar_t0, drum_table(events, self.sr))
        self._pend_covered = 0

    def waiting_for(self):
        """The bar time a queued section waits for, or None."""
        p = self.pending
        return p[4] if p is not None and p[0] != "clear" and self.stems is None else None

    def clear(self):
        self.pending = ("clear",)

    def fire(self, clip, now=False):
        """A one-shot (2, n) float32 at sr, on the next line of the quantize
        grid (at once with the grid off or no section playing). `now`: a
        burst, mixed outside the A/B crossfade, so it is heard over the
        tape too; returns its number, and `burst_fired` tells when it
        started sounding."""
        if now:
            self.burst_n += 1
            self.burst_queue.append((clip, self.burst_n))
            return self.burst_n
        self.shot_queue.append(clip)
        return None

    def _grid_offset(self, i0, m, lead):
        """Block offset of the first quantize grid line inside this block's
        loop frames [i0, i0 + m) (the loop's end is one), None when there
        is none; 0 with the grid off."""
        q, bf = self.quant, self.beat_frames
        if q <= 0 or not bf:
            return 0
        step = max(1, int(round(4 * bf * q)))
        g = -(-i0 // step) * step
        if self.n and i0 + m > self.n:
            g = min(g, self.n)
        return lead + (g - i0) if g < i0 + m else None

    def pos_s(self):
        return self.pos / self.sr

    def _swap(self, now=None, bs=0):
        p, self.pending = self.pending, None
        if p[0] == "clear":
            if self.stems is not None:
                self.stopped = True
            self.stems, self.events, self.n, self.pos = None, [], 0, 0
            self._ev = None
            self.repeat, self._rep_anchor, self._heard = None, None, None
            return
        stems, events, bpm, bars, bar_t0, table = p
        from_silence = self.stems is None
        self.stems, self.events, self.bpm, self.bars = stems, events, bpm, bars
        self._ev = table
        self._covered, self._pend_covered = self._pend_covered, 0
        self.n = next(iter(stems.values())).shape[1]
        self.pos, self._lead = 0, 0
        self.beat_frames = int(round(60.0 / bpm * self.sr))
        self.loop_t0 = time.monotonic() if now is None else now
        if bar_t0 is not None and from_silence and now is not None:
            if bar_t0 > now:                          # inside this block: silence first
                self._lead = min(bs, int(round((bar_t0 - now) * self.sr)))
            else:                                     # the bar just passed: enter late
                self.pos = int((now - bar_t0) * self.sr) % self.n
                self._covered = self.pos
            self.loop_t0 = now + (self._lead - self.pos) / self.sr
        if from_silence:
            self.started = (self._lead, self.pos)
        self.swapped = True

    # ---- the MIDI out's window: the hits `lookahead` frames ahead of the block
    def _next_table(self):
        """What plays after this loop's end: the pending section's hits,
        our own again, or None once a clear waits."""
        p = self.pending
        if p is None:
            return self._ev
        return None if p[0] == "clear" else p[5]

    def _collect(self, table, pos, m, off, nxt):
        """Hits from loop frame `pos` of `table` for `m` frames, at block
        offset `off` on; past the loop's end `nxt` takes over from 0."""
        n = self.n
        while m > 0 and n:
            take = min(m, n - pos) if pos < n else m
            table_hits(table, pos, pos + take, off - pos, self.hits)
            m, off, pos, table = m - take, off + take, 0, nxt

    def _loop_hits(self, i0, m, lead, breaking, bar_f):
        """The normal path: the window ahead of [i0, i0+m), from where the
        last one ended (a section just started from silence has its first
        frames listed now, late by the lead at worst); a pending section
        starts at the first block after the next bar (as the swap does), so
        its first hits are listed from there."""
        L, n = self.lookahead, self.n
        a0, end = i0 + L, i0 + L + m
        a = max(i0, min(a0, self._covered))
        cut = None
        if self.pending is not None and not breaking and bar_f:
            b = (i0 // bar_f + 1) * bar_f                  # the next bar boundary
            cut = i0 + m * -(-(b - i0) // m)               # the block start after it
        own_end = end if cut is None else min(end, cut)
        if own_end > a:
            self._collect(self._ev, a % n, own_end - a, lead + (a - a0), self._ev)
        if cut is not None and cut < end:
            nxt = self._next_table()
            start = max(a, cut)
            if nxt is not None:
                self._collect(nxt, start - cut, end - start, lead + (start - a0), nxt)
            self._pend_covered = end - cut
        self._covered = end - (n if i0 + m >= n else 0)

    def _fill_hits(self, k, bs):
        """A break: the window over the fill's slice map, each slice's hits
        where they sound (reversed slices reversed), then what follows the
        drop: the pending section from 0, or the loop on from the bar."""
        f = self._fill
        idx, retrig, end = f["idx"], f["retrig"], f["end"]
        L = self.lookahead
        a0, b = k + L, k + L + bs
        a = max(k, min(a0, f["covered"]))
        f["covered"] = b
        w1 = min(b, len(idx))
        p = a
        while p < w1:
            r = retrig[p]
            q = p + 1
            while q < w1 and retrig[q] == r:
                q += 1
            if r >= 0:
                src0 = int(idx[p])
                rev = q - p > 1 and idx[q - 1] < src0
                if rev:                                    # frames src0, src0-1, ...
                    tmp = []
                    table_hits(self._ev, src0 - (q - p) + 1, src0 + 1, 0, tmp)
                    self.hits.extend((p - a0 + (src0 - t), role, v) for t, role, v in tmp)
                else:
                    table_hits(self._ev, src0, src0 + (q - p), p - a0 - src0, self.hits)
            p = q
        if b > len(idx):                                   # past the drop
            start = max(a, len(idx))
            if self.pending is not None:
                nxt = self._next_table()
                if nxt is not None:
                    self._collect(nxt, start - len(idx), b - start, start - a0, nxt)
                self._pend_covered = b - len(idx)
            elif self.n:
                self._collect(self._ev, (end + start - len(idx)) % self.n, b - start,
                              start - a0, self._ev)
            f["after"] = b - len(idx)

    def block(self, bs, now, ramp):
        """The next bs frames of the section, {stem: (2, bs) float32}, the
        one-shots in "events". Stem on/off is applied by LiveAudio after the
        racks. Empty dict when nothing plays."""
        out = {}
        self.hits = []
        self.grid_i0, self.grid_lead = None, 0
        self.grid_off = 0
        bar_f = 4 * self.beat_frames
        breaking = self.fill_armed or self._fill is not None
        if self.pending is not None and (self.stems is None or (self.pos == 0 and not breaking)):
            wait = self.waiting_for()
            if wait is None or now + bs / self.sr >= wait:   # a timed start: not yet
                self._swap(now, bs)
        if self._fill is not None:
            self._rep_anchor, self._heard = None, None   # a break wins; a held repeat re-anchors after
            on_beat = self._fill_block(out, bs)
        elif self.stems is not None:
            lead = min(self._lead, bs)                # a timed start: silence up to it
            self._lead -= lead
            m = bs - lead
            i0 = self.pos
            self.grid_i0, self.grid_lead = i0, lead
            self.grid_off = self._grid_offset(i0, m, lead)
            rep = self.repeat
            if rep is not None:                       # a hold: the cell, not the transport
                idx, g = self._repeat_idx(i0, m, rep)
                self._resume = True
            else:
                idx, g = (np.arange(m) + i0) % self.n, None
                self._rep_anchor = None
                if self._resume:                      # just let go: no click on the way back
                    self._resume = False
                    f = min(m, int(0.002 * self.sr))
                    if f > 1:
                        g = np.ones(m, np.float32)
                        g[:f] = np.linspace(0.0, 1.0, f, dtype=np.float32)
                self._heard = None
            for name, data in self.stems.items():
                blk = data[:, idx]
                if g is not None:
                    blk = blk * g
                if lead:
                    blk = np.concatenate([np.zeros((2, lead), np.float32), blk], axis=1)
                out[name] = blk
            if self._ev is not None:
                if rep is not None:
                    self._repeat_hits(i0, m, lead, rep)
                else:
                    self._loop_hits(i0, m, lead, breaking, bar_f)
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
            # an armed break waits for the phrase's last bar, then starts on
            # the next beat: from the next block, to the phrase's end
            if self.fill_armed and on_beat and self.beat_frames                     and self.pos >= self.phrase_end(self.pos) - bar_f:
                self._fill_start(self.pos)
        else:
            on_beat = True
        o = self.grid_off                             # the grid: one-shots and bursts start there
        if o is not None:
            if self.shot_queue:
                q, self.shot_queue = self.shot_queue, []
                self.shots.extend((c, -o) for c in q)
            if self.burst_queue:
                q, self.burst_queue = self.burst_queue, []
                self.bursts.extend((c, -o) for c, _k in q)
                self.burst_fired = (q[-1][1], now + o / self.sr)
        if self.shots:
            ev = out.get("events")
            ev = np.zeros((2, bs), np.float32) if ev is None else ev.copy()
            _, self.shots = self._sum_clips(self.shots, bs, ev)
            out["events"] = ev
        if self.bursts:
            out["burst"], self.bursts = self._sum_clips(self.bursts, bs)
        return out

    def _fill_block(self, out, bs):
        """One block of the break into `out`; `pos` follows the slice that
        sounds, so the picture glitches with it. At the end: the pending
        section, or the loop on from the bar. Returns whether a beat passed."""
        f = self._fill
        idx, gain, k, end = f["idx"], f["gain"], f["k"], f["end"]
        self.grid_i0 = end - (len(idx) - k)       # the break keeps the bar's grid
        if self._ev is not None:
            self._fill_hits(k, bs)
        m = min(bs, len(idx) - k)
        self.grid_off = self._grid_offset(self.grid_i0, m, 0)
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
            self._covered = self.pos + f["after"]
            if self.pending is not None:
                self._swap()
                self.dropped = True
            elif self.pos == 0:
                self.loop_t0 = time.monotonic()
            on_beat = True
        return on_beat

    @staticmethod
    def _sum_clips(clips, bs, blk=None):
        """Mix the next `bs` frames of every (clip, played) into one (2, bs)
        block (`blk`, added into, or a new one); played < 0 is a clip that
        starts that many frames into the block. Returns the block and the
        clips still sounding."""
        if blk is None:
            blk = np.zeros((2, bs), np.float32)
        keep = []
        for clip, k in clips:
            o, k = max(0, -k), max(0, k)
            n = min(bs - o, clip.shape[1] - k)
            if n > 0:
                blk[:, o:o + n] += clip[:, k:k + n]
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
        self._ev = None                       # drum_table of the render: the MIDI out
        self.lookahead = 0
        self.hits = []

    def set_events(self, events):
        """The render's event timeline (its meta's `events`), for the MIDI out."""
        self._ev = drum_table(events, self.sr)

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
        self.hits = []
        if self.state != "playing" or self.stems is None:
            return None
        i0 = self.pos
        end = min(self.n, i0 + bs)
        if self._ev is not None:
            a = i0 + self.lookahead
            table_hits(self._ev, a, min(self.n, a + bs), -a, self.hits)
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
        self.racks = {ch: Rack(self.sr) for ch in ALL_CHANNELS}   # live in, 4 stems, 7 voices
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
        # the drum voices: in or out, and soloed (then every other voice is
        # out); set through switch(), which lands them on the grid
        self.voice_on = {v: True for v in VOICES}
        self.voice_solo = set()
        self.quant = 0.0                     # the grid, a fraction of a bar (1/8 = eighths); 0 = at once
        self._sw_req = []                    # (kind, key, on) asked, under _sw_lock
        self._sw_lock = threading.Lock()
        self._sw_wait = []                   # ...and waiting for the grid (audio thread)
        self._aud_before = frozenset()
        self._limiter = Limiter(threshold_db=-1.0, release_ms=100.0)
        self.inserts = None                  # gacha_vst.InsertChain on the master, or None
        # the drum hits as MIDI notes (see the module note): a gacha_midi.MidiOut
        self.midi_out = None
        self.midi_channel = 10
        self.midi_channels = None            # role -> channel instead (a Model:Cycles), or None
        self.midi_lead_ms = 0.0              # how early a note leaves: the receiver's latency
        self.midi_gate_ms = 50.0             # note on to note off
        self.midi_notes = dict(DRUM_NOTES)   # role -> note, -1 = not sent
        self.midi_drums_only = False         # our own drum voices silent while sending (the sub stays)
        self.midi_follow_mutes = False       # F1, the voice mutes and solos take notes out too;
                                             # off: every hit goes out, the receiver mutes
        self.midi_sent = 0
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
    def switch(self, kind, key, on):
        """A stem ("stem", F1-F4), a drum voice ("voice", in or out) or a
        solo ("solo") switched, from any thread: it lands on the next line
        of the quantize grid while a section plays, at once otherwise. The
        audio gate ramps there, and the MIDI out follows from the same
        frame on."""
        with self._sw_lock:
            self._sw_req.append((kind, key, bool(on)))

    def _audible(self):
        """The roles whose hits go out: every role unless the notes follow
        the mutes, then the drums stem in, the voice in, and not soloed out."""
        if not self.midi_follow_mutes:
            return frozenset(VOICES)
        if not self.on.get("drums", True):
            return frozenset()
        solo = self.voice_solo
        return frozenset(v for v in VOICES
                         if self.voice_on.get(v, True) and (not solo or v in solo))

    def _land_switches(self, grid_off):
        """The switches asked for land at `grid_off`, the player's grid line
        in this block (None: none this block, they keep waiting; 0: at
        once). Returns the offset they landed at, or None. What was audible
        before is kept for the hits before that frame."""
        with self._sw_lock:
            if self._sw_req:
                self._sw_wait.extend(self._sw_req)
                self._sw_req = []
        self._aud_before = self._audible()
        if not self._sw_wait or grid_off is None:
            return None
        for kind, key, on in self._sw_wait:
            if kind == "stem":
                self.on[key] = on
            elif kind == "voice":
                self.voice_on[key] = on
            elif kind == "solo":
                if on:
                    self.voice_solo.add(key)
                else:
                    self.voice_solo.discard(key)
        self._sw_wait = []
        return int(grid_off)

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
        self.racks = {ch: Rack(sr) for ch in ALL_CHANNELS}
        for ch, r in self.racks.items():
            r.amounts.update(old_racks[ch].amounts)
            r.vol = old_racks[ch].vol
        old_player = self.player
        self.player = SectionPlayer(sr)
        self.player.on = self.on
        self.player.gap = old_player.gap
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
        self._voice_gain = {}                     # voice -> the same
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

    def _midi_sync(self, mo, heard, sr):
        """The MIDI clock's grid and the transport from this block: `heard`
        is when the block's first frame is heard, less the lead, the time
        base the notes leave in. The section's beat grid while one plays
        (a break included), the live tempo otherwise; Start on a section's
        downbeat from silence (or the next bar, entered late), Stop on a
        clear."""
        pl = self.player
        bf = pl.beat_frames
        if pl.playing and bf and pl.grid_i0 is not None:
            beat_s = bf / sr
            b = (pl.grid_i0 // bf) * bf               # the beat this block sits in
            mo.grid(heard + (pl.grid_lead + b - pl.grid_i0) / sr, beat_s)
        else:
            beat_s = float(self.beat_s)
            mo.grid(None, beat_s)
        if pl.started is not None:
            lead, pos = pl.started
            pl.started = None
            due = heard + (lead - pos) / sr
            bar_s = 4 * beat_s
            while due < heard - 0.001 and bar_s > 0:   # entered late: the next bar
                due += bar_s
            mo.start(due)
        if pl.stopped:
            pl.stopped = False
            mo.stop()

    def _tape_weight(self, ab):
        """The tape's gain under the pedal: equal power against the section,
        or 1 while hold_tape keeps its sound whole."""
        return 1.0 if self.hold_tape else float(np.sqrt(1.0 - ab))

    def _ring_read(self, delay):
        idx = (self._ar + self._wpos - self.block - delay) % self._ring_n
        return self._ring[:, idx]

    @staticmethod
    def _gate(cur, target, bs, off):
        """A gain ramp over the block from `cur` to `target`, from frame
        `off` on (None or 0: the whole block), 32 frames long at the least
        so it never clicks."""
        o = 0 if not off else min(int(off), bs)
        o = min(o, max(0, bs - 32))
        r = np.full(bs, cur, np.float32)
        r[o:] = np.linspace(cur, target, bs - o, dtype=np.float32)
        return r

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
        # the MIDI out's window: hits are listed this far ahead of the block,
        # so a note can leave before the block it sits in is even rendered
        mo = self.midi_out
        midi = mo is not None and mo.is_open
        out_lat = self.out_latency_ms / 1000.0
        lead_s = self.midi_lead_ms / 1000.0
        L = int(math.ceil(max(0.0, lead_s - out_lat) * sr)) if midi else 0
        self.player.lookahead = self.song.lookahead = L
        self.player.quant = self.quant
        t_block = time.perf_counter()
        sec = self.player.block(bs, now, ramp)
        song = self.song.block(bs)
        sec_w = np.sqrt(a)
        # the switches land on the grid line of this block, if any; the hits
        # before it keep what was audible, those from it on take the new state
        sw_off = self._land_switches(self.player.grid_off)
        if midi:
            self._midi_sync(mo, t_block + out_lat - lead_s, sr)
        if midi and (self.player.hits or self.song.hits):
            aud_after, aud_before = self._audible(), self._aud_before
            base = t_block + out_lat - lead_s + L / sr        # >= t_block by the choice of L
            gate = self.midi_gate_ms / 1000.0
            ch, notes = int(self.midi_channel), self.midi_notes
            chans = self.midi_channels or {}
            w = float(np.sqrt(ab))                            # the section under the pedal
            for hits, scale in ((self.player.hits, w), (self.song.hits, 1.0)):
                if scale < 0.05:
                    continue
                for off, role, vel in hits:
                    aud = aud_after if sw_off is None or off >= sw_off else aud_before
                    note = notes.get(role, -1)
                    if role not in aud or note is None or note < 0:
                        continue
                    mo.note(chans.get(role, ch), note, round(127 * vel * scale),
                            base + off / sr, gate)
                    self.midi_sent += 1
        # the drum voices: each gated (mute, solo; the receiver's kit
        # replacing ours leaves the sub) before its own rack, summed into
        # the drums stem; a rack with a tail rings on after its mute
        drums_only = midi and self.midi_drums_only
        solo = self.voice_solo
        bus = None
        for v in VOICES:
            blk = sec.get(v)
            on = self.voice_on.get(v, True) and (not solo or v in solo)
            if drums_only and v != "sub":
                on = False
            g = 1.0 if on else 0.0
            cur = self._voice_gain.get(v, 1.0)        # every voice starts in
            rack = self.racks[v]
            if g != cur:
                blk = np.zeros((2, bs), np.float32) if blk is None else blk * sec_w
                blk = blk * self._gate(cur, g, bs, sw_off)
                self._voice_gain[v] = g
            elif blk is None or not g:
                if rack.idle:
                    continue
                blk = np.zeros((2, bs), np.float32)
            else:
                blk = blk * sec_w
            blk = rack.process(blk)
            bus = blk if bus is None else bus + blk
        burst = sec.get("burst")                    # the bursts: into the events
        for stem in STEM_CHANNELS:                  # rack, but not under A/B
            if stem == "drums":
                blk = bus
            else:
                blk = sec.get(stem)
                blk = None if blk is None else blk * sec_w
            if burst is not None and stem == "events":
                blk = burst if blk is None else blk + burst
            if song is not None and stem in song and not (stem == "drums" and drums_only):
                blk = song[stem] if blk is None else blk + song[stem]
            if blk is None:
                blk = np.zeros((2, bs), np.float32)     # tails keep ringing
            blk = self.racks[stem].process(blk)
            on = 1.0 if self.on.get(stem, True) else 0.0
            cur = self._stem_gain.get(stem, on)
            if cur != on:                                # F1-F4: a ramp, no click
                blk = blk * self._gate(cur, on, bs, sw_off)
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
