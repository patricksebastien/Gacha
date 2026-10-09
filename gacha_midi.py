#!/usr/bin/env python3
"""
gacha_midi.py — MIDI input for the performer's actions.

Several controllers at once (a foot controller for the sections, a pad for
the events, a knob box for the grade), each a port ticked in the Controls
tab. Every message from any of them lands on one Qt signal on the GUI
thread, where gacha.py matches it against the action map:

    midi = MidiIn()
    midi.message.connect(on_midi)      # (port, kind, channel, number, value)
    midi.open(["FCB1010:FCB1010 MIDI 1", ...])   # labels from midi_inputs()

Kinds are "note" (note on, velocity > 0; the note off, or a note on at
velocity 0, comes as the same note at value 0, the release of a held
action), "cc" (control change) and "pc" (program change, which is what an
FCB1010 sends out of the box); channels are 1 to 16. Clock, active sensing
and the rest are dropped here.

A binding is a small string that the settings store and the Controls tab
shows: "note 60 ch1", "cc 11 ch1", "pc 5 ch1". The port is not part of it:
two controllers on different channels tell themselves apart by the channel,
and a binding survives the controller moving to another USB socket.

Port names from ALSA end in the client:port numbers ("FCB1010:FCB1010 MIDI 1
20:0"), which change with every plug; the label drops them so the ticked
ports are remembered across launches and re-enumerations.

MIDI out, the other way (MidiOut): the drum hits of a playing section or
song as notes on one output port, for a drum machine in a DAW (Bitwig over
a loopMIDI cable, say) to play them. The audio thread knows every hit ahead
of time (sections are rendered before they play) and hands each one over
with the moment it should sound, on time.perf_counter(); a sender thread
sleeps up to a few ms before it and spins the rest, so a note lands within
about a millisecond of its due time whatever the audio block size. Note
offs follow after a gate time. GM drum notes are the default map
(DRUM_NOTES), every role's note settable, -1 = not sent. The notes go out
on one channel, or, for an Elektron Model:Cycles, each role on its own
track channel (CYCLES_CHANNELS: kick 1, snare 2, closed hat 3, open hat 4,
ride 5, hat stutter 6) where the note is the pitch the track plays at,
CYCLES_ROOT for its own tuning; see MODEL_CYCLES_MIDI.md.

    out = MidiOut(); out.open("loopMIDI Port")  # a label from midi_outputs()
    out.note(10, 36, 100, due=time.perf_counter() + 0.02, gate=0.05)
    out.all_off(); out.close()

MIDI clock and transport on the same port, for a drum machine or a synth
that follows MIDI sync (an Elektron Model:Cycles, anything with CLK IN):
with clock_on the sender thread puts out the 24 ticks per beat (0xF8) at
all times, at the tempo it is told (grid(None, beat_s): the live clock
while nothing plays) and, while a section plays, pulled onto the section's
own beat grid (grid(beat_t, beat_s): the perf_counter moment a beat is
heard, minus the lead) by a gentle slew, a tenth of a tick per block at
most, so the receiver's beat sits on our hits without a tick ever doubling
or dropping. With transport_on, start(due) puts a Start (0xFA) half a
millisecond before the tick at `due`, which is then placed exactly there,
so the receiver's step 1 is our downbeat; stop() sends Stop (0xFC) at once,
and so do close_port() and all_off() while a Start is out.

Control change out, for the Model:Cycles tab (the box's sound over CC,
CYCLES_PARAMS below): cc(channel, cc, value) leaves on the same port and
thread. CCs are gathered for a couple of ms and go out together, one value
per controller (the last one wins, a repeat of the value last sent is
dropped), and a controller gets a new value every CC_MIN_S at most: the DIN
wire takes a millisecond per message, and an LFO on a knob would otherwise
drown the notes. retrig_start(channel, note, velocity, period_s, length_s,
fade) is the box's RETRIG function done from here, since it has no CC: the
note again every period until retrig_stop(channel), the velocity fading in
(fade > 0) or out (fade < 0) over length_s like the box's FDE and LEN.

Built on mido over python-rtmidi; without them everything here is inert and
MIDI_AVAILABLE is False.
"""
import heapq
import math
import re
import sys
import threading
import time

from PySide6.QtCore import QObject, Signal

try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:              # pragma: no cover - the tab says what to install
    mido = None
    MIDI_AVAILABLE = False

KINDS = ("note", "cc", "pc")
_TRAILING_ADDR = re.compile(r"\s+\d+:\d+$")
# the drum roles the engine and the section renderer put in their event
# timelines, with the General MIDI percussion notes they go out as
DRUM_ROLES = ("kick", "snare", "hihat", "ohat", "ride", "glitch")
DRUM_NOTES = {"kick": 36, "snare": 38, "hihat": 42, "ohat": 46, "ride": 51, "glitch": 44}
DRUM_LABELS = {"kick": "kick", "snare": "snare", "hihat": "closed hat", "ohat": "open hat",
               "ride": "ride", "glitch": "hat stutter"}
# a Model:Cycles as the drum machine: one track per role, the box's default
# track channels; a note on a track channel plays that track chromatically,
# and CYCLES_ROOT is the note that leaves the track's tuning alone: C4 (60),
# the pitch the box's own pads send (the manual's 49-semitone range on the
# AUTO channel, notes 12 to 60, is a different mapping; 36 was tried first)
CYCLES_CHANNELS = {"kick": 1, "snare": 2, "hihat": 3, "ohat": 4, "ride": 5, "glitch": 6}
CYCLES_ROOT = 60
# the box's sound over CC, Appendix A of the manual (MODEL_CYCLES_MIDI.md):
# (key, cc, label, kind, default, what) per track, on the track's channel.
# kind: "u" 0..127 sent as is; "b" -64..63, sent + 64; "semi" -24..+24
# semitones spread over 0..127 (64 = 0); "pct" 0..100 % over 0..127;
# "swing" 50..80 % over 0..127; "on" a switch, 0 or 127; "list" a menu,
# its index sent (the menu's order, 0 first: the manual gives no values,
# verify on the box); "dt" the delay time 1..128 in 128ths, sent - 1
# the Model-TG firmware (github.com/TinyGregAudio/Model-TG, on the box since
# 2026-10) adds a seventh machine, the Sampler: machine index 6 on the box
# (stock OS 1.13 clamps the index to 5, so the value means nothing there)
CYCLES_MACHINES = ["kick", "snare", "metal", "perc", "tone", "chord", "sampler"]
CYCLES_LFO_WAVES = ["tri", "sine", "square", "saw", "env", "ramp", "random"]
# Model-TG adds attack, filter and resonance to the five stock machines as
# LFO destinations (parameter indices 23..25, after the stock list; a
# Sampler track does not list them, its own filter sits in sweep / contour)
CYCLES_LFO_DESTS = ["none", "pitch", "fine tune", "decay", "color", "shape", "sweep",
                    "contour", "delay send", "reverb send", "vol + dist", "pan", "punch",
                    "gate", "attack", "filter", "resonance"]
# the Sampler's four page knobs sit in the machine slots color, shape, sweep
# and contour (the same track words the stock page writes, so CC 16..19 by
# the stock mapping, to verify on the box) and change job with the playback
# mode, which only the box sets (SETTINGS + PRESET); the tab is told which:
# (mode, the short names for the cells, the long names for the tooltips)
CYCLES_SAMPLER_KNOBS = ("color", "shape", "sweep", "contour")
_SAMPLE_WINDOW = (("start", "end", "filt", "res"),
                  ("sample start (above end plays it reversed)", "sample end",
                   "filter cutoff, a 4-pole ladder low-pass", "filter resonance"))
CYCLES_SAMPLER_MODES = [
    ("one shot", *_SAMPLE_WINDOW),
    ("loop", *_SAMPLE_WINDOW),
    ("slice", ("start", "end", "filt", "res"),
     ("start, inside the slice the note picks", "end, inside the slice",
      "filter cutoff", "filter resonance")),
    ("granular", ("pos", "size", "dens", "spray"),
     ("grain position in the sample", "grain size, 10 ms to 1 s", "grain density",
      "position spray")),
    ("stretch", ("speed", "size", "dens", "spray"),
     ("playhead speed: 0 frozen, 96 the original, 127 about 3.8 times",
      "grain size", "grain density", "position spray")),
    ("pluck", ("strike", "len", "tone", "fdbk"),
     ("where the string is struck", "strike length, 1 ms to 1 s", "string tone",
      "string feedback")),
    ("wave", ("wave", "frame", "smooth", "detune"),
     ("wavetable position", "frame size, 32 to 4096 samples (2048 for Serum and "
      "Vital tables)", "frame smoothing", "detune")),
]
# Slice mode: the note picks the slice, slice 1 at note 52 (the box's first
# trig key), one a semitone, so the root 60 plays slice 9
CYCLES_SLICE_NOTE = 52
# the multiplier: the tempo-synced set (x the tempo) then the fixed one
# (x 120 BPM, free of the tempo); the box's own order to verify
_MULTS = ("1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1k", "2k")
CYCLES_LFO_MULTS = [f"x{m}" for m in _MULTS] + [f"{m} fixed" for m in _MULTS]
CYCLES_PARAMS = [
    ("machine", 70, "machine", "list", 0,
     "the machine the track runs: kick, snare, metal, perc, tone, chord, and the "
     "Model-TG firmware's sampler (OS 1.13 takes it over CC; the menu order is a "
     "guess, verify on the box)"),
    ("pitch", 65, "pitch", "semi", 0,
     "tuning, -24 to +24 semitones; the box slews a CC jump into a glide, a firmware "
     "quirk with no switch, so a step sounds as a portamento; not sent to a sampler "
     "track, where the pitch CC pins the pitch and ends its note tracking"),
    ("decay", 80, "decay", "u", 64, "the amp envelope's decay, 0 to 127"),
    ("color", 16, "color", "u", 64, "machine dependent: kick punch, snare noise..."),
    ("shape", 17, "shape", "u", 64, "machine dependent; chord: picks the chord"),
    ("sweep", 18, "sweep", "u", 64, "machine dependent: a pitch or filter sweep"),
    ("contour", 19, "contour", "u", 64, "machine dependent: the sweep's envelope"),
    ("volume", 7, "vol + dist", "u", 60,
     "level of the sound, 60 clean unity, above it distorts up to 127"),
    ("pan", 10, "pan", "b", 0, "left to right, 0 in the middle"),
    ("dsend", 12, "delay send", "u", 0, "into the box's delay, 0 off"),
    ("rsend", 13, "reverb send", "u", 0, "into the box's reverb, 0 off"),
    ("level", 95, "level", "u", 100, "the track's LEVEL knob, 0 to 127"),
    ("chance", 14, "chance", "pct", 100,
     "the odds a trig of the box's own pattern plays, 0 to 100 % (our notes always "
     "play; a trig condition on the box overrides it)"),
    ("swing", 15, "swing", "swing", 50,
     "the swing of the box's own pattern, 50 % straight to 80 %"),
    ("punch", 66, "punch", "on", False,
     "punch: pseudo-compression and distortion, an accent, off or on"),
    ("gate", 67, "gate", "on", False,
     "gate: the envelope holds at full until the note off (our gate time) then decays; "
     "off, it decays from the note on and ignores the note off"),
    ("mute", 94, "mute", "on", False,
     "the track muted on the box (its notes and ours), off or on; the polarity, 0 "
     "unmuted and 127 muted, to verify"),
    ("lfo_speed", 102, "LFO speed", "b", 16,
     "-64 to 63; negative runs the cycle backwards; 8, 16, 32 sit on straight beats"),
    ("lfo_mult", 103, "LFO mult", "list", 4,
     "the speed times this: x the tempo (clocked, follows our MIDI clock) or a fixed "
     "120 BPM; the menu order to verify"),
    ("lfo_fade", 104, "LFO fade", "b", 0,
     "-64 to 63: negative fades the modulation in from the trig, positive fades it out"),
    ("lfo_dest", 105, "LFO dest", "list", 0,
     "what the LFO moves (Appendix B); the menu order, none first, to verify"),
    ("lfo_wave", 106, "LFO wave", "list", 0,
     "triangle, sine, square, saw, envelope (a one-shot from the trig, the speed its "
     "decay), ramp, random"),
    ("lfo_phase", 107, "LFO phase", "u", 0,
     "where the cycle starts at a trig, 0 the beginning, 64 the middle"),
    ("lfo_reset", 108, "LFO reset", "on", False,
     "off free-running, on restarts the cycle at every note"),
    ("lfo_depth", 109, "LFO depth", "b", 0,
     "-64 to 63, negative inverts, 0 none (the fine LSB, CC 110, goes out as 0)"),
]
CYCLES_FX_PARAMS = [
    ("delay_time", 85, "delay time", "dt", 48,
     "1 to 128 in 128ths of a whole note, following the tempo: 32 a quarter, 48 a "
     "dotted eighth, 64 a half"),
    ("delay_fb", 86, "delay feedback", "u", 64, "0 one echo to 127 forever"),
    ("rev_size", 87, "reverb size", "u", 64, "the space, 1 to 126, 127 infinite"),
    ("rev_tone", 88, "reverb tone", "b", 0,
     "-64 to 63: negative damps the highs, positive the lows"),
]
CYCLES_LISTS = {"machine": CYCLES_MACHINES, "lfo_mult": CYCLES_LFO_MULTS,
                "lfo_dest": CYCLES_LFO_DESTS, "lfo_wave": CYCLES_LFO_WAVES}
CYCLES_RANGES = {"u": (0, 127), "b": (-64, 63), "semi": (-24, 24), "pct": (0, 100),
                 "swing": (50, 80), "dt": (1, 128)}
# the retrig done from here (the box's own has no CC): the rate as a
# fraction of a whole note (a bar in 4/4), the box's own list; the length
# of the velocity curve in sixteenth steps, 0 = infinite (a flat curve)
CYCLES_RETRIG_RATES = [(f"1/{n}", 1.0 / n) for n in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24,
                                                    32, 40, 48, 64, 80)]
CYCLES_RETRIG_LENS = [("1/8 step", 0.125), ("1/4 step", 0.25), ("1/2 step", 0.5),
                      ("1 step", 1.0), ("2 steps", 2.0), ("4 steps", 4.0), ("8 steps", 8.0),
                      ("16 steps", 16.0), ("inf", 0.0)]


def cycles_cc_value(kind, v):
    """A panel value of this kind as the 0..127 the box takes."""
    if kind == "u":
        x = v
    elif kind == "b":
        x = v + 64
    elif kind == "semi":
        x = 64 + v * 64.0 / 24.0
    elif kind == "pct":
        x = v * 127.0 / 100.0
    elif kind == "swing":
        x = (v - 50) * 127.0 / 30.0
    elif kind == "on":
        x = 127 if v else 0
    elif kind == "dt":
        x = v - 1
    else:                                 # a list: its index
        x = v
    return max(0, min(127, int(round(x))))


def port_label(name):
    """'FCB1010:FCB1010 MIDI 1 20:0' -> 'FCB1010:FCB1010 MIDI 1'."""
    return _TRAILING_ADDR.sub("", name)


def midi_inputs():
    """[(label, full port name)] of every MIDI input right now, in system
    order, one entry per label."""
    if not MIDI_AVAILABLE:
        return []
    try:
        names = mido.get_input_names()
    except Exception:
        return []
    out, seen = [], set()
    for n in names:
        lab = port_label(n)
        if lab not in seen:
            seen.add(lab)
            out.append((lab, n))
    return out


def midi_outputs():
    """[(label, full port name)] of every MIDI output right now."""
    if not MIDI_AVAILABLE:
        return []
    try:
        names = mido.get_output_names()
    except Exception:
        return []
    out, seen = [], set()
    for n in names:
        lab = port_label(n)
        if lab not in seen:
            seen.add(lab)
            out.append((lab, n))
    return out


def format_binding(kind, channel, number):
    return f"{kind} {number} ch{channel}"


def parse_binding(text):
    """'cc 11 ch1' -> ('cc', 1, 11); None when it is not a binding."""
    if not text:
        return None
    m = re.fullmatch(r"(note|cc|pc)\s+(\d+)\s+ch(\d+)", text.strip())
    if not m:
        return None
    kind, number, channel = m.group(1), int(m.group(2)), int(m.group(3))
    if not (0 <= number <= 127 and 1 <= channel <= 16):
        return None
    return kind, channel, number


class MidiIn(QObject):
    """The open input ports. rtmidi calls back on its own thread; the signal
    is queued to the thread this object lives in (the GUI's)."""

    message = Signal(str, str, int, int, int)   # port label, kind, channel 1-16, number, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ports = {}        # label -> mido port

    @property
    def open_labels(self):
        return sorted(self._ports)

    def open(self, labels, present=None):
        """Make the set of open ports equal to `labels` (those that exist):
        opens the missing ones, closes the ones no longer wanted. `present`
        is midi_inputs() when the caller just scanned. Returns
        {label: error} for the ones that could not be opened."""
        want = set(labels)
        for lab in list(self._ports):
            if lab not in want:
                self._close(lab)
        errors = {}
        if not MIDI_AVAILABLE:
            return errors
        present = dict(midi_inputs() if present is None else present)
        for lab in want:
            if lab in self._ports or lab not in present:
                continue
            try:
                port = mido.open_input(
                    present[lab], callback=lambda msg, lab=lab: self._on(lab, msg))
            except Exception as e:                     # busy, vanished, ...
                errors[lab] = str(e)
                continue
            self._ports[lab] = port
        return errors

    def _close(self, lab):
        port = self._ports.pop(lab, None)
        if port is not None:
            try:
                port.callback = None
                port.close()
            except Exception:
                pass

    def close_all(self):
        for lab in list(self._ports):
            self._close(lab)

    def _on(self, lab, msg):
        t = msg.type
        if t == "note_on" and msg.velocity > 0:
            self.message.emit(lab, "note", msg.channel + 1, msg.note, msg.velocity)
        elif t in ("note_off", "note_on"):      # the release: the same note at 0
            self.message.emit(lab, "note", msg.channel + 1, msg.note, 0)
        elif t == "control_change":
            self.message.emit(lab, "cc", msg.channel + 1, msg.control, msg.value)
        elif t == "program_change":
            self.message.emit(lab, "pc", msg.channel + 1, msg.program, 127)


class MidiOut:
    """One output port and a scheduler: notes are handed in with the
    perf_counter() moment they should leave, from any thread (the audio
    thread does), and a sender thread puts them on the wire then. The port
    can be swapped or closed at any time; notes still due go out on the new
    port, or are dropped with the old one, after note offs for what sounds.
    The same thread runs the MIDI clock and the transport (module note)."""

    SPIN_S = 0.002                    # sleep up to here before the due time, then spin
    CC_GATHER_S = 0.002               # CCs from one moment leave together, this much later
    CC_MIN_S = 0.010                  # between two values of one controller (DIN: 1 ms a message)
    SLEW = 0.1                        # of a tick period, the most one grid() moves the clock
    PULL = 0.1                        # of the phase error one grid() takes up: the callback's
                                      # own timing wobble is smoothed out of the ticks
    START_AHEAD_S = 0.0005            # Start leaves this long before the downbeat's tick
    PPQN = 24

    def __init__(self):
        self.label = None
        self._port = None
        self._heap = []               # (due, seq, kind, channel 1-16, note, velocity)
        self._seq = 0
        self._sounding = set()        # (channel, note) with a note off still to come
        self._cond = threading.Condition()
        self._stop = False
        self._thread = None
        self.sent = 0                 # notes on that left, for the status line
        self.late_ms = 0.0            # how late the last notes on left, smoothed
        # clock and transport
        self._clock_on = False
        self.transport_on = False
        self._period = 60.0 / 120.0 / self.PPQN      # seconds per tick
        self._tick_due = None         # the next tick's moment, None while the clock is off
        self._running = False         # a Start went out and no Stop since
        self.ticks = 0
        self.bpm = 120.0              # what the ticks say, for the status line
        # control changes: gathered, one value per controller, paced
        self._cc_pending = {}         # (channel, cc) -> value waiting to leave
        self._cc_last = {}            # (channel, cc) -> the value last sent
        self._cc_ok_at = {}           # (channel, cc) -> when the next may leave
        self._cc_due = None           # the next flush, None while nothing waits
        self.ccs = 0                  # CCs that left, for the status line
        # retrigs: channel -> [note, velocity, period, length, fade, t0, next]
        self._retrigs = {}
        # mirrors: other ports (the Erae's) given the clock and the transport
        self._sinks = {}              # sink -> {"clock": bool, "transport": bool, "stop": bool}

    @property
    def is_open(self):
        return self._port is not None

    @property
    def clock_on(self):
        return self._clock_on

    @clock_on.setter
    def clock_on(self, on):
        with self._cond:
            self._clock_on = bool(on)
            self._retick_locked()
            self._cond.notify()

    def _wants(self, what):
        return any(f[what] for f in self._sinks.values())

    def _retick_locked(self):
        """The ticks run while the port wants them or a mirror does; a
        clock already running keeps its phase."""
        if self._clock_on or self._wants("clock"):
            if self._tick_due is None:
                self._tick_due = time.perf_counter()
        else:
            self._tick_due = None

    @property
    def mirrored(self):
        """A mirror is listening (the grid and the transport matter even
        with no port open)."""
        return bool(self._sinks)

    def mirror(self, sink, clock=False, transport=False, stop=False):
        """Give `sink` (anything with realtime(byte)) the clock ticks and
        the Start / Stop bytes too, from the sender thread, whatever the
        panel's own ticks say; `stop` False keeps the Stops from it (a
        looper that should run on when a section clears). Both False
        forgets the sink."""
        with self._cond:
            if clock or transport:
                self._sinks[sink] = {"clock": bool(clock), "transport": bool(transport),
                                     "stop": bool(stop)}
            else:
                self._sinks.pop(sink, None)
            self._retick_locked()
            self._cond.notify()
        self._ensure_thread()

    def _fan(self, byte, what):
        """A realtime byte to every mirror that asked for `what`."""
        for sink, flags in list(self._sinks.items()):
            if flags[what]:
                try:
                    sink.realtime(byte)
                except Exception:
                    pass

    def _ensure_thread(self):
        with self._cond:
            if self._thread is None:
                self._stop = False
                self._thread = threading.Thread(target=self._run, name="gacha-midi-out",
                                                daemon=True)
                self._thread.start()

    @property
    def running(self):
        """A Start went out and no Stop since."""
        return self._running

    def open(self, label, present=None):
        """Open the output with this label (closing the one before); returns
        an error text, or None. `present` is midi_outputs() if just scanned."""
        if not MIDI_AVAILABLE:
            return "mido / python-rtmidi missing"
        present = dict(midi_outputs() if present is None else present)
        if label not in present:
            self.close_port()
            return f"{label}: not present"
        if label == self.label and self._port is not None:
            return None
        try:
            port = mido.open_output(present[label])
        except Exception as e:
            self.close_port()
            return str(e)
        with self._cond:
            old = self._port
            self._all_off_locked(old)
            self._port, self.label = port, label
            self._cc_last.clear()               # a new port knows nothing yet
            self._retick_locked()
            self._cond.notify()
        self._ensure_thread()
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        return None

    def _push(self, due, kind, channel=0, note=0, velocity=0):
        self._seq += 1
        heapq.heappush(self._heap, (due, self._seq, kind, channel, note, velocity))

    def note(self, channel, note, velocity, due, gate=0.05):
        """A note on at `due` (perf_counter seconds) and its off `gate` s
        later; a due time in the past goes out at once."""
        if self._port is None or not 0 <= note <= 127:
            return
        velocity = max(1, min(127, int(velocity)))
        with self._cond:
            self._push(due, "on", channel, note, velocity)
            self._push(due + gate, "off", channel, note, 0)
            self._cond.notify()

    def cc(self, channel, cc, value, force=False):
        """A control change, out within a few ms: the last value given for
        a controller is the one that leaves, a value equal to the one last
        sent on it is dropped unless `force`, and a controller gets a new
        value every CC_MIN_S at most (the rest wait their turn)."""
        if self._port is None or not 0 <= cc <= 127:
            return
        value = max(0, min(127, int(round(value))))
        with self._cond:
            key = (channel, cc)
            if force:
                self._cc_last.pop(key, None)
            self._cc_pending[key] = value
            if self._cc_due is None:
                self._cc_due = time.perf_counter() + self.CC_GATHER_S
            self._cond.notify()

    def retrig_start(self, channel, note, velocity, period_s, length_s=0.0, fade=0):
        """The note on `channel` again every `period_s` from now until
        retrig_stop(channel): the velocity runs from `velocity` scaled by the
        fade (-128..127, the box's FDE: negative fades out, positive in, both
        complete at the ends) over `length_s`, then stays; 0 = infinite, a
        flat curve. A retrig already on the channel starts over."""
        if self._port is None or not 0 <= note <= 127 or period_s <= 0:
            return
        now = time.perf_counter()
        with self._cond:
            self._retrigs[channel] = [int(note), max(1, min(127, int(velocity))),
                                      float(period_s), max(0.0, float(length_s)),
                                      max(-128, min(127, int(fade))), now, now]
            self._cond.notify()

    def retrig_stop(self, channel=None):
        """The retrig on `channel` over, or every one; the note sounding
        gets its off as scheduled."""
        with self._cond:
            if channel is None:
                self._retrigs.clear()
            else:
                self._retrigs.pop(channel, None)

    @property
    def retrigging(self):
        return bool(self._retrigs)

    # ---- clock and transport (audio thread)
    def grid(self, beat_t, beat_s):
        """The beat grid the clock follows: `beat_s` seconds per beat and,
        `beat_t` given, the perf_counter moment of a beat (any beat) the
        ticks should fall on; None keeps them free-running at that tempo.
        Called every block; the phase is pulled over by PULL of the error
        each time, SLEW of a tick at most, a lock with a time constant of
        ten blocks: the block's own lateness (a few ms either way) would
        otherwise land in the ticks as jitter, and a receiver reading the
        tempo off short windows shows it a beat or so high."""
        if beat_s <= 0:
            return
        with self._cond:
            self._period = beat_s / self.PPQN
            self.bpm = 60.0 / beat_s
            if beat_t is not None and self._tick_due is not None:
                per = self._period
                target = beat_t + round((self._tick_due - beat_t) / per) * per
                err = target - self._tick_due
                self._tick_due += max(-self.SLEW * per, min(self.SLEW * per, self.PULL * err))
            self._cond.notify()

    def start(self, due):
        """Start at `due` (perf_counter): the message just before, the tick
        exactly there, so the receiver's first step is that moment."""
        if not (self._port is not None and self.transport_on) and not self._wants("transport"):
            return
        with self._cond:
            self._push(due - self.START_AHEAD_S, "start")
            if self._tick_due is not None:
                self._tick_due = due
            self._cond.notify()

    def stop(self):
        """Stop, now, if a Start went out."""
        with self._cond:
            if self._running:
                self._push(time.perf_counter(), "stop")
                self._cond.notify()

    def all_off(self):
        """Note offs for everything that sounds, the queue dropped, Stop
        if running."""
        with self._cond:
            self._all_off_locked(self._port)

    def _all_off_locked(self, port):
        self._heap.clear()
        self._retrigs.clear()
        self._cc_pending.clear()
        self._cc_due = None
        for ch, n in list(self._sounding):
            self._send(port, mido.Message("note_off", channel=ch - 1, note=n, velocity=0))
        self._sounding.clear()
        if self._running:
            if self.transport_on:
                self._send(port, mido.Message("stop"))
            self._fan(0xFC, "stop")
            self._running = False

    def close_port(self):
        """Close the port; the sender thread stays for the next open()."""
        with self._cond:
            port, self._port, self.label = self._port, None, None
            self._all_off_locked(port)
        if port is not None:
            try:
                port.close()
            except Exception:
                pass

    def close(self):
        self.close_port()
        with self._cond:
            self._stop = True
            self._cond.notify()
        t = self._thread
        if t is not None:
            t.join(1.0)
        self._thread = None

    @staticmethod
    def _send(port, msg):
        if port is None:
            return
        try:
            port.send(msg)
        except Exception:
            pass

    def _cc_flush_locked(self, port, now):
        """Every pending CC whose controller is free to take one, the
        rest wait until the first of them is."""
        later = math.inf
        for key, val in list(self._cc_pending.items()):
            ok = self._cc_ok_at.get(key, 0.0)
            if ok > now:
                later = min(later, ok)
                continue
            del self._cc_pending[key]
            if self._cc_last.get(key) == val:
                continue
            self._cc_last[key] = val
            self._cc_ok_at[key] = now + self.CC_MIN_S
            if port is not None:
                self._send(port, mido.Message("control_change", channel=key[0] - 1,
                                              control=key[1], value=val))
                self.ccs += 1
        self._cc_due = later if later < math.inf else None

    def _retrig_fire_locked(self, port, now):
        """The retrigs whose next note is due: the note on now, its off
        half a period later (50 ms at most), the next one a period on."""
        for ch, r in self._retrigs.items():
            note, vel, per, length, fade, t0, nxt = r
            if nxt > now + 0.0005:
                continue
            frac = min(1.0, (nxt - t0) / length) if length > 0 else 0.0
            if fade > 0:                            # in: from what the fade leaves, up to full
                v = vel * ((1.0 - fade / 128.0) + fade / 128.0 * frac)
            else:                                   # out: from full, down by the fade
                v = vel * (1.0 + fade / 128.0 * frac)
            v = max(1, min(127, int(round(v))))
            if port is not None:
                self._send(port, mido.Message("note_on", channel=ch - 1, note=note, velocity=v))
                self._sounding.add((ch, note))
                self.sent += 1
                self._push(nxt + min(0.05, per / 2), "off", ch, note, 0)
            nxt += per
            while nxt < now - per:                  # a stall: skip the notes missed
                nxt += per
            r[6] = nxt

    def _run(self):
        if sys.platform == "win32":        # 1 ms scheduler ticks while we send
            try:
                import ctypes
                ctypes.windll.winmm.timeBeginPeriod(1)
            except Exception:
                pass
        try:
            while True:
                with self._cond:
                    while not self._stop and not self._heap and self._tick_due is None \
                            and self._cc_due is None and not self._retrigs:
                        self._cond.wait()
                    if self._stop:
                        return
                    tick = self._tick_due if self._tick_due is not None else math.inf
                    ccd = self._cc_due if self._cc_due is not None else math.inf
                    rt = min((r[6] for r in self._retrigs.values()), default=math.inf)
                    due = min(self._heap[0][0] if self._heap else math.inf, tick, ccd, rt)
                    wait = due - time.perf_counter()
                    if wait > self.SPIN_S:
                        self._cond.wait(wait - self.SPIN_S)
                        continue                     # something newer may be first now
                    while time.perf_counter() < due:  # the last ms or two
                        pass
                    port = self._port
                    if tick <= due:                  # the tick first: it moves the receiver's step
                        now = time.perf_counter()
                        self._tick_due = tick + self._period
                        if self._tick_due < now - self._period:   # far behind (a stall): skip ahead
                            self._tick_due = now + self._period
                        if port is not None and self._clock_on:
                            self._send(port, mido.Message("clock"))
                        self._fan(0xF8, "clock")
                        self.ticks += 1
                        continue
                    if rt <= due:                    # a retrig's note, before any CC
                        self._retrig_fire_locked(port, time.perf_counter())
                        continue
                    if ccd <= due:
                        self._cc_flush_locked(port, time.perf_counter())
                        continue
                    due, _seq, kind, ch, note, vel = heapq.heappop(self._heap)
                    if kind == "on":
                        msg = mido.Message("note_on", channel=ch - 1, note=note, velocity=vel)
                        self._sounding.add((ch, note))
                        self.sent += 1
                        late = 1000.0 * (time.perf_counter() - due)
                        self.late_ms = 0.8 * self.late_ms + 0.2 * late
                    elif kind == "off":
                        msg = mido.Message("note_off", channel=ch - 1, note=note, velocity=0)
                        self._sounding.discard((ch, note))
                    elif kind == "start":
                        self._running = True
                        if self.transport_on:
                            self._send(port, mido.Message("start"))
                        self._fan(0xFA, "transport")
                        continue
                    else:
                        self._running = False
                        if self.transport_on:
                            self._send(port, mido.Message("stop"))
                        self._fan(0xFC, "stop")
                        continue
                    self._send(port, msg)
        finally:
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass
