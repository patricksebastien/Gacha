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
    SLEW = 0.1                        # of a tick period, the most one grid() moves the clock
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
            self._tick_due = time.perf_counter() if on else None
            self._cond.notify()

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
            if self._thread is None:
                self._stop = False
                self._thread = threading.Thread(target=self._run, name="gacha-midi-out",
                                                daemon=True)
                self._thread.start()
            if self._clock_on:
                self._tick_due = time.perf_counter()
            self._cond.notify()
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

    # ---- clock and transport (audio thread)
    def grid(self, beat_t, beat_s):
        """The beat grid the clock follows: `beat_s` seconds per beat and,
        `beat_t` given, the perf_counter moment of a beat (any beat) the
        ticks should fall on; None keeps them free-running at that tempo.
        Called every block; the phase is pulled over by SLEW of a tick at
        most each time."""
        if beat_s <= 0:
            return
        with self._cond:
            self._period = beat_s / self.PPQN
            self.bpm = 60.0 / beat_s
            if beat_t is not None and self._tick_due is not None:
                per = self._period
                target = beat_t + round((self._tick_due - beat_t) / per) * per
                err = target - self._tick_due
                self._tick_due += max(-self.SLEW * per, min(self.SLEW * per, err))
            self._cond.notify()

    def start(self, due):
        """Start at `due` (perf_counter): the message just before, the tick
        exactly there, so the receiver's first step is that moment."""
        if self._port is None or not self.transport_on:
            return
        with self._cond:
            self._push(due - self.START_AHEAD_S, "start")
            if self._tick_due is not None:
                self._tick_due = due
            self._cond.notify()

    def stop(self):
        """Stop, now, if a Start went out."""
        with self._cond:
            if self._running and self._port is not None:
                self._push(time.perf_counter(), "stop")
                self._cond.notify()

    def all_off(self):
        """Note offs for everything that sounds, the queue dropped, Stop
        if running."""
        with self._cond:
            self._all_off_locked(self._port)

    def _all_off_locked(self, port):
        self._heap.clear()
        for ch, n in list(self._sounding):
            self._send(port, mido.Message("note_off", channel=ch - 1, note=n, velocity=0))
        self._sounding.clear()
        if self._running:
            self._send(port, mido.Message("stop"))
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
                    while not self._stop and not self._heap and self._tick_due is None:
                        self._cond.wait()
                    if self._stop:
                        return
                    tick = self._tick_due if self._tick_due is not None else math.inf
                    due = min(self._heap[0][0] if self._heap else math.inf, tick)
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
                        if port is not None:
                            self._send(port, mido.Message("clock"))
                            self.ticks += 1
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
                        msg = mido.Message("start")
                        self._running = True
                    else:
                        msg = mido.Message("stop")
                        self._running = False
                    self._send(port, msg)
        finally:
            if sys.platform == "win32":
                try:
                    import ctypes
                    ctypes.windll.winmm.timeEndPeriod(1)
                except Exception:
                    pass
