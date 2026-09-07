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

Kinds are "note" (note on, velocity > 0), "cc" (control change) and "pc"
(program change, which is what an FCB1010 sends out of the box); channels
are 1 to 16. Note offs, clock, active sensing and the rest are dropped here.

A binding is a small string that the settings store and the Controls tab
shows: "note 60 ch1", "cc 11 ch1", "pc 5 ch1". The port is not part of it:
two controllers on different channels tell themselves apart by the channel,
and a binding survives the controller moving to another USB socket.

Port names from ALSA end in the client:port numbers ("FCB1010:FCB1010 MIDI 1
20:0"), which change with every plug; the label drops them so the ticked
ports are remembered across launches and re-enumerations.

Built on mido over python-rtmidi; without them everything here is inert and
MIDI_AVAILABLE is False.
"""
import re

from PySide6.QtCore import QObject, Signal

try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:              # pragma: no cover - the tab says what to install
    mido = None
    MIDI_AVAILABLE = False

KINDS = ("note", "cc", "pc")
_TRAILING_ADDR = re.compile(r"\s+\d+:\d+$")


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
        elif t == "control_change":
            self.message.emit(lab, "cc", msg.channel + 1, msg.control, msg.value)
        elif t == "program_change":
            self.message.emit(lab, "pc", msg.channel + 1, msg.program, 127)
