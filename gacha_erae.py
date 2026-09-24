#!/usr/bin/env python3
"""
gacha_erae.py — the Erae 2's API zones as the show's control surface.

The Erae 2 (Embodme) is a pressure-sensitive LED surface. Its layout, made
in Erae Lab, can hold *API zones*: rectangles that send nothing of their
own but, once the host enables the API over SysEx, report every finger
(down / move / up, x, y in the zone's own LED units as floats, z the
pressure) and can be drawn on pixel by pixel. The rest of the layout (pads,
faders) keeps sending plain notes and CCs, which the Controls tab's MIDI
map already learns.

This module turns each API zone into one control for the set and paints
the app's state back on it, so the surface shows what the app is doing:

    the volume as a fader that fills with the level, the A/B pedal, the
    section pads (next, stop, event, burst, glitch, break, drop, randomize),
    the stem toggles lit when a stem is in, an X/Y pad for the picture's
    glitch and rgb shift that springs back to zero when the finger lifts,
    four faders for the focused rack (lowpass, delay, reverb, drive), an
    X/Y pad picking the shader and its opacity, and the focus buttons with
    the shader, video-only and live-input switches.

ROLES below is the order the controls take over the zones, lowest zone id
first; change the order there to match the layout on the device.

Everything speaks the Erae API V2 (Rev 004, 2025) on the device's first
MIDI port, named "Erae 2": enable with a receiver prefix (every reply
starts with it), boundary request, clear, draw image (at most 32 pixels
per message, RGB 7 bits a channel, origin bottom left), fingerstream. All
binary payloads are 7-bitized (a lead byte carrying the high bits of the
next seven). python-rtmidi is used directly for the SysEx; rtmidi calls
back on its own thread, and the signals here carry everything to the GUI
thread.

Checked on the device on 2026-09-11: fingers arrive at about 44 messages a
second per finger, x and y in LEDs with sub-LED precision, z from 0 (a
brush) to about 0.2 (firm), several fingers at once, and 24 image rows a
frame at 12 frames a second are absorbed without a drop.
"""
import math
import struct
import time
from functools import reduce

from PySide6.QtCore import QObject, QTimer, Signal

try:
    import rtmidi
    ERAE_AVAILABLE = True
except ImportError:                                 # the app runs without it
    rtmidi = None
    ERAE_AVAILABLE = False

PORT = "Erae 2"                     # the device's first port; the second (MIDIIN2) is silent for the API
PREFIX = [0x0A, 0x0B]               # our receiver prefix: replies start with it
ERAE2 = [0x00, 0x21, 0x50, 0x00, 0x01, 0x00, 0x02, 0x01, 0x01, 0x04]   # manufacturer + Erae 2 id
MAX_ZONES = 16                      # zone ids asked for at start
PIXELS_PER_MSG = 32                 # the spec's cap for a draw-image message
FLASH_S = 0.15                      # a pressed button lights this long

# colours, 7 bits a channel
AMBER = (0x7F, 0x48, 0x08)
TEAL = (0x10, 0x68, 0x60)
GREEN = (0x18, 0x7F, 0x20)
RED = (0x7F, 0x10, 0x10)
BLUE = (0x18, 0x40, 0x7F)
MAGENTA = (0x7F, 0x18, 0x60)
ORANGE = (0x7F, 0x30, 0x00)
VIOLET = (0x50, 0x20, 0x7F)
WHITE = (0x70, 0x70, 0x70)
LIME = (0x50, 0x7F, 0x10)
OFF = (0, 0, 0)


def dim(c, k):
    return tuple(int(v * k) for v in c)


def bitize7(data):
    out = []
    for i in range(0, len(data), 7):
        chunk = data[i:i + 7]
        out.append(sum((el & 0x80) >> (j + 1) for j, el in enumerate(chunk)))
        out.extend(el & 0x7F for el in chunk)
    return out


def unbitize7(bits):
    out = []
    for i in range(0, len(bits), 8):
        lead, chunk = bits[i], bits[i + 1:i + 8]
        out.extend(((lead << (j + 1)) & 0x80) | b for j, b in enumerate(chunk))
    return out


def xor(bs):
    return reduce(lambda a, b: a ^ b, bs, 0)


def frac(coord, size):
    """A finger coordinate in LEDs -> 0..1 along that axis, the outer half
    LED on both ends counting as the ends (a finger stops short of the rim)."""
    if size <= 1:
        return 0.0
    return max(0.0, min(1.0, (coord - 0.5) / (size - 1)))


# ------------------------------------------------------------------ controls
class Control:
    """One API zone's control. `attach(w, h)` learns the size; `touch()`
    turns a finger event into [(ident, value | None)]; `frame(read, now)`
    paints it as rows of (r, g, b), row 0 at the bottom."""
    name = "control"

    def __init__(self):
        self.w = self.h = 0

    def attach(self, w, h):
        self.w, self.h = w, h

    def touch(self, action, fid, x, y, z):
        return []

    def frame(self, read, now):
        return [[OFF] * self.w for _ in range(self.h)]

    def blank(self):
        return [[OFF] * self.w for _ in range(self.h)]


class Fader(Control):
    """One value along the zone's long side, filled up to it."""

    def __init__(self, ident, colour, name=None):
        super().__init__()
        self.ident, self.colour = ident, colour
        self.name = name or ident
        self.vertical = True

    def attach(self, w, h):
        super().attach(w, h)
        self.vertical = h >= w

    def touch(self, action, fid, x, y, z):
        if action not in ("down", "move"):
            return []
        v = frac(y, self.h) if self.vertical else frac(x, self.w)
        return [(self.ident, v)]

    def frame(self, read, now):
        v = read(self.ident)
        v = 0.0 if v is None else float(v)
        px = self.blank()
        if self.vertical:
            top = v * (self.h - 1)
            for row in range(self.h):
                if row < top:
                    for cx in range(self.w):
                        px[row][cx] = dim(self.colour, 0.35)
            r = int(round(top))
            for cx in range(self.w):
                px[r][cx] = self.colour
        else:
            right = v * (self.w - 1)
            for cx in range(self.w):
                if cx < right:
                    for row in range(self.h):
                        px[row][cx] = dim(self.colour, 0.35)
            c = int(round(right))
            for row in range(self.h):
                px[row][c] = self.colour
        return px


class Faders(Control):
    """A bank of vertical faders side by side, one ident each."""

    def __init__(self, items, name="faders"):
        super().__init__()
        self.items = list(items)                 # [(ident, colour)]
        self.name = name

    def _col(self, x):
        n = len(self.items)
        return max(0, min(n - 1, int(frac(x, self.w) * n * 0.9999)))

    def touch(self, action, fid, x, y, z):
        if action not in ("down", "move"):
            return []
        return [(self.items[self._col(x)][0], frac(y, self.h))]

    def frame(self, read, now):
        px = self.blank()
        n = len(self.items)
        for i, (ident, colour) in enumerate(self.items):
            x0 = int(round(i * self.w / n))
            x1 = int(round((i + 1) * self.w / n)) - (1 if self.w >= 2 * n else 0)
            v = read(ident)
            v = 0.0 if v is None else float(v)
            top = v * (self.h - 1)
            for row in range(self.h):
                lit = row < top
                for cx in range(x0, max(x0 + 1, x1)):
                    if lit:
                        px[row][cx] = dim(colour, 0.35)
            r = int(round(top))
            for cx in range(x0, max(x0 + 1, x1)):
                px[r][cx] = colour
        return px


class Pads(Control):
    """A grid of buttons. `kind`: "button" (flashes when pressed), "toggle"
    (lit when `read(ident)` is true), "beat" (a button that pulses with
    read("beat"), the live clock's beat while a section plays)."""

    def __init__(self, items, name="pads"):
        super().__init__()
        self.items = list(items)                 # [(ident, colour, kind)]
        self.name = name
        self.cols = self.rows = 1
        self.pressed = {}                        # index -> time pressed

    def attach(self, w, h):
        super().attach(w, h)
        n = len(self.items)
        best = None
        for cols in range(1, n + 1):
            rows = math.ceil(n / cols)
            cell = min(w / cols, h / rows)
            if best is None or cell > best[0] + 1e-9:
                best = (cell, cols, rows)
        self.cols, self.rows = best[1], best[2]

    def _cell(self, i):
        """Pixel box (x0, y0, x1, y1) of pad i, gaps of one LED when there
        is room; pads fill the grid top-left first."""
        c, r = i % self.cols, i // self.cols
        cw, ch = self.w / self.cols, self.h / self.rows
        x0, x1 = int(round(c * cw)), int(round((c + 1) * cw))
        ytop, ybot = self.h - int(round(r * ch)), self.h - int(round((r + 1) * ch))
        gap = 1 if cw >= 3 and ch >= 3 else 0
        return x0, ybot, x1 - gap, ytop - gap

    def _hit(self, x, y):
        c = min(self.cols - 1, int(frac(x, self.w) * self.cols * 0.9999))
        r = min(self.rows - 1, int((1.0 - frac(y, self.h)) * self.rows * 0.9999))
        i = r * self.cols + c
        return i if i < len(self.items) else None

    def touch(self, action, fid, x, y, z):
        if action != "down":
            return []
        i = self._hit(x, y)
        if i is None:
            return []
        self.pressed[i] = time.monotonic()
        return [(self.items[i][0], None)]

    def frame(self, read, now):
        px = self.blank()
        for i, (ident, colour, kind) in enumerate(self.items):
            x0, y0, x1, y1 = self._cell(i)
            flash = now - self.pressed.get(i, -1.0) < FLASH_S
            if flash:
                c = WHITE
            elif kind == "toggle":
                c = colour if read(ident) else dim(colour, 0.12)
            elif kind == "beat":
                phase = read("beat")
                c = dim(colour, 0.35 + 0.65 * (1.0 - phase)) if phase is not None else dim(colour, 0.35)
            else:
                c = dim(colour, 0.35)
            for row in range(max(0, y0), min(self.h, y1)):
                for cx in range(max(0, x0), min(self.w, x1)):
                    px[row][cx] = c
        return px


class XY(Control):
    """Two values from one finger: x and y. `spring`: both back to zero
    when the last finger lifts (a gesture, not a setting)."""

    def __init__(self, ident_x, ident_y, colour, spring=False, name="xy"):
        super().__init__()
        self.ident_x, self.ident_y, self.colour, self.spring = ident_x, ident_y, colour, spring
        self.name = name
        self.fingers = set()

    def touch(self, action, fid, x, y, z):
        if action in ("down", "move"):
            self.fingers.add(fid)
            return [(self.ident_x, frac(x, self.w)), (self.ident_y, frac(y, self.h))]
        self.fingers.discard(fid)
        if self.spring and not self.fingers:
            return [(self.ident_x, 0.0), (self.ident_y, 0.0)]
        return []

    def frame(self, read, now):
        vx, vy = read(self.ident_x), read(self.ident_y)
        vx = 0.0 if vx is None else float(vx)
        vy = 0.0 if vy is None else float(vy)
        px = self.blank()
        cx, cy = int(round(vx * (self.w - 1))), int(round(vy * (self.h - 1)))
        for row in range(self.h):
            for c in range(self.w):
                px[row][c] = dim(self.colour, 0.06)
        for row in range(self.h):
            px[row][cx] = dim(self.colour, 0.45)
        for c in range(self.w):
            px[cy][c] = dim(self.colour, 0.45)
        px[cy][cx] = self.colour
        return px


def default_roles():
    """The controls, in the order they take the zones (lowest id first)."""
    return [
        Fader("volume", AMBER, "main volume"),
        Fader("ab", TEAL, "A/B"),
        Pads([("next", GREEN, "beat"), ("stop", RED, "button"),
              ("event", BLUE, "button"), ("burst", MAGENTA, "button"),
              ("glitch", ORANGE, "button"), ("break", ORANGE, "button"),
              ("drop", WHITE, "button"), ("randomize_fx", VIOLET, "button")],
             "section"),
        Pads([("stem_drums", AMBER, "toggle"), ("stem_layers", TEAL, "toggle"),
              ("stem_chops", LIME, "toggle"), ("stem_events", BLUE, "toggle"),
              ("record", RED, "toggle"), ("tap", WHITE, "button"),
              ("downbeat", WHITE, "button"), ("kind_next", VIOLET, "button")],
             "stems, take, clock"),
        XY("fx_glitch", "fx_rgb", MAGENTA, spring=True, name="picture: glitch x rgb"),
        Faders([("afx_lowpass", TEAL), ("afx_delay", BLUE),
                ("afx_reverb", VIOLET), ("afx_drive", ORANGE)], "focused rack"),
        XY("shader_pick", "shader_mix", LIME, spring=False, name="shader: pick x mix"),
        Pads([("focus_live", AMBER, "toggle"), ("focus_drums", AMBER, "toggle"),
              ("focus_layers", AMBER, "toggle"), ("focus_chops", AMBER, "toggle"),
              ("focus_events", AMBER, "toggle"), ("shader", LIME, "toggle"),
              ("video_only", WHITE, "toggle"), ("through", GREEN, "toggle")],
             "focus, switches"),
    ]


# ------------------------------------------------------------------ the surface
class EraeSurface(QObject):
    """The Erae 2 on its port: the API zones as controls, the rest of the
    layout forwarded as plain MIDI. `open()` joins, `close()` clears the
    zones and leaves API mode. Everything reaches the GUI thread through
    the signals; `action` is what a control asks of the app, `midi` the
    layout's own notes and CCs in the MidiIn signal's shape."""

    action = Signal(str, object)                 # ident, value 0..1 or None (a button)
    midi = Signal(str, str, int, int, int)       # port label, kind, channel, number, value
    log = Signal(str)
    _finger = Signal(int, int, int, float, float, float)   # zone, action code, finger, x, y, z
    _boundary = Signal(int, int, int)            # zone, w, h

    ACTIONS = {0: "down", 1: "move", 2: "up"}

    def __init__(self, read, roles=None, parent=None):
        super().__init__(parent)
        self.read = read                         # ident -> 0..1 | bool | None: the app's state
        self.roles = roles if roles is not None else default_roles()
        self._in = self._out = None
        self.zones = {}                          # zone id -> (w, h) answered
        self.controls = {}                       # zone id -> Control
        self._last = {}                          # zone id -> last frame sent
        self.label = PORT
        self.open_ok = False
        self._finger.connect(self._on_finger)
        self._boundary.connect(self._on_boundary)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.redraw)

    # ------------------------------------------------------------ ports
    @staticmethod
    def present():
        """Is an Erae 2 plugged in (its first port listed)?"""
        if not ERAE_AVAILABLE:
            return False
        try:
            return any(n.startswith(PORT) for n in rtmidi.MidiIn().get_ports())
        except Exception:
            return False

    def open(self):
        """Open both ports, enable the API and ask every zone its size; the
        controls attach half a second later. Returns an error text or None."""
        if not ERAE_AVAILABLE:
            return "python-rtmidi missing (pip install python-rtmidi)"
        try:
            inp, out = rtmidi.MidiIn(), rtmidi.MidiOut()
            ins, outs = inp.get_ports(), out.get_ports()
            pi = next((i for i, n in enumerate(ins) if n.startswith(PORT)), None)
            po = next((i for i, n in enumerate(outs) if n.startswith(PORT)), None)
            if pi is None or po is None:
                return f"no '{PORT}' port (is it plugged in?)"
            self.label = ins[pi]
            inp.open_port(pi)
            out.open_port(po)
        except Exception as e:                   # busy (Erae Lab open?), vanished
            return f"cannot open '{PORT}': {e}"
        inp.ignore_types(sysex=False, timing=True, active_sense=True)
        inp.set_callback(self._on_message)
        self._in, self._out = inp, out
        self.open_ok = True
        self._send([0x02])                       # leave API mode first, as the spec asks
        QTimer.singleShot(60, self._enable)
        return None

    def _enable(self):
        if not self.open_ok:
            return
        self._send([0x01] + PREFIX)
        for z in range(MAX_ZONES):
            self._send([0x10, z])
        QTimer.singleShot(500, self._attach)

    def _attach(self):
        if not self.open_ok:
            return
        ids = sorted(z for z, (w, h) in self.zones.items() if w < 0x7F and h < 0x7F)
        if not ids:
            self.log.emit("erae: no API zone answered (load the layout with the API zones, "
                          "close Erae Lab)")
            return
        self.controls, self._last = {}, {}
        for z, ctl in zip(ids, self.roles):
            w, h = self.zones[z]
            ctl.attach(w, h)
            self.controls[z] = ctl
            self._send([0x20, z])
        spare = len(ids) - len(self.controls)
        self.log.emit(f"erae: {len(self.controls)} zone{'s' if len(self.controls) != 1 else ''}: "
                      + ", ".join(f"{z} {c.name} ({self.zones[z][0]}x{self.zones[z][1]})"
                                  for z, c in self.controls.items())
                      + (f", {spare} more unused" if spare > 0 else ""))
        self._timer.start(100)
        self.redraw(force=True)

    def close(self):
        """Blank the zones, leave API mode, close the ports."""
        self._timer.stop()
        if self._out is not None:
            for z in self.controls:
                try:
                    self._send([0x20, z])
                except Exception:
                    pass
            try:
                self._send([0x02])
            except Exception:
                pass
        for p in (self._in, self._out):
            if p is not None:
                try:
                    p.cancel_callback() if hasattr(p, "cancel_callback") else None
                    p.close_port()
                except Exception:
                    pass
        self._in = self._out = None
        self.open_ok = False
        self.controls, self.zones, self._last = {}, {}, {}

    # ------------------------------------------------------------ out
    def _send(self, body):
        if self._out is not None:
            self._out.send_message([0xF0] + ERAE2 + body + [0xF7])

    def _image(self, z, x, y, w, h, rgb):
        data = bitize7(rgb)
        self._send([0x23, z, x, y, w, h] + data + [xor(data)])

    def redraw(self, force=False):
        """Paint every control whose picture changed since the last time:
        row by row, in chunks of at most 32 pixels, only the rows that
        differ. Ten times a second from the timer; nothing is sent while
        nothing changes."""
        if not self.controls:
            return
        now = time.monotonic()
        for z, ctl in self.controls.items():
            try:
                px = ctl.frame(self.read, now)
            except Exception as e:               # noqa: BLE001 - a read failing must not kill the loop
                self.log.emit(f"erae: {ctl.name}: {e}")
                continue
            last = self._last.get(z)
            for row, cells in enumerate(px):
                if not force and last is not None and last[row] == cells:
                    continue
                for x0 in range(0, ctl.w, PIXELS_PER_MSG):
                    chunk = cells[x0:x0 + PIXELS_PER_MSG]
                    flat = [c for rgb in chunk for c in rgb]
                    self._image(z, x0, row, len(chunk), 1, flat)
            self._last[z] = px

    # ------------------------------------------------------------ in (rtmidi thread)
    def _on_message(self, event, _data=None):
        msg, _dt = event
        if not msg:
            return
        st = msg[0]
        if st == 0xF0:
            n = len(PREFIX)
            if msg[1:1 + n] != PREFIX:
                return                           # not ours (a dump, another host)
            body = msg[1 + n:-1]
            if len(body) < 2:
                return
            if body[0] == 0x7F:                  # a reply, not a finger
                if body[1] == 0x01 and len(body) >= 5:
                    self._boundary.emit(body[2], body[3], body[4])
                return
            if len(body) < 26:
                return
            action, zone = body[0] & 0x07, body[1]
            try:
                fid = struct.unpack("<Q", bytes(unbitize7(body[2:12])))[0]
                x, y, z = struct.unpack("<fff", bytes(unbitize7(body[12:26])))
            except (struct.error, ValueError):
                return
            self._finger.emit(zone, action, fid & 0x7FFFFFFF, x, y, z)
            return
        kind = st & 0xF0
        ch = (st & 0x0F) + 1
        if kind == 0x90 and len(msg) >= 3 and msg[2] > 0:
            self.midi.emit(self.label, "note", ch, msg[1], msg[2])
        elif kind == 0xB0 and len(msg) >= 3:
            self.midi.emit(self.label, "cc", ch, msg[1], msg[2])
        elif kind == 0xC0 and len(msg) >= 2:
            self.midi.emit(self.label, "pc", ch, msg[1], 127)

    # ------------------------------------------------------------ in (GUI thread)
    def _on_boundary(self, z, w, h):
        self.zones[z] = (w, h)

    def _on_finger(self, zone, code, fid, x, y, z):
        ctl = self.controls.get(zone)
        if ctl is None:
            return
        for ident, value in ctl.touch(self.ACTIONS.get(code, "up"), fid, x, y, z):
            self.action.emit(ident, value)
