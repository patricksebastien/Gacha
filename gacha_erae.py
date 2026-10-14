#!/usr/bin/env python3
"""
gacha_erae.py — the Erae 2 as the show's control surface: eight pages.

The Erae 2 (Embodme) is a pressure-sensitive surface of 42 x 24 RGB LEDs
with eight layouts picked by the N1..N8 buttons. Patrick's layout gives
every one of the eight a single *API zone* covering the whole surface:
a rectangle that sends nothing of its own but, once the host enables the
API over SysEx, reports every finger (down / move / up, x, y in LEDs as
floats, z the pressure) and can be drawn on pixel by pixel. So each
layout is a blank 42 x 24 page this module draws and reads: a page of
faders, pads and X/Y pads for one part of the performance, lit with the
app's state. PAGES below is the catalogue; which page sits on which N
button is the Erae tab's choice (`pages` given to the surface).

What the device tells us that the API does not: pressing N1..N8 sends a
control change on channel 16, CC 102 + k at 127 for layout k coming up
(and 0 for the one left), so the surface always knows the page on
screen (verified on the device on 2026-10-08, with firmware 2.0.8x).

Sync: the Erae's own looper locks to the MIDI clock arriving on its USB
port when its clock source is "USB device" (Settings > MIDI clock on the
device). Windows gives a MIDI port to one program, and this module holds
the Erae's, so the app's MIDI clock and transport (gacha_midi.MidiOut)
are mirrored here through `realtime()`, called from the sender thread;
the surface's own SysEx and those bytes share one lock so a tick never
lands inside a picture. Verified on 2026-10-08: the Home screen follows
our tempo and the picture stays clean with the clock running.

Everything speaks the Erae API V2 on the device's first MIDI port, named
"Erae 2": enable with a receiver prefix (every reply starts with it),
boundary request, clear, draw image (at most 32 pixels per message, RGB 7
bits a channel, origin bottom left), fingerstream. All binary payloads are
7-bitized (a lead byte carrying the high bits of the next seven).
python-rtmidi is used directly for the SysEx; rtmidi calls back on its own
thread, and the signals here carry everything to the GUI thread.

Checked on the device: fingers arrive at about 44 messages a second per
finger, x and y in LEDs with sub-LED precision, z from 0 (a brush) to
about 0.35 (a hard press), several fingers at once, and 24 image rows a
frame at 12 frames a second are absorbed without a drop.

    surf = EraeSurface(read, pages=["mix", "racks", ...])   # read(ident) -> the app's state
    surf.action.connect(on_action)       # (ident, 0..1 | None): a knob or a press
    surf.hold.connect(on_hold)           # (ident, down): a held action
    surf.open()                          # the port, API mode, the pages drawn
    midi_out.mirror(surf, clock=True, transport=True)   # the clock to the device
"""
import math
import struct
import threading
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
W, H = 42, 24                       # the whole surface, what a page is drawn for
LAYOUT_CC = 102                     # N1..N8: CC 102..109 on channel 16, 127 = this layout up
LAYOUT_CH = 16
FLASH_S = 0.15                      # a pressed button lights this long
LONG_S = 0.7                        # a long press (a scene's save)
HIT_WAIT_S = 0.015            # s: a drum pad's pressure peak is read over this long
Z_FULL = 0.25                       # the pressure read as 1.0 (a hard press is about 0.35)

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
PINK = (0x7F, 0x40, 0x50)
OFF = (0, 0, 0)
KIND_COLOURS = {"groove": GREEN, "intro": TEAL, "outro": VIOLET, "break": ORANGE,
                "random": WHITE}


def dim(c, k):
    k = max(0.0, min(1.0, k))
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


def pressure(z):
    return max(0.0, min(1.0, z / Z_FULL))


def split(x0, x1, n, gap=1):
    """`n` cells over [x0, x1) with `gap` LEDs between them: [(x, w)]."""
    total = x1 - x0 + gap
    edges = [x0 + int(round(i * total / n)) for i in range(n + 1)]
    return [(edges[i], max(1, edges[i + 1] - gap - edges[i])) for i in range(n)]


# ------------------------------------------------------------------ widgets
class Widget:
    """A piece of a page at `rect` = (x, y, w, h), y from the bottom.
    Fingers reach it as local LED coordinates; what it asks of the app
    goes into `out` as ("set", ident, 0..1), ("press", ident) or
    ("hold", ident, down). `paint` writes into the page's rows."""

    def __init__(self, rect, label=""):
        self.x, self.y, self.w, self.h = rect
        self.label = label

    def contains(self, fx, fy):
        return self.x <= fx < self.x + self.w and self.y <= fy < self.y + self.h

    def down(self, fid, lx, ly, z, now, out):
        pass

    def move(self, fid, lx, ly, z, now, out):
        pass

    def up(self, fid, lx, ly, z, now, out):
        pass

    def tick(self, now, out):
        """Once a frame: what ripens with time (a long press)."""
        pass

    def paint(self, px, read, now):
        pass

    def idents(self):
        """The action ids this widget reaches, for the documentation."""
        return []

    # painting helpers, in local coordinates, clipped to the rect and the page
    def put(self, px, lx, ly, c):
        x, y = self.x + lx, self.y + ly
        if 0 <= lx < self.w and 0 <= ly < self.h and 0 <= y < len(px) and 0 <= x < len(px[0]):
            px[y][x] = c

    def fill(self, px, c, lx0=0, ly0=0, lw=None, lh=None):
        lw = self.w if lw is None else lw
        lh = self.h if lh is None else lh
        for ly in range(max(0, ly0), min(self.h, ly0 + lh)):
            for lx in range(max(0, lx0), min(self.w, lx0 + lw)):
                self.put(px, lx, ly, c)


class Fader(Widget):
    """One value along the long side, filled up to it; `bipolar` fills from
    the middle; `meter` names a second value drawn as a thin bar on the
    right edge (the output level next to the volume)."""

    def __init__(self, rect, ident, colour, label=None, bipolar=False, meter=None):
        super().__init__(rect, label or ident)
        self.ident, self.colour, self.bipolar, self.meter = ident, colour, bipolar, meter
        self.vertical = self.h >= self.w

    def _value(self, lx, ly):
        return frac(ly, self.h) if self.vertical else frac(lx, self.w)

    def down(self, fid, lx, ly, z, now, out):
        out.append(("set", self.ident, self._value(lx, ly)))

    move = down

    def idents(self):
        return [self.ident]

    def paint(self, px, read, now):
        v = read(self.ident)
        v = 0.0 if v is None else max(0.0, min(1.0, float(v)))
        n = self.h if self.vertical else self.w
        mw = 1 if self.meter and self.vertical and self.w >= 3 else 0
        lo, hi = (0.5, v) if self.bipolar else (0.0, v)
        a, b = sorted((lo * (n - 1), hi * (n - 1)))
        for i in range(n):
            lit = a - 0.5 <= i <= b + 0.5 if self.bipolar else i < b
            c = dim(self.colour, 0.3) if lit else dim(self.colour, 0.04)
            if self.vertical:
                for lx in range(self.w - mw):
                    self.put(px, lx, i, c)
            else:
                for ly in range(self.h):
                    self.put(px, i, ly, c)
        head = int(round(v * (n - 1)))
        if self.vertical:
            for lx in range(self.w - mw):
                self.put(px, lx, head, self.colour)
            if self.bipolar:
                self.put(px, 0, int(round(0.5 * (n - 1))), WHITE)
        else:
            for ly in range(self.h):
                self.put(px, head, ly, self.colour)
        if mw:
            lv = read(self.meter)
            lv = 0.0 if lv is None else max(0.0, min(1.0, float(lv)))
            top = lv * (n - 1)
            for ly in range(n):
                c = (GREEN if ly < 0.7 * n else AMBER) if ly <= top else OFF
                self.put(px, self.w - 1, ly, c)


class XY(Widget):
    """Two values from one finger, x and y, and the pressure as a third
    (`iz`) when asked. `spring`: back to zero when the last finger lifts
    (a gesture, not a setting)."""

    def __init__(self, rect, ix, iy, colour, label=None, iz=None, spring=False):
        super().__init__(rect, label or f"{ix} x {iy}")
        self.ix, self.iy, self.iz, self.colour, self.spring = ix, iy, iz, colour, spring
        self.fingers = set()

    def _emit(self, lx, ly, z, out):
        out.append(("set", self.ix, frac(lx, self.w)))
        out.append(("set", self.iy, frac(ly, self.h)))
        if self.iz:
            out.append(("set", self.iz, pressure(z)))

    def down(self, fid, lx, ly, z, now, out):
        self.fingers.add(fid)
        self._emit(lx, ly, z, out)

    move = down

    def up(self, fid, lx, ly, z, now, out):
        self.fingers.discard(fid)
        if self.spring and not self.fingers:
            out.append(("set", self.ix, 0.0))
            out.append(("set", self.iy, 0.0))
            if self.iz:
                out.append(("set", self.iz, 0.0))

    def idents(self):
        return [self.ix, self.iy] + ([self.iz] if self.iz else [])

    def paint(self, px, read, now):
        vx, vy = read(self.ix), read(self.iy)
        vx = 0.0 if vx is None else float(vx)
        vy = 0.0 if vy is None else float(vy)
        vz = read(self.iz) if self.iz else None
        base = dim(self.colour, 0.05 + 0.25 * float(vz or 0.0))
        self.fill(px, base)
        cx, cy = int(round(vx * (self.w - 1))), int(round(vy * (self.h - 1)))
        for ly in range(self.h):
            self.put(px, cx, ly, dim(self.colour, 0.4))
        for lx in range(self.w):
            self.put(px, lx, cy, dim(self.colour, 0.4))
        self.put(px, cx, cy, self.colour)


class Pad(Widget):
    """A button. `kind`: "button" (a press, flashes), "toggle" (lit when
    read(ident) is true), "hold" (held down and let go: the hold signal,
    lit while the app says it is held), "beat" (a press that pulses with
    read("beat")), "radio" (sets `ident` to `value`, lit when the app's
    value sits within `band` of it), "hit" (a drum pad: the finger's peak
    pressure over its first HIT_WAIT_S, or up to its lift if sooner, goes
    out as the value, a velocity; the first contact frame alone reads near
    zero). `ix`, `iy`, `iz`: knobs set from the
    finger's place and pressure at the press, and all along for a hold.
    `long`: another action after LONG_S of holding (the press then waits
    for the finger to lift, and a long hold does not press)."""

    def __init__(self, rect, ident, colour, label=None, kind="button", ix=None, iy=None,
                 iz=None, long=None, value=None, band=None):
        super().__init__(rect, label or ident)
        self.ident, self.colour, self.kind = ident, colour, kind
        self.ix, self.iy, self.iz, self.long = ix, iy, iz, long
        self.value, self.band = value, band
        self.fingers = {}                        # fid -> (t_down, lx, ly, long fired)
        self.pressed = -1.0
        self.hits = {}                           # fid -> [t_down, peak z] of a hit not yet fired

    def _axes(self, lx, ly, z, out):
        if self.ix:
            out.append(("set", self.ix, frac(lx, self.w)))
        if self.iy:
            out.append(("set", self.iy, frac(ly, self.h)))
        if self.iz:
            out.append(("set", self.iz, pressure(z)))

    def _press(self, now, out, z=0.0):
        self.pressed = now
        if self.kind == "radio":
            out.append(("set", self.ident, self.value))
        elif self.kind == "hit":
            self.last_vel = max(0.2, pressure(z))
            out.append(("set", self.ident, self.last_vel))
        else:
            out.append(("press", self.ident))

    def _hit(self, fid, z, now, out, done=False):
        """A hit pad's finger: its pressure peak is kept until the window
        has passed (or the finger lifts, `done`), then the hit goes out."""
        h = self.hits.get(fid)
        if h is None:
            return
        h[1] = max(h[1], z)
        if done or now - h[0] >= HIT_WAIT_S:
            del self.hits[fid]
            self._press(now, out, h[1])

    def down(self, fid, lx, ly, z, now, out):
        first = not self.fingers
        self.fingers[fid] = [now, lx, ly, False]
        self._axes(lx, ly, z, out)
        if self.kind == "hold":
            if first:
                self.pressed = now
                out.append(("hold", self.ident, True))
        elif self.kind == "hit":
            self.hits[fid] = [now, z]
        elif self.long is None:
            self._press(now, out, z)

    def move(self, fid, lx, ly, z, now, out):
        f = self.fingers.get(fid)
        if f is None:
            return
        f[1], f[2] = lx, ly
        if self.kind == "hold":
            self._axes(lx, ly, z, out)
        elif self.kind == "hit":
            self._hit(fid, z, now, out)

    def up(self, fid, lx, ly, z, now, out):
        f = self.fingers.pop(fid, None)
        if f is None:
            return
        if self.kind == "hold":
            if not self.fingers:
                out.append(("hold", self.ident, False))
        elif self.kind == "hit":
            self._hit(fid, z, now, out, done=True)
        elif self.long is not None and not f[3]:
            self._press(now, out)

    def tick(self, now, out):
        """Long presses ripen here (the frame loop); a hit whose finger
        stays still past the window too."""
        for fid in list(self.hits):
            self._hit(fid, 0.0, now, out)
        if self.long is None:
            return
        for f in self.fingers.values():
            if not f[3] and now - f[0] >= LONG_S:
                f[3] = True
                self.pressed = now
                out.append(("press", self.long))

    def idents(self):
        ids = [self.ident] + ([self.long] if self.long else [])
        return ids + [i for i in (self.ix, self.iy, self.iz) if i]

    def paint(self, px, read, now):
        flash = now - self.pressed < FLASH_S
        if flash:
            c = WHITE if self.kind != "hit" else dim(WHITE, 0.4 + 0.6 * getattr(self, "last_vel", 1.0))
        elif self.kind == "toggle":
            c = self.colour if read(self.ident) else dim(self.colour, 0.1)
        elif self.kind == "hold":
            c = self.colour if (self.fingers or read(self.ident)) else dim(self.colour, 0.18)
        elif self.kind == "radio":
            v = read(self.ident)
            on = v is not None and abs(float(v) - self.value) <= (self.band or 0.0) / 2
            c = self.colour if on else dim(self.colour, 0.1)
        elif self.kind == "beat":
            phase = read("beat")
            c = dim(self.colour, 0.3 + 0.7 * (1.0 - phase)) if phase is not None \
                else dim(self.colour, 0.3)
        else:
            c = dim(self.colour, 0.3)
        self.fill(px, c)
        if self.fingers and (self.ix or self.iy):   # the finger's place while held
            for f in self.fingers.values():
                self.put(px, int(round(f[1] - 0.5)), int(round(f[2] - 0.5)), WHITE)


class Strip(Widget):
    """The fragments: the bar as a row of slices, read("frag_div") picking
    how many (16 at most, a note per slice), the one under the playhead
    lit as the loop runs, a slice held (a hold on frag_k) sounding; `iy`
    sets a knob from the finger's height at the press."""

    def __init__(self, rect, colour, label="fragments: the bar in slices", iy=None):
        super().__init__(rect, label)
        self.colour, self.iy = colour, iy
        self.fingers = {}                        # fid -> slice k
        self._read = lambda ident: None          # the last frame's read, for the hit test

    def _n(self, read):
        v = read("frag_divs")                    # the slice count as an int
        return max(1, min(16, int(v))) if v else 16

    def _cells(self, n):
        return split(0, self.w, n, gap=1 if self.w >= 2 * n else 0)

    def _hit(self, lx, n):
        for k, (x0, w) in enumerate(self._cells(n), 1):
            if x0 <= lx < x0 + w + 1:
                return k
        return min(n, max(1, int(frac(lx, self.w) * n) + 1))

    def down(self, fid, lx, ly, z, now, out):
        k = self._hit(lx, self._n(self._read))
        self.fingers[fid] = k
        if self.iy:
            out.append(("set", self.iy, frac(ly, self.h)))
        if list(self.fingers.values()).count(k) == 1:
            out.append(("hold", f"frag_{k}", True))

    def up(self, fid, lx, ly, z, now, out):
        k = self.fingers.pop(fid, None)
        if k is not None and k not in self.fingers.values():
            out.append(("hold", f"frag_{k}", False))

    def idents(self):
        return [f"frag_{k}" for k in range(1, 17)] + ([self.iy] if self.iy else [])

    def paint(self, px, read, now):
        self._read = read                        # the hit test needs the slice count
        n = self._n(read)
        pos = read("pos")
        head = None if pos is None else int((pos % 1.0) * n)
        held = read("frag_held") or set()
        sounding = read("frag_slice")
        for k, (x0, w) in enumerate(self._cells(n), 1):
            if k in held or any(v == k for v in self.fingers.values()):
                c = self.colour
            elif sounding == k:
                c = dim(self.colour, 0.7)
            elif head == k - 1:
                c = dim(WHITE, 0.6)
            else:
                c = dim(self.colour, 0.08 if (k - 1) % 4 else 0.16)
            self.fill(px, c, x0, 0, w, self.h)


class Scenes(Widget):
    """The eight scenes: a tap loads one, a long press saves into it; the
    loaded one lit (pulsing while a morph runs), an empty slot dark."""

    def __init__(self, rect, colour=PINK, label="scenes: tap loads, hold saves"):
        super().__init__(rect, label)
        self.colour = colour
        self.pads = []
        self.fingers = {}                        # fid -> pad
        rows = 2 if self.h >= 2 * self.w / 4 else 1
        per = 8 // rows
        for r in range(rows):
            y0, hh = split(0, self.h, rows, gap=1)[rows - 1 - r]
            for i, (x0, w) in enumerate(split(0, self.w, per, gap=1)):
                n = r * per + i + 1
                self.pads.append(Pad((self.x + x0, self.y + y0, w, hh), f"scene_load_{n}",
                                     colour, f"scene {n}", long=f"scene_save_{n}"))

    def _pad(self, lx, ly):
        fx, fy = self.x + lx, self.y + ly
        return next((p for p in self.pads if p.contains(fx, fy)), None)

    def down(self, fid, lx, ly, z, now, out):
        p = self._pad(lx, ly)
        if p is not None:
            self.fingers[fid] = p
            p.down(fid, lx - (p.x - self.x), ly - (p.y - self.y), z, now, out)

    def up(self, fid, lx, ly, z, now, out):
        p = self.fingers.pop(fid, None)
        if p is not None:
            p.up(fid, 0, 0, z, now, out)

    def idents(self):
        return [i for p in self.pads for i in p.idents()]

    def tick(self, now, out):
        for p in self.pads:
            p.tick(now, out)

    def paint(self, px, read, now):
        cur = read("scene_cur")
        morph = read("morph")
        for n, p in enumerate(self.pads, 1):
            flash = now - p.pressed < FLASH_S
            if flash:
                c = WHITE
            elif cur == n - 1:
                k = 1.0 if morph is None else 0.5 + 0.5 * math.cos(now * 6.0)
                c = dim(self.colour, k)
            elif read(f"scene_empty_{n}"):
                c = dim(self.colour, 0.04)
            else:
                c = dim(self.colour, 0.2)
            p.fill(px, c)


class Status(Widget):
    """Two rows at the top of every page: the bar's sixteenths with the
    beat marked, the loop's bars, the A/B pedal; under it record, through,
    Link, MIDI out, the section kind, the output level, the volume."""

    def __init__(self, rect, label="status strip"):
        super().__init__(rect, label)

    def paint(self, px, read, now):
        top, low = self.h - 1, max(0, self.h - 2)
        bar = read("bar")
        cell = None if bar is None else int(bar * 16) % 16
        for i in range(16):
            base = 0.22 if i % 4 == 0 else 0.07
            c = WHITE if i == cell else dim(WHITE, base)
            self.put(px, i, top, c)
        pos, bars = read("pos"), read("bars")
        bars = int(bars or 8)
        for i in range(min(8, bars)):
            on = pos is not None and int(pos) == i
            self.put(px, 17 + i, top, GREEN if on else dim(GREEN, 0.1))
        ab = read("ab")
        ab = 0.0 if ab is None else float(ab)
        for i in range(16):
            self.put(px, 26 + i, top, TEAL if i <= ab * 15 else dim(TEAL, 0.06))
        rec = read("record")
        self.put(px, 0, low, RED if rec and int(now * 2) % 2 == 0 else dim(RED, 0.5 if rec else 0.08))
        self.put(px, 2, low, LIME if read("through") else dim(LIME, 0.08))
        self.put(px, 4, low, BLUE if read("link") else dim(BLUE, 0.08))
        self.put(px, 6, low, WHITE if read("midi_out") else dim(WHITE, 0.08))
        kind = read("kind")
        kc = KIND_COLOURS.get(kind, BLUE) if kind else OFF
        for i in range(8, 12):
            self.put(px, i, low, kc if kind else dim(WHITE, 0.04))
        lv = read("level")
        lv = 0.0 if lv is None else max(0.0, min(1.0, float(lv)))
        for i in range(15):
            on = i <= lv * 14
            c = (GREEN if i < 11 else AMBER) if on else dim(GREEN, 0.04)
            self.put(px, 13 + i, low, c)
        vol = read("volume")
        vol = 0.0 if vol is None else float(vol)
        for i in range(13):
            self.put(px, 29 + i, low, AMBER if i <= vol * 12 else dim(AMBER, 0.06))


# ------------------------------------------------------------------ pages
class Page:
    """One layout's zone: widgets at rects, fingers routed to the widget
    they landed on until they lift, the picture as rows of (r, g, b),
    row 0 at the bottom. `touch()` returns what the widgets asked."""

    def __init__(self, name, title, widgets, summary=""):
        self.name, self.title, self.widgets, self.summary = name, title, widgets, summary
        self.w, self.h = W, H
        self.owners = {}                         # finger id -> widget
        self.held = {}                           # ident -> how many holds are down

    def attach(self, w, h):
        self.w, self.h = w, h

    def blank(self):
        return [[OFF] * self.w for _ in range(self.h)]

    def widget_at(self, fx, fy):
        for wd in reversed(self.widgets):
            if wd.contains(fx, fy):
                return wd
        return None

    def _track(self, out):
        for ev in out:
            if ev[0] == "hold":
                n = self.held.get(ev[1], 0) + (1 if ev[2] else -1)
                if n > 0:
                    self.held[ev[1]] = n
                else:
                    self.held.pop(ev[1], None)

    def touch(self, action, fid, x, y, z, now):
        out = []
        if action == "down":
            wd = self.widget_at(x, y)
            if wd is None:
                return out
            self.owners[fid] = wd
            wd.down(fid, x - wd.x, y - wd.y, z, now, out)
        else:
            wd = self.owners.get(fid)
            if wd is None:
                return out
            if action == "move":
                wd.move(fid, x - wd.x, y - wd.y, z, now, out)
            else:
                self.owners.pop(fid, None)
                wd.up(fid, x - wd.x, y - wd.y, z, now, out)
        self._track(out)
        return out

    def release_all(self):
        """Every hold let go (the surface closing): what to tell the app."""
        out = [(ident, False) for ident in self.held]
        self.held.clear()
        self.owners.clear()
        for wd in self.widgets:
            if hasattr(wd, "fingers"):
                wd.fingers.clear()
            if hasattr(wd, "hits"):
                wd.hits.clear()
        return out

    def frame(self, read, now):
        out = []
        px = self.blank()
        for wd in self.widgets:
            wd.tick(now, out)
            wd.paint(px, read, now)
        self._track(out)
        return px, out

    def describe(self):
        """The widgets and their actions, one line each."""
        lines = [f"{self.title}: {self.summary}"]
        for wd in self.widgets:
            if isinstance(wd, Status):
                continue
            ids = wd.idents()
            kind = type(wd).__name__.lower()
            if isinstance(wd, Pad):
                kind = {"button": "pad", "toggle": "switch", "hold": "hold pad",
                        "beat": "pad", "radio": "pick", "hit": "drum pad"}.get(wd.kind, "pad")
            lines.append(f"  {wd.label} ({kind}): " + ", ".join(ids))
        return "\n".join(lines)


def _pads_row(y, h, items, x0=0, x1=W, gap=1):
    """A row of pads: items = (ident, colour, kind, label, extra kwargs)."""
    out = []
    for (x, w), it in zip(split(x0, x1, len(items), gap), items):
        ident, colour, kind, label = it[:4]
        kw = it[4] if len(it) > 4 else {}
        out.append(Pad((x, y, w, h), ident, colour, label, kind=kind, **kw))
    return out


def _faders_row(y, h, items, x0=0, x1=W, gap=1):
    """Faders side by side: items = (ident, colour, label, extra kwargs)."""
    out = []
    for (x, w), it in zip(split(x0, x1, len(items), gap), items):
        ident, colour, label = it[:3]
        kw = it[3] if len(it) > 3 else {}
        out.append(Fader((x, y, w, h), ident, colour, label, **kw))
    return out


STATUS_H = 2
TOP = H - STATUS_H                      # the first row the status strip takes
CHANNELS = ("live", "drums", "layers", "chops", "events")
VOICES = ("kick", "snare", "hihat", "ohat", "ride", "glitch", "sub")
VOICE_SHORT = {"kick": "kick", "snare": "snare", "hihat": "c.hat", "ohat": "o.hat",
               "ride": "ride", "glitch": "stutter", "sub": "sub"}
AUDIO_FX = ("highpass", "lowpass", "drive", "crush", "lofi", "chorus", "phaser", "pitch",
            "delay", "reverse", "reverb")


def page_mix():
    ws = [Status((0, TOP, W, STATUS_H))]
    ws.append(Fader((0, 8, 6, 14), "volume", AMBER, "main volume", meter="level"))
    cols = split(8, 38, 5, gap=1)
    for (x, w), ch in zip(cols, CHANNELS):
        ws.append(Fader((x, 8, w, 14), f"avol_{ch}", TEAL if ch != "live" else LIME,
                        f"{ch} volume"))
        tog = "through" if ch == "live" else f"stem_{ch}"
        ws.append(Pad((x, 4, w, 3), tog, LIME if ch == "live" else TEAL,
                      "live input through" if ch == "live" else f"{ch} stem", kind="toggle"))
    ws.append(Pad((38, 18, 4, 4), "link", BLUE, "Ableton Link", kind="toggle"))
    ws.append(Pad((38, 13, 4, 4), "midi_out", WHITE, "MIDI out", kind="toggle"))
    ws.append(Pad((38, 8, 4, 4), "metronome", WHITE, "metronome", kind="toggle"))
    ws.append(Scenes((0, 0, 32, 3)))
    ws.append(Pad((33, 0, 4, 3), "scene_morph", PINK, "scene morph", kind="toggle"))
    ws.append(Pad((38, 0, 4, 3), "video_off", RED, "video off", kind="toggle"))
    return Page("mix", "Mix", ws, "the main volume with the output level, the five channel "
                "volumes with the stems and the live input under them, Link, MIDI out, the "
                "metronome, the eight scenes (tap loads, hold saves) and the morph switch")


def page_racks():
    ws = [Status((0, TOP, W, STATUS_H))]
    items = [(f"focus_{ch}", AMBER, "toggle", f"focus {ch}") for ch in CHANNELS]
    items += [(f"focus_{v}", ORANGE, "toggle", f"focus {VOICE_SHORT[v]}") for v in VOICES]
    ws += _pads_row(18, 4, items, gap=1)
    fx = [(f"afx_{k}", BLUE if k not in ("pitch",) else VIOLET, k,
           {"bipolar": k == "pitch"}) for k in AUDIO_FX]
    fx.append(("avol", AMBER, "channel volume"))
    ws += _faders_row(0, 17, fx, gap=1)
    return Page("racks", "Racks", ws, "the focus pads (the five channels, the seven drum "
                "voices), then the focused rack's eleven effects and its volume as faders; "
                "two fingers move two faders at once")


def page_picture():
    ws = [Status((0, TOP, W, STATUS_H))]
    ws.append(XY((0, 6, 12, 16), "fx_glitch", "fx_rgb", MAGENTA,
                 "picture: glitch x rgb shift, pressure = pixel", iz="fx_pixel", spring=True))
    ws.append(XY((13, 6, 12, 16), "shader_pick", "shader_mix", LIME, "shader: pick x mix"))
    looks = [("fx_flash", WHITE, "flash"), ("fx_pump", ORANGE, "pump"),
             ("fx_trails", BLUE, "trails"), ("fx_zoom", TEAL, "zoom"),
             ("fx_kaleido", VIOLET, "kaleido"), ("fx_grain", AMBER, "grain")]
    ws += _faders_row(6, 16, looks, x0=25, x1=W, gap=1)
    row1 = [("shader", LIME, "toggle", "shader layer"), ("video_only", WHITE, "toggle", "video only"),
            ("next_fx", MAGENTA, "button", "next fx look"),
            ("randomize_fx", VIOLET, "button", "randomize fx"),
            ("grade_neutral", WHITE, "button", "grade neutral")]
    row0 = [("blend_next", LIME, "button", "next blend"),
            ("video_second", TEAL, "toggle", "webcam on B"),
            ("video_random", BLUE, "button", "random video"),
            ("video_prev", BLUE, "button", "previous video"),
            ("video_next", BLUE, "button", "next video")]
    ws += _pads_row(3, 2, row1, gap=1)
    ws += _pads_row(0, 2, row0, gap=1)
    return Page("picture", "Picture", ws, "glitch x rgb shift on a spring pad with the "
                "pressure on pixel, the shader pick x mix pad, six look faders, and the "
                "picture switches and video pads")


def page_section():
    ws = [Status((0, TOP, W, STATUS_H))]
    row3 = [("next", GREEN, "beat", "next section"), ("stop", RED, "button", "stop"),
            ("event", BLUE, "button", "event"),
            ("burst", MAGENTA, "button", "burst (height = length)", {"iy": "burst_len"}),
            ("glitch", ORANGE, "button", "glitch"), ("break", ORANGE, "button", "break"),
            ("drop", WHITE, "button", "drop"), ("kind_next", VIOLET, "button", "next kind")]
    ws += _pads_row(17, 5, row3, gap=1)
    row2 = [(f"voice_{v}", AMBER, "toggle", VOICE_SHORT[v]) for v in VOICES]
    row2.append(("voices_reset", WHITE, "button", "all in"))
    ws += _pads_row(11, 5, row2, gap=1)
    row1 = [(f"solo_{v}", ORANGE, "toggle", f"solo {VOICE_SHORT[v]}") for v in VOICES]
    row1.append(("quant_next", WHITE, "button", "next quantize"))
    ws += _pads_row(5, 5, row1, gap=1)
    row0 = [("picture_next", TEAL, "button", "next picture mode"),
            ("dissolve_next", TEAL, "button", "next dissolve"),
            ("half", BLUE, "toggle", "half time"), ("double", BLUE, "toggle", "double time"),
            ("pitch_lock", VIOLET, "toggle", "pitch lock"),
            ("tap", WHITE, "button", "tap tempo"), ("downbeat", WHITE, "button", "downbeat"),
            ("live", LIME, "toggle", "live mode")]
    ws += _pads_row(0, 4, row0, gap=1)
    return Page("section", "Section", ws, "the section pads (next on the beat, stop, event, "
                "burst with its length by height, glitch, break, drop, kind), the seven voices "
                "and their solos, the gears, the clock pads")


def page_time():
    ws = [Status((0, TOP, W, STATUS_H))]
    ws.append(Pad((0, 6, 16, 16), "repeat", AMBER, "repeat: x = position, y = division",
                  kind="hold", ix="rep_pos", iy="rep_div"))
    ws.append(Pad((17, 6, 16, 16), "freeze", VIOLET,
                  "freeze: x = pitch, y = grain size, pressure = smear",
                  kind="hold", ix="freeze_pitch", iy="freeze_size", iz="freeze_smear"))
    ws.append(Pad((34, 12, 8, 10), "brake", RED, "tape stop (height = length)", kind="hold",
                  iy="brake_len"))
    ws.append(Pad((34, 6, 4, 5), "nudge_down", BLUE, "nudge down", kind="hold"))
    ws.append(Pad((38, 6, 4, 5), "nudge_up", BLUE, "nudge up", kind="hold"))
    row0 = [("rep_rev", AMBER, "toggle", "repeat reversed"),
            ("half", BLUE, "toggle", "half time"), ("double", BLUE, "toggle", "double time"),
            ("pitch_lock", VIOLET, "toggle", "pitch lock"),
            ("lfo_trig", PINK, "button", "LFO retrigger"),
            ("tap", WHITE, "button", "tap tempo"), ("downbeat", WHITE, "button", "downbeat"),
            ("drop", WHITE, "button", "drop")]
    ws += _pads_row(0, 5, row0, gap=1)
    return Page("time", "Time", ws, "the held gestures: repeat with the position and the "
                "division under the finger, freeze with pitch, grain size and the pressure "
                "as smear, the tape stop with its length by height, the nudges, the gears")


def page_fragments():
    ws = [Status((0, TOP, W, STATUS_H))]
    ws.append(Strip((0, 14, W, 8), LIME))
    ws.append(XY((0, 0, 16, 13), "frag_motion", "frag_pitch", LIME,
                 "fragments: motion x pitch, pressure = repeats", iz="frag_repeats"))
    divs = [("frag_div", LIME, "radio", f"{d} slices", {"value": (i + 0.5) / 4, "band": 0.25})
            for i, d in enumerate((4, 8, 16, 32))]
    ws += _pads_row(9, 4, divs, x0=17, x1=33, gap=1)
    ws += _pads_row(9, 4, [("frag_dir", TEAL, "button", "grain direction"),
                           ("frag_play", TEAL, "button", "slice play mode")],
                    x0=34, x1=W, gap=1)
    ws += _faders_row(0, 8, [("frag_speed", VIOLET, "speed", {"bipolar": True}),
                             ("frag_size", LIME, "grain size"), ("frag_rate", LIME, "grain rate"),
                             ("frag_shape", TEAL, "grain shape")], x0=17, x1=33, gap=1)
    ws += _faders_row(0, 8, [("frag_a", WHITE, "attack"), ("frag_r", WHITE, "release")],
                      x0=34, x1=W, gap=1)
    return Page("fragments", "Fragments", ws, "the bar as slices with the playhead running, "
                "a slice held sounding as grains; motion x pitch with the pressure as repeats; "
                "the slice count, direction and play mode; speed, grain size, rate, shape, "
                "attack and release")


def page_macros():
    ws = [Status((0, TOP, W, STATUS_H))]
    ws += _faders_row(12, 10, [(f"macro_{n}", PINK, f"macro {n}") for n in range(1, 9)], gap=1)
    ws += _pads_row(9, 2, [(f"lfo_{n}", VIOLET, "toggle", f"LFO {n}") for n in range(1, 9)],
                    x0=0, x1=33, gap=1)
    ws += _faders_row(0, 8, [(f"lfo_{n}_depth", VIOLET, f"LFO {n} depth") for n in range(1, 9)],
                      x0=0, x1=33, gap=1)
    ws.append(Pad((34, 0, 4, 11), "lfo_trig", PINK, "LFO retrigger"))
    ws.append(Pad((38, 0, 4, 11), "lfo_off", RED, "every LFO off"))
    return Page("macros", "Macros and LFOs", ws, "the eight macro faders, the eight LFOs "
                "with their depths, retrigger and all off")


def page_drums():
    ws = [Status((0, TOP, W, STATUS_H))]
    top = [("hit_kick", RED, "hit", "kick"), ("hit_snare", AMBER, "hit", "snare"),
           ("hit_hihat", LIME, "hit", "closed hat"), ("hit_ohat", GREEN, "hit", "open hat")]
    ws += _pads_row(11, 11, top, gap=1)
    low = [("hit_ride", TEAL, "hit", "ride"), ("hit_glitch", ORANGE, "hit", "hat stutter"),
           ("burst", MAGENTA, "button", "burst (height = length)", {"iy": "burst_len"}),
           ("event", BLUE, "button", "event")]
    ws += _pads_row(0, 10, low, gap=1)
    return Page("drums", "Drums", ws, "a drum pad: the six voices' hits with the pressure as "
                "velocity (the Model:Cycles plays them too on its tracks), the burst with its "
                "length by height, the event")


def page_loop():
    return Page("loop", "Loop (native)", [], "nothing is drawn: this layout is meant to hold "
                "the Erae's own pads (Erae Lab, a note each, learned in the Controls tab) so "
                "the device's looper can record them; a zone here stays dark")


def page_empty():
    return Page("empty", "Empty", [], "nothing drawn, nothing read")


PAGES = {"mix": page_mix, "racks": page_racks, "picture": page_picture,
         "section": page_section, "time": page_time, "fragments": page_fragments,
         "macros": page_macros, "drums": page_drums, "loop": page_loop, "empty": page_empty}
DEFAULT_PAGES = ["mix", "racks", "picture", "section", "time", "fragments", "macros", "loop"]


def make_page(name):
    return PAGES.get(name, page_empty)()


SETUP_DOC = """\
On the device (Erae Lab): one API zone per layout, filling the surface (42 x 24), \
zone index 0 on layout N1 up to 7 on N8, finger data rate at its highest. \
Settings > MIDI clock > clock source: USB device, so the looper locks to the clock \
sent from here. Firmware 2.0.74 or later for the looper under an external clock.
Here: the surface takes the "Erae 2" port (Erae Lab must be closed) and the \
Controls tab's MIDI list lets go of it; the layout's own pads and faders still \
reach the MIDI map as plain notes and CCs. N1..N8 tell the app which page is on \
screen. The looper records the device's own pads only (a native layout), never \
the drawn pages."""


# ------------------------------------------------------------------ the surface
class EraeSurface(QObject):
    """The Erae 2 on its port: the API zones as pages, the rest of the
    layout forwarded as plain MIDI. `open()` joins, `close()` clears the
    zones and leaves API mode. Everything reaches the GUI thread through
    the signals; `action` is a knob or a press a page asks for, `hold` a
    held action going down or up, `midi` the layout's own notes and CCs
    in the MidiIn signal's shape, `page` the layout on screen (0..7)."""

    action = Signal(str, object)                 # ident, value 0..1 or None (a button)
    hold = Signal(str, bool)                     # ident, down
    midi = Signal(str, str, int, int, int)       # port label, kind, channel, number, value
    page = Signal(int)                           # the layout on screen, 0-based
    finger = Signal(int, int, int, float, float, float)   # zone, action code, finger, x, y, z
    log = Signal(str)
    _boundary = Signal(int, int, int)            # zone, w, h
    _layout = Signal(int)

    ACTIONS = {0: "down", 1: "move", 2: "up"}

    def __init__(self, read, pages=None, parent=None):
        super().__init__(parent)
        self.read = read                         # ident -> 0..1 | bool | None: the app's state
        self.page_names = list(pages) if pages else list(DEFAULT_PAGES)
        self._in = self._out = None
        self._lock = threading.Lock()            # the port: SysEx here, clock from the sender thread
        self.zones = {}                          # zone id -> (w, h) answered
        self.pages = {}                          # zone id -> Page
        self.ids = []                            # zone ids in order, N1 first
        self.active = 0                          # index into ids of the layout on screen
        self._last = {}                          # zone id -> last frame sent
        self._ticks = 0
        self.label = PORT
        self.open_ok = False
        self.messages = 0                        # image messages sent, for the status
        self.finger.connect(self._on_finger)
        self._boundary.connect(self._on_boundary)
        self._layout.connect(self._on_layout)
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
        pages attach half a second later. Returns an error text or None."""
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
        self.ids = ids
        self.pages, self._last = {}, {}
        for i, z in enumerate(ids):
            name = self.page_names[i] if i < len(self.page_names) else "empty"
            pg = make_page(name)
            pg.attach(*self.zones[z])
            self.pages[z] = pg
            self._send([0x20, z])
        self.log.emit(f"erae: {len(self.pages)} page{'s' if len(self.pages) != 1 else ''}: "
                      + ", ".join(f"N{i + 1} {pg.title} ({self.zones[z][0]}x{self.zones[z][1]})"
                                  for i, (z, pg) in enumerate(self.pages.items())))
        self._timer.start(100)
        self.redraw(force=True)

    def set_pages(self, names):
        """Another page catalogue on the N buttons, live."""
        self.page_names = list(names)
        if self.pages:
            for pg in self.pages.values():
                for ident, down in pg.release_all():
                    self.hold.emit(ident, down)
            self._attach()

    def close(self):
        """Every hold let go, the zones blank, API mode left, the ports closed."""
        self._timer.stop()
        for pg in self.pages.values():
            for ident, down in pg.release_all():
                self.hold.emit(ident, down)
        if self._out is not None:
            for z in self.pages:
                try:
                    self._send([0x20, z])
                except Exception:
                    pass
            try:
                self._send([0x02])
            except Exception:
                pass
        with self._lock:
            ports, self._in, self._out = (self._in, self._out), None, None
        for p in ports:
            if p is not None:
                try:
                    p.cancel_callback() if hasattr(p, "cancel_callback") else None
                    p.close_port()
                except Exception:
                    pass
        self.open_ok = False
        self.pages, self.zones, self._last, self.ids = {}, {}, {}, []

    # ------------------------------------------------------------ out
    def _send(self, body):
        with self._lock:
            if self._out is not None:
                self._out.send_message([0xF0] + ERAE2 + body + [0xF7])

    def realtime(self, byte):
        """A MIDI clock, Start, Stop or Continue byte from gacha_midi's
        sender thread, mirrored onto the device's port."""
        with self._lock:
            if self._out is not None:
                try:
                    self._out.send_message([byte])
                except Exception:
                    pass

    def _image(self, z, x, y, w, h, rgb):
        data = bitize7(rgb)
        self._send([0x23, z, x, y, w, h] + data + [xor(data)])
        self.messages += 1

    def active_zone(self):
        return self.ids[self.active] if self.ids and self.active < len(self.ids) else None

    def active_page(self):
        z = self.active_zone()
        return self.pages.get(z) if z is not None else None

    def last_frame(self, z=None):
        """The rows last sent to zone z (the page on screen by default)."""
        z = self.active_zone() if z is None else z
        return self._last.get(z)

    def redraw(self, force=False):
        """Paint the page on screen, only the rows that changed since the
        last time, in chunks of at most 32 pixels; the other pages every
        fifth tick, so a switch finds them close to current. Ten times a
        second from the timer; nothing is sent while nothing changes."""
        if not self.pages:
            return
        now = time.monotonic()
        self._ticks += 1
        az = self.active_zone()
        for z, pg in self.pages.items():
            if z != az and self._ticks % 5 and not force:
                continue
            try:
                px, asks = pg.frame(self.read, now)
            except Exception as e:               # noqa: BLE001 - a read failing must not kill the loop
                self.log.emit(f"erae: {pg.title}: {e}")
                continue
            for ev in asks:                      # a long press ripened
                self._emit(ev)
            last = self._last.get(z)
            for row, cells in enumerate(px):
                if not force and last is not None:
                    if last[row] == cells:
                        continue
                    diff = [i for i, (a, b) in enumerate(zip(last[row], cells)) if a != b]
                    lo, hi = diff[0], diff[-1] + 1     # the changed span only
                else:
                    lo, hi = 0, pg.w
                for x0 in range(lo, hi, PIXELS_PER_MSG):
                    chunk = cells[x0:min(hi, x0 + PIXELS_PER_MSG)]
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
            self.finger.emit(zone, action, fid & 0x7FFFFFFF, x, y, z)
            return
        kind = st & 0xF0
        ch = (st & 0x0F) + 1
        if kind == 0xB0 and len(msg) >= 3 and ch == LAYOUT_CH \
                and LAYOUT_CC <= msg[1] < LAYOUT_CC + 8:
            if msg[2] >= 64:                     # the layout coming up (0 = the one left)
                self._layout.emit(msg[1] - LAYOUT_CC)
            return
        if kind == 0x90 and len(msg) >= 3:
            self.midi.emit(self.label, "note", ch, msg[1], msg[2])
        elif kind == 0x80 and len(msg) >= 3:
            self.midi.emit(self.label, "note", ch, msg[1], 0)
        elif kind == 0xB0 and len(msg) >= 3:
            self.midi.emit(self.label, "cc", ch, msg[1], msg[2])
        elif kind == 0xC0 and len(msg) >= 2:
            self.midi.emit(self.label, "pc", ch, msg[1], 127)

    # ------------------------------------------------------------ in (GUI thread)
    def _on_boundary(self, z, w, h):
        self.zones[z] = (w, h)

    def _on_layout(self, k):
        if k != self.active:
            self.active = k
            self.page.emit(k)
            self.redraw(force=False)

    def _on_finger(self, zone, code, fid, x, y, z):
        pg = self.pages.get(zone)
        if pg is None:
            return
        if zone in self.ids and self.ids.index(zone) != self.active:
            self._on_layout(self.ids.index(zone))   # a touch tells the page too
        for ev in pg.touch(self.ACTIONS.get(code, "up"), fid, x, y, z, time.monotonic()):
            self._emit(ev)

    def _emit(self, ev):
        if ev[0] == "set":
            self.action.emit(ev[1], float(ev[2]))
        elif ev[0] == "press":
            self.action.emit(ev[1], None)
        else:
            self.hold.emit(ev[1], bool(ev[2]))
