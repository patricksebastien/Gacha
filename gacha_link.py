#!/usr/bin/env python3
"""Ableton Link for the live clock.

Link is the tempo and beat grid shared by Ableton Live, Traktor, Reason,
the Elektron boxes and a few hundred other apps on the same network: every
peer sees the same tempo and the same bar phase. This module wraps it with
``aalink`` (Ableton's Link library with Python bindings and prebuilt
wheels: ``python -m pip install aalink``) into a small, polled object;
``gacha.py`` reads it from a timer and steers its TapTempo clock with what
it finds, and pushes taps, the bpm box and the section engine's tempo back
out, so the app is a peer like any other.

aalink is written for asyncio: its callbacks are posted to an event loop.
The app runs Qt's loop instead, so the Link object is built inside a
throwaway asyncio loop that is never run again and the state is read on
demand (tempo, beat, phase are cheap atomic reads on Link's side) rather
than pushed through callbacks.
"""

import asyncio
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import aalink
    LINK_AVAILABLE = True
except ImportError:                     # the app runs without Link
    aalink = None
    LINK_AVAILABLE = False

INSTALL_HINT = "python -m pip install aalink"


class LinkClock:
    """A Link peer, polled. ``enabled`` joins / leaves the network session;
    while enabled, ``snapshot()`` gives the session's tempo and where in the
    bar it is right now, and ``tempo`` (settable) pushes a change to every
    peer. Only the tempo is used so far: the bar phase is read but the app
    keeps its own bar, and start/stop is not wired."""

    QUANTUM = 4.0                       # beats per bar: the bar phase we follow

    def __init__(self, bpm=120.0):
        if not LINK_AVAILABLE:
            raise RuntimeError(f"aalink missing: {INSTALL_HINT}")
        # aalink asks for a running asyncio loop at construction (for the
        # callbacks, which this class does not use); give it one and put it
        # away
        self._loop = asyncio.new_event_loop()

        async def make():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return aalink.Link(float(bpm))
        self._link = self._loop.run_until_complete(make())
        self._link.quantum = self.QUANTUM

    # ------------------------------------------------------------ session
    @property
    def enabled(self):
        return bool(self._link.enabled)

    @enabled.setter
    def enabled(self, on):
        self._link.enabled = bool(on)

    @property
    def peers(self):
        """Other Link apps in the session (0 when alone or disabled)."""
        return int(self._link.num_peers)

    # ------------------------------------------------------------ tempo
    @property
    def tempo(self):
        return float(self._link.tempo)

    @tempo.setter
    def tempo(self, bpm):
        self._link.tempo = float(min(999.0, max(20.0, bpm)))     # Link's range

    def snapshot(self):
        """(bpm, beat, phase) at this instant: the session tempo, the running
        beat count and the position inside the bar in beats, 0 <= phase < 4."""
        lk = self._link
        return float(lk.tempo), float(lk.beat), float(lk.phase)

    def close(self):
        try:
            self._link.enabled = False
        except Exception:
            pass
        try:
            self._loop.close()
        except Exception:
            pass
