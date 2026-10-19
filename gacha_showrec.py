#!/usr/bin/env python3
"""
gacha_showrec.py — a whole performance to disk, as it was heard.

One press starts it, the next stops it (Ctrl+R, the Perform tab's button,
a MIDI binding); every recording is a folder of its own under recordings/:

    recordings/2026-10-16_213045/
        master.wav        stereo, 24-bit: the output itself (after the racks,
                          the VST inserts, the gain, the limiter, the click)
        live.flac         the five buses, stereo 24-bit each, after their
        drums.flac        racks and their F1-F4 (a muted stem is silence),
        layers.flac       before the freeze, the inserts and the gain: the
        chops.flac        stems to mix again in a DAW, sample-aligned with
        events.flac       the master
        midi.mid          what left the MIDI out (notes, CCs), placed as heard
        meta.json         the start, the tempo, the sections as they landed

The buses go one file each because FLAC stops at 8 channels (libsndfile
refuses 10), and a DAW takes stereo files one per track anyway.

ShowRecorder lives in the audio engine: the audio thread hands it a
(12, block) array per block (push()), a writer thread of its own encodes
and writes, so the audio thread never waits for the disk; libsndfile runs
without the interpreter's lock (cffi lets it go), so the encoding does not
hold the audio thread up either. FLAC and WAV are integers: a bus louder
than full scale (the master limiter comes after it) is clipped in its file,
and counted (meta.json "clipped"). A plain WAV stops at 4 GB, about four
hours of 24-bit stereo at 48 kHz.

write_midi() and write_meta() are the GUI's side: the MIDI out's port is
there (gacha_midi.MidiOut.tap), the sections too.
"""
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BUSES = ("live", "drums", "layers", "chops", "events")   # rows 2.. of a pushed block, in pairs
ROWS = 2 + 2 * len(BUSES)                                # the master first
MIDI_KINDS = ("note_on", "note_off", "control_change", "program_change", "pitchwheel")
TPB = 480                                                # MIDI ticks per beat


def new_folder(base):
    """recordings/<date>_<time>, with _2, _3 ... if that one exists."""
    base = Path(base)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    p, n = base / stamp, 2
    while p.exists():
        p, n = base / f"{stamp}_{n}", n + 1
    p.mkdir(parents=True)
    return p


class ShowRecorder:
    """The master and the buses to files in `folder`, from the audio
    thread's blocks, written by a thread of its own."""

    BATCH_S = 0.25                    # the writer wakes this often

    def __init__(self, folder, sr):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.sr = int(sr)
        self.t0 = None                # perf_counter moment frame 0 is heard
        self.t0_mono = None           # monotonic moment frame 0 was rendered (the player's clock)
        self.frames = 0
        self.error = None
        self.clipped = {b: 0 for b in BUSES}
        self._q = queue.SimpleQueue()
        self._files = []
        try:
            self._files.append(sf.SoundFile(str(self.folder / "master.wav"), "w", self.sr, 2,
                                            subtype="PCM_24", format="WAV"))
            for b in BUSES:
                self._files.append(sf.SoundFile(str(self.folder / f"{b}.flac"), "w", self.sr, 2,
                                                subtype="PCM_24", format="FLAC"))
        except Exception:
            for f in self._files:
                f.close()
            raise
        self._thread = threading.Thread(target=self._run, name="gacha-showrec", daemon=True)
        self._thread.start()

    def push(self, rows, t_heard, now):
        """The audio thread: one block, (ROWS, n) float32, the master in
        rows 0-1 and BUSES after it; `t_heard` the perf_counter moment its
        first frame is heard, `now` the block's monotonic time."""
        if self.t0 is None:
            self.t0, self.t0_mono = t_heard, now
        self.frames += rows.shape[1]
        self._q.put(rows)

    def _run(self):
        done = False
        while not done:
            blocks = []
            try:
                b = self._q.get(timeout=self.BATCH_S)
                while True:
                    if b is None:
                        done = True
                        break
                    blocks.append(b)
                    b = self._q.get_nowait()
            except queue.Empty:
                pass
            if blocks and self.error is None:
                try:
                    self._write(np.concatenate(blocks, axis=1))
                except Exception as e:           # disk full, the drive gone: say so, stop writing
                    self.error = str(e)
            if not done:
                time.sleep(self.BATCH_S)         # a quarter second of audio per write
        for f in self._files:
            try:
                f.close()
            except Exception as e:
                self.error = self.error or str(e)

    def _write(self, data):
        for i, f in enumerate(self._files):
            ch = data[2 * i:2 * i + 2]
            if i:                                # a bus: over full scale is clipped, counted
                over = int(np.count_nonzero(np.abs(ch) > 1.0))
                if over:
                    self.clipped[BUSES[i - 1]] += over
                    ch = np.clip(ch, -1.0, 1.0)
            f.write(np.ascontiguousarray(ch.T))

    def state(self):
        """For the GUI's status line."""
        return {"s": self.frames / self.sr, "error": self.error,
                "clipped": sum(self.clipped.values())}

    def close(self):
        """Stop: what is queued is written, the files closed. Returns what
        the GUI needs for the MIDI file and meta.json."""
        self._q.put(None)
        self._thread.join(30.0)
        return {"folder": str(self.folder), "sr": self.sr, "frames": self.frames,
                "seconds": self.frames / self.sr, "t0": self.t0, "t0_mono": self.t0_mono,
                "clipped": dict(self.clipped), "error": self.error}


def write_midi(path, events, t0, lead_s, bpm, marks=()):
    """midi.mid out of the MIDI out's tap: `events` [(perf_counter sent,
    mido message)], placed at sent + lead (the moment the receiver was
    meant to sound it, the master's time line) minus `t0` (when the
    master's frame 0 is heard); what came before t0 is left out, a note
    still on at the end gets its off there. `marks` [(seconds, text)] go
    in as markers. One track, the tempo `bpm` (the grid only: every event
    sits at its time in seconds). Returns the count of messages written."""
    import mido
    tempo = mido.bpm2tempo(bpm if bpm and bpm > 0 else 120.0)
    tick = TPB * 1e6 / tempo                             # ticks per second

    items = []                                           # (tick, order, message)
    on = set()
    for t, msg in events:
        if msg.type not in MIDI_KINDS:
            continue
        s = t + lead_s - t0
        if s < 0:
            continue
        items.append((int(round(s * tick)), 1, msg))
    for s, text in marks:
        if s >= 0:
            items.append((int(round(s * tick)), 0, mido.MetaMessage("marker", text=str(text))))
    items.sort(key=lambda it: (it[0], it[1]))
    end = items[-1][0] if items else 0
    mf = mido.MidiFile(type=0, ticks_per_beat=TPB)
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    last, n = 0, 0
    for tk, _o, msg in items:
        tr.append(msg.copy(time=tk - last))
        last = tk
        if msg.type == "note_on" and msg.velocity > 0:
            on.add((msg.channel, msg.note))
            n += 1
        elif msg.type in ("note_on", "note_off"):
            on.discard((msg.channel, msg.note))
            n += 1
        elif not msg.is_meta:
            n += 1
    for ch, note in sorted(on):
        tr.append(mido.Message("note_off", channel=ch, note=note, velocity=0, time=end - last))
        last = end
    tr.append(mido.MetaMessage("end_of_track", time=0))
    mf.save(str(path))
    return n


def write_meta(path, info, **extra):
    meta = {"buses": list(BUSES), "files": ["master.wav"] + [f"{b}.flac" for b in BUSES],
            "sr": info.get("sr"), "seconds": round(info.get("seconds") or 0.0, 3),
            "clipped": info.get("clipped"), "error": info.get("error")}
    meta.update(extra)
    Path(path).write_text(json.dumps(meta, indent=1))
