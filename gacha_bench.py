#!/usr/bin/env python3
"""
gacha_bench.py — what one audio block costs, offline, state by state.

Runs LiveAudio._dsp_block (no device) at a block size: nothing playing,
then a section with and without racks, varispeed, the stretch, a repeat,
the freeze, the fragments and a break, and prints the mean, the 99th
percentile and the worst block against the block's own length. --prof
adds a profile of the heavier states. (The DSP is a tenth of a 128-frame
block; what crackled was the GUI holding the interpreter lock, settled
by the engine in its own process, gacha_audioproc.py.)

    python gacha_bench.py 128
"""
import cProfile
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gacha_audio as ga                              # noqa: E402
from gacha_fx import STEM_CHANNELS, VOICES            # noqa: E402
from gacha_freeze import FREEZE                       # noqa: E402

SR = 48000
BS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 128
BLOCK_US = 1e6 * BS / SR
rng = np.random.default_rng(0)


def make_la(block=BS):
    la = ga.LiveAudio(None, "x", block=block, sr=SR)
    la._dsp_init()
    return la


def section(la, bpm=120.0, bars=8, hits=True):
    """Eight bars of noise on every stem and voice, a kick and a hat per beat."""
    n = int(round(60.0 / bpm * SR)) * 4 * bars
    stems = {k: (rng.standard_normal((2, n)) * 0.05).astype(np.float32)
             for k in STEM_CHANNELS + VOICES}
    events = []
    if hits:
        beat = 60.0 / bpm
        for b in range(4 * bars):
            events.append([b * beat, "", 0, 0.1, 1.0, "kick", 1.0])
            events.append([b * beat + beat / 2, "", 0, 0.1, 1.0, "hihat", 0.7])
    la.player.queue(stems, events, bpm, bars)


def run(la, n=3000, label="", prof=False):
    x = (rng.standard_normal((2, la.block)) * 0.1).astype(np.float32)
    now = time.monotonic()
    for _ in range(50):
        la._dsp_block(x, now)
    ts = np.zeros(n)
    pr = cProfile.Profile() if prof else None
    if pr:
        pr.enable()
    for i in range(n):
        t0 = time.perf_counter()
        la._dsp_block(x, now + i * la.block / SR)
        ts[i] = time.perf_counter() - t0
    if pr:
        pr.disable()
    us = ts * 1e6
    print(f"{label:34s} mean {us.mean():7.0f} us  p99 {np.percentile(us, 99):7.0f}  "
          f"max {us.max():7.0f}   ({100 * us.mean() / BLOCK_US:4.0f}% of {BLOCK_US:.0f} us)")
    if pr:
        pstats.Stats(pr).sort_stats("tottime").print_stats(14)
    return us


def main():
    prof = "--prof" in sys.argv
    print(f"block {BS} frames = {BLOCK_US:.0f} us")
    la = make_la()
    run(la, label="idle, nothing playing", prof=prof)

    la = make_la()
    la.racks["live"].amounts.update(reverb=0.3, delay=0.3, lowpass=0.2)
    run(la, label="idle + 3 fx on live")

    la = make_la()
    section(la)
    run(la, label="section playing", prof=prof)

    la = make_la()
    section(la)
    for k in ("drums", "layers", "chops", "events"):
        la.racks[k].amounts.update(reverb=0.3, delay=0.3)
    la.racks["kick"].amounts.update(drive=0.4)
    run(la, label="section + fx on 4 stems + kick")

    la = make_la()
    section(la)
    la.player.rate = 1.05
    run(la, label="section, rate 1.05 varispeed")

    la = make_la()
    section(la)
    la.player.locked = True
    la.player.rate = 1.05
    run(la, label="section, rate 1.05 locked (WSOLA)", prof=prof)

    la = make_la()
    section(la)
    la.player.set_repeat(1 / 16)
    run(la, label="section + repeat 1/16")

    la = make_la()
    section(la)
    la.freeze.held = True
    run(la, label="section + freeze held", prof=prof)

    la = make_la()
    section(la)
    la.freeze.held = True
    FREEZE["smear"] = 0.7
    run(la, label="section + freeze held, smear .7")
    FREEZE["smear"] = 0.0

    la = make_la()
    section(la)
    la.player.frag.press(1)
    run(la, label="section + fragments 1 voice", prof=prof)

    la = make_la()
    section(la)
    for k in (1, 2, 3):
        la.player.frag.press(k)
    run(la, label="section + fragments 3 voices")

    la = make_la()
    section(la)
    la.player.fill("glitch")
    run(la, n=1500, label="section + glitch break")



if __name__ == "__main__":
    main()
