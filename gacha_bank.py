#!/usr/bin/env python3
"""
gacha_bank.py — cut a sample set out of a folder of long recordings.

The engine loads every selected sample into RAM as 44.1 kHz float32
stereo (about 350 KB per second), so hours-long session recordings
cannot be used as they are. This tool picks N files at random from a
folder and saves one random excerpt of each into samples/<name>/, where
the GUI sees it as a new set. Excerpts are 24-bit wav, cut at zero
crossings with a short fade, and windows that are near silence are
re-rolled a few times before the loudest attempt is kept. Each excerpt
is then normalized, --normalize picks how:

    lufs   (default) -14 LUFS integrated (ITU-R BS.1770, gated), peaks
           clipped at full scale: the treatment Reaper's "normalize to
           LUFS-I" gave the default set, so sets audition at one level.
           --lufs changes the target.
    peak   scale so the loudest sample sits at --peak dBFS (default -0.1).
           No clipping, but quiet-with-spikes material stays quiet. This
           is what the engine does on load anyway.
    none   keep the source level.

    python3 gacha_bank.py puredata /path/to/recordings
    python3 gacha_bank.py puredata /path/to/recordings --count 75 --seed 11
    python3 gacha_bank.py field ~/rec --min-len 8 --max-len 60 --overwrite
    python3 gacha_bank.py field ~/rec --normalize peak --peak -1

Output names keep the source file's stem plus the start offset, so a
sample can be traced back: 2019-07-06T23.17.57_1832s.wav
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

SAMPLES_DIR = Path(__file__).parent / "samples"
AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".mp3", ".w64", ".ogg")
SILENCE_DB = -45.0          # a window quieter than this is re-rolled
FADE_SEC = 0.01
TARGET_LUFS = -14.0         # what the default set was rendered at
TARGET_PEAK_DB = -0.1
NORMALIZE_MODES = ("lufs", "peak", "none")


def db(x):
    r = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return 20 * np.log10(r) if r > 0 else -120.0


def _biquad(fs, fc, q, gain_db, shelf):
    """RBJ biquad coefficients (b, a) as used by the BS.1770 K-weighting."""
    a_ = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    if shelf:
        b = [a_ * ((a_ + 1) + (a_ - 1) * c + 2 * np.sqrt(a_) * alpha),
             -2 * a_ * ((a_ - 1) + (a_ + 1) * c),
             a_ * ((a_ + 1) + (a_ - 1) * c - 2 * np.sqrt(a_) * alpha)]
        a = [(a_ + 1) - (a_ - 1) * c + 2 * np.sqrt(a_) * alpha,
             2 * ((a_ - 1) - (a_ + 1) * c),
             (a_ + 1) - (a_ - 1) * c - 2 * np.sqrt(a_) * alpha]
    else:
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def integrated_lufs(data, sr):
    """ITU-R BS.1770-4 integrated loudness: K-weighting, 400 ms blocks
    with 75% overlap, -70 LUFS absolute and -10 LU relative gating."""
    x = data.astype("float64")
    for fc, q, g, shelf in ((1681.974450955533, 0.7071752369554196, 4.0, True),
                            (38.13547087602444, 0.5003270373238773, 0.0, False)):
        b, a = _biquad(sr, fc, q, g, shelf)
        x = lfilter(b, a, x, axis=0)
    block, hop = int(0.4 * sr), int(0.1 * sr)
    if len(x) < block:
        z = np.mean(np.square(x), axis=0).sum()
        return -0.691 + 10 * np.log10(z) if z > 0 else -120.0
    n = 1 + (len(x) - block) // hop
    idx = np.arange(block)[None, :] + hop * np.arange(n)[:, None]
    z = np.square(x)[idx].mean(axis=1).sum(axis=1)          # per block
    with np.errstate(divide="ignore"):
        l = -0.691 + 10 * np.log10(z)
    keep = l > -70
    if not keep.any():
        return -120.0
    gamma = -0.691 + 10 * np.log10(z[keep].mean()) - 10
    keep &= l > gamma
    if not keep.any():
        return -120.0
    return -0.691 + 10 * np.log10(z[keep].mean())


def normalize_lufs(data, sr, target):
    """Gain to the target loudness, peaks hard-clipped at full scale.
    Returns (audio, gain_db, clipped fraction)."""
    cur = integrated_lufs(data, sr)
    gain_db = target - cur
    out = data * (10 ** (gain_db / 20))
    clipped = float(np.mean(np.abs(out) > 1.0))
    return np.clip(out, -1.0, 1.0), gain_db, clipped


def normalize_peak(data, target_db):
    """Scale so the loudest sample sits at target_db dBFS."""
    peak = float(np.abs(data).max())
    if peak <= 0:
        return data, 0.0
    gain_db = target_db - 20 * np.log10(peak)
    return data * (10 ** (gain_db / 20)), gain_db


def normalize(data, sr, mode, lufs, peak_db):
    """Dispatch on --normalize. Returns (audio, note for the log)."""
    if mode == "lufs":
        data, gain, clipped = normalize_lufs(data, sr, lufs)
        return data, f"{gain:+5.1f} dB" + (f" clip {clipped:.2%}" if clipped else "")
    if mode == "peak":
        data, gain = normalize_peak(data, peak_db)
        return data, f"{gain:+5.1f} dB"
    return data, ""


def excerpt(rng, path, min_len, max_len, tries=8):
    """One random window of `path` as (audio, sr, start_sec). Whole file
    when it is shorter than min_len. Returns None if unreadable or empty."""
    try:
        info = sf.info(str(path))
    except Exception as e:
        print(f"  ! skipping {path.name}: {e}")
        return None
    sr, total = info.samplerate, info.frames
    if total < sr:                              # under a second: not worth it
        return None
    if total <= min_len * sr:
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
        return data, sr, 0.0
    length = int(rng.uniform(min_len, min(max_len, total / sr)) * sr)
    best = None
    for _ in range(tries):
        start = rng.randrange(0, total - length + 1)
        data, _ = sf.read(str(path), start=start, frames=length,
                          dtype="float32", always_2d=True)
        level = db(data)
        if best is None or level > best[0]:
            best = (level, data, start)
        if level > SILENCE_DB:
            break
    level, data, start = best
    if level <= -90:                            # digital silence everywhere
        return None
    return data, sr, start / sr


def trim_and_fade(data, sr):
    """Snap the edges to the nearest zero crossings, then a short fade."""
    mono = data.mean(axis=1)
    zc = np.where(np.diff(np.signbit(mono)))[0]
    if len(zc) > 2:
        data = data[zc[0]:zc[-1] + 1]
    n = min(int(FADE_SEC * sr), len(data) // 4)
    if n > 0:
        data[:n] *= np.linspace(0, 1, n)[:, None]
        data[-n:] *= np.linspace(1, 0, n)[:, None]
    return data


def safe_stem(path):
    return "".join(c if c.isalnum() or c in "-_." else "." for c in path.stem)


def build(name, source, count, min_len, max_len, seed, overwrite,
          mode="lufs", lufs=TARGET_LUFS, peak_db=TARGET_PEAK_DB):
    source = Path(source).expanduser()
    files = sorted(p for p in source.rglob("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f"no audio files under {source}")
    out_dir = SAMPLES_DIR / name
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        sys.exit(f"{out_dir} already has files; pass --overwrite to add to it")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(files)
    how = {"lufs": f"{lufs:g} LUFS", "peak": f"peak {peak_db:g} dBFS",
           "none": "source level"}[mode]
    print(f"{len(files)} audio files in {source}; cutting {count} excerpts "
          f"of {min_len:g}-{max_len:g}s into {out_dir} (seed {seed}, {how})")
    written, total_sec, total_bytes = 0, 0.0, 0
    for path in files:
        if written >= count:
            break
        res = excerpt(rng, path, min_len, max_len)
        if res is None:
            continue
        data, sr, start = res
        data = trim_and_fade(data, sr)
        data, note = normalize(data, sr, mode, lufs, peak_db)
        out = out_dir / f"{safe_stem(path)}_{int(start)}s.wav"
        sf.write(out, data, sr, subtype="PCM_24")
        written += 1
        total_sec += len(data) / sr
        total_bytes += out.stat().st_size
        print(f"  {written:3d}/{count}  {out.name:<44} {len(data)/sr:5.1f}s  "
              f"{note:<18} <- {path.name}")
    print(f"\n✔ {written} samples, {total_sec/60:.1f} min of audio, "
          f"{total_bytes/1e6:.0f} MB on disk. Refresh the Samples tab in the "
          f"GUI and tick '{name}'.")
    if written < count:
        print(f"  (only {written} usable files found)")


def main():
    if hasattr(sys.stdout, "reconfigure"):      # Windows consoles: no crash on ✔
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(
        description="Cut a gacha sample set out of a folder of recordings.")
    ap.add_argument("name", help="set name, becomes samples/<name>/")
    ap.add_argument("source", help="folder to draw from (searched recursively)")
    ap.add_argument("--count", type=int, default=75, help="excerpts to cut")
    ap.add_argument("--min-len", type=float, default=4.0, metavar="S")
    ap.add_argument("--max-len", type=float, default=30.0, metavar="S")
    ap.add_argument("--seed", type=int, default=None,
                    help="reproducible pick of files and windows")
    ap.add_argument("--overwrite", action="store_true",
                    help="add to a set folder that already has files")
    ap.add_argument("--normalize", choices=NORMALIZE_MODES, default="lufs",
                    help="lufs: integrated loudness with clipped peaks, like "
                         "the default set (default); peak: loudest sample to "
                         "--peak dBFS; none: keep the source level")
    ap.add_argument("--lufs", type=float, default=TARGET_LUFS, metavar="L",
                    help=f"target for --normalize lufs (default {TARGET_LUFS:g})")
    ap.add_argument("--peak", type=float, default=TARGET_PEAK_DB, metavar="DB",
                    help=f"target for --normalize peak (default {TARGET_PEAK_DB:g})")
    a = ap.parse_args()
    if a.min_len <= 0 or a.max_len < a.min_len:
        sys.exit("need 0 < --min-len <= --max-len")
    seed = a.seed if a.seed is not None else random.randrange(10 ** 6)
    build(a.name, a.source, max(1, a.count), a.min_len, a.max_len, seed,
          a.overwrite, a.normalize, a.lufs, a.peak)


if __name__ == "__main__":
    main()
