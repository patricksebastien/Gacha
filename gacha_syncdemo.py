#!/usr/bin/env python3
"""
gacha_syncdemo.py — a test clip for the picture-follows-sound mode.

The soundtrack is a slow melody of sustained, softly plucked notes. On every
note change the picture changes with it: a big number counts the notes, the
note name sits under it, and the background takes the note's colour (hue by
pitch class, the same mapping the GUI's tint uses). A bar along the bottom
fills over each note and a frame counter runs top left, so speed, direction
and drift are visible too. When Gacha builds a song out of this clip's audio
and the picture follows, the number on screen is the note you hear.

Written straight to the fast-seeking spec of gacha_transcode.py: 1080p,
30 fps, MJPEG .mov (every frame a keyframe), 44.1 kHz stereo audio.

    python3 gacha_syncdemo.py                      # videos/sync_demo.mov
    python3 gacha_syncdemo.py --seed 3 --notes 48  # another melody
    python3 gacha_syncdemo.py --out videos/demo2.mov --height 720
"""
import argparse
import colorsys
import random
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SR = 44100
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# D dorian pentatonic over two octaves: MIDI numbers
SCALE = [50, 52, 55, 57, 60, 62, 64, 67, 69, 72]
# a monospace face from the system (the app ships no fonts); Pillow's own
# bitmap font is the last resort
FONT_CANDIDATES = [
    "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "UbuntuMono-R.ttf",
    "NotoSansMono-Regular.ttf", "FreeMono.ttf",                    # Linux
    "consola.ttf", "cour.ttf",                                     # Windows
    "Menlo.ttc", "Monaco.ttf", "Courier New.ttf",                  # macOS
]
FONT_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts"),
             str(Path.home() / ".local/share/fonts"), "C:/Windows/Fonts",
             "/System/Library/Fonts", "/Library/Fonts"]


def font(size):
    """A monospace font at `size` pixels: the first candidate found in the
    usual font folders, else Pillow's default face."""
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)          # on the font path
        except OSError:
            pass
        for d in FONT_DIRS:
            hits = list(Path(d).rglob(name)) if Path(d).is_dir() else []
            if hits:
                try:
                    return ImageFont.truetype(str(hits[0]), size)
                except OSError:
                    pass
    try:
        return ImageFont.load_default(size=size)           # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def compose(rng, n):
    """A random walk on the scale with a melodic shape: mostly steps, a
    leap now and then, a pull back to the middle when it strays, and a
    rhythm of 1, 2 and 3 second notes. Returns [(midi, seconds)]."""
    i = len(SCALE) // 2
    out = []
    for k in range(n):
        if k == n - 1:
            i = 0 if rng.random() < 0.5 else len(SCALE) // 2   # home at the end
        else:
            step = rng.choice([-2, -1, -1, 1, 1, 2] if rng.random() < 0.85
                              else [-4, -3, 3, 4])
            i = min(len(SCALE) - 1, max(0, i + step))
            if rng.random() < 0.3:            # gravity towards the middle
                i += (1 if i < len(SCALE) // 2 else -1) if i != len(SCALE) // 2 else 0
        dur = rng.choice([1.0, 1.0, 2.0, 2.0, 2.0, 3.0]) if k < n - 1 else 3.0
        out.append((SCALE[i], dur))
    return out


def tone(midi, sec):
    """One note: a two-voice detuned pad (sine plus soft harmonics, slow
    vibrato, 20 ms attack, 60 ms release) and a brighter pluck on the
    attack so an onset detector has something to grab."""
    f = 440.0 * 2 ** ((midi - 69) / 12)
    n = int(sec * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.002 * np.sin(2 * np.pi * 5.0 * t)
    pad = np.zeros(n)
    for detune in (0.997, 1.003):
        ph = 2 * np.pi * f * detune * np.cumsum(vib) / SR
        for h, a in ((1, 1.0), (2, 0.35), (3, 0.18), (4, 0.08), (5, 0.04)):
            pad += a * np.sin(h * ph)
    pad *= 0.5 / 1.65
    pluck = np.zeros(n)
    for h, a in ((1, 0.8), (2, 0.6), (3, 0.5), (4, 0.35), (6, 0.2), (8, 0.1)):
        pluck += a * np.sin(2 * np.pi * f * h * t) * np.exp(-t * (6 + 1.2 * h))
    pluck *= 0.35 / 2.55
    env = np.ones(n)
    a, r = int(0.02 * SR), int(0.06 * SR)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    mono = (pad + pluck) * env
    # a little width: the second voice a hair later on the right
    left = mono
    right = np.concatenate([np.zeros(60), mono[:-60]])
    return np.stack([left, right], axis=1)


def write_wav(path, audio):
    pcm = np.clip(audio * 32767, -32767, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def note_colour(midi):
    hue = (midi % 12) / 12
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.55)
    return (int(r * 255), int(g * 255), int(b * 255))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="videos/sync_demo.mov")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--notes", type=int, default=40)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=4, help="mjpeg -q:v, 2 best")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    melody = compose(rng, args.notes)
    total = sum(d for _, d in melody)
    H = args.height
    W = H * 16 // 9
    fps = args.fps

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav.tmp")
    audio = np.concatenate([tone(m, d) for m, d in melody])
    write_wav(wav, audio)

    big = font(int(H * 0.55))
    mid = font(int(H * 0.11))
    small = font(int(H * 0.045))

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-i", str(wav),
           "-c:v", "mjpeg", "-q:v", str(args.quality), "-pix_fmt", "yuvj420p",
           "-c:a", "pcm_s16le", "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_frames = int(round(total * fps))
    starts = np.cumsum([0.0] + [d for _, d in melody])
    frame_i = 0
    print(f"{len(melody)} notes, {total:.0f} s, {n_frames} frames -> {out}")
    for k, (midi, dur) in enumerate(melody):
        base = Image.new("RGB", (W, H), note_colour(midi))
        d = ImageDraw.Draw(base)
        num = str(k + 1)
        name = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
        bw, bh = d.textbbox((0, 0), num, font=big)[2:]
        d.text(((W - bw) / 2, H * 0.42 - bh / 2 - H * 0.04), num,
               font=big, fill=(255, 255, 255))
        mw = d.textbbox((0, 0), name, font=mid)[2]
        d.text(((W - mw) / 2, H * 0.72), name, font=mid, fill=(255, 255, 255))
        d.text((H * 0.03, H * 0.03), f"note {k + 1}/{len(melody)}   {dur:.0f} s",
               font=small, fill=(255, 255, 255))
        end = int(round(starts[k + 1] * fps))
        while frame_i < end:
            fr = base.copy()
            d = ImageDraw.Draw(fr)
            t = frame_i / fps
            prog = (t - starts[k]) / dur
            bar_h = int(H * 0.035)
            d.rectangle([0, H - bar_h, int(W * prog), H], fill=(255, 255, 255))
            d.text((W - H * 0.03, H * 0.03), f"f {frame_i:05d}  {t:8.3f} s",
                   font=small, fill=(255, 255, 255), anchor="ra")
            proc.stdin.write(fr.tobytes())
            frame_i += 1
    proc.stdin.close()
    rc = proc.wait()
    wav.unlink(missing_ok=True)
    if rc:
        sys.exit(f"ffmpeg failed ({rc})")
    print(f"done: {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
