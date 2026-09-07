#!/usr/bin/env python3
"""
gacha_transcode.py — bring every backdrop video in videos/ to a spec the
GUI can seek instantly, using the ffmpeg and ffprobe installed on the
system (no Python video package).

The backdrop jumps to a new spot at every section change and swaps clips
on drops, so a video must seek fast. Phone and drone clips are 4K H.264 or
HEVC at 60 fps with an audio track: heavy to decode, and a seek stalls the
picture for 100-300 ms right on the downbeat. The default target is
1080p, 30 fps, MJPEG in a .mov, where every frame is a keyframe and a
seek lands in about 15 ms. The audio track is kept as it is (copied,
not re-encoded): the app can build songs out of a video's own soundtrack.
--no-audio drops it.

Every video under videos/ (its set subfolders included) is probed. A file
already at spec is left alone. Anything else is re-encoded next to itself
and the original moves to videos/originals/, mirroring the set folder it
came from, so the Videos tab keeps working and nothing is lost. The stock
videos/gacha.mp4 and anything already in originals/ are skipped.

    python3 gacha_transcode.py                    # report + convert
    python3 gacha_transcode.py --dry-run          # only say what would change
    python3 gacha_transcode.py --height 720 --fps 24 --quality 4
    python3 gacha_transcode.py --codec h264 --quality 20 --gop 15
    python3 gacha_transcode.py --force            # redo files already at spec
    python3 gacha_transcode.py --no-audio         # silent backdrops only
    python3 gacha_transcode.py videos/field/a.MP4 # just this one
"""

import argparse
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

VIDEOS_DIR = Path(__file__).parent / "videos"
ORIGINALS = VIDEOS_DIR / "originals"
STOCK = VIDEOS_DIR / "gacha.mp4"
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mts", ".m2ts")

# codec name -> (ffmpeg encoder, container extension, quality flag, default)
CODECS = {
    "mjpeg": ("mjpeg", ".mov", "-q:v", 6),        # 2 (best) .. 31
    "h264": ("libx264", ".mp4", "-crf", 18),      # 0 (lossless) .. 51
}


def need(tool):
    if shutil.which(tool) is None:
        sys.exit(f"{tool} not found: install ffmpeg (it ships ffprobe too)")


def probe(path):
    """Video stream facts + whether an audio stream exists, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    streams = json.loads(out).get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return None
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {"codec": video.get("codec_name", "?"),
            "width": int(video.get("width", 0)),
            "height": int(video.get("height", 0)),
            "fps": fps,
            "audio": any(s.get("codec_type") == "audio" for s in streams)}


def at_spec(info, codec, height, fps, no_audio):
    return (info["codec"] == codec and info["height"] <= height
            and info["fps"] <= fps + 0.05 and not (no_audio and info["audio"]))


def encode_cmd(src, dst, codec, height, fps, quality, gop, scale_up, no_audio):
    encoder, _, qflag, _ = CODECS[codec]
    # scale only downwards unless asked; keep aspect, even dimensions
    vf = [f"scale=-2:'min({height},ih)'" if not scale_up else f"scale=-2:{height}",
          f"fps={fps:g}"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y",
           "-i", str(src), "-map", "0:v:0", "-vf", ",".join(vf),
           "-c:v", encoder, qflag, str(quality)]
    cmd += ["-an"] if no_audio else ["-map", "0:a?", "-c:a", "copy"]
    if codec == "h264":
        cmd += ["-preset", "fast", "-pix_fmt", "yuv420p",
                "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
                "-movflags", "+faststart"]
    cmd.append(str(dst))
    return cmd


def remux_cmd(src, dst):
    """Same picture, audio dropped: no re-encode."""
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src), "-an", "-c:v", "copy", str(dst)]


def candidates(only=()):
    """Every video under videos/ except the stock clip and originals/, or
    just the given files when some are named."""
    if only:
        for p in only:
            p = Path(p).resolve()
            if VIDEOS_DIR.resolve() not in p.parents:
                sys.exit(f"{p} is not under {VIDEOS_DIR}")
            yield p
        return
    for p in sorted(VIDEOS_DIR.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        if p == STOCK or ORIGINALS in p.parents:
            continue
        yield p


def describe(info):
    return (f"{info['codec']} {info['width']}x{info['height']} "
            f"{info['fps']:.3g}fps{' +audio' if info['audio'] else ''}")


def main():
    if hasattr(sys.stdout, "reconfigure"):      # Windows consoles: no crash on ✔
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(
        description="Re-encode backdrop videos for instant seeking; the audio "
                    "track is kept and originals move to videos/originals/.")
    ap.add_argument("--codec", choices=CODECS, default="mjpeg",
                    help="mjpeg: every frame a keyframe, fastest seeks, big "
                         "files (default); h264: small files, seeks limited "
                         "by --gop")
    ap.add_argument("--height", type=int, default=1080,
                    help="target height in pixels, width follows (default 1080)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="target frame rate (default 30)")
    ap.add_argument("--quality", type=float, default=None,
                    help="mjpeg -q:v 2..31 (default 6) or h264 -crf 0..51 "
                         "(default 18); lower is better")
    ap.add_argument("--gop", type=int, default=15,
                    help="h264 only: keyframe every N frames (default 15, "
                         "half a second at 30 fps)")
    ap.add_argument("--scale-up", action="store_true",
                    help="also upscale clips smaller than --height")
    ap.add_argument("--force", action="store_true",
                    help="re-encode files that already meet the spec")
    ap.add_argument("--no-audio", action="store_true",
                    help="drop the audio track (default: copy it as it is)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, change nothing")
    ap.add_argument("files", nargs="*",
                    help="only these videos (default: everything under videos/)")
    a = ap.parse_args()
    quality = a.quality if a.quality is not None else CODECS[a.codec][3]
    ext = CODECS[a.codec][1]

    need("ffmpeg")
    need("ffprobe")
    if not VIDEOS_DIR.is_dir():
        sys.exit(f"no {VIDEOS_DIR}")

    todo, fine, bad = [], 0, 0
    for src in candidates(a.files):
        info = probe(src)
        if info is None:
            print(f"  ? {src.relative_to(VIDEOS_DIR)}: not a readable video")
            bad += 1
            continue
        if at_spec(info, a.codec, a.height, a.fps, a.no_audio) and not a.force:
            fine += 1
            continue
        # a file at spec except for an unwanted audio track only needs a remux
        remux = (info["codec"] == a.codec and info["height"] <= a.height
                 and info["fps"] <= a.fps + 0.05 and not a.force)
        todo.append((src, info, remux))

    print(f"{fine} video(s) already at spec ({a.codec}, <= {a.height}p, "
          f"<= {a.fps:g} fps{', no audio' if a.no_audio else ''}); "
          f"{len(todo)} to convert"
          + (f"; {bad} unreadable" if bad else ""))
    if not todo:
        return

    for src, info, remux in todo:
        rel = src.relative_to(VIDEOS_DIR)
        dst = src.with_suffix(ext)
        keep = ORIGINALS / rel
        action = "remux (drop audio)" if remux else "encode"
        print(f"\n  {rel}  [{describe(info)}]  ->  {dst.name}  {action}")
        if a.dry_run:
            continue
        if keep.exists():
            print(f"    ! {keep.relative_to(VIDEOS_DIR)} exists already, skipping")
            continue
        tmp = dst.with_name(dst.stem + ".transcoding" + dst.suffix)
        cmd = remux_cmd(src, tmp) if remux else encode_cmd(
            src, tmp, a.codec, a.height, a.fps, quality, a.gop, a.scale_up,
            a.no_audio)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            tmp.unlink(missing_ok=True)
            print(f"    ✗ ffmpeg failed ({e.returncode}), original untouched")
            continue
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(keep))          # original parked first...
        tmp.replace(dst)                          # ...then the result takes its name
        before, after = keep.stat().st_size / 1e6, dst.stat().st_size / 1e6
        out = probe(dst)
        print(f"    ✔ {describe(out) if out else '?'}  {before:.0f} MB -> {after:.0f} MB"
              f"   original in {keep.relative_to(VIDEOS_DIR).parent}/")
    if not a.dry_run:
        print("\nRefresh the Videos tab in the GUI to pick up the new files.")


if __name__ == "__main__":
    main()
