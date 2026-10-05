"""
gacha_fetch.py — bring a clip from the web into videos/ with yt-dlp.

The Videos tab has a URL field: paste a YouTube, Vimeo or any other page
yt-dlp knows and press Fetch. This runs yt-dlp as a subprocess (never
imported: when a site changes and breaks it, `pip install -U yt-dlp` is
the whole fix and the app is not touched), downloads the best picture
up to 1080p as an .mp4 into videos/web/, then hands the file to
gacha_transcode.py so it lands as the fast-seeking MJPEG .mov the
backdrop wants, the original parked in videos/originals/web/. Both steps
run on a worker thread and stream their lines to the log; nothing of it
touches the audio or video threads of the perform screen.

yt-dlp needs an ffmpeg to merge the picture and the sound YouTube serves
as two files; it gets the one the app found (the system's, or the static
build of imageio-ffmpeg). The transcode wants ffprobe, which only a
system ffmpeg brings: without it the clip is kept as downloaded, which
plays fine at 1080p and merely seeks slower.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from gacha_engine import NO_WINDOW, ffmpeg_exe

BASE = Path(__file__).parent
VIDEOS_DIR = BASE / "videos"
WEB_DIR = VIDEOS_DIR / "web"                 # the set folder fetched clips form
TRANSCODE = BASE / "gacha_transcode.py"
INSTALL_HINT = "pip install yt-dlp"
MAX_HEIGHT = 1080

# H.264 in an mp4 first (PyAV decodes it fastest and the transcoder reads
# anything), then any mp4 up to 1080p, then whatever is best
FORMAT = (f"bv*[height<={MAX_HEIGHT}][vcodec^=avc1]+ba[ext=m4a]"
          f"/bv*[height<={MAX_HEIGHT}][ext=mp4]+ba"
          f"/b[height<={MAX_HEIGHT}]/b")

_PROGRESS = re.compile(r"^\[download\]\s+(\d+(?:\.\d+)?)%")


def yt_dlp_cmd():
    """How to run yt-dlp: the module in this interpreter (what
    requirements.txt installs), else the yt-dlp on PATH, else None."""
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        pass
    exe = shutil.which("yt-dlp")
    return [exe] if exe else None


def fetch_available():
    return yt_dlp_cmd() is not None


def fetch_cmd(url, folder=WEB_DIR):
    """The yt-dlp call: one video (never the playlist it sits in), best
    picture up to 1080p merged into an mp4, filenames Windows-safe, the
    progress as lines, and the final path printed last."""
    cmd = yt_dlp_cmd() + [
        "--no-playlist", "--newline", "--no-quiet", "--progress",
        "--windows-filenames", "--trim-filenames", "80",
        "-f", FORMAT, "--merge-output-format", "mp4",
        "--ffmpeg-location", ffmpeg_exe(),
        "-P", str(folder), "-o", "%(title)s [%(id)s].%(ext)s",
        "--print", "after_move:filepath",
        "--", url]
    return cmd


def looks_like_url(text):
    return bool(re.match(r"^https?://\S+$", text.strip()))


class FetchWorker(QThread):
    """Downloads one URL, then transcodes the file it got. Signals: `line`
    for the log, `progress(percent)` while downloading (-1 once past it),
    `done(ok, path)` with the file now in videos/ (or "" on failure)."""
    line = Signal(str)
    progress = Signal(float)
    done = Signal(bool, str)

    def __init__(self, url, folder=WEB_DIR, transcode=True):
        super().__init__()
        self.url = url.strip()
        self.folder = Path(folder)
        self.transcode = transcode
        self._proc = None

    def run(self):
        try:
            path = self._download()
            if path is None:
                self.done.emit(False, "")
                return
            self.progress.emit(-1)
            if self.transcode:
                path = self._transcode(path)
            self.done.emit(True, str(path))
        except Exception as e:
            self.line.emit(f"fetch error: {e}")
            self.done.emit(False, "")

    def _stream(self, cmd):
        """Run cmd, emit its lines (download progress thinned to every
        5 %), return (exit code, last line)."""
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", **NO_WINDOW)
        last, shown = "", -10.0
        for ln in self._proc.stdout:
            ln = ln.rstrip()
            if not ln:
                continue
            m = _PROGRESS.match(ln)
            if m:
                pct = float(m.group(1))
                self.progress.emit(pct)
                if pct - shown < 5 and pct < 100:
                    continue
                shown = pct
            last = ln
            self.line.emit("  " + ln)
        rc = self._proc.wait()
        self._proc = None
        return rc, last

    def _download(self):
        cmd = yt_dlp_cmd()
        if cmd is None:
            self.line.emit(f"yt-dlp not found: {INSTALL_HINT}")
            return None
        self.folder.mkdir(parents=True, exist_ok=True)
        self.line.emit(f"fetching {self.url}")
        rc, last = self._stream(fetch_cmd(self.url, self.folder))
        if rc != 0:
            self.line.emit(f"✗ yt-dlp failed ({rc}); if the site changed, "
                           f"update it: {INSTALL_HINT} -U")
            return None
        path = Path(last.strip())            # --print after_move:filepath, last
        if not path.is_file():
            self.line.emit(f"✗ no file came back ({last})")
            return None
        self.line.emit(f"✔ {path.name}")
        return path

    def _transcode(self, path):
        """gacha_transcode.py on that one file; the .mov it writes
        replaces the download in the tree, or the download stays when
        the transcoder cannot run."""
        if shutil.which("ffprobe") is None:
            self.line.emit("  no ffprobe on PATH: kept as downloaded (plays fine, "
                           "seeks slower); install ffmpeg to transcode "
                           "(winget install Gyan.FFmpeg)")
            return path
        self.line.emit(f"transcoding {path.name} for instant seeking")
        rc, _ = self._stream([sys.executable, "-u", "-X", "utf8", str(TRANSCODE),
                              str(path)])
        if rc != 0:
            self.line.emit("✗ transcode failed, the download is kept")
            return path
        for ext in (".mov", ".mp4"):         # whatever codec the tool defaults to
            out = path.with_suffix(ext)
            if out.is_file() and out != path:
                return out
        return path

    def stop(self):
        """Kill the subprocess (window closing); the partial download is
        yt-dlp's .part file, resumed by the next fetch of the same URL."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
        self.wait(3000)
