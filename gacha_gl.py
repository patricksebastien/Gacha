#!/usr/bin/env python3
"""
gacha_gl.py — OpenGL backdrop for gacha.py.

Two layers, always: the video plays underneath and runs through an effects
shader (the same effects the numpy path offers, at full resolution), and an
optional generative shader is composited on top with its own opacity and
blend mode. Generative shaders are Shadertoy-style fragment shaders in
./shaders/*.frag: they get the usual iTime / iResolution / iChannel0 (the
processed video) / iChannel1 (their own previous frame), plus music data
from the playing song (iLoud, iBeat, iBar, iBPM, iSection, iRoot, iDrop,
iSwell, iSongPos).

gacha.py falls back to its numpy VideoBackdrop when no usable OpenGL 3.3
context exists.
"""

import random
import re
import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QImage, QOpenGLContext, QSurfaceFormat, QVector2D
from PySide6.QtMultimedia import QMediaPlayer, QVideoSink
from PySide6.QtOpenGL import (
    QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLShader,
    QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVersionFunctionsFactory,
    QOpenGLVersionProfile, QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QVBoxLayout

SHADERS_DIR = Path(__file__).parent / "shaders"


class FrameHistory:
    """The last few seconds of decoded video, downscaled and sparse, so the
    backdrop can run backwards for a bar or two. QMediaPlayer cannot play
    in reverse and a shader only ever sees the current frame, so the past
    has to be kept somewhere: 12 frames a second at half size for 5 s is
    about 120 MB at 1080p, and the effects run on top as usual."""

    def __init__(self, seconds=5.0, fps=12.0, scale=0.5):
        self.span, self.step, self.scale = seconds, 1.0 / fps, scale
        self.frames = deque()            # (time, QImage), oldest first
        self._last = 0.0
        self.rev_start = None            # wall time a reverse run began

    def push(self, img, now):
        if now - self._last < self.step * 0.9 or img.isNull():
            return
        self._last = now
        small = img.scaled(max(2, int(img.width() * self.scale)),
                           max(2, int(img.height() * self.scale)),
                           Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.frames.append((now, small))
        while self.frames and now - self.frames[0][0] > self.span:
            self.frames.popleft()

    def rewind(self, now, speed=1.0):
        """Frame for a reverse run: time runs back from the moment the run
        began, `speed` times faster than real time. Holds the oldest frame
        when the history runs out."""
        if not self.frames:
            return None
        if self.rev_start is None:
            self.rev_start = now
        target = self.rev_start - (now - self.rev_start) * speed
        for t, img in reversed(self.frames):
            if t <= target:
                return img
        return self.frames[0][1]

    def stop(self):
        self.rev_start = None

# GL constants (PySide6 does not export them)
GL_TRIANGLE_STRIP, GL_FLOAT = 0x0005, 0x1406
GL_TEXTURE0, GL_TEXTURE_2D = 0x84C0, 0x0DE1
GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE = 0x1908, 0x80E1, 0x1401
GL_COLOR_BUFFER_BIT = 0x4000
GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER = 0x2801, 0x2800
GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T = 0x2802, 0x2803
GL_LINEAR, GL_CLAMP_TO_EDGE = 0x2601, 0x812F
GL_FRAMEBUFFER = 0x8D40

BLEND_MODES = ["mix", "add", "screen"]


def gl_available():
    """True if this platform can host a QOpenGLWidget with OpenGL 3.3 core.
    GACHA_NO_GL=1 forces the numpy fallback."""
    import os
    from PySide6.QtGui import QGuiApplication
    if os.environ.get("GACHA_NO_GL"):
        return False
    app = QGuiApplication.instance()
    # headless platforms create contexts fine but cannot render GL widgets
    if app is not None and app.platformName() in ("offscreen", "minimal", "vnc"):
        return False
    try:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            return False
        f = ctx.format()
        return (f.majorVersion(), f.minorVersion()) >= (3, 3)
    except Exception:
        return False


def shader_files():
    """Generative shaders available in ./shaders, sorted."""
    if not SHADERS_DIR.is_dir():
        return []
    return sorted(p for p in SHADERS_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in (".frag", ".glsl"))


# ---------------------------------------------------------------- shaders

# fullscreen quad from gl_VertexID: no vertex buffer, no attribute state
VS = """#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)) * 2.0 - 1.0;
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

# The effects pass: same effects and order as the numpy VideoFX, per pixel.
FX_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uVideo;      // the decoded frame (top row first)
uniform sampler2D uPrev;       // previous output, for trails
uniform vec2 uRes, uVidRes;
uniform float uTime, uHasPrev;
uniform float uBrightness, uSaturation, uHue, uTintHue, uTintAmt, uInvert;
uniform float uBits, uDither, uMonoTones, uMonoThr, uSolar, uEdges, uScan;
uniform float uGrain, uVign, uPixel, uRgbShift, uGlitch, uGlitchMode, uTrails, uZoom;
uniform float uKaleido, uRepeat, uRepeatK, uRepeatFrac, uSwell;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 hueRotate(vec3 c, float turns) {
    float a = 6.2831853 * turns, co = cos(a), si = sin(a);
    mat3 m = mat3(
        0.213 + 0.787*co - 0.213*si, 0.213 - 0.213*co + 0.143*si, 0.213 - 0.213*co - 0.787*si,
        0.715 - 0.715*co - 0.715*si, 0.715 + 0.285*co + 0.140*si, 0.715 - 0.715*co + 0.715*si,
        0.072 - 0.072*co + 0.928*si, 0.072 - 0.072*co - 0.283*si, 0.072 + 0.928*co + 0.072*si);
    return m * c;
}
vec3 hue2rgb(float h) {
    return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
}
// cover-crop the video into the widget, zoomed toward the centre
vec2 coverUV(vec2 uv, float zoom) {
    float wr = uRes.x / uRes.y, ir = uVidRes.x / uVidRes.y;
    vec2 s = (ir > wr) ? vec2(wr / ir, 1.0) : vec2(1.0, ir / wr);
    uv = (uv - 0.5) * s * (1.0 - zoom) + 0.5;
    return vec2(uv.x, 1.0 - uv.y);                  // QImage rows are top-down
}
vec3 video(vec2 uv, float zoom) { return texture(uVideo, coverUV(uv, zoom)).rgb; }

void main() {
    vec2 uv = vUV;
    float aspect = uRes.x / uRes.y;
    float zoom = clamp(uZoom, 0.0, 0.45);
    if (uRepeat > 0.5) zoom += 0.18 * uRepeatFrac;   // beat-repeat: zoom, snap back
    if (uKaleido > 0.5) {                            // polar fold, upright
        vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
        float r = length(p), a = atan(p.y, p.x), w = 6.2831853 / uKaleido;
        float k = floor(a / w), t = a - k * w;
        if (mod(k, 2.0) > 0.5) t = w - t;
        uv = r * vec2(cos(t), sin(t)) / vec2(aspect, 1.0) + 0.5;
    }
    if (uPixel > 0.0) {                              // pixelate
        vec2 cell = vec2(uPixel, uPixel * aspect);
        uv = (floor(uv / cell) + 0.5) * cell;
    }
    if (uGlitch > 0.0) {                             // horizontal glitch bands
        // a handful of bands per tick, each with its own place, height and
        // shift; heights are log-uniform between hmin and hmax, per section:
        // mode 0 thin sharp lines (0.4-1% of the frame), 1 medium bands
        // (4-14%), 2 anything from 4% up to 38% slabs
        float tick = floor(uTime * 14.0);
        float hmin = uGlitchMode < 0.5 ? 0.004 : 0.04;
        float hmax = uGlitchMode < 0.5 ? 0.01 : (uGlitchMode < 1.5 ? 0.14 : 0.38);
        for (int i = 0; i < 6; i++) {
            float fi = float(i);
            if (hash(vec2(fi + 7.0, tick)) > uGlitch * 0.6) continue;
            float h = hmin * pow(hmax / hmin, hash(vec2(fi + 31.0, tick)));
            float y0 = hash(vec2(fi + 53.0, tick)) * (1.0 - h);
            if (uv.y >= y0 && uv.y < y0 + h)
                uv.x += (hash(vec2(tick, fi + 97.0)) - 0.5) * 0.4 * uGlitch;
        }
    }
    vec3 col;
    if (uRgbShift > 0.0) {                           // chromatic aberration
        col.r = video(uv - vec2(uRgbShift, 0.0), zoom).r;
        col.g = video(uv, zoom).g;
        col.b = video(uv + vec2(uRgbShift, 0.0), zoom).b;
    } else col = video(uv, zoom);
    if (uEdges > 0.0) {
        vec2 d = 1.0 / uRes;
        float gx = luma(video(uv + vec2(d.x, 0.0), zoom)) - luma(col);
        float gy = luma(video(uv + vec2(0.0, d.y), zoom)) - luma(col);
        col = mix(col, vec3(clamp((abs(gx) + abs(gy)) * 3.0 * 255.0 / 255.0 * 4.0, 0.0, 1.0)), uEdges);
    }
    if (uHue != 0.0) col = hueRotate(col, uHue);
    if (uSaturation != 1.0) col = mix(vec3(luma(col)), col, uSaturation);
    if (uTintAmt > 0.0) col = mix(col, luma(col) * hue2rgb(uTintHue), uTintAmt);
    col *= uBrightness;
    if (uSolar > 0.0) {
        float thr = 1.0 - 0.6 * uSolar;
        col = mix(col, 1.0 - col, step(thr, col));
    }
    if (uMonoTones > 0.5) {
        float g = luma(col);
        if (uMonoTones < 2.5) g = step(uMonoThr, g);
        else if (uMonoTones > 2.5) g = floor(clamp(g, 0.0, 0.999) * uMonoTones) / (uMonoTones - 1.0);
        col = vec3(g);
    }
    if (uBits > 0.5) {
        float levels = pow(2.0, uBits);
        col = floor(clamp(col, 0.0, 0.999) * levels) / (levels - 1.0);
    }
    if (uDither > 0.0) {                             // 4x4 Bayer, 1 bit
        ivec2 q = ivec2(mod(gl_FragCoord.xy, 4.0));
        int idx = q.y * 4 + q.x;
        float bayer[16] = float[16](0.,8.,2.,10., 12.,4.,14.,6., 3.,11.,1.,9., 15.,7.,13.,5.);
        float bw = step(bayer[idx] / 16.0, luma(col));
        col = mix(col, vec3(bw), uDither);
    }
    if (uScan > 0.0 && mod(gl_FragCoord.y, 2.0) < 1.0) col *= 1.0 - 0.7 * uScan;
    if (uGrain > 0.0) col += (hash(gl_FragCoord.xy + fract(uTime) * 100.0) - 0.5) * 0.35 * uGrain;
    if (uVign > 0.0) {
        vec2 p = vUV * 2.0 - 1.0;
        col *= 1.0 - uVign * clamp((p.x * p.x + p.y * p.y * 1.3) * 0.7, 0.0, 1.0);
    }
    if (uRepeat > 0.5 && mod(uRepeatK, 2.0) > 0.5) col = 1.0 - col;
    if (uSwell > 0.0) col += (1.0 - col) * 0.75 * uSwell * uSwell;
    if (uInvert > 0.5) col = 1.0 - col;
    col = clamp(col, 0.0, 1.0);
    if (uTrails > 0.0 && uHasPrev > 0.5)
        col = max(col, texture(uPrev, vUV).rgb * (0.55 + 0.42 * uTrails));
    fragColor = vec4(col, 1.0);
}
"""

# Final pass: video layer + generative layer + the legibility scrim.
COMPOSITE_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uFx, uGen, uOverlay;
uniform float uGenOn, uOpacity, uBlend, uScrim, uOverlayA;
void main() {
    vec3 base = texture(uFx, vUV).rgb;
    vec3 col = base;
    if (uGenOn > 0.5) {
        vec4 g = texture(uGen, vUV);
        float a = clamp(g.a, 0.0, 1.0) * uOpacity;
        if (uBlend < 0.5)      col = mix(base, g.rgb, a);                 // mix
        else if (uBlend < 1.5) col = base + g.rgb * a;                    // add
        else                   col = 1.0 - (1.0 - base) * (1.0 - g.rgb * a); // screen
    }
    if (uOverlayA > 0.0) {                       // word cloud, drawn by Qt (rows top-down)
        vec4 o = texture(uOverlay, vec2(vUV.x, 1.0 - vUV.y));
        col = mix(col, o.rgb, o.a * uOverlayA);
    }
    col = mix(col, vec3(0.047, 0.027, 0.02), uScrim * 0.215);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

# Shadertoy-compatible wrapper for ./shaders/*.frag
GEN_PREAMBLE = """#version 330 core
in vec2 vUV;
out vec4 gacha_outColor;
uniform vec3  iResolution;
uniform float iTime, iTimeDelta;
uniform int   iFrame;
uniform vec4  iMouse, iDate;
uniform sampler2D iChannel0;   // the processed video
uniform sampler2D iChannel1;   // this shader's previous frame
uniform vec3  iChannelResolution[2];
// gacha: the playing song
uniform float iLoud;      // loudness 0..1
uniform float iBeat;      // phase inside the beat 0..1
uniform float iBar;       // phase inside the bar 0..1
uniform float iBPM;
uniform int   iSection;   // 0 intro, 1 groove, 2 break, 3 outro
uniform int   iRoot;      // root note 0..11 (C..B), -1 unknown
uniform float iDrop;      // 1 on a drop, decaying to 0
uniform float iSwell;     // reverse-cymbal build 0..1
uniform float iSongPos;   // position in the song 0..1
#line 1
"""
GEN_POSTAMBLE = """
void main() { mainImage(gacha_outColor, vUV * iResolution.xy); }
"""


# ---------------------------------------------------------------- backdrop

class GLBackdrop(QOpenGLWidget):
    """OpenGL backdrop: video -> effects shader -> generative layer -> screen.
    Same surface as gacha.VideoBackdrop: child, state_fn, current,
    set_video(), pick_random(), jump(), plus set_shader()/shader_mix/blend."""

    def __init__(self, child, videos_fn, default_video):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSwapInterval(1)
        super().__init__()
        self.setFormat(fmt)
        self.child = child
        self.state_fn = None
        self.current = None
        self.videos_fn, self.default_video = videos_fn, default_video
        self._start_random = False
        self._pending = None             # newest decoded frame (QImage)
        self._frozen = False
        self._history = FrameHistory()   # for the reverse effect
        self._gl_ok = False
        self._t0 = time.monotonic()
        self._frame_n = 0
        self._last_t = self._t0
        self.log = []                    # compile errors etc., for the GUI
        self.frame_ms = 0.0              # smoothed CPU time of paintGL
        # generative layer
        self.shader_path = None
        self.shader_mix = 0.5
        self.shader_blend = 0             # index into BLEND_MODES
        self._gen_prog = None
        self._gen_pending = None          # path to compile on next paint
        self._gen_cache = {}              # path -> (mtime, program): compile once
        self._gen_min_opacity = {}        # path -> opacity a shader insists on
        self._prewarm = []                # shaders still to compile in idle time
        self._overlay_img = None          # word cloud QImage waiting for upload
        self._overlay_tex = None
        self._overlay_keep = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(child)
        # video source
        self._vplayer = None
        if videos_fn():
            self._sink = QVideoSink()
            self._sink.videoFrameChanged.connect(self._on_frame)
            self._vplayer = QMediaPlayer()          # no audio output = silent
            self._vplayer.setVideoSink(self._sink)
            self._vplayer.setLoops(QMediaPlayer.Loops.Infinite)
            if default_video.exists():
                self.set_video(default_video)
            else:
                self.pick_random()
        self._timer = QTimer(self)                 # steady 60 fps repaint
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    # ---- video source (same behaviour as the numpy backdrop) ----
    def set_video(self, path, random_start=False):
        if self._vplayer is None:
            return None
        self.current = Path(path)
        self._start_random = random_start
        self._vplayer.stop()
        self._vplayer.setSource(QUrl.fromLocalFile(str(self.current)))
        self._vplayer.play()
        return self.current

    def pick_random(self):
        files = self.videos_fn()
        if self._vplayer is None or not files:
            return None
        files = [f for f in files if f != self.default_video] or files
        pool = [f for f in files if f != self.current] or files
        return self.set_video(random.choice(pool), random_start=True)

    def jump(self):
        if self._vplayer is not None and self._vplayer.duration() > 0:
            self._vplayer.setPosition(random.randrange(self._vplayer.duration()))

    def _on_frame(self, frame):
        if not frame.isValid():
            return
        if self._start_random:
            self._start_random = False
            self.jump()
            return
        img = frame.toImage()
        self._history.push(img, time.monotonic())
        if not self._frozen:
            self._pending = img

    # ---- generative layer ----
    def last_frame(self):
        """The most recently uploaded video frame (QImage) or None."""
        return getattr(self, "_frame_img", None)

    def set_overlay(self, img):
        """A transparent RGBA QImage drawn over everything (the word cloud).
        Uploaded on the next paint; its opacity comes from state['words']."""
        self._overlay_img = img.convertToFormat(QImage.Format_RGBA8888) \
            if img is not None else None

    def _upload_overlay(self):
        img = self._overlay_img
        self._overlay_img = None
        if img is None:
            return
        w, h = img.width(), img.height()
        if img.bytesPerLine() != w * 4:
            img = img.copy()
        if self._overlay_tex is None or (self._overlay_tex.width(),
                                         self._overlay_tex.height()) != (w, h):
            if self._overlay_tex is not None:
                self._overlay_tex.destroy()
            tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            tex.setFormat(QOpenGLTexture.RGBA8_UNorm)
            tex.setSize(w, h)
            tex.allocateStorage()
            tex.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
            tex.setWrapMode(QOpenGLTexture.ClampToEdge)
            self._overlay_tex = tex
        self._overlay_tex.bind()
        self.gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA,
                                GL_UNSIGNED_BYTE, img.constBits())
        self._overlay_keep = img

    def set_shader(self, path):
        """Select a ./shaders file (or None). Compiled on the next paint."""
        self.shader_path = Path(path) if path else None
        self._gen_pending = self.shader_path or ""

    # ---- GL lifecycle ----
    def initializeGL(self):
        ctx = self.context()
        self.gl = QOpenGLVersionFunctionsFactory.get(
            QOpenGLVersionProfile(ctx.format()), ctx)
        if self.gl is None:
            self.log.append("no OpenGL 3.3 functions")
            return
        self.gl.initializeOpenGLFunctions()
        self._fx = self._program(FX_FS, "effects")
        self._comp = self._program(COMPOSITE_FS, "composite")
        if not (self._fx and self._comp):
            return
        self._vao = QOpenGLVertexArrayObject()   # core profile needs one bound,
        self._vao.create()                       # even with no attributes
        self._video_tex = None
        self._vid_size = (0, 0)
        self._fbo_fx = [None, None]
        self._fbo_gen = [None, None]
        self._ping = 0
        self._has_prev = False
        self._fbo_size = None
        self._gl_ok = True
        ctx.aboutToBeDestroyed.connect(self._cleanup_gl)

    def _cleanup_gl(self):
        """Free GL objects while the context is still current (on exit)."""
        self.makeCurrent()
        for tex in (getattr(self, "_video_tex", None),
                    getattr(self, "_overlay_tex", None)):
            if tex is not None:
                tex.destroy()
        self._video_tex = None
        self._overlay_tex = None
        self._fbo_fx = [None, None]
        self._fbo_gen = [None, None]
        self._gen_prog = None
        self.doneCurrent()

    def _program(self, frag_src, name):
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VS) \
            and prog.addShaderFromSourceCode(QOpenGLShader.Fragment, frag_src) \
            and prog.link()
        if not ok:
            self.log.append(f"{name} shader: {prog.log().strip()}")
            return None
        return prog

    def _compile_gen(self, path):
        """Compile a Shadertoy-style shader file; None on failure (logged).
        Programs are cached by path and file mtime, so a section change
        never stalls on the compiler and an edited file is picked up."""
        path = Path(path)
        try:
            mtime = path.stat().st_mtime
            hit = self._gen_cache.get(path)
            if hit and hit[0] == mtime:
                return hit[1]
            src = path.read_text()
        except Exception as e:
            self.log.append(f"{path.name}: {e}")
            return None
        # optional directive: "// gacha: opacity=1.0" for shaders that only
        # work when they replace the picture rather than mix with it
        m = re.search(r"gacha:\s*opacity\s*=\s*([0-9.]+)", src)
        self._gen_min_opacity[path] = float(m.group(1)) if m else 0.0
        src = re.sub(r"^\s*precision\s.*$", "", src, flags=re.M)   # GLSL ES only
        src = re.sub(r"^\s*#version.*$", "", src, flags=re.M)
        prog = QOpenGLShaderProgram()
        ok = prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VS) \
            and prog.addShaderFromSourceCode(
                QOpenGLShader.Fragment, GEN_PREAMBLE + src + GEN_POSTAMBLE) \
            and prog.link()
        if not ok:
            self.log.append(f"{path.name}: {prog.log().strip()[:400]}")
            self._gen_cache[path] = (mtime, None)
            return None
        self._gen_cache[path] = (mtime, prog)
        return prog

    def prewarm(self, paths):
        """Queue shaders to compile one per frame in the background, so the
        first use of each is instant."""
        self._prewarm = [Path(p) for p in paths if Path(p) not in self._gen_cache]

    def resizeGL(self, w, h):
        self._fbo_size = None            # rebuilt lazily at device pixel size

    def _ensure_fbos(self):
        dpr = self.devicePixelRatioF()
        size = (max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr)))
        if size == self._fbo_size:
            return
        self._fbo_size = size
        self._fbo_fx = [QOpenGLFramebufferObject(*size) for _ in range(2)]
        self._fbo_gen = [QOpenGLFramebufferObject(*size) for _ in range(2)]
        for fbo in self._fbo_fx + self._fbo_gen:
            self._set_filters(fbo.texture())
        self._has_prev = False

    def _set_filters(self, tex_id):
        gl = self.gl
        gl.glBindTexture(GL_TEXTURE_2D, tex_id)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def _upload_frame(self, img):
        """Newest decoded frame -> the persistent video texture (0.6 ms)."""
        if img.format() in (QImage.Format_ARGB32, QImage.Format_RGB32,
                            QImage.Format_ARGB32_Premultiplied):
            fmt = GL_BGRA
        else:
            img = img.convertToFormat(QImage.Format_RGBA8888)
            fmt = GL_RGBA
        w, h = img.width(), img.height()
        if img.bytesPerLine() != w * 4:
            img = img.copy()             # tight rows
        if self._video_tex is None or self._vid_size != (w, h):
            if self._video_tex is not None:
                self._video_tex.destroy()
            tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            tex.setFormat(QOpenGLTexture.RGBA8_UNorm)
            tex.setSize(w, h)
            tex.allocateStorage()
            tex.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
            tex.setWrapMode(QOpenGLTexture.ClampToEdge)
            self._video_tex, self._vid_size = tex, (w, h)
        self._video_tex.bind()
        self.gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, fmt,
                                GL_UNSIGNED_BYTE, img.constBits())
        self._frame_img = img            # keep the buffer alive for the call

    def _bind_tex(self, unit, tex_id):
        self.gl.glActiveTexture(GL_TEXTURE0 + unit)
        self.gl.glBindTexture(GL_TEXTURE_2D, tex_id)

    def _draw(self):
        self._vao.bind()
        self.gl.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        self._vao.release()

    def paintGL(self):
        if not self._gl_ok:
            return
        t_start = time.perf_counter()
        self._paint()
        self.frame_ms = 0.9 * self.frame_ms + 0.1 * (time.perf_counter() - t_start) * 1000

    def _paint(self):
        gl = self.gl
        st = (self.state_fn() if self.state_fn else None) or {}
        self._frozen = bool(st.get("repeat") or st.get("freeze"))
        rev = st.get("reverse")
        if rev:                                  # play the history backwards
            img = self._history.rewind(time.monotonic(), float(rev))
            if img is not None:
                self._upload_frame(img)
            self._pending = None
        else:
            self._history.stop()
            if self._pending is not None:
                self._upload_frame(self._pending)
                self._pending = None
        if self._overlay_img is not None:
            self._upload_overlay()
        if self._video_tex is None:
            gl.glClearColor(16 / 255, 9 / 255, 7 / 255, 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT)
            return
        self._ensure_fbos()
        W, H = self._fbo_size
        now = time.monotonic()
        t = now - self._t0
        dt = now - self._last_t
        self._last_t = now
        self._frame_n += 1
        cur, prev = self._ping, 1 - self._ping

        # ---- pass 1: video + effects -> fbo_fx[cur] ----
        self._fbo_fx[cur].bind()
        gl.glViewport(0, 0, W, H)
        p = self._fx
        p.bind()
        self._bind_tex(0, self._video_tex.textureId())
        self._bind_tex(1, self._fbo_fx[prev].texture())
        p.setUniformValue1i("uVideo", 0)
        p.setUniformValue1i("uPrev", 1)
        p.setUniformValue("uRes", QVector2D(W, H))
        p.setUniformValue("uVidRes", QVector2D(*self._vid_size))
        p.setUniformValue1f("uTime", t)
        p.setUniformValue1f("uHasPrev", 1.0 if self._has_prev else 0.0)
        self._set_fx_uniforms(p, st)
        self._draw()

        # ---- pass 2: generative layer -> fbo_gen[cur] ----
        if self._gen_pending is not None:
            self._gen_prog = self._compile_gen(self._gen_pending) \
                if self._gen_pending else None
            self._gen_pending = None
        elif self._prewarm:                       # one background compile per frame
            self._compile_gen(self._prewarm.pop(0))
        opacity = max(float(self.shader_mix),
                      self._gen_min_opacity.get(self.shader_path, 0.0))
        gen_on = self._gen_prog is not None and opacity > 0 \
            and not st.get("clean")              # clean bar: video only
        if gen_on:
            self._fbo_gen[cur].bind()
            gl.glViewport(0, 0, W, H)
            g = self._gen_prog
            g.bind()
            self._bind_tex(0, self._fbo_fx[cur].texture())
            self._bind_tex(1, self._fbo_gen[prev].texture())
            g.setUniformValue1i("iChannel0", 0)
            g.setUniformValue1i("iChannel1", 1)
            g.setUniformValue("iResolution", float(W), float(H), 1.0)
            g.setUniformValue1f("iTime", t)
            g.setUniformValue1f("iTimeDelta", dt)
            g.setUniformValue1i("iFrame", self._frame_n)
            m = st.get("music", {})
            g.setUniformValue1f("iLoud", float(m.get("loud", 0.5)))
            g.setUniformValue1f("iBeat", float(m.get("beat", 0.0)))
            g.setUniformValue1f("iBar", float(m.get("bar", 0.0)))
            g.setUniformValue1f("iBPM", float(m.get("bpm", 120.0)))
            g.setUniformValue1i("iSection", int(m.get("section", 1)))
            g.setUniformValue1i("iRoot", int(m.get("root", -1)))
            g.setUniformValue1f("iDrop", float(m.get("drop", 0.0)))
            g.setUniformValue1f("iSwell", float(m.get("swell", 0.0)))
            g.setUniformValue1f("iSongPos", float(m.get("song_pos", 0.0)))
            self._draw()

        # ---- pass 3: composite into the widget's own framebuffer ----
        gl.glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        dpr = self.devicePixelRatioF()
        gl.glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
        c = self._comp
        c.bind()
        self._bind_tex(0, self._fbo_fx[cur].texture())
        self._bind_tex(1, self._fbo_gen[cur].texture())
        c.setUniformValue1i("uFx", 0)
        c.setUniformValue1i("uGen", 1)
        c.setUniformValue1f("uGenOn", 1.0 if gen_on else 0.0)
        c.setUniformValue1f("uOpacity", opacity)
        c.setUniformValue1f("uBlend", float(self.shader_blend))
        c.setUniformValue1f("uScrim", 1.0 if self.child.isVisible() else 0.0)
        words = float(st.get("words", 0.0)) if self._overlay_tex is not None else 0.0
        if words > 0.0:
            self._bind_tex(2, self._overlay_tex.textureId())
        c.setUniformValue1i("uOverlay", 2)
        c.setUniformValue1f("uOverlayA", words)
        self._draw()
        self._has_prev = True
        self._ping = prev

    def _set_fx_uniforms(self, p, st):
        g = st.get
        f = p.setUniformValue1f
        f("uBrightness", float(g("brightness", 1.0)))
        f("uSaturation", float(g("saturation", 1.0)))
        f("uHue", float(g("hue", 0.0)))
        tint = g("tint") or (0.0, 0.0)
        f("uTintHue", float(tint[0]))
        f("uTintAmt", float(tint[1]))
        f("uInvert", 1.0 if g("invert") else 0.0)
        f("uBits", float(g("bits", 0) or 0))
        f("uDither", float(g("dither", 0.0)))
        mono = g("mono") or (0, 0.5)
        f("uMonoTones", float(mono[0]))
        f("uMonoThr", float(mono[1]))
        f("uSolar", float(g("solarize", 0.0)))
        f("uEdges", float(g("edges", 0.0)))
        f("uScan", float(g("scanlines", 0.0)))
        f("uGrain", float(g("grain", 0.0)))
        f("uVign", float(g("vignette", 0.0)))
        f("uPixel", float(g("pixelate", 0)) / 480.0)       # numpy worked at 480 px
        f("uRgbShift", float(g("rgbshift", 0)) / 480.0)
        f("uGlitch", float(g("glitch", 0.0)))
        f("uGlitchMode", float(g("glitch_mode", 2)))
        f("uTrails", float(g("trails", 0.0)))
        f("uZoom", float(g("zoom", 0.0)))
        kal = g("kaleido")
        f("uKaleido", float(kal[0]) if kal else 0.0)
        rpt = g("repeat")
        f("uRepeat", 1.0 if rpt else 0.0)
        f("uRepeatK", float(rpt[0]) if rpt else 0.0)
        f("uRepeatFrac", float(rpt[1]) if rpt else 0.0)
        f("uSwell", float(g("swell", 0.0)))
