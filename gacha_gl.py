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

The canvas: the picture does not have to fill the window. It can be cropped
(the edges of a tape cut off), placed (slid, zoomed, tilted in perspective
for a projector hitting the wall at an angle) and what is around it is a
background: a colour, a photo or a clip of its own. Every video effect and
every shader either keeps to the canvas (works on the picture alone, the
background untouched) or spills over the whole screen (a vignette darkens
the corners of the wall, a feedback shader rushes out of the picture onto
the background); `keep_fx` and `keep_shaders` say which.

gacha.py falls back to its numpy VideoBackdrop when no usable OpenGL 3.3
context exists.
"""

import math
import random
import re
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QMatrix3x3, QOpenGLContext, QSurfaceFormat, QVector2D
from PySide6.QtOpenGL import (
    QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLShader,
    QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVersionFunctionsFactory,
    QOpenGLVersionProfile, QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QVBoxLayout

from gacha_video import VideoSource

SHADERS_DIR = Path(__file__).parent / "shaders"


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

# the video effects that have a place on the screen, in the order of the
# bits of the effects shader's uKeep (gacha.py's VIDEO_EFFECTS keys; reverse
# and clean are not pictures and have no place); each one either keeps to
# the canvas or works over the whole screen, the background included
KEEP_FX = ["color", "tint", "pump", "flash", "pixel", "bits", "dither", "mono", "solar",
           "edges", "lines", "grain", "gran", "zoom", "trails", "glitch", "rgb", "vign",
           "kaleido"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
CANVAS_FITS = ["fill", "fit"]
CANVAS_DEFAULT = {"fit": "fill", "crop": (0.0, 0.0, 0.0, 0.0), "x": 0.0, "y": 0.0,
                  "zoom": 1.0, "tilt_x": 0.0, "tilt_y": 0.0,
                  "blend": 0, "mix": 1.0, "feather": 0.0}
# how the placed video meets the background: index = the shader's uPicBlend
PIC_BLENDS = ["over", "add", "multiply", "screen", "overlay", "difference", "lighten",
              "darken"]
BG_MODES = ["colour", "file"]
BG_DEFAULT = {"mode": "colour", "colour": (16 / 255, 9 / 255, 7 / 255), "path": ""}
EYE = 2.5                 # the tilt's perspective: the eye this many screen heights away


# ---------------------------------------------------------------- the canvas

def canvas_quad(vid_size, W, H, canvas):
    """Where the (cropped) picture lies on a W x H screen: its four corners
    in screen uv (bottom left (0, 0)), in canvas order (0,0), (1,0), (1,1),
    (0,1); its size in screen uv before the tilt; its aspect.

    `canvas`: fit (fill = the picture covers the screen at zoom 1, fit =
    the whole picture shows), crop (left, right, top, bottom as fractions
    of the source), x and y (-1..1 of the screen, 0 centred), zoom, tilt_x
    and tilt_y (degrees: a perspective tilt about the horizontal and the
    vertical axis, the eye EYE screen heights away)."""
    vw, vh = vid_size
    l, r, t, b = canvas.get("crop", CANVAS_DEFAULT["crop"])
    cw, ch = max(1e-6, vw * (1.0 - l - r)), max(1e-6, vh * (1.0 - t - b))
    a = cw / ch if vw > 0 and vh > 0 else 16 / 9
    A = W / max(1, H)
    # the picture in screen heights: w0 x h0 at zoom 1
    if (a >= A) == (canvas.get("fit", "fill") == "fit"):   # bound by the width
        w0, h0 = A, A / a
    else:                                                   # bound by the height
        w0, h0 = a, 1.0
    z = max(0.01, float(canvas.get("zoom", 1.0)))
    w, h = w0 * z, h0 * z
    tx = math.radians(float(canvas.get("tilt_x", 0.0)))
    ty = math.radians(float(canvas.get("tilt_y", 0.0)))
    ox = float(canvas.get("x", 0.0)) * A * 0.5
    oy = float(canvas.get("y", 0.0)) * 0.5
    pts = []
    for cx, cy in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        x, y, zz = cx * w, cy * h, 0.0
        x, zz = x * math.cos(ty), -x * math.sin(ty)              # about the vertical axis
        y, zz = y * math.cos(tx) - zz * math.sin(tx), y * math.sin(tx) + zz * math.cos(tx)
        d = EYE / max(0.2, EYE - zz)                            # nearer is bigger
        pts.append(((x * d + ox) / A + 0.5, y * d + oy + 0.5))
    return pts, (w / A, h), a


def square_to_quad(pts):
    """The 3x3 homography taking the unit square's corners (0,0), (1,0),
    (1,1), (0,1) to `pts`, acting on (u, v, 1)."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
    dx1, dx2, sx = x1 - x2, x3 - x2, x0 - x1 + x2 - x3
    dy1, dy2, sy = y1 - y2, y3 - y2, y0 - y1 + y2 - y3
    if abs(sx) < 1e-9 and abs(sy) < 1e-9:                      # affine
        return np.array([[x1 - x0, x3 - x0, x0], [y1 - y0, y3 - y0, y0], [0.0, 0.0, 1.0]])
    det = dx1 * dy2 - dx2 * dy1
    if abs(det) < 1e-12:
        det = 1e-12
    g = (sx * dy2 - dx2 * sy) / det
    h = (dx1 * sy - sx * dy1) / det
    return np.array([[x1 - x0 + g * x1, x3 - x0 + h * x3, x0],
                     [y1 - y0 + g * y1, y3 - y0 + h * y3, y0],
                     [g, h, 1.0]])


def canvas_geometry(vid_size, W, H, canvas):
    """Everything the shaders need for the canvas: `to_screen` (canvas uv
    -> screen uv) and `to_zone` (the inverse), `k` (canvas uv per screen uv,
    without the tilt), `aspect`, and `px`, the size a shader keeping to the
    canvas renders at (its aspect at the screen's height, whatever the
    zoom, so a knob on the zoom allocates nothing)."""
    pts, (wu, hu), a = canvas_quad(vid_size, W, H, canvas)
    to_screen = square_to_quad(pts)
    try:
        to_zone = np.linalg.inv(to_screen)
    except np.linalg.LinAlgError:
        to_zone = np.eye(3)
    pw = max(16, min(4096, int(round(H * a / 8.0)) * 8))
    return {"to_screen": to_screen, "to_zone": to_zone,
            "k": (1.0 / max(1e-6, wu), 1.0 / max(1e-6, hu)),
            "aspect": a, "px": (pw, max(16, H))}


def _mat3(m):
    return QMatrix3x3([float(v) for v in np.asarray(m).reshape(9)])


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

# GLSL shared by the passes that need to know where the canvas is
ZONE_GLSL = """
uniform mat3 uToZone;          // screen uv -> canvas uv, homogeneous (the tilt is a perspective)
uniform vec2 uZoneK;           // canvas uv per screen uv, without the tilt
uniform float uFeather;        // the edge's fade, as a fraction of the canvas's height
vec2 toZone(vec2 s) {
    vec3 q = uToZone * vec3(s, 1.0);
    if (q.z <= 1e-4) return vec2(-10.0);            // behind the eye: nowhere
    return q.xy / q.z;
}
// 1 on the canvas, 0 off it: a one-pixel soft edge, or the feather, the
// same width on every side
float inside(vec2 z) {
    float zaspect = (uRes.x / uRes.y) * uZoneK.y / uZoneK.x;
    vec2 e = max(uZoneK / uRes, vec2(uFeather / zaspect, uFeather));
    vec2 a = smoothstep(vec2(0.0), e, z) * smoothstep(vec2(0.0), e, 1.0 - z);
    return a.x * a.y;
}
"""

# The effects pass: same effects and order as the numpy VideoFX, per pixel,
# on the canvas or over the whole screen as each effect's uKeep bit says.
FX_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uVideo;      // the decoded frame (top row first)
uniform sampler2D uTake;       // the section's take frame, dissolved in by uAB
uniform sampler2D uPrev;       // previous output, for trails
uniform sampler2D uBg;         // the background picture (a photo or a clip), top row first
uniform vec2 uRes, uVidRes, uBgRes;
uniform float uTime, uHasPrev;
uniform float uBrightness, uSaturation, uHue, uTintHue, uTintAmt, uInvert;
uniform float uBits, uDither, uMonoTones, uMonoThr, uSolar, uEdges, uScan;
uniform float uGrain, uVign, uPixel, uRgbShift, uGlitch, uGlitchMode, uTrails, uZoom;
uniform float uKaleido, uRepeat, uRepeatK, uRepeatFrac, uSwell;
uniform float uGran;           // the granular picture: strength, and its bands
uniform int uGranN;            // (left edge, width, playhead x, lit), fractions of the width
uniform vec4 uGranHeads[16];
// source grade (colour correction of the picture itself, before any effect)
uniform float uGExposure, uGBlack, uGGamma, uGContrast, uGSat, uGWarm, uGTint, uGSharpen;
uniform float uAB;             // 0 = the live picture, 1 = the take frame
uniform float uABMode;         // 0 mix, 1 add, 2 screen (same order as the shader layer)
uniform vec2 uTakeScale;       // the B layer's cover-crop into the canvas: a 16:9
                               // webcam over a 5:4 tape is cropped, not squeezed
// the canvas: where the (cropped) picture lies, what is around it, and
// which effects keep to it
uniform vec4 uCrop;            // left, right, top, bottom cut off the source, fractions
uniform float uBgOn;           // 1: uBg is the background, else uBgColor
uniform vec3 uBgColor;
uniform int uKeep;             // a bit per effect, KEEP_FX order: 1 = it keeps to the canvas
uniform float uPicBlend;       // how the video meets the background: PIC_BLENDS index
uniform float uPicMix;         // the video's opacity over it, 0..1
//@KEEP
//@ZONE

bool keep(int i) { return (uKeep & (1 << i)) != 0; }
// an effect's result `e` over `col`: on the canvas alone when it keeps to
// it, everywhere otherwise
vec3 ap(int k, vec3 col, vec3 e, float ins) { return keep(k) ? mix(col, e, ins) : e; }

// the source pixel: the live picture dissolved toward the take frame by
// uAB, the picture's side of the A/B pedal. Every mode is pure tape at 0
// and pure take at 1, like the sound; they differ in the middle: mix is a
// plain dissolve, add and screen superimpose the two at full strength
// halfway (each side fades in over the first half of its travel and out
// over the second), add clipping bright on bright, screen never clipping.
vec3 src(vec2 pv, vec2 pt) {
    vec3 a = texture(uVideo, pv).rgb;
    if (uAB <= 0.0) return a;
    vec3 b = texture(uTake, pt).rgb;
    if (uABMode < 0.5) return mix(a, b, uAB);
    a *= min(1.0, 2.0 * (1.0 - uAB));
    b *= min(1.0, 2.0 * uAB);
    if (uABMode < 1.5) return a + b;
    return 1.0 - (1.0 - a) * (1.0 - b);
}

vec3 grade(vec3 c) {
    c = max(c * uGExposure + uGBlack, 0.0);
    c = pow(c, vec3(1.0 / max(uGGamma, 0.05)));
    c = (c - 0.5) * uGContrast + 0.5;
    // warmth tilts red against blue, tint green against magenta; both keep
    // the luma of a grey roughly where it was
    c *= vec3(1.0 + 0.12 * uGWarm, 1.0 - 0.05 * abs(uGWarm), 1.0 - 0.12 * uGWarm);
    c *= vec3(1.0 + 0.06 * uGTint, 1.0 - 0.10 * uGTint, 1.0 + 0.06 * uGTint);
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, uGSat);
    return clamp(c, 0.0, 1.0);
}

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
// the cropped source at canvas uv z (0,0 bottom left), the B layer
// cover-cropped into the same canvas, sharpened (unsharp mask against its
// 4 neighbours) or blurred (3x3 box) in the video's own texels so a
// 720x576 tape is treated at tape resolution, then graded
vec3 pic(vec2 z) {
    vec2 pv = vec2(uCrop.x + z.x * (1.0 - uCrop.x - uCrop.y),
                   uCrop.z + (1.0 - z.y) * (1.0 - uCrop.z - uCrop.w));   // rows top-down
    vec2 pt = (z - 0.5) * uTakeScale + 0.5;
    pt = vec2(pt.x, 1.0 - pt.y);
    vec3 c = src(pv, pt);
    if (uGSharpen > 0.0) {
        vec2 d = 1.0 / uVidRes;
        vec2 dx = vec2(d.x, 0.0), dy = vec2(0.0, d.y);
        vec3 b = (src(pv + dx, pt + dx) + src(pv - dx, pt - dx)
                + src(pv + dy, pt + dy) + src(pv - dy, pt - dy)) * 0.25;
        c = max(c + (c - b) * uGSharpen, 0.0);
    } else if (uGSharpen < 0.0) {
        // blur: a 3x3 box whose radius grows with the knob (1 to 10 source
        // texels); below 1 the box is blended in instead
        float a = -uGSharpen;
        vec2 d = max(a, 1.0) / uVidRes;
        vec3 b = c;
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                if (i != 0 || j != 0) {
                    vec2 o = vec2(float(i), float(j)) * d;
                    b += src(pv + o, pt + o);
                }
        c = mix(c, b / 9.0, min(a, 1.0));
    }
    return grade(c);
}
// the background at screen uv s: the colour, or the picture cover-cropped
// into the screen
vec3 bgc(vec2 s) {
    if (uBgOn < 0.5) return uBgColor;
    float wr = uRes.x / uRes.y, ir = uBgRes.x / uBgRes.y;
    vec2 sc = (ir > wr) ? vec2(wr / ir, 1.0) : vec2(1.0, ir / wr);
    vec2 p = (s - 0.5) * sc + 0.5;
    return texture(uBg, vec2(p.x, 1.0 - p.y)).rgb;
}
// the video `p` laid onto the background `bg` where it is (`a`): over
// it, or blended into it: add (a double exposure), multiply (the
// background's colour or picture tints the video), screen, overlay,
// difference, lighten, darken; then the video's own opacity
vec3 lay(vec3 bg, vec3 p, float a) {
    vec3 r;
    if (uPicBlend < 0.5)      r = p;
    else if (uPicBlend < 1.5) r = bg + p;
    else if (uPicBlend < 2.5) r = bg * p;
    else if (uPicBlend < 3.5) r = 1.0 - (1.0 - bg) * (1.0 - p);
    else if (uPicBlend < 4.5) r = mix(2.0 * bg * p, 1.0 - 2.0 * (1.0 - bg) * (1.0 - p),
                                      step(0.5, bg));
    else if (uPicBlend < 5.5) r = abs(bg - p);
    else if (uPicBlend < 6.5) r = max(bg, p);
    else                      r = min(bg, p);
    return mix(bg, clamp(r, 0.0, 1.0), a * uPicMix);
}

// the warps: a polar fold, upright; square cells; horizontal tears
vec2 kaleido(vec2 uv, float aspect) {
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
    float r = length(p), a = atan(p.y, p.x), w = 6.2831853 / uKaleido;
    float k = floor(a / w), t = a - k * w;
    if (mod(k, 2.0) > 0.5) t = w - t;
    return r * vec2(cos(t), sin(t)) / vec2(aspect, 1.0) + 0.5;
}
vec2 pixelate(vec2 uv, vec2 cell) { return (floor(uv / cell) + 0.5) * cell; }
vec2 glitch(vec2 uv, float xk) {
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
            uv.x += (hash(vec2(tick, fi + 97.0)) - 0.5) * 0.4 * uGlitch * xk;
    }
    return uv;
}

void main() {
    vec2 s = vUV;
    float aspect = uRes.x / uRes.y;
    float zaspect = aspect * uZoneK.y / uZoneK.x;    // the canvas's own
    vec2 z0 = toZone(vUV);                           // the canvas under this pixel, unwarped
    // the zooms: the effect's with the loudness, the beat-repeat's punch
    float zoom = clamp(uZoom, 0.0, 0.45);
    float zk = keep(K_ZOOM) ? zoom : 0.0, zs = zoom - zk;    // on the canvas / the screen
    if (uRepeat > 0.5) { if (keep(K_FLASH)) zk += 0.18 * uRepeatFrac; else zs += 0.18 * uRepeatFrac; }
    // the warps that spill off the canvas work on the screen, the ones
    // that keep to it on the canvas; between the two kinds the order is moot
    if (zs > 0.0) s = (s - 0.5) * (1.0 - zs) + 0.5;
    if (uKaleido > 0.5 && !keep(K_KALEIDO)) s = kaleido(s, aspect);
    if (uPixel > 0.0 && !keep(K_PIXEL)) s = pixelate(s, vec2(uPixel, uPixel * aspect));
    if (uGlitch > 0.0 && !keep(K_GLITCH)) s = glitch(s, 1.0);
    vec2 z = toZone(s);
    // the picture's rectangle is decided here, before the warps that keep
    // to it: those move what the picture shows, never its edges (a fold or
    // a tear that reaches past the edge reads the edge, the background
    // stays out), while a warp on the screen above moved the rectangle itself
    float ins = inside(z);
    if (zk > 0.0) z = (z - 0.5) * (1.0 - zk) + 0.5;
    if (uKaleido > 0.5 && keep(K_KALEIDO)) z = kaleido(z, zaspect);
    if (uPixel > 0.0 && keep(K_PIXEL)) z = pixelate(z, vec2(uPixel, uPixel * aspect) * uZoneK);
    if (uGlitch > 0.0 && keep(K_GLITCH)) z = glitch(z, uZoneK.x);
    z = clamp(z, 0.0, 1.0);
    vec3 col;
    if (uRgbShift > 0.0) {                           // chromatic aberration
        vec2 d = vec2(uRgbShift, 0.0);
        if (keep(K_RGB)) {
            vec2 dz = d * uZoneK;
            col = lay(bgc(s), vec3(pic(z - dz).r, pic(z).g, pic(z + dz).b), ins);
        } else {
            vec2 dz = toZone(s + d) - toZone(s);
            col = vec3(lay(bgc(s - d), pic(z - dz), inside(z - dz)).r,
                       lay(bgc(s), pic(z), ins).g,
                       lay(bgc(s + d), pic(z + dz), inside(z + dz)).b);
        }
    } else col = lay(bgc(s), pic(z), ins);
    if (uGran > 0.0) {
        // the granular picture: the frame is the buffer, now at its right
        // edge, and every grain a band at its spot, as wide as the grain,
        // lifted by its window with soft edges, a bright playhead line where
        // it reads; outside the bands the frame darkens a little
        bool kp = keep(K_GRAN);
        float x = kp ? z0.x : vUV.x;
        float lit = 0.0, line = 0.0;
        float px1 = 1.5 / uRes.x * (kp ? uZoneK.x : 1.0);
        for (int i = 0; i < 16; i++) {
            if (i >= uGranN) break;
            vec4 hd = uGranHeads[i];
            if (hd.y <= 0.0) continue;
            float e = max(px1, 0.08 * hd.y);
            float in_ = clamp((x - hd.x) / e, 0.0, 1.0) * clamp((hd.x + hd.y - x) / e, 0.0, 1.0);
            lit = max(lit, in_ * hd.w);
            line = max(line, clamp(1.0 - abs(x - hd.z) / px1, 0.0, 1.0) * in_);
        }
        vec3 e = col * (1.0 - 0.3 * uGran + 0.6 * uGran * lit) + 0.16 * uGran * lit;
        e = mix(e, 1.0 - e, line * uGran);
        col = ap(K_GRAN, col, e, ins);
    }
    if (uEdges > 0.0) {
        float gx, gy;
        if (keep(K_EDGES)) {
            vec2 d = uZoneK / uRes;
            gx = luma(pic(z + vec2(d.x, 0.0))) - luma(col);
            gy = luma(pic(z + vec2(0.0, d.y))) - luma(col);
        } else {
            vec2 d = 1.0 / uRes;
            vec2 dzx = toZone(s + vec2(d.x, 0.0)) - toZone(s);
            vec2 dzy = toZone(s + vec2(0.0, d.y)) - toZone(s);
            gx = luma(lay(bgc(s + vec2(d.x, 0.0)), pic(z + dzx), inside(z + dzx))) - luma(col);
            gy = luma(lay(bgc(s + vec2(0.0, d.y)), pic(z + dzy), inside(z + dzy))) - luma(col);
        }
        vec3 e = mix(col, vec3(clamp((abs(gx) + abs(gy)) * 12.0, 0.0, 1.0)), uEdges);
        col = ap(K_EDGES, col, e, ins);
    }
    if (uHue != 0.0) col = ap(K_COLOR, col, hueRotate(col, uHue), ins);
    if (uSaturation != 1.0) col = ap(K_COLOR, col, mix(vec3(luma(col)), col, uSaturation), ins);
    if (uTintAmt > 0.0) col = ap(K_TINT, col, mix(col, luma(col) * hue2rgb(uTintHue), uTintAmt), ins);
    if (uBrightness != 1.0) col = ap(K_PUMP, col, col * uBrightness, ins);
    if (uSolar > 0.0) {
        float thr = 1.0 - 0.6 * uSolar;
        col = ap(K_SOLAR, col, mix(col, 1.0 - col, step(thr, col)), ins);
    }
    if (uMonoTones > 0.5) {                          // 1 = smooth gray, 2 = threshold, 3+ = tones
        float g = luma(col);
        if (uMonoTones > 1.5 && uMonoTones < 2.5) g = step(uMonoThr, g);
        else if (uMonoTones > 2.5) g = floor(clamp(g, 0.0, 0.999) * uMonoTones) / (uMonoTones - 1.0);
        col = ap(K_MONO, col, vec3(g), ins);
    }
    if (uBits > 0.5) {
        float levels = pow(2.0, uBits);
        col = ap(K_BITS, col, floor(clamp(col, 0.0, 0.999) * levels) / (levels - 1.0), ins);
    }
    if (uDither > 0.0) {                             // 4x4 Bayer, 1 bit
        ivec2 q = ivec2(mod(gl_FragCoord.xy, 4.0));
        int idx = q.y * 4 + q.x;
        float bayer[16] = float[16](0.,8.,2.,10., 12.,4.,14.,6., 3.,11.,1.,9., 15.,7.,13.,5.);
        float bw = step(bayer[idx] / 16.0, luma(col));
        col = ap(K_DITHER, col, mix(col, vec3(bw), uDither), ins);
    }
    if (uScan > 0.0 && mod(gl_FragCoord.y, 2.0) < 1.0)
        col = ap(K_LINES, col, col * (1.0 - 0.7 * uScan), ins);
    if (uGrain > 0.0)
        col = ap(K_GRAIN, col, col + (hash(gl_FragCoord.xy + fract(uTime) * 100.0) - 0.5) * 0.35 * uGrain, ins);
    if (uVign > 0.0) {
        vec2 p = (keep(K_VIGN) ? z0 : vUV) * 2.0 - 1.0;
        col = ap(K_VIGN, col, col * (1.0 - uVign * clamp((p.x * p.x + p.y * p.y * 1.3) * 0.7, 0.0, 1.0)), ins);
    }
    if (uRepeat > 0.5 && mod(uRepeatK, 2.0) > 0.5) col = ap(K_FLASH, col, 1.0 - col, ins);
    if (uSwell > 0.0) col = ap(K_FLASH, col, col + (1.0 - col) * 0.75 * uSwell * uSwell, ins);
    if (uInvert > 0.5) col = ap(K_FLASH, col, 1.0 - col, ins);
    col = clamp(col, 0.0, 1.0);
    if (uTrails > 0.0 && uHasPrev > 0.5)
        col = ap(K_TRAILS, col, max(col, texture(uPrev, vUV).rgb * (0.55 + 0.42 * uTrails)), ins);
    fragColor = vec4(col, 1.0);
}
""".replace("//@KEEP", "\n".join(f"const int K_{k.upper()} = {i};" for i, k in enumerate(KEEP_FX))) \
   .replace("//@ZONE", ZONE_GLSL)

# Final pass: video layer + generative layer + the legibility scrim. A
# shader that kept to the canvas was drawn at the canvas's own size and is
# laid back into it here, through the canvas's perspective.
COMPOSITE_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uFx, uGen;
uniform vec2 uRes;
uniform float uGenOn, uOpacity, uBlend, uScrim;
uniform float uGenZone;        // 1: uGen is the canvas alone, not the screen
//@ZONE
void main() {
    vec3 base = texture(uFx, vUV).rgb;
    vec3 col = base;
    if (uGenOn > 0.5) {
        vec2 p = vUV;
        float m = 1.0;
        if (uGenZone > 0.5) {
            p = toZone(vUV);
            m = inside(p);
        }
        vec4 g = texture(uGen, p);
        float a = clamp(g.a, 0.0, 1.0) * uOpacity * m;
        if (uBlend < 0.5)      col = mix(base, g.rgb, a);                 // mix
        else if (uBlend < 1.5) col = base + g.rgb * a;                    // add
        else                   col = 1.0 - (1.0 - base) * (1.0 - g.rgb * a); // screen
    }
    col = mix(col, vec3(0.047, 0.027, 0.02), uScrim * 0.215);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
""".replace("//@ZONE", ZONE_GLSL)

# The canvas cut out of the effects pass, at its own size, for a shader
# that keeps to it: what it gets as iChannel0.
EXTRACT_FS = """#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uFx;
uniform mat3 uToScreen;        // canvas uv -> screen uv
void main() {
    vec3 q = uToScreen * vec3(vUV, 1.0);
    fragColor = texture(uFx, q.xy / max(q.z, 1e-4));
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
        # no vsync wait inside our own draw calls: with the interval at 1
        # the driver's queue fills and a draw call blocks until the display
        # takes a frame, several ms of the GUI thread gone each paint; the
        # 60 fps timer paces the paints, and the window is composed by the
        # desktop, so nothing tears
        fmt.setSwapInterval(0)
        super().__init__()
        self.setFormat(fmt)
        self.child = child
        self.state_fn = None
        self.current = None
        self.input = None                # live capture device while video in is on
        self._input_opts = None          # its capture mode
        self.source2 = None              # a second capture device (the webcam), kept open
        self.input2 = None               # its device string while it is on
        self._input2_opts = None
        self.enabled = True              # False: no decoding, no repaint, a black widget
        self.detached = False            # in a window of its own: no interface over it, no scrim
        self.second_on_b = False         # the second input is the B layer, in place of the take frames
        self._src_error2 = None
        self.videos_fn, self.default_video = videos_fn, default_video
        self._start_random = False
        self._shown = None               # Frame now on the video texture
        self._shown_ov = None            # Frame now on the take texture
        self._src_error = None           # last decoder error already logged
        self._frozen = False
        self._gl_ok = False
        self._t0 = time.monotonic()
        self._frame_n = 0
        self._last_t = self._t0
        self._t_acc = 0.0                # the effects' clock (uTime, iTime), in seconds
        self.time_rate = 1.0             # how fast it runs: the section's head (a tape stop
                                         # slows the strobes and the grain to a stop with it)
        self.log = []                    # compile errors etc., for the GUI
        self.frame_ms = 0.0              # smoothed CPU time of paintGL
        self.frames = 0                  # paints so far (the stats HUD reads the rate)
        # generative layer
        self.shader_path = None
        self.shader_mix = 0.5
        self.shader_blend = 0             # index into BLEND_MODES
        self.grade = {}                   # source grade: exposure, black, gamma, ...
        self.override = None              # a Frame from RAM (the section's take frame)
        self.override_ab = 0.0            # how far the picture is dissolved into it, 0..1
        self.override_blend = 0           # index into BLEND_MODES: how the two pictures meet
        self._gen_prog = None
        self._gen_pending = None          # path to compile on next paint
        self._gen_cache = {}              # path -> (mtime, program): compile once
        self._gen_min_opacity = {}        # path -> opacity a shader insists on
        self._prewarm = []                # shaders still to compile in idle time
        # the canvas: where the picture lies on the screen (canvas_quad),
        # what is around it, and which effects and shaders keep to it
        self.canvas = dict(CANVAS_DEFAULT)
        self.background = dict(BG_DEFAULT)
        self.keep_fx = set(KEEP_FX)       # the effects that keep to the canvas
        self.keep_shaders = set()         # the shader stems that keep to it
        self.source_bg = None             # the background clip's own decoder
        self._bg_clip = None              # its path while a clip is the background
        self._bg_img = None               # a background photo (QImage)
        self._shown_bg = None             # what is on the background texture
        self._src_error_bg = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(child)
        # video source: decoded on its own thread, polled from paintGL
        self.source = VideoSource(parent=self) if videos_fn() else None
        if self.source is not None:
            if default_video.exists():
                self.set_video(default_video)
            else:
                self.pick_random()
        self._timer = QTimer(self)                 # steady 60 fps repaint
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    # ---- video source (same behaviour as the numpy backdrop) ----
    def set_input(self, device, options=None):
        """Live video in: show a capture device (see video_inputs()) instead
        of the clips, at the capture mode `options` (a video_input_modes()
        entry, None = the device's default), and ignore clip changes until
        `None` switches back."""
        if self.source is None:
            return
        self.input, self._input_opts = device, options
        if self.enabled:
            self._open_main(random_start=True)

    def _open_main(self, random_start=False):
        """The first source shows what is chosen: the capture device, else
        the current clip."""
        if self.input:
            self.source.open(self.input, options=self._input_opts)
        elif self.current is not None:
            self.source.open(self.current, random_start)

    def set_second(self, device, options=None):
        """A second capture device (a webcam next to the VHS grabber) opened
        alongside the first. With `set_second_on_b` it is the B side of the
        A/B pedal, where the section's take frames go otherwise, so the
        pedal dissolves the tape into the webcam. None closes it; the first
        input is untouched."""
        self.input2, self._input2_opts = device, options
        if device:
            if self.source2 is None:
                self.source2 = VideoSource(parent=self)
            self._src_error2 = None
            if self.enabled:
                self.source2.open(device, options=options)
        elif self.source2 is not None:
            self.source2.close()

    def set_enabled(self, on):
        """Video off: the decoders let their file or device go (a capture
        device is free for other programs), the 60 fps repaint stops and
        the widget paints black once; the effects, the shader layer and the
        section's frames cost nothing until it is on again, when what was
        chosen (the input, the clip, the second input) reopens as it was."""
        on = bool(on)
        if on == self.enabled:
            return
        self.enabled = on
        if on:
            if self.source is not None:
                self._open_main(random_start=True)
            if self.input2 and self.source2 is not None:
                self.source2.open(self.input2, options=self._input2_opts)
            if self._bg_clip is not None and self.source_bg is not None:
                self.source_bg.open(self._bg_clip)
            self._timer.start(16)
        else:
            self._timer.stop()
            for s in (self.source, self.source2, self.source_bg):
                if s is not None:
                    s.close()
            self._shown = self._shown_ov = self._shown_bg = None
            self.update()                    # one paint: black

    def set_second_on_b(self, on):
        """On: the second input's picture is the B layer under the pedal
        (`override_ab`) whenever no take frame or burst is set as `override`.
        Off: B is the take frames again. Without a second device nothing
        changes."""
        self.second_on_b = bool(on)

    def set_background(self, mode, colour, path=""):
        """What is around the canvas: a colour (r, g, b in 0..1), or with
        mode 'file' a photo, or a clip looping on a decoder of its own at
        its own pace, untouched by the song."""
        self.background = {"mode": mode, "colour": tuple(colour), "path": str(path or "")}
        self._bg_img = None
        self._shown_bg = None
        clip = None
        if mode == "file" and path:
            p = Path(path)
            if p.suffix.lower() in IMAGE_SUFFIXES:
                img = QImage(str(p))
                if img.isNull():
                    self.log.append(f"background: cannot read {p.name}")
                else:
                    if img.height() > 2160:          # a photo straight off a camera
                        img = img.scaledToHeight(2160, Qt.SmoothTransformation)
                    self._bg_img = img.convertToFormat(QImage.Format_RGBA8888)
            elif p.is_file():
                clip = p
            else:
                self.log.append(f"background: no such file {p.name}")
        self._bg_clip = clip
        if clip is not None:
            if self.source_bg is None:
                self.source_bg = VideoSource(parent=self)
            self._src_error_bg = None
            if self.enabled:
                self.source_bg.open(clip)
        elif self.source_bg is not None:
            self.source_bg.close()

    def second_frame(self):
        """The second input's latest Frame while it stands in for the take
        frames, else None."""
        if self.second_on_b and self.input2 and self.source2 is not None:
            return self.source2.latest()
        return None

    def live_source(self):
        """The capture device behind the A picture, or None when a clip is
        (or the video is off: the device is closed then)."""
        src = self.source
        return src if (self.enabled and self.input and src is not None and src.is_live) else None

    def set_video(self, path, random_start=False):
        if self.source is None or self.input:
            return None
        self.current = Path(path)
        if self.enabled:
            self.source.open(self.current, random_start)
        return self.current

    def pick_random(self):
        files = self.videos_fn()
        if self.source is None or not files:
            return None
        files = [f for f in files if f != self.default_video] or files
        pool = [f for f in files if f != self.current] or files
        return self.set_video(random.choice(pool), random_start=True)

    def jump(self):
        if self.source is not None and not self.input:
            self.source.jump()

    # ---- generative layer ----
    def last_frame(self):
        """The most recently uploaded video frame (QImage) or None."""
        return getattr(self, "_frame_img", None)

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
        self._extract = self._program(EXTRACT_FS, "canvas cut-out")
        if not (self._fx and self._comp and self._extract):
            return
        self._vao = QOpenGLVertexArrayObject()   # core profile needs one bound,
        self._vao.create()                       # even with no attributes
        self._video_tex = None
        self._vid_size = (0, 0)
        self._take_tex = None
        self._take_size = (0, 0)
        self._bg_tex = None
        self._bg_size = (0, 0)
        self._fbo_fx = [None, None]
        self._fbo_gen = [None, None]      # the shader layer's pair: the screen's size,
        self._gen_size = None             # or the canvas's for a shader keeping to it
        self._fbo_zone = None             # the canvas cut out of the effects pass, then
        self._ping = 0
        self._has_prev = False
        self._fbo_size = None
        self._gl_ok = True
        # a second init (the widget moved to another window gets a new
        # context): the shaders compiled for the old one went with it, the
        # textures are uploaded again from the next frames
        self._gen_cache = {}
        self._gen_prog = None
        self._gen_pending = self.shader_path or None
        self._prewarm = list(shader_files())
        self._shown = self._shown_ov = self._shown_bg = None
        ctx.aboutToBeDestroyed.connect(self._cleanup_gl)

    def _cleanup_gl(self):
        """Free GL objects while the context is still current (on exit)."""
        self.makeCurrent()
        for name in ("_video_tex", "_take_tex", "_bg_tex"):
            tex = getattr(self, name, None)
            if tex is not None:
                tex.destroy()
            setattr(self, name, None)
        self._fbo_fx = [None, None]
        self._fbo_gen = [None, None]
        self._fbo_zone = None
        self._gen_size = None
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
        for fbo in self._fbo_fx:
            self._set_filters(fbo.texture())
        self._has_prev = False

    def _ensure_gen_fbos(self, size, zone):
        """The shader layer's pair at `size` (the screen's, or the canvas's
        own for a shader that keeps to it, plus the cut-out it reads then),
        cleared when made so a feedback shader starts from black."""
        if size != self._gen_size:
            self._gen_size = size
            self._fbo_gen = [QOpenGLFramebufferObject(*size) for _ in range(2)]
            self._fbo_zone = None
            for fbo in self._fbo_gen:
                self._set_filters(fbo.texture())
                self._clear_fbo(fbo)
        if zone and self._fbo_zone is None:
            self._fbo_zone = QOpenGLFramebufferObject(*size)
            self._set_filters(self._fbo_zone.texture())
            self._clear_fbo(self._fbo_zone)

    def _clear_fbo(self, fbo):
        fbo.bind()
        self.gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self.gl.glClear(GL_COLOR_BUFFER_BIT)

    def _set_filters(self, tex_id):
        gl = self.gl
        gl.glBindTexture(GL_TEXTURE_2D, tex_id)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def _upload_frame(self, img):
        """Newest decoded frame -> the persistent video texture (0.6 ms)."""
        self._video_tex, self._vid_size = self._upload(img, self._video_tex, self._vid_size)
        self._frame_img = img            # what last_frame() hands out: the live picture

    def _upload_take(self, img):
        """The section's take frame -> its own texture, dissolved in by uAB."""
        self._take_tex, self._take_size = self._upload(img, self._take_tex, self._take_size)

    def _upload_bg(self, img):
        """The background photo or the background clip's frame -> its texture."""
        self._bg_tex, self._bg_size = self._upload(img, self._bg_tex, self._bg_size)

    def _upload(self, img, tex, size):
        """A QImage into `tex` (reallocated when the size changes); returns
        the (texture, size) pair to keep."""
        if img.format() in (QImage.Format_ARGB32, QImage.Format_RGB32,
                            QImage.Format_ARGB32_Premultiplied):
            fmt = GL_BGRA
        else:
            img = img.convertToFormat(QImage.Format_RGBA8888)
            fmt = GL_RGBA
        w, h = img.width(), img.height()
        if img.bytesPerLine() != w * 4:
            img = img.copy()             # tight rows
        if tex is None or size != (w, h):
            if tex is not None:
                tex.destroy()
            tex = QOpenGLTexture(QOpenGLTexture.Target2D)
            tex.setFormat(QOpenGLTexture.RGBA8_UNorm)
            tex.setSize(w, h)
            tex.allocateStorage()
            tex.setMinMagFilters(QOpenGLTexture.Linear, QOpenGLTexture.Linear)
            tex.setWrapMode(QOpenGLTexture.ClampToEdge)
            size = (w, h)
        tex.bind()
        self.gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, fmt,
                                GL_UNSIGNED_BYTE, img.constBits())
        return tex, size

    def _take_scale(self, same, aspect):
        """The B texture's cover-crop into the canvas (its aspect `aspect`,
        the cropped video's): a 16:9 webcam over a 5:4 tape is cropped, not
        squeezed. (1, 1) when B is the video itself."""
        tw, th = self._take_size
        if same or tw <= 0 or th <= 0:
            return (1.0, 1.0)
        ta = tw / th
        return (aspect / ta, 1.0) if ta > aspect else (1.0, ta / aspect)

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
        if not self.enabled:                 # video off: the background, nothing else
            gl = self.gl
            gl.glClearColor(16 / 255, 9 / 255, 7 / 255, 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT)
            return
        t_start = time.perf_counter()
        self._paint()
        self.frame_ms = 0.9 * self.frame_ms + 0.1 * (time.perf_counter() - t_start) * 1000
        self.frames += 1

    def _paint(self):
        gl = self.gl
        st = (self.state_fn() if self.state_fn else None) or {}
        self._frozen = bool(st.get("repeat") or st.get("freeze"))
        src = self.source
        for s, attr, who in ((self.source, "_src_error", "video"),
                             (self.source2, "_src_error2", "video input 2"),
                             (self.source_bg, "_src_error_bg", "background")):
            if s is not None and s.error and s.error != getattr(self, attr):
                setattr(self, attr, s.error)
                self.log.append(f"{who}: {s.error}")
        # the B layer: a take frame or burst set by the window, else the
        # second input (the webcam) when it stands in for them
        ov = self.override
        if ov is None:
            ov = self.second_frame()
        if ov is not None and ov.img is not None and ov is not self._shown_ov \
                and not self._frozen:
            self._upload_take(ov.img)
            self._shown_ov = ov
        if src is not None:
            rev = st.get("reverse")              # a real rewind, any speed
            if "speed" in st:                    # the song's sample timeline
                src.set_speed(float(st["speed"]))
            else:
                src.set_speed(-float(rev) if rev else 1.0)
            fr = src.latest()
            if fr is not None and fr.img is not None and fr is not self._shown \
                    and not self._frozen:
                self._upload_frame(fr.img)
                self._shown = fr                 # keeps fr.arr alive with it
        # the background: a photo (uploaded once), a clip's frames as they
        # come, or nothing (the colour)
        bg = None
        if self._bg_img is not None:
            bg = self._bg_img
        elif self._bg_clip is not None and self.source_bg is not None:
            fr = self.source_bg.latest()
            bg = fr if (fr is not None and fr.img is not None) else None
        if bg is not None and bg is not self._shown_bg:
            self._upload_bg(bg if isinstance(bg, QImage) else bg.img)
            self._shown_bg = bg
        bg_on = bg is not None and self._bg_tex is not None
        # the picture's A/B: 0 = the live picture, 1 = the B layer (take frame
        # or webcam); with nothing on B, or one texture still missing, no dissolve
        vid, take = self._video_tex, self._take_tex
        ab = min(1.0, max(0.0, float(self.override_ab))) if ov is not None else 0.0
        if vid is None:
            vid, self._vid_size = take, self._take_size   # nothing live yet: the take alone
            ab = 1.0 if take is not None else 0.0
        if vid is None:
            gl.glClearColor(16 / 255, 9 / 255, 7 / 255, 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT)
            return
        if take is None:
            take, ab = vid, 0.0
        self._ensure_fbos()
        W, H = self._fbo_size
        now = time.monotonic()
        dt = now - self._last_t
        self._last_t = now
        self._t_acc += max(0.0, dt) * float(self.time_rate)
        t = self._t_acc
        self._frame_n += 1
        cur, prev = self._ping, 1 - self._ping

        # ---- pass 1: video + effects -> fbo_fx[cur] ----
        self._fbo_fx[cur].bind()
        gl.glViewport(0, 0, W, H)
        p = self._fx
        p.bind()
        self._bind_tex(0, vid.textureId())
        self._bind_tex(1, self._fbo_fx[prev].texture())
        self._bind_tex(2, take.textureId())
        p.setUniformValue1i("uVideo", 0)
        p.setUniformValue1i("uPrev", 1)
        p.setUniformValue1i("uTake", 2)
        p.setUniformValue1f("uAB", ab)
        p.setUniformValue1f("uABMode", float(self.override_blend))
        geo = canvas_geometry(self._vid_size, W, H, self.canvas)
        p.setUniformValue("uTakeScale", QVector2D(*self._take_scale(take is vid, geo["aspect"])))
        p.setUniformValue("uRes", QVector2D(W, H))
        p.setUniformValue("uVidRes", QVector2D(*self._vid_size))
        p.setUniformValue1f("uTime", t)
        p.setUniformValue1f("uHasPrev", 1.0 if self._has_prev else 0.0)
        # the canvas: its place, the crop, the background, who keeps to it
        p.setUniformValue("uToZone", _mat3(geo["to_zone"]))
        p.setUniformValue("uZoneK", QVector2D(*geo["k"]))
        p.setUniformValue1f("uFeather", float(self.canvas.get("feather", 0.0)))
        p.setUniformValue1f("uPicBlend", float(self.canvas.get("blend", 0)))
        p.setUniformValue1f("uPicMix", float(self.canvas.get("mix", 1.0)))
        l, r, tp, bt = self.canvas.get("crop", CANVAS_DEFAULT["crop"])
        p.setUniformValue("uCrop", float(l), float(r), float(tp), float(bt))
        self._bind_tex(3, self._bg_tex.textureId() if bg_on else 0)
        p.setUniformValue1i("uBg", 3)
        p.setUniformValue1f("uBgOn", 1.0 if bg_on else 0.0)
        p.setUniformValue("uBgRes", QVector2D(*(self._bg_size if bg_on else (1, 1))))
        cr, cg, cb = self.background.get("colour", BG_DEFAULT["colour"])
        p.setUniformValue("uBgColor", float(cr), float(cg), float(cb))
        p.setUniformValue1i("uKeep", sum(1 << i for i, k in enumerate(KEEP_FX)
                                         if k in self.keep_fx))
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
        # a shader that keeps to the canvas is drawn at the canvas's own
        # size over the canvas cut out of the effects pass, and laid back
        # into it by the composite; the others see the whole screen
        zone = gen_on and self.shader_path is not None \
            and self.shader_path.stem in self.keep_shaders
        if gen_on:
            gw, gh = geo["px"] if zone else (W, H)
            self._ensure_gen_fbos((gw, gh), zone)
            if zone:
                self._fbo_zone.bind()
                gl.glViewport(0, 0, gw, gh)
                x = self._extract
                x.bind()
                self._bind_tex(0, self._fbo_fx[cur].texture())
                x.setUniformValue1i("uFx", 0)
                x.setUniformValue("uToScreen", _mat3(geo["to_screen"]))
                self._draw()
            self._fbo_gen[cur].bind()
            gl.glViewport(0, 0, gw, gh)
            g = self._gen_prog
            g.bind()
            self._bind_tex(0, (self._fbo_zone if zone else self._fbo_fx[cur]).texture())
            self._bind_tex(1, self._fbo_gen[prev].texture())
            g.setUniformValue1i("iChannel0", 0)
            g.setUniformValue1i("iChannel1", 1)
            g.setUniformValue("iResolution", float(gw), float(gh), 1.0)
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
        self._bind_tex(1, self._fbo_gen[cur].texture() if self._fbo_gen[cur] else 0)
        c.setUniformValue1i("uFx", 0)
        c.setUniformValue1i("uGen", 1)
        c.setUniformValue1f("uGenOn", 1.0 if gen_on else 0.0)
        c.setUniformValue1f("uGenZone", 1.0 if zone else 0.0)
        c.setUniformValue("uToZone", _mat3(geo["to_zone"]))
        c.setUniformValue("uZoneK", QVector2D(*geo["k"]))
        c.setUniformValue1f("uFeather", float(self.canvas.get("feather", 0.0)))
        c.setUniformValue("uRes", QVector2D(W, H))
        c.setUniformValue1f("uOpacity", opacity)
        c.setUniformValue1f("uBlend", float(self.shader_blend))
        c.setUniformValue1f("uScrim", 1.0 if (self.child.isVisible() and not self.detached)
                            else 0.0)
        self._draw()
        self._has_prev = True
        self._ping = prev

    def _set_grade_uniforms(self, p, st):
        """The source grade, plus its very slight breathing with the music
        when `react` is up: exposure and saturation a few percent with the
        loudness, warmth a touch toward the root note's colour (warm notes
        on the red side of the hue circle, cool ones on the cyan side)."""
        gr = self.grade
        f = p.setUniformValue1f
        react = float(gr.get("react", 0.0))
        expo = float(gr.get("exposure", 1.0))
        sat = float(gr.get("saturation", 1.0))
        warm = float(gr.get("warmth", 0.0))
        m = st.get("music") if react > 0 else None
        if m:
            loud = float(m.get("loud", 0.5)) - 0.5             # -0.5 .. 0.5
            expo *= 1.0 + 0.08 * react * loud                   # +-4 % at most
            sat *= 1.0 + 0.12 * react * loud                    # +-6 % at most
            root = int(m.get("root", -1))
            if root >= 0:
                warm += 0.15 * react * math.cos(2 * math.pi * root / 12.0)
        f("uGExposure", expo)
        f("uGBlack", float(gr.get("black", 0.0)))
        f("uGGamma", float(gr.get("gamma", 1.0)))
        f("uGContrast", float(gr.get("contrast", 1.0)))
        f("uGSat", sat)
        f("uGWarm", warm)
        f("uGTint", float(gr.get("tint", 0.0)))
        f("uGSharpen", float(gr.get("sharpen", 0.0)))

    def _set_fx_uniforms(self, p, st):
        self._set_grade_uniforms(p, st)
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
        mono = g("mono")                    # (tones, threshold); tones 0 = smooth gray
        f("uMonoTones", float(mono[0] or 1) if mono else 0.0)   # 0 here = mono off
        f("uMonoThr", float(mono[1]) if mono else 0.5)
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
        gr = g("gran")
        f("uGran", float(gr["amt"]) if gr else 0.0)
        heads = (gr.get("heads") or [])[:16] if gr else []
        p.setUniformValue1i("uGranN", len(heads))
        for i, (x0, w, hx, lit) in enumerate(heads):
            p.setUniformValue(f"uGranHeads[{i}]", float(x0), float(w), float(hx), float(lit))
