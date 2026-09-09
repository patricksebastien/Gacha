// gacha shader: Super 8 home movie. The picture is projected film: it only
// changes 18 times a second (frames are held from iChannel1 in between),
// is graded like faded Kodachrome with red halation round the highlights,
// and carries grain, dust, hairs and dark scratches that come and go. It
// fills the frame: no gate, no border, no vignette, so it mixes with the
// picture at any opacity. Exposure flickers frame to frame, more when
// loud, and the lamp browns and shivers under a swell. Downbeats sometimes
// land on a splice (one dark frame with the tape line across it) or on a
// camera start, a couple of overexposed orange flash frames. A drop is the
// end of the reel: the film slips in the gate, the frame line crosses the
// picture and the tail burns orange and white. The reel gets dustier and
// more scratched as the song goes on (iSongPos).
const float FPS = 18.0;

// hashes that stay random for large inputs (the sine trick falls apart once
// iTime is in the thousands, which on stage it is)
float hash(float n) { n = fract(n * 0.1031); n *= n + 33.33; n *= n + n; return fract(n); }
float hash2(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// smooth value noise, for grain that reads as blobs rather than pixels
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2(1, 0)), f.x),
               mix(hash2(i + vec2(0, 1)), hash2(i + vec2(1, 1)), f.x), f.y);
}

// faded reversal stock: warm, lifted blacks, creamy whites, olive greens
vec3 filmGrade(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    // channel mixer: reds warm, greens toward olive, blues a touch teal
    c = vec3(dot(c, vec3(1.05, 0.03, -0.03)),
             dot(c, vec3(0.07, 0.86, 0.02)),
             dot(c, vec3(0.02, 0.05, 0.86)));
    float l = luma(c);
    c = l + (c - l) * 1.3;                          // dense colour
    c = mix(c, c * c * (3.0 - 2.0 * c), 0.6);       // gentle S-curve
    vec3 black = vec3(0.07, 0.048, 0.042);          // faded warm blacks
    vec3 white = vec3(1.0, 0.96, 0.88);             // creamy highlights
    return black + c * (white - black);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float asp = iResolution.x / iResolution.y;

    // ---- 18 fps: hold the last developed frame until the next one is due
    float ff = floor(iTime * FPS);
    float dt = iTimeDelta;
    bool fresh = dt <= 0.0 || dt > 0.5 || floor((iTime - dt) * FPS) != ff;
    if (!fresh) { fragColor = texture(iChannel1, uv); return; }
    ff = mod(ff, 8192.0);                            // keep the seeds small

    float wear = 0.6 + 0.8 * iSongPos;
    float beatPulse = pow(1.0 - iBeat, 6.0);
    float barLen = 240.0 / max(iBPM, 30.0);
    float barCount = floor(iTime / barLen);
    float frameInBar = floor(iBar * barLen * FPS);

    vec2 suv = uv;

    // ---- drop: the loop is lost, the film slips and the frame line shows
    float slip = smoothstep(0.15, 0.9, iDrop);
    float slipY = suv.y + slip * (0.35 + 0.1 * hash(ff));
    float frameLine = 0.0;
    if (slip > 0.001) {
        float fy = fract(slipY);
        frameLine = smoothstep(0.014, 0.004, min(fy, 1.0 - fy)) * slip;
        suv.y = fy;
    }

    // ---- soft lens: a small blur, softer toward the edges
    vec2 p = (uv - 0.5) * vec2(asp, 1.0);
    float edge = smoothstep(0.25, 0.85, length(p));
    float rs = (1.6 + 3.5 * edge) / iResolution.y;
    vec3 col = texture(iChannel0, suv).rgb * 0.36;
    col += texture(iChannel0, suv + vec2( rs,  rs)).rgb * 0.16;
    col += texture(iChannel0, suv + vec2(-rs,  rs)).rgb * 0.16;
    col += texture(iChannel0, suv + vec2( rs, -rs)).rgb * 0.16;
    col += texture(iChannel0, suv + vec2(-rs, -rs)).rgb * 0.16;

    // ---- halation: highlights bleed a red-orange halo into their surroundings
    float hr = 0.011;
    float hal = 0.0;
    for (int i = 0; i < 8; i++) {
        float a = float(i) * 0.7854;
        vec3 s = texture(iChannel0, suv + vec2(cos(a) / asp, sin(a)) * hr).rgb;
        hal += max(luma(s) - 0.55, 0.0);
    }
    hal /= 8.0 * 0.45;
    col += vec3(1.0, 0.42, 0.12) * hal * 0.4;

    col = filmGrade(col);

    // a break is an older, more faded shot
    if (iSection == 2) {
        float l = luma(col);
        col = mix(col, vec3(l) * vec3(1.0, 0.94, 0.86), 0.45);
        col = mix(col, vec3(0.5), 0.15);
    }

    // ---- exposure flicker: every frame a little different, the beat and
    // the loudness stir it, a swell makes the lamp shiver and brown
    float flick = (hash(ff * 0.71 + 3.3) - 0.5) * (0.10 + 0.14 * iLoud + 0.6 * iSwell);
    flick += (hash(ff * 0.29 + 9.1) - 0.5) * 0.12 * beatPulse * iLoud;
    flick += sin(ff * 0.8) * 0.015;
    col *= 1.0 + flick;
    col *= mix(vec3(1.0), vec3(1.0, 0.78, 0.5), iSwell * 0.45) * (1.0 - 0.3 * iSwell);

    // ---- dust: specks per frame, dark mostly, a few bright; more just
    // after a beat and as the reel wears
    {
        vec2 cells = vec2(14.0, 8.0);
        vec2 cid = floor(uv * cells);
        vec2 seed = cid + ff * vec2(17.13, 31.71);
        float prob = (0.05 + 0.10 * beatPulse * iLoud) * wear;
        if (hash2(seed) < prob) {
            vec2 c = (cid + 0.2 + 0.6 * vec2(hash2(seed + 1.7), hash2(seed + 2.9))) / cells;
            float rad = (0.0009 + 0.003 * pow(hash2(seed + 4.1), 2.0));
            float d = length((uv - c) * vec2(asp, 1.0));
            float k = smoothstep(rad, rad * 0.4, d);
            float bright = step(0.72, hash2(seed + 6.3));
            col = mix(col, bright > 0.5 ? vec3(0.95, 0.93, 0.88) : vec3(0.03, 0.02, 0.02),
                      k * (0.6 + 0.35 * hash2(seed + 7.7)));
        }
    }

    // ---- hairs: a bent fibre caught in the gate for a few frames, wobbling
    for (int h = 0; h < 2; h++) {
        float life = 2.0 + floor(3.0 * hash(float(h) * 3.1 + 0.4));
        float s = floor(ff / life) * 7.0 + float(h) * 101.0;
        float prob = (0.14 + 0.2 * beatPulse * iLoud) * wear;
        if (hash(s + 0.13) > prob) continue;
        vec2 a = vec2(0.1 + 0.8 * hash(s + 1.1), 0.1 + 0.8 * hash(s + 2.2));
        a += (vec2(hash(ff * 0.9 + s), hash(ff * 1.1 + s + 0.5)) - 0.5) * 0.004;
        float ang = hash(s + 3.3) * 6.2832;
        vec2 dir = vec2(cos(ang), sin(ang)), nrm = vec2(-dir.y, dir.x);
        float len = 0.04 + 0.16 * hash(s + 4.4);
        float curv = (hash(s + 5.5) - 0.5) * 6.0;
        vec2 d = (uv - a) * vec2(asp, 1.0);
        float t = dot(d, dir), n = dot(d, nrm) - curv * t * t;
        float w = 1.1 / iResolution.y;
        float k = smoothstep(w * 1.6, w * 0.3, abs(n)) * smoothstep(len, len * 0.7, abs(t));
        col = mix(col, vec3(0.02, 0.015, 0.01), k * 0.85);
    }

    // ---- scratches: lines running the length of the film, living for a
    // while, wandering a little, brighter and darker along their way
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float life = 5.0 + floor(28.0 * hash(fi * 2.7 + 0.9));
        float s = floor(ff / life) * 13.0 + fi * 57.0;
        float thr = mix(0.82, 0.55, iSongPos);
        if (hash(s + 0.5) < thr) continue;
        float x = hash(s + 1.5);
        x += sin(ff * (0.3 + 0.4 * hash(s + 2.5)) + fi) * 0.004 + (hash(ff + s) - 0.5) * 0.002;
        float w = (0.7 + 1.4 * hash(s + 3.5)) / iResolution.x;
        float k = smoothstep(w * 1.5, w * 0.2, abs(uv.x - x));
        float along = 0.3 + 0.7 * vnoise(vec2(uv.y * 30.0, ff * 0.5 + s));
        along *= smoothstep(0.0, 0.15, vnoise(vec2(uv.y * 4.0 + s, ff * 0.2)) - 0.25 + 0.4 * hash(s + 4.5));
        // scratches are dark grey, a little darker or lighter each
        vec3 sc = vec3(0.18 + 0.14 * hash(s + 6.5));
        col = mix(col, sc, k * along * 0.85);
    }

    // ---- grain: soft blobs a few pixels wide, per frame, heavier in the
    // shadows and when loud, with a little colour speckle
    {
        float gsz = max(1.0, iResolution.y / 330.0);
        vec2 gp = fragCoord / gsz + ff * vec2(37.7, 91.3);
        // two octaves, the second rotated, so the noise grid never shows
        float g = vnoise(gp) - 0.5;
        g += (vnoise(mat2(0.8, 0.6, -0.6, 0.8) * gp * 1.9 + 17.0) - 0.5) * 0.6;
        vec3 gc = vec3(hash2(gp + 0.3), hash2(gp + 1.3), hash2(gp + 2.3)) - 0.5;
        float l = luma(col);
        float amt = (0.08 + 0.07 * iLoud) * (0.45 + 0.55 * (1.0 - l)) * (0.8 + 0.3 * iSongPos);
        col += (g * 1.4 + gc * 0.3) * amt;
    }

    // ---- downbeat events: a splice, or a camera start with its flash frames
    float ev = hash(barCount * 0.913 + 0.37);
    bool splice = ev > 0.72, camStart = ev > 0.52;
    if (splice && frameInBar < 1.0) {
        // one dark frame with the tape line across it
        col *= 0.1;
        float y0 = 0.3 + 0.4 * hash(barCount + 0.7);
        float ly = y0 + (uv.x - 0.5) * 0.03 * (hash(barCount + 0.9) - 0.5);
        float dl = abs(uv.y - ly);
        col += vec3(0.85, 0.82, 0.75) * smoothstep(0.008, 0.002, dl);
        col *= 1.0 - 0.7 * smoothstep(0.012, 0.008, dl) * smoothstep(0.004, 0.008, dl);
    } else if (camStart) {
        // the first frames after a cut are overexposed and orange from the
        // shutter opening slowly
        float fr = frameInBar - (splice ? 1.0 : 0.0);
        float flare = exp(-max(fr, 0.0) * 0.8) * step(fr, 3.5);
        vec3 fc = vec3(1.0, 0.7, 0.4);
        col = mix(col, fc, flare * 0.55) + fc * flare * flare * 0.6;
    }

    // ---- the frame line of a slipped frame is dark
    col = mix(col, vec3(0.02, 0.015, 0.01), frameLine);

    // ---- drop: the reel runs out, the tail burns orange then white from a
    // corner and the whole picture blows out for an instant
    if (iDrop > 0.001) {
        vec2 bc = vec2(0.5 * asp + 0.1, -0.6);            // bottom right corner
        float bd = length(p - bc);
        float R = 2.2 * pow(iDrop, 1.5);
        float burn = smoothstep(R + 0.4, R - 0.3, bd) * (0.5 + 0.5 * iDrop);
        burn *= 0.8 + 0.4 * vnoise(p * 6.0 + ff * 0.7);      // the edge of the burn boils
        vec3 orange = vec3(1.0, 0.42, 0.06), hot = vec3(1.0, 0.9, 0.7);
        col = mix(col, orange, burn * 0.9);
        col = mix(col, hot, burn * burn * 0.7);
        col += hot * pow(iDrop, 10.0);
    }

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
