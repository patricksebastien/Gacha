// gacha shader: bloom. The bright parts of the video bleed into a soft
// glow that gets wider and hotter with the loudness, tightens on each beat,
// lingers through the previous frame, and flares white on a drop. Colours
// stay the video's own.
vec3 bright(vec2 uv, float thr) {
    vec3 c = texture(iChannel0, uv).rgb;
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    return c * smoothstep(thr, thr + 0.25, l);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 px = vec2(1.0 / iResolution.y);                 // square taps
    float thr = 0.6 - 0.3 * iLoud;                        // louder: more of the frame glows
    float radius = (0.02 + 0.05 * iLoud) * (0.7 + 0.5 * (1.0 - iBeat));   // tight on the beat
    vec3 glow = vec3(0.0);
    float wsum = 0.0;
    // spiral rotated by a per-pixel random angle: no stamped copies, the
    // leftover noise is averaged out by the temporal smoothing below
    float rot = fract(sin(dot(fragCoord, vec2(12.9898, 78.233)) + iTime) * 43758.5453) * 6.2831853;
    const int N = 32;
    for (int i = 0; i < N; i++) {
        float f = (float(i) + 0.5) / float(N);
        float a = float(i) * 2.39996 + rot, r = sqrt(f) * radius;
        float wgt = 1.0 - f * 0.7;                        // softer toward the rim
        glow += bright(uv + vec2(cos(a), sin(a)) * r, thr) * wgt;
        wsum += wgt;
    }
    glow /= wsum;
    glow *= 0.8 + 1.25 * iDrop;                           // half strength: soft, not white
    // temporal smoothing: blend with the previous frame's glow
    vec3 prev = max(texture(iChannel1, uv).rgb - texture(iChannel0, uv).rgb, 0.0);
    glow = mix(glow, prev, 0.6);
    vec3 vid = texture(iChannel0, uv).rgb;
    fragColor = vec4(vid + glow, 1.0);
}
