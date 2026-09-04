// gacha shader: a slow plasma that borrows the video's own colours. The
// fractal field warps the picture and modulates its brightness; where the
// video is dark the frame's average colour fills in, so the plasma never
// invents a palette of its own. It breathes with the beat.
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = fract(sin(dot(i, vec2(12.9898, 78.233))) * 43758.5453);
    float b = fract(sin(dot(i + vec2(1, 0), vec2(12.9898, 78.233))) * 43758.5453);
    float c = fract(sin(dot(i + vec2(0, 1), vec2(12.9898, 78.233))) * 43758.5453);
    float d = fract(sin(dot(i + vec2(1, 1), vec2(12.9898, 78.233))) * 43758.5453);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, amp = 0.5;
    for (int i = 0; i < 5; i++) { v += amp * noise(p); p = p * 2.03 + 17.1; amp *= 0.5; }
    return v;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 p = uv * vec2(iResolution.x / iResolution.y, 1.0) * 2.0;
    float t = iTime * 0.15 + iBar * 0.2;
    float n = fbm(p + vec2(t, -t * 0.7) + fbm(p * 0.7 - t) * 1.5);
    // the frame's average colour, from a coarse 4x4 grid of taps
    vec3 avg = vec3(0.0);
    for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
        avg += texture(iChannel0, (vec2(float(x), float(y)) + 0.5) / 4.0).rgb;
    avg /= 16.0;
    // the video, gently warped by the field
    vec3 vid = texture(iChannel0, uv + (vec2(n, fbm(p + 9.0)) - 0.5) * 0.08).rgb;
    float l = dot(vid, vec3(0.299, 0.587, 0.114));
    // only the really dark pixels borrow the average colour, and faintly
    vec3 base = mix(avg * 0.8, vid, smoothstep(0.02, 0.25, l));
    float pattern = smoothstep(0.25, 0.85, n);                    // crisper cells
    float breathe = 0.5 + 0.5 * pow(1.0 - iBeat, 2.0) * iLoud;
    vec3 col = base * (0.35 + 1.4 * pattern * (0.7 + breathe));
    col += vid * smoothstep(0.6, 1.0, l) * 0.6;                   // highlights punch through
    fragColor = vec4(col, 0.8);
}
