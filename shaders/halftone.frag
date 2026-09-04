// gacha shader: CMYK halftone print. Three rotated dot screens, cyan,
// magenta and yellow, with dot size from each ink's coverage, multiplied on
// paper. The screens rotate slowly, the paper tone comes from the root
// note, a drop prints the negative.
vec3 hue2rgb(float h) {
    return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
}
float screen(vec2 uv, float angle, float cells, float ink) {
    float c = cos(angle), s = sin(angle);
    vec2 p = mat2(c, -s, s, c) * (uv * vec2(iResolution.x / iResolution.y, 1.0)) * cells;
    vec2 f = fract(p) - 0.5;
    float r = sqrt(clamp(ink, 0.0, 1.0)) * 0.72;
    return smoothstep(r, r - 0.12, length(f));       // 1 inside the dot
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 vid = texture(iChannel0, uv).rgb;
    if (iDrop > 0.5) vid = 1.0 - vid;
    vec3 ink = 1.0 - vid;                              // CMY coverage
    float cells = 60.0 + 30.0 * iLoud;
    float spin = iTime * 0.02;
    float c = screen(uv, 0.26 + spin, cells, ink.r);
    float m = screen(uv, 1.31 + spin, cells, ink.g);
    float y = screen(uv, 0.0 + spin, cells, ink.b);
    float hue = iRoot < 0 ? 0.1 : float(iRoot) / 12.0;
    vec3 paper = mix(vec3(0.93, 0.9, 0.84), hue2rgb(hue), 0.12);
    vec3 col = paper;
    col *= mix(vec3(1.0), vec3(0.0, 0.62, 0.9), c);    // cyan ink
    col *= mix(vec3(1.0), vec3(0.93, 0.0, 0.55), m);   // magenta ink
    col *= mix(vec3(1.0), vec3(1.0, 0.9, 0.0), y);     // yellow ink
    fragColor = vec4(col, 1.0);
}
