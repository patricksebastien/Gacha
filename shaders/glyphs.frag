// gacha shader: the picture rebuilt from a small set of procedural glyphs,
// brightness picking a denser glyph. Glyphs refresh on every beat, the
// loudness sets their size (tiny when quiet, four times bigger when loud),
// colours stay the video's own.
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float glyph(int id, vec2 f) {                          // f in 0..1 inside the cell
    vec2 c = abs(f - 0.5);
    if (id == 0) return 0.0;                                            // space
    if (id == 1) return step(max(c.x, c.y), 0.12) > 0.5 ? 1.0 : 0.0;   // dot
    if (id == 2) return (c.y < 0.08 && c.x < 0.35) ? 1.0 : 0.0;        // dash
    if (id == 3) return ((c.y < 0.08 && c.x < 0.35) || (c.x < 0.08 && c.y < 0.35)) ? 1.0 : 0.0; // plus
    if (id == 4) return (max(c.x, c.y) < 0.36 && max(c.x, c.y) > 0.24) ? 1.0 : 0.0; // box
    if (id == 5) return ((c.y < 0.08 && c.x < 0.35) || (c.x < 0.08 && c.y < 0.35) || abs(c.x - c.y) < 0.07 && c.x < 0.3) ? 1.0 : 0.0; // star
    return max(c.x, c.y) < 0.38 ? 1.0 : 0.0;                            // block
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float size = 0.5 + 1.5 * iLoud;                    // glyph size x0.5 quiet .. x2 loud
    float cols = 200.0 / size;                         // 400 columns down to 100
    vec2 grid = vec2(cols, cols / (iResolution.x / iResolution.y));
    vec2 cell = floor(uv * grid), f = fract(uv * grid);
    vec3 vc = texture(iChannel0, (cell + 0.5) / grid).rgb;
    float l = dot(vc, vec3(0.299, 0.587, 0.114));
    float beatCount = floor(iTime * iBPM / 60.0);
    float jitter = (hash(cell + beatCount) - 0.5) * 0.15;   // glyph choice wobbles per beat
    int id = int(clamp(floor((l + jitter) * 7.0), 0.0, 6.0));
    float g = glyph(id, f);
    vec3 col = vc * (0.6 + 0.9 * l) * g * 1.4;
    col += vc * 0.05;                                    // faint ghost of the video
    fragColor = vec4(col, 1.0);
}
