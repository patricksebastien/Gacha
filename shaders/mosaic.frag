// gacha shader: the video cut into tiles, ~40 when quiet up to ~200 when loud.
// Each beat a different set of tiles
// flips, brightens or shows the frame from a shifted position; a drop
// scatters all of them.
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    // loudness sets the tile count: ~40 tiles when quiet, ~200 when loud,
    // in six steps so the grid snaps instead of jittering
    float aspect = iResolution.x / iResolution.y;
    float level = floor(clamp(iLoud, 0.0, 0.999) * 6.0) / 5.0;    // 0, .2 .. 1
    float total = 40.0 + 160.0 * level * level;
    float rows = max(1.0, floor(sqrt(total / aspect) + 0.5));
    float cols = max(2.0, floor(rows * aspect + 0.5));
    vec2 grid = vec2(cols, rows);
    vec2 cell = floor(uv * grid);
    float beatCount = floor(iTime * iBPM / 60.0);
    float h = hash(cell + beatCount * 0.37);
    vec2 local = fract(uv * grid);
    vec2 src = uv;
    if (h < 0.25 + iDrop) {                              // shifted tile
        src = (cell + vec2(hash(cell + 3.1), hash(cell + 7.3)) * 2.0 - 1.0) / grid + local / grid;
        src = clamp(src, 0.0, 1.0);
    } else if (h < 0.40) {                               // mirrored tile
        src = (cell + vec2(1.0 - local.x, local.y)) / grid;
    }
    vec3 col = texture(iChannel0, src).rgb;
    float flash = step(0.85, hash(cell + beatCount)) * (1.0 - iBeat) * (0.5 + iLoud);
    col = col * (1.0 + 0.7 * flash) + 0.06 * flash;      // brighten the tile's own colour, a hint of lift
    fragColor = vec4(col, 1.0);
}
