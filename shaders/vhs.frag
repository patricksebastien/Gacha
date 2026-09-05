// gacha shader: worn VHS tape, over the whole frame. A tracking band rolls
// up the full height once every couple of bars, the picture rolls and
// wobbles top to bottom, chroma bleeds, dropouts hit on the beat, a noisy
// head-switching band sits at the bottom. The tape wears out as the song
// goes on (iSongPos).
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float wear = 0.3 + 0.7 * iSongPos;
    float beatCount = floor(iTime * iBPM / 60.0);
    // vertical roll: the whole frame slips up and down a little, more when
    // loud, and jumps on the beat now and then
    float roll = sin(iTime * 0.7) * 0.006 * wear * (0.5 + iLoud);
    roll += step(0.8, hash(vec2(beatCount, 3.0))) * (1.0 - iBeat) * 0.03 * wear;
    uv.y = fract(uv.y + roll);
    // tracking band: a bright, torn strip that travels the full height,
    // bottom to top, every 2 bars (top to bottom in the video)
    float bandPos = fract(iTime * iBPM / 60.0 / 8.0);
    float band = smoothstep(0.06, 0.0, abs(uv.y - bandPos));
    uv.x += band * (hash(vec2(floor(uv.y * 300.0), floor(iTime * 20.0))) - 0.5) * 0.25 * wear;
    uv.x += band * 0.02;
    // tracking wobble over the whole height, in several waves
    uv.x += sin(uv.y * 40.0 + iTime * 3.0) * 0.003 * wear * (0.5 + iLoud);
    uv.x += sin(uv.y * 3.0 - iTime * 0.8) * 0.004 * wear;
    // head-switching noise at the bottom
    if (uv.y < 0.05) uv.x += (hash(vec2(floor(uv.y * 200.0), floor(iTime * 30.0))) - 0.5) * 0.15;
    // chroma bleed: colour smeared sideways, luma sharp
    vec3 c = texture(iChannel0, uv).rgb;
    float bleed = 0.006 + 0.01 * band;
    vec3 blur = (texture(iChannel0, uv + vec2(bleed, 0.0)).rgb + texture(iChannel0, uv - vec2(bleed, 0.0)).rgb) * 0.5;
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    vec3 col = blur - dot(blur, vec3(0.299, 0.587, 0.114)) + l;
    col.r = texture(iChannel0, uv + vec2(0.004 * wear, 0.0)).r * 0.5 + col.r * 0.5;
    // dropouts: white streaks over the whole height, more on the beat and when loud
    float row = floor(uv.y * 240.0);
    float tick = floor(iTime * 24.0);
    float drop = step(0.995 - 0.03 * (1.0 - iBeat) * iLoud * wear - 0.1 * band, hash(vec2(row, tick)));
    // a dropout is a short streak, not a full line: random start and length
    float x0 = hash(vec2(row, tick + 0.5)), len = 0.05 + 0.3 * hash(vec2(tick, row));
    drop *= step(x0, uv.x) * step(uv.x, x0 + len);
    col = mix(col, vec3(0.9), drop);
    // the band itself: brighter, noisier, washed out
    col += band * (hash(fragCoord * 0.5 + iTime * 60.0) - 0.3) * 0.5 * wear;
    col = mix(col, vec3(dot(col, vec3(0.299, 0.587, 0.114))) * 1.2, band * 0.6);
    // grain, desaturation, scanlines
    col += (hash(fragCoord + fract(iTime) * 100.0) - 0.5) * 0.12 * wear;
    col = mix(col, vec3(dot(col, vec3(0.299, 0.587, 0.114))), 0.25 * wear);
    if (mod(fragCoord.y, 2.0) < 1.0) col *= 0.88;
    fragColor = vec4(col, 1.0);
}
