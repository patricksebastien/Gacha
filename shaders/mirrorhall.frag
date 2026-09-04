// gacha: opacity=1.0
// gacha shader: mirror wall. The frame is split into full-height panels
// side by side, each showing the video at its true aspect but cropped to a
// different horizontal slice, so the wall tiles the whole screen. Nothing
// moves within a bar; the slices reshuffle every bar, the panel count follows
// the loudness, panels light up one by one
// with the beat, a drop lights them all.
float hash(float n) { return fract(sin(n * 127.1) * 43758.5453); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float n = 4.0 + floor(iLoud * 4.0);                  // 4 to 8 panels
    float k = floor(uv.x * n);                           // which panel
    float local = fract(uv.x * n);                       // 0..1 across it
    float panelAspect = (iResolution.x / iResolution.y) / n;   // panel w/h
    // the video at true aspect fills the panel's height: only a horizontal
    // slice of width panelAspect / videoAspect fits; each panel takes a
    // different slice so the wall reads as fragments of one picture
    float videoAspect = iResolution.x / iResolution.y;   // iChannel0 fills the frame
    float sliceW = panelAspect / videoAspect;
    // which slice each panel shows is reshuffled on every bar, so the
    // picture keeps rearranging itself with the music
    float barCount = floor(iTime * iBPM / 240.0);
    float sliceX = mix(0.0, 1.0 - sliceW, hash(k + 3.7 + barCount * 0.61));
    vec2 q = vec2(sliceX + local * sliceW, uv.y);
    vec3 col = texture(iChannel0, q).rgb;
    // one panel lights on each beat, the others sit a little darker
    float beatCount = floor(iTime * iBPM / 60.0);
    float lit = step(0.5, 1.0 - abs(mod(beatCount, n) - k)) * (1.0 - iBeat);
    col *= 0.5 + 0.8 * lit + 0.5 * iDrop;               // stronger lit/unlit contrast
    fragColor = vec4(col, 1.0);
}
