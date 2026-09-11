// gacha shader: strips. Like lanes, but the lane is a real piece of the
// picture: a vertical strip about 100 px wide and the full height, repeated
// side by side across the screen, so one slice of the frame tiles it in
// vertical columns. A new random strip on every beat, drifting a little
// until the next; wider when loud (about 60 to 160 px), and a drop opens it
// up to a third of the screen. Replaces the picture, so it pins its opacity.
// gacha: opacity=1.0
float hash11(float p) { return fract(sin(p * 127.1) * 43758.5453); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float beat = iTime * iBPM / 60.0;            // beats so far, fractional
    float n = floor(beat);
    // the strip's width in pixels, then where it is cut from the frame
    float w = 60.0 + 100.0 * iLoud;
    w = mix(w, iResolution.x / 3.0, iDrop);
    float left = hash11(n + 0.37);
    left += (hash11(n + 5.1) - 0.5) * 0.08 * fract(beat);
    left = clamp(left, 0.0, 1.0) * (iResolution.x - w);
    float x = left + mod(fragCoord.x, w);        // this pixel's place in the strip
    vec2 uv = vec2(x / iResolution.x, fragCoord.y / iResolution.y);
    vec3 col = texture(iChannel0, uv).rgb;
    fragColor = vec4(col, 1.0);
}
