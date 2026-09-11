// gacha shader: lanes. One narrow vertical lane of the picture, a few
// pixels wide and the full height, repeated side by side across the whole
// screen: the frame becomes a barcode of one of its own columns. A new lane
// is drawn at random on every beat and drifts slowly until the next; its
// width breathes with the loudness; a drop widens it into broad stripes for
// a moment. Replaces the picture, so it pins its opacity.
// gacha: opacity=1.0
float hash11(float p) { return fract(sin(p * 127.1) * 43758.5453); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float beat = iTime * iBPM / 60.0;            // beats so far, fractional
    float n = floor(beat);
    // the lane: a random column per beat, drifting a little inside the beat
    float lane = hash11(n + 0.37);
    lane += (hash11(n + 5.1) - 0.5) * 0.06 * fract(beat);
    lane = clamp(lane, 0.0, 1.0) * iResolution.x;
    // its width in pixels: 3 to about 12 with the loudness, broad on a drop
    float w = 3.0 + 9.0 * iLoud;
    w = mix(w, 48.0 + 120.0 * iDrop, iDrop);
    float x = lane + mod(fragCoord.x, w);        // this pixel's place in the lane
    vec2 uv = vec2(x / iResolution.x, fragCoord.y / iResolution.y);
    vec3 col = texture(iChannel0, uv).rgb;
    fragColor = vec4(col, 1.0);
}
