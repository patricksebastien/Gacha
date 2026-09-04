// gacha shader: slit-scan. A narrow slit sweeps across the frame once per
// bar; whatever passes under it is frozen into the picture, so motion turns
// into time-sliced streaks. Slit width grows with loudness, the direction
// flips with the section, a drop wipes the canvas clean.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 vid = texture(iChannel0, uv).rgb;
    vec3 prev = texture(iChannel1, uv).rgb;
    bool vertical = (iSection % 2) == 1;
    float axis = vertical ? uv.x : uv.y;
    float s = fract(iBar);
    if ((iSection / 2) % 2 == 1) s = 1.0 - s;
    float w = 0.012 + 0.05 * iLoud;
    float slit = smoothstep(w, 0.0, abs(axis - s));
    vec3 col = mix(prev, vid, max(slit, iDrop));
    // keep the frozen canvas from drifting to pure grey over time
    col = mix(col, vid, 0.02);
    fragColor = vec4(col, 1.0);
}
