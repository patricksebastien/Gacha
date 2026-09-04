// gacha shader: thermal camera. Brightness becomes a heat palette with
// animated iso-lines. The palette rotates with the section, the contour
// count pulses with loudness, a drop whites the whole frame out.
vec3 heat(float t) {
    // black -> deep blue -> purple -> red -> orange -> yellow -> white
    vec3 a = mix(vec3(0.0, 0.0, 0.1), vec3(0.35, 0.0, 0.6), smoothstep(0.0, 0.25, t));
    a = mix(a, vec3(0.9, 0.1, 0.1), smoothstep(0.25, 0.5, t));
    a = mix(a, vec3(1.0, 0.6, 0.0), smoothstep(0.5, 0.75, t));
    a = mix(a, vec3(1.0, 1.0, 0.85), smoothstep(0.75, 1.0, t));
    return a;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 vid = texture(iChannel0, uv).rgb;
    float t = dot(vid, vec3(0.299, 0.587, 0.114));
    t = clamp(t * (1.1 + 0.4 * iLoud) + 0.03 * sin(iTime + uv.y * 6.0), 0.0, 1.0);
    vec3 col = heat(t);
    if (iSection == 2) col = col.bgr;                    // breaks run cold
    else if (iSection == 0) col = col.gbr * 0.9;         // intro: ghostly
    float bands = 6.0 + 6.0 * iLoud;
    float iso = smoothstep(0.08, 0.0, abs(fract(t * bands + iTime * 0.3) - 0.5) - 0.42);
    col *= 1.0 - 0.6 * iso;
    col = mix(col, vec3(1.0), iDrop * 0.8);
    fragColor = vec4(col, 1.0);
}
