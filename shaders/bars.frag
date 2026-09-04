// gacha shader: stage lights. Soft vertical bars sweep across the picture,
// one new bar per beat, in the frame's own average colour; a drop turns
// them all on at once.
float hash(float n) { return fract(sin(n * 127.1) * 43758.5453); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float beatCount = floor(iTime * iBPM / 60.0);
    // bars in the frame's own average colour, never a hue of their own
    vec3 avg = vec3(0.0);
    for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
        avg += texture(iChannel0, (vec2(float(x), float(y)) + 0.5) / 4.0).rgb;
    avg = normalize(avg / 16.0 + 0.05) * 1.1;
    vec3 light = vec3(0.0);
    float alpha = 0.0;
    for (int i = 0; i < 6; i++) {                 // the last six beats' bars
        float b = beatCount - float(i);
        float age = iBeat + float(i);              // beats since it lit
        float dir = hash(b) < 0.5 ? 1.0 : -1.0;
        float x0 = hash(b + 0.3);
        float x = fract(x0 + dir * age * 0.12);    // slides sideways
        float w = 0.04 + 0.05 * hash(b + 0.7);
        float bar = smoothstep(w, 0.0, abs(uv.x - x));
        float fade = exp(-age * 0.7) * (0.4 + 0.8 * iLoud);
        light += avg * (0.8 + 0.4 * hash(b + 0.9)) * bar * fade;
        alpha += bar * fade;
    }
    // haze: bars are brighter where the video is bright
    float vidl = dot(texture(iChannel0, uv).rgb, vec3(0.299, 0.587, 0.114));
    light *= 0.6 + 0.8 * vidl;
    light += iDrop * avg * 0.7;
    fragColor = vec4(light, clamp(alpha + iDrop * 0.6, 0.0, 1.0));
}
