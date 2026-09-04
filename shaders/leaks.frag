// gacha shader: light leaks. Soft washes of colour bleed in from the edges
// on each beat, tinted from the frame's own average colour so they belong
// to the picture. Louder beats leak more; a drop floods the frame.
float hash(float n) { return fract(sin(n * 127.1) * 43758.5453); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 avg = vec3(0.0);
    for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
        avg += texture(iChannel0, (vec2(float(x), float(y)) + 0.5) / 4.0).rgb;
    avg = normalize(avg / 16.0 + 0.05);
    // push the frame's colour further from grey so the leak reads as colour
    float al = dot(avg, vec3(0.299, 0.587, 0.114));
    avg = clamp(al + (avg - al) * 2.2, 0.0, 1.0);
    float beatCount = floor(iTime * iBPM / 60.0);
    vec3 leak = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float b = beatCount - float(i);
        float age = iBeat + float(i);
        // a spot on one of the four edges
        float side = floor(hash(b) * 4.0), along = hash(b + 0.5);
        vec2 c = side < 1.0 ? vec2(along, 0.0) : side < 2.0 ? vec2(along, 1.0)
               : side < 3.0 ? vec2(0.0, along) : vec2(1.0, along);
        float d = length((uv - c) * vec2(iResolution.x / iResolution.y, 1.0));
        float glowr = 0.45 + 0.3 * hash(b + 0.9);
        float k = exp(-d * d / (glowr * glowr)) * exp(-age * 0.7) * (1.0 + 1.6 * iLoud);
        vec3 tint = avg.gbr * 0.3 + avg * 0.7;
        leak += tint * k;
    }
    leak += avg * iDrop * 0.8;
    vec3 vid = texture(iChannel0, uv).rgb;
    fragColor = vec4(vid + leak, 1.0);
}
