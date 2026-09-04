// gacha shader: sparks, very subtle. Tiny particles are born on bright spots,
// drift upward and fade, carrying the colour of the spot they came from.
// Louder passages release more of them; a drop bursts a whole shower.
float hash(float n) { return fract(sin(n * 127.1) * 43758.5453); }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float aspect = iResolution.x / iResolution.y;
    vec3 col = vec3(0.0);
    float alpha = 0.0;
    float thr = 0.5 - 0.3 * iLoud - 0.3 * iDrop;
    for (int i = 0; i < 48; i++) {
        float fi = float(i);
        float speed = 0.3 + 0.35 * hash(fi + 0.1);
        float t = fract(iTime * speed + hash(fi));           // life 0..1
        float cycle = floor(iTime * speed + hash(fi));
        // birth: the brightest of four candidate spots
        vec2 birth = vec2(0.0); vec3 src = vec3(0.0); float best = -1.0;
        for (int c = 0; c < 4; c++) {
            vec2 cand = vec2(hash(fi + cycle * 3.1 + float(c) * 0.71), hash(fi + cycle * 7.7 + float(c) * 0.37));
            vec3 v = texture(iChannel0, cand).rgb;
            if (lum(v) > best) { best = lum(v); birth = cand; src = v; }
        }
        if (best < thr) continue;
        vec2 pos = birth + vec2(sin(t * 7.0 + fi) * 0.03, t * 0.3);
        float r = (0.004 + 0.004 * hash(fi + 2.2)) * (1.0 - 0.5 * t) * (0.8 + 0.4 * iLoud);
        float d = length((uv - pos) * vec2(aspect, 1.0));
        float dot_ = smoothstep(r, r * 0.25, d) * (1.0 - t * t);
        col += (src * 0.8 + 0.15) * dot_;  // faint, source-coloured
        alpha += dot_ * 0.35;              // barely there
    }
    fragColor = vec4(col, clamp(alpha, 0.0, 1.0));
}
