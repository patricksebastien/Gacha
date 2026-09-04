// gacha shader: the video as an embossed relief. Brightness is height, a
// light orbits the picture once per bar, bright areas cast shading over
// their neighbours. A drop flashes the light straight on.
float h(vec2 uv) { return dot(texture(iChannel0, uv).rgb, vec3(0.299, 0.587, 0.114)); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 d = 2.0 / iResolution.xy;
    float strength = 6.0 + 10.0 * iLoud;
    vec3 n = normalize(vec3((h(uv - vec2(d.x, 0.0)) - h(uv + vec2(d.x, 0.0))) * strength,
                            (h(uv - vec2(0.0, d.y)) - h(uv + vec2(0.0, d.y))) * strength, 1.0));
    float a = iBar * 6.2831853;
    vec3 L = normalize(vec3(cos(a), sin(a), 0.5 + iDrop * 2.0));
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(reflect(-L, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);
    vec3 vid = texture(iChannel0, uv).rgb;
    vec3 col = vid * (0.25 + 0.9 * diff) + spec * 0.6 * (0.5 + iLoud);
    fragColor = vec4(col, 1.0);
}
