// gacha shader: the video itself, melted. Noise displaces the picture more
// the louder the song gets, and a drop rips it apart for a moment.
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    float a = fract(sin(dot(i, vec2(12.9898, 78.233))) * 43758.5453);
    float b = fract(sin(dot(i + vec2(1, 0), vec2(12.9898, 78.233))) * 43758.5453);
    float c = fract(sin(dot(i + vec2(0, 1), vec2(12.9898, 78.233))) * 43758.5453);
    float d = fract(sin(dot(i + vec2(1, 1), vec2(12.9898, 78.233))) * 43758.5453);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float amt = 0.02 + 0.08 * iLoud + 0.25 * iDrop;
    vec2 q = uv * 3.0 + iTime * 0.4;
    vec2 d = vec2(noise(q) - 0.5, noise(q + 31.7) - 0.5) * amt;
    vec3 col = texture(iChannel0, uv + d).rgb;
    // chromatic split on the beat
    float split = 0.004 * (1.0 - iBeat) * (0.5 + iLoud);
    col.r = texture(iChannel0, uv + d + vec2(split, 0.0)).r;
    col.b = texture(iChannel0, uv + d - vec2(split, 0.0)).b;
    fragColor = vec4(col, 1.0);
}
