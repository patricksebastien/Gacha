// gacha shader: datamosh. Fake codec corruption: blocks of the previous
// frame stick and drift while only strong edges keep updating, like a
// broken stream. Louder passages stick more, a drop resets the picture.
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 grid = vec2(32.0 * iResolution.x / iResolution.y, 32.0);
    vec2 cell = floor(uv * grid);
    float beatCount = floor(iTime * iBPM / 60.0);
    float h = hash(cell + beatCount * 0.61);
    float stick = 0.35 + 0.5 * iLoud;
    // stuck blocks drift a little each frame
    vec2 drift = (vec2(hash(cell + 1.3), hash(cell + 4.7)) - 0.5) * 0.004;
    vec3 prev = texture(iChannel1, uv + drift).rgb;
    vec3 vid = texture(iChannel0, uv).rgb;
    vec2 d = 1.0 / iResolution.xy;
    float edge = abs(lum(texture(iChannel0, uv + vec2(d.x, 0.0)).rgb) - lum(vid))
               + abs(lum(texture(iChannel0, uv + vec2(0.0, d.y)).rgb) - lum(vid));
    vec3 col = (h < stick) ? mix(prev, vid, smoothstep(0.08, 0.3, edge)) : vid;
    // colour smearing typical of broken motion vectors
    if (h < stick * 0.5) col = col.brg * 0.15 + col * 0.85;
    col = mix(col, vid, iDrop);
    fragColor = vec4(col, 1.0);
}
