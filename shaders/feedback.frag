// gacha shader: zoom feedback. The previous frame is enlarged a little each
// pass and blended under the video's bright parts, so lights rush toward the
// viewer and leave straight trails. No rotation, so the picture stays sharp.
// Zoom speed follows the loudness, a drop kicks it, breaks slow it down.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 c = uv - 0.5;
    float speed = 0.012 + 0.03 * iLoud + 0.06 * iDrop;
    if (iSection == 2) speed *= 0.4;                       // breaks: gentle
    vec3 prev = texture(iChannel1, c * (1.0 - speed) + 0.5).rgb * (0.86 + 0.06 * iLoud);
    vec3 vid = texture(iChannel0, uv).rgb;
    float l = dot(vid, vec3(0.299, 0.587, 0.114));
    vec3 col = max(prev, vid * smoothstep(0.35, 0.8, l));
    fragColor = vec4(col, 1.0);
}
