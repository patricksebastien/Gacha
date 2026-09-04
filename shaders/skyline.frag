// gacha shader: skyline equaliser made of the picture. Each column rises
// from the bottom to the height of its own brightness, showing the video
// inside the bar under a dimmed sky, with a faint cap. Columns multiply and jump with the
// loudness; a drop sends them all to the top.
float colLum(float x) {
    float s = 0.0;
    for (int i = 0; i < 8; i++) s += dot(texture(iChannel0, vec2(x, (float(i) + 0.5) / 8.0)).rgb, vec3(0.299, 0.587, 0.114));
    return s / 8.0;
}
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float cols = 48.0 + 48.0 * iLoud;
    float cx = (floor(uv.x * cols) + 0.5) / cols;
    // heights relative to the frame's average brightness, so the skyline
    // has relief on dark and bright footage alike, and maxes out quickly
    float avg = 0.0;
    for (int i = 0; i < 8; i++) avg += colLum((float(i) + 0.5) / 8.0);
    avg = avg / 8.0 + 0.03;
    float rel = colLum(cx) / avg;                          // 1 = average column
    float hgt = clamp(pow(rel * 0.55, 1.3) * (0.9 + 0.3 * iLoud) + iDrop, 0.0, 1.0);
    hgt *= 0.9 + 0.1 * (1.0 - iBeat);                    // small bounce on the beat
    vec3 vid = texture(iChannel0, uv).rgb;
    float inside = step(uv.y, hgt);
    float top = smoothstep(0.012, 0.0, abs(uv.y - hgt));
    // inside the column the video shows as is; above it the video is dimmed
    // rather than blacked out, and the caps are only a faint highlight
    vec3 col = mix(vid * 0.35, vid, inside) + top * (vid * 0.4 + 0.15);
    float alpha = mix(0.6, 0.0, inside) + top * 0.35;     // sky dimmed, column transparent
    fragColor = vec4(col, alpha);
}
