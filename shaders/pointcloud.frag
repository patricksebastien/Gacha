// gacha shader: a LiDAR-style point cloud of the video. The picture is
// sampled on a grid of dots; brightness sets dot size and doubles as fake
// depth, so the cloud sways with a slow parallax as if the camera drifted.
// A faint scan line in the frame's own colour sweeps across once per bar
// and lights the points it passes,
// loudness swells the dots and thickens the cloud, a drop scatters them.
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float aspect = iResolution.x / iResolution.y;
    float rows = 60.0 + 30.0 * iLoud;                        // denser when loud
    vec2 grid = vec2(floor(rows * aspect), rows);
    // camera drift: near (bright) points move more than far (dark) ones
    vec2 sway = vec2(sin(iTime * 0.5), cos(iTime * 0.37)) * (0.012 + 0.02 * iLoud);
    float sweep = fract(iBar);                                 // scan position 0..1
    // the beam takes the frame's own average colour, brightened, so it
    // never brings a colour of its own into the picture
    vec3 scan = vec3(0.0);
    for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
        scan += texture(iChannel0, (vec2(float(x), float(y)) + 0.5) / 4.0).rgb;
    scan = normalize(scan / 16.0 + 0.05);
    scan = mix(scan, vec3(dot(scan, vec3(0.299, 0.587, 0.114))), 0.5) * 1.2;   // half desaturated
    vec3 col = vec3(0.0);
    vec2 cell0 = floor(uv * grid);
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        vec2 cell = cell0 + vec2(float(x), float(y));
        vec2 cuv = (cell + 0.5) / grid;
        vec3 vc = texture(iChannel0, cuv).rgb;
        float l = lum(vc);
        vec2 pos = cuv + (l - 0.5) * sway;                     // parallax
        pos += (vec2(hash(cell), hash(cell + 3.7)) - 0.5) * iDrop * 4.0 / grid;
        float r = (0.22 + 0.55 * l) * (0.85 + 0.35 * iLoud) / grid.y;
        float dist = length((uv - pos) * vec2(aspect, 1.0));
        float dot_ = smoothstep(r, r * 0.55, dist);
        float near = exp(-abs(pos.x - sweep) * 14.0);          // lit by the scan
        vec3 c = vc * (0.5 + 0.7 * l + 1.6 * near);            // lit points: their own colour
        col = max(col, c * dot_);
    }
    col += scan * exp(-abs(uv.x - sweep) * 250.0) * 0.25;     // the beam itself, faint
    col += texture(iChannel0, uv).rgb * 0.06;                  // faint ghost of the video
    fragColor = vec4(col, 1.0);
}
