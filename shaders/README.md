# Shaders

Generative layers for the video backdrop, one Shadertoy-style fragment
shader per file. `gacha_gl.py` lists every `.frag` (or `.glsl`) in here at
launch, compiles them in the background and composites the chosen one over
the processed video with its own opacity and blend mode. Drop a new file in
and it shows up in the picker; edit one while gacha is running and it
recompiles on the next frame (compile errors land in the log).

## Writing one

Write `void mainImage(out vec4 fragColor, in vec2 fragCoord)` as on
Shadertoy. The wrapper supplies the usual uniforms plus music data from the
playing song:

| uniform | meaning |
|---|---|
| `iResolution`, `iTime`, `iTimeDelta`, `iFrame`, `iMouse`, `iDate` | the Shadertoy set |
| `iChannel0` | the processed video frame |
| `iChannel1` | this shader's own previous output, for trails and feedback |
| `iChannelResolution[2]` | sizes of the two channels |
| `iLoud` | loudness 0..1 |
| `iBeat`, `iBar` | phase inside the current beat / bar, 0..1 |
| `iBPM` | tempo |
| `iSection` | 0 intro, 1 groove, 2 break, 3 outro |
| `iRoot` | root note 0..11 (C..B), -1 when unknown |
| `iDrop` | 1 on a drop, decaying to 0 |
| `iSwell` | reverse-cymbal build 0..1 |
| `iSongPos` | position in the song 0..1 |

Leave out `#version` and `precision` lines; the wrapper strips them and
adds its own. A shader that only makes sense when it replaces the picture
rather than mixing with it can pin its minimum opacity with a comment
directive, e.g. `// gacha: opacity=1.0` (see `mirrorhall.frag`).

House style: borrow the video's own colours (its average, or a sample from
`iChannel0`) rather than inventing a palette, react to `iBeat`/`iLoud`, and
give `iDrop` a moment.

## The set

| file | effect |
|---|---|
| `bars.frag` | stage lights: soft vertical bars sweep in, one per beat, all on at a drop |
| `datamosh.frag` | fake codec corruption, blocks of the previous frame stick and drift |
| `feedback.frag` | zoom feedback, bright parts rush toward the viewer and leave trails |
| `glow.frag` | bloom that widens with loudness, tightens on beats, flares on a drop |
| `glyphs.frag` | picture rebuilt from procedural glyphs, refreshed every beat |
| `halftone.frag` | CMYK halftone print with three slowly rotating dot screens |
| `leaks.frag` | light leaks washing in from the edges on each beat |
| `mirrorhall.frag` | mirror wall of full-height panels (replaces the picture) |
| `mosaic.frag` | video cut into tiles that flip and shift on each beat |
| `plasma.frag` | slow plasma warping the picture in its own colours |
| `pointcloud.frag` | LiDAR-style dot cloud with fake depth and parallax |
| `relief.frag` | embossed relief lit by a light orbiting once per bar |
| `skyline.frag` | skyline equaliser, each column rising to its own brightness |
| `slitscan.frag` | slit-scan sweeping once per bar, freezing motion into streaks |
| `sparks.frag` | tiny particles born on bright spots, drifting up and fading |
| `super8.frag` | Super 8 home movie: 18 fps frame hold, faded Kodachrome grade with halation, grain, dust, hairs and dark scratches, exposure flicker, splices and flash frames on downbeats, the reel burning out on a drop |
| `thermal.frag` | thermal camera palette with animated iso-lines |
| `vhs.frag` | worn VHS over the whole frame: a tracking band rolling the full height every 2 bars, frame roll, wobble, chroma bleed, dropouts on the beat |
| `warp.frag` | the picture melted by noise, ripped apart on a drop |
