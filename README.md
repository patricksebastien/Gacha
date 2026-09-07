# Gacha

**A slot machine for songs.** Point it at folders of audio samples and every run produces a brand-new track: drums carved out of random samples, textures looped and reversed behind them, glitch bursts, tempo-synced dub delays, and a different arrangement every single time.

No two pulls are alike. Every sample gets used.

## How it works

1. **Loads the samples you ticked** in the Samples tab (wav, flac, aiff, mp3). Each subfolder of `samples/` is a set; tick a whole set or hand-pick files across sets.
2. **Carves a drum kit** from randomly chosen samples: a short chunk is cut at the loudest transient, then sculpted into a kick (lowpass + drive), snare (bandpass + room), closed and open hihat (see below) and ride (long, airy decay). Each role claims its own source samples, so a TV jingle can become your snare while a flute becomes the ride.
3. **Sequences a song** on a 16th-note grid at a single steady tempo: kick, snare, hats and occasional rides play style-driven patterns with per-bar mutations, ghost notes, fills, subtle swing and stereo spread. Some hat steps open up, offbeat 8ths in house and garage, sparser elsewhere, and an open hat rings until the next closed hat chokes it like a pedal closing. Hihats occasionally shatter into ultra-fast stutter rolls.
4. **Layers textures** per section: long samples are looped with crossfades, reversed, or time-stretched to exact bar lengths, then run through randomized effect chains (reverb, delay, gain-compensated saturation, chorus, phaser, bitcrush, lo-fi codec, ladder filter, pitch shift). A chop layer slices a sample into pieces and replays them rhythmically. All delays are synced to the tempo grid.
5. **Adds a sub bass** on the first kick of a bar, pitched to the root note heard in each section's textures, with a long release.
6. **Leads into drops** with reverse cymbal swells, DJ beat-repeat rolls and short silences.
7. **Structures the result** into sections (intro, grooves, breaks, outro) and writes a mastered wav plus a `.json` section map, so any render can later be sliced and recombined.

## Requirements

Python 3.10+ and:

```bash
pip install numpy soundfile librosa pedalboard PySide6 av
```

[pedalboard](https://github.com/spotify/pedalboard) (Spotify's audio effects library) powers all effect chains, [PyAV](https://github.com/PyAV-Org/PyAV) (ffmpeg) decodes the backdrop videos. The app is the Qt GUI; `gacha_engine.py` is the render engine it drives.

## Quick start

Drop your samples into a subfolder of `samples/`, one folder per set, e.g. `samples/default/`, `samples/field/`, `samples/vinyl/`. Then:

```bash
python3 gacha.py
```

Pick the sets and files to use in the **Samples** tab, hit **Generate**. Renders land in `output/`.

### Sets from long recordings

The engine loads every ticked sample into RAM as float32 stereo, about 350 MB per quarter hour, so hours-long session recordings cannot be used as they are. `gacha_bank.py` cuts a set out of such a folder: it picks files at random and saves one random excerpt of each, re-rolling windows that are near silence. `--normalize` sets how each excerpt is levelled: `lufs`, the default, brings it to -14 LUFS integrated with peaks clipped at full scale, which is what Reaper's *normalize to LUFS-I* did to the default set, so sets audition at one level (`--lufs` changes the target); `peak` scales the loudest sample to `--peak` dBFS without clipping; `none` keeps the source level. Note that the engine peak-normalizes every sample on load anyway, so on-disk level matters for auditioning and for how clipping shapes a sample, not for its raw gain in the mix.

```bash
# 75 excerpts of 4 to 30 s from a folder of recordings, into samples/puredata/
python3 gacha_bank.py puredata ~/recordings --count 75 --seed 11

# longer excerpts, added to an existing set
python3 gacha_bank.py puredata ~/recordings --min-len 8 --max-len 60 --overwrite
```

Excerpt names keep the source stem and the start offset, so `2019-07-06T23.17.57_1832s.wav` is 30 minutes and 32 seconds into that session. A new set appears unticked after Refresh; tick its folder to use it. With long excerpts, `random_start` set to `one-shots` or `all` in the Layers & FX tab makes the cuts land anywhere in them rather than at their start.

## The app

`gacha.py` is the Qt window. `gacha_engine.py` is the render engine it runs in a subprocess for every Generate, `gacha_gl.py` is the OpenGL video backdrop with the shaders in `shaders/`, `gacha_live.py` is the audio input and tap-tempo clock behind the live mode, `gacha_video.py` decodes the backdrop video on its own thread (any speed, forwards or backwards, frame-exact seeks, and V4L2 capture devices), and three command-line tools prepare material: `gacha_bank.py` cuts sample sets out of long recordings, `gacha_transcode.py` re-encodes backdrop videos for instant seeking, and `gacha_syncdemo.py` renders a test clip for the follow mode (a slow melody of sustained notes; on every note the picture shows its number, its name and its colour, with a progress bar and a frame counter, so what you hear is checked against what you see).

All parameters as widgets in four tabs (General, Drums, Sections, Layers & FX), a live render log, and a built-in player:

* **Samples** tab: the library as a tree, one node per set folder, with checkboxes. Ticking a folder ticks every file in it, a partly ticked folder shows the mixed state, and single files can be ticked across sets. Only ticked files go into a render, and the selection is remembered between launches. On the first launch only the `default` set is ticked, and folders start collapsed. Double-click a file to audition it. New files dropped into a fully ticked folder join it on Refresh; a brand-new folder starts unticked. Loose files directly in `samples/` show up as their own group. Each render's JSON records which samples it used, as `set/file.wav`.
* **Video audio** (Video tab): tick *use the audio of the ticked videos as sample material* and the next render cuts its drums, textures and chops out of the soundtracks of the ticked videos (all of them when none is ticked), on top of any ticked samples or with no samples at all; the tracks are extracted once with ffmpeg into `videos/.audio/`. Every render records a **sample timeline** in its JSON (`events`: when each sound was placed, from which sample, at which source offset, how long, at what rate and direction), and while such a song plays the backdrop **follows** it: a kick cuts to the frame it was carved from, a texture runs along with its loop, backwards or stretched as the audio is, a chop jumps slice by slice, so what you see is what you hear. *Picture follows* picks the layers: textures only, textures and chops, everything with hits winning over chops over textures (which, since a kick or a hat is nearly always sounding, means the picture lives on the drums), or *varies* (the default): which layer leads is re-rolled at every section and every 4-bar phrase, the textures and chops a little more often than the drums, the other layers filling in behind the lead so something is always on screen; off leaves the backdrop to its usual clip switching. Ticking it sets `random_start` (Layers & FX) to *all* when it was off, so cuts land anywhere in the film rather than at its start; change it back if you want otherwise.
* **Video input** (General tab): pick a V4L2 capture device, a webcam or a cheap USB composite grabber with a VHS deck on it, and the live picture replaces the clips as the backdrop. The effects, the shader layer and the words run on it as usual; clip switching and jumps are held off until the input is set back to off. Rewinds work on the live picture too: the last seconds of captured frames stay in memory, a rewind scrubs back through them and then fast-forwards back to live, a backwards-then-forwards jog. The list is read at launch.
* **Outputs** tab: every past render, newest first. Double-click to listen. New renders start playing automatically.
* **Mixer** tab: the section editor. Pick any render, double-click its sections (intro, grooves, breaks, outro) to audition each slice, then assemble your favourite parts from *different* renders into a final arrangement. Reorder, repeat sections at will, and render the result to a single stitched track with seamless crossfades.

Video files (mp4, mov, m4v, webm, mkv, any extension case) in `videos/` play as a looping ambient backdrop behind the interface. The **Videos** tab lists them as the same kind of tree as the samples: every subfolder is a set, loose files in `videos/` are a group of their own, and checkboxes pick which clips a song may use. On the first launch only the loose files are ticked, so `videos/originals/` stays a parking place for source files until you tick it. Double-click a clip to loop it right away. The stock `videos/gacha.mp4` shows at launch and is never picked for a song unless it is the only clip.

What happens when a song starts depends on how many clips are ticked:

* **One**: it plays for the whole song, jumping to a random spot at every section change.
* **Several**: a random one starts the song, and the clip changes with the music. A drop, meaning a groove after an intro or a break, or a section led in by a beat-repeat, cymbal swell or gap, swaps the clip once at least 4 bars have passed since the last swap. An ordinary section boundary swaps it about one time in three, and only after 16 bars without a swap. Every other boundary is a jump within the current clip. The *switch between the ticked videos* checkbox in the Video tab turns swapping off.
* **None**: every clip in the folder is fair game, as if all were ticked.

Phone and camera clips are usually 4K H.264 or HEVC at 60 fps with an audio track, which is heavy to decode and, more importantly, slow to seek: the backdrop jumps to a new spot at every section change and swaps clips on drops, and an H.264 seek stalls the picture for 100 to 300 ms right on the downbeat. `gacha_transcode.py` brings the whole folder to a fast-seeking spec with the system's ffmpeg and ffprobe, no Python video package needed:

```bash
python3 gacha_transcode.py --dry-run      # what would change
python3 gacha_transcode.py                # 1080p, 30 fps, MJPEG .mov, audio removed
python3 gacha_transcode.py --height 720 --fps 24 --quality 4
python3 gacha_transcode.py --codec h264 --quality 20 --gop 15   # smaller files, keyframe every 15 frames
python3 gacha_transcode.py videos/field/clip.MP4                # just this one
```

Every video under `videos/` and its set folders is probed. Files already at spec are left alone, anything else is re-encoded next to itself and the original moves to `videos/originals/`, mirroring the set folder it came from. A file that only has an audio track to lose is remuxed without re-encoding. The stock clip and `originals/` are skipped, and `--force` redoes files already at spec.

MJPEG files are larger, roughly 5 MB per second at the default quality, but they live in the git-ignored videos folder. The folder is rescanned on Refresh, so new files are picked up without a restart, and everything in it except the stock clip is git-ignored. While a song plays, the video jumps to a random spot at every section change, so the picture cuts with the music. Turn that off with the Playback checkbox in the Video tab.

Everything about the backdrop lives in the **Video** tab of the parameter panel:

* **Playback**: jump to a random spot in the clip at every section change, switch between the ticked clips on drops, and hide the interface and the mouse pointer the moment a song starts. Move the mouse, click or press a key to bring them back, and they hide again after an adjustable idle time (default 1.5 s). When a song ends, every effect switches off, the interface comes back and the video keeps looping clean. `F11` or the button goes fullscreen and hides the interface at once, `Esc` or `F11` again leaves it. `Space` jumps to the next video FX: a new random effect look plus the next shader, or a random one in random mode. `S` switches the shader layer off and on; while it is off, `Space` only cycles the video effects. `C` toggles *video only*: the picture as it is, no effects, no shader layer, no words, while the picture keeps following the song (timeline, section jumps, clip switches, rewinds, the outro fade). `1` to `9` select a shader directly in list order and `0` turns the layer off. None of these keys bring the interface back.
* **Audio input** (VJ mode): tick it, or press `L`, and an audio input drives the visuals instead of a song. Pick the capture device (a mic, a line in, or on PipeWire/Pulse a monitor source that carries whatever the machine is playing); the level bar shows the loudness the effects see, normalised against a slowly decaying running peak so quiet and loud sources both use the full range. With **auto** ticked (the default) the app listens along: after about 4 s it has a tempo, from spectral-flux onsets autocorrelated under a tempo prior, locks the beat phase to the bass onsets so kicks rather than offbeat hats mark the beat, and follows the root note as the dominant pitch class of a running chroma from 30 Hz up, which is what the tint effect uses. The status line shows what it hears. Tap tempo is the fallback and the override: press `T` on the beat four or more times, the first tap on the one, and auto switches off and follows your taps; `D` restarts the bar on the current tempo and the spinbox can be typed into as well. `X` marks a drop by hand: the flash and the shaders' `iDrop` fire, the bar restarts on it, the look and the shader roll again and the clip may switch. Every 8 bars of the live clock also count as a section, so the look, the shader in random mode and the clip jumps keep changing the way they do in a song. Loudness, beat, bar, tempo, root and drop reach the shaders as `iLoud`, `iBeat`, `iBar`, `iBPM`, `iRoot` and `iDrop`; section stays groove and swells are not detected. Automatic drop detection from loudness is in the code but off, since on rendered songs it caught few real drops and fired falsely every half minute. Starting a song switches live off; the interface hides while live is on like it does for a song.
* **Source grade**: colour correction of the picture itself, made for the VHS input: **exposure**, **black** level (negative pulls the tape's grey blacks down), **gamma** for the mid-tones, **contrast** around mid grey, **saturation**, **warmth** (amber to blue) and **tint** (green to magenta). It runs on the GPU inside the video sampler, so it sits before every effect and the shader layer and stays on in video-only mode and clean flashes; the *neutral* button puts everything back. **react** lets the grade breathe with the music, very slightly: at most a few percent of exposure and saturation with the loudness, and a touch of warmth toward the root note's colour (notes on the red side of the hue circle warm the picture, those on the cyan side cool it). Meant for MIDI knobs later.
* **Audio through**: tick it, or press `A`, and the chosen hardware input plays out of the chosen hardware output in realtime through the app: the live path of a show, where a VHS deck's sound reaches the PA through Gacha so that effects, and later the generated material, can join it on the way. The rack is empty for now, so what comes out is the deck's own sound. It runs in `gacha_audio.py` on its own thread on pedalboard's audio streams, one for the input and one for the output, with Python between them owning every sample: 256-frame buffers make about 11 ms of buffering plus whatever the devices add; the block size can be raised if the status line ever shows drops. Only ALSA hardware devices are listed (PipeWire's own devices report no channels to the library), and a device PipeWire or another app is actively playing through cannot be opened; idle ones are released to us automatically. With live mode on, the analysis listens to this stream instead of opening a capture of its own. The main volume slider sets its level, ramped so it never clicks. **Sync** holds the sound back by a settable number of milliseconds so it meets the picture, which arrives late: the grabber hands over a frame one frame period after it happened (40 ms on a PAL MS210x, measured from the kernel's capture timestamps), the paint loop and its 25 ms effect throttle add a dozen more, and the screen or projector adds its own. Set it while watching a tape, on mouths and cuts; it changes live without a click. The status line shows input and output peaks, DSP load, buffering, the sync delay, any drops, and with the video input on the picture's measured capture latency as a starting point for the sync value.
* **Takes**: press `R`, or the record button, and the live feed is recorded into RAM: the tape's dry sound from the audio through and the frames from the video input, starting with a **pre-roll** of the seconds before the press (10 by default, up to 12), because you decide something was interesting after it happened. Press again to stop. A **limit** (60 s by default) stops a forgotten take; the next press then starts a new one. The takes together are the sample bank the engine will build sections from at the end of a piece, so short ones are fine, and all of them count. A minute of 720x576 frames is about 2.5 GB; the store is capped at 12 GB and the oldest takes go first. Each take remembers how far behind the world its frames were (the grabber's 40 ms), so an audio offset finds the frame that was on screen. *save takes* writes every take as wav + MJPEG mov into `takes/<date>/` for rehearsal and offline work; *clear* forgets them. `gacha_takes.py` is the store.
* **New random look on every section**: which stylistic effects are active and how hard is re-rolled every 4-bar phrase, and the section-level choices (kaleidoscope, mono tones, shader) change with each section, so long sections keep moving.
* **Effects**, each a strength from 0 (off) to 1, driven by the playing song's section map, root notes and loudness:

| Row | Effects |
| --- | --- |
| Color | **color** saturation and hue per section, intros bloom in, breaks wash out. **tint** toward the colour of the section's root note. **pump** brightness follows loudness. **flash** drop FX in sync with the audio: the frame freezes and stutters in the exact rhythm of a beat-repeat, builds to white under a reverse cymbal, holds still in a gap and inverts on the drop. |
| Pixels | **pixel** pixelation during breaks, switching on and off per 2-bar phrase with a new block size that breathes with the loudness. **bits** colour depth, 1.0 is 1 bit, 0.75 is 2 bits, 0.5 is 3 bits, 0.25 is 4 bits. **dither** 1-bit Bayer dithering. |
| Tone | **mono** grayscale where each section rolls its tone count: smooth gray, or black and white at a random threshold, or 3, 4, 6 or 8 tones, with strength setting how often the hard tones win. **solarize**, **edges**, **scanlines**. |
| Grain | **grain**. |
| Motion | **zoom** punches in with loudness. **trails** bright parts linger. **glitch** horizontal tears, more when loud, a few bands at a time. Each section picks a flavour: thin sharp lines of 0.4 to 1% of the frame height, medium bands of 4 to 14%, or anything from 4% up to 38% slabs. **rgb shift** chromatic aberration pulsing with loudness. **reverse** rewind: on a downbeat, with probability = strength, the video runs backwards for the chosen **length** (1/8 bar by default, up to 2 bars, or random per bar) at a random speed from 1x to 3x; every clean flash jogs the same way, and a reverse cymbal swell rewinds the picture faster and faster into the drop. This is a real rewind of the video, at full resolution and any speed, and playback carries on forwards from wherever it stopped. It is instant on the all-intra files `gacha_transcode.py` writes; long-GOP camera originals decode a whole keyframe group per step, which is fine at 1080p and too slow to keep up at 4K, so transcode those. |
| Frame | **vignette**. **kaleido** upright kaleidoscope with 4, 6 or 8 mirrored segments; strength is how often a section gets it. **words** now and then a 4-bar phrase fills the screen with a cloud of keys and values from the render's JSON in mixed sizes, opacities and fonts (`bpm: 121`, `style = clave`, `"root": E`, the knobs, some of the samples it was made from, the playing section's kind, bars, root and transition) plus the song's own name; the live input, which has no JSON, gets random words instead. Strength is how often. Fonts are the `.ttf` files in `fonts/` (open-licensed terminal-style monospace faces ship with the repo, drop in your own). **clean** on a downbeat, with probability = strength, a 1/16-note flash of the video as it is: no effects, no shader layer, no words. |

The Randomize button rolls new strengths for all of them.

**Rendering.** On a machine with OpenGL 3.3 the backdrop is drawn on the GPU (`gacha_gl.py`): every effect above runs as a fragment shader at full resolution and 60 fps. Without OpenGL the GUI falls back to a numpy renderer that shrinks frames to 480 pixels wide.

**Shader layer.** On top of the video, a second layer runs a generative fragment shader from `shaders/`, composited with its own opacity and blend mode (mix, add, screen). The layer starts off when the app launches and switches to "random" as soon as a song has been generated; "random" picks a new shader every section. The shaders are Shadertoy-style: write a `mainImage(out vec4 fragColor, in vec2 fragCoord)` and you get `iTime`, `iResolution`, `iChannel0` (the processed video) and `iChannel1` (the shader's own previous frame), plus the music: `iLoud` (0-1), `iBeat` and `iBar` (phase 0-1), `iBPM`, `iSection` (0 intro, 1 groove, 2 break, 3 outro), `iRoot` (0-11, -1 unknown), `iDrop` (1 on a drop, decaying), `iSwell` and `iSongPos`. The alpha you output is the layer's coverage. A shader that only works by replacing the picture can put `// gacha: opacity=1.0` in its source to override a lower layer opacity. Eighteen originals ship:

| Shader | Look |
| --- | --- |
| `bars` | Stage-light bars sweeping in, one per beat. |
| `datamosh` | Fake codec corruption, blocks stick and drift, a drop resets. |
| `feedback` | Zoom feedback toward the viewer, bright parts leave straight trails. |
| `glow` | Bloom on the bright parts, pumping with the beat. |
| `glyphs` | The picture rebuilt from procedural glyphs, refreshed per beat. |
| `halftone` | CMYK print screens, rotating slowly, negative on a drop. |
| `leaks` | Light leaks from the edges on each beat, in the frame's own colours. |
| `mirrorhall` | Mirror wall: full-height panels each showing a different slice of the picture, one lighting up per beat. |
| `mosaic` | Tiles that shift, mirror and flash, 45 to 220 of them with loudness. |
| `plasma` | A fractal field in the video's own colours, breathing with the beat. |
| `pointcloud` | LiDAR-style dot cloud with parallax and a scan line per bar. |
| `relief` | Embossed heightfield lit by a light orbiting once per bar. |
| `skyline` | Column equaliser made of the picture. |
| `slitscan` | A slit sweeps per bar and freezes time slices behind it. |
| `sparks` | Particles born on bright pixels, drifting up. |
| `thermal` | Heat palette with animated iso-lines, cold in breaks. |
| `vhs` | A tracking band rolling up the full height every 2 bars, frame roll and wobble top to bottom, chroma bleed, dropouts on the beat, wearing out toward the end. |
| `warp` | The video melted by noise, torn by drops. |

Keys `1` to `9` reach the first nine in this alphabetical order; the combo and `Space` reach them all. Compile errors show in the render log. Shaders you bring in from elsewhere keep their own licence.

### Perform

The Perform tab is the live section engine, what the feet will drive on stage once MIDI is mapped; for now every control has a key. It needs audio through on (`A`), because the section is mixed into that path. **Material** is what sections are built from: the takes recorded during the show, or the ticked sample sets while rehearsing (nothing ticked means every sample). **next section** (`N`) renders an 8-bar loop from the material at the live tempo, `gacha_section.py` doing in about a second what the engine does for a song, and starts it on the next bar; while it loops, the following one is rendered ahead, so the next press is instant. **next kind** picks the shape: groove, break, build, sparse or random. **stop** (`Shift+N`) silences the section on the bar; the tape stays. **event** (`E`) fires a one-shot cut from the material on the next beat. The four **stems** (`F1` to `F4`) switch drums, layers, chops and events in and out with a ramp. The **A/B** slider (`[` and `]` step by 10) mixes the tape through against the section at equal power, 0 to 100 %, later an expression pedal. **Picture** chooses what the screen shows while a section plays: the frames the sounds were cut from (takes with video), the tape as it comes in, or auto, which switches to the take frames once A/B passes 50 %. The live clock follows the section, so the visuals, the shaders and the tap tempo are on its bars.

### Keys

The Keys tab lists every key the performer has, grouped as picture, tape, clock and section engine, with what each does. It is built from one table in `gacha.py` that also gives each action an id; the MIDI column is empty for now and will take the controller bindings, one per action, so the same table becomes the mapping.

## Parameters

Everything below lives in the parameter panel of the GUI, grouped in the same tabs. Names are given as the engine's job keys; the widget labels match them.

| Parameter | Default | Description |
| --- | --- | --- |
| `seed` | random | Reproducible run for the *same sample selection*. With `count` > 1, seeds increment. |
| `duration` | `180` | Target length in seconds. |
| `count` | `1` | Number of songs to render in one go. |
| `bpm` | random 84-128 | Tempo. |
| `drum_style` | `random` | Rhythmic skeleton, see below. |
| `intro_style` | `ambient` | How the song opens, see below. |
| `intro_bars` | random | Cap the intro length in bars. `0` skips the intro entirely. |
| `outro_style` | `random` | How the song ends, see below. `none` is a hard stop on the last bar. |
| `num_samples` | all | Build the song from only N randomly picked samples among the ticked ones. |
| `pan_drums` | `0.3` | Random pan spread per snare/hihat hit (kick stays centered). |
| `pan_layers` | `0.6` | Random static pan per background layer. |
| `pan_events` | `0.5` | Random pan per chop slice and one-shot. |

### Drums

| Parameter | Default | Description |
| --- | --- | --- |
| `swing` | random 0-0.06 | Delay of every offbeat 16th as a fraction of a step. `0.33` is a triplet feel, max `0.5`. |
| `humanize` | `0.5` | How much like a person the drums are played, samples and synth alike. Velocities wobble, downbeats lean in, offbeat 16ths sit back, and timing drifts up to a few milliseconds. Every hit exists in three velocity layers: soft hits are lowpassed and shorter, hard hits are saturated. `0` is a machine. |
| `mutation_chance` | `0.3` | Per bar, chance a drum pattern gets mutated. |
| `mutation_amount` | `0.1` | Fraction of steps a mutation touches. |
| `glitch_chance` | `0.15` | Per bar, chance of a hihat stutter burst. |
| `gain_kick` | `0.95` | Kick level. |
| `gain_snare` | `0.8` | Snare level. |
| `gain_hihat` | `0.45` | Closed hihat level. `0` removes closed hats from the sequence. |
| `gain_ohat` | `0.4` | Open hihat level. `0` removes open hats from the sequence. Any drum volume at `0` drops that voice entirely. |
| `gain_ride` | `0.3` | Ride level. |
| `synth_kick`, `synth_snare`, `synth_hihat`, `synth_ohat`, `synth_ride` | `0` | Levels of a synthesized drum machine kit, see below. Each synth voice plays the same pattern as its sample voice, so the two layer; set the sample volumes to 0 for a pure drum machine. |
| `synth_style` | `auto` | Flavour of the synth kit: `house`, `techno`, `dnb`, or `auto` from the drum style (house for four-floor, ukg, minimal, one-drop, boom-bap; techno for halftime, idm, dembow, clave; dnb for dnb, breakbeat, footwork). In the GUI, picking a flavour also sets the five synth volumes to a mix that suits it (house leans on the open hat, techno on the kick, drum and bass on the snare); `auto` leaves the volumes alone. |

Both hats come out of the same source sample, so they read as one instrument. The trick that makes any sample pass for a cymbal: the transient at the onset is mixed with a cloud of tiny grains picked from the whole sample, which turns a flute or a voice into a noisy sizzle in its own colour, then a steep highpass takes the body away. The closed hat is 30 to 70 ms with a snappy decay and a touch of drive. The open hat is 150 to 450 ms with a slow decay, a denser cloud, a resonant band around 7 to 10 kHz for the ring of the cymbal edge, and a small room. Every open hat is cut off by the next closed hat in the bar, or by the next downbeat, with a 5 ms fade.

**Synth kit.** A drum machine in the 909/808 spirit, built from oscillators and noise with no sample involved, jittered a little per song. Kick: a sine with an exponential pitch drop and a highpassed click, soft-clipped. House sits around 48 Hz with a 0.4 s decay, techno drops to 44 Hz with a longer, harder, lowpassed body, drum and bass is tighter and punchier with more click. Snare: two detuned tones that dip in pitch like a 909 plus highpassed noise, snappy in house, darker and shorter in techno, bright and short in drum and bass. Hats: six square waves at the 808's metallic ratios through a bandpass, plus a little noise; the open hat is the same metal ringing longer, and it is choked like the sample one. Ride: inharmonic partials with their own decays, a noise shimmer and a stick click. When a sample voice is at 0 its samples stay free for textures and one-shots.

### Sub bass

A synth sub sits under the kick. Each section's background textures are analysed for their dominant pitch class, and that becomes the section's root note in the sub octave, C1 to B1 (33 to 62 Hz), fifths included, so it always stays a sub and never turns into a bass note. At most one note per bar, on the bar's first kick, and a weighted coin decides whether that bar gets one. Most notes play the root, some the fifth. Each note is a sine oscillator with an ADSR: a one-beat hold, then a long release that rings under the bar. It plays dry on the right and through a slow phaser on the left. The sub stays out of the intro and enters at the drop. Its level is matched to the mix, and the root per section is written to the JSON.

| Parameter | Default | Description |
| --- | --- | --- |
| `sub_level` | `0.6` | Sub level relative to the mix. `0` turns it off. |
| `sub_phaser` | `0.5` | Phaser depth on the left oscillator. `0` gives two dry sines. |
| `sub_pan` | `0.3` | Dry oscillator panned right, phased one panned left by this much. Keep it low for a mono-safe low end. |
| `sub_chance` | `0.6` | Per bar, chance of a sub note on the bar's first kick. |
| `sub_release` | `1.5` | Release in seconds after the one-beat hold. |

### Sections

| Parameter | Default | Description |
| --- | --- | --- |
| `section_bars` | `4,4,8,8,16` | Section lengths in bars to pick from. Repeat a value to make it more likely. |
| `break_chance` | `0.25` | Chance a middle section drops to a break. |
| `fade_in` | `0.3` | Fade-in at the start of the song, in seconds. |
| `fade_out` | `0.03` | Fade-out at the end, in seconds. The default is only a click guard; with an outro style the ending effect's own safety fade handles the silence, and with `none` the outro stays loopable in the Mixer. |

### Drops

A drop is the start of any non-break section. These transitions can lead into it. Each chance applies fully after an intro or a break, and at half strength between two grooves. `0` disables the effect.

| Parameter | Default | Description |
| --- | --- | --- |
| `cymbal_chance` | `0.3` | Reverse cymbal: the ride hit through a huge reverb, played backwards under an opening lowpass, so the wash builds for one or two bars and the hit lands on the downbeat. |
| `repeat_chance` | `0.25` | DJ beat-repeat on the last bar before the drop. The mix is captured and retriggered: accelerating roll, steady 16th stutter, or halving beat by beat, sometimes under a rising highpass. |
| `gap_chance` | `0` | Half a beat to two beats of silence right before the downbeat. Off by default. |

The transitions used are listed per section in the render's JSON.

### Layers and effects

| Parameter | Default | Description |
| --- | --- | --- |
| `layers_min` | `1` | Fewest background textures per groove section. Intros and breaks get 2 when the range allows. |
| `layers_max` | `3` | Most background textures per groove section. |
| `layer_level_min` | `0.18` | Quietest texture gain. |
| `layer_level_max` | `0.4` | Loudest texture gain. |
| `chop_chance` | `0.6` | Per section, chance of a rhythmic chop layer. |
| `chop_gain` | `0.4` | Chop slice level. |
| `fx_min` | `1` | Fewest random effects per texture chain. |
| `fx_max` | `3` | Most random effects per texture chain. The pool has 9. |
| `reverse_chance` | `0.35` | Chance a texture plays backwards. |
| `stretch_chance` | `0.3` | Chance a texture is time-stretched to whole bars. |
| `saturation` | `0.5` | Drive of the texture saturator. It is gain-compensated, so a saturated layer sits at the same level in the mix, only denser. `0` removes it from the effect pool. |
| `random_start` | `off` | Where cuts are taken from a sample. `off`: from its beginning. `one-shots`: the leftover one-shot events start at a random point in the file. `all`: background textures and collage intro hits too. Drum hits already seek the loudest transient and chops already sample the whole file. Use it for long recordings, especially ones with a quiet opening. |

Chances are clamped to 0-1 and swapped min/max ranges are sorted, so nothing you type can crash a render.

The GUI writes these as a JSON job file and runs `python3 gacha_engine.py job.json` in a subprocess, so the engine has no command line of its own.

### Outro styles

The ending is a ring-out effect in the spirit of Marmelade's "Ending FX" stage: the body of the song stays dry, the effect blends in over the last bars, then rings out into a tail appended past the last bar, and a safety fade guarantees true silence at the very end. The last section's `end_sec` in the JSON includes the tail, so the Mixer keeps the ending. In the GUI the backdrop video fades to black across the outro.

| Style | Ending |
| --- | --- |
| `hall_wash` | A huge hall swallows the last bars. |
| `dub_echo` | Dotted-eighth dub delay, synced to the tempo, feeding back into a room. |
| `tape_stop` | The reel slows to a halt, pitch dropping with it. |
| `filter_close` | A resonant lowpass closes the door. |
| `shimmer_freeze` | Octave-up shimmer frozen in a cathedral. |
| `bitcrush_collapse` | The mix crumbles into a few bits. |
| `codec_rot` | GSM and low-bitrate MP3 codecs chew the ending up. |
| `glitch_stutter` | A 32nd-note stutter with drive. |
| `overdrive_bloom` | Heavy overdrive blooming into reverb. |
| `smear` | Deep chorus smeared across a long room. |

| Parameter | Default | Description |
| --- | --- | --- |
| `outro_tail` | per style | Seconds of ring-out appended past the last bar, 2.5 to 7 by default depending on the style. |
| `outro_bars` | `2` | The effect blends in over the last N bars. |
| `outro_wet` | `1.0` | Maximum wetness of the ending effect. |
| `outro_amount` | `0.5` | The style's primary knob: decay, feedback or crush depth. |

### Drum styles

`four-floor`, `breakbeat`, `boom-bap`, `halftime`, `dnb`, `minimal`, `ukg`, `dembow`, `one-drop`, `footwork`, `clave`, `idm`

`random` picks one per song. Tempo is deliberately independent: dnb at 90 BPM is your call.

### Intro styles

| Style | Feel |
| --- | --- |
| `ambient` | Textures first, drums may sit out entirely. |
| `sparse` | One drum voice keeps time under the textures. |
| `none` | No intro. Full kit from bar 1. |
| `build` | Hats, then kick, then snare stack up bar by bar. |
| `drums-first` | The dry kit alone, textures join later. |
| `reverse-swell` | A reversed sample crescendos straight into the drop. |
| `collage` | Scattered one-shots set the scene, no beat. |
| `filtered` | Everything plays behind a lowpass that opens up, club-door style. |

## Output

Each render writes two files:

```
output/gacha_ember_20260830_213599_seed42_bpm120.wav   the song (44.1 kHz, 16-bit stereo)
output/gacha_ember_20260830_213599_seed42_bpm120.json  seed, bpm, style, every knob, samples used, section map
```

The word after `gacha_` is drawn from the seed, so reproducing a run gives it the same name.

The JSON section map is what powers the Mixer tab. Finals stitched in the Mixer are saved as `output/final_*.wav`.


# IDEAS

- generate a sample bank based on the sample bank :)
- have separated tracks for drum, layers, events etc...
- realtime audio (adc) and webcam
- add font decorative
- add real drums
- be able to select the samples to use for drum, layers etc..
- save parameter
- keyboard shotcut to trig video fx
- show the matrix build (intro, intro2, seq 1 - fx,) in a visual way
- sidechain
- apply fx on main for one bar (ie: reverb)
- libpd?
- slow techno a la rross implementation
- split output audio for the 4 layers