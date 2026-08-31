# Gacha

**A slot machine for songs.** Point it at a folder of audio samples and every run produces a brand-new track: drums carved out of random samples, textures looped and reversed behind them, glitch bursts, tempo-synced dub delays, and a different arrangement every single time.

No two pulls are alike. Every sample gets used.

## How it works

1. **Loads every sample** in `samples/` (wav, flac, aiff, mp3).
2. **Carves a drum kit** from randomly chosen samples: a short chunk is cut at the loudest transient, then sculpted into a kick (lowpass + drive), snare (bandpass + room), hihat (tight highpass slice) and ride (long, airy decay). Each role claims its own source samples, so a TV jingle can become your snare while a flute becomes the ride.
3. **Sequences a song** on a 16th-note grid at a single steady tempo: kick, snare, hats and occasional rides play style-driven patterns with per-bar mutations, ghost notes, fills, subtle swing and stereo spread. Hihats occasionally shatter into ultra-fast stutter rolls.
4. **Layers textures** per section: long samples are looped with crossfades, reversed, or time-stretched to exact bar lengths, then run through randomized effect chains (reverb, delay, distortion, chorus, phaser, bitcrush, lo-fi codec, ladder filter, pitch shift). A chop layer slices a sample into pieces and replays them rhythmically. All delays are synced to the tempo grid.
5. **Structures the result** into sections (intro, grooves, breaks, outro) and writes a mastered wav plus a `.json` section map, so any render can later be sliced and recombined.

## Requirements

Python 3.10+ and:

```bash
pip install numpy soundfile librosa pedalboard PySide6
```

[pedalboard](https://github.com/spotify/pedalboard) (Spotify's audio effects library) powers all effect chains. PySide6 is only needed for the GUI.

## Quick start

Drop your samples into `samples/`, then:

```bash
# one random 3-minute song
python3 gacha.py

# reproduce a run you liked
python3 gacha.py --seed 42

# fast 2-minute track, full drums from bar 1
python3 gacha.py --bpm 140 --duration 120 --intro-style none

# five pulls in one go
python3 gacha.py --count 5
```

Renders land in `output/`.

## GUI

```bash
python3 gacha_gui.py
```

All parameters as widgets, a live render log, and a built-in player:

* **Outputs** tab: every past render, newest first. Double-click to listen. New renders start playing automatically.
* **Samples** tab: audition the raw material.
* **Mixer** tab: the section editor. Pick any render, double-click its sections (intro, grooves, breaks, outro) to audition each slice, then assemble your favourite parts from *different* renders into a final arrangement. Reorder, repeat sections at will, and render the result to a single stitched track with seamless crossfades.

Drop a `gacha.mp4` next to the scripts and it plays as a looping ambient backdrop behind the interface.

## Options

| Flag | Default | Description |
| --- | --- | --- |
| `--seed N` | random | Reproducible run. With `--count`, seeds increment. |
| `--duration S` | `180` | Target length in seconds. |
| `--count N` | `1` | Number of songs to render in one go. |
| `--bpm N` | random 84-128 | Tempo. |
| `--drum-style S` | `random` | Rhythmic skeleton, see below. |
| `--intro-style S` | `ambient` | How the song opens, see below. |
| `--intro-bars N` | random | Cap the intro length in bars. |
| `--num-samples N` | all | Build the song from only N randomly picked samples. |
| `--pan-drums X` | `0.3` | Random pan spread per snare/hihat hit (kick stays centered). |
| `--pan-layers X` | `0.6` | Random static pan per background layer. |
| `--pan-events X` | `0.5` | Random pan per chop slice and one-shot. |

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
output/chop_20260830_213599_seed42_bpm120.wav   the song (44.1 kHz, 16-bit stereo)
output/chop_20260830_213599_seed42_bpm120.json  seed, bpm, style, section map
```

The JSON section map is what powers the Mixer tab. Finals stitched in the Mixer are saved as `output/final_*.wav`.
