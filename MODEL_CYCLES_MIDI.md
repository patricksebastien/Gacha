# Elektron Model:Cycles: MIDI reference for Gacha

What the box can do over MIDI, and what that means for Gacha. Compiled
2026-09-28 from the official Model:Cycles user manual (OS 1.13, the current
and final OS, released 2021-05-26, Appendix A and chapter 12.3) plus the
OS 1.13 release notes and Elektronauts threads for the parts the manual
leaves out. Anything marked *verify* is not in the manual and needs a
minute with the device. Since 2026-10 the box runs the Model-TG firmware
(a patch over OS 1.13 with a Sampler machine); what that changes is in
section 9, everything before it is the stock surface, which Model-TG
keeps.

Hookup here: the Model:Cycles hangs on the UMC204HD's DIN MIDI, so Gacha
sees it as `UMC204HD 192k MIDI In` (from the box) and `UMC204HD 192k MIDI
Out` (to the box). No USB MIDI port of its own is enumerated. DIN is
31.25 kbaud: about 1 ms per 3-byte message, so a burst of six notes on one
tick spreads over ~6 ms.

## 1. Channels

Six tracks, each with its own receive and send channel, plus an AUTO
channel, an FX channel and a program-change channel. All set in
CONFIG > MIDI > IN CHAN / OUT CHAN, each can be OFF.

| Setting | Default | What it carries |
|---|---|---|
| T1..T6 IN | 1..6 | notes (chromatic play of that track) and CC for that track's parameters |
| FX IN | *verify* (own channel) | CC 85..88 delay and reverb parameters |
| AUTO IN | *verify* (10 on other Elektrons) | whatever the *active* track is: notes 0..5 trigger tracks 1..6, notes 12..60 play the active track chromatically |
| PCH IN | AUTO or a channel | program change in |
| T1..T6 OUT | 1..6 | what the track sends when MOUT is on: notes, mutes, knob CCs |
| FX OUT | own channel | delay and reverb knob CCs |
| PCH OUT | channel | program change out on pattern change |

Several tracks may share one channel; the lowest-numbered track wins
conflicts.

## 2. Notes in (Gacha or anything else playing the box)

Gate needed: CONFIG > MIDI > FILTER > RCV NOTE on. INP FROM must be MID or
M+U.

* On a **track channel** the track plays chromatically. Note 60 (C4) is
  the root, the pitch the pad itself plays (set by ear 2026-09-28; the
  pad's default note is TRACK SETUP > PAD and is what MOUT sends).
* On the **AUTO channel**: notes 0..5 (C-2..F-2) fire tracks 1..6 at their
  pad note regardless of which track is active; notes 12..60 play the
  active track over 49 semitones. Same thing the [TRIG] keys do in
  CHROMATIC mode, KBT (-60..60) transposes that range.
* Velocity is honoured (pads are velocity sensitive; PAD menu FIX/VDEP/DST
  decides what it modulates). Sequencer trig VEL is 0..127.
* Note off matters only when the track's GATE is on: the amp envelope then
  holds until note off or LEN, else decays from note on and ignores note
  off. So for drum-style hits gate is irrelevant; for Tone or Chord tracks
  with GATE on, Gacha's gate time becomes the note length.
* All Notes Off (CC 123) is honoured since OS 1.13.
* Notes received on the track or AUTO channels record into the sequencer
  in LIVE RECORDING mode.

## 3. CC in (Appendix A, all per track channel unless FX)

Gate needed: FILTER > RCV CC on. Send on the track's IN channel; the FX
CCs on the FX IN channel. Values are 0..127; bipolar parameters sit at 64
for zero.

Track parameters:

| Parameter | CC | Range on the box | Notes |
|---|---|---|---|
| Note | 3 | C0..G10 | the trig note (*verify* mapping) |
| Mute | 94 | off/on | *verify* polarity: on other Elektrons 0 = unmuted, 1+ = muted. Global mute, survives pattern change |
| Track level | 95 | 0..127 | the LEVEL knob, not p-lockable |
| Pan | 10 | L..R | 64 = centre |
| Pitch | 65 | -24..+24 semitones | 64 = 0; **incoming CC jumps are slewed**, a big jump audibly glisses (firmware quirk, no off switch); p-locks on the box itself step instantly |
| Decay | 80 | 0..127 | |
| Color | 16 | 0..127 | machine dependent |
| Shape | 17 | 0..127 | machine dependent; Chord: picks the chord (Appendix C) |
| Sweep | 18 | 0..127 | machine dependent |
| Contour | 19 | 0..127 | machine dependent |
| Delay send | 12 | off, 1..127 | |
| Reverb send | 13 | off, 1..127 | |
| Volume + Dist | 7 | 0..127 | 60 = clean unity, above it distorts |
| Swing | 15 | 50..80 % | |
| Chance | 14 | 0..100 % | |
| Machine | 70 | Kick, Snare, Metal, Perc, Tone, Chord | added in OS 1.13; *verify* value per machine (menu order 0..5 is the guess) |

Playback: Punch CC 66, Gate CC 67 (on/off toggles, p-lockable).

Retrig: none. The RETRIG key (chapter 9.6: rate 1/1..1/80, length, fade,
always-on, per track, stored in the pattern) has no CC, no NRPN and no
note message; the box retrigs only from its own pads and [TRIG] keys.
From outside, a retrig is just the note sent again at the rate (Gacha's
M:C tab does that, with the box's velocity curve).

Velocity modulation (Appendix B, the PAD menu's VDEP/DST): what a note's
velocity shapes, pitch to LFO depth; set on the box, not over CC.

LFO (per track): Speed 102 (-64..63, 64 = 0, negative runs backwards),
Multiplier 103 (x1..x2k tempo-synced or fixed 120 BPM set), Fade 104
(-64..63), Destination 105 (Appendix B list: pitch, fine tune, decay,
color, shape, sweep, contour, delay send, reverb send, vol+dist, pan,
punch, gate), Waveform 106 (tri, sine, square, saw, env, ramp, random),
Start phase 107 (0..127), Reset 108 (off/on), Depth 109 with LSB 110
(-64.0..63.0).

FX (on FX IN channel): Delay time 85 (in 128ths, tempo-relative), Delay
feedback 86, Reverb size 87, Reverb tone 88 (-64..63).

NRPN: the FILTER page says "CC/NRPN" but Appendix A documents CCs only and
no NRPN table exists anywhere public. Treat the box as CC-only.

Not implemented: pitch bend, mod wheel, aftertouch, breath, bank select,
RPN. Nothing in the manual, nothing on the forums.

## 4. Program change and transport in

* SYNC > PCH IN on, channel PCH IN. Program 0..95 = pattern A01..F16
  (bank = program // 16, pattern = program % 16). A change while running
  is **queued to the end of the current pattern**; the box has no
  immediate or quantised-to-bar option. Pattern chains are volatile, not
  reachable by MIDI.
* SYNC > CLK IN on: the box follows MIDI clock (0xF8) **and** transport
  (start 0xFA, continue 0xFB, stop 0xFC) from the same port. No way to take
  clock without transport or vice versa. Song position pointer is not
  mentioned. OS 1.13 fixed timing issues with external clock + transport.
* Tempo can also be tapped or set on the box; there is no Ableton Link.

## 5. What the box sends out

* **MOUT** per track (TRACK SETUP, [FUNC]+[TRACK]): the track's sequencer
  NOTE, VEL and LEN go out on its T OUT channel as note on/off, the pads
  and the chromatic [TRIG] keys send notes, mute on/off sends (CC 94),
  the track knobs and FX knobs send their CCs when FILTER > ENCS is I+E,
  and notes arriving on the AUTO channel are forwarded to MIDI OUT. The
  track keeps sounding internally too. Only NOTE, VEL and LEN leave; no
  retrigs, no p-locks.
* SYNC > CLK OUT: MIDI clock and start/stop when the box is the master.
* SYNC > PCH OUT: a program change on the PCH OUT channel on every pattern
  change.
* OUT TO must be MID or M+U (M+U throttles USB to DIN speed). OUT/THRU
  must be OUT, otherwise the port just echoes MIDI IN and Gacha would get
  its own notes back.

## 6. SysEx

* Backup (CONFIG > BACKUP) sends the project or the current pattern as
  SysEx over DIN or USB; the box listens for incoming dumps at all times
  and a received project **overwrites the active project**. Presets and
  preset folders can be sent as SysEx from the +Drive file menus.
* OS upgrade is a SysEx transfer in the startup menu.
* The format is not published. The only reverse-engineered scrap is the
  Elektron manufacturer header `F0 00 20 3C`, device id `0x11` for Cycles
  (`0x0F` Model:Samples), and a "get kit" request
  `F0 00 20 3C 11 00 6A 01 01 00 00 00 00 05 F7` with 7-bit-packed
  payloads like the Analog Rytm (libanalogrytm). No pattern or preset
  byte layout exists. Real-time parameter control over SysEx is not
  a thing on this box; everything live goes over CC and notes.
* Elektron Transfer 1.10.4 (2026-06) is the official tool and does the
  same dumps over USB.

## 7. What this means for Gacha

Already there in `gacha_midi.py` and `gacha.py`:

* **Box -> Gacha actions.** MidiIn takes note on, CC and program change on
  any channel and the Controls tab learns them, so with MOUT on a track
  and ENCS I+E, a pad, a knob or the sequencer's own notes on channels 1..6
  can drive any Gacha action today, and PCH OUT can pick sections by
  program number. Pick `UMC204HD 192k MIDI In` in the inputs.
* **Gacha -> box, drum hits.** MidiOut sends the drum hits as notes on one
  channel with a fixed note per role. Aiming it at `UMC204HD 192k MIDI
  Out` on the AUTO channel with the map kick 0, snare 1, hihat 2, ohat 3,
  ride 4, glitch 5 fires tracks 1..6 at their pad notes without touching
  the code (needs RCV NOTE on and the AUTO channel set to Gacha's out
  channel).

* **Gacha -> box, one track per drum (done 2026-09-28).** The
  *Model:Cycles* tick in the MIDI out panel sends each role on its own
  track channel (`CYCLES_CHANNELS` in `gacha_midi.py`: kick 1, snare 2,
  closed hat 3, open hat 4, ride 5, hat stutter 6) and the note map
  becomes the pitch per track, root C4 = 60 (`CYCLES_ROOT`, the *M:C*
  button; 36 was tried first and sounded off, 60 is the pad's own pitch).
  Box side: RCV NOTE on, T1..T6 IN at 1..6, INP FROM MID.

* **Clock and transport out (done 2026-09-28).** The *MIDI clock* and
  *start / stop* ticks on the panel's sync row: 24 ppqn from the sender
  thread, free-running at the live tempo, slewed onto the section's beat
  grid while one plays (frame positions and perf_counter, since this
  Python's monotonic clock is 15.6 ms coarse on Windows), Start half a
  millisecond before the downbeat's tick from silence, Stop when the
  section clears on its bar. Box side: SYNC > CLK IN on. Clock in (the box
  as master) is still open.

* **CC out and the M:C tab (done 2026-10-07).** `MidiOut.cc(channel, cc,
  value, force)` on the same port and thread: CCs gathered for 2 ms and
  sent together, one value per controller, a repeat of the last value
  dropped (force sends it), at most one value per controller every 10 ms
  (`CC_MIN_S`) so an LFO on a knob cannot drown the notes on the DIN wire.
  The tables (`CYCLES_PARAMS`, `CYCLES_FX_PARAMS`, the menus, the retrig
  lists) and `cycles_cc_value()` in `gacha_midi.py`; the tab in `gacha.py`
  (`_tab_cycles`): a cell per parameter per track on channels 1..6, the
  FX pair on the FX channel box, all of them MIDI-learnable, LFO and macro
  destinations, and the "mc" scene group. Menus go out as the menu index,
  the mute as 0/127: to verify by ear, like the AUTO channel default.
* **Retrig from here (done 2026-10-07).** `MidiOut.retrig_start(channel,
  note, velocity, period_s, length_s, fade)` / `retrig_stop(channel)`: the
  note again every period from the sender thread, the velocity curve as
  the box's FDE and LEN; a hold button per track on the tab, held from a
  MIDI note or CC. Sent as plain notes, so the box needs RCV NOTE only.

Not there yet, in rough order of payoff:

1. **Clock and transport in.** MidiIn drops 0xF8/0xFA/0xFC; the box as
   master would need a tempo estimate from the ticks into the live clock.
   One direction only, never both.
2. **CC out, the next layer.** The tab's cells are knobs, so the mutes
   under the F-keys, the sends under the A/B pedal, a reroll of color /
   shape / sweep / contour at a section start are a macro, an LFO or a
   scene away; what is missing is the automatic part (reroll per section,
   the pedal as a crossfader into FM). Mind the pitch slew.
3. **Program change out.** `MidiOut.pc(channel, program)` when a section
   starts, so the box's pattern follows Gacha's sections; it lands at the
   end of the box's current pattern, so send it a pattern early or keep
   the box's patterns the length of a section.
4. **Feedback guard.** With MOUT tracks sending into Gacha and Gacha's
   MidiOut sending back to the same box, a learnt binding on channels 1..6
   can trigger on Gacha's own echoed notes if OUT/THRU is ever set to
   THRU. Ignore the M:C input while its notes are ours, or filter by the
   out channel.

Settings to make on the box for the first test:
FILTER: RCV NOTE on, RCV CC on, ENCS I+E. PORTS: INP FROM MID, OUT TO MID,
OUT/THRU OUT. SYNC: CLK IN on (when Gacha sends clock), PCH IN on. IN
CHAN: T1..T6 = 1..6, AUTO = 10, PCH = AUTO. OUT CHAN: T1..T6 = 1..6, PCH =
10. TRACK SETUP on each track: MOUT on.

## Sources

* Model:Cycles User Manual, OS 1.13 (2024-10-30 PDF), chapters 7.5, 7.6,
  9.4, 9.8, 10, 12.3, 12.5, Appendix A, B, C.
  https://www.elektron.se/wp-content/uploads/2024/10/Model-Cycles-User-Manual_ENG_OS1.13_241030.pdf
* OS release notes: https://www.elektron.se/release-notes/modelcycles-os-release-notes
* Support page (OS 1.13, Transfer 1.10.4): https://www.elektron.se/support-downloads/modelcycles
* CC list cross-check: https://midi.guide/d/elektron/model-cycles/
* SysEx header and device id: https://www.elektronauts.com/t/more-midi-info-sysex-format-etc/125714
* CC slew quirk: https://www.elektronauts.com/t/model-cycles-smooths-incoming-cc-changes/253912
* Mute CC per track channel: https://www.elektronauts.com/t/midi-in-model-cycles-from-roland-fc-300/218650
* MOUT and track channels: https://www.elektronauts.com/t/model-cycles-as-midi-keyboard/172750, https://www.elektronauts.com/t/model-cycles-external-midi-in-chromatic-play/182468
* OUT/THRU and port settings: https://www.elektronauts.com/t/how-to-send-receive-midi-cc/169871

## 8. Ideas (2026-09-28)

What the FM box and the Erae could bring to the set, grounded in the arc
(tape, FX, takes, sections under the pedal, back to tape). Tags: *now*
works with what is built, *CC* / *PC* / *clock* need that piece of code.

The box as Gacha's instrument:

* Carved kit becomes an FM kit per section (*now*; silence own drums).
* Reroll the kit like the samples: machine (CC 70) and random color,
  shape, sweep, contour within ranges at every section start (*CC*).
* Style to voice: house punchy kick with punch on, techno long and dark,
  drum and bass tight with the click up (*CC*).
* Sub bass tuned to the tape: a Kick machine track with a long decay sent
  the section's root + 60 minus octaves; a Tone machine as synth bass
  (*now*, map only).
* Chord drone in key: Chord machine on T6 fed the section root, chord by
  shape, wavetable sweep moving over the section (*CC* + a new role).
* Drop spikes reverb size and delay feedback on the FX channel; swell ramps
  LFO depth; break mutes all but the kick via CC 94 (*CC*).
* Pedal as crossfader into FM: track levels and sends follow A/B; never
  pitch, the box slews pitch CC (*CC*).
* The box's LFO as free automation per section, tempo-locked once clocked
  (*CC* + *clock*).
* Velocity shapes timbre through the box's PAD menu (*now*, box side).
* Two sequencers on six voices: the box's own patterns run under Gacha's
  hits, a program change per section picks which (*PC*).
* Chance (CC 14) and swing (CC 15) from the grade knob (*CC*).
* Recursion through the takes: box audio into the second input and the
  live rack, record the FM kit as a take, let the engine carve the next
  kit out of it (*now*, no code).
* The FM kit through Gacha's racks and VST chain (*now*, same routing).

The box driving Gacha:

* MOUT tracks as a sequenced controller: conditional trigs (1:4, fill)
  bound to burst or glitch, chance trigs bound to shader or picture
  events (*now*).
* Pads with velocity as six more buttons (*now*).
* One knob two worlds: ENCS I+E, the color knob also sweeps a rack or the
  rgb shift (*now*).
* PCH OUT as the setlist: the box's pattern chain walks next, drop, break
  (*now*).
* Box as clock master, tap tempo on the box (*clock in*; one direction of
  clock only, ever).
* A picture sequencer: a MOUT track at zero volume driving video glitches
  (*now*).

The Erae on top:

* Six finger pads: pressure = velocity, x = pitch, y = color, floats at
  44 Hz so a drag is a continuous timbre morph.
* The mixer the box lacks: six level faders (CC 95) with LEDs following,
  press to mute, sends on a second row.
* Chord zone in key: x = chord (shape), y = sweep, pressure = velocity,
  note = section root.
* Drones under pressure: GATE on, the held finger holds the note, Kick
  machine as an 808 drone.
* LFO gesture zone: speed by x, depth by pressure, zero on release.
* Machine morph zone: x picks the machine, y a macro preset; the randomize
  pad rerolls the box with the effects.
* LEDs pulse per hit so the surface shows the kit playing.

Order to start: take recursion and tuned sub bass (no code), then CC out
(reroll, drop, pedal crossfade), then clock out (delay and LFO lock), then
the chord role and the Erae pads once the box's values are confirmed by
ear.

## 9. Model-TG firmware (2026-10-08)

The box now runs [Model-TG](https://github.com/TinyGregAudio/Model-TG)
(checked at commit 70b39dd, 2026-10-02): unofficial code patched into OS
1.13 that adds a seventh machine, the **Sampler** (seven playback modes:
one shot, loop, slice, granular, stretch, pluck, wave; stereo; slicing by
equal parts, transients or markers; resampling of a track, the mix or USB
audio), a retrig page with beat-repeat and twelve master FX, **Attack,
Filter and Resonance** on the five stock machines, slide trigs, scale lock
and a lower CPU load. Its docs (`docs/USER_GUIDE.md`, `docs/INTERNALS.md`)
say nothing about MIDI; everything below is read out of the firmware
source (`src/model_tg.s`, `src/phase1.s`) and is *verify* on the box.

What the MIDI surface becomes:

* **The stock CC table stands.** Model-TG patches the UI, the voices and
  the save format; it adds no CC handler and no NRPN, and the only MIDI
  code in it is the mute setter (so CC 94 out still follows a mute) and
  some diagnostics that no shipped build installs.
* **Machine CC 70: the Sampler is index 6.** The stock table clamps the
  machine index to 5 (Chord); Model-TG lifts that, and every lookup it
  patches treats 6 as the Sampler. Whether the CC path takes the raw index
  or a scaled value is the same open question as the stock menus
  (*verify*).
* **The Sampler's four knobs are the machine slots.** Its page writes
  Sample Start, Sample End, Filter and Resonance into the track words the
  stock page writes for Color, Shape, Sweep and Contour (trackData +22,
  +24, +26, +28), so CC 16..19 should land on them. In Granular, Stretch,
  Pluck and Wave the same four words take new jobs (Position / Grain size
  / Density / Spray; Speed / Grain size / Density / Spray; Strike pos /
  Strike len / Tone / Feedback; Wave pos / Frame size / Smooth / Detune):
  the parameters do not change, only what they do. The mode itself is set
  on the box only (SETTINGS + PRESET), no CC. *verify* that CC 16..19 reach
  a Sampler track: the stock CC handler might resolve them through the
  per-machine array that has no row for machine 6, in which case they do
  nothing there.
* **Pitch CC 65 on a Sampler track pins the pitch** and ends note
  tracking (the firmware author measured it: the common Pitch parameter is
  kept off the Sampler page for that reason). Tune a Sampler track by the
  note instead. Decay (CC 80), level, pan, sends, volume, mute, chance,
  swing, gate and the LFO CCs are the common parameters and stay as they
  are.
* **Slice mode: the note is the slice.** Slice 1 is note 52 (the first trig
  key in keyboard mode), one slice a semitone, so the root 60 Gacha sends is
  slice 9 and 52..67 are the sixteen slices (17..32 above). Slices play at
  their own speed.
* **Attack, Filter, Resonance (stock machines): no CC.** They are real
  parameters (indices 23, 24, 25; two of the firmware's "Error"
  placeholders given a job) that p-lock, save and take the LFO, but no CC
  number maps to them. They are new **LFO destinations** (CC 105) after
  the stock list on an FM track; a Sampler track does not list them (its own
  filter is in Sweep / Contour). The exact menu position is *verify*, like
  the rest of that list.
* **Mute on a Sampler track** (CC 94 or FUNC + pad) fades it out and ends
  its note instead of cutting.
* **Retrig page, master FX, resampling, slide trigs, scale lock:** buttons
  and knobs on the box only, nothing over MIDI. The box's new beat-repeat
  does not change the retrig Gacha does from the M:C tab (still plain
  notes). Delay and reverb switch off while nothing sounds, which changes
  nothing on the wire.
* **Transfer identity (CYC / SMP)** is for uploading samples over USB with
  Elektron Transfer, not DIN MIDI; it resets to CYC at every power-on.

In Gacha (2026-10-08): `CYCLES_MACHINES` gained "sampler" (index 6),
`CYCLES_LFO_DESTS` attack / filter / resonance at the end,
`CYCLES_SAMPLER_MODES` the four knobs' names per mode, `CYCLES_SLICE_NOTE`
52. The M:C tab: a track whose machine cell says sampler shows the color,
shape, sweep and contour cells with the mode's names as a prefix (the job
in the tooltip), its pitch cell is greyed and never sent (send all
included), and a **mode** row per track (a local menu, not sent, remembered
with the tab's values) tells the tab which mode the box is in so the names
match.

To verify with the box the first time: the machine menu reaches the
Sampler at CC 70 = 6; CC 16..19 move Start / End / Filter / Res on a
Sampler track; a Model:Cycles tick drum on a Slice-mode track picks slice 9
at the root and walks the slices with the note map; the LFO destination
menu's new entries.
