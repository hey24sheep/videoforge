# Director prompt (master template)

Use this when asking Claude (or any LLM) to design a new episode for videoforge. Fill the `{…}` slots, paste the
whole block. The output is meant to become `series/<name>/<ep>/episode.yaml` (see README "Episode format").
The style list is generated from `styles/styles.yaml` (`python -m vf styles`); pick by key.
After writing the YAML run `python -m vf lint <episode.yaml>`: it enforces most rules below automatically.

```text
Act as an expert AI Cinematographer, Prompt Engineer, and Hollywood Scriptwriter specializing in local
open-source production pipelines. Design a {length: 60}-second sequence for {platforms: YouTube 16:9 and/or
Instagram/YouTube Shorts 9:16}, rendered locally on an RTX 4060 Laptop (8 GB VRAM) with the videoforge pipeline:
Z-Image-Turbo key frames -> Wan2.2-TI2V-5B image-to-video (5 s max per clip, steady AI camera) -> automatic clip QC
(re-rolls/repairs) -> RIFE 60 fps -> a virtual cinema camera in the compositor (eased push/pan/truck/orbit/crash
zoom/dolly zoom + handheld + motion blur + depth parallax) -> Chatterbox voices (ASR-checked) -> procedural sound.

Topic / premise: {premise}
Series bible (characters, world, tone) if any: {bible}

[STYLE] anime | neo_noir | comic | realistic | clay3d | isometric | cyberpunk | ink_wash | papercraft | pixel
(or honour the requested style: {style or "free choice"}); say why it suits 8 GB small-model limits.

[DELIVER]
1. Style choice + one-paragraph justification.
2. Shot list of {n: 10-14} shots (each needs <= 5 s of AI clip; dur x speed <= 5), each with:
   - id, dur, transition, and `camera:` (the move is made by the compositor, never by the video model):
     {move: push_in|pull_out|pan_left|pan_right|tilt_up|tilt_down|truck_left|truck_right|drift|crash_zoom|
      dolly_zoom|orbit_left|orbit_right|rise|settle, amount: 0.03-0.12, handheld: 0-0.4, ease: io|out}
   - KEYFRAME prompt: the complete first frame. Every prop, light source, wet/dry/smoke state the motion will use
     must already be visible here. Reuse the character sheet `{name}` verbatim.
   - MOTION prompt: what moves inside a STILL frame, physically. No camera words.
   - on-screen text if any (titles centred at x 0.5; captions are automatic)
3. Voiceover & dialogue: dramatic, mature, motive-driven; no flat or childish lines. Per line: speaker, `at`,
   text (`say:` when spoken wording differs from the caption, e.g. numbers), delivery notes.
4. Sound cues with timestamps (synth: rain, drone, piano, ring, heartbeat, thunder, hit, riser, tick).
5. The story must be SHOWN, not only told: every twist line needs a shot that proves it (a legible name on a page,
   an object in hand) - add compositor text/inserts where the model can't draw it.
Return the shot list, dialogue and cues as YAML that matches videoforge's episode.yaml schema.
```

## Rules that make shots artifact-free on this hardware (learned; `vf lint` checks the marked ones)
- **One action per shot, <= 5 s of clip.** Longer clips drift; chain shots and cut on action. [lint]
- **The AI camera never moves.** Camera words in a motion prompt make piers, rails and buildings wobble. Write the
  move in `camera:`; the compositor makes it exactly, with motion blur and parallax. [lint]
- **Every shot moves.** No dead stills: at least a slow push or drift, plus a little handheld where it suits.
- **Props and states live in the keyframe.** Smoke needs a visible cigarette/ashtray/vent; a face that drips must be
  described wet; a receiver at the ear starts at the ear. The video model can't add what the first frame lacks. [lint]
- **No hand-to-object actions** (lift, pick up, hand over, hang up, turn around): props teleport and hands melt.
  Use held poses (grip tightens, fingers shift) and cut on the action. [lint]
- **No walking or running close enough to see legs.** Legs skate at 480p. Silhouette wides, legs out of frame, or
  a `truck_*` camera move past a standing figure. [lint]
- **Mouths stay closed during voice-over** ("lips pressed together"); there is no lip-sync yet. Prefer the speaker
  off-screen, from behind, or in shadow when a line plays. [lint]
- **Motionless subjects are 2.5D stills** (`gen: {still: true}`): an empty room, a coat on a rail, a ledger, a
  figure standing still in the rain. The keyframe is moved by the virtual camera with depth parallax, plus a
  `rain` layer outdoors. Video models invent walk-ins, hands and gaits in "nothing moves" shots (judge-confirmed on
  a first episode: 6 of 14 shots); a camera move over a still cannot. Use AI motion only where something must move.
- **One identity anchor per character and location**: the same `{name}` sheet and, for recurring places, the same
  keyframe wording (a recurring location must not change design between shots).
- **Indoor shots get no rain/sky words**; the style's motion wrapper is neutral, keep it that way.
- **Positive phrasing only** for distilled models (CFG 1): negatives are ignored.
- **Text belongs to the compositor, never to the video model.** Titles at x 0.5. [lint]
- **Stylised beats realistic on 8 GB**: papercraft, ink wash, clay 3D, comic, isometric and pixel hide small-model
  errors; realistic faces/hands are judged hardest.
