# videoforge reference

Details behind the README: the episode spec, the cast lock, styles, camera, caching, quality gates and models.
Start with the README; come here for the full field list.

## Episode format (episode.yaml)
```yaml
title: My Episode
style: papercraft            # optional; else the series style
fps: 60
look: {bloom: 0.3}           # overrides style/series grade
gen: {size: [832, 480], retries: 2}    # defaults for every shot's gen: block
camera: {move: push_in, amount: 0.05, handheld: 0.2}   # default virtual camera (style/series/episode)
audio:
  music: {src: ../assets/music.wav, gain_db: -9, duck_db: -8, fade_in: 1, fade_out: 2}
  beds: [{src: "synth:rain?dur=60", gain_db: -17}]          # ambience, never ducked
  sfx: [{src: ../assets/hit.wav, at: 12.4, gain_db: -3}, {src: "synth:ring?dur=3", at: 0.2}]
  # src may be a file or synth:<rain|drone|piano|ring|heartbeat|thunder|hit|riser|tick>?key=value (vf/synth.py)
dialogue:                    # wav: file, or text spoken by the series voice (TTS stage), or dur: for silent captions
  - {who: narrator, at: 1.0, text: "Line.", wav: ../voice/l1.wav, note: "delivery notes"}
  - {who: caller, at: 4.0, text: "2:40.", say: "Two forty.", fx: phone, exaggeration: 0.7}
shots:
  - id: ai_shot              # AI footage: key frame (text->image) -> clip (image->video), cached
    dur: 5
    gen: {keyframe: "the complete first frame: every prop/state the motion uses, {name} from series characters",
          motion: "what moves inside a still frame (no camera words)", seed: 1, keyframe_seed: 1,
          keyframe_candidates: 3, image_from: other_shot_id, speed: 1.0, profile: fast, sharpen: 0.35,
          still: false,      # true = no AI motion: the keyframe moved by the camera (2.5D parallax) - see director.md
          judge: [{q: "Is a telephone receiver held against the man's ear in the last frames?", expect: yes}],
          qc: {locked: true, people: true, smoke: false, water: false}, lip_sync: false}
    camera: {move: dolly_zoom, amount: 0.12, handheld: 0.1, parallax: 1.0, ease: io}   # vf/camera.py
  - id: open                 # optional
    dur: 5
    transition: {type: fade, dur: 0.5}     # cut | fade | dip | dip_white | flash
    # moves: push_in pull_out pan_left/right tilt_up/down truck_left/right drift crash_zoom dolly_zoom
    #        orbit_left/right rise settle none; + handheld, blur (180-degree shutter), parallax (depth), focus,
    #        start/end, ease; legacy keyframes zoom/roll/shake stack on top. `camera: false` = locked off.
    camera: {move: push_in, amount: 0.06, handheld: 0.15, blur: 1.0, zoom: [[0, 1.0], [5, 1.02]]}
    flash: [[2.4, 0.6]]                     # impact flashes [time, strength]
    allow_black: false                      # silence the black-frame QC for intended blacks
    caption_y: 0.42                         # move this shot's captions off a clue (Shorts safe y 0.10-0.78)
    layers:                                 # drawn bottom to top
      - {type: video, src: clips/open.mp4, fit: cover, focus: [0.5, 0.5], zoom: 1.0, speed: 1, loop: false}
      - {type: image, src: ../assets/bg.png, zoom: [[0, 1], [5, 1.1]], focus: [[0, [0.4, 0.5]], [5, [0.6, 0.5]]]}
      - {type: sprite, src: cutout_frames/, x: 0.3, y: 0.92, h: 0.7, fps: 16, anchor: [0.5, 1.0], mirror: false}
      - {type: text, text: "TITLE", font: title, size: 0.08, x: 0.5, y: 0.4, at: 0.5, until: 4.5,
         anim: fade|pop|slide|type, stroke: 0.04, plate: 0.0, short: {y: 0.3}}   # per-format override
      - {type: gradient, top: '#101820', bottom: '#000', glow: '#ff9a4a', glow_at: [0.5, 0.6]}
      - {type: text, in_scene: true, text: "1205", font: segoepr.ttf, rotate: -4, soft: 0.05, blend: screen}
        # in-scene props the model can't draw: writing in fog, ink in a ledger (blend: multiply), a stamp
      - {type: rain, intensity: 0.75, angle: 7, speed: 1.0, opacity: 0.14, layers: 3}   # procedural depth rain
      - {type: solid, color: '#000', radial: {center: [0.5, 0.42], radius: 0.28}, opacity: [[0, 0], [2, 0.9]]}
        # a light source dying: darken the bulb and its glow, not the whole frame (invert: true = all but it)
```
Any value can be keyframed `[[t, v], ...]` (shot-local seconds, eased). Any layer can override settings per
format with `yt: {...}` / `short: {...}` (yt4k inherits `yt:`); 9:16 uses the same normalised coordinates on a
vertical frame, so move the focus / text for vertical framing there. Text must stay in the platform safe area
(QC error otherwise): 9:16 x 0.06-0.88 (the right strip holds the like/comment buttons), y 0.10-0.78.

**Text is an overlay by default:** titles and captions are drawn after grading and camera moves, so white stays
white and text never shakes. Add `in_scene: true` to a text layer to make it part of the scene (graded, moved by
the camera). **Colour:** sources are display images; the default `tonemap: knee` keeps them exact and only rolls
off highlights (bloom, flashes); `tonemap: aces` gives a filmic curve. Output is BT.709-tagged H.264 with dither.

## Cast lock: the same face in every shot and every episode (vf/cast.py)
Add a cast sheet to `series.yaml` once; every episode of the series then uses it with no per-shot work:
```yaml
cast_lock: true                # whole series; cast_lock_except: [ep01] keeps listed episodes on the old path
cast_threshold: 0.40           # SFace cosine: same person ~0.55-0.89, different people ~0.02-0.33
cast:
  hero:
    name: Ada Quinn            # name words ("Ada", "Quinn") tag shots whose prompt mentions them
    description: "..."         # optional: defaults to characters.hero ({hero} in prompts)
    refs: [cast/hero_ref1.png]  # 1-3 clear images of the face (series folder relative)
    aliases: [the courier]     # optional extra words that tag a shot
```
**Add a character:** generate or pick 1-3 clear, frontal-ish images of the face, put them in `series/<name>/cast/`,
add an entry like the one above (plus `characters.<key>` if prompts use `{key}`), then check the tagging with
`python -m vf.cast plan <episode.yaml>`. Per episode: `cast_lock: true|false` and `cast: {key: {refs: [...]}}`
override the series (new wardrobe, ageing, a guest). Per shot: `cast: [hero, rival]` (explicit), `cast: []`
(nobody), `gen: {cast_candidates: 2, cast_rerolls: 1, cast_klein: true, cast_refine: false, cast_seed: 7}`.

For each cast shot the keyframe stage draws Z-Image candidates, redraws each with FLUX.2 Klein using the
references' face crops, scores every face with SFace (2x upscale and rotated passes catch small and tilted
faces), keeps the best one above the threshold and re-rolls new seeds below it (bounded), keeping the best take
and flagging it. The report `out/cast_qc.json` (also printed) scores each cast shot's keyframe and four sampled
frames of its finished clip; `python -m vf.cast qc <episode.yaml>` writes it for any episode without generating.

## Styles
`styles/styles.yaml`: anime, neo_noir, comic, realistic, clay3d, isometric, cyberpunk, ink_wash, papercraft, pixel.
Each sets prompt wrapping, negatives, grade, motion timing (stepped vs interpolated) and model defaults, plus an
"8 GB artifact risk" rating. Precedence: default < style < series < episode.

## Camera and motion (vf/camera.py, vf/interp.py)
The AI model always films with a steady camera (moving it makes structures wobble); every shot then gets a real
camera move in the compositor: eased presets, organic handheld sway, motion blur from a 180-degree shutter and 2.5D
parallax from a depth map of the keyframe. Clips are RIFE-interpolated to exactly `fps / speed` frames per second,
so 24 fps AI footage plays smoothly at 60 fps, slow motion included (no 2:3 judder). ~1 ms/frame.

## Caching (why re-renders are fast)
`vf/memo.py`, three tiers, key = function code (+ every videoforge function it calls) + inputs (arrays by
content, files by size+mtime). AI generators (i2v, t2i, TTS, prompt embeddings, clip QC, retime, identity) key on
an explicit `GEN_VERSION`/`version=` instead of their code: editing loaders or speed code never throws away
generated media; bump the version when the output would change.
- `mem`: per process (decoded sources, text bitmaps, GPU frames)
- `project`: `<episode>/.vfcache` (shot segments, audio mix)
- `global`: `videoforge/cache` (AI clips, upscales, TTS lines, key art: reused across every series)
Change a shot -> only its segment re-renders. Change renderer code -> all segments re-render (still ~32 s/format).

## Quality gates (vf/qc.py)
Every frame: black / blown-out frames, flicker spikes not at cuts/flashes (stored with each cached segment, so
cache hits still report them). Layout: each text's rendered box inside the safe area (checked before rendering).
Sources: straight-edged matte slabs in cut-outs. QC errors stop the build (layout) or withhold the final file
(frames) unless `--force`; warnings are listed in `out/qc_<fmt>.json`.

## Models (tuned for an 8 GB RTX 4060) - details in docs/video_model.md, docs/model_optimization_plan.md
All open models, run locally; nothing calls a paid API.
| Stage | Model | Notes |
|---|---|---|
| Video, default | Kandinsky 6.0 Lite distill (`profile: k6`) | 6 steps, BF16 block-offloaded; `hires: true` for face close-ups |
| Video, fast | Wan2.2-TI2V-5B-Turbo (`profile: fast`) | ~50 s per 5 s clip at 832x480 |
| Video, quality | Wan2.2-I2V-A14B GGUF Q4 + lightx2v (`profile: quality`) | ~10 min per clip |
| Key frames | Z-Image-Turbo (W8A8 INT8) | best-of-N by face identity |
| Key-frame edits, cast lock | FLUX.2 Klein 4B (W8A8 INT8) | redraws with reference faces; also used for prop edits |
| Talking shots | InfiniteTalk via Wan2GP (INT8) | audio-driven acting, cut points chosen so no mouth moves without voice |
| Lip-sync (older path) | MuseTalk, LatentSync | LatentSync at 8 steps / guidance 1.0 |
| Voices | Chatterbox, Qwen3-TTS, Kokoro; checked by faster-whisper, ECAPA, Step-Audio-2-mini | directed takes, best take kept |
| Music | ACE-Step 1.5 turbo | `tools/make_music.py`, ranked with CLAP (`tools/clap_rank.py`) |
| Checks | Qwen3-VL-4B judge, YuNet + SFace, SyncNet, Depth-Anything-V2-Small | yes/no frame questions, identity, sync, parallax depth |
| Finishing | RIFE (ncnn), Real-ESRGAN anime (ncnn), NVIDIA RTX VSR + frame generation | interpolation and upscaling |
