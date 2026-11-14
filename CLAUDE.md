# videoforge - notes for Claude sessions

Generic pipeline for the user's YouTube / Instagram series. Read README.md first (it is the instruction set).

## User rules
- Every series/episode can differ; never hard-code a story, characters or style into vf/.
- **Before a final render, ASK which formats** (yt = 16:9 1080p, short = 9:16 1080x1920, yt4k). Not always both.
- Speed target: a 1-minute episode renders in 1-2 min after edits (AI footage is generated once and cached).
  Measured: ~32 s/format cold at 1080p60; one-shot edit ~9 s.
- Quality target: no artifacts or defects. Auto-check and regenerate (QC) rather than shipping defects.
- Dialogue must be dramatic, mature and motive-driven; the user rejects flat or childish lines.
- Downloads of open models are fine; never create accounts or enter credentials; no paid APIs.
- Show a watchable video early; keep review rounds short (the user dislikes long slow loops); weight story and
  visible artifacts more than polish when judging.

## Workflow and gates (2026-11-06): read `docs/workflow.md` before any episode work
- Script first: `python -m vf script-qa <drafts> --bible series/<s>/bible.md --ref <approved REF.md>` + blind Opus/Sonnet critics
  (`prompts/script_critic.md`); only the winner goes to production.
- Judging v2 (2026-11-10, replaces the gap-list loop for drama decisions):
  `python -m vf script-judge transcript <episode.yaml>` (viewer transcript of a built episode);
  `python -m vf script-judge prompts --premise P.md --entry NAME=file ... --out DIR` -> one subagent per persona
  (showrunner+exec Opus, exec Sonnet, viewer Haiku; never Fable) writes DIR/<persona>.answer.md;
  `python -m vf script-judge table DIR` -> drama 70 % + logic 30 % (3 worst "lost" points), Borda rank, watch-next.
  Write 2-3 spines, rank them against the owner-approved previous episode; max 2 patch rounds; the owner picks.
- Generic judges (2026-11-11): `vf judge script|dialogue|music|clip|image|video ...` runs the open-model judges;
  `vf judge critics script|dialogue|video --entry NAME=path ... --out DIR` writes blind Opus/Sonnet critic prompts
  (rubrics in prompts/judges/), `vf judge table DIR` derives the scores. Use these for anything outside an episode
  (music picks, reels, single clips) instead of ad-hoc checks.
- After every render: `python -m vf check <episode.yaml>` (brightness dips, frozen frames, SyncNet lip-sync offset). Exit 1 = BLOCK.
- Lip-sync shots with `speed != 1` need the voice time-compressed by the speed (automatic since 2026-11-06); LatentSync runs at
  8 steps / guidance 1.0 (the old 20-step default spills past 8 GB and takes hours).
- A dialogue line spoken in person (not on the phone) gets `fx: null`; `fx: phone` can be a character's series default.
- Never add dips, flicker, crash zooms or transitions the owner did not ask for.

## Layout
- `vf/` pipeline (memo cache, spec, gpu kernels, layers, render, captions, audio, qc, build CLI, styles, gen/)
- `styles/styles.yaml` 10 style profiles; `prompts/director.md` master episode-design prompt (user's brief + rules)
- `series/<name>/series.yaml` + `series/<name>/<ep>/episode.yaml`; outputs in `<ep>/out/`
- `cache/` global cache (AI clips, TTS, key art); `<ep>/.vfcache` project cache
- Python: `%LOCALAPPDATA%\Programs\Python\Python312\python.exe` (CuPy, OpenCV, PyYAML) for render;
  `D:\claude_work\hero_film\tools\envs\gen\Scripts\python.exe` (torch, diffusers) for AI via `vf.gen.bridge`.

## AI stages (vf/gen, run in the gen venv through vf.gen.bridge.run_gen)
- video profiles (vf/gen/video.py; numbers in docs/video_model.md):
  - `fast` (default): Wan2.2-TI2V-5B-Turbo, fp8 weights resident, 4 steps, tiny TAEHV decoder, 832x480, 121 frames
    @ 24 fps: ~50 s per 5 s clip warm (~62 s first). Faces soften/drift after ~1.5 s; fine for most shots.
  - `quality`: Wan2.2-I2V-A14B GGUF Q4_K_M + lightx2v 4-step, 81 frames @ 16 fps, ~10 min; hero close-ups and
    last_image. Compute-bound (>=100 s/step on this 55 W GPU): it cannot reach 2 min.
  - a persistent worker (vf/gen/server.py) keeps models loaded; prompt embeddings are cached on disk and encoded
    in batches (the text encoder is freed right after). Builds `park_gen()` instead of killing it: weights move to
    RAM (~2 s) and come back in ~1 s, so a one-shot re-roll is ~18 s + the clip. Models load from fp8 "baked"
    weights (hero_film/models/hf/vf_baked, 5-7 s). The worker parks itself after 120 s idle, exits after 1 h.
    `gen_status()` shows owners/RAM/VRAM. cuDNN attention on (-7 %/step).
  - FP8 compute (`VF_FP8_COMPUTE=1`): -17 % per clip step, -32 % per keyframe, but measurably softer on
    structures (SSIM 0.825 vs 0.851): OFF by default (quality first). Own cache keys when on.
  - Over 8 GB VRAM Windows silently pages to system RAM and steps get 4-15x slower: keep the resident set < 8 GB.
- Pass keyword args only, omit unset optionals (the cache keys on the arguments actually passed). Editing
  vf/gen/video.py, taehv.py or fetch_models.py invalidates every cached clip.
- Shots use `gen:` blocks (vf/stages.py): keyframe/image -> i2v -> Real-ESRGAN upscale -> video layer.
  Episode/series `gen:` = defaults for every shot (e.g. `size: [1024, 576]`, `upscale: 0`). The only upscaler is
  anime-tuned: use `upscale: 0` + a bigger `size` for photographic styles.
  Clip size cost (fast profile, 121 frames, measured 2026-11-01, GPU at full clocks, no throttling):
  832x480 = 11 s/step, peak 7.1 GB, ~47 s/clip warm. 576x1024 = 22-26 s/step (1.5x tokens -> ~2x attention),
  peak 8.2 GB = over the card, Windows spills 1-2.8 GB to RAM -> stalls (53 s steps, a 233 s encode); 75-130 s/clip.
  Chunking the DiT feed-forward did not lower the peak. Add ~90 s per batch for prompt encoding + model loads.
  **Default = 832x480 / 480x832 (the user's choice for speed)**, set in styles.yaml `_defaults.gen`; only use
  576x1024 if the user asks for sharper clips.
- key frames (vf/gen/image.py): Z-Image-Turbo (Apache-2.0), fp8-stored DiT resident, 9 steps CFG 0; Qwen3 prompt
  embeddings cached. `python -m vf keyframes <ep> --formats short` -> out/keyframes_sheet.jpg (review before clips;
  re-roll one with `keyframe_seed:`). Image and video models swap through `vf.gen.claim` (one resident at a time).
- voices (vf/gen/tts.py, runs in the `tts` venv as a child process): Chatterbox + faster-whisper check. Each take is
  transcribed; a garbled take is redone (4 takes max), a repeated tail ("Come alone. Come alone.") is cut at the
  last scripted word; word timings go to `<wav>.words.json` (Shorts captions pop on the spoken word).
  `python -m vf voices <ep>` prints line timings: set `at:` from them. `say:` = spoken wording ("two forty")
  when the caption differs ("2:40"). Voices: `{ref, exaggeration, cfg_weight, temperature, fx: phone|radio|room,
  gain_db}`. Chatterbox's built-in voice is MALE (~86 Hz); for women make a synthetic ref with Kokoro (installed
  in the tts venv, e.g. af_heart). Never clone a real person.
- sound (vf/synth.py): any audio src may be `synth:<kind>?k=v` (rain, drone, piano, ring, heartbeat, thunder, hit,
  riser, tick), cached. `audio.beds:` = undocked ambience layers. Still to build: a music model (ACE-Step) and
  AI-clip QC with seed retries.

## Kandinsky 6 profile (`gen: {profile: k6}`, vf/k6.py, the default profile since 2026-11-02)
- **Generation:** `vf/gen/k6_batch.py`, Kandinsky 6.0 Lite distill (MIT), run as its own process.
  - fp8 DiT resident, SageAttention 1.0.6 patched into the attention dispatch, prompts cached, TAEHV decode.
  - Steps and sizes: 6 steps, 384x672 at ~1.3 min/clip; `hires: true` gives 480x864 at ~4.3 min/clip for face close-ups.
- **Finishing:** `k6.finish`, one cached step after another, each in its own process:
  1. Lip-sync (optional): `lipsync: {line, engine: musetalk|latentsync, mask}`. MuseTalk works for frontal clean faces; bearded 3/4 faces fail, so use VO.
  2. RTX VSR 1x `DEBLUR_HIGH`.
  3. Exposure match (`expo:`).
  4. NVIDIA frame generation (`vf/gen/framegen.py`) to fps/speed.
  5. RTX VSR 4x `ULTRA` (`vf/gen/rtx_vsr.py`).
- **RAM rule:** run one heavy model process at a time. Concurrent K6 + pip + TTS hung the GPU driver.
- **Licences:**
  - nvidia-vfx: a free NVIDIA SDK, not open source.
  - MuseTalk: MIT.
  - LatentSync: Apache code, OpenRAIL++ weights. Its InsightFace dependency is non-commercial, so it is replaced by YuNet.

## Resource limits (done 2026-11-02; see memory film-pipeline-performance for numbers)
k6_batch: lazy text encoder (only for uncached prompts, then freed), no audio models, 4 CPU threads, thermal guard
(K6_TEMP_HI 82 / K6_TEMP_LO 72). Wrap heavy runs in `python -m vf.gen.resmon -- <cmd>` to log RAM/CPU/GPU temp.
## (old) TODO after restart (user, 2026-11-02: "CPU, RAM and GPU temps go high, fix the pipeline")
1. `k6_batch.load`: when every job's prompt embedding is cached (`cache/k6_emb`), don't load the 16 GB Qwen2.5-VL text
   encoder at all (load transformer + VAE only); load it only for uncached prompts, then free it.
2. Thermal guard in k6_batch / long loops: poll `nvidia-smi --query-gpu=temperature.gpu`; above ~80 C pause until < 70 C.
3. Cap CPU threads: `torch.set_num_threads(4)`, `OMP_NUM_THREADS=4`, ffmpeg `-threads 4`.
4. One heavy process at a time (gpu_lock around every finish step and the judge), never pip/TTS during generation.
State when paused: new pipeline is in the code (k6 profile, grade, prop + silent-mouth judge checks, nose-exhale smoke
prompt, SageAttention 2.2 auto kernel, lossless NVENC intermediates) but untested end to end; 5/15 clips cached in
cache/k6. Next: implement 1-4, run the finishing test on one clip, then the full build.

## LTX-2.5 engine (2026-11-11)
`gen: {engine: ltx}` (shot, episode or series `gen:`) makes the raw clip with LTX-2.5 distilled 22B INT8 through Wan2GP
(vf/gen/ltx_batch.py, talk venv); QC, trim, lip-sync and finishing are the K6 path's. Blind A/B on 5 hard shots (same
keyframes): LTX 5.4 vs K6 3.0, ~47 s vs ~78 s per clip, ~2 GB VRAM peak. Sizes are multiples of 64 (`VF_LTX_SIZE`,
default 384x704; `k6_size` per shot). K6 stays the default so approved episodes keep their cached clips.
576x1024 tested blind on the same 5 shots: 5.1 vs 5.3 for the small size, ~80 s vs ~47 s per clip: keep 384x704.

## Small checker models (gen venv)
- judge (vf/gen/judge.py): Qwen3-VL-4B (Apache-2.0, fp8-stored, 5.5 GB peak) answers yes/no questions about 8
  sampled frames: run_gen('judge', [dict(clip=, questions=[dict(q=, expect='yes'|'no')], checklist=['appear','text'],
  describe=False)]). ~0.6 s/question/clip + 10 s load. On 16 hand-labelled test clips 'appear' caught 9/11 defective clips with
  0/5 false alarms; blind to warping/morphing (clipqc's job). Shots carry `gen: {judge: [{q, expect}]}` written from
  their intent; stages.generate re-rolls on a fail. Generic "glitch?" questions never fire: be specific.
- depth (vf/gen/depth.py): Depth-Anything-V2-Small (Apache-2.0; Base/Large are NC - never use them): 16-bit PNG,
  near = bright, 25 ms/frame, 0.7 GB; feeds the camera's 2.5D parallax.

## Artifact prevention (built 2026-11-02 after the first episode review: user scored it 5/10 for artifacts)
- `vf lint` (vf/lint.py) runs before every build; prompts/director.md lists the rules. Never write camera moves,
  walking, hand-prop actions or mouth movement into motion prompts; props/wet/smoke states go in the keyframe.
- The AI camera is always steady (styles motion_suffix); every shot gets a compositor camera move (vf/camera.py).
  User rule: no held stills, film-like moves with motion blur.
- Clip QC + repair + re-roll (vf/clipqc.py via stages.generate; CPU, ~1-3 s/clip). It cannot see semantic errors
  (a prop teleporting): that's the VLM judge (vf/gen/judge.py).
- 24->60 fps: RIFE retime (vf/interp.py, ~12 s/clip); nearest-frame holds were the "shaky dock".
- Identity: cast anchors per series + best-of-3 keyframes by SFace similarity (vf/identity.py).
- vf/gpulock.py: hold `gpu_lock()` around ad-hoc GPU work when several agents share the 8 GB card.

## Cast lock (vf/cast.py, built 2026-11-03): one face per character, every shot, every episode
- series.yaml `cast:` sheet: per character `name`, optional `description` (default = `characters.<key>`), 1-3 `refs`,
  optional `aliases`/`threshold`/`anchor`. `cast_lock: true` turns it on for the series; `cast_lock_except: [ep01]`
  keeps listed episodes off (e.g. a pilot made before the lock; opt in = remove it from the list, or
  `cast_lock: true` in its episode.yaml). Episode `cast:` overrides/extends the sheet (new wardrobe, guest character).
- Shot tagging: `cast: [ada]` on the shot (or `gen:`), `cast: []` = none; else inferred from the keyframe (or
  motion) prompt: `{key}`, the key word, name words (e.g. Ada, Quinn), aliases, the description text.
- Keyframes for cast shots: `cast_candidates` (2) Z-Image stills -> FLUX.2 Klein edit with the refs' face crops
  (cache/cast_refs) -> SFace score (2x upscale + rotated passes for small/tilted faces; score = mean over refs of
  the best face, min over the shot's characters) -> keep the best >= threshold (0.40), else re-roll new seeds
  (`cast_rerolls`, 1) and keep the best, flagged. Klein outputs: cache/cast_kf/<hash>.png (never GC'd).
  `image:` shots are only measured unless `gen: {cast_refine: true}`. `cast_klein: false` = gate Z-Image only.
  Klein encodes every prompt: new cast shots need `--allow-encode` (guard: 20 GB free RAM, clear GPU).
- QC: out/cast_qc.json + printed table (keyframe score, clip median/min over 4 sampled frames, flags), written by
  `vf keyframes` / the build for cast-locked episodes; any episode on demand: `python -m vf.cast qc <episode.yaml>`;
  `python -m vf.cast plan <episode.yaml>` shows which shots get which character.
- Shots/episodes that don't opt in run the old code path with the same arguments (cache keys unchanged).
- SFace numbers: same person 0.55-0.89 (robust pass up to ~0.97 on rotated copies), different people 0.02-0.33.

## Performance facts
- All compositing on the GPU via two CUDA RawKernels (bilinear `sample`, fused `finish` grade). Never add
  per-pixel NumPy work on full frames to the hot path; write a kernel or work at 1/4 res.
- CuPy's gaussian_filter is slow at large sigma: use the bloom pyramid / `_blur_sep` at low res.
- First run of a kernel compiles (seconds); profile after warm-up.
- Frames are piped raw to h264_nvenc; no PNGs between stages. YouTube/Instagram max 60 fps: don't render 120.
- Vulkan device 0 is the AMD iGPU: ncnn tools (RIFE, Real-ESRGAN) need `-g 1`.

## Memory bank (in the repo)
Read `.claude/memory/MEMORY.md` at the start of every session and open the files that apply: they are the owner's
standing rules and lessons (same content as the local auto-memory). Add new lessons there as one file + one index line.
Default mark (owner 2026-11-11): every `vf build` / `vf.mograph` render carries "hey24sheep.com/videoforge" bottom right
(vf/render.py `watermark`). Users may turn it off (`--no-watermark`, VF_WATERMARK=0, `watermark: false`, MCP
`watermark=False`); never turn it off yourself unless the owner asks. Repo is Apache-2.0 with a NOTICE file.
AI root: env `VF_AI_ROOT` (default `D:\claude_work\hero_film`) holds the AI venvs, tools and models.
In a cloud session (no GPU, no D:/ models): scripts, specs, docs and `python -m vf.mograph` only; see README "Running in the cloud".

## Context hygiene (user rule, 2026-11-02)
- `STATE.md` (this folder) is the working-state file: decisions, progress, constraints, files, blockers, next steps.
  Update it after every meaningful change and before any compaction. Facts only; mark guesses UNVERIFIED.
- Auto-compaction is set to 60 % (`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` in `~/.claude/settings.json`); a SessionStart
  `compact` hook prints STATE.md back after each compaction. After a compaction: compare it with the summary,
  tell the user what is missing, re-read the files before acting.
- Claude cannot read its own context %; never estimate it. Ask the user to run `/context`.

## Thermal limit (owner, 2026-11-14): CPU, GPU and RAM must never pass 80 C
`vf/thermal.py`: jobwatch suspends a job's process tree at VF_TEMP_MAX (80) and resumes at VF_TEMP_RESUME (70); gpu_lock
waits for a cool machine. `vf thermal` reads the sensors. CPU = ACPI zone (reads low under load); no RAM sensor without
LibreHardwareMonitor. Always start long jobs through jobwatch so the guard covers them.

## Long-process watchdog (user rule, 2026-11-02)
- Any process running 30+ min gets rechecked (newest output/cache file times, GPU/VRAM, competing processes, log);
  if stuck or slow, diagnose, fix, restart. Start a 30-min background check with every long job; run builds with
  PYTHONUNBUFFERED=1 (buffered stdout left build logs empty and hid an hour-long stall).

## Resource hygiene (audit + fixes 2026-11-03; see vf/proc.py, vf/gc.py)
- VRAM: every GPU process runs under an allocator cap, so overflow raises OOM instead of silently paging into shared
  RAM. k6_batch: K6_SAMPLE_CAP 6.9 GB for sampling, K6_VRAM_CAP 6.5 for decode. Gen worker and Klein: process_cap
  7.2 GB (vf/gen/guard.py); t2i/encoders restore that cap, never 1.0. PYTORCH_CUDA_ALLOC_CONF limits fragmentation.
  Measured: 480x864x33 sampling reserved 5.25 GB, no spill. 576x1024 spilled 1 GB before the cap; untested after.
  Log `reserved` + `mem_get_info`, not max_memory_allocated (it hid ~0.5 GB spills on earlier builds).
- One GPU user at a time: gpu_lock serialises threads AND processes; every finishing step (_gen) holds it. The
  speculative early finishing was removed (it finished every rejected take and overlapped the judge).
- Processes: builds run in a Windows Job Object (children die with the build); every subprocess/encoder has a
  timeout and is killed on exceptions (vf/proc.py writer/run); a build stops the gen worker it started.
- RAM: platelock, k6 expo/grade and RTX VSR stream frames (no whole-clip lists).
- Disk: final VSR output is near-lossless cq13 (was lossless 135 MB/clip). Every build writes a v2 manifest of all
  cache paths it touched; `vf gc` (auto after each successful build, VF_GC=0 to skip) moves cache files no latest
  build uses to the Recycle Bin. It refuses on old manifests, other live builds or a held GPU lock. GC_DIRS lists
  what it may touch (never t2i/keyframes, models, embeddings, i2v, say, depth).
- Don't keep manual per-build copies of outputs; keep the current build plus the last approved snapshot only.

## Reply style (owner, 2026-11-06): caveman, always
Use the `caveman` skill (`.claude/skills/caveman/SKILL.md`) on every reply: short words, answer first, no fluff.
Exact numbers/paths/commands and every "unverified / still flawed" disclosure stay. Scripts, captions and prompts for the
video are NOT caveman.

## Start long jobs with jobwatch (2026-11-06, after a build died with the session)
- A build started from the Claude session's shell dies when that session's process exits (whole process tree): ffmpeg
  went first and the K6 writer failed with `OSError: [Errno 22]`. Start every build/keyframes/voices run detached:
  `python tools/jobwatch.py run --name <x> --resumable --env VF_GC=0 -- python -m vf build <episode.yaml> --formats short`
- Restart checker: `python tools/jobwatch.py check` (also runs from the SessionStart hook on startup/resume/compact):
  running / STALLED (no log for 12 min) / ok / FAILED with known cause + fix / KILLED (no exit code). Resume with
  `python tools/jobwatch.py restart [--id ID]` (only when the GPU is clear and nothing else runs; builds are cache-backed).
- k6_batch.write keeps ffmpeg's stderr (`<clip>.ffmpeg.log`), retries a failed encode once from RAM, and a failed clip
  no longer kills the rest of the batch.

## Directed voices + talking shots (2026-11-07, owner: "real acting, dialogue delivery with emotions", "mouth moves, no voice")
- Voices: `voices.<who>.direct:` (vf/voicedir.py). Each line is acted N times (Chatterbox clone `cbox`, Qwen3-TTS
  VoiceDesign + speaker embedding `xvec`, Qwen Base clone `qclone`), measured by open models (faster-whisper words,
  ECAPA voice match to `anchor`, Step-Audio-2-mini blind judge: emotion word, intensity, naturalness, match to the
  line's `direction:`), and only a take passing every gate is used (words exact, naturalness >= 9.3, emotion not
  `banned` / in the line's `emotion:`, voice match >= 0.6). Report: out/voices.json (FLAG = no take passed).
  Lessons: cloning a calm reference gives calm takes (the clone copies the ref's acting); a 2-word line cloned from an
  expressive ref loses the emotion, so an emotional line may need `xvec`. One voice per process (Qwen modes OOM together).
- Talking shots: `gen: {talk: {line: <prefix>}}` (vf/talk.py) replaces K6 + LatentSync. InfiniteTalk (talk venv, Wan2GP)
  acts the keyframe over the line +- ~1 s, audio fed 80 ms late (measured mouth lead), pauses removed from the line
  (the model never closes the mouth in a short pause; the mix uses the same pause-free take), then cut points move so
  no open mouth without voice shows (neighbour shots absorb the change; dialogue `at:` never moves). out/talk.json.
  ~7-8 min per 81-frame window on this card (peak ~3.3 GB VRAM, 9 GB working-set cap).
- `vf check` adds `silent_mouth` (tools/silent_mouth.py on each talking shot of the final video, fail > 100 ms) and reads
  shot times from out/timeline.json. ALWAYS also look at a per-frame mouth strip yourself before sending.
- The owner's app viewer plays audio ~1 s ahead of the picture: sync is judged in the phone's own player (proof/sync_test.mp4).
- AFTER EVERY LAUNCH run `python tools/jobwatch.py watch --id <job id>` in the background (owner 2026-11-07: "why do I
  have to check and tell you"). It exits with one EVENT line on: error, abort, FLAG, QC warning, stage milestone,
  deadlock (child waiting for a GPU lock its own parent holds), stall (log quiet + GPU idle 5 min), job end, or the
  30-min heartbeat. Handle it, PushNotification the owner on problems and milestones, start the next watch.

## Integrated 2026-11-09 (final-pass fixes, now pipeline features; read docs/workflow.md)
- `vf/autolayout.py`: episode generators (beats + lines -> shot lengths and `at:` from real voice takes).
- `gen.ink` (vf/inktext.py): text written into a prop on the keyframe, carried onto the clip as ink. Replaces in-scene text layers.
- `vf cut <episode.yaml>` (vf/cut.py): intro (episode label filled in) + episode + outro, check, join, phone copy < 28 MB.
  series.yaml: `intro:`, `outro:`, `release: {name}`; episode.yaml: `episode: {no, title}`.
- `audio: {lufs: X}` per spec (vf/build.py): own loudness target (music-only intro at -18).
- `vf check` WARN gates: floating props on talk shots (Qwen3-VL, out/prop_review.jpg, series `check_props`), loudness.
- `vf lint`: talk shot naming an unheld prop (it floats); in-scene text layers.
- Package + run tool (2026-11-09): `vf.cmd` (repo root) / `pip install -e .` -> `vf`; `vf doctor`; `vf new <series> <ep> --no N
  --title T`; `vf run <gen.py|episode.yaml> [--until keyframes] [--preview] [--detach]` (gen -> lint -> voices -> gen -> keyframes
  -> cut). `release: {wrap: false}` = no intro/outro (trailers). Speed: `vf check` caches per-shot SyncNet / silent-mouth /
  prop results (cache/check, keyed on the shot's frames) and decodes the playback scan at 108x192 on NVDEC: a check
  went 91 s -> 18 s and 432 s -> 131 s on unchanged re-releases.
- Known: the mixer ignores `audio.music.at` (only `start` = offset into the file). A bed then plays from 0.
- Reusing approved shots elsewhere (trailers): pin `gen.seed` (the kept take, out/clip_qc.json), `gen.seconds` (raw clip
  frames / 24 - 0.5), `gen.image` (the exact keyframe: cast_qc.json or `vf keyframes` of the source), `gen.motion_style`
  (the source episode's style), `retries: 0`, `cast_gate: final` -> the K6 cache returns the same clips (trailer_gen.py).
- Talk-stage speed (2026-11-10, private 2-take talk bench, 2 new takes each run):
  batch (vf/talk.py prefetch -> tools/talk_video.py --jobs): the 14B model loads ONCE per build, not per shot (~60-90 s saved
  per extra talk shot; a 2.5-min episode builds in ~20 min). SageAttention 2.2 in the talk venv (`--attention auto` = sage2): 1044 -> 992 s
  (-5 %), lip-sync unchanged (80/40 ms, conf 7.6/6.8). `--preload 3000` (VF_TALK_PRELOAD): 992 -> 949 s, VRAM peak 5 GB, no spill.
  InfiniteTalk is compute-bound (~81 s/step per 81-frame window): the next lever is fewer frames (a line + 1.6 s margin
  > 3.24 s needs a 2nd window, +~180 s). Not changed: a tighter margin leaves less room for clean cut points.
- Computation caches (2026-11-10). Cached: K6 + Z-Image prompt embeddings (cache/k6_emb, prompt_embeds), Klein prompt
  embeddings (NEW cache/klein_emb: bit-identical images, 53 -> 33 s per run, no text-encoder load / RAM guard on a hit),
  judge answers, voice takes, clips, finishing steps, render segments, mixes, vf check results (cache/check).
  Tested and REJECTED: WanGP MagCache step skipping on InfiniteTalk (`gen.talk.step_cache: mag`, opt-in, off): -45 % time
  (949 -> 518 s) and SyncNet passed, but frames showed crushed contrast, green beard cast and blotchy faces (4 distilled
  steps leave nothing to skip). Never enable it without a frame A/B. Not cached: InfiniteTalk's own per-take prep inside
  WanGP (~27 s/take: T5 + CLIP + wav2vec + VAE encode), small next to ~380-560 s of diffusion.

## Model precision (2026-11-10, docs/model_optimization_plan.md section 6)
- Defaults now: K6 DiT BF16 block-offloaded (`VF_K6_PREC`), Z-Image W8A8 INT8 + regional compile (`VF_ZIMAGE_PREC`,
  `VF_ZIMAGE_COMPILE`), Klein W8A8 INT8 (`VF_KLEIN_PREC`); `fp8store` = old path. Each was faster AND closer to the model's
  BF16 output on the same seed. On this card "fp8 storage + BF16 compute" (diffusers layerwise casting) is SLOWER than plain
  BF16 offload and drifts more than INT8. Old fp8-keyed cache entries are looked up first (no regeneration of approved work).
- InfiniteTalk settings tuned 2026-11-10 (docs/model_optimization_plan.md 6b): keep audio_cfg 2.0 / shift 7 / 4 steps. audio_cfg 1.0 is
  -42 % but weaker mouths + lower SyncNet on unseen takes: drafts only. compile and BF16 bring nothing. Validate any tuning on unseen takes.
- InfiniteTalk stays WanGP quanto INT8 W8A8: FP8 measured 3.5x the error for ~10 %. GEMM facts on the 4060: FP8 _scaled_mm
  1.7-2.0x BF16; torch _int_mm 0.8-1.0x; WanGP fused INT8 Triton 1.3x (relerr 0.011).
- Precision A/B toolkit: `tools/precision/` (gemm_bench.py = matmul speed/error per precision; dump_jobs.py = an episode's
  K6/Klein jobs without generating; ab_compare.py = SSIM/PSNR vs the bf16 reference + sheet). Evidence is kept with the private test series.
  Rule: A/B vs the model's own BF16 output, same seed, and look.

## Private series
Private series live outside this repo (gitignored `series/*` except demo and showreel). If a `CLAUDE.local.md` exists here, read it: it holds private series rules and imports their CLAUDE.md.
