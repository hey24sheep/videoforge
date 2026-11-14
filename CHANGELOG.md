# videoforge changelog

## Unreleased
- `vf cut` writes the archive automatically after a passed release (`VF_ARCHIVE=0` to skip). `vf config export` /
  `vf config import` save and restore every setting.
- `vf archive <episode.yaml>`: keeps the kept voice take of every line, voice refs, every shot's keyframe, cast refs and
  lock.json (voice settings, seeds, prompts, kept takes, VF_* settings, videoforge commit) in `<episode>/archive/`, so a
  lost render can be remade with the same people and voices. Every build now writes `out/shots.json` for it.
- `tools/junkcheck.py`: blocks a push that adds junk (logs, review sheets, critic files, episode build output, build
  snapshots, test files, local settings, media over 25 MB outside allowed folders). `--install` makes it the pre-push hook.
- Thermal guard (`vf/thermal.py`), safe for a 45 °C room: jobs run at below-normal priority on half the cores, start
  only when cool, are throttled from 72 °C, paused at 80 °C or earlier when the temperature rises fast (2 s reads once warm; resume at 70 °C), killed at 88 °C or after 20 min over
  80 °C, and paused when no sensor reads (fail closed). The CPU reading adds an 8 °C margin to the ACPI zone.
  `vf thermal` shows the temperatures; `vf status` too; settings in `vf config` (group thermal).
- `gen.ink` styles: `style: engrave` (cut into metal or pressed into paper: groove shading from one light) and `style: emboss` (raised), with `depth` and `light`.
- `gen.ink` `bake: true`: the text goes into the keyframe the video model animates, so it moves with the prop. The tracked stamp it replaces can drift off a prop that moves (text floats).

## 1.3.0 (2026-11-13): terminal CLI
- `vf` with no command shows the commands in groups (Make, Check, Look after); `vf --version`; colour help on every
  command (`rich`, `rich-argparse`). Output to a file or pipe stays plain text, so job logs read the same.
- `vf status [--watch] [--all] [--json]`: running and recent jobs, `vf run` stage bar, last log line or known failure
  cause, GPU load, VRAM, temperature.
- `vf config [get|set|unset] [--json]`: every `VF_*` setting with value, source (env / file / default) and default.
  `vf config set` saves to `vf.config.json` (not committed); an environment variable still wins.
- Progress bar over the render loop of `vf build`; coloured stage rules in `vf run`; coloured `vf doctor`.
- `vf`, `vf status`, `vf config` and `vf --version` start without the render stack (no cupy import).
- Fix: tools/jobwatch.py could report a finished job as running or STALLED when Windows reused its pid.

## 1.2.1 (2026-11-10): cleanup
- `vf.gen.k6_batch` run on its own now defaults to `K6_PREC=bf16`, the same as `vf` itself.
- Unused precision bakes removed from models/hf/vf_baked: K6 fp8 and int8, Z-Image fp8store and fp8compute, Wan 5B fp8compute.
  Any path switched back on re-bakes on first use.
- Added `tools/precision/` (gemm_bench, dump_jobs, ab_compare) and the `docs/ab_2026-11-10/` frame sheets.

## 1.2.0 (2026-11-10): model precision pass
- K6: new clips run the DiT in full BF16, block-offloaded: -21 % per step and exact model precision (`VF_K6_PREC=fp8store` = old).
- Z-Image: W8A8 INT8 linears (`vf/gen/int8lin.py`) + regional torch.compile: -40 % per still, closer to BF16 (`VF_ZIMAGE_PREC`, `VF_ZIMAGE_COMPILE`).
- Klein: W8A8 INT8: -10 %, closer to BF16 (`VF_KLEIN_PREC`).
- Cached clips/stills under the old fp8 keys are still used: approved episodes do not regenerate.
- Rejected after measurement: InfiniteTalk FP8 (3.5x error for ~10 %), LoRA merge (~3 %). Skipped: LoRA training, step cuts, INT4.

## 1.1.0 (2026-11-10)
Run tool and package
- `vf run <gen.py|episode.yaml> [--until STAGE] [--preview] [--detach]`: gen -> lint -> voices -> keyframes -> release cut.
- `vf doctor` (environment check), `vf new` (episode scaffold), `vf.cmd` launcher, `pyproject.toml` (`vf` command).
- `vf cut`: series intro (`{episode}` label) + episode + outro, checks, join, phone copy < 28 MB; `release: {wrap: false}`.
- `vf/autolayout.py`: episode generators; shot lengths and line times from the real voice takes.
Quality features
- `gen.ink` (vf/inktext.py): text written into props; `audio: {lufs}` per spec; lint rules (unheld props in talk shots,
  in-scene text layers, K6-only steps on talk shots); `vf check` WARN gates (floating props, loudness).
Speed
- `vf check` caches per-shot results (cache/check) and scans playback at 108x192 on NVDEC (ep01 91 -> 18 s).
- Talk stage: one InfiniteTalk model load per build (batched takes), SageAttention 2.2 (talk venv), `--preload 3000`.
- Klein prompt embeddings cached (cache/klein_emb): bit-identical, 53 -> 33 s per run.
- Rejected after A/B: MagCache step skipping on InfiniteTalk (faces break; opt-in `gen.talk.step_cache`, off).
- `gen.motion_style`, pinned seed/seconds/image: reuse approved takes elsewhere (trailers) with zero new clips.

## 1.0.0
K6 + finishing chain, cast lock, directed voices, InfiniteTalk talk shots, checks, jobwatch.
