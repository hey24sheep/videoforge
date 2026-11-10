# videoforge changelog

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
