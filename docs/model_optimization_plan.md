# Model optimisation plan: RTX 4060 Laptop (8 GB, Ada), 31 GB RAM

Date: 2026-11-10. Status: PLAN. Nothing is changed yet; every item is an opt-in flag until an A/B pass and owner OK.

## 1. Measured on this GPU (torch 2.11 cu128, talk venv)

GEMM throughput at 14B-transformer shapes (M=8192 tokens):

| Shape | BF16 | FP8 e4m3 `_scaled_mm` | INT8 `_int_mm` | FP8 storage -> BF16 compute |
|---|---|---|---|---|
| attn proj 5120x5120 | 26.2 TF | **53.3 TF (2.04x)** | 26.3 (1.00x) | 28.3 TF |
| FFN up 5120->13824 | 29.2 TF | **50.9 TF (1.74x)** | 22.9 (0.78x) | 29.7 TF |
| FFN down 13824->5120 | 28.5 TF | **53.4 TF (1.88x)** | 23.9 (0.84x) | 28.3 TF |

What this shows:
- **FP8 compute is the one real speed lever on Ada.** The FP8 tensor cores run about 2× BF16.
- **INT8 is storage only.** INT8 matrix maths is no faster than BF16 here, so int8 weights only save memory.
- **"FP8 storage, BF16 compute" costs almost nothing.** We use it for Z-Image, Klein, the judge and K6. It saves VRAM but gains no speed.
- FP4 / NVFP4 is not available. That needs Blackwell.

## 2. Inventory: what runs today

| Model | Role | Weights / compute | Peak VRAM | Time (unit) | Bottleneck |
|---|---|---|---|---|---|
| InfiniteTalk (Wan2.1-I2V-14B + IT module + lightx2v LoRA r64) | talk shots | quanto int8 / bf16, mmgp profile 5, Sage2 | ~5 GB | ~81 s per step per 81-frame window, 4 steps | **biggest: about 70% of a build** |
| K6 lite distill | motion clips | fp8 stored / bf16, Sage2, TAEHV decode | <7 GB | 11.4 s/step | second |
| Z-Image-Turbo | keyframes | fp8 stored / bf16 (fp8 compute opt-in, softer) | ~6 GB | per still | small |
| FLUX.2 Klein 4B | cast / edit | fp8 stored / bf16; text encoder offloaded; embeddings cached | ~5 GB | per still | small |
| Qwen3-VL-4B judge | checks | fp8 stored / bf16 | 4.9 GB | per frame set | small (cached) |
| Chatterbox / Qwen3-TTS, whisper int8 CPU, ACE-Step bf16 | audio | as is | — | — | negligible |

## 3. Ranked candidates

Gains below are estimates until each one is A/B'd.

| # | Change | Model | Expected gain | Quality risk | Effort |
|---|---|---|---|---|---|
| **1** | **FP8 e4m3 compute for the DiT linears.** Per-output-channel weight scales and per-token activation scales, through `_scaled_mm`. Keep these in BF16: patch/text/time embeddings, adaLN modulation, the audio cross-attention, the first and last 2 blocks, and the head. | InfiniteTalk 14B | linears are about 65% of a step; 1.8× on those gives **about -25 to -30% per talk shot** (81 s → ~58 s per step) | low–med (lip-sync is the sensitive part, so the audio path stays BF16) | med: Wan2GP has an FP8 path; we need to build an fp8 checkpoint from the bf16 source |
| **2** | **Merge the lightx2v LoRA into the base weights**, then quantise. | InfiniteTalk | removes the extra rank-64 matmul on every linear: **about -3 to -6%** | none to slightly better: one rounding step instead of LoRA-on-int8 | low |
| 3 | Same FP8 compute recipe with protected layers. The earlier "all-FP8" try was softer (SSIM 0.825 vs 0.851); keeping sensitive layers BF16 should close that gap. | K6 | **about -20% per clip** | med (motion texture) | med |
| 4 | FP8 compute with protected layers (replaces today's all-or-nothing `VF_FP8_COMPUTE`) | Z-Image, Klein | about -20 to -30% per still | med (fine structure); owner eye test | low (the code path exists) |
| 5 | `torch.compile` (inductor, fixed shapes) on the DiT blocks | K6, Z-Image | 5–15% | none (same maths) | med: compile time on Windows and recompiles on shape change |
| 6 | Talk step count 4 → 3 | InfiniteTalk | -25% | **high** (MagCache showed faces break before metrics notice) | low |
| — | Rejected: INT8 compute (slower here), TensorRT (weeks of work, gain ≈ FP8), FP4 (no hardware), FP8 judge (not a bottleneck), new layers or retraining (no data or compute for it on 8 GB) | | | | |

"Add layers" in practice means items 1–4: give each layer its own precision. Sensitive layers stay BF16, and the bulk GEMMs run in FP8. Training new layers is not feasible locally.

## 3b. Tuning (added 2026-11-10)

Tuning here has three kinds. **Fine-tuning** means training a small LoRA. **Settings tuning** means picking the best inference knobs. **Kernel tuning** means compiling and autotuning the GPU code.

### Fine-tuning (LoRA training on this 8 GB card)

| # | What | Model | Fits 8 GB? | Gain | Risk / cost |
|---|---|---|---|---|---|
| **T1** | **Character LoRA** per cast member, trained from our own approved cast refs (synthetic, no real person) | Z-Image-Turbo (Apache-2.0) | **yes, reported**: 512 px, batch 1, layer + text-encoder offload (ai-toolkit / musubi-tuner) | the face is right on the first still: fewer cast rerolls, and it may skip the Klein face-edit pass (~30–50 s per cast keyframe) | ~1–2 h training per character; needs 15–30 clean refs per character; can over-fit (same pose every time) |
| T2 | **Series style LoRA** (a series' signature look) | Z-Image-Turbo | yes, same setup | steadier look, fewer keyframe rerolls | ~1–2 h; can flatten variety |
| — | Klein 4B LoRA | Klein | **no**: the documented minimum is 12 GB | — | skip |
| — | InfiniteTalk / Wan 14B LoRA | Wan 14B | **no**: needs 16–24 GB even with block swap | — | skip |
| — | K6 LoRA | K6 | no tooling found | — | skip |

### Settings tuning (no training; a sweep on the bench, same seeds, blind critics choose)

| # | Knob | Today | Sweep | Expected |
|---|---|---|---|---|
| S1 | Z-Image steps / shift | 9 steps | 6, 7, 8 × shift 2–4 | -10–30% per still if 7 steps holds |
| S2 | K6 steps | 6 | 4, 5 | -15–30% per clip (risky: it is already distilled) |
| S3 | InfiniteTalk audio CFG / shift | audio CFG 2, default shift | audio CFG 1.5–3, shift ±1 | quality only (lip-sync and face stability); no speed change |
| S4 | FP8 scale calibration | — | per-channel scales fit on ~200 of our own frames (SmoothQuant-style) | the quality half of plan #1/#3/#4: fixes the softness |

### Kernel / system tuning

| # | What | Gain | Note |
|---|---|---|---|
| K1 | `torch.compile` `max-autotune` + CUDA graphs (fixed shapes) | 5–15% | same as #5 above |
| K2 | **Nunchaku SVDQuant INT4 (W4A4)** for Z-Image-Turbo: official INT4 build for RTX 40 (r32 = fastest, r256 = best quality); its own INT4 tensor-core kernels, not torch `_int_mm` | the project reports about 2–3× vs BF16 (UNVERIFIED here); about 3 GB VRAM | replaces #4 for Z-Image if the A/B passes. The Klein INT4 build is a community fork, so it is lower trust |
| K3 | Laptop power mode: plugged in + Windows "Best performance" + vendor GPU mode | throttling showed up earlier at 55 W | **your setting**: I don't change system settings |

## 4. A/B protocol (every item)

1. **Same inputs:** the same seeds, keyframes, audio and frame counts. The bench is a private 2-take talk bench (seeds 701/702).
2. **Reference:** the current pipeline output for those inputs.
3. **Gates, all must pass:**
   - SSIM / LPIPS vs the reference;
   - SyncNet offset and confidence no worse;
   - silent_mouth PASS;
   - Qwen3-VL blind 2-critic score no lower;
   - **my own per-frame review of every shot**. Metrics alone missed the MagCache face breakage.
4. **Owner looks at a side-by-side** before anything becomes the default.
5. Ship as a flag (`gen.talk.precision: fp8`, `VF_FP8_PROTECT=1`, …) with a cache-key change. Old caches stay valid.

## 5. Suggested order and time

1. Items **2 + 1 together** (InfiniteTalk merge, then FP8). Build the checkpoint (~1 h CPU/RAM), run the bench (~16 min), review. **~2–3 h total.** This is the biggest payoff: about -30% on the slowest stage.
2. Item 3 (K6 FP8 protected): ~1.5 h.
3. Item 4 (Z-Image/Klein protected FP8): ~1 h.
4. Item 5 (compile): ~1 h, only if 1–4 land.
5. Tuning:
   - **K2** Nunchaku Z-Image INT4: ~1 h.
   - **S1/S2** step sweeps: ~1 h each.
   - **T1** character LoRAs: ~2 h per character, plus your eye test on faces.
   - T2 style LoRA: only if you want a locked look.

Sources: [Z-Image 8 GB LoRA](https://www.runcomfy.com/trainer/ai-toolkit/z-image-8gb-vram-lora-training), [Klein training min 12 GB](https://docs.bfl.ai/flux_2/flux2_klein_training), [musubi-tuner](https://docsearch.algolia.com/mcp/docs/repo/kohya-ss/musubi-tuner), [Wan LoRA VRAM reports](https://redlib.hbubli.cc/r/StableDiffusion/comments/1jbbn4j/how_much_memory_to_train_wan_lora), [Nunchaku Z-Image-Turbo](https://huggingface.co/nunchaku-ai/nunchaku-z-image-turbo), [Nunchaku in Diffusers](https://huggingface.co/blog/nunchaku-diffusers), [Klein 4B Nunchaku (community)](https://huggingface.co/tonera/FLUX.2-klein-4B-Nunchaku).

Disk: an fp8 14B checkpoint is about 14 GB (same as int8). Old weights go to `_trash`, never deleted.

## 6. RESULTS (2026-11-10, run under the owner's rule: no change that risks quality or creativity)

Reference = each model's own BF16 output, same seed. "Closer" means SSIM/PSNR vs BF16 went up, and the frames were reviewed by eye.

| Model | Before | Now (default) | Speed | Fidelity vs BF16 | Status |
|---|---|---|---|---|---|
| K6 | fp8-stored, BF16 compute | **BF16, block-offloaded** (`VF_K6_PREC`) | **-21%** per step (16.9 → 13.3 s, 12.6 → 10.1 s) | exact (it *is* BF16); fp8store was SSIM 0.959–0.984 | ADOPTED |
| Z-Image | fp8-stored | **W8A8 INT8 + regional torch.compile** (`VF_ZIMAGE_PREC`, `VF_ZIMAGE_COMPILE`) | **-40%** per still (18.1 → 10.9 s, +17 s one-off compile per process) | much closer: SSIM 0.92–0.98 (fp8store 0.82–0.87: the old path visibly drifted, e.g. a different coat) | ADOPTED |
| Klein | fp8-stored | **W8A8 INT8** (`VF_KLEIN_PREC`) | -10% (10.9 → 9.6 s), load 40 → 15 s | closer: SSIM 0.997 (fp8store 0.988) | ADOPTED |
| InfiniteTalk | quanto INT8 W8A8 (WanGP Triton) | unchanged | FP8 would be only ~10% faster on linears | FP8 error 0.038 vs INT8 0.011 (3.5× worse) | REJECTED |
| lightx2v LoRA merge | runtime LoRA | unchanged | ~3% at most | needs a 2nd int8 rounding | REJECTED |
| Tuning T1/T2 (LoRAs), S1/S2 (fewer steps), S3 (audio CFG) | | | | quality/creativity risk | SKIPPED (owner rule) |
| Nunchaku INT4 (K2) | | not tried | | 4-bit = quality risk; INT8 already gave -40% | SKIPPED |

Cache safety: every clip/still/Klein frame already cached under the old fp8 key is still used (lookup tries the old key
first), so approved episodes and the trailer re-cut with zero regeneration (checked: trailer 14/14 K6 clips, ep02 44/44
Klein frames, 599 t2i stills). Only NEW generations use the new paths. Old path: set the env var to `fp8store`.
Code: `vf/gen/int8lin.py` (W8A8 linears on WanGP's fused Triton kernel; norms/embeddings/modulation/final layers stay BF16).

Re-run any of this: `tools/precision/` (gemm_bench, dump_jobs, ab_compare). Frame sheets are kept with the private test series.

## 6b. InfiniteTalk checks + settings tuning (2026-11-10, after owner OK)

Precision: INT8 (current) vs the model's own BF16 weights, two test takes: SyncNet 6.77/7.54 vs 6.84/7.56,
INT8 914 s vs BF16 1176 s. INT8 kept. The detective mouth smear (f8-f16) is in BF16 too: model-inherent.
torch.compile: 1001 s vs 914 s, 6.5 GB VRAM, SSIM 0.89. REJECTED. BF16 weights deleted after the test.

Settings sweep (search set: detective, seed 701, 10 variants). Steps 3/6/8, shift 3/5, audio CFG 1.0/1.5/3.
Only audio CFG 1.0 was faster: one model pass per step instead of two, -40 %. It also looked better on that take.
Steps 3 broke the face (glowing teeth, green chin). Steps 6/8 were slower with no gain.

Hold-out (anti-overfit): base vs acfg 1.0 vs acfg 1.0 + shift 3 on 4 unseen takes (2 new characters, new seeds).

| Take | Base: time / SyncNet conf | acfg 1.0 | acfg 1.0 + shift 3 |
|---|---|---|---|
| mara702 (81 f) | 536 s / 7.54 | 311 s / 6.97 | 312 s / 6.89 |
| det703 (57 f) | 378 s / 7.23 | 216 s / 7.05 | 216 s / 6.93 |
| waltoath (69 f) | 466 s / 7.02 | 264 s / 7.12 | 264 s / 7.50 |
| walthid (93 f) | 694 s / 6.76 | 402 s / 5.80 | 402 s / 6.35 |
| mean | 7.14 | 6.73 | 6.92 |

Identity and face sharpness (tools/precision/talk_score.py) went up with acfg 1.0 on all 4 takes.
Frame review (talk_hold_*.jpg): with acfg 1.0 the mouth opens less on every take. The sharpness gain is a side effect
of less mouth motion. SyncNet agrees: lower on 3/4 takes.
**Verdict: REJECTED as default.** The search-set win did not hold on unseen takes. Lip-sync is the quality bar.
Defaults unchanged: audio CFG 2.0, shift 7, 4 steps, INT8.
Opt-in only: `talk: {audio_cfg: 1.0}` per shot gives ~-42 % for a draft/preview. Not for delivery.
