# AI video stage (vf/gen/video.py)

> **Updated 2026-11-02:** builds now call `park_gen()` (weights to RAM, ~1 s back), not `stop_gen()`. The
> worker idles out after 3600 s (parks after 120 s). Models load from fp8 baked weights in 5-7 s. Cache keys
> use `GEN_VERSION` + PROFILES (not code), so editing this file no longer invalidates clips. Clip length follows
> the shot (`dur x speed` + 0.5 s QC tail, max 121 frames); step time grows superlinearly with frames
> (117f 11 s, 97f ~8 s, 57f ~4 s per step). Where this page disagrees, CLAUDE.md is current.

## Calling it
- From the main interpreter: `from vf.gen.bridge import run_gen, stop_gen`, then
  `run_gen('video', [dict(image=r'C:\abs\still.png', prompt='...', seed=1)])`.
- Jobs go to a **persistent worker** (`vf.gen.server`, localhost:47219). It starts on first use and keeps its
  models loaded. It exits after `VF_GEN_IDLE` seconds idle (default 600). `stop_gen()` frees its RAM and VRAM at
  once.
  - While loaded, the fast engine holds ~5–7 GB of VRAM. **Call `stop_gen()` before GPU rendering.**
  - `persistent=False` runs a one-shot child process instead.
  - The log is `cache/gen_server.log`.
- Results include `out`, `fps`, `frames`, `cached`, `seconds`, `steps_s`, `times` (load / te_load / encode /
  vae_encode / vae_decode), `peak_vram_gb` and `peak_ram_gb`.
- **The fps depends on the profile** (fast = 24, quality = 16). Choose `frames` as 4k+1 for that fps
  (5 s = 121 frames @24, 81 @16), or omit it to get the profile's 5 s default. Place the clip using the returned
  `fps`.
- `i2v(image, prompt, seed=0, frames=None, width=None, height=None, last_image=None, steps=None, shift=None,
  profile=None)`. Unset values come from the profile. Pass keyword args only. The cache keys on the arguments
  passed and on the code, so **editing video.py, taehv.py or fetch_models.py regenerates every clip**.
- A job with `last_image` defaults to `quality`, the only engine that supports it.

## Profiles
| profile | model | 5 s clip | per clip (warm) | VRAM / RAM | quality |
|---|---|---|---|---|---|
| **fast** (default) | Wan2.2-TI2V-5B-Turbo (yetter-ai diffusers port of quanhaol's 4-step I2V distill); fp8-stored DiT resident in VRAM; UniPC, shift 5; taew2_2 tiny decoder | 121 f @24 fps, 832x480 | **~50 s**; ~62 s for the first clip of a session; +~16 s for a new prompt | 7.1 GB peak / 14.7 GB | holds the palette, costume and glowing eyes; identity drifts after ~1.5 s; gold detail gets mushy; seed-dependent smear in fast moves |
| fast_vae | same with the full Wan2.2 VAE decode | same | ~140 s | same | looks identical to fast at a side-by-side crop (TAE PSNR 25 dB vs full) |
| fastwan | FastWan2.2-TI2V-5B-FullAttn, DMD 3 steps (1000/757/522) | same | ~40 s | 7.1 GB | **re-styles the image after frame 1** (a T2V-only distill); not usable for I2V |
| quality | Wan2.2-I2V-A14B GGUF Q4_K_M + lightx2v Seko-V1 4-step, group offload | 81 f @16 fps, 832x480 | ~590 s gen, ~650 s wall | 5.1 GB / 23.6 GB | best: identity, eye glow and costume stable over all frames |

## Lever-by-lever results
Test image: kael_cu_chosen.png. GPU: RTX 4060 Laptop, **power-capped at 55 W** (nvidia-smi), measuring
~11 TFLOPS bf16.

1. **Persistent worker:** saves process start plus model load per call: 5B ~12 s, 14B ~30–60 s. A cache hit
   now takes 0.0 s (it was ~7 s through a fresh child process).
2. **Prompt-embedding cache** (global memo): a repeated prompt skips UMT5 (11 GB) entirely. A miss costs
   ~12 s load plus ~4 s encode on the GPU via cpu_offload.
3. **Decode:**
   - Full Wan2.2 VAE, bf16 tiled: 87–91 s. Untiled: 176 s, and it spills.
   - **taew2_2 tiny decoder: 0.5–0.8 s, 1.3 GB peak, PSNR 25.3 dB vs full, visually equivalent.** This is our own
     re-implementation (`vf/gen/taehv.py`, strict weight load, lightx2v/Autoencoders weights).
   - lighttaew2_2 gave a PSNR of 15.9: wrong normalisation/architecture, so it is not used.
   - The 14B path's ~80 s VAE time is mostly the encode of the 81-frame conditioning video. Not changed.
4. **Per step:**
   - 14B is compute-bound. One step at 832x480x81 (32.8k tokens) is ~1.7–2.3 s of attention per block × 40
     blocks plus ~1.5 s of linears per block, so ≥100 s/step on this GPU. GGUF dequant and offload are not the
     bottleneck, and no kernel trick brings 4 steps under 120 s.
   - The 5B model at 12k tokens takes 11.5 s/step with the fp8 weights fully resident.
   - **Critical fix:** DiT 4.8 GB + VAE 1.4 GB + activations > 8 GB made WDDM page into shared memory, giving
     42–190 s/step. Now the VAE is swapped onto the GPU only for its call, and `empty_cache` runs per step.
   - Not tried:
     - SageAttention/triton and torch.compile: attention is only ~2 s of each 5B step, and the Windows
       triton/sage wheels aren't on PyPI.
     - GGUF CUDA kernels: remote code.
     - Native fp8 matmul: next lever if needed, maybe 1.5–2× on the linears.
5. **Fewer frames or lower resolution:**
   - Not needed for the 5B model.
   - For 14B, 640x368 still costs ~0.6 s of attention per block, so ≥60 s/step. No.
6. **Step caching:** skipped. With 4 distinct distilled steps there is nothing to reuse; skipping 1 of 4 saves
   ≤11 s at a high quality risk.
7. **Alternative model:** the TI2V-5B Turbo became `fast`. FastWan was rejected (see above).

## Recommendation
- Use `fast` for drafts and most shots: ~50 s per 5 s clip.
- Hero close-ups and identity-critical shots should use `profile='quality'` (~10 min, cached once), or
  regenerate `fast` with other seeds.
- Add AI-clip QC: a frame-diff spike above ~30 flags smeared fast moves, e.g. seed 10 at f40.
- If the laptop runs in a higher power mode than 55 W, all numbers drop.

## Failure modes
- VRAM over 8 GB never errors on Windows; it silently pages and becomes 4–15× slower. Watch `steps_s`.
- Switching engines in one worker now moves the old weights off the GPU explicitly. Before that fix the
  closures leaked them and caused paging.
- RAM: 14B experts (19 GB) and UMT5 (11 GB) are never resident together; jobs are encoded before the experts load.
- `expandable_segments` is unsupported on Windows (harmless warning).
