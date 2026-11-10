---
name: film-pipeline-performance
description: "Measured speed facts for local video work on this laptop (RTX 4060 8 GB at 55 W, AMD iGPU, 31 GB RAM) — what's slow and what fixed it"
metadata:
  node_type: memory
  type: project
  originSessionId: bcdc64f6-e189-45d4-91c7-0dd42090e64c
  modified: 2026-11-03T07:09:44.401Z
---

Measured 2026-11-01, used by [[videoforge-pipeline]]. The user wants a 1-min episode to render in 1–2 min after
edits, and a 5 s AI clip in at most 1–2 min.

**Render (videoforge):** 60 s at 1080p60, both 16:9 and 9:16, takes 41 s cold and ~6 s when cached.
- Composite on the GPU with CuPy RawKernels (bilinear sample, fused grade, RGB→NV12).
- Copy frames out through pinned buffers asynchronously; encode with NVENC p4, cq 18, maxrate 20M.
- No PNGs between stages, and max 60 fps (YouTube/IG don't play more).
- NVENC (one engine, ~2.4 ms/frame) is the floor.
- Per-pixel grain balloons the bitrate; use 2×2 clumps.

**AI video:**
- Wan2.2-TI2V-5B-Turbo with fp8 weights resident and the TAEHV decoder: ~50 s per 5 s clip **at 832x480 only**.
  At 576x1024 it is 2x the compute per step and peaks at 8.2 GB, which spills: 75–130 s per clip.
- Z-Image-Turbo key frames: ~21 s at 576x1024; at 864x1536 it spills (97 s).
- Wan2.2-I2V-A14B Q4 runs ≥100 s per step on this power-capped GPU (~10 min per clip); keep it for hero shots.
- The full Wan VAE decode takes ~90 s; TAEHV takes <1 s and looks the same.
- A persistent model worker plus a cached prompt-embedding store remove ~60 s of loading per call.

**Measured 2026-11-02:**
- Models load from fp8 "baked" weights in 5–7 s; before, 14–23 s.
- `park_gen` moves weights to RAM and back in ~1 s, so a one-shot re-roll takes ~18 s plus the clip.
- Step time grows faster than frame count: 117f ≈11 s, 57f ≈4 s per step. Generate only what the shot needs.
- The full 14-shot episode with QC re-rolls took ~18 min. Rebuilds after spec edits take 35–50 s.
- RIFE retime: ~12 s per clip. Clip QC: 1–3 s per clip on the CPU. VLM judge: ~0.6 s per question plus a 10 s load.
- Compositor: ~15 ms per frame warm (limited by NVENC). The virtual camera adds ~1 ms.
- FP8 compute is 17–32 % faster but softer. It stays off by default.

**Kandinsky 6.0 Lite distill (MIT; 3B DiT + Qwen2.5-VL-7B text encoder; Hunyuan VAE), measured 2026-11-02:**
- It gives much better motion, identity and smoke than Wan-5B on the same keyframe.
- At 480x864 (50k tokens): 45 s/step streamed, and the full VAE decode takes 148 s, so ~11 min per clip.
- **At 384x672, 6 steps, it's ~2.3 min per clip.**
  - The DiT is stored fp8 (layerwise casting, bf16 compute) and streamed at block level.
  - Prompt embeddings are cached.
  - Audio is off.
  - Decoding uses TAEHV (madebyollin taehv.pth, from GitHub): 1.5 s.
- Each step is 23 s and limited by compute. Peak VRAM is 2.7 GB.
- 4 steps loses the smoke. Keeping the DiT resident saves only ~10% and peaks at 6.2 GB.
- Resident at 480x864 spills over 8 GB: 164 s per step.
- Group offload with pinned memory fails (CUDA OOM while pinning 27 GB). Use `low_cpu_mem_usage=True`.

**k6 profile (vf/k6.py, measured 2026-11-02):**
- **Generation step times:**
  - Streamed: 23.9 s/step.
  - DiT fp8-resident with prompts cached: 20.6 s/step.
  - Resident plus SageAttention 1.0.6 (PyPI, Triton, patched into the K6 attention dispatch): **12.6 s/step**. PSNR is 36–46 dB vs native, visually identical.
  - 480x864: 43 s/step, peak 6.4 GB. Use it only for face close-ups (`hires: true`).
  - 576x1024 (`gen: {k6_size: [576, 1024]}`), even at only 65 frames: peak 7.8 GB, **986 MB spilled**, 10 min build
    (2026-11-03, detective_phone). Visibly sharper than 480x864, but over the card; needs a memory fix before reuse.
  - Root cause (2026-11-03 audit): sampling ran with NO allocator cap; reserved grew past the card (fragmentation,
    ~1.3-1.9 GB over allocated) and WDDM paged it. Earlier "6.4 GB" 480x864 builds also spilled up to 476 MB, hidden
    because only max_memory_allocated was logged. Fixed: K6_SAMPLE_CAP 6.9 GB + PYTORCH_CUDA_ALLOC_CONF; after the
    fix 480x864x33 reserved 5.25 GB, no spill. Always log reserved + mem_get_info. Details: videoforge CLAUDE.md
    "Resource hygiene"; cache cleanup is `vf gc` (auto after builds).
- **Finishing chain**, about 30 s per clip in total:
  1. Lip-sync (optional).
  2. Deblur: RTX VSR at 1x, `DEBLUR_HIGH` (sharpness 71 vs 35).
  3. Exposure match.
  4. NVIDIA VideoFrameGeneration to 60/speed fps: 6.4 s vs RIFE 12 s, same look.
  5. VSR 4x `ULTRA`.
- Diffusers' own `sage` backend needs sageattention 2.1+. The 2.2 Windows wheel from woct0rdho's GitHub was blocked as an untrusted binary, so use the PyPI 1.0.6 patch instead.
- **Lip-sync:**
  - LatentSync 1.6 runs at about 10.8 GB peak with spill, 14 min per 5 s.
  - Its alignment needs MuseTalk/LatentSync crops built from scale-from-mouth geometry (see `vf/gen/latentsync_run.py`).
  - It still loses beards on 3/4 faces.

**Resource fix (after the user said "CPU, RAM, GPU temps going high"; measured 2026-11-02 with vf/gen/resmon.py):**
- **What changed in k6_batch:**
  - The 16 GB text encoder and the audio models are not loaded; the text encoder loads lazily only for a new prompt, then is freed.
  - Threads are capped at 4.
  - Thermal guard pauses at 82 C and resumes at 72 C.
- **Results:**
  - Process RAM: about 21 GB before, now 2.7 GB steady (8.7 GB while loading). A new prompt costs one 20 GB spike for about 22 s.
  - CPU: 13% average.
  - GPU: 62 C peak.
  - Speed is unchanged: 12.6 s/step.
- **Finishing chain:** 32 s/clip, 2.3 GB RAM, 57 C.
- **Monitoring:** wrap any heavy run in `python -m vf.gen.resmon -- <cmd>`.

**GPU hang, 2026-11-02:**
- What was running:
  - a Kandinsky 6 batch (~21 GB of process RAM)
  - pip installs
  - a TTS/spec process
  - an ffmpeg encode
- Free RAM fell to 2 GB. The K6 process got stuck inside the driver: unkillable, nvidia-smi and CuPy hang.
- Rules:
  - Run K6 alone, with nothing else heavy on the box.
  - Restart the batch process between batches.
  - Never install packages while it generates.
- Recovery needs a reboot or the user.

**Gotchas:**
- Above 8 GB of VRAM, Windows silently pages to system RAM and steps get 4–15× slower.
- Vulkan device 0 is the AMD iGPU, so ncnn tools need `-g 1`.
- CuPy's gaussian_filter is slow at large sigma; use a blur pyramid.
- First kernel runs compile for seconds, so profile after a warm-up.
- Don't benchmark while another GPU job runs: it halves the speed.

**Why:** these were the root causes of hours-long passes.
**How to apply:**
- Check VRAM residency and the GPU index before blaming a model.
- Cache at function, episode and global level.
- Park the AI worker (`park_gen`) before rendering; `stop_gen` only for a full reset.

- Kandinsky 6 (2026-11-02): never use the TAEHV tiny decoder for final clips; it causes blocky skin and near-black frames. Use the full VAE decode with the DiT moved to the CPU during decode, tiling, and `torch.cuda.set_per_process_memory_fraction` (6.5 GB cap) so overflow raises an OOM and retries with smaller tiles instead of spilling into shared RAM. Spilling made the decode 4x slower (732 s vs 181 s per clip).

**InfiniteTalk (2026-11-10):** compute-bound, ~81 s/step per 81-frame window on the 4060 (int8 14B, 4 steps). Batched takes in one process (load once), SageAttention 2.2 (-5 %), preload 3000 MB (-4 %). Biggest remaining lever = frames: any line + 1.6 s margin over 3.24 s needs a second window (+~180 s).
**Computation caching (2026-11-10):** prompt embeddings are cached for K6, Z-Image and now Klein (cache/klein_emb, bit-identical, -40 % per run). Step-skip caches (MagCache/TeaCache) on 4-step distilled InfiniteTalk = -45 % time but visibly broken faces (SyncNet still passed): rejected; always A/B frames, metrics alone miss it.

## Model precision pass (2026-11-10, videoforge 1.2.0)
- On the RTX 4060 8 GB, diffusers "fp8 storage + bf16 compute" (layerwise casting) is a TRAP: slower than plain BF16
  block-offload (K6 16.9 vs 13.3 s/step) and drifts from the model (Z-Image SSIM 0.82-0.87 vs its own bf16; different coat).
- W8A8 INT8 via WanGP's fused Triton kernel (vf/gen/int8lin.py; per-channel weights, per-row activations, norms/embeds
  kept bf16): 1.3x bf16 matmul, relerr 0.011. Z-Image 18.1 -> 12.6 s (+ regional compile -> 10.9 s), Klein -10 %, both closer to bf16.
- FP8 _scaled_mm is 1.7-2x bf16 but per-tensor activation scale -> relerr 0.038: rejected for InfiniteTalk (only ~10 % over its INT8).
- Always A/B vs the model's own BF16 output (same seed), not vs the previous path. Keep old cache keys reachable (fallback lookup).
- torch.compile can't trace WanGP Triton kernels: wrap the call in torch.compiler.disable; regional compile_repeated_blocks works.

**InfiniteTalk tuning (2026-11-10):** INT8 ≈ BF16 (SyncNet within 0.07); compile slower. audio_cfg 1.0 = one pass per step, -42 % time, but on 4 unseen takes SyncNet mean 7.14 -> 6.73 and mouths open less: rejected as default (looked better on the single search take = overfit). Always validate a tuning win on unseen takes + mouth frames before adopting. Steps 3 breaks faces; 6/8 no gain.

**CPU-only (2026-11-11, measured):** `vf build` finishes without NVIDIA. Encoder = vf/enc.py: NVENC when a test encode works, else libx264 (segments, full cut) / libx265 (phone); `VF_ENCODER=auto|nvenc|cpu`. Compositor falls back to NumPy when CuPy can't start (`VF_GPU=0` forces it). Demo episode 60 s 540x960 preview: GPU+NVENC 9 s; GPU+libx264 12.6 s; CPU-only 12 min 44 s (~85x). CPU-only PSNR vs GPU 39 dB (NumPy pixels differ a little; not investigated). CPU-encoded segments are cached under their own key (`.cpu`). AI stages (k6, lipsync, upscale) still write lossless temps with NVENC: fail on CUDA cards without NVENC (A100/H100).
