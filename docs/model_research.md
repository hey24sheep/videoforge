# Model research log: talk stage speed and heat

What was looked at to make the talk stage (InfiniteTalk on Wan2.1-I2V-14B) faster and cooler on an 8 GB RTX 4060 laptop
with 31 GB RAM, when, and why each idea was kept, rejected or left open. Measured results live in
[model_optimization_plan.md](model_optimization_plan.md) section 6; this file is the research log. Add a row for every
new idea, with its date and source, before testing it.

Licence rule: open weights usable commercially only (Apache-2.0, MIT, or a licence without a non-commercial clause).

## Why the talk stage is slow and hot (2026-11-14)

- The model is about 17 GB at int8 (14B base + InfiniteTalk module). The card holds about 5 GB of it; the rest is
  copied from RAM on every one of the 4 steps (about 105 s per step).
- On 4 cores the CPU sensor reached 80-82 C on every step and 88-89 C at the video decode, so the thermal guard killed
  two builds.
- On 2 cores (`VF_TALK_CORES=2`) it still reached 81-83 C with the GPU at 68-69 C. The laptop's CPU and GPU share one
  cooler, so GPU load heats the CPU sensor. Conclusion: less GPU work helps heat more than fewer cores or less copying.
  UNVERIFIED: no real per-core sensor (the CPU reading is the ACPI zone + 8 C); LibreHardwareMonitor would confirm it.

## Test machine and setup (2026-11-14)

- Laptop: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB, Ada, driver 617.14), AMD Ryzen 9 8945H (16 threads), 31 GB RAM,
  Windows 11 Home. CPU and GPU share one cooler. The owner's target: safe in a 45 C room.
- Thermal guard on every run (vf/thermal.py via jobwatch): throttle 72 C, pause 80 C, resume 70 C, kill 88 C.
- InfiniteTalk baseline: WanGP commit f0b54e0 (2026-11-06), Wan2.1-I2V-14B-480P + InfiniteTalk single, quanto INT8
  W8A8, lightx2v step-distill LoRA, 4 steps, audio_cfg 2.0, 480x832, 81-frame windows, SageAttention 2.2, offload
  profile 4 + VF_TALK_PIN 0.2, VF_TALK_CORES 2, torch 2.11.0+cu128. Measured 118-144 s per step including guard pauses.
- StableAvatar: repo Francis-Rings/StableAvatar at 3862518 (2026-09-25); weights FrancisRing/StableAvatar
  (StableAvatar-1.3B/transformer3d-square.pt + Wan2.1-Fun-V1.1-1.3B-InP + wav2vec2-base-960h, 23 GB); own venv with
  torch 2.11.0+cu128, diffusers 0.30.1, transformers 4.51.3, accelerate 1.15.0. Runner: tools/research/stableavatar_8gb.py.
  480x832, 81 frames, 25 steps, text scale 3.0, audio scale 5.0, seed 79. Weights loaded with `weights_only=True`.
  Model files, venv and clone were moved to the Recycle Bin afterwards; the repo, Hugging Face id and versions above
  recreate them.
- HuMo: repo Phantom-video/HuMo at 845f447 (2026-01-26) read, not run.

## Ideas

| Date | Idea | Status | Why |
|---|---|---|---|
| 2026-11-10 | FP8 weights for InfiniteTalk | Rejected | 3.5x the error of WanGP quanto INT8 for about 10 % speed (measured, section 6). |
| 2026-11-10 | `audio_cfg` 1.0 (half the compute per step) | Drafts only | -42 % time but weaker mouths and lower SyncNet on unseen takes (measured, 6b). |
| 2026-11-10 | `torch.compile`, BF16 compute | Rejected | No gain on this card (measured, 6b). |
| 2026-11-10 | MagCache step skipping | Rejected | -45 % time, SyncNet passed, but crushed contrast, green beard cast, blotchy faces: 4 distilled steps leave nothing to skip (measured). |
| 2026-11-10 | SageAttention 2.2, `--preload 3000`, one model load per build | Kept | -5 % and -4 %, lip-sync unchanged; batch saves 60-90 s per extra talk shot (measured). |
| 2026-11-14 | GPUDirect Storage / fastsafetensors (disk straight to GPU) | Rejected | Linux only. |
| 2026-11-14 | DirectStorage on Windows for model weights | Rejected | Only hobby projects, no PyTorch path. |
| 2026-11-14 | Pinned (page-locked) RAM, WanGP profile 4 + `VF_TALK_PIN=0.2` | Open | The GPU copies pinned blocks by DMA without the CPU. At 0.2 only 11 of 39 blocks fit and free RAM fell to about 5 GB; step time about the same (105 s). Default stays profile 5 until a clean build finishes. |
| 2026-11-14 | 2 cores for the talk stage + cool wait before decode | Kept | Avoids the 88 C kill at decode. Did not stop 80 C pauses during steps (see above). |
| 2026-11-14 | GGUF Q4_K_M / Q3_K_M of Wan2.1-I2V-14B-480P (city96) + InfiniteTalk Q4 (Kijai) | Open, low value | 11.3 GB / 8.6 GB + 1.4 GB, about 30 % less than int8, not half. WanGP loads GGUF, but its fast GGUF CUDA kernels (claimed +15 %) have no wheel for Python 3.12 / torch 2.11 (built for py310 torch 2.7.1, py311 torch 2.10 cu130). Without them weights are unpacked slowly: likely same speed or slower. Cuts copying, not GPU math, so little heat gain. One guide reports Q4 visibly worse than Q5 for face consistency. |
| 2026-11-14 | W4A8 INT8 (4-bit weights, 8-bit math, WanGP) | Blocked | Works on RTX 30/40, but no InfiniteTalk W4A8 file exists and WanGP's `--save-quantized` only writes 8-bit. |
| 2026-11-14 | NVFP4 / Nunchaku FP4 | Rejected | Needs an RTX 50 card (Blackwell FP4); Ada has none. |
| 2026-11-14 | Nunchaku / SVDQuant INT4 for Wan | Rejected | No Wan or InfiniteTalk model found. |
| 2026-11-14 | TalkVerse 5B (Wan2.2-TI2V-5B LoRA, CVPR 2026) | Rejected | Snap Inc. non-commercial licence. |
| 2026-11-14 | StableAvatar (Wan2.1-1.3B based, MIT) | Rejected (tested) | Measured on this laptop, 480x832, 81 frames, 25 steps: does not fit 8 GB as shipped (its 3 guidance passes run as one batch: OOM); with the passes run one at a time it ran ~100-140 s per step, so ~45-60 min per take against ~7-11 min for InfiniteTalk 14B (4 steps), and heated the machine to CPU 87 C / GPU 74 C with repeated guard pauses. Stopped at step 12 of 25. Needs no weight streaming, but 25-50 undistilled steps x 3 passes is far more compute than 4 distilled steps of the 14B. Quality not judged (512x512 "square" weights were used; the 480x832 "rec-vec" set was not downloaded). |
| 2026-11-14 | HuMo 1.7B (ByteDance, Apache-2.0) | Rejected (not run) | Same class as StableAvatar: Wan 1.3B base, 50 steps, two guidance scales (3 passes); authors quote 8 min per 480p clip on a 32 GB GPU; torchrun + FSDP setup; generates from a reference face instead of animating our keyframe. StableAvatar's measured 6x slowdown applies. |
| 2026-11-14 | OmniAvatar 1.3B | Rejected | Hugging Face repo states no licence (no licence = no permission to use). |
| 2026-11-14 | DirectStorage -> D3D12 buffer -> CUDA external memory (Windows answer to GPUDirect Storage) | Rejected for the talk stage | Real on Windows: described in the fastsafetensors docs, and a community DLL reports 4.1x faster LLM loads (self-reported). But it speeds up loading from disk, and the talk step is compute-bound: about 12 GB copied RAM -> GPU per step is about 1-2 s of a 105 s step (CLAUDE.md: ~81 s/step of pure compute). It could only cut the first model load (about 90 s). |
| 2026-11-14 | Zero-copy mapped host memory (cudaHostRegister on mmap'd weights) | Rejected | The GPU reads weights over PCIe at every use; a 2026 preprint measured llama.cpp-style mapped weights over PCIe 146.9 -> 3.79 tok/s. Staging into VRAM (what WanGP does) is the right policy on a discrete GPU. |
| 2026-11-14 | LongCat-Video-Avatar 1.5 (MIT, in WanGP) | Rejected | About 64 GB of bf16 weights (larger than InfiniteTalk) and 8 steps against our 4: slower here. |
| 2026-11-14 | Motar (arXiv 2609.10317, 77M motion + 1.7B renderer, 15 fps on H200) | Watch | No code or weights released. |
| 2026-11-14 | Avatar-Forever (arXiv 2608.12107, LTX-2.3 22B, 4 steps, 27 fps on H100) | Rejected | 22B base: bigger than what we run; weights licence not stated. |
| 2026-11-14 | DynaForcing (arXiv 2608.17707, Wan2.2-S2V 14B) | Rejected | 14B, no release. |
| 2026-11-14 | Hallo-Live (arXiv 2604.23632, Ovi-based, 20 fps on 2x H200) | Watch | Code and models said to be public (github fudan-generative-vision/Hallo-Live); licence and size not checked; generates its own audio track (Ovi), not driven by our voice takes. |
| 2026-11-14 | Persistent talk worker (load the 14B once per session) | Open | Saves the model load (about 90 s) per build, not per-step heat. |

## Key fact

Small undistilled models are not faster: a 1.3B model at 25-50 steps x 3 guidance passes does more GPU work than the
14B at 4 distilled steps with 1-2 passes (measured: StableAvatar about 6x slower). A small model only wins with a
few-step distillation (none released for these avatar models as of 2026-11-14).


The talk step is compute-bound: loading tricks (GPUDirect, DirectStorage, pinning, Q4) cut copying, which is about 1-2 s
of each 105 s step. Only less compute helps per step: a smaller model, fewer frames, fewer steps or sparse attention.

## Next test (when chosen)

Blind A/B on two real talk shots, same keyframe and voice take: InfiniteTalk 14B int8 vs each 1.3B candidate that
passes the licence check. Measure SyncNet offset and confidence, silent-mouth, face match, s/step, peak CPU and GPU
temperature, guard pauses; plus a blind Opus frame review. `tools/talk_bench.py` measures speed and heat per setting.

## Sources

- city96 Wan2.1-I2V-14B-480P GGUF: https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf
- Kijai InfiniteTalk GGUF: https://huggingface.co/Kijai/WanVideo_comfy_GGUF
- WanGP (GGUF kernels, W4A8, NVFP4 notes in docs/CHANGELOG.md and docs/INSTALLATION.md): https://github.com/deepbeepmeep/WanGP
- GGUF kernel wheels: https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.29
- StableAvatar: https://huggingface.co/FrancisRing/StableAvatar , https://arxiv.org/abs/2508.08248
- HuMo: https://comfyui-wiki.com/en/models/humo/humo
- OmniAvatar 1.3B: https://huggingface.co/OmniAvatar/OmniAvatar-1.3B
- TalkVerse: https://arxiv.org/abs/2512.14938
- Q3 14B GGUF on 8 GB, same speed (one user report): https://www.reddit.com/r/StableDiffusion/comments/1koefcg/new_wan21vace14bggufs/
- Wan 2.2 GGUF guide (Q4 vs Q5 faces): https://wan27.org/zh/blog/wan-2-2-gguf-guide
- fastsafetensors (Windows DirectStorage design): https://github.com/foundation-model-stack/fastsafetensors/blob/main/docs/overview.md
- DirectStorage + CUDA LLM loader (community, self-reported): https://news.ycombinator.com/item?id=46949763
- File-backed weights over PCIe (preprint): https://arxiv.org/abs/2608.12114
- LongCat-Video-Avatar: https://huggingface.co/meituan-longcat/LongCat-Video-Avatar
- Motar: https://arxiv.org/abs/2609.10317 ; Avatar-Forever: https://arxiv.org/abs/2608.12107 ;
  DynaForcing: https://arxiv.org/abs/2608.17707 ; Hallo-Live: https://arxiv.org/abs/2604.23632
