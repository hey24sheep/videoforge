"""Kandinsky 6.0 Lite distill image-to-video, batch, tuned for an 8 GB laptop (RTX 4060, measured 2026-11-02).
Usage (gen venv): python -m vf.gen.k6_batch jobs.json   jobs = [{id, image, prompt, out, seed?, frames?, width?,
height?, steps?}]. Skips jobs whose `out` exists. Writes lossless h264 (NVENC) at 24 fps.

Speed: fp8-stored DiT resident on the GPU + SageAttention (auto kernel), prompt embeddings cached, TAEHV decode,
no audio stream. 384x672 x 121 frames ~9-11 s/step; 480x864 ~43 s/step.
Resources (user, 2026-11-02: "CPU, RAM and GPU temps go high"):
  - the 16 GB Qwen2.5-VL text encoder (+ CLIP) is loaded ONLY for prompts not in cache/k6_emb, then freed;
    the audio VAE + vocoder are never loaded (sample_audio=False)
  - CPU threads capped (K6_THREADS, default 4)
  - thermal guard: above K6_TEMP_HI (default 82 C) the run pauses between denoising steps until K6_TEMP_LO (72 C)
"""
import gc, hashlib, json, os, subprocess, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

THREADS = int(os.environ.get('K6_THREADS', 4))
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, str(THREADS))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.7,max_split_size_mb:128')
import torch
from PIL import Image

torch.set_num_threads(THREADS)
M = os.path.join(_AI, 'models', 'hf', 'kandinsky6-lite-distill')
TAE = os.path.join(_AI, 'models', 'hf', 'taehv_hunyuan', 'taehv.pth')
EMB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cache', 'k6_emb')
FFMPEG = os.environ.get('VF_FFMPEG', r'C:\Users\trey5\AppData\Local\Microsoft\WinGet\Packages'
                        r'\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe')
H0, W0 = int(os.environ.get('K6_H', 672)), int(os.environ.get('K6_W', 384))  # per-job width/height override
STEPS = int(os.environ.get('K6_STEPS', 6))
DECODE = os.environ.get('K6_DECODE', 'vae')  # vae (full, default) | tae (tiny, fast) | both (A/B test)
# 2026-11-02 A/B: the TAEHV tiny decoder caused the blocky skin and crushed near-black frames; the full VAE fixes both.
# The full VAE decode must never spill into shared system RAM (8 GB card): the DiT leaves the GPU during decode, the
# VAE decodes in small tiles, and a hard allocator cap turns any overflow into an OOM that is retried with smaller tiles.
# Kandinsky clips come out very dark (median luma 2-19/255). Lifting them after 8-bit quantisation stretches a few
# code values into visible steps (blotchy shadows, 2026-11-02). So a fixed filmic lift y=(1+k)x/(1+kx) is applied
# here in float, before the 8-bit write; finishing (vf/k6._expo) only trims what is left.
PRELIFT = float(os.environ.get('K6_PRELIFT', 2.0))
# 2026-11-03 audit: sampling had NO cap, so the caching allocator grew past the 8 GB card and Windows paged the excess
# to shared RAM (576x1024: 986 MB spilled, steps 45 -> 97 s; 480x864 builds spilled up to 476 MB unnoticed because only
# max_memory_allocated was logged). Now the whole process runs under SAMPLE_CAP: at the cap the allocator frees its
# cache and retries, and a real overflow raises OOM instead of spilling.
SAMPLE_CAP = float(os.environ.get('K6_SAMPLE_CAP', 6.9))  # GB for sampling (card 8 GB minus desktop/driver/context)
VRAM_CAP = float(os.environ.get('K6_VRAM_CAP', 6.5))  # GB the decode may reserve (card 8 GB, desktop/driver ~0.5-1)
# (tile px, tile stride px, tile frames, frame stride) from largest to smallest
TILES = [(256, 192, 16, 12), (192, 160, 12, 8), (128, 96, 8, 4)]
# measured 2026-11-02 (384x672 x 121 f, 4 steps) vs PyTorch SDPA: streamed 23.9 s/step; resident 20.6; resident +
# SageAttention 1.0.6 (Triton) 12.6 s/step 42.5 dB; 2.2 auto (INT8 QK + FP8 PV) 11.4 s/step 42.4 dB; 2.2 explicit
# FP16-PV/fp32-accum 14.0 s/step 39.1 dB (slower AND less faithful) -> sageattn auto dispatcher.
RESIDENT = os.environ.get('K6_RESIDENT', '1') == '1'
PREC = os.environ.get('K6_PREC', 'bf16')  # bf16 (default since 2026-11-10, block-offloaded) | fp8store (old; re-bakes on first use) | int8
ATTN = os.environ.get('K6_ATTN', 'sage')  # 'sage' = sageattention.sageattn patched in; '' = PyTorch SDPA
TEMP_HI, TEMP_LO = int(os.environ.get('K6_TEMP_HI', 82)), int(os.environ.get('K6_TEMP_LO', 72))
cuda, cpu = torch.device('cuda'), torch.device('cpu')


def gpu_temp():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def cool_down(where=''):
    t = gpu_temp()
    if t < TEMP_HI:
        return 0.0
    t0 = time.time()
    print(f'  thermal guard {where}: GPU {t} C >= {TEMP_HI} C, pausing...', flush=True)
    while t > TEMP_LO and time.time() - t0 < 300:
        time.sleep(3)
        t = gpu_temp()
    print(f'  resumed at {t} C after {time.time() - t0:.0f}s', flush=True)
    return time.time() - t0


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return float('nan')


def _emb_key(a, kw):
    return hashlib.sha1(repr((a, sorted((k, v) for k, v in kw.items() if not torch.is_tensor(v)))).encode()).hexdigest()


def _baked_transformer(diffusers):
    """The fp8-prepared DiT, baked once to one safetensors file (vf/gen/video.load_baked) and then loaded straight
    to the GPU: ~53 s and an 8.7 GB RAM staging copy -> a few seconds. Falls back to from_pretrained on any error."""
    try:
        from .video import load_baked
        if PREC == 'bf16':
            return None
        if PREC == 'int8':
            from .int8lin import int8_linears
            return load_baked(diffusers.Kandinsky6Transformer3DModel, M, 'transformer', 'k6lite_distill_int8',
                              int8_linears)
        return load_baked(diffusers.Kandinsky6Transformer3DModel, M, 'transformer', 'k6lite_distill_fp8',
                          lambda m: m.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                                               compute_dtype=torch.bfloat16))
    except Exception as e:
        print(f'  baked transformer unavailable ({type(e).__name__}: {e}); loading normally', flush=True)
        return None


def _canon_kw(prompt):
    """The exact encode_prompt arguments Kandinsky6TI2VAPipeline.__call__ passes for our calls (guidance 1.0, one
    video, default negative prompt and length, fp8 DiT whose compute dtype is bf16), so pre-pass keys match."""
    return dict(prompt=[prompt], negative_prompt=None, do_classifier_free_guidance=False, num_videos_per_prompt=1,
                prompt_embeds=None, pooled_prompt_embeds=None, negative_prompt_embeds=None,
                negative_pooled_prompt_embeds=None, max_sequence_length=1024, device=cuda, dtype=torch.bfloat16)


def _emb_path(prompt):
    return os.path.join(EMB, _emb_key((), _canon_kw(prompt)) + '.pt')


def encode_missing(jobs):
    """Encode every uncached prompt in one pass BEFORE the DiT loads: only the text encoders are in memory, under
    the guardrails of vf/gen/guard.py (explicit --allow-encode, RAM/GPU headroom, allocator cap). Returns the count."""
    miss = list(dict.fromkeys(j['prompt'] for j in jobs if not os.path.exists(_emb_path(j['prompt']))))
    if not miss:
        return 0
    from .guard import allow_encoder, encode_cap
    allow_encoder('kandinsky6 text encoder (Qwen2.5-VL, 16 GB)', len(miss), what=miss)
    import diffusers
    from diffusers.hooks import apply_group_offloading
    os.makedirs(EMB, exist_ok=True)
    t = time.time()
    te = diffusers.Kandinsky6TI2VAPipeline.from_pretrained(M, torch_dtype=torch.bfloat16, transformer=None, vae=None,
                                                           audio_vae=None, vocoder=None)
    try:
        for n in ('text_encoder', 'text_encoder_2'):
            # use_stream=False: streams pin host memory that the driver books as 'shared GPU memory' and keeps
            # (2026-11-03: 2.6 GB shared for the rest of the run)
            apply_group_offloading(getattr(te, n), onload_device=cuda, offload_device=cpu, offload_type='leaf_level',
                                   use_stream=False, low_cpu_mem_usage=True)
        with encode_cap(), torch.no_grad():
            for p in miss:
                r = te.encode_prompt(**_canon_kw(p))
                torch.save(tuple(x.cpu() if torch.is_tensor(x) else x for x in r), _emb_path(p))
    finally:
        del te
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch._C, '_host_emptyCache'):
            torch._C._host_emptyCache()
    print(f'  encoded {len(miss)} new prompt(s) in {time.time() - t:.0f}s; text encoder freed, RAM {rss_gb():.1f} GB',
          flush=True)
    return len(miss)


def load():
    import diffusers
    from diffusers.hooks import apply_group_offloading
    from .taehv import TinyWanDecoder
    # text encoders + audio models are NOT loaded here (16 GB + 2.2 GB of RAM); see encode_missing above
    tr = _baked_transformer(diffusers) if RESIDENT and PREC != 'bf16' else None
    extra = dict(transformer=tr) if tr is not None else {}
    pipe = diffusers.Kandinsky6TI2VAPipeline.from_pretrained(M, torch_dtype=torch.bfloat16, text_encoder=None,
                                                             text_encoder_2=None, audio_vae=None, vocoder=None, **extra)
    tr = pipe.transformer
    if not extra and PREC != 'bf16':
        tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    if RESIDENT and PREC != 'bf16':  # fp8 DiT (~3.2 GB) stays on the card
        tr.to(cuda)
    else:
        apply_group_offloading(tr, onload_device=cuda, offload_device=cpu, offload_type='block_level',
                               num_blocks_per_group=1, use_stream=False, low_cpu_mem_usage=True)  # streams pin host RAM (booked as shared)
    if ATTN in ('sage', 'sage1'):  # diffusers' own 'sage' backend needs >=2.1 and a dispatch path K6 doesn't use
        import diffusers.models.transformers.transformer_kandinsky6 as k6m
        import sageattention as sa
        orig_attn = k6m.dispatch_attention_fn

        def fast(q, k, v, attn_mask=None, **kw):
            if attn_mask is None and q.dtype in (torch.float16, torch.bfloat16) and q.shape[-1] in (64, 96, 128):
                return sa.sageattn(q, k, v, tensor_layout='NHD', is_causal=False)
            return orig_attn(q, k, v, attn_mask=attn_mask, **kw)
        k6m.dispatch_attention_fn = fast
    elif ATTN:
        tr.set_attention_backend(ATTN)
    pipe.vae.to(cuda)
    type(pipe)._execution_device = property(lambda self: cuda)
    os.makedirs(EMB, exist_ok=True)

    def lazy_encode(*a, **kw):
        p = os.path.join(EMB, _emb_key(a, kw) + '.pt')
        if os.path.exists(p):
            return tuple(x.to(cuda) if torch.is_tensor(x) else x for x in torch.load(p, weights_only=True))
        # guardrail (2026-11-03): never load the 16 GB text encoder next to the resident DiT. encode_missing() runs
        # before load(); a miss here means the call's arguments changed and the cache key no longer matches.
        raise RuntimeError(f'k6: prompt embedding missing after the encode pre-pass (key {os.path.basename(p)}); '
                           f'encode_prompt arguments changed: update _canon_kw')
    pipe.encode_prompt = lazy_encode
    if DECODE == 'vae':  # the tiny decoder is unused: don't park it on the card
        return pipe, None
    tae = TinyWanDecoder(latent_channels=16, patch_size=1)
    sd = torch.load(TAE, map_location='cpu', weights_only=True)
    tae.load_state_dict({k: v for k, v in sd.items() if k.startswith('decoder.')}, strict=True)
    return pipe, tae.to(cuda, torch.float16).eval().requires_grad_(False)


def _cap(gb):
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2**30 / total))


def _vram_line():
    free, total = torch.cuda.mem_get_info()
    return (f'alloc peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, reserved peak '
            f'{torch.cuda.max_memory_reserved() / 2**30:.2f} GB, card free {free / 2**30:.2f}/{total / 2**30:.2f} GB')


def full_decode(pipe, lat):
    """Full-VAE decode in small spatial/temporal tiles with the DiT parked on the CPU. Returns (T,3,H,W) [0,1] on CPU."""
    tr, vae = pipe.transformer, pipe.vae
    t = time.time()
    tr.to(cpu)
    gc.collect()
    torch.cuda.empty_cache()
    _cap(VRAM_CAP)
    torch.cuda.reset_peak_memory_stats()
    z = lat.to(vae.dtype) / vae.config.scaling_factor
    try:
        for i, (px, st, nf, fs) in enumerate(TILES):
            vae.enable_tiling(tile_sample_min_height=px, tile_sample_min_width=px, tile_sample_min_num_frames=nf,
                              tile_sample_stride_height=st, tile_sample_stride_width=st,
                              tile_sample_stride_num_frames=fs)
            try:
                fv = vae.decode(z, return_dict=False)[0]
                break
            except torch.OutOfMemoryError:
                fv = None
                gc.collect()
                torch.cuda.empty_cache()
                print(f'  vae decode: tiles {px}px/{nf}f over the {VRAM_CAP} GB cap, retrying smaller', flush=True)
        if fv is None:
            raise RuntimeError(f'vae decode does not fit in {VRAM_CAP} GB even at the smallest tiles')
        fv = ((fv[0].float().cpu() + 1) / 2).permute(1, 0, 2, 3)  # (C,T,H,W) [-1,1] -> (T,C,H,W) [0,1]
        print(f'  vae decode {time.time() - t:.0f}s tiles {px}px/{nf}f  VRAM peak alloc '
              f'{torch.cuda.max_memory_allocated() / 2**30:.2f} GB reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB',
              flush=True)
        return fv
    finally:
        del z
        _cap(SAMPLE_CAP)  # back to the sampling cap, never uncapped
        torch.cuda.empty_cache()
        t2 = time.time()
        tr.to(cuda)
        print(f'  DiT back on GPU in {time.time() - t2:.1f}s', flush=True)


def write(frames, out):
    """frames: (T, 3, H, W) float [0,1] on GPU -> lossless h264 (NVENC)."""
    frames = frames.clamp(0, 1)
    if PRELIFT > 0:
        frames = (1 + PRELIFT) * frames / (1 + PRELIFT * frames)
    a = (frames * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    tmp = out + '.part.mp4'
    elog = out + '.ffmpeg.log'
    # 2026-11-06: ffmpeg died mid-pipe (the whole process tree was killed when the Claude session exited) and the
    # bare `OSError: [Errno 22]` lost a finished 5-min clip with no reason. Now: ffmpeg's stderr goes to a file, a
    # failed encode is retried once from the frames still in RAM, and the error names ffmpeg's exit code + message.
    err = None
    for attempt in range(2):
        with open(elog, 'wb') as ef:
            p = subprocess.Popen([FFMPEG, '-v', 'error', '-y', '-threads', str(THREADS), '-f', 'rawvideo', '-pix_fmt',
                                  'rgb24', '-s', f'{a.shape[2]}x{a.shape[1]}', '-r', '24', '-i', '-', '-c:v',
                                  'h264_nvenc', '-tune', 'lossless', '-preset', 'p4', '-pix_fmt', 'yuv420p', tmp],
                                 stdin=subprocess.PIPE, stderr=ef)
            rc = None
            try:
                p.stdin.write(a.tobytes())
                p.stdin.close()
                rc = p.wait(timeout=600)
            except (OSError, ValueError) as e:  # ffmpeg gone: Errno 22 / EPIPE on Windows
                err = f'{type(e).__name__}: {e}'
            finally:
                if p.poll() is None:  # an exception mid-pipe must not orphan ffmpeg
                    p.kill()
                    p.wait()
                rc = p.returncode if rc is None else rc
        if rc == 0 and err is None:
            os.replace(tmp, out)
            if os.path.exists(elog) and not os.path.getsize(elog):
                os.remove(elog)
            return
        msg = open(elog, 'rb').read().decode('utf-8', 'replace').strip()[-600:]
        print(f'  ffmpeg write failed (attempt {attempt + 1}/2, rc {rc}, {err or "no pipe error"}): {msg or "(no stderr)"}',
              flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
        err = None
        time.sleep(2)
    raise RuntimeError(f'ffmpeg failed twice writing {out} (see {elog})')


def main(path):
    jobs = [j for j in json.load(open(path)) if not os.path.exists(j['out'])]
    if not jobs:
        print('nothing to do', flush=True)
        return
    t = time.time()
    if any(not os.path.exists(_emb_path(j['prompt'])) for j in jobs):
        # The encoder pass runs in its own process: freeing the 16 GB encoder in-process left RSS at 17.6 GB for the
        # whole batch (2026-11-03, free RAM fell to 3.4 GB). A child returns all of it to Windows when it exits.
        import subprocess
        rc = subprocess.run([sys.executable, '-m', 'vf.gen.k6_batch', '--encode-only', path], timeout=1800).returncode
        if rc:
            raise SystemExit(f'k6 encode pass failed (exit {rc})')
    _cap(SAMPLE_CAP)  # before any weight lands on the card
    pipe, tae = load()
    print(f'loaded in {time.time() - t:.0f}s; {len(jobs)} clips; RAM {rss_gb():.1f} GB', flush=True)
    failed = []
    for j in jobs:
        cool_down('before clip')
        t0 = time.time()
        W, H = int(j.get('width', W0)), int(j.get('height', H0))
        img = Image.open(j['image']).convert('RGB')
        sw, sh = img.size  # cover-crop to the clip aspect then resize
        tw = min(sw, sh * W // H)
        th = tw * H // W
        img = img.crop(((sw - tw) // 2, (sh - th) // 2, (sw - tw) // 2 + tw, (sh - th) // 2 + th)).resize(
            (W, H), Image.LANCZOS)
        marks, paused, temps = [], [0.0], []

        def cb(p, i, ts, kw):
            torch.cuda.synchronize()
            temps.append(gpu_temp())
            paused[0] += cool_down(f'step {i + 1}')
            marks.append(time.time())
            return kw
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            try:
                lat, _ = pipe(prompt=j['prompt'], image=img, height=H, width=W, num_frames=int(j.get('frames', 121)),
                              num_inference_steps=int(j.get('steps', STEPS)), guidance_scale=1.0, sample_audio=False,
                              output_type='latent', generator=torch.Generator('cpu').manual_seed(int(j.get('seed', 7))),
                              return_dict=False, callback_on_step_end=cb)
            except torch.OutOfMemoryError:  # skip this clip, keep the batch going; the parent reports the miss
                print(f"FAILED {j['id']}: {W}x{H}x{j.get('frames', 121)} does not fit in the {SAMPLE_CAP} GB sampling "
                      f"cap (refusing to spill into shared RAM): use a smaller k6_size or fewer frames", flush=True)
                failed.append(j['id'])
                gc.collect()
                torch.cuda.empty_cache()
                continue
            print(f'  sampled in {time.time() - t0:.0f}s, {_vram_line()}', flush=True)
            peak_sample = torch.cuda.max_memory_allocated()
            fv = full_decode(pipe, lat) if DECODE in ('vae', 'both') else None
            if DECODE == 'vae':
                frames = fv
            else:
                frames = tae.decode((lat / pipe.vae.config.scaling_factor).float())[0]
            if DECODE == 'both':
                write(fv, j['out'][:-4] + '.vae.mp4')
            del fv
        del lat
        os.makedirs(os.path.dirname(j['out']), exist_ok=True)
        try:
            write(frames, j['out'])
        except RuntimeError as e:  # keep the batch going: one bad encode must not throw away the clips after it
            print(f"FAILED {j['id']}: {e}", flush=True)
            failed.append(j['id'])
            del frames
            torch.cuda.empty_cache()
            continue
        del frames
        torch.cuda.empty_cache()
        step = (marks[-1] - marks[0] - paused[0]) / max(1, len(marks) - 1) if len(marks) > 1 else float('nan')
        print(f"DONE {j['id']} {time.time() - t0:.0f}s  {W}x{H}  step {step:.1f}s  VRAM peak "
              f"{max(peak_sample, torch.cuda.max_memory_allocated()) / 2**30:.1f} GB  RAM {rss_gb():.1f} GB  GPU max "
              f"{max(temps or [0])} C  paused {paused[0]:.0f}s -> {j['out']}", flush=True)
    if failed:
        raise SystemExit(f'k6: {len(failed)} clip(s) failed (see the FAILED lines above: VRAM cap or encoder): {failed}')


if __name__ == '__main__':
    if sys.argv[1] == '--encode-only':
        encode_missing([j for j in json.load(open(sys.argv[2])) if not os.path.exists(j['out'])])
    else:
        main(sys.argv[1])
