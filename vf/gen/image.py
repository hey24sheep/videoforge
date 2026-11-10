"""Key-frame stage: text -> still image. Runs ONLY inside the gen venv; from the main interpreter use
vf.gen.bridge.run_gen('image', jobs).

    t2i(prompt, width=720, height=1280, seed=0, steps=None, out=None) -> png path   (global cache)
    run_jobs(jobs, emit=print)  /  python -m vf.gen.image jobs.json

Model: Z-Image-Turbo (Tongyi-MAI, Apache-2.0): 6B single-stream DiT distilled to 8 steps at CFG 0. Strong on
photographic/cinematic looks and on following long prompts. Speed plan for 8 GB:
  - DiT stored in fp8, computed in bf16: 12 GB -> ~6 GB, fully resident in VRAM (no offload traffic).
  - Qwen3-4B text encoder loads only on a prompt-embedding miss (embeddings memoised on the global tier), layer by
    layer on the GPU, and is dropped before the DiT loads.
  - Shares the GPU with the video stage through vf.gen.claim (only one of them is resident at a time).
Width/height must be multiples of 16; the result is RGB PNG.
"""
import gc, json, os, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
import numpy as np

from ..memo import memo, memo_file, key_for, tier_dir

REPO = 'Tongyi-MAI/Z-Image-Turbo'
MODEL_ID = 'z-image-turbo fp8-store bf16-compute flowmatch'
STEPS = 9          # 9 scheduler steps = 8 DiT evaluations (the distill's target)
ATTN_BACKEND = ''  # diffusers attention backend for the DiT ('' = PyTorch default; cuDNN measured no faster here)
FP8_COMPUTE = os.environ.get('VF_FP8_COMPUTE', '0') == '1'  # fp8 tensor-core matmuls (-32 % per still, different but equal-quality pixels): own cache keys
# 2026-11-10 A/B (docs/model_optimization_plan.md): W8A8 int8 is 30 % faster than fp8-store (12.6 vs 18.1 s per 560x976 still)
# and much CLOSER to the model's own bf16 output (SSIM 0.92-0.97 vs 0.82-0.87). New stills use int8; a still already cached
# under the fp8store key is reused (approved episodes stay identical). VF_ZIMAGE_PREC=fp8store brings the old path back.
PREC = os.environ.get('VF_ZIMAGE_PREC', 'int8')  # int8 (default) | fp8store | bf16 (reference, offloaded, slow)
PARK_GB = 6.0      # RAM a parked DiT + VAE takes
ENCODE_BATCH = 8
_M = {}
LAST = {}


def _torch():
    import torch
    return torch


def model_dir():
    from huggingface_hub import snapshot_download
    pat = ['model_index.json', 'scheduler/*', 'tokenizer/*', 'text_encoder/*', 'transformer/*', 'vae/*']
    try:
        return snapshot_download(REPO, allow_patterns=pat, local_files_only=True)
    except Exception:
        return snapshot_download(REPO, allow_patterns=pat)


def _free(*names):
    torch = _torch()
    for n in names:  # drop the references only: moving weights to the CPU first left that copy committed in RAM
        _M.pop(n, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release():
    _M.pop('parked', None)
    _free('te', 'tok', 'pipe', 'dit', 'vae')


def park():
    """DiT + VAE to CPU RAM (~6 GB; VRAM freed). _load_pipe moves them back in ~1 s. The text encoder is freed."""
    _free('te', 'tok')
    if 'pipe' in _M and not _M.get('parked'):
        _M['dit'].to('cpu')
        _M['vae'].to('cpu')
        _M['parked'] = True
    gc.collect()
    _torch().cuda.empty_cache()


def _unpark():
    if _M.get('parked'):
        t0 = time.time()
        _M['dit'].to('cuda')
        _M['vae'].to('cuda')
        _M['parked'] = False
        LAST['unpark_s'] = round(time.time() - t0, 1)


def _claim():
    from . import claim
    claim('image', release, park, size_gb=PARK_GB)


# ------------------------------------------------------------------ text encoder (only on embed miss)
_GUARD_OK = [False]  # set by encode_prompts after its guard check


def _load_te():
    if 'te' not in _M and not _GUARD_OK[0]:  # a single uncached prompt_embeds() call: same guardrail
        from .guard import allow_encoder
        allow_encoder('z-image text encoder (Qwen3-4B, 8.5 GB)', 1)
    _claim()
    if 'te' in _M:
        return
    if 'pipe' in _M:  # never hold Qwen3-4B and the DiT together: free the DiT (reload from baked weights ~6 s).
        # 2026-11-03: parking it in RAM next to the 8.5 GB encoder took free RAM down to 7.6 GB
        release()
    torch = _torch()
    from transformers import AutoModel, AutoTokenizer
    from accelerate import cpu_offload
    d = model_dir()
    _M['tok'] = AutoTokenizer.from_pretrained(d, subfolder='tokenizer')
    te = AutoModel.from_pretrained(d, subfolder='text_encoder', torch_dtype=torch.bfloat16).eval()
    cpu_offload(te, execution_device=torch.device('cuda'))
    _M['te'] = te


T2I_VRAM_CAP = float(os.environ.get('VF_T2I_VRAM_CAP', 6.8))  # GB the allocator may reserve while drawing
GEN_VERSION = 't2i-1'  # bump when the generated pixels would change; loader/speed edits don't


@memo('global', version=lambda: f'qwen3-1|{REPO}')
def prompt_embeds(prompt, max_len=512):
    """Qwen3 hidden_states[-2] of the chat-templated prompt, masked to its real tokens (as ZImagePipeline does).
    Returns float32 (n_tokens, dim)."""
    if max_len == 512 and prompt in _PRE:
        return _PRE.pop(prompt)
    return _encode([prompt], max_len)[0]


_PRE = {}  # prompt -> embedding from a batched encode_prompts() pass, consumed by prompt_embeds()


def _encode(prompts, max_len=512):
    torch = _torch()
    _load_te()
    tok, te = _M['tok'], _M['te']
    texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True) for p in prompts]
    t = tok(texts, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt')
    mask = t.attention_mask.bool()
    with torch.no_grad():
        h = te(input_ids=t.input_ids.to('cuda'), attention_mask=mask.to('cuda'),
               output_hidden_states=True).hidden_states[-2]
    return [h[k][mask[k].to(h.device)].float().cpu().numpy() for k in range(len(prompts))]


def _emb_cached(prompt):
    fn = prompt_embeds.__wrapped__
    p = os.path.join(tier_dir('global'), fn.__qualname__, key_for(fn, (prompt,), {}) + '.npy')
    return os.path.exists(p) and os.environ.get('VF_NO_CACHE') != '1'


def encode_prompts(prompts):
    """Embed every uncached prompt in batches of ENCODE_BATCH (one cpu-offloaded Qwen3 pass each), store them in the
    memo, then free the encoder (8 GB of RAM)."""
    miss = [p for p in dict.fromkeys(prompts) if not _emb_cached(p)]
    if miss:
        from .guard import allow_encoder
        allow_encoder('z-image text encoder (Qwen3-4B, 8.5 GB)', len(miss), what=miss)
        _GUARD_OK[0] = True
    try:
        for i in range(0, len(miss), ENCODE_BATCH):
            chunk = miss[i:i + ENCODE_BATCH]
            _PRE.update(zip(chunk, _encode(chunk)))
        for p in miss:
            prompt_embeds(p)
    finally:
        _PRE.clear()
        _GUARD_OK[0] = False
        if 'te' in _M:
            _free('te', 'tok')
    return len(miss)


# ------------------------------------------------------------------ DiT + VAE
def _load_pipe():
    _claim()
    if 'te' in _M:
        _free('te', 'tok')
    if 'pipe' in _M:
        _unpark()
        return _M['pipe']
    torch = _torch()
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline, ZImageTransformer2DModel
    d = model_dir()
    t0 = time.time()
    from .video import load_baked, fp8_linears  # one-time bake: the repo ships fp32 shards (24 GB read)
    if PREC in ('', 'fp8store'):
        pass
    if PREC == 'int8':
        from .int8lin import int8_linears
        dit = load_baked(ZImageTransformer2DModel, d, 'transformer', f'zimage-{os.path.basename(d)[:12]}-int8', int8_linears)
    elif PREC == 'bf16':
        dit = ZImageTransformer2DModel.from_pretrained(d, subfolder='transformer', torch_dtype=torch.bfloat16)
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(dit, onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
                               offload_type='block_level', num_blocks_per_group=1, use_stream=False, low_cpu_mem_usage=True)
    elif FP8_COMPUTE:  # fp8 tensor-core matmuls: 18.2 -> 12.4 s per 560x976 still (A/B in docs)
        dit = load_baked(ZImageTransformer2DModel, d, 'transformer', f'zimage-{os.path.basename(d)[:12]}-fp8compute',
                         fp8_linears)
    else:
        dit = load_baked(ZImageTransformer2DModel, d, 'transformer', f'zimage-{os.path.basename(d)[:12]}-fp8store',
                         lambda m: m.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                                              compute_dtype=torch.bfloat16))
    if PREC == 'int8' and os.environ.get('VF_ZIMAGE_COMPILE', '1') == '1':
        # 2026-11-10 A/B: regional torch.compile of the DiT blocks: 13.1 -> 10.9 s per still after a one-off ~17 s compile
        # per process (the gen worker keeps it), and CLOSER to the bf16 model (SSIM 0.979 vs 0.967 eager). A compile
        # error falls back to eager instead of failing the still.
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        try:
            dit.compile_repeated_blocks(fullgraph=False)
        except Exception as e:
            print(f'[t2i] compile skipped: {type(e).__name__}: {e}', file=sys.stderr, flush=True)
    gc.collect()
    vae = AutoencoderKL.from_pretrained(d, subfolder='vae', torch_dtype=torch.bfloat16).to('cuda')
    # 2026-11-03: the full-frame decode after the resident DiT reached 7.9 GB and spilled 1.2 GB into shared memory;
    # tiled decode keeps it small (seams are blended by diffusers' tile overlap)
    vae.enable_tiling()
    if ATTN_BACKEND:
        try:
            dit.set_attention_backend(ATTN_BACKEND)
        except Exception as e:
            print(f'[t2i] attention backend {ATTN_BACKEND!r} unavailable: {e}', file=sys.stderr, flush=True)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(d, subfolder='scheduler')
    pipe = ZImagePipeline(scheduler=sched, vae=vae, text_encoder=None, tokenizer=None, transformer=dit)
    pipe.set_progress_bar_config(disable=True)
    _M.update(pipe=pipe, dit=dit, vae=vae)
    LAST['load_s'] = round(time.time() - t0, 1)
    return pipe


# ------------------------------------------------------------------ the cached call
@memo_file('global', '.png', version=lambda: f'{GEN_VERSION}|{MODEL_ID}|{STEPS}' + ('|fp8compute' if FP8_COMPUTE else '') + (f'|{PREC}' if PREC not in ('', 'fp8store') else ''))
def _t2i(prompt, width=720, height=1280, seed=0, steps=None, out=None):
    """One still from `prompt`. Same prompt + seed + size + code -> same cached file."""
    torch = _torch()
    assert MODEL_ID and STEPS
    W, H = int(width) // 16 * 16, int(height) // 16 * 16
    emb = prompt_embeds(prompt)
    pipe = _load_pipe()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    # hard cap: an overflow raises OOM instead of silently paging into shared system memory (8 GB card)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, T2I_VRAM_CAP / total))
    try:
        lat = pipe(prompt_embeds=[torch.from_numpy(np.array(emb)).to('cuda', torch.bfloat16)], height=H, width=W,
                   num_inference_steps=int(steps or STEPS), guidance_scale=0.0, output_type='latent',
                   generator=torch.Generator('cuda').manual_seed(int(seed))).images
        # decode with the DiT off the card: DiT (~6.2 GB) + VAE decode overflowed the 6.8 GB cap (2026-11-03 OOM)
        dit = pipe.transformer
        if PREC != 'bf16':  # the bf16 reference is group-offloaded
            dit.to('cpu')
        torch.cuda.empty_cache()
        try:
            with torch.no_grad():
                v = pipe.vae
                x = lat.to(v.dtype) / v.config.scaling_factor + v.config.shift_factor
                img = pipe.image_processor.postprocess(v.decode(x, return_dict=False)[0], output_type='pil')[0]
        finally:
            del lat
            if PREC != 'bf16':
                dit.to('cuda')
    finally:
        from .guard import _BASE  # the worker's standing cap (process_cap), never uncapped
        torch.cuda.set_per_process_memory_fraction(_BASE[0])
    img.convert('RGB').save(out, format='PNG')
    LAST.update(gen_s=round(time.time() - t0, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))


_t2i.__wrapped__.__qualname__ = _t2i.__wrapped__.__name__ = 't2i'  # cache dir + keys stay those of the old t2i


def _path(args, kw, prec):
    global PREC
    fn, keep = _t2i.__wrapped__, PREC
    PREC = prec
    try:
        return os.path.join(tier_dir('global'), fn.__qualname__, key_for(fn, args, kw) + '.png')
    finally:
        PREC = keep


def cached_path(job, args=()):
    """The still this call resolves to: an old fp8store still if one is cached, else the current precision's."""
    old = _path(args, job, 'fp8store')
    return old if PREC != 'fp8store' and os.path.exists(old) else _path(args, job, PREC)


def t2i(*args, **kw):
    """One still from `prompt` (cached; same arguments as before). Reuses a still cached under the old fp8store key."""
    p = cached_path(kw, args)
    if os.path.exists(p):
        from ..memo import touch
        return touch(p)
    return _t2i(*args, **kw)


def run_jobs(jobs, emit=None):
    """Encode every unseen prompt first (text encoder loaded once), then load the DiT once and draw."""
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    todo = [j for j in jobs if not os.path.exists(cached_path(j))]
    if todo:
        encode_prompts([j['prompt'] for j in todo])  # batched; frees Qwen3 afterwards
    out = []
    for i, j in enumerate(jobs):
        t0 = time.time()
        hit = os.path.exists(cached_path(j))
        LAST.clear()
        try:
            r = dict(i=i, ok=True, out=t2i(**j), cached=hit, seconds=round(time.time() - t0, 2), **LAST)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 2))
        out.append(r)
        emit(r)
    if todo:
        _free('te', 'tok')
        gc.collect()
        _torch().cuda.empty_cache()
    return out


def status():
    return dict(pipe='pipe' in _M, parked=bool(_M.get('parked')), te='te' in _M)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.image jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
