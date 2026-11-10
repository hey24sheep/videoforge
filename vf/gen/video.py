"""AI video stage: image -> clip. Two engines behind one cached call; runs ONLY inside the gen venv (torch + diffusers).
From the main interpreter use vf.gen.bridge.run_gen('video', jobs) (persistent worker, models stay loaded).

    i2v(image, prompt, seed=0, frames=None, width=None, height=None, last_image=None, steps=None, shift=None,
        profile=None) -> mp4 path            (None = the profile's default)
    run_jobs(jobs, emit=print)  /  i2v_batch(jobs)  /  python -m vf.gen.video jobs.json

Profiles (PROFILES below; numbers on the RTX 4060 Laptop, see docs/video_model.md):
  fast     (default) Wan2.2-TI2V-5B-Turbo (I2V step/CFG distill, UniPC 4 steps, shift 5), fp8-stored DiT resident in
           VRAM, tiny taew2_2 decoder. 24 fps: 121 frames = 5.04 s at 832x480. ~50 s/clip warm, ~65 s first clip.
  fast_vae same with the full Wan2.2 VAE decode (+90 s; visually ~identical, kept for A/B).
  fastwan  FastWan2.2-TI2V-5B DMD 3 steps: ~40 s/clip but re-styles the input after frame 1 (T2V-only distill).
  quality  Wan2.2-I2V-A14B GGUF Q4_K_M + lightx2v 4-step Seko-V1, 16 fps, 81 frames = 5.06 s, ~10 min/clip.
           Best identity hold. The only profile with last_image (jobs with last_image default to it).
Result dicts carry `fps`: frames/fps differ per profile, so place the clip with the returned fps.

RAM plan (31 GB box): UMT5-XXL (11 GB bf16) is loaded only on a prompt-embedding miss (embeddings are memoised on the
global tier), encoded in batches (encode_prompts) and freed right after, so it is never resident beside an engine.
park() moves the 5B DiT to CPU RAM (~5 GB) so the renderer / TTS get the whole GPU; the next job moves it back in ~1 s.
"""
import gc, json, os, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

_HERE = os.path.dirname(os.path.abspath(__file__))
_VF_ROOT = os.path.dirname(os.path.dirname(_HERE))
_DEPS = os.path.join(_VF_ROOT, '.pydeps')  # `gguf` lives here (kept out of the shared gen venv)
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.append(_DEPS)
os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))

import numpy as np

from ..memo import memo, memo_file, key_for, tier_dir
from . import fetch_models as fm

TE_ID = 'umt5-xxl-bf16 (Wan2.1/2.2 shared)'
PROFILES = {
    'fast': dict(engine='ti2v5b', repo='turbo', model='Wan2.2-TI2V-5B-Turbo fp8-store unipc4 taew2_2', fps=24,
                 frames=121, width=832, height=480, steps=4, shift=5.0, decoder='tae'),
    'fast_vae': dict(engine='ti2v5b', repo='turbo', model='Wan2.2-TI2V-5B-Turbo fp8-store unipc4 fullvae', fps=24,
                     frames=121, width=832, height=480, steps=4, shift=5.0, decoder='full'),
    'fastwan': dict(engine='ti2v5b', repo='fastwan', model='FastWan2.2-TI2V-5B-FullAttn fp8-store dmd3 taew2_2',
                    fps=24, frames=121, width=832, height=480, steps=3, shift=None, dmd=(1000, 757, 522),
                    decoder='tae'),
    'quality': dict(engine='a14b', model='wan2.2-i2v-a14b-q4km+lightx2v-seko-v1', fps=16, frames=81,
                    width=832, height=480, steps=4, shift=5.0, decoder='full'),
}
DEFAULT_PROFILE = 'fast'
ATTN_BACKEND = '_native_cudnn'   # diffusers attention backend for the 5B DiT ('' = PyTorch default dispatch)
FP8_COMPUTE = os.environ.get('VF_FP8_COMPUTE', '0') == '1'  # fp8 tensor-core matmuls instead of fp8-storage/bf16-compute (see docs/video_model.md)
PARK_GB = 5.0                    # RAM a parked 5B engine takes (DiT 4.8 GB + tiny decoder)
_M = {}      # module-level model cache: 'te', 'tok', 'pipe', 'engine'
LAST = {}    # stats of the last generated clip
TIMES = {}   # stage timers of the current clip


def _torch():
    import torch
    return torch


def _free(*names):
    torch = _torch()
    for n in names:
        _M.pop(n, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync():
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------ text encoder (loaded only on embed miss)
def _claim():
    from . import claim
    claim('video', release, park, size_gb=PARK_GB)


_GUARD_OK = [False]  # set by encode_prompts after its guard check


def _load_te():
    if 'te' not in _M and not _GUARD_OK[0]:  # a single uncached prompt_embeds() call: same guardrail
        from .guard import allow_encoder
        allow_encoder('wan text encoder (UMT5-XXL, 11 GB)', 1)
    _claim()
    if 'te' in _M:
        return
    if (_M.get('engine') or ('',))[0] == 'a14b':
        _free('pipe', 'engine')  # never hold UMT5 and the 14B experts in RAM together
    torch = _torch()
    from transformers import AutoTokenizer, UMT5EncoderModel
    from accelerate import cpu_offload
    t0 = time.time()
    base = fm.base_dir()
    _M['tok'] = AutoTokenizer.from_pretrained(base, subfolder='tokenizer')
    te = UMT5EncoderModel.from_pretrained(base, subfolder='text_encoder', torch_dtype=torch.bfloat16).eval()
    cpu_offload(te, execution_device=torch.device('cuda'))  # layer-by-layer on the GPU
    _M['te'] = te
    TIMES['te_load'] = round(time.time() - t0, 1)


_PRE = {}  # prompt -> embedding computed by a batched encode_prompts() pass, consumed by prompt_embeds()
ENCODE_BATCH = 8


def _encode(prompts, max_len=512):
    """UMT5 last hidden state per prompt, zero-padded past its length exactly like WanPipeline does."""
    torch = _torch()
    _load_te()
    t0 = time.time()
    tok, te = _M['tok'], _M['te']
    t = tok(list(prompts), padding='max_length', max_length=max_len, truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors='pt')
    with torch.no_grad():
        e = te(t.input_ids.to('cuda'), t.attention_mask.to('cuda')).last_hidden_state.float().cpu()
    out = []
    for k in range(len(prompts)):
        x = e[k].clone()
        x[int(t.attention_mask[k].sum()):] = 0
        out.append(x.numpy())
    TIMES['encode'] = round(TIMES.get('encode', 0) + time.time() - t0, 1)
    return out


@memo('global', version=lambda: f'umt5-1|{TE_ID}')
def prompt_embeds(prompt, max_len=512):
    """UMT5 last hidden state, zero-padded past the prompt length exactly like WanPipeline does. (max_len, 4096) f32."""
    if max_len == 512 and prompt in _PRE:
        return _PRE.pop(prompt)
    return _encode([prompt], max_len)[0]


def _emb_cached(prompt):
    fn = prompt_embeds.__wrapped__
    p = os.path.join(tier_dir('global'), fn.__qualname__, key_for(fn, (prompt,), {}) + '.npy')
    return os.path.exists(p) and os.environ.get('VF_NO_CACHE') != '1'


def encode_prompts(prompts):
    """Embed every uncached prompt in batches (one pass of the cpu-offloaded UMT5 per ENCODE_BATCH prompts instead of
    one per prompt: ~2 s each -> ~3 s per 4), store them in the memo, then free the encoder (11 GB of RAM)."""
    miss = [p for p in dict.fromkeys(prompts) if not _emb_cached(p)]
    if miss:
        from .guard import allow_encoder
        allow_encoder('wan text encoder (UMT5-XXL, 11 GB)', len(miss), what=miss)
        _GUARD_OK[0] = True
    try:
        for i in range(0, len(miss), ENCODE_BATCH):
            chunk = miss[i:i + ENCODE_BATCH]
            _PRE.update(zip(chunk, _encode(chunk)))
        for p in miss:
            prompt_embeds(p)  # memo writes the cache entry from _PRE
    finally:
        _PRE.clear()
        _GUARD_OK[0] = False
        if 'te' in _M:
            _free('te', 'tok')
    return len(miss)


# ------------------------------------------------------------------ engines
def _timed(obj, name, key):
    """Wrap obj.name so its (synchronised) wall time accumulates into TIMES[key]."""
    f = getattr(obj, name)

    def w(*a, **k):
        _sync(); t0 = time.time()
        r = f(*a, **k)
        _sync(); TIMES[key] = round(TIMES.get(key, 0) + time.time() - t0, 1)
        return r
    setattr(obj, name, w)


def _load_a14b():
    torch = _torch()
    _free('te', 'tok')
    from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig, UniPCMultistepScheduler,
                           WanImageToVideoPipeline, WanTransformer3DModel)
    from diffusers.hooks import apply_group_offloading
    base = fm.base_dir()
    q = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    tr = {}
    for which, sub in (('high', 'transformer'), ('low', 'transformer_2')):
        tr[which] = WanTransformer3DModel.from_single_file(
            fm.gguf_path(which), quantization_config=q, config=base, subfolder=sub, torch_dtype=torch.bfloat16)
        gc.collect()
    vae = AutoencoderKLWan.from_pretrained(base, subfolder='vae', torch_dtype=torch.float32)
    sched = UniPCMultistepScheduler.from_pretrained(base, subfolder='scheduler')
    with open(os.path.join(base, 'model_index.json')) as f:
        boundary = json.load(f).get('boundary_ratio', 0.9)
    pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=sched,
                                   image_processor=None, image_encoder=None, transformer=tr['high'],
                                   transformer_2=tr['low'], boundary_ratio=boundary)
    pipe.load_lora_weights(fm.lora_path('high'), adapter_name='lx_high')
    pipe.load_lora_weights(fm.lora_path('low'), adapter_name='lx_low', load_into_transformer_2=True)
    for t in (pipe.transformer, pipe.transformer_2):
        apply_group_offloading(t, onload_device=torch.device('cuda'), offload_device=torch.device('cpu'),
                               offload_type='block_level', num_blocks_per_group=1, use_stream=False)
    vae.to('cuda')
    vae.enable_tiling()
    return pipe


def _dmd_scheduler(timesteps):
    """DMD sampler as a diffusers scheduler: each step predicts x0, then re-noises it to the next timestep with fresh
    noise (FastVideo's `dmd_denoising_steps`); the last step returns x0."""
    torch = _torch()
    from diffusers import FlowMatchEulerDiscreteScheduler

    class DMDScheduler(FlowMatchEulerDiscreteScheduler):
        generator = None

        def set_timesteps(self, num_inference_steps=None, device=None, **kw):
            ts = list(timesteps)[:num_inference_steps or len(timesteps)]
            self.timesteps = torch.tensor(ts, dtype=torch.float32, device=device)
            self.sigmas = torch.tensor([t / 1000 for t in ts] + [0.0], dtype=torch.float32, device=device)
            self.num_inference_steps = len(ts)
            self._step_index = None
            self._begin_index = None

        def step(self, model_output, timestep, sample, return_dict=True, **kw):
            i = int((self.timesteps == timestep).nonzero()[0])
            s, s_next = self.sigmas[i], self.sigmas[i + 1]
            x0 = sample.float() - s * model_output.float()
            if s_next > 0:
                noise = torch.randn(x0.shape, generator=self.generator, device='cpu', dtype=torch.float32).to(x0.device)
                x0 = (1 - s_next) * x0 + s_next * noise
            out = x0.to(sample.dtype)
            return (out,) if not return_dict else type('O', (), {'prev_sample': out})()

    return DMDScheduler(num_train_timesteps=1000, shift=1.0)


def _swap_in(vae, name, evict):
    """Run vae.<name> on the GPU, keeping it in CPU RAM otherwise; `evict` (the DiT) leaves the GPU meanwhile only if
    the job is big (full decode). 8 GB can't hold DiT 4.8 GB + VAE 1.4 GB + activations: spilling into shared memory
    made each step 5-15x slower."""
    f = getattr(vae, name)

    def w(*a, **k):
        torch = _torch()
        # decode is always big; encode is big too above ~0.4 MP (the I2V condition is encoded as a whole clip:
        # at 576x1024 it spilled past 8 GB and one clip took 233 s before its first step)
        x0 = a[0] if a else None
        big = name == 'decode' or (x0 is not None and hasattr(x0, 'shape') and x0.shape[-1] * x0.shape[-2] > 420000)
        if big and evict is not None:
            evict.to('cpu')
        vae.to('cuda')
        try:
            a = [x.to('cuda') if hasattr(x, 'to') else x for x in a]
            return f(*a, **k)
        finally:
            vae.to('cpu')
            torch.cuda.empty_cache()
            if big and evict is not None:
                evict.to('cuda')
    setattr(vae, name, w)


def _load_ti2v5b(repo):
    torch = _torch()
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanTransformer3DModel
    d = fm.turbo_dir() if repo == 'turbo' else fm.fast_dir()
    if FP8_COMPUTE:  # fp8 weights + fp8 activations on the tensor cores (per-tensor scales)
        tr = load_baked(WanTransformer3DModel, d, 'transformer', f'wan5b-{repo}-{os.path.basename(d)[:12]}-fp8compute',
                        lambda m: fp8_linears(m.blocks))
    else:
        # fp8 storage, bf16 compute: 10 GB -> 4.8 GB so the whole DiT stays resident in 8 GB VRAM (no offload traffic)
        tr = load_baked(WanTransformer3DModel, d, 'transformer', f'wan5b-{repo}-{os.path.basename(d)[:12]}-fp8store',
                        lambda m: m.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                                             compute_dtype=torch.bfloat16))
    _attn_backend(tr)
    gc.collect()
    vae = AutoencoderKLWan.from_pretrained(fm.fast_dir(), subfolder='vae', torch_dtype=torch.bfloat16)
    vae.enable_tiling()
    sched = (UniPCMultistepScheduler.from_pretrained(d, subfolder='scheduler') if repo == 'turbo'
             else _dmd_scheduler(PROFILES['fastwan']['dmd']))
    pipe = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae, scheduler=sched, image_processor=None,
                                   image_encoder=None, transformer=tr, transformer_2=None, boundary_ratio=None,
                                   expand_timesteps=True)
    _swap_in(vae, 'encode', None)   # 1 frame: cheap, DiT stays put
    _swap_in(vae, 'decode', tr)     # full-VAE decode only on request (tiny decoder is the default)
    return pipe


def _tae():
    if 'tae' not in _M:
        from .taehv import TinyWanDecoder
        _M['tae'] = TinyWanDecoder.load('taew2_2')
    return _M['tae']


def _attn_backend(model):
    """cuDNN SDPA: measured 7.6 vs 8.0-8.2 s/step (97 f, 480x832) over PyTorch's default mem-efficient kernel on
    Windows (no flash-attention build); exact attention, deterministic per seed."""
    if ATTN_BACKEND:
        try:
            model.set_attention_backend(ATTN_BACKEND)
        except Exception as e:
            print(f'[video] attention backend {ATTN_BACKEND!r} unavailable: {e}', file=sys.stderr, flush=True)


def fp8_linears(root, skip=('embed', 'proj_out', 'norm', 'modulation', 'adaLN', 'time', 'final')):
    """Replace every nn.Linear under `root` (dims multiple of 16) by an fp8 tensor-core matmul: weight stored e4m3
    with a per-tensor scale, activation quantised per call (per-tensor amax). Quantises on the GPU one layer at a
    time, so the bf16 copy never has to fit in VRAM. Returns the number of layers replaced."""
    torch = _torch()
    nn = torch.nn
    F8 = torch.float8_e4m3fn

    class FP8Linear(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.in_features, self.out_features = lin.in_features, lin.out_features
            if lin.weight.is_meta:  # skeleton for load_baked: the tensors come from the baked file
                self.register_buffer('w8', torch.empty(lin.weight.shape, dtype=F8, device='meta'))
                self.register_buffer('ws', torch.empty((), dtype=torch.float32, device='meta'))
                self.register_buffer('bias', None if lin.bias is None else
                                     torch.empty(lin.bias.shape, dtype=torch.bfloat16, device='meta'))
                return
            w = lin.weight.data.to('cuda', torch.bfloat16)
            s = (w.abs().amax().float() / 440.0).clamp(min=1e-12)
            self.register_buffer('w8', (w.float() / s).to(F8).contiguous())
            self.register_buffer('ws', s.reshape(()))
            self.register_buffer('bias', None if lin.bias is None else lin.bias.data.to('cuda', torch.bfloat16))

        def forward(self, x):
            shp = x.shape
            x2 = x.reshape(-1, shp[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            s = (x2.abs().amax().float() / 440.0).clamp(min=1e-12)  # 440 < 448: bf16 rounding can't overflow e4m3
            xq = (x2 * (1.0 / s).to(torch.bfloat16)).to(F8)
            y = torch._scaled_mm(xq, self.w8.t(), scale_a=s, scale_b=self.ws, bias=self.bias,
                                 out_dtype=torch.bfloat16)
            return y.reshape(*shp[:-1], self.out_features)

    n = 0
    for name, mod in list(root.named_modules()):
        for cn, ch in list(mod.named_children()):
            full = f'{name}.{cn}' if name else cn
            if (isinstance(ch, nn.Linear) and ch.in_features % 16 == 0 and ch.out_features % 16 == 0
                    and not any(k in full for k in skip)):
                setattr(mod, cn, FP8Linear(ch))
                n += 1
    gc.collect()
    return n


BAKE_DIR = os.path.join(os.environ['HF_HOME'], 'vf_baked')


def _set_tensor(model, name, t, is_buffer):
    torch = _torch()
    mod_name, _, leaf = name.rpartition('.')
    mod = model.get_submodule(mod_name) if mod_name else model
    if is_buffer:
        mod._buffers[leaf] = t
    else:
        mod._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)


def load_baked(cls, src, subfolder, tag, prepare):
    """`cls.from_pretrained(src, subfolder, bf16)` + `prepare(model)` (e.g. fp8 layerwise casting), cached as one
    safetensors of the prepared tensors (exact dtypes, buffers included) and loaded straight to the GPU.
    Bit-identical weights; ~5 s instead of ~20 s, and no fp32/bf16 staging copy in RAM (Z-Image ships fp32 shards:
    loading them left ~10 GB of the worker's RAM committed after the model was freed). Returns the model on cuda."""
    torch = _torch()
    from safetensors import safe_open
    from safetensors.torch import save_file
    path = os.path.join(BAKE_DIR, f'{tag}.safetensors')
    if os.path.exists(path):
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=True):
            m = cls.from_config(cls.load_config(src, subfolder=subfolder))
        prepare(m)
        with safe_open(path, framework='pt', device='cuda') as f:
            bufs = set(json.loads(f.metadata().get('buffers', '[]')))
            for k in f.keys():
                _set_tensor(m, k, f.get_tensor(k), k in bufs)
        left = [n for n, t in list(m.named_parameters()) + list(m.named_buffers()) if t.is_meta]
        if left:
            raise RuntimeError(f'baked {tag}: {len(left)} tensors missing, e.g. {left[:3]}; delete {path}')
        return m.eval()
    m = cls.from_pretrained(src, subfolder=subfolder, torch_dtype=torch.bfloat16)
    prepare(m)
    m.to('cuda')
    try:  # bake once for next time
        os.makedirs(BAKE_DIR, exist_ok=True)
        seen, sd, bufs = set(), {}, []
        for kind, items in (('p', m.named_parameters()), ('b', m.named_buffers())):
            for n, t in items:
                if t.data_ptr() in seen:
                    continue
                seen.add(t.data_ptr())
                sd[n] = t.detach().to('cpu').contiguous()
                if kind == 'b':
                    bufs.append(n)
        tmp = path + f'.{os.getpid()}.tmp'
        save_file(sd, tmp, metadata={'buffers': json.dumps(bufs), 'src': str(src)})
        del sd
        os.replace(tmp, path)
    except Exception as e:
        print(f'[gen] could not bake {tag}: {e}', file=sys.stderr, flush=True)
    gc.collect()
    return m


def park():
    """Weights to CPU RAM (VRAM freed; `_load_pipe` moves them back in ~1 s). The 14B engine (19 GB) and the text
    encoder are freed instead."""
    _free('te', 'tok')
    if (_M.get('engine') or ('',))[0] == 'a14b':
        _drop_pipe()
    elif 'pipe' in _M and not _M.get('parked'):
        _M['pipe'].transformer.to('cpu')
        if 'tae' in _M:
            _M['tae'].to('cpu')
        _M['parked'] = True
    gc.collect()
    _torch().cuda.empty_cache()


def _unpark():
    if _M.get('parked'):
        t0 = time.time()
        _M['pipe'].transformer.to('cuda')
        if 'tae' in _M:
            _M['tae'].to('cuda')
        _M['parked'] = False
        TIMES['unpark'] = round(time.time() - t0, 1)


def release():
    """Give the GPU/RAM back (another generator is about to load)."""
    _drop_pipe()
    _free('te', 'tok', 'tae')


def _drop_pipe():
    old = _M.pop('pipe', None)
    if old is not None:
        # wrapper closures form reference cycles: break them and let gc free the GPU weights. (Moving the weights to
        # the CPU first, as before, left that 5-6 GB copy committed in the worker's RAM after the free.)
        for n in ('transformer', 'transformer_2', 'vae'):
            try:
                if getattr(old, n, None) is not None:
                    vars(getattr(old, n)).pop('encode', None)
                    vars(getattr(old, n)).pop('decode', None)
                    setattr(old, n, None)
            except Exception:
                pass
        del old
    _M.pop('parked', None)
    _free('pipe', 'engine')


def _load_pipe(p):
    _claim()
    key = (p['engine'], p.get('repo'))
    if _M.get('engine') == key:
        _unpark()
        return _M['pipe']
    _drop_pipe()
    t0 = time.time()
    pipe = _load_a14b() if p['engine'] == 'a14b' else _load_ti2v5b(p['repo'])
    _timed(pipe.vae, 'encode', 'vae_encode')
    _timed(pipe.vae, 'decode', 'vae_decode')
    pipe.set_progress_bar_config(disable=True)
    _M.update(pipe=pipe, engine=key)
    gc.collect()
    TIMES['load'] = round(time.time() - t0, 1)
    return pipe


# ------------------------------------------------------------------ io helpers
def _fit(img, w, h):
    """Cover-resize + centre-crop a PIL image to exactly (w, h)."""
    from PIL import Image
    img = img.convert('RGB')
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.LANCZOS)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


def _write_mp4(frames, path, fps):
    import imageio.v2 as iio
    u8 = (np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8)
    with iio.get_writer(path, format='FFMPEG', fps=fps, codec='libx264', quality=None, macro_block_size=1,
                        ffmpeg_params=['-crf', '14', '-preset', 'medium']) as w:  # imageio adds yuv420p
        for f in u8:
            w.append_data(f)


def resolve(job):
    """Fill a job's unset options from its profile (last_image forces the quality engine unless a profile is given)."""
    j = dict(job)
    prof = j.get('profile') or ('quality' if j.get('last_image') else DEFAULT_PROFILE)
    p = PROFILES[prof]
    if j.get('last_image') and p['engine'] != 'a14b':
        raise ValueError(f"profile {prof!r} has no last_image support; use profile='quality'")
    for k in ('frames', 'width', 'height', 'steps', 'shift'):
        if j.get(k) is None:
            j[k] = p[k]
    j['profile'] = prof
    return j, p


# ------------------------------------------------------------------ the cached call
GEN_VERSION = 'i2v-1'  # bump when the generated pixels would change (sampler, model, decode); loaders/speed don't


@memo_file('global', '.mp4', version=lambda: f'{GEN_VERSION}|{PROFILES!r}|{DEFAULT_PROFILE}'
                                             + ('|fp8compute' if FP8_COMPUTE else ''))
def i2v(image, prompt, seed=0, frames=None, width=None, height=None, last_image=None, steps=None, shift=None,
        profile=None, out=None):
    """Animate `image` (path) following `prompt`; see PROFILES for defaults. frames must be 4k+1."""
    torch = _torch()
    from PIL import Image
    j, p = resolve(dict(seed=seed, frames=frames, width=width, height=height, last_image=last_image, steps=steps,
                        shift=shift, profile=profile))
    # cache key = GEN_VERSION + PROFILES + DEFAULT_PROFILE + the arguments (see the decorator)
    t_all = time.time()
    emb = prompt_embeds(prompt)  # memoised; loads UMT5 only for unseen prompts (run_jobs pre-encodes in batches)
    if 'te' in _M:
        _free('te', 'tok')  # never keep the 11 GB encoder beside the engine
    pipe = _load_pipe(p)
    if p.get('dmd'):
        pipe.scheduler.generator = torch.Generator('cpu').manual_seed(int(seed) + 1)
    else:
        from diffusers import UniPCMultistepScheduler
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=float(j['shift']))
    W, H = int(j['width']), int(j['height'])
    first = _fit(Image.open(image), W, H)
    last = _fit(Image.open(last_image), W, H) if last_image else None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    marks = []

    def _step(pp, i, t, kw):  # progress on stderr (stdout is reserved for the JSON result lines)
        _sync()
        torch.cuda.empty_cache()  # keep the caching allocator from growing past physical VRAM (WDDM then pages)
        marks.append(round(time.time() - t0, 1))
        print(f'[i2v] step {i + 1}/{j["steps"]} t={int(t)} at {marks[-1]}s', file=sys.stderr, flush=True)
        return kw

    kw = dict(last_image=last) if last is not None else {}
    if p['engine'] == 'a14b':
        kw['guidance_scale_2'] = 1.0
    res = pipe(image=first, prompt_embeds=torch.from_numpy(np.array(emb))[None].to('cuda', torch.bfloat16),
               height=H, width=W, num_frames=int(j['frames']), num_inference_steps=int(j['steps']),
               guidance_scale=1.0, generator=torch.Generator('cpu').manual_seed(int(seed)),
               output_type='latent' if p['decoder'] == 'tae' else 'np', callback_on_step_end=_step, **kw)
    if p['decoder'] == 'tae':  # tiny decoder on the normalised latents: ~0.5 s instead of ~90 s, PSNR ~25 dB vs full
        _sync(); td = time.time()
        vid = _tae().decode(res.frames)[0].permute(0, 2, 3, 1).float().cpu().numpy()
        _sync(); TIMES['vae_decode'] = round(time.time() - td, 1)
    else:
        vid = res.frames[0]
    t1 = time.time()
    _write_mp4(vid, out, p['fps'])
    enc = TIMES.get('vae_encode', 0)
    steps_s = [round(b - a, 1) for a, b in zip([enc] + marks[:-1], marks)]
    LAST.update(gen_seconds=round(time.time() - t0, 1), steps_s=steps_s, write_s=round(time.time() - t1, 1),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), frames=int(len(vid)), fps=p['fps'],
                profile=j['profile'], total_s=round(time.time() - t_all, 1))


def cached_path(job):
    """Where i2v(**job) lives in the global cache (exists -> the call is free)."""
    fn = i2v.__wrapped__
    return os.path.join(tier_dir('global'), fn.__qualname__, key_for(fn, (), job) + '.mp4')


def _peak_ram_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, 'peak_wset', mi.rss) / 2**30, 2)
    except Exception:
        return None


def run_jobs(jobs, emit=None):
    """Run i2v over a list of kwarg dicts: encode every new prompt first, then load the engine once and generate.
    Cached jobs never touch the GPU. `emit(result)` is called per job (CLI: print a JSON line)."""
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    todo = [j for j in jobs if not os.path.exists(cached_path(j))]
    if todo:
        TIMES.clear()
        encode_prompts([j['prompt'] for j in todo])  # batched; frees UMT5 afterwards (never resident with a DiT)
    results = []
    for i, j in enumerate(jobs):
        t0 = time.time()
        hit = os.path.exists(cached_path(j))
        LAST.clear()
        if not (i == 0 and todo):
            TIMES.clear()
        try:
            fps = resolve(j)[1]['fps']
            path = i2v(**j)
            r = dict(i=i, ok=True, out=path, cached=hit, fps=fps, seconds=round(time.time() - t0, 2),
                     **{k: v for k, v in LAST.items() if k != 'fps'}, times=dict(TIMES), peak_ram_gb=_peak_ram_gb())
        except Exception as e:  # one bad job must not kill the batch
            import traceback
            traceback.print_exc()
            r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 2))
        results.append(r)
        emit(r)
    if todo:
        _free()  # hand activation memory back (weights stay resident for the next request)
    return results


def status():
    """Resident state + memory, for leak checks (server `status` command)."""
    torch = _torch()
    return dict(engine=_M.get('engine'), parked=bool(_M.get('parked')), te='te' in _M, tae='tae' in _M,
                vram_alloc_gb=round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                vram_reserved_gb=round(torch.cuda.memory_reserved() / 2 ** 30, 2))


def i2v_batch(jobs):
    return run_jobs(jobs, emit=lambda r: None)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.video jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
