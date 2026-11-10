"""FLUX.2 Klein 4B (Apache-2.0) keyframe refine / edit, gen venv. Usage: python -m vf.gen.klein jobs.json
jobs = [{src, out, prompt, refs?: [image paths], seed?, width?, height?}]; skips jobs whose `out` exists.

Edit mode: `src` is the composition to keep (an existing keyframe); `refs` are extra reference images, e.g. a face
crop of the character from the series' canonical keyframe, so the redraw keeps the same person (measured
2026-11-02: without a reference Klein redrew the detective younger, with a darker beard).
8 GB card: transformer stored fp8 (layerwise casting) resident, Qwen3-4B text encoder leaf-offloaded (it is only
used for the prompt). 4 steps, guidance 1.0 (distilled). ~5-10 s per 576x1024 frame after a ~16 s load.
"""
import json, os, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

THREADS = int(os.environ.get('K6_THREADS', 4))
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, str(THREADS))
import torch
from PIL import Image

torch.set_num_threads(THREADS)
M = os.path.join(_AI, 'models', 'hf', 'flux2-klein-4b')
STYLE = ('Keep the exact same shot, framing, composition, pose, clothing, props, text placement and lighting. '
         'Make it a sharp photorealistic 35mm neo-noir film still: natural skin texture, detailed eyes, clean edges, '
         'fine rain, no painterly smoothing, do not add any text.')


EMB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cache', 'klein_emb')


def _emb_path(text):
    """Prompt embeddings on disk (2026-11-10): a cached prompt never loads the Qwen3-4B text encoder again."""
    import hashlib
    return os.path.join(EMB, hashlib.sha1(('klein-qwen3|512|9,18,27|' + text).encode()).hexdigest()[:24] + '.pt')


# 2026-11-10 A/B: W8A8 int8 is ~10 % faster than fp8-store and closer to the model's bf16 output (SSIM 0.997 vs 0.988).
PREC = os.environ.get('VF_KLEIN_PREC', 'int8')  # int8 (default) | fp8store | bf16 (reference, offloaded)


def _prep(tr, cuda):
    if PREC == 'int8':
        from .int8lin import int8_linears
        int8_linears(tr)
        tr.to(cuda)
    elif PREC == 'bf16':
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(tr, onload_device=cuda, offload_device=torch.device('cpu'), offload_type='block_level',
                               num_blocks_per_group=1, use_stream=False, low_cpu_mem_usage=True)
    else:
        tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        tr.to(cuda)


def load(text_encoder=True):
    import diffusers
    from diffusers.hooks import apply_group_offloading
    cuda = torch.device('cuda')
    if not text_encoder:   # every prompt embedding is cached: transformer + VAE only
        pipe = diffusers.Flux2KleinPipeline.from_pretrained(M, torch_dtype=torch.bfloat16, text_encoder=None, tokenizer=None)
        _prep(pipe.transformer, cuda)
        pipe.vae.to(cuda)
        type(pipe)._execution_device = property(lambda self: cuda)
        return pipe
    from .guard import allow_encoder
    allow_encoder('klein text encoder (Qwen3-4B)', 1)  # only for prompts missing from cache/klein_emb
    pipe = diffusers.Flux2KleinPipeline.from_pretrained(M, torch_dtype=torch.bfloat16)
    _prep(pipe.transformer, cuda)
    pipe.vae.to(cuda)
    apply_group_offloading(pipe.text_encoder, onload_device=cuda, offload_device=torch.device('cpu'),
                           offload_type='leaf_level', use_stream=False, low_cpu_mem_usage=True)  # no pinned 'shared' memory
    type(pipe)._execution_device = property(lambda self: cuda)
    return pipe


def main(path):
    jobs = [j for j in json.load(open(path)) if not os.path.exists(j['out'])]
    if not jobs:
        print('nothing to do', flush=True)
        return
    t = time.time()
    from .guard import process_cap
    process_cap()
    def text_of(j):
        who = (' The person must keep exactly the same face and identity as the person in the reference image.'
               if j.get('refs') else '')
        return f"{j['prompt']}. {j.get('style', STYLE)}{who}"
    miss = [x for x in dict.fromkeys(text_of(j) for j in jobs) if not os.path.exists(_emb_path(x))]
    pipe = load(text_encoder=bool(miss))
    if miss:
        os.makedirs(EMB, exist_ok=True)
        with torch.no_grad():
            for x in miss:
                e, _ = pipe.encode_prompt(prompt=x, device=torch.device('cuda'))
                torch.save(e.detach().to('cpu'), _emb_path(x) + '.part')
                os.replace(_emb_path(x) + '.part', _emb_path(x))
    print(f'loaded in {time.time() - t:.0f}s; {len(jobs)} frames; {len(miss)} new prompt embedding(s)', flush=True)
    for j in jobs:
        t = time.time()
        W, H = int(j.get('width', 576)), int(j.get('height', 1024))
        imgs = [Image.open(j['src']).convert('RGB').resize((W, H), Image.LANCZOS)]
        imgs += [Image.open(r).convert('RGB') for r in j.get('refs', [])]
        # `style:` per job replaces STYLE, e.g. when the edit itself changes a prop ("keep the same props" fights it)
        pe = torch.load(_emb_path(text_of(j))).to('cuda', torch.bfloat16)
        img = pipe(prompt_embeds=pe, image=imgs, height=H, width=W, guidance_scale=1.0,
                   num_inference_steps=4, generator=torch.Generator('cpu').manual_seed(int(j.get('seed', 7)))).images[0]
        os.makedirs(os.path.dirname(j['out']), exist_ok=True)
        img.save(j['out'] + '.part.png')
        os.replace(j['out'] + '.part.png', j['out'])
        print(f"DONE klein {os.path.basename(j['out'])} {time.time() - t:.1f}s", flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
