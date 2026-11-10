"""Director models: an open video-language model compares two takes of the same shot (old vs new cut) blind, in both
orders, and says which is better made and why. Runs ONLY inside the gen venv; from the main interpreter use
vf.gen.bridge.run_gen('director', jobs, persistent=False).

    job = dict(a=dict(video=r'C:\\abs\\base.mp4', t0=14.4, t1=19.0),     # take A (any clip range)
               b=dict(video=r'C:\\abs\\new.mp4',  t0=14.4, t1=19.0),     # take B
               caption='what the shot should show (its motion prompt)', frames=8, model='q9b')
    run_jobs(jobs, emit=print) -> [dict(i, ok, b={axis: B's share 0..1}, agree={axis: both orders agree}, raw, ...)]
    python -m vf.gen.director jobs.json

Every pair is asked twice with the order swapped and averaged, so a preference for the first/second position cancels;
`agree` says whether the two orders picked the same take (when they don't, the answer is noise for that axis).

Models:
  'q9b'  (default) Qwen3.5-9B (Apache-2.0, Mar 2026) in 4-bit NF4 (bitsandbytes), vision tower bf16; a neutral
         director rubric (DIRECTOR below): realism, cinematography, consistency, overall; answers 1/2/0 + a reason.
  'ur4b' UnifiedReward-2.0-qwen35-4b (MIT) with the authors' pairwise video prompt, fp8-stored layers. Measured
         2026-11-05 on 16 past case shots: the two orders agreed 24/47 times (chance), most shares ~0.5: too weak
         to gate on. Kept for comparison only.
Both run under the process VRAM cap (7.2 GB) with fused attention (no full attention matrix).
"""
import gc, json, os, re, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
import numpy as np

REPOS = {'ur4b': 'CodeGoat24/UnifiedReward-2.0-qwen35-4b', 'q9b': 'Qwen/Qwen3.5-9B'}
DIRECTOR_VERSION = 'director-3'   # bump when answers would change (model, prompt, frames, size)
LONG_EDGE = {'ur4b': 640, 'q9b': 512}
_M = {}
LAST = {}

UR_PAIR = ("You are presented with two generated videos (Video 1 and Video 2) along with a shared text caption. "
           "Your task is to comparatively evaluate the two videos across three specific dimensions:\n\n"
           "- Alignment Score: How well each video matches the caption in terms of content.\n"
           "- Coherence Score: How logically consistent and visually coherent each video is (absence of visual glitches, distorted objects, etc.).\n"
           "- Style Score: How aesthetically appealing each video is, regardless of caption accuracy.\n\n"
           "For each dimension, you must assign a relative score to Video 1 and Video 2, such that:\n"
           "- Each score is a float between 0 and 1 (inclusive).\n"
           "- The scores for Video 1 and Video 2 must sum to exactly 1.0 for each dimension.\n"
           "- The higher the score, the better that video is in the corresponding dimension *relative to the other*.\n\n"
           "This format emphasizes comparative quality rather than absolute evaluation.\n\n"
           "Please provide your evaluation in the format below:\n\n"
           "Alignment Score:\n Video 1: X\n Video 2: Y\n\n"
           "Coherence Score:\n Video 1: X\n Video 2: Y\n\n"
           "Style Score:\n Video 1: X\n Video 2: Y\n\n"
           "Your task is provided as follows:\n"
           "Text Caption: [{caption}]")
UR_AXES = ('alignment', 'coherence', 'style')

# Neutral: never says which take is new, what changed, or what the owner asked for (review rule: no leading prompts).
DIRECTOR = ("You are the director of a neo-noir short film, reviewing two takes of the same shot. The first {n} images "
            "are Take 1 in time order; the next {n} images are Take 2 in time order. The shot should show: {caption}\n\n"
            "Answer each question with the take number (1 or 2), or 0 if the takes are equally good, then a short reason "
            "naming what you see.\n"
            "Q1 Realism: which take looks more like real filmed footage, with nothing that looks pasted in, composited, "
            "melted, smeared or AI-generated?\n"
            "Q2 Cinematography: which take is better framed, composed and lit for a neo-noir film?\n"
            "Q3 Consistency: in which take do faces, bodies, clothes and objects keep a more stable, consistent look "
            "from the first frame to the last?\n"
            "Q4 Overall: which take would you put in the film?\n\n"
            "Answer in exactly this format:\nQ1: <1|2|0> - reason\nQ2: <1|2|0> - reason\nQ3: <1|2|0> - reason\n"
            "Q4: <1|2|0> - reason")
Q_AXES = ('realism', 'cinematography', 'consistency', 'overall')

# Per-take (pointwise) director questions, read from the Yes/No logits like vf/gen/judge.py. Pairwise prompts were
# position-biased on BOTH models (2026-11-05: Qwen3.5-9B picked "Take 1" for realism/consistency/overall and "Take 2"
# for cinematography in both orders); one take at a time has no position to prefer. Neutral wording, one take, no
# history (review rule). good = the answer a good shot gets.
RATE = ('These are {n} frames, in time order, from one shot of a neo-noir short film. The shot should show: {caption}\n'
        'Question: {q}\nAnswer with only Yes or No.')
RATE_QS = [
    dict(key='artificial', good='no', q='Does any part of this shot look pasted in, composited, melted, smeared or '
                                         'AI-generated?'),
    dict(key='cinematic', good='yes', q='Is this shot framed, composed and lit like a shot from a professional '
                                        'neo-noir film?'),
    dict(key='consistent', good='yes', q='Do the faces, bodies, clothes and objects keep the same shape and look from '
                                         'the first frame to the last?'),
    dict(key='matches', good='yes', q='Does the shot show what it should show?'),
    dict(key='horror', good='no', q='Does this shot look more like a horror film than a neo-noir crime film?'),
    dict(key='accept', good='yes', q='Would a film director accept this shot into the final cut as it is?'),
]
CRITIQUE = ('These are {n} frames, in time order, from one shot of a neo-noir short film. The shot should show: '
            '{caption}\nAs the director of this film, give your notes on this shot in at most three short sentences: what '
            'works, and anything that looks wrong or unconvincing (name where it is).')


def release():
    for k in ('model', 'proc', 'which'):
        _M.pop(k, None)
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(which):
    if _M.get('which') == which:
        return _M['proc'], _M['model']
    release()
    from . import claim
    claim('director', release)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, AutoModelForImageTextToText
    try:
        d = snapshot_download(REPOS[which], local_files_only=True)
    except Exception:
        d = snapshot_download(REPOS[which])
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(d)
    if which == 'q9b':
        # 9B bf16 = 19 GB; vocab 248k x 4096: the input embedding and the untied lm_head are 2 GB EACH in bf16.
        # Loading straight to the GPU OOMed at the 7.2 GB cap three ways (embeddings are never quantised; the lm_head
        # passes through the GPU in bf16 to be quantised; accelerate's CPU offload streams it back every call).
        # So: NF4-quantise on the CPU (bitsandbytes 0.50 CPU backend, ~36 s), then move ONLY the decoder layers and
        # the vision tower to the GPU (3.6 GB measured); embedding + lm_head stay in system RAM and run there.
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True, llm_int8_skip_modules=['visual', 'conv1d', 'lm_head'])
        model = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.bfloat16, quantization_config=q,
                                                            device_map='cpu').eval()
        _cpu_ends(model)
    else:                # 4.7B bf16 = 9.4 GB: LLM layers stored fp8, computed bf16 (like vf/gen/judge.py)
        model = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.bfloat16).eval()
        from diffusers.hooks import apply_layerwise_casting
        # conv1d: the linear-attention layers call F.conv1d on the raw weight (torch fallback, no hook) -> keep bf16
        apply_layerwise_casting(model.model.language_model.layers, storage_dtype=torch.float8_e4m3fn,
                                compute_dtype=torch.bfloat16, skip_modules_pattern=('norm', 'conv1d'))
        model.to('cuda')
    gc.collect()
    _M.update(proc=proc, model=model, which=which)
    LAST['load_s'] = round(time.time() - t0, 1)
    return proc, model


def _cpu_ends(model):
    """Keep the embedding and lm_head on the CPU and move everything else to the GPU. The two are wrapped in
    parameter-less modules that hold them OUTSIDE the module tree (so no .to() can move them): ids/hidden states go
    to the CPU, only the small result (prompt embeddings, last-position logits) comes back to the GPU."""
    import torch
    lm = model.model.language_model
    emb, head = lm.embed_tokens, model.lm_head

    class CpuEmbed(torch.nn.Module):
        def __init__(self, e):
            super().__init__()
            self._e = [e]

        @property
        def weight(self):
            return self._e[0].weight

        def forward(self, ids):
            return self._e[0](ids.to('cpu')).to('cuda')

    class CpuHead(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self._h = [h]

        @property
        def weight(self):
            return self._h[0].weight

        def forward(self, x):
            h = self._h[0]
            return h(x.to('cpu', h.weight.dtype)).to('cuda')
    lm.embed_tokens = CpuEmbed(emb)
    model.lm_head = CpuHead(head)
    for name, child in model.model.named_children():
        if name != 'language_model':
            child.to('cuda')
    for name, child in lm.named_children():
        if name != 'embed_tokens':
            child.to('cuda')


class _fused_attention:
    """Keep SDPA off the math kernel, which materialises the full attention matrix (a 938 MB block OOMed at the
    7.2 GB cap). Qwen3.5's text layers call SDPA with grouped KV heads (enable_gqa, head dim 256): this Windows torch
    build has no fused kernel for that, so the KV heads are expanded first (~60 MB at 4k tokens). Scoped: restored on
    exit, nothing else in the process is affected."""
    def __enter__(self):
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend
        self.F, self.orig = F, F.scaled_dot_product_attention
        orig = self.orig

        def sdpa(q, k, v, *a, enable_gqa=False, **kw):
            if enable_gqa and k.shape[1] != q.shape[1]:
                r = q.shape[1] // k.shape[1]
                k, v = k.repeat_interleave(r, dim=1), v.repeat_interleave(r, dim=1)
            return orig(q, k, v, *a, **kw)
        F.scaled_dot_product_attention = sdpa
        self.ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])
        self.ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self.ctx.__exit__(*exc)
        self.F.scaled_dot_product_attention = self.orig
        return False


def _frames(video, t0, t1, n, long_edge):
    """n frames evenly over [t0, t1) of a video (i * N / n, the UnifiedReward sampler)."""
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    i0 = int(round(t0 * fps))
    i1 = int(round(t1 * fps)) if t1 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in [i0 + int(k * (i1 - i0) / n) for k in range(n)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = long_edge / max(h, w)
        W, H = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
        out.append(cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    if len(out) < n:
        raise ValueError(f'{video}: got {len(out)} of {n} frames in {t0}-{t1} s')
    return out


def _parse_ur(txt):
    r = {}
    for ax, name in zip(UR_AXES, ('Alignment', 'Coherence', 'Style')):
        m = re.search(name + r'\s*Score:?\s*Video 1:\s*([0-9.]+)\s*Video 2:\s*([0-9.]+)', txt, re.S | re.I)
        if m:
            v1, v2 = float(m.group(1)), float(m.group(2))
            r[ax] = (v1 / (v1 + v2)) if v1 + v2 > 0 else 0.5
    return r   # Video 1's share


def _parse_q(txt):
    r, why = {}, {}
    for k, ax in enumerate(Q_AXES, 1):
        m = re.search(rf'Q{k}\s*:\s*\**\s*([012])\b\s*[-:\u2013\u2014]?\s*(.*)', txt)
        if m:
            v = int(m.group(1))
            r[ax] = {1: 1.0, 2: 0.0, 0: 0.5}[v]      # Take 1's share
            why[ax] = m.group(2).strip()[:300]
    return r, why


def _ask(which, f1, f2, caption):
    import torch
    from PIL import Image
    proc, model = _load(which)
    text = (UR_PAIR.format(caption=caption) if which == 'ur4b' else DIRECTOR.format(n=len(f1), caption=caption))
    content = [dict(type='image') for _ in f1 + f2] + [dict(type='text', text=text)]
    prompt = proc.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    H, W = f1[0].shape[:2]
    inp = proc(text=[prompt], images=[Image.fromarray(x) for x in f1 + f2],
               size={'shortest_edge': 32 * 32, 'longest_edge': H * W}, return_tensors='pt').to('cuda')
    # batch of one, no padding: an all-ones mask only forces SDPA onto the math kernel
    inp.pop('attention_mask', None)
    with torch.inference_mode(), _fused_attention():
        out = model.generate(**inp, max_new_tokens=110 if which == 'ur4b' else 260, do_sample=False)
    txt = proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
    LAST['tokens'] = int(inp['input_ids'].shape[1])
    del inp, out
    torch.cuda.empty_cache()
    return txt


def compare(a, b, caption, frames=8, model='q9b'):
    """B's share per axis (0..1, 0.5 = equal), averaged over both orders; agree = both orders picked the same side."""
    le = LONG_EDGE[model]
    fa = _frames(a['video'], a.get('t0', 0), a.get('t1'), frames, le)
    fb = _frames(b['video'], b.get('t0', 0), b.get('t1'), frames, le)
    t1, t2 = _ask(model, fa, fb, caption), _ask(model, fb, fa, caption)   # B second, then B first
    if model == 'ur4b':
        (p1, w1), (p2, w2), axes = (_parse_ur(t1), {}), (_parse_ur(t2), {}), UR_AXES
    else:
        (p1, w1), (p2, w2), axes = _parse_q(t1), _parse_q(t2), Q_AXES
    share, agree, why = {}, {}, {}
    for ax in axes:
        o = [x for x in ((1 - p1[ax]) if ax in p1 else None, p2.get(ax)) if x is not None]
        share[ax] = round(float(np.mean(o)), 3) if o else None
        agree[ax] = (len(o) == 2 and ((o[0] > 0.5 and o[1] > 0.5) or (o[0] < 0.5 and o[1] < 0.5) or o[0] == o[1] == 0.5))
        why[ax] = [w for w in (w1.get(ax), w2.get(ax)) if w]
    return share, agree, why, [t1, t2]


_SPLIT = '\x00Q\x00'


def rate(clip, caption, frames=8, critique=True, model='q9b'):
    """One take: p(good answer) per RATE_QS key (the frames + preamble are encoded once; each question runs on a copy
    of that prefix cache), plus optional director notes. -> (scores {key: p_good}, p_yes {key: p}, notes)"""
    import copy, torch
    from PIL import Image
    proc, mdl = _load(model)
    tok = proc.tokenizer
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', 'yes', ' Yes')})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ('No', 'no', ' No')})
    fr = _frames(clip['video'], clip.get('t0', 0), clip.get('t1'), frames, LONG_EDGE[model])
    imgs = [Image.fromarray(x) for x in fr]
    H, W = fr[0].shape[:2]

    def templ(text):
        content = [dict(type='image') for _ in fr] + [dict(type='text', text=text)]
        return proc.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    pre, suf = templ(RATE.format(n=len(fr), caption=caption, q=_SPLIT)).split(_SPLIT)
    inp = proc(text=[pre], images=imgs, size={'shortest_edge': 32 * 32, 'longest_edge': H * W},
               return_tensors='pt').to('cuda')
    inp.pop('attention_mask', None)
    p_yes = {}
    with torch.inference_mode(), _fused_attention():
        r = mdl(**inp, use_cache=True, logits_to_keep=1)
        cache = r.past_key_values
        del r
        for qd in RATE_QS:
            ids = tok(qd['q'] + suf, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
            c = copy.deepcopy(cache)          # hybrid (linear-attention) cache can't be cropped back: copy it
            lg = mdl(input_ids=ids, past_key_values=c, use_cache=True, logits_to_keep=1).logits[0, -1].float()
            pr = torch.softmax(lg, -1)
            y, n = pr[yes].sum().item(), pr[no].sum().item()
            p_yes[qd['key']] = round(y / max(y + n, 1e-9), 3)
            del c, lg, pr
    LAST['tokens'] = int(inp['input_ids'].shape[1])
    del cache, inp
    notes = None
    if critique:
        inp = proc(text=[templ(CRITIQUE.format(n=len(fr), caption=caption))], images=imgs,
                   size={'shortest_edge': 32 * 32, 'longest_edge': H * W}, return_tensors='pt').to('cuda')
        inp.pop('attention_mask', None)
        with torch.inference_mode(), _fused_attention():
            out = mdl.generate(**inp, max_new_tokens=120, do_sample=False)
        notes = proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        del inp, out
    torch.cuda.empty_cache()
    scores = {qd['key']: (p_yes[qd['key']] if qd['good'] == 'yes' else round(1 - p_yes[qd['key']], 3)) for qd in RATE_QS}
    return scores, p_yes, notes


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    from ..gpulock import gpu_lock
    from .guard import process_cap
    out = []
    with gpu_lock():
        process_cap()
        import torch
        for i, j in enumerate(jobs):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            try:
                m = j.get('model', 'q9b')
                info = dict(model=m, version=DIRECTOR_VERSION, seconds=None, tokens=None, load_s=None)
                if j.get('mode') == 'rate':
                    scores, p_yes, notes = rate(j['clip'], j.get('caption', ''), int(j.get('frames', 8)),
                                                j.get('critique', True), m)
                    r = dict(i=i, ok=True, scores=scores, p_yes=p_yes, notes=notes, **info)
                else:
                    share, agree, why, raw = compare(j['a'], j['b'], j.get('caption', ''), int(j.get('frames', 8)), m)
                    r = dict(i=i, ok=all(v is not None for v in share.values()), b=share, agree=agree, why=why,
                             raw=raw, **info)
                    if not r['ok']:
                        r['error'] = 'unparsed answer'
                r.update(seconds=round(time.time() - t0, 1), tokens=LAST.get('tokens'), load_s=LAST.get('load_s'),
                         peak_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            except Exception as e:
                import traceback
                traceback.print_exc()
                r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 1))
            out.append(r)
            emit(r)
        release()
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.director jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
