"""Semantic clip checker: a small open VLM looks at sampled frames of an AI clip and answers yes/no questions.
Runs ONLY inside the gen venv; from the main interpreter use vf.gen.bridge.run_gen('judge', jobs).

    job = dict(clip=r'C:\\abs\\shot.mp4',
               questions=[dict(q='Is a lit cigarette visible in his hand or mouth?', expect='yes'), ...],
               frames=8, describe=True, checklist=False)
    run_jobs(jobs, emit=print) -> [dict(i, ok, fail, answers=[{q, answer, p_yes, expect, pass}], description,
                                        seconds, cached)]
    python -m vf.gen.judge jobs.json

- Every question is asked separately and scored from the first-token logits: p_yes = P(Yes) / (P(Yes) + P(No)),
  answer = 'yes' when p_yes >= 0.5. `expect` may be omitted (then the answer is informational, never a fail).
- `checklist=True` appends every DEFAULT_CHECKS question (expect 'no'); `checklist=['appear']` only those keys.
  Leave out 'appear' when the shot is MEANT to bring someone/something into frame.
- `describe=True` adds one paragraph of what happens (story check: compare it with the shot's intent).
- `mode='images'` (default) shows the frames as separate timestamped pictures; 'video' feeds them as a video
  (Qwen3-VL merges frame pairs: half the tokens, but it missed 2 of 9 "appears" defects that 'images' caught).
- The frames + preamble are encoded once per clip; each question only adds ~30 tokens on a cached prefix.

What it can and cannot see (measured 2026-11-02 on 16 hand-labelled test clips, 4B, images, 8 frames):
  reliable  : things that APPEAR/VANISH (hands, a person walking in, a block smear), props/states you ask about
              by name (rain indoors, numbers on glass, receiver at the ear), what happens (describe).
  blind     : warping/morphing (hat changes shape), "jumps", generic "glitch/artifact" questions, physics
              ("does it float") -> ~0 true positives. Those belong to vf/clipqc.py (flow/pixel checks).

Model: Qwen3-VL-4B-Instruct (Apache-2.0, default) with fp8-stored / bf16-computed LLM layers: 4.9 GB resident,
~5.5 GB peak. `model='2b'` = Qwen3-VL-2B-Instruct (Apache-2.0) bf16, ~4.3 GB peak, 2x faster, poorly calibrated
(p_yes near 0.5 on the hard cases). Shares the GPU through vf.gen.claim like every generator.
Never swap in a research/non-commercial licensed VLM (e.g. Qwen2.5-VL-3B): the output is monetised.
"""
import gc, json, os, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
import numpy as np

from ..memo import memo, key_for, tier_dir

REPOS = {'2b': 'Qwen/Qwen3-VL-2B-Instruct', '4b': 'Qwen/Qwen3-VL-4B-Instruct'}
JUDGE_VERSION = 'judge-2'  # bump when answers would change (prompt wording, frame sampling, scoring)
LONG_EDGE = 640            # frames are resized so the long edge is this (multiple of 32): 352x640 -> 220 tokens
_M = {}
LAST = {}

# Generic per-clip checks (expect 'no'), kept to the ones the model answers reliably. Measured (4B, images):
#   appear: 9/11 defective clips flagged, 0/5 clean clips flagged (misses: the two hat-morph clips).
#   text  : flags the one clip with lettering (numbers on fogged glass), 0 false positives.
# Rejected (0 true positives on both models): "warped/melted/deformed", "blocky smears/glitches", "jumps between
# frames", "visual glitch or AI artifact", "moves by itself/floats", "number of people changes".
DEFAULT_CHECKS = {
    'appear': dict(q='Is there a person, hand or object in the last frames that is not present in the first frame?',
                   expect='no'),
    'text': dict(q='Does any readable text or lettering appear in the frames?', expect='no'),
}

ASK = ('These are {n} frames sampled evenly from one {dur:.1f}-second AI-generated video clip, in order.\n'
       'Question: {q}\nAnswer with only Yes or No.')
DESCRIBE = ('These are {n} frames sampled evenly from one {dur:.1f}-second video clip, in order. In one short '
            'paragraph, describe what is visible and what happens from start to end: who or what is in the shot, '
            'every movement, anything that appears or disappears, and any visual glitch.')


def _torch():
    import torch
    return torch


def release():
    # drop the references (no .to('cpu'): that would copy GBs of weights into RAM just to free them)
    for k in ('model', 'proc', 'which', 'yes', 'no'):
        _M.pop(k, None)
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(which='4b'):
    from . import claim
    claim('judge', release)
    if _M.get('which') == which:
        return _M['proc'], _M['model']
    release()
    torch = _torch()
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    repo = REPOS[which]
    try:
        d = snapshot_download(repo, local_files_only=True)
    except Exception:
        d = snapshot_download(repo)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(d)
    model = Qwen3VLForConditionalGeneration.from_pretrained(d, dtype=torch.bfloat16).eval()
    if which != '2b':  # 4B bf16 is 8.9 GB: store the LLM layers in fp8, compute in bf16
        from diffusers.hooks import apply_layerwise_casting
        apply_layerwise_casting(model.model.language_model.layers, storage_dtype=torch.float8_e4m3fn,
                                compute_dtype=torch.bfloat16, skip_modules_pattern=('norm',))
    model.to('cuda')
    gc.collect()
    _M.update(proc=proc, model=model, which=which)
    tok = proc.tokenizer
    _M['yes'] = [tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', 'yes', ' Yes')]
    _M['no'] = [tok.encode(w, add_special_tokens=False)[0] for w in ('No', 'no', ' No')]
    LAST['load_s'] = round(time.time() - t0, 1)
    return proc, model


def _sample(clip, n):
    """n frames evenly over the clip (first and last included), resized to LONG_EDGE. -> (frames RGB, idx, fps, N)"""
    import cv2
    cap = cv2.VideoCapture(clip)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if N <= 0:
        raise FileNotFoundError(clip)
    idx = sorted(set(np.linspace(0, N - 1, int(n)).round().astype(int).tolist()))
    want, got, i = set(idx), {}, 0
    while len(got) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = f
        i += 1
    cap.release()
    idx = [k for k in idx if k in got]
    h, w = got[idx[0]].shape[:2]
    s = LONG_EDGE / max(h, w)
    W, H = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
    frames = [cv2.cvtColor(cv2.resize(got[k], (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for k in idx]
    return frames, idx, fps, N


def _inputs(frames, idx, fps, N, text, mode, prompt_only=False, prompt=None):
    """Processor inputs for frames + text. prompt_only -> just the chat-templated string; prompt -> use that
    (already templated) string instead of templating `text`."""
    from PIL import Image
    proc = _M['proc']
    if mode == 'video':
        from transformers.video_utils import VideoMetadata
        content = [dict(type='video'), dict(type='text', text=text)]
        if prompt is None:
            prompt = proc.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                              add_generation_prompt=True)
        if prompt_only:
            return prompt
        H, W = frames[0].shape[:2]
        meta = VideoMetadata(total_num_frames=N, fps=fps, width=W, height=H, duration=N / fps, frames_indices=idx)
        px = H * W * len(frames)
        return proc(text=[prompt], videos=[np.stack(frames)], video_metadata=[meta], do_sample_frames=False,
                    size={'shortest_edge': 32 * 32, 'longest_edge': px}, return_tensors='pt')
    content = []
    for f, k in zip(frames, idx):
        content += [dict(type='text', text=f'Frame at {k / fps:.1f} s:'), dict(type='image')]
    content.append(dict(type='text', text=text))
    if prompt is None:
        prompt = proc.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                          add_generation_prompt=True)
    if prompt_only:
        return prompt
    H, W = frames[0].shape[:2]
    return proc(text=[prompt], images=[Image.fromarray(f) for f in frames],
                size={'shortest_edge': 32 * 32, 'longest_edge': H * W}, return_tensors='pt')


_SPLIT = '\x00Q\x00'


def _attn():
    """Keep SDPA off the math kernel (GQA + mask sends it there: 4B prefix peak 6.4 -> 5.45 GB, ~30% faster).
    Scoped, so other generators in the same worker are unaffected."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])


def _ask_all(fr, idx, fps, N, qs, mode):
    """p_yes per question. The frames + preamble are encoded ONCE (vision tower + KV cache of the prefix); each
    question only runs its ~30 suffix tokens on top of the cached prefix, then the cache is cropped back."""
    torch = _torch()
    model, tok = _M['model'], _M['proc'].tokenizer
    text = ASK.format(n=len(fr), dur=N / fps, q=_SPLIT)
    full = _inputs(fr, idx, fps, N, text, mode, prompt_only=True)
    pre, suf = full.split(_SPLIT)
    inp = _inputs(fr, idx, fps, N, None, mode, prompt=pre).to('cuda')
    L = int(inp['input_ids'].shape[1])
    out = []
    with torch.inference_mode(), _attn():
        r = model(**inp, use_cache=True, logits_to_keep=1)  # sets model.model.rope_deltas for the continuation
        cache = r.past_key_values
        del r
        for q in qs:
            ids = tok(q + suf, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
            lg = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1].float()
            p = torch.softmax(lg, -1)
            y, n = p[_M['yes']].sum().item(), p[_M['no']].sum().item()
            out.append(y / max(y + n, 1e-9))
            cache.crop(-int(ids.shape[1]))  # drop this question's tokens: back to the shared prefix
    del cache, inp
    return out, L


def _describe(inp, max_new=150):
    torch = _torch()
    with torch.inference_mode(), _attn():
        out = _M['model'].generate(**inp.to('cuda'), max_new_tokens=max_new, do_sample=False,
                                   repetition_penalty=1.1)  # greedy 2B otherwise loops ("He looks at the camera.")
    return _M['proc'].batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()


@memo('global', version=lambda: f'{JUDGE_VERSION}|{LONG_EDGE}')
def judge_clip(clip, questions=(), frames=8, describe=True, mode='images', model='4b'):
    """-> dict(answers=[{q, answer, p_yes, expect, pass}], description, fail, n_tokens)"""
    torch = _torch()
    assert JUDGE_VERSION and DEFAULT_CHECKS
    fr, idx, fps, N = _sample(clip, frames)
    _load(model)
    dur = N / fps
    qds = [dict(q=qd) if isinstance(qd, str) else dict(qd) for qd in questions]
    ps, ntok = _ask_all(fr, idx, fps, N, [qd['q'] for qd in qds], mode) if qds else ([], 0)
    answers = []
    for qd, p in zip(qds, ps):
        a = 'yes' if p >= 0.5 else 'no'
        exp = (qd.get('expect') or '').strip().lower() or None
        answers.append(dict(q=qd['q'], answer=a, p_yes=round(p, 3), expect=exp, **{'pass': exp is None or a == exp}))
    desc = None
    if describe:
        inp = _inputs(fr, idx, fps, N, DESCRIBE.format(n=len(fr), dur=dur), mode)
        ntok = int(inp['input_ids'].shape[1])
        desc = _describe(inp)
        del inp
    torch.cuda.empty_cache()
    return dict(answers=answers, description=desc, fail=any(not a['pass'] for a in answers), n_tokens=ntok)


def _args(job):
    qs = [dict(q=q) if isinstance(q, str) else dict(q) for q in (job.get('questions') or [])]
    ck = job.get('checklist')
    if ck:
        qs += [dict(DEFAULT_CHECKS[k]) for k in (DEFAULT_CHECKS if ck is True else ck)]
    a = dict(clip=os.path.abspath(job['clip']), questions=qs)
    for k in ('frames', 'describe', 'mode', 'model'):  # omit unset optionals (the cache keys on passed args)
        if k in job:
            a[k] = job[k]
    return a


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    out = []
    for i, j in enumerate(jobs):
        t0 = time.time()
        LAST.clear()
        try:
            a = _args(j)
            fn = judge_clip.__wrapped__
            hit = os.path.exists(os.path.join(tier_dir('global'), fn.__qualname__, key_for(fn, (), a) + '.pkl'))
            res = judge_clip(**a)
            r = dict(i=i, ok=True, cached=hit, seconds=round(time.time() - t0, 2), **res, **LAST)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 2))
        out.append(r)
        emit(r)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.judge jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
