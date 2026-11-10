"""Image-quality model for blur / softness / quality regressions: VisualQuality-R1-7B (MIT; Qwen2.5-VL-7B base,
NeurIPS 2025) scores crops of a take 1-5 in its answer-only (non-thinking) mode. Runs ONLY inside the gen venv;
from the main interpreter use vf.gen.bridge.run_gen('iqa', jobs, persistent=False).

    job = dict(clip=dict(video=r'C:\\abs\\cut.mp4', t0=14.4, t1=19.0), frames=12)
    run_jobs(jobs, emit=print) -> [dict(i, ok, frames=[{t, face, body, frame}], summary={face, body, frame}, ...)]

Scores are only meaningful RELATIVE to another take of the same shot (new vs base): the model was trained on
everyday photos and will mark night, rain and grain down in absolute terms.
Loading: NF4 quantised on the CPU, decoder layers + vision tower on the GPU, the 1.1 GB embedding and lm_head in
system RAM (vf/gen/director.py's _cpu_ends), process capped at 7.2 GB.
"""
import gc, json, os, re, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
import numpy as np

REPO = 'TianheWu/VisualQuality-R1-7B'
IQA_VERSION = 'vqr1-7b-3'   # 3: face crops only (whole-frame score dropped on an intended fade to black)
# 2:   # 2: expected score from the digit probabilities; face + frame only, 8 frames
PROMPT = ("You are doing the image quality assessment task. Here is the question: "
          "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
          "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality.")
TEMPLATE = "{Question} Please only output the final answer with only one score in <answer> </answer> tags."
_M = {}


def release():
    for k in ('model', 'proc'):
        _M.pop(k, None)
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load():
    if 'model' in _M:
        return _M['proc'], _M['model']
    from . import claim
    claim('iqa', release)
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from .director import _cpu_ends
    try:
        d = snapshot_download(REPO, local_files_only=True)
    except Exception:
        d = snapshot_download(REPO)
    proc = AutoProcessor.from_pretrained(d)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True, llm_int8_skip_modules=['visual', 'lm_head'])
    model = AutoModelForImageTextToText.from_pretrained(d, dtype=torch.bfloat16, quantization_config=q,
                                                        device_map='cpu').eval()
    _cpu_ends(model)
    gc.collect()
    _M.update(proc=proc, model=model)
    return proc, model


def score(img):
    """1-5 quality of one RGB array (answer-only mode)."""
    import torch
    from PIL import Image
    from .director import _fused_attention
    proc, model = _load()
    msg = [dict(role='user', content=[dict(type='image'), dict(type='text', text=TEMPLATE.format(Question=PROMPT))])]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    h, w = img.shape[:2]
    inp = proc(text=[text], images=[Image.fromarray(img)], size={'shortest_edge': 28 * 28, 'longest_edge': h * w},
               return_tensors='pt').to('cuda')
    inp.pop('attention_mask', None)
    with torch.inference_mode(), _fused_attention():
        out = model.generate(**inp, max_new_tokens=24, do_sample=False, output_scores=True, return_dict_in_generate=True)
    gen = out.sequences[0, inp['input_ids'].shape[1]:].tolist()
    ans = proc.batch_decode([gen], skip_special_tokens=True)[0]
    # vqr1-7b-2: the greedy answer moves in coarse steps (1.00, 1.50); the expectation over the digit tokens of the
    # integer and the first decimal is continuous: E[int] + E[tenth] / 10
    tok = proc.tokenizer
    if 'digits' not in _M:
        _M['digits'] = {d: tok.convert_tokens_to_ids(str(d)) for d in range(10)}
    dig = _M['digits']
    pos = [k for k, t in enumerate(gen) if t in dig.values()]
    exp = None
    if pos:
        def e(k, rng):
            pr = torch.softmax(out.scores[k][0].float(), -1)
            ps = torch.stack([pr[dig[d]] for d in rng])
            return float((ps / ps.sum() * torch.tensor(list(rng), dtype=torch.float32, device=ps.device)).sum())
        exp = e(pos[0], range(0, 6))
        if len(pos) > 1 and pos[1] <= pos[0] + 2:
            exp += e(pos[1], range(10)) / 10
    del inp, out
    m = re.search(r'<answer>\s*([0-9.]+)', ans) or re.search(r'([0-9]+(?:\.[0-9]+)?)', ans)
    return (round(exp, 3) if exp is not None else (float(m.group(1)) if m else None)), ans


def check(clip, frames=8, whole=False):
    from .smear import _read, _pieces
    import cv2
    fr = []
    for t, f in _read(clip['video'], clip.get('t0', 0), clip.get('t1'), frames):
        rec = dict(t=t)
        for kind, img in _pieces(f):
            if kind == 'face':          # body crops scored ~0.5 everywhere: no information
                rec[kind] = score(img)[0]
        if not whole:
            fr.append(rec)
            continue
        h, w = f.shape[:2]
        s = 896 / max(h, w)
        rec['frame'] = score(cv2.cvtColor(cv2.resize(f, (round(w * s / 28) * 28, round(h * s / 28) * 28),
                                                     interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))[0]
        fr.append(rec)
    summ = {k: round(float(np.median([x[k] for x in fr if x.get(k) is not None])), 3)
            for k in ('face', 'body', 'frame') if any(x.get(k) is not None for x in fr)}
    return fr, summ


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    from ..gpulock import gpu_lock
    from .guard import process_cap
    import torch
    out = []
    with gpu_lock():
        process_cap()
        for i, j in enumerate(jobs):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            try:
                fr, summ = check(j['clip'], int(j.get('frames', 8)))
                r = dict(i=i, ok=True, frames=fr, summary=summ, version=IQA_VERSION, seconds=round(time.time() - t0, 1),
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
        sys.exit('usage: python -m vf.gen.iqa jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
