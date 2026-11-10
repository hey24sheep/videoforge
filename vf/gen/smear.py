"""Smear / melt / distortion check: Qwen3.5-9B looks at full-resolution pieces of a take (a 3x3 tile grid and the
face crop) and answers two yes/no questions per piece; the probabilities are read from the Yes/No logits.
Runs ONLY inside the gen venv; from the main interpreter use vf.gen.bridge.run_gen('smear', jobs, persistent=False).

    job = dict(clip=dict(video=r'C:\\abs\\cut.mp4', t0=14.4, t1=19.0), frames=12, tiles=False)
    run_jobs(jobs, emit=print) -> [dict(i, ok, frames=[{t, face: {smear, distort}, tiles: [{smear, distort} x9]}],
                                        summary={face_smear, face_distort, tile_smear, tile_distort, worst}, ...)]
    python -m vf.gen.smear jobs.json

Why tiles: the earlier whole-frame judge (Qwen3-VL-4B, 8 small frames) never fired on "warped/melted" (0 true
positives, vf/gen/judge.py). A 1080x1920 frame shrunk to 640 px hides a smeared eye; a 360x640 tile at native
resolution does not. Model loading (NF4 on the CPU, embedding/lm_head in RAM, 4.2 GB peak) is vf/gen/director.py's.
"""
import copy, json, os, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
import numpy as np

SMEAR_VERSION = 'smear-3'   # bump when answers would change (model, questions, crops, frames)
# smear-1 asked "is any part smeared ... a blurry blob": on noir footage the worst tiles were intended optics (the
# out-of-focus foreground hat, rain haze, lamp glare), so approved and rejected takes scored alike. smear-2 tells it
# to ignore intended camera effects and only judge in-focus people and things.
QS = dict(
    smear='Ignore intended camera effects: out-of-focus blur, motion blur, rain, fog, glare and film grain. Is any '
          'in-focus face, hand, body or piece of clothing in this image melted, smeared or smudged into a shapeless '
          'blob?',
    distort='Does any face, hand, body, piece of clothing or object in this image look distorted, malformed or '
            'unnaturally shaped?',
    sharp='Is this face in sharp focus, with fine detail such as skin texture, eyelashes and single strands of hair '
          'clearly visible?',
)
# smear-3 (owner 2026-11-06 "tune it first"): the 3x3 tiles never separated good from bad takes (intended optics),
# so they are off by default; a face-anchored BODY crop checks clothes/body; 12 frames instead of 6 (brief smears);
# a face-sharpness question for blur regressions (p43 "blurry and smearing again"), compared new vs base.
KIND_QS = dict(face=('smear', 'distort', 'sharp'), body=('smear', 'distort'), tile=('smear', 'distort'))
ASK = 'This is a crop from one frame of a film shot.\nQuestion: {q}\nAnswer with only Yes or No.'
_SPLIT = '\x00Q\x00'
LAST = {}


def _read(video, t0, t1, n):
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    i0 = int(round(t0 * fps))
    i1 = int(round(t1 * fps)) if t1 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for k in range(n):      # evenly inside the shot, away from the cut frames
        i = i0 + int((k + 0.5) * (i1 - i0) / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            out.append((round(i / fps - t0, 2), f))
    cap.release()
    return out


def _pieces(f, tiles=False):
    """The largest face (1.6x box) and a body crop anchored on it (3x the face wide, from the chin down 4.5 faces),
    plus optional 3x3 tiles, at native resolution, as RGB arrays sized to multiples of 32."""
    import cv2
    from ..identity import _det
    H, W = f.shape[:2]
    pcs = []
    for r in range(3 if tiles else 0):
        for c in range(3):
            pcs.append(('tile', f[r * H // 3:(r + 1) * H // 3, c * W // 3:(c + 1) * W // 3]))
    s = 0.5
    sm = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    _, fa = _det(sm.shape[1], sm.shape[0]).detect(sm)
    if fa is not None and len(fa):
        a = fa[np.argmax(fa[:, 2] * fa[:, 3])][:4] / s
        cx, cy, side = a[0] + a[2] / 2, a[1] + a[3] / 2, 1.6 * max(a[2], a[3])
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        crop = f[y0:int(min(H, cy + side / 2)), x0:int(min(W, cx + side / 2))]
        if crop.size and min(crop.shape[:2]) >= 48:
            pcs.append(('face', crop))
        fw, fh = a[2], a[3]
        bx0, bx1 = int(max(0, cx - 1.5 * fw)), int(min(W, cx + 1.5 * fw))
        by0, by1 = int(min(H - 1, a[1] + fh * 0.9)), int(min(H, a[1] + fh * 5.4))
        body = f[by0:by1, bx0:bx1]
        if body.size and min(body.shape[:2]) >= 64:
            pcs.append(('body', body))
    out = []
    for kind, p in pcs:
        h, w = p.shape[:2]
        sc = min(1.0, 640 / max(h, w)) if kind != 'face' else 448 / max(h, w)
        nw, nh = max(32, round(w * sc / 32) * 32), max(32, round(h * sc / 32) * 32)
        out.append((kind, cv2.cvtColor(cv2.resize(p, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC),
                                       cv2.COLOR_BGR2RGB)))
    return out


def _ask(img, keys):
    """p_yes for every QS question on one image (image + preamble encoded once, each question on a cache copy)."""
    import torch
    from PIL import Image
    from .director import _load, _fused_attention
    proc, mdl = _load('q9b')
    tok = proc.tokenizer
    if 'yn' not in LAST:
        LAST['yn'] = (sorted({tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', 'yes', ' Yes')}),
                      sorted({tok.encode(w, add_special_tokens=False)[0] for w in ('No', 'no', ' No')}))
    yes, no = LAST['yn']
    content = [dict(type='image'), dict(type='text', text=ASK.format(q=_SPLIT))]
    pre, suf = proc.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False).split(_SPLIT)
    h, w = img.shape[:2]
    inp = proc(text=[pre], images=[Image.fromarray(img)], size={'shortest_edge': 32 * 32, 'longest_edge': h * w},
               return_tensors='pt').to('cuda')
    inp.pop('attention_mask', None)
    res = {}
    with torch.inference_mode(), _fused_attention():
        r = mdl(**inp, use_cache=True, logits_to_keep=1)
        cache = r.past_key_values
        del r
        for k in keys:
            q = QS[k]
            ids = tok(q + suf, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
            c = copy.deepcopy(cache)
            pr = torch.softmax(mdl(input_ids=ids, past_key_values=c, use_cache=True, logits_to_keep=1).logits[0, -1].float(), -1)
            y, n = pr[yes].sum().item(), pr[no].sum().item()
            res[k] = round(y / max(y + n, 1e-9), 3)
            del c, pr
    del cache, inp
    return res


def check(clip, frames=12, tiles=False):
    fr = []
    for t, f in _read(clip['video'], clip.get('t0', 0), clip.get('t1'), frames):
        rec = dict(t=t, tiles=[], face=None, body=None)
        for kind, img in _pieces(f, tiles):
            a = _ask(img, KIND_QS[kind])
            if kind == 'tile':
                rec['tiles'].append(a)
            else:
                rec[kind] = a
        fr.append(rec)
    summ = {}
    for part in ('face', 'body'):
        for k in KIND_QS[part]:
            v = [x[part][k] for x in fr if x[part]]
            if v:   # smear/distort: the worst frame; sharp: the typical frame
                summ[f'{part}_{k}'] = round(float(np.median(v) if k == 'sharp' else max(v)), 3)
    for k in ('smear', 'distort'):
        tv = [(x['tiles'][j][k], x['t'], j) for x in fr for j in range(len(x['tiles']))]
        if tv:
            summ[f'tile_{k}'] = round(float(np.percentile([v for v, _, _ in tv], 90)), 3)
            summ[f'tile_{k}_max'] = max(tv)[0]
        if tv:
            v, t, j = max(tv)
            summ[f'worst_tile_{k}'] = dict(p=v, t=t, row=j // 3, col=j % 3)
    return fr, summ


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    from ..gpulock import gpu_lock
    from .guard import process_cap
    from .director import release
    import torch
    out = []
    with gpu_lock():
        process_cap()
        for i, j in enumerate(jobs):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            try:
                fr, summ = check(j['clip'], int(j.get('frames', 12)), bool(j.get('tiles', False)))
                r = dict(i=i, ok=True, frames=fr, summary=summ, version=SMEAR_VERSION, seconds=round(time.time() - t0, 1),
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
        sys.exit('usage: python -m vf.gen.smear jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
