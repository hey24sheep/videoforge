"""Frame features for the learned frame checker (vf/framecheck.py). Runs ONLY inside the gen venv; from the main
interpreter use vf.gen.bridge.run_gen('framefeat', jobs, persistent=False).

    job = dict(grid=r'C:\\abs\\shot.jpg', tile=[216, 384], n=38, out=r'C:\\abs\\feat.npz', texts=[...])
    run_jobs(jobs, emit=print) -> [dict(i, ok, out, n, seconds)]

A grid is a contact sheet of numbered tiles (10 per row, the review format of vf/framecheck.py). Each tile goes
through SigLIP2-base (google/siglip2-base-patch16-224, Apache-2.0, ~0.4 GB) on the CPU: the image embedding (768,
L2-normalised) plus its similarity to each text in `texts` (zero-shot defect phrases). CPU on purpose: the checker
runs beside a build that holds the GPU. 4 threads, ~0.08 s per tile.
"""
import gc, json, os, sys, time

import numpy as np

REPO = 'google/siglip2-base-patch16-224'
FEAT_VERSION = 'siglip2-b16-224-1'   # bump when the numbers would change (model, crop, preprocessing)
COLS = 10
_M = {}


def release():
    _M.clear()
    gc.collect()


def _model():
    if 'm' not in _M:
        import torch
        from transformers import AutoModel, AutoProcessor
        torch.set_num_threads(int(os.environ.get('VF_FRAMEFEAT_THREADS', '4')))
        _M['m'] = AutoModel.from_pretrained(REPO, dtype=torch.float32).eval()
        _M['p'] = AutoProcessor.from_pretrained(REPO)
    return _M['m'], _M['p']


def tiles(grid, tile, n):
    import cv2
    g = cv2.imread(grid)
    if g is None:
        raise FileNotFoundError(grid)
    w, h = tile
    out = []
    for i in range(n):
        r, c = divmod(i, COLS)
        t = g[r * h:(r + 1) * h, c * w:(c + 1) * w]
        if t.shape[0] != h or t.shape[1] != w:
            break
        t = t.copy()
        t[:34, :70] = t[34:68, :70]   # hide the burned-in tile number (top left) so it is not a feature
        out.append(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
    return out


def _vec(o):
    """transformers 5 returns an output object (pooler_output = the projected embedding); 4.x a tensor."""
    return getattr(o, 'pooler_output', o)


def embed(images, texts, batch=16):
    import torch
    m, p = _model()
    embs = []
    with torch.inference_mode():
        for i in range(0, len(images), batch):
            x = p(images=images[i:i + batch], return_tensors='pt')
            e = _vec(m.get_image_features(**x))
            embs.append(torch.nn.functional.normalize(e, dim=-1).numpy())
        img = np.concatenate(embs) if embs else np.zeros((0, 768), np.float32)
        if texts:
            key = tuple(texts)
            if _M.get('tkey') != key:
                t = p(text=list(texts), return_tensors='pt', padding='max_length', max_length=64)
                _M['temb'] = torch.nn.functional.normalize(_vec(m.get_text_features(**t)), dim=-1).numpy()
                _M['tkey'] = key
            sims = img @ _M['temb'].T
        else:
            sims = np.zeros((len(img), 0), np.float32)
    return img.astype(np.float32), sims.astype(np.float32)


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    out = []
    for i, j in enumerate(jobs):
        t0 = time.time()
        try:
            ims = tiles(j['grid'], j.get('tile', [216, 384]), int(j['n']))
            img, sims = embed(ims, j.get('texts') or [])
            os.makedirs(os.path.dirname(os.path.abspath(j['out'])), exist_ok=True)
            np.savez_compressed(j['out'], img=img, sims=sims, version=FEAT_VERSION)
            r = dict(i=i, ok=True, out=j['out'], n=len(img), seconds=round(time.time() - t0, 2))
        except Exception as e:   # one bad grid must not stop the others
            r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}')
        emit(r)
        out.append(r)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.framefeat jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
