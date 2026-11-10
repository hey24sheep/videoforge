"""Monocular depth for 2.5D camera parallax. Runs ONLY inside the gen venv; from the main interpreter use
vf.gen.bridge.run_gen('depth', jobs).

    job = dict(image=r'C:\\abs\\key.png')                       -> one depth map
    job = dict(clip=r'C:\\abs\\shot.mp4', frames=[0, 60, 120])  -> one depth map per frame index (default [0])
    run_jobs(jobs, emit=print) -> [dict(i, ok, paths=[png...], cached, seconds, ms_per_frame)]
    python -m vf.gen.depth jobs.json

Output: 16-bit grayscale PNG at the SOURCE resolution, near = bright (65535), far = dark, normalised 0..1.
All frames of one job share one normalisation (robust 0.5/99.5 percentiles over the whole set), so depth does not
flicker between the frames of a clip. Read with cv2.imread(p, cv2.IMREAD_UNCHANGED) / 65535.

Model: Depth-Anything-V2-Small (Apache-2.0, 25M params, ~100 MB). Its output is relative inverse depth (disparity),
so it is good for layering/parallax, not metric distance. Base/Large are CC-BY-NC: never switch to them.
"""
import gc, json, os, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
import numpy as np

from ..memo import memo_file, key_for, tier_dir

REPO = 'depth-anything/Depth-Anything-V2-Small-hf'
DEPTH_VERSION = 'dav2s-1'  # bump when the written depth would change (model, normalisation, resize)
BATCH = 4
_M = {}
LAST = {}


def _torch():
    import torch
    return torch


def release():
    _M.pop('model', None)  # drop references only (moving weights to the CPU first would just fill RAM)
    _M.pop('proc', None)
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load():
    from . import claim
    claim('depth', release)
    if 'model' in _M:
        return _M['proc'], _M['model']
    torch = _torch()
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    try:
        d = snapshot_download(REPO, local_files_only=True)
    except Exception:
        d = snapshot_download(REPO)
    t0 = time.time()
    _M['proc'] = AutoImageProcessor.from_pretrained(d)
    _M['model'] = AutoModelForDepthEstimation.from_pretrained(d, dtype=torch.float16).to('cuda').eval()
    LAST['load_s'] = round(time.time() - t0, 2)
    return _M['proc'], _M['model']


def _read_frames(src, frames):
    """RGB uint8 frames. An image -> [img]; a clip -> the requested indices (clamped to the clip length)."""
    import cv2
    if frames is None:
        im = cv2.imread(src, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(src)
        return [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)]
    cap = cv2.VideoCapture(src)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        raise FileNotFoundError(src)
    want = [min(max(int(f), 0), n - 1) for f in frames]
    got, i = {}, 0
    need = set(want)
    while need - set(got) and i <= max(want):  # sequential decode: seeking in h264 is slow and inexact
        ok, f = cap.read()
        if not ok:
            break
        if i in need:
            got[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    last = got[max(got)] if got else None
    return [got.get(w, last) for w in want]


@memo_file('global', '/', version=lambda: f'{DEPTH_VERSION}|{REPO}')
def depth_maps(src, frames=None, out=None):
    """Write d00000.png ... (one per requested frame, in order) into the cache folder `out`."""
    import cv2
    torch = _torch()
    imgs = _read_frames(src, frames)
    proc, model = _load()
    H, W = imgs[0].shape[:2]
    torch.cuda.synchronize()
    t0 = time.time()
    preds = []
    with torch.inference_mode():
        for b in range(0, len(imgs), BATCH):
            x = proc(images=imgs[b:b + BATCH], return_tensors='pt')['pixel_values'].to('cuda', torch.float16)
            d = model(pixel_values=x).predicted_depth  # (B, h, w) relative disparity, bigger = nearer
            d = torch.nn.functional.interpolate(d[:, None].float(), size=(H, W), mode='bicubic', align_corners=False)
            preds.append(d[:, 0])
        d = torch.cat(preds)
        lo, hi = torch.quantile(d[:, ::4, ::4].flatten().float(), torch.tensor([0.005, 0.995], device='cuda'))
        d = ((d - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
        d = (d * 65535.0 + 0.5).to(torch.int32).cpu().numpy().astype(np.uint16)
    torch.cuda.synchronize()
    LAST['ms_per_frame'] = round((time.time() - t0) * 1000 / len(imgs), 1)
    for k, m in enumerate(d):
        cv2.imwrite(os.path.join(out, f'd{k:05d}.png'), m)


def _args(job):
    src = job.get('image') or job.get('clip')
    a = dict(src=os.path.abspath(src))
    if job.get('clip'):
        a['frames'] = [int(f) for f in (job.get('frames') or [0])]
    return a


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    out = []
    for i, j in enumerate(jobs):
        t0 = time.time()
        LAST.clear()
        try:
            a = _args(j)
            hit = os.path.isdir(os.path.join(tier_dir('global'), depth_maps.__wrapped__.__qualname__,
                                             key_for(depth_maps.__wrapped__, (), a)))
            d = depth_maps(**a)
            n = len(a.get('frames') or [0])
            r = dict(i=i, ok=True, paths=[os.path.join(d, f'd{k:05d}.png') for k in range(n)], cached=hit,
                     seconds=round(time.time() - t0, 3), **LAST)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 3))
        out.append(r)
        emit(r)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.depth jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
