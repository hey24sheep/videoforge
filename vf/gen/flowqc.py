"""Motion-compensated face flicker for the review harness (vf/qa.py). Runs ONLY inside the gen venv; from the main
interpreter use vf.gen.bridge.run_gen('flowqc', jobs, persistent=False).

    job = dict(video=r'C:\\abs\\cut.mp4', i0=864, track=[[cx, cy, w, h], ...])   # one row per frame of the shot
    run_jobs(jobs, emit=print) -> [dict(i, ok, face=[...], mouth=[...], seconds)]
    python -m vf.gen.flowqc jobs.json

For each frame t of the shot, frames t-1 and t are cut with frame t's (smoothed) face box: 1.6x the face, resized
to 256x256, so real motion stays motion. RAFT gives the flow t->t-1; frame t-1 is warped onto t and the L (CIE Lab)
difference is what motion cannot explain: flicker, boiling texture, a mouth that pops. Per frame it returns the
99th percentile of that residual over the face (central 60 % of the crop) and over the mouth zone (rows 58-75 %,
columns 35-65 %, a fixed place in the face box: mouth landmarks are unreliable on profiles).

Why (2026-11-05): the old jitter number (2nd temporal difference) grows with motion speed and face sharpness; it
failed the owner-approved pier3 (a sharp profile moving 2x more than the soft base) while the face had LESS
unexplained change than the base. Model: torchvision RAFT-large C_T_SKHT_V2 (BSD-3, 20 MB, ships with torchvision),
0.7 GB VRAM, ~10 s per 250-frame shot.
"""
import gc, json, os, sys, time

import numpy as np

FLOWQC_VERSION = 'raft-l-1'   # bump when the numbers would change (model, crop, zones, iterations)
S = 256
BATCH = 8
ITERS = 12
_M = {}


def release():
    _M.pop('net', None)
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _net():
    if 'net' not in _M:
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        _M['net'] = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).cuda().eval()
    return _M['net']


def _warp(img, flo):
    import torch, torch.nn.functional as F
    _, _, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing='ij')
    g = torch.stack(((xx + flo[:, 0]) / (w - 1) * 2 - 1, (yy + flo[:, 1]) / (h - 1) * 2 - 1), -1)
    return F.grid_sample(img, g, mode='bilinear', padding_mode='border', align_corners=True)


def _pairs(video, i0, track):
    import cv2
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i0))
    prev, out = None, []
    for j in range(len(track)):
        ok, f = cap.read()
        if not ok:
            break
        if prev is not None:
            cx, cy, w, h = track[j][:4]
            s = 1.6 * max(w, h)
            M = np.float32([[S / s, 0, S / 2 - cx * S / s], [0, S / s, S / 2 - cy * S / s]])
            out.append(tuple(cv2.warpAffine(x, M, (S, S), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
                             for x in (prev, f)))
        prev = f
    cap.release()
    return out


def measure(video, i0, track):
    import cv2, torch
    net = _net()
    rgb = lambda xs: torch.tensor(np.array([cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in xs]),
                                  dtype=torch.float32).permute(0, 3, 1, 2).cuda() / 127.5 - 1
    c0, c1 = int(S * 0.2), int(S * 0.8)
    m0, m1, n0, n1 = int(S * .58), int(S * .75), int(S * .35), int(S * .65)
    face, mouth = [], []
    P = _pairs(video, i0, track)
    with torch.inference_mode():
        for k in range(0, len(P), BATCH):
            B = P[k:k + BATCH]
            a, b = rgb([x for x, _ in B]), rgb([y for _, y in B])
            fb = net(b, a, num_flow_updates=ITERS)[-1]          # each pixel of t -> where it was in t-1
            aw = ((_warp(a, fb) + 1) * 127.5).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy()
            for i in range(len(B)):
                lb = cv2.cvtColor(B[i][1].astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0]
                la = cv2.cvtColor(np.ascontiguousarray(aw[i][..., ::-1]).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0]
                r = np.abs(lb - la) * 2.55                      # 0-255 scale, like the old jitter numbers
                face.append(round(float(np.percentile(r[c0:c1, c0:c1], 99)), 2))
                mouth.append(round(float(np.percentile(r[m0:m1, n0:n1], 99)), 2))
    return face, mouth


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    from ..gpulock import gpu_lock
    from .guard import process_cap
    out = []
    with gpu_lock():
        process_cap(2.0)   # measured peak 0.7 GB
        for i, j in enumerate(jobs):
            t0 = time.time()
            try:
                face, mouth = measure(os.path.abspath(j['video']), j['i0'], j['track'])
                if len(face) < 3:
                    raise ValueError(f'only {len(face)} frame pairs')
                r = dict(i=i, ok=True, face=face, mouth=mouth, version=FLOWQC_VERSION, seconds=round(time.time() - t0, 2))
            except Exception as e:
                import traceback
                traceback.print_exc()
                r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 2))
            out.append(r)
            emit(r)
        release()
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.flowqc jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
