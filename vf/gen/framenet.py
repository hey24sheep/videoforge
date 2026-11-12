"""Our own temporal transformer for the learned frame checker (vf/framecheck.py). Runs ONLY inside the gen venv; from
the main interpreter use vf.gen.bridge.run_gen('framenet', jobs, persistent=False).

    train: dict(mode='train', data=r'C:\\abs\\set.npz', out=r'C:\\abs\\framenet.pt', folds=5)
           -> dict(ok, out, cv=dict(auc_any, auc_type{...}, score_spearman, score_mae))
    predict: dict(mode='predict', model=r'...framenet.pt', data=r'C:\\abs\\new.npz', out=r'C:\\abs\\pred.npz')

Input per shot: the frame sequence of SigLIP2 features (vf/gen/framefeat.py: 768-d embedding + defect-phrase sims) at
12 fps. Model: linear 783 -> 128, sinusoidal time code, a learned [SHOT] token, 2 transformer encoder layers (4 heads,
dropout 0.2), heads: per-frame "any defect" + one logit per defect type, and the shot's critic score from the [SHOT]
token. ~0.45 M weights; the vision transformer under it (SigLIP2) stays frozen: with a few thousand highly correlated
frames, fine-tuning it would memorise the shots. CPU, ~20-60 s per fit.

set.npz layout (written by vf/framecheck.py): X = (frames, d) float32 concatenated, off = (shots+1) int64 offsets,
y = (frames,) defect weight 0/0.5/1, Y = (frames, T) types 0/1, score = (shots,) (nan = unknown), group = (shots,)
int64 shot-id groups (all takes of one shot id share a fold), types = names.
"""
import json, math, os, sys, time

import numpy as np

NET_VERSION = 'fnet-1'
D, HEADS, LAYERS = 128, 4, 2


def _net(d_in, n_types):
    import torch
    import torch.nn as nn

    class FrameNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(0.1), nn.Linear(d_in, D))
            self.cls = nn.Parameter(torch.zeros(1, 1, D))
            layer = nn.TransformerEncoderLayer(D, HEADS, 2 * D, dropout=0.2, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, LAYERS)
            self.frame = nn.Linear(D, 1 + n_types)
            self.score = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))

        def forward(self, x):   # x: (1, n, d_in), one shot at a time (shots differ in length)
            n = x.shape[1]
            t = torch.arange(n, dtype=torch.float32, device=x.device)[:, None] / 12.0
            f = torch.arange(D // 2, dtype=torch.float32, device=x.device)[None]
            pe = torch.cat([torch.sin(t * 2 ** (f / 8)), torch.cos(t * 2 ** (f / 8))], 1)[None]
            h = self.inp(x) + 0.1 * pe
            h = self.enc(torch.cat([self.cls, h], 1))
            return self.frame(h[:, 1:])[0], self.score(h[:, 0])[0, 0]

    return FrameNet()


def _shots(z):
    off = z['off']
    return [(z['X'][off[i]:off[i + 1]], z['y'][off[i]:off[i + 1]], z['Y'][off[i]:off[i + 1]], float(z['score'][i]))
            for i in range(len(off) - 1)]


def _fit(shots, n_types, epochs=60, seed=0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = _net(shots[0][0].shape[1], n_types)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos = np.concatenate([s[1] >= 0.75 for s in shots]).mean()
    w_pos = float(min(5.0, (1 - pos) / max(pos, 1e-3)))
    for ep in range(epochs):
        net.train()
        for i in rng.permutation(len(shots)):
            X, y, Y, sc = shots[i]
            n = len(X)
            if n > 16 and rng.rand() < 0.5:   # random temporal crop
                a = rng.randint(0, n - 12)
                b = rng.randint(a + 12, n + 1)
                X, y, Y = X[a:b], y[a:b], Y[a:b]
            x = torch.from_numpy(X)[None]
            x = x + 0.02 * torch.randn_like(x)
            logits, s = net(x)
            yt = torch.from_numpy(np.c_[(y >= 0.75).astype(np.float32), Y].astype(np.float32))
            wt = torch.ones_like(yt)
            wt[:, 0] = torch.where(yt[:, 0] > 0, torch.tensor(w_pos), torch.tensor(1.0))
            wt[:, 0] = torch.where(torch.from_numpy((y > 0) & (y < 0.75)), torch.tensor(0.3), wt[:, 0])  # low: weak
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, weight=wt)
            if not math.isnan(sc):
                loss = loss + 0.5 * (s - sc / 10.0) ** 2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
    return net.eval()


def _predict(net, X):
    import torch
    with torch.inference_mode():
        logits, s = net(torch.from_numpy(X)[None])
    p = torch.sigmoid(logits).numpy()
    return p[:, 0], p[:, 1:], float(s) * 10.0


def train(j):
    import torch
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    torch.set_num_threads(int(os.environ.get('VF_FRAMEFEAT_THREADS', '4')))
    z = dict(np.load(j['data'], allow_pickle=False))
    shots = _shots(z)
    types = [str(t) for t in z['types']]
    groups = z['group']
    ug = np.unique(groups)
    rng = np.random.RandomState(0)
    rng.shuffle(ug)
    k = int(j.get('folds', 5))
    yt, pt, st, ps, tt = [], [], [], [], {t: ([], []) for t in types}
    for f in range(k):
        te = set(ug[f::k].tolist())
        net = _fit([s for s, g in zip(shots, groups) if g not in te], len(types))
        for (X, y, Y, sc), g in zip(shots, groups):
            if g not in te:
                continue
            p, P, s = _predict(net, X)
            yt.append(y >= 0.75)
            pt.append(p)
            for i, t in enumerate(types):
                tt[t][0].append(Y[:, i] > 0)
                tt[t][1].append(P[:, i])
            if not math.isnan(sc):
                st.append(sc)
                ps.append(s)
    yt, pt = np.concatenate(yt), np.concatenate(pt)
    cv = dict(auc_any=round(float(roc_auc_score(yt, pt)), 3),
              score_spearman=round(float(spearmanr(st, ps).correlation), 3),
              score_mae=round(float(np.mean(np.abs(np.array(st) - np.array(ps)))), 2),
              auc_type={t: round(float(roc_auc_score(np.concatenate(a), np.concatenate(b))), 3)
                        for t, (a, b) in tt.items() if np.concatenate(a).any() and not np.concatenate(a).all()})
    net = _fit(shots, len(types))
    torch.save(dict(state=net.state_dict(), d_in=shots[0][0].shape[1], types=types, version=NET_VERSION, cv=cv), j['out'])
    return dict(out=j['out'], cv=cv)


def predict(j):
    import torch
    m = torch.load(j['model'], weights_only=True)   # tensors + plain types only
    net = _net(int(m['d_in']), len(m['types']))
    net.load_state_dict(m['state'])
    net.eval()
    z = dict(np.load(j['data'], allow_pickle=False))
    off = z['off']
    p_all, P_all, s_all = [], [], []
    for i in range(len(off) - 1):
        p, P, s = _predict(net, z['X'][off[i]:off[i + 1]])
        p_all.append(p)
        P_all.append(P)
        s_all.append(s)
    np.savez(j['out'], p=np.concatenate(p_all), P=np.concatenate(P_all), score=np.array(s_all), off=off)
    return dict(out=j['out'])


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    out = []
    for i, j in enumerate(jobs):
        t0 = time.time()
        try:
            r = dict(i=i, ok=True, **(train(j) if j['mode'] == 'train' else predict(j)),
                     seconds=round(time.time() - t0, 1))
        except Exception as e:
            r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}')
        emit(r)
        out.append(r)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.framenet jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
