"""Learned frame checker: a small model trained on the critics' every-frame reviews flags defect frames in a new
build, scores each shot, and writes a fix plan ranked by what fixed the same defect before.

    vf framecheck check <episode.yaml> [--video V]     grids + features + prediction + fix plan (CPU only)
    vf framecheck learn <grid_dir> [--labels F]        add a reviewed round (grids + labels.json) to the training set
    vf framecheck train                                retrain; prints leave-one-shot-out accuracy
    vf framecheck grids <episode.yaml> [--video V] [--out DIR]   only the review grids (+ intent.txt)
    vf framecheck fixes record <episode.yaml> <shot> <action,...> [--labels before.json]
    vf framecheck fixes resolve <after labels.json>    score pending fixes against the next review
    vf framecheck fixes table                          what worked, per defect type

How it fits the review loop (docs/workflow.md): critics read every frame on the grids and write labels.json (one
object per shot: score + defects with type, severity, frame ranges). `learn` stores the round, `train` fits the
model, and `check` runs on every later build before the critics. It runs on the CPU, so it can run while a build
holds the GPU. It does not replace the critics: it is a cheap first pass that gets better with every reviewed round.

Model: each 12 fps tile -> SigLIP2-base image embedding + similarity to a few defect phrases (vf/gen/framefeat.py),
plus how much the frame differs from its neighbours, the shot's first frame and the shot's mean. On top: a PCA +
logistic-regression head per defect type and one for "any defect", and a ridge regression from per-shot aggregates
to the critics' 0-10 score. Validation is grouped by shot id, so a shot never trains and tests at once.

Fix policy: every applied fix is logged (defect types before, actions taken). When the next review comes in, a
defect type that is gone (or lower in severity) counts as a win for each action taken on that shot. Actions are ranked per
defect type by the posterior mean of a Beta(1 + wins, 1 + losses), with the built-in order as the prior. The ranking follows
what worked on this pipeline.

Data lives in VF_FRAMECHECK_DIR (default cache/framecheck: local, never committed): datasets.json, feats/, model.pkl,
fixes.jsonl. A series' own reviews stay on this machine.
"""
import glob, hashlib, json, os, pickle, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.environ.get('VF_FRAMECHECK_DIR', os.path.join(ROOT, 'cache', 'framecheck'))
FPS, TILE, COLS = 12, (216, 384), 10
SEV = {'low': 0.5, 'medium': 1.0, 'high': 1.0}
TRUST = 0.75   # a defect type is reported only if its held-out AUC reaches this
SCORE_OK = 0.5   # the predicted shot score is shown only above this held-out Spearman
MIN_POS = 25   # a defect type gets its own head only with this many positive frames from 3+ shots

TYPES = ['identity_drift', 'face_morph', 'hair_change', 'body_morph', 'extra_person', 'object_morph',
         'object_vanish', 'object_jump', 'text_garbled', 'text_cut', 'wrong_light', 'intent_breach', 'dark_frames',
         'hard_cut', 'flicker', 'no_motion', 'mouth_silent', 'glow_eyes', 'other']

# zero-shot phrases fed to SigLIP2 (features, not verdicts: the trained head decides what they mean)
TEXTS = ['a distorted melted face', 'a deformed hand with extra fingers', 'garbled unreadable text',
         'text cut off at the edge of the picture', 'a nearly black underexposed image', 'a blurry smeared image',
         'glowing eyes', 'two people', 'a crowd of people', 'an empty scene with no people', 'a close-up of a face',
         'a clean sharp film still', 'a broken glitchy AI image', 'a telephone', 'a hand']

# fix actions: what the agent changes in the episode generator. The order per defect is the prior.
ACTIONS = {
    'reseed': 'new `seed` (insert shots: keyframe AND clip change)',
    'reseed_clip': 'new clip/talk seed with the same keyframe (talk `seed`, or pin `keyframe_seed`)',
    'reseed_keyframe': 'new `keyframe_seed` only; look at the keyframe with `vf run --until keyframes` first',
    'prompt_state': 'state the correct attribute in the keyframe prompt (hair colour, light, time of day, object state)',
    'add_judge': 'add a yes/no `judge` question for this defect so QC re-rolls catch it',
    'restage': 'reframe: closer/wider, move or remove the prop, hands out of frame, nobody in a wide street',
    'simplify_motion': 'motion prompt asks for less: "nothing moves", one small action, no hand-prop action',
    'drop_lock': 'remove `lock=` (platelock): it made props jump, double and warp',
    'ink': 'bake the text into the prop with `gen.ink` instead of letting the model draw it',
    'shorten': 'cut the shot to its clean frames (`mindur` / trim), or move the cut',
    'keep_disclose': 'keep it and tell the owner (approved asset, or no fix is worth the cost)',
}
PRIOR = {
    'identity_drift': ['reseed_keyframe', 'prompt_state', 'restage', 'add_judge'],
    'face_morph': ['restage', 'reseed_clip', 'simplify_motion'],
    'hair_change': ['prompt_state', 'add_judge', 'reseed'],
    'body_morph': ['restage', 'simplify_motion', 'reseed_clip'],
    'extra_person': ['restage', 'reseed_keyframe', 'add_judge'],   # a person in the keyframe survives clip re-rolls
    'object_morph': ['simplify_motion', 'drop_lock', 'reseed_clip'],
    'object_vanish': ['drop_lock', 'simplify_motion', 'reseed_clip'],
    'object_jump': ['drop_lock', 'simplify_motion', 'reseed_clip'],
    'text_garbled': ['ink', 'restage', 'reseed_keyframe'],
    'text_cut': ['restage', 'ink'],
    'wrong_light': ['prompt_state', 'reseed_keyframe'],
    'intent_breach': ['prompt_state', 'restage', 'add_judge', 'simplify_motion'],
    'dark_frames': ['shorten', 'keep_disclose'],
    'hard_cut': ['shorten', 'reseed_clip'],
    'flicker': ['drop_lock', 'reseed_clip', 'simplify_motion'],
    'no_motion': ['reseed_clip', 'simplify_motion'],
    'mouth_silent': ['reseed_clip', 'shorten'],
    'glow_eyes': ['reseed_clip', 'prompt_state'],
    'other': ['reseed_clip', 'restage'],
}


def _p(*a):
    return os.path.join(STORE, *a)


def _sha(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


# ---------- grids ----------
def _video(yaml_path, video):
    if video:
        return os.path.abspath(video)
    root = os.path.dirname(os.path.abspath(yaml_path))
    c = [p for p in glob.glob(os.path.join(root, 'out', '*.mp4')) if 'phone' not in p and '_FULL' not in p]
    if not c:
        raise SystemExit('no rendered episode video under out/: pass --video')
    return max(c, key=os.path.getmtime)


def grids(yaml_path, video=None, out_dir=None, log=print):
    """Per-shot contact sheets at 12 fps (216x384 tiles, numbered, 10 per row) + intent.txt: the critics' review
    format and the checker's input. Returns {shot id: (grid path, tile count)}."""
    import cv2, yaml
    from .checks import shots_of
    video = _video(yaml_path, video)
    root = os.path.dirname(os.path.abspath(yaml_path))
    out_dir = out_dir or os.path.join(root, 'out', 'framecheck')
    os.makedirs(out_dir, exist_ok=True)
    ep = yaml.safe_load(open(yaml_path, encoding='utf-8')) or {}
    spec = {s.get('id'): s for s in ep.get('shots', [])}
    cap = cv2.VideoCapture(video)
    res, intent = {}, []
    for s in shots_of(yaml_path):
        g = (spec.get(s['id']) or {}).get('gen') or {}
        intent.append(f"- {s['id']} ({'talk' if s['talk'] else 'insert'}, {s['end'] - s['start']:.2f} s): "
                      f"{g.get('keyframe') or '(image or finished clip)'} | MOTION: {g.get('motion', '')}")
        tiles = []
        for i in range(max(1, int(round((s['end'] - s['start']) * FPS)))):
            cap.set(cv2.CAP_PROP_POS_MSEC, (s['start'] + (i + 0.5) / FPS) * 1000)
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.resize(f, TILE, interpolation=cv2.INTER_AREA)
            cv2.putText(f, str(i + 1), (6, 24), 0, 0.7, (0, 255, 255), 2)
            tiles.append(f)
        if not tiles:
            continue
        n = len(tiles)
        while len(tiles) % COLS:
            tiles.append(np.zeros_like(tiles[0]))
        p = os.path.join(out_dir, f"{s['id']}.jpg")
        cv2.imwrite(p, np.vstack([np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        res[s['id']] = (p, n)
    cap.release()
    open(os.path.join(out_dir, 'intent.txt'), 'w', encoding='utf-8').write('\n'.join(intent) + '\n')
    log(f'[framecheck] {len(res)} grids -> {out_dir}')
    return res


def _count_tiles(grid):
    """Tiles in a grid: the ones carrying the yellow tile number (padding tiles at the end have none)."""
    import cv2
    g = cv2.imread(grid).astype(np.int16)
    w, h = TILE
    rows, n = g.shape[0] // h, 0
    for i in range(rows * COLS):
        r, c = divmod(i, COLS)
        z = g[r * h:r * h + 30, c * w:c * w + 60]
        yellow = (z[..., 1] > 150) & (z[..., 2] > 150) & (z[..., 0] < z[..., 1] - 60)
        if yellow.sum() >= 6:
            n = i + 1
    return n


# ---------- features ----------
def features(grid_list, log=print):
    """[(grid, n)] -> list of {'img': (n,768), 'sims': (n,k)}; cached on the grid file's bytes."""
    from .gen.bridge import run_gen
    os.makedirs(_p('feats'), exist_ok=True)
    tkey = hashlib.sha1('|'.join(TEXTS).encode()).hexdigest()[:8]
    outs, jobs = [], []
    for g, n in grid_list:
        o = _p('feats', f'{_sha(g)}_{n}_{tkey}.npz')
        outs.append(o)
        if not os.path.exists(o):
            jobs.append(dict(grid=os.path.abspath(g), tile=list(TILE), n=int(n), out=o, texts=TEXTS))
    if jobs:
        t = time.time()
        log(f'[framecheck] features for {len(jobs)} grid(s) on the CPU ...')
        run_gen('framefeat', jobs, persistent=False)
        log(f'[framecheck] features done in {time.time() - t:.0f}s')
    return [dict(np.load(o)) for o in outs]


def _frame_x(f):
    """Per-frame feature rows for one shot: embedding, defect-phrase sims, temporal change."""
    E, S = f['img'], f['sims']
    n = len(E)
    prev = np.r_[1.0, (E[1:] * E[:-1]).sum(1)] if n > 1 else np.ones(1)
    nxt = np.r_[(E[1:] * E[:-1]).sum(1), 1.0] if n > 1 else np.ones(1)
    first = E @ E[0]
    m = E.mean(0)
    m /= np.linalg.norm(m) + 1e-8
    mean = E @ m
    t = np.linspace(0, 1, n)
    temporal = np.stack([prev, nxt, np.minimum(prev, nxt), first, mean, t, np.full(n, min(n, 120) / 120.0)], 1)
    return np.hstack([E, S, temporal]).astype(np.float32)


def _shot_x(p_any, P, f):
    """Per-shot aggregate row: frame defect probabilities + temporal stability."""
    E = f['img']
    cs = (E[1:] * E[:-1]).sum(1) if len(E) > 1 else np.ones(1)
    q = np.sort(p_any)[::-1]
    return np.r_[p_any.mean(), q[0], q[:max(1, len(q) // 10)].mean(), (p_any > 0.5).mean(),
                 P.max(0) if P.size else [], cs.min(), (1 - cs).mean(), (E @ E[0]).min(),
                 len(E) / 120.0].astype(np.float32)


# ---------- dataset ----------
def _datasets():
    p = _p('datasets.json')
    return json.load(open(p)) if os.path.exists(p) else []


def learn(grid_dir, labels=None, log=print):
    """Register a reviewed round: every <shot>.jpg in grid_dir with its entry in labels.json."""
    labels = labels or os.path.join(grid_dir, 'labels.json')
    L = json.load(open(labels, encoding='utf-8'))
    ds = [d for d in _datasets() if os.path.abspath(d['labels']) != os.path.abspath(labels)]
    shots = []
    for e in L:
        g = os.path.join(grid_dir, f"{e['shot']}.jpg")
        if os.path.exists(g):
            shots.append(e['shot'])
    ds.append(dict(grid_dir=os.path.abspath(grid_dir), labels=os.path.abspath(labels), shots=shots,
                   added=time.strftime('%Y-%m-%d %H:%M')))
    os.makedirs(STORE, exist_ok=True)
    json.dump(ds, open(_p('datasets.json'), 'w'), indent=1)
    log(f'[framecheck] dataset {labels}: {len(shots)} shots ({len(ds)} rounds stored)')
    return ds


def _frame_labels(e, n):
    y = np.zeros(n, np.float32)
    Y = np.zeros((n, len(TYPES)), np.float32)
    for d in e.get('defects', []):
        k = TYPES.index(d['type']) if d.get('type') in TYPES else TYPES.index('other')
        w = SEV.get(d.get('severity'), 1.0)
        for a, b in d.get('frames') or [[1, n]]:
            a, b = max(1, int(a)), min(n, int(b))
            y[a - 1:b] = np.maximum(y[a - 1:b], w)
            Y[a - 1:b, k] = 1
    return y, Y


def _load_all(log=print):
    rows = []
    for d in _datasets():
        L = {e['shot']: e for e in json.load(open(d['labels'], encoding='utf-8'))}
        gl = []
        for s in d['shots']:
            g = os.path.join(d['grid_dir'], f'{s}.jpg')
            gl.append((g, _count_tiles(g)))
        for (g, n), f in zip(gl, features(gl, log)):
            s = os.path.splitext(os.path.basename(g))[0]
            y, Y = _frame_labels(L[s], len(f['img']))
            rows.append(dict(shot=s, round=d['labels'], f=f, y=y, Y=Y, score=float(L[s].get('score', np.nan))))
    return rows


# ---------- model ----------
def _head():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), PCA(48, random_state=0),
                         LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000))


class _Const:
    """A head for a type with no positives (or no negatives) in a training fold."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.tile([1 - self.p, self.p], (len(X), 1))


def _fit(rows, types):
    from sklearn.linear_model import Ridge
    X = np.vstack([_frame_x(r['f']) for r in rows])
    yb = np.concatenate([r['y'] for r in rows]) >= 0.75   # medium/high defect frames
    any_h = _head().fit(X, yb)
    heads = {}
    Yall = np.vstack([r['Y'] for r in rows])
    for k in types:
        yk = Yall[:, TYPES.index(k)] > 0
        heads[k] = _head().fit(X, yk) if 0 < yk.sum() < len(yk) else _Const(float(yk.mean()))
    model = dict(any=any_h, heads=heads, types=types)
    A = np.vstack([_shot_x(*_predict_frames(model, r['f']), r['f']) for r in rows])
    sc = np.array([r['score'] for r in rows])
    ok = ~np.isnan(sc)
    model['score'] = Ridge(alpha=3.0).fit(A[ok], sc[ok])
    return model


def _predict_frames(model, f):
    X = _frame_x(f)
    p = model['any'].predict_proba(X)[:, 1]
    P = np.stack([model['heads'][k].predict_proba(X)[:, 1] for k in model['types']], 1) if model['types'] \
        else np.zeros((len(X), 0))
    return p, P


def train(log=print):
    """Fit on every stored round; report grouped (leave-shots-out) accuracy; save model.pkl."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    rows = _load_all(log)
    if len(rows) < 8:
        raise SystemExit(f'only {len(rows)} labelled shots: review more rounds first')
    Yall = np.vstack([r['Y'] for r in rows])
    types = []
    for k in TYPES:
        c = Yall[:, TYPES.index(k)] > 0
        nshots = sum((r['Y'][:, TYPES.index(k)] > 0).any() for r in rows)
        if c.sum() >= MIN_POS and nshots >= 3:
            types.append(k)
    # grouped CV: all takes of a shot id in the same fold
    ids = sorted({r['shot'] for r in rows})
    rng = np.random.RandomState(0)
    rng.shuffle(ids)
    folds = [set(ids[i::5]) for i in range(5)]
    yt, pt, st, ps, tt = [], [], [], [], {k: ([], []) for k in types}
    for fo in folds:
        tr = [r for r in rows if r['shot'] not in fo]
        te = [r for r in rows if r['shot'] in fo]
        m = _fit(tr, types)
        for r in te:
            p, P = _predict_frames(m, r['f'])
            yt.append(r['y'] >= 0.75)
            pt.append(p)
            for j, k in enumerate(types):
                tt[k][0].append(r['Y'][:, TYPES.index(k)] > 0)
                tt[k][1].append(P[:, j])
            if not np.isnan(r['score']):
                st.append(r['score'])
                ps.append(float(m['score'].predict(_shot_x(p, P, r['f'])[None])[0]))
    yt, pt = np.concatenate(yt), np.concatenate(pt)
    rep = dict(shots=len(rows), frames=int(len(yt)), defect_frames=int(yt.sum()), types=types,
               auc_any=round(float(roc_auc_score(yt, pt)), 3),
               score_spearman=round(float(spearmanr(st, ps).correlation), 3),
               score_mae=round(float(np.mean(np.abs(np.array(st) - np.array(ps)))), 2),
               auc_type={k: round(float(roc_auc_score(np.concatenate(a), np.concatenate(b))), 3)
                         for k, (a, b) in tt.items() if np.concatenate(a).any()},
               trained=time.strftime('%Y-%m-%d %H:%M'))
    model = _fit(rows, types)
    # our own temporal transformer (vf/gen/framenet.py) on the same rows; keep whichever does better held-out
    net = _train_net(rows, types, log)
    model['engine'] = 'linear'
    common = [k for k in rep['auc_type'] if net and k in net['auc_type']]
    if common and np.mean([net['auc_type'][k] for k in common]) > np.mean([rep['auc_type'][k] for k in common]):
        model['engine'] = 'framenet'
    best = net if model['engine'] == 'framenet' else rep
    model['trusted'] = [k for k, v in best['auc_type'].items() if v >= TRUST]
    model['score_ok'] = best['score_spearman'] >= SCORE_OK
    rep['framenet'] = net
    rep['engine'] = model['engine']
    rep['trusted'] = model['trusted']
    rep['score_ok'] = model['score_ok']
    model['report'] = rep
    os.makedirs(STORE, exist_ok=True)
    pickle.dump(model, open(_p('model.pkl'), 'wb'))
    json.dump(rep, open(_p('train_report.json'), 'w'), indent=1)
    log(f"[framecheck] trained on {rep['shots']} shots / {rep['frames']} frames ({rep['defect_frames']} defect).")
    log(f"  held-out: frame AUC {rep['auc_any']}, shot score Spearman {rep['score_spearman']}, MAE {rep['score_mae']}")
    for k, v in rep['auc_type'].items():
        log(f'  {k:15s} AUC {v}' + (f"   framenet {net['auc_type'].get(k)}" if net else ''))
    if net:
        log(f"  framenet held-out: frame AUC {net['auc_any']}, score Spearman {net['score_spearman']}, MAE {net['score_mae']}")
    log(f"  using: {model['engine']}; reported types (held-out AUC >= {TRUST}): {', '.join(model['trusted']) or 'none'}; "
        f"shot score {'shown' if model['score_ok'] else f'hidden (Spearman < {SCORE_OK})'}")
    return rep


def _pack(rows_or_feats, types, out):
    """Shots -> one npz for vf/gen/framenet.py (frame features without the hand-made temporal columns)."""
    rows = rows_or_feats
    X = np.vstack([np.hstack([r['f']['img'], r['f']['sims']]) for r in rows]).astype(np.float32)
    off = np.r_[0, np.cumsum([len(r['f']['img']) for r in rows])].astype(np.int64)
    n = len(X)
    y = np.concatenate([r.get('y', np.zeros(len(r['f']['img']))) for r in rows]).astype(np.float32)
    Y = np.vstack([r['Y'][:, [TYPES.index(k) for k in types]] if 'Y' in r else np.zeros((len(r['f']['img']), len(types)))
                   for r in rows]).astype(np.float32) if types else np.zeros((n, 0), np.float32)
    ids = sorted({r['shot'] for r in rows})
    np.savez(out, X=X, off=off, y=y, Y=Y, score=np.array([r.get('score', np.nan) for r in rows], np.float32),
             group=np.array([ids.index(r['shot']) for r in rows], np.int64), types=np.array(types))
    return out


def _train_net(rows, types, log):
    from .gen.bridge import run_gen
    try:
        data = _pack(rows, types, _p('trainset.npz'))
        log('[framecheck] training the temporal transformer (5-fold + final fit, CPU) ...')
        r = run_gen('framenet', [dict(mode='train', data=data, out=_p('framenet.pt'), folds=5)], persistent=False)[0]
        log(f"[framecheck] framenet done in {r['seconds']}s")
        return r['cv']
    except Exception as e:   # the linear model still works without it
        log(f'[framecheck] framenet failed (linear model kept): {e}')
        return None


# ---------- fix policy ----------
def _ledger():
    p = _p('fixes.jsonl')
    return [json.loads(l) for l in open(p, encoding='utf-8')] if os.path.exists(p) else []


def _write_ledger(rows):
    os.makedirs(STORE, exist_ok=True)
    with open(_p('fixes.jsonl'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def record(episode, shot, actions, before_labels=None, log=print):
    """Log a fix about to be built: the shot's defect types now (from the last review) and the actions taken."""
    bad = [a for a in actions if a not in ACTIONS]
    if bad:
        raise SystemExit(f'unknown action(s) {bad}; known: {sorted(ACTIONS)}')
    before = {}
    if before_labels:
        e = {x['shot']: x for x in json.load(open(before_labels, encoding='utf-8'))}.get(shot) or {}
        for d in e.get('defects', []):
            before[d['type']] = max(before.get(d['type'], 0), SEV.get(d.get('severity'), 1.0) * 2)
        before_score = e.get('score')
    else:
        before_score = None
    rows = _ledger()
    rows.append(dict(episode=episode, shot=shot, actions=list(actions), before=before, before_score=before_score,
                     after=None, after_score=None, at=time.strftime('%Y-%m-%d %H:%M')))
    _write_ledger(rows)
    log(f'[framecheck] fix logged: {shot} <- {actions} (pending the next review)')


def resolve(after_labels, episode=None, log=print):
    """Fill every pending fix whose shot is in this review."""
    A = {x['shot']: x for x in json.load(open(after_labels, encoding='utf-8'))}
    rows, n = _ledger(), 0
    for r in rows:
        if r['after'] is None and r['shot'] in A and (episode is None or r['episode'] == episode):
            after = {}
            for d in A[r['shot']].get('defects', []):
                after[d['type']] = max(after.get(d['type'], 0), SEV.get(d.get('severity'), 1.0) * 2)
            r['after'], r['after_score'] = after, A[r['shot']].get('score')
            n += 1
    _write_ledger(rows)
    log(f'[framecheck] {n} pending fix(es) resolved from {after_labels}')
    return n


def policy():
    """{defect type: [(action, posterior mean, wins, losses)]} best first."""
    stats = {}
    for r in _ledger():
        if r['after'] is None:
            continue
        for t, sev in r['before'].items():
            win = r['after'].get(t, 0) < sev
            share = 1.0 / len(r['actions'])   # several actions on one shot split the credit
            for a in r['actions']:
                w, l = stats.get((t, a), (0.0, 0.0))
                stats[(t, a)] = (w + win * share, l + (not win) * share)
    out = {}
    for t in TYPES:
        pri = PRIOR.get(t, [])
        cand = list(dict.fromkeys(pri + [a for (tt, a) in stats if tt == t]))
        rank = []
        for i, a in enumerate(cand):
            w, l = stats.get((t, a), (0.0, 0.0))
            prior = 0.6 - 0.05 * i if a in pri else 0.35
            # Beta(1+w, 1+l) mean shrunk toward the prior with 2 pseudo-trials
            mean = (prior * 2 + w) / (2 + w + l)
            rank.append((a, round(mean, 3), round(w, 2), round(l, 2)))
        out[t] = sorted(rank, key=lambda x: -x[1])
    return out


def fix_table(log=print):
    pol = policy()
    rows = _ledger()
    log(f"[framecheck] fix ledger: {len(rows)} fixes, {sum(r['after'] is not None for r in rows)} resolved")
    for t, rank in pol.items():
        if any(w + l for _, _, w, l in rank):
            log(f'  {t:15s} ' + '  '.join(f'{a} {m:.2f} ({w:g}/{w + l:g})' for a, m, w, l in rank[:4]))
    return pol


# ---------- check ----------
def check(yaml_path, video=None, log=print):
    """Grids + features + prediction for every shot; writes out/framecheck.json and out/framecheck_fixes.md."""
    if not os.path.exists(_p('model.pkl')):
        raise SystemExit('no trained model: vf framecheck learn <grid_dir> && vf framecheck train')
    # model.pkl is written only by train() on this machine (sklearn objects); never load one from elsewhere
    model = pickle.load(open(_p('model.pkl'), 'rb'))
    root = os.path.dirname(os.path.abspath(yaml_path))
    g = grids(yaml_path, video, log=log)
    ids = list(g)
    feats = features([g[s] for s in ids], log)
    pol = policy()
    shots = []
    net = None
    if model.get('engine') == 'framenet':
        from .gen.bridge import run_gen
        rows = [dict(shot=s, f=f) for s, f in zip(ids, feats)]
        data = _pack(rows, model['types'], os.path.join(root, 'out', 'framecheck', 'set.npz'))
        o = os.path.join(root, 'out', 'framecheck', 'pred.npz')
        run_gen('framenet', [dict(mode='predict', model=_p('framenet.pt'), data=data, out=o)], persistent=False)
        net = dict(np.load(o))
    for i, (s, f) in enumerate(zip(ids, feats)):
        if net is not None:
            a, b = net['off'][i], net['off'][i + 1]
            p, P, score = net['p'][a:b], net['P'][a:b], float(np.clip(net['score'][i], 0, 10))
        else:
            p, P = _predict_frames(model, f)
            score = float(np.clip(model['score'].predict(_shot_x(p, P, f)[None])[0], 0, 10))
        trusted = model.get('trusted', model['types'])
        cols = [j for j, k in enumerate(model['types']) if k in trusted]
        types = sorted(((model['types'][j], float(P[:, j].max()), [int(i) + 1 for i in np.where(P[:, j] > 0.5)[0]])
                        for j in cols), key=lambda x: -x[1])
        types = [t for t in types if t[1] > 0.5]
        bad = np.where((P[:, cols] > 0.5).any(1))[0] if cols else np.array([], int)
        risk = sum(v for _, v, _ in types)
        shots.append(dict(shot=s, score=round(score, 1) if model.get('score_ok', True) else None, risk=round(risk, 2),
                          bad_frames=_ranges(bad + 1), types=[dict(type=k, p=round(v, 2), frames=_ranges(np.array(fr)))
                                                              for k, v, fr in types],
                          fixes=[dict(type=k, actions=[a for a, *_ in pol[k][:3]]) for k, _, _ in types[:2]]))
    shots.sort(key=lambda x: -x['risk'])
    res = dict(model=model['report'], video=_video(yaml_path, video), shots=shots)
    json.dump(res, open(os.path.join(root, 'out', 'framecheck.json'), 'w'), indent=1)
    rp = model['report']
    md = [f"# framecheck fix plan ({time.strftime('%Y-%m-%d %H:%M')})", '',
          f"Model: {rp['shots']} reviewed shots, engine {model.get('engine', 'linear')}. Reported defect types (held-out AUC "
          f">= {TRUST}): {', '.join(model.get('trusted', [])) or 'none'}. Predicted shot score: "
          f"{'shown' if model.get('score_ok', True) else 'not shown, not reliable yet'}. A first pass, not a verdict: critics still read every frame.", '',
          '| shot | risk | predicted score | defect frames | likely defects | fixes to try (best first) |', '|---|---|---|---|---|---|']
    for x in shots:
        md.append(f"| {x['shot']} | {x['risk']} | {x['score'] if x['score'] is not None else '-'} | {x['bad_frames'] or '-'} | "
                  + ', '.join(f"{t['type']} {t['p']}" for t in x['types'][:3]) + ' | '
                  + '; '.join(f"{f['type']}: " + ' > '.join(f['actions']) for f in x['fixes']) + ' |')
    md += ['', 'Actions:'] + [f'- `{a}`: {d}' for a, d in ACTIONS.items()]
    open(os.path.join(root, 'out', 'framecheck_fixes.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    log(f"[framecheck] {len(shots)} shots; highest risk first:")
    for x in shots[:10]:
        log(f"  {x['shot']:16s} risk {x['risk']:4.2f}  frames {x['bad_frames'] or '-':18s} "
            + ', '.join(f"{t['type']} {t['p']}" for t in x['types'][:2]))
    log(f"[framecheck] plan -> {os.path.join(root, 'out', 'framecheck_fixes.md')}")
    return res


def _ranges(idx):
    idx = sorted(int(i) for i in idx)
    out, a = [], None
    for i, v in enumerate(idx):
        if a is None:
            a = v
        if i == len(idx) - 1 or idx[i + 1] != v + 1:
            out.append(f'{a}-{v}' if v != a else f'{a}')
            a = None
    return ','.join(out)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='vf framecheck')
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('episode')
    c.add_argument('--video')
    gr = sub.add_parser('grids')
    gr.add_argument('episode')
    gr.add_argument('--video')
    gr.add_argument('--out')
    le = sub.add_parser('learn')
    le.add_argument('grid_dir')
    le.add_argument('--labels')
    sub.add_parser('train')
    fx = sub.add_parser('fixes')
    fx.add_argument('what', choices=['record', 'resolve', 'table'])
    fx.add_argument('args', nargs='*')
    fx.add_argument('--labels')
    fx.add_argument('--episode')
    a = ap.parse_args(argv)
    if a.cmd == 'check':
        check(a.episode, a.video)
    elif a.cmd == 'grids':
        grids(a.episode, a.video, a.out)
    elif a.cmd == 'learn':
        learn(a.grid_dir, a.labels)
    elif a.cmd == 'train':
        train()
    elif a.what == 'record':
        if len(a.args) != 3:
            raise SystemExit('usage: vf framecheck fixes record <episode name> <shot> <action,action> [--labels before.json]')
        record(a.args[0], a.args[1], a.args[2].split(','), a.labels)
    elif a.what == 'resolve':
        resolve(a.args[0], a.episode)
    else:
        fix_table()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
