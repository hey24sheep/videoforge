"""Cast lock: every recurring character keeps one face in every shot of an episode and across a series.

The local stand-in for multi-reference character lock (Seedance-style): each character has a reference set in the
series cast sheet; every keyframe that shows the character is
  1. drawn as `cast_candidates` Z-Image stills (composition),
  2. redrawn by FLUX.2 Klein in edit mode with the character's reference face crops (identity),
  3. scored by SFace cosine similarity against the references (robust: 2x upscale + rotated passes for small or
     tilted faces), and the best one kept if it clears the threshold; below it the shot re-rolls (new seeds,
     `cast_rerolls` rounds), else the best take is kept and flagged.
A QC report (out/cast_qc.json) scores each cast shot's keyframe and sampled frames of its finished clip.

series.yaml:
    cast_lock: true                 # on for every episode of the series ...
    cast_lock_except: [ep01]        # ... except these episode folders (until the owner approves them)
    cast_threshold: 0.40            # SFace cosine; same person ~0.55-0.89, different people ~0.02-0.27
    cast:
      ada:
        name: Ada Quinn             # also matched in prompts ("Ada", "Quinn")
        description: "..."          # optional; defaults to characters.ada (the {ada} prompt text)
        refs: [cast/mara_ref.png]   # 1-3 images (series folder relative); the first face found in each is used
        aliases: [the woman]        # optional extra words that tag a shot
        anchor: cast/ada.png        # optional: the legacy best-of-3 anchor (default cast/<key>.png)
      detective: cast/detective.png # legacy form: a path = one ref + the anchor
episode.yaml:
    cast_lock: true|false           # overrides the series switch
    cast: {ada: {refs: [...]}}      # per-episode overrides (new wardrobe, ageing) or episode-only characters
shot:
    cast: [ada]                     # explicit; `cast: []` / `cast: false` = none. Else inferred from the keyframe
                                    # (or motion) prompt: {key}, the key, name words, aliases, the description.
    gen: {cast_candidates: 2, cast_rerolls: 1, cast_klein: true, cast_refine: false, cast_seed: 7}
    cast_gate: clip | final         # clip: judge the clip only (keyframe never redrawn/flagged); final: the face is
                                    # composited later (e.g. a stamped photo): reported on the final media only,
                                    # never redrawn or re-rolled
Automatic regeneration (2026-11-03): a supplied `image:` scoring below the threshold gets the Klein redraw + re-roll
(`cast_refine: false` opts out); a generated clip whose sampled-frame median falls below it re-rolls its seed up to
`cast_rerolls` times even when `retries: 0`; still below -> best take kept and flagged (log + cast_qc.json).
                                    # cast_refine: also run the Klein identity pass on a supplied `image:`

Shots that don't opt in are untouched (no new arguments reach any cached generator).
CLI (render python, no GPU): python -m vf.cast plan <episode.yaml>   - which shots get which cast
                             python -m vf.cast qc <episode.yaml>     - score keyframes/clips -> out/cast_qc.json
"""
import hashlib, json, os, re, subprocess, sys, tempfile, time
import numpy as np
import cv2
import yaml
from .memo import memo, tier_dir

VERSION = 'cast-1'
THRESHOLD = 0.40
ROTATIONS = (30, -30, 60, -60, 90, -90)
KLEIN_STYLE = ('Keep the exact same shot, framing, composition, pose, clothing, props, background and lighting. '
               'Photorealistic, natural skin texture, detailed eyes, clean edges, do not add any text.')


# ------------------------------------------------------------------ cast sheet
def _episode_raw(spec):
    try:
        return yaml.safe_load(open(spec['path'], encoding='utf-8')) or {}
    except (OSError, KeyError):
        return {}


def sheet(spec):
    """{key: dict(key, name, description, refs=[abs paths], aliases, threshold)} = series cast + episode overrides."""
    se = spec.get('series') or {}
    chars = se.get('characters') or {}
    ep = _episode_raw(spec)
    out = {}
    for src, base in ((se.get('cast') or {}, spec.get('series_dir')), (ep.get('cast') or {}, spec.get('root'))):
        for key, v in src.items():
            if isinstance(v, str):
                v = dict(refs=[v])
            v = dict(v or {})
            cur = out.get(key, dict(key=key, name=key, description=chars.get(key, ''), refs=[], aliases=[],
                                    threshold=float(se.get('cast_threshold', THRESHOLD))))
            if 'refs' in v:
                cur['refs'] = [_abs(spec, p, base) for p in (v['refs'] if isinstance(v['refs'], list) else [v['refs']])]
            for k in ('name', 'description', 'aliases', 'threshold'):
                if k in v:
                    cur[k] = v[k]
            out[key] = cur
    for c in out.values():
        c['refs'] = [p for p in c['refs'] if p][:3]
        c['threshold'] = float(c['threshold'])
    return out


def _abs(spec, p, base):
    if os.path.isabs(p):
        return os.path.normpath(p)
    for b in (base, spec.get('root'), spec.get('series_dir'), _ROOT):
        if b and os.path.exists(os.path.join(b, p)):
            return os.path.normpath(os.path.join(b, p))
    return os.path.normpath(os.path.join(base or _ROOT, p))


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def enabled(spec):
    """Episode `cast_lock` wins; else series `cast_lock` unless this episode folder is in `cast_lock_except`."""
    ep = _episode_raw(spec)
    if 'cast_lock' in ep:
        return bool(ep['cast_lock'])
    se = spec.get('series') or {}
    return bool(se.get('cast_lock')) and os.path.basename(spec['root']) not in (se.get('cast_lock_except') or [])


_TITLES = {'det', 'detective', 'mr', 'mrs', 'ms', 'miss', 'dr', 'officer', 'sgt', 'the'}


def shot_cast(spec, s, cast=None):
    """Character keys in a shot: explicit `cast:` (shot or gen), else inferred from its keyframe/motion prompt."""
    cast = sheet(spec) if cast is None else cast
    g = s.get('gen') or {}
    for v in (s.get('cast', None), g.get('cast', None)):
        if v is False:
            return []
        if v is not None:
            v = [v] if isinstance(v, str) else list(v)
            bad = [k for k in v if k not in cast]
            if bad:
                raise ValueError(f"shot {s['id']}: cast {bad} not in the cast sheet ({', '.join(cast) or 'empty'})")
            return v
    text = str(g.get('keyframe') or g.get('motion') or '')
    low = text.lower()
    found = []
    for key, c in cast.items():
        words = {key.lower()} | {w.lower() for w in re.findall(r"[A-Za-z']+", str(c['name']))
                                  if len(w) >= 3 and w.lower() not in _TITLES} | {a.lower() for a in c.get('aliases') or []}
        hit = '{%s}' % key in text or any(re.search(rf'\b{re.escape(w)}\b', low) for w in words)
        d = str(c.get('description') or '').strip().lower()
        if hit or (len(d) > 20 and d in low):
            found.append(key)
    return found


# ------------------------------------------------------------------ faces (robust)
def _detect(img):
    from .identity import _det
    h, w = img.shape[:2]
    _, f = _det(w, h).detect(img)
    return [] if f is None else list(f)


def _embed_faces(img, min_px=32):
    """SFace features of every face found; small/tilted faces get a 2x upscale and rotated passes."""
    from .identity import _rec
    h, w = img.shape[:2]
    feats = []

    def run(im, scale):
        got = 0
        for f in _detect(im):
            if f[2] / scale < min_px:
                continue
            feats.append(_rec().feature(_rec().alignCrop(im, f)).astype(np.float32).ravel())
            got += 1
        return got

    s = min(1.0, 1280 / max(h, w))
    base = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    n = run(base, s)
    k = min(2.0, 2560 / max(h, w))
    if k > 1.2:  # small faces: the detector's floor is ~20 px and SFace wants 112 px crops
        up = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_CUBIC)
        n += run(up, k)
    else:
        up, k = base, s
    # tilted heads (lying down, leaning): YuNet misses them or finds them with bad landmarks (measured: a 35-degree
    # tilt scored 0.27 upright vs 0.76 after rotating). Rotated passes only ADD faces; the score is the best match.
    c = (up.shape[1] / 2, up.shape[0] / 2)
    for a in ROTATIONS:
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        run(cv2.warpAffine(up, M, (up.shape[1], up.shape[0]), borderMode=cv2.BORDER_REFLECT), k)
    return np.stack(feats) if feats else np.zeros((0, 128), np.float32)


@memo('global', version=lambda: f'cast-faces-2|{ROTATIONS}')
def faces(image_path):
    img = cv2.imread(image_path)
    return np.zeros((0, 128), np.float32) if img is None else _embed_faces(img)


def _cos(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T


def ref_feats(c):
    """One feature per reference image (its largest/first face)."""
    out = []
    for p in c['refs']:
        f = faces(p)
        if len(f):
            out.append(np.asarray(f[0]))
    return np.stack(out) if out else None


def score_feats(feats, refs):
    """Mean over references of the best-matching face; None when no face or no reference."""
    if refs is None or feats is None or not len(feats):
        return None
    return float(_cos(np.asarray(feats), refs).max(axis=0).mean())


def score(image_path, chars, cast):
    """{key: score|None} for the characters in an image."""
    f = faces(image_path)
    return {k: score_feats(f, ref_feats(cast[k])) for k in chars}


def _total(sc):
    v = [x for x in sc.values()]
    return -1.0 if not v or any(x is None for x in v) else min(v)


# ------------------------------------------------------------------ Klein identity pass
def face_crop(path):
    """Square crop around the largest face of a reference (Klein reads a face crop better than a full frame)."""
    img = cv2.imread(path)
    if img is None:
        return path
    h, w = img.shape[:2]
    f = sorted(_detect(img), key=lambda f: -f[2] * f[3])
    if not f:
        return path
    x, y, fw, fh = f[0][:4]
    side = int(max(fw, fh) * 2.0)
    cx, cy = x + fw / 2, y + fh * 0.45
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, x0 + side)), int(min(h, y0 + side))
    st = os.stat(path)
    k = hashlib.blake2b(f'{path}|{st.st_size}|{st.st_mtime_ns}|crop2'.encode(), digest_size=10).hexdigest()
    out = os.path.join(tier_dir('global'), 'cast_refs', f'{os.path.splitext(os.path.basename(path))[0]}_{k}.png')
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        crop = img[y0:y1, x0:x1]
        if crop.shape[0] < 512:
            crop = cv2.resize(crop, (512, int(512 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(out, crop)
    return out


def _file_key(p):
    st = os.stat(p)
    return f'{os.path.abspath(p)}|{st.st_size}|{st.st_mtime_ns}'


def klein_job(src, refs, prompt, seed, W, H, style=None):
    base = '|'.join([VERSION, _file_key(src)] + [_file_key(r) for r in refs] + [prompt, str(seed), str(W), str(H),
                                                                            style or KLEIN_STYLE])
    prec = os.environ.get('VF_KLEIN_PREC', 'int8')   # vf/gen/klein.py PREC (2026-11-10: int8 default)
    d = os.path.join(tier_dir('global'), 'cast_kf')
    old = os.path.join(d, hashlib.blake2b(base.encode(), digest_size=16).hexdigest() + '.png')   # fp8store frames
    key = base if prec == 'fp8store' else f'{base}|{prec}'
    out = old if os.path.exists(old) else os.path.join(d, hashlib.blake2b(key.encode(), digest_size=16).hexdigest() + '.png')
    return dict(src=os.path.abspath(src), out=out, refs=refs, prompt=prompt, seed=int(seed), width=int(W),
                height=int(H), style=style or KLEIN_STYLE)


def run_klein(jobs, log=print, timeout=1800):
    """Run Klein edit jobs in their own gen-venv process (skips outputs that exist). Returns the set of outs made."""
    todo = [j for j in jobs if not os.path.exists(j['out'])]
    if todo:
        from .gen.bridge import GEN_PY, ROOT, _env, stop_gen
        from .gpulock import gpu_lock
        fd, path = tempfile.mkstemp(prefix='vf_cast_klein_', suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(todo, f)
        try:
            with gpu_lock('vf cast klein'):
                try:  # stop, not park: a parked Z-Image DiT holds ~6 GB of RAM and Klein's encoder guard needs
                    stop_gen()  # 20 GB free (measured: 13.8 GB free with it parked -> refused). We hold the GPU lock,
                except Exception:  # so no client is mid-job; the worker restarts on the next run_gen (~7 s).
                    pass
                t = time.time()
                p = subprocess.run([GEN_PY, '-m', 'vf.gen.klein', path], cwd=ROOT, env=_env(), capture_output=True,
                                   text=True, encoding='utf-8', errors='replace', timeout=timeout)
            if p.returncode:
                tail = (p.stderr or p.stdout or '').strip().splitlines()[-3:]
                log(f'    cast: Klein failed (exit {p.returncode}): {" | ".join(tail)}')
            else:
                log(f'    cast: Klein {len(todo)} frame(s) in {time.time() - t:.0f}s')
        finally:
            os.remove(path)
    return {j['out'] for j in jobs if os.path.exists(j['out'])}


# ------------------------------------------------------------------ keyframe stage
def _chars(spec):
    cast = sheet(spec)
    return dict({k: c['description'] for k, c in cast.items() if c.get('description')},
                **((spec.get('series') or {}).get('characters') or {}))


def t2i_jobs(spec, s, formats, r=0):
    """Z-Image composition candidates of a cast shot for round r (round 0 joins the stage's main batch, so all
    new prompts are encoded in one text-encoder load: a second load next to the resident DiT is refused for RAM)."""
    from .stages import _kf_size
    from .styles import wrap_prompt, shot_style
    g = s['gen']
    W, H = _kf_size(formats, g)
    pinned = 'keyframe_seed' in g
    k = 1 if pinned else int(g.get('cast_candidates', 2))
    seed0 = int(g.get('keyframe_seed', g.get('seed', 1))) + (0 if pinned else r * 1000)
    p = wrap_prompt(shot_style(spec.get('style'), g), g['keyframe'], _chars(spec), kind='keyframe')
    return [dict(prompt=p, width=W, height=H, seed=seed0 + c * 101) for c in range(k)]


def lock_keyframes(spec, shots, formats, log=print):
    """Cast-locked keyframes for `shots` (each has g['_cast_keys']): candidates -> Klein with refs -> gate -> re-roll.
    Sets g['_image'] and g['_cast'] (the decision record)."""
    from .gen.bridge import run_gen
    from .styles import wrap_prompt, shot_style
    cast = sheet(spec)
    chars = _chars(spec)
    wrap = lambda t, gg=None: wrap_prompt(shot_style(spec.get('style'), gg), t, chars, kind='keyframe')
    state = {}
    for s in shots:
        g = s['gen']
        keys = g['_cast_keys']
        refs = [face_crop(p) for k in keys for p in cast[k]['refs']]
        missing = [k for k in keys if ref_feats(cast[k]) is None]
        if missing:
            log(f"    {s['id']}: cast {missing} has no usable reference face: identity gate off for them")
        thr = max(cast[k]['threshold'] for k in keys)
        state[s['id']] = dict(s=s, keys=[k for k in keys if k not in missing], refs=refs, thr=thr, takes=[],
                              done=False)
    rounds = 1 + max(int(st['s']['gen'].get('cast_rerolls', 1)) for st in state.values()) if state else 0
    for r in range(rounds):
        live = [st for st in state.values() if not st['done'] and r <= int(st['s']['gen'].get('cast_rerolls', 1))]
        if not live:
            break
        # 1. compositions (Z-Image) - supplied images are used as they are
        t2i, own = [], []
        for st in live:
            g = st['s']['gen']
            if g.get('image'):
                st['cands'] = [os.path.abspath(g['_image'])]
                continue
            if r == 0 and g.get('_cast_r0'):  # drawn in the stage's main Z-Image batch (one encoder load)
                st['cands'] = list(g.pop('_cast_r0'))
                continue
            st['cands'] = []
            for j in t2i_jobs(spec, st['s'], formats, r):
                t2i.append(j)
                own.append(st)
        if t2i:
            from .gpulock import gpu_lock
            with gpu_lock('vf cast keyframes'):
                try:
                    res = run_gen('image', t2i)
                except OSError as e:  # the shared worker was stopped under us (another session's cleanup): once more
                    log(f'    cast: gen worker connection lost ({e}); retrying')
                    res = run_gen('image', t2i)
            for st, rr in zip(own, res):
                st['cands'].append(rr['out'])
        # 2. identity pass (Klein edit with the reference face crops)
        kj, kown = [], []
        for st in live:
            g = st['s']['gen']
            use = g.get('cast_klein', True) and st['refs'] and (not g.get('image') or g.get('cast_refine')
                                                                      or g.get('_cast_auto'))
            if not use:
                continue
            im = cv2.imread(st['cands'][0])
            H, W = im.shape[:2]
            W, H = W // 16 * 16, H // 16 * 16
            text = g.get('keyframe') or g.get('cast_prompt') or g.get('motion') or ''
            for i, c in enumerate(st['cands']):
                kj.append(klein_job(c, st['refs'], wrap(text, g), int(g.get('cast_seed', 7)) + 100 * r + i, W, H,
                                    g.get('cast_style') or (spec.get('series') or {}).get('cast_style')))
                kown.append(st)
        made = run_klein(kj, log) if kj else set()
        for st, j in zip(kown, kj):
            if j['out'] in made:
                st['cands'].append(j['out'])
        # 3. gate
        for st in live:
            g = st['s']['gen']
            for c in st.pop('cands'):
                sc = score(c, st['keys'], cast) if st['keys'] else {}
                st['takes'].append(dict(path=c, round=r, klein=os.path.basename(os.path.dirname(c)) == 'cast_kf',
                                        scores={k: None if v is None else round(v, 3) for k, v in sc.items()},
                                        total=round(_total(sc), 3) if st['keys'] else None))
            best = max(st['takes'], key=lambda t: (t['total'] if t['total'] is not None else -1, t['klein']))
            st['best'] = best
            ok = not st['keys'] or (best['total'] is not None and best['total'] >= st['thr'])
            tag = ', '.join(f"{'K' if t['klein'] else 'Z'}{t['total']}" for t in st['takes'] if t['round'] == r)
            log(f"    {st['s']['id']}: cast {st['keys']} round {r}: [{tag}] best {best['total']} "
                f"{'PASS' if ok else 'below ' + str(st['thr'])}")
            blind = st['keys'] and all(any(v is None for v in t['scores'].values())
                                       for t in st['takes'] if t['round'] == r)
            if blind:  # back view / wide shot: nothing to measure, a re-roll can't help
                log(f"    {st['s']['id']}: no measurable face for {st['keys']}: kept candidate (tag `cast: []` if "
                    f"the face is not meant to be seen)")
            if ok or blind or g.get('image') and not (g.get('cast_refine') or g.get('_cast_auto')):
                st['done'] = True
    for st in state.values():
        g = st['s']['gen']
        b = st['best']
        g['_image'] = b['path']
        g['_cast'] = dict(cast=st['keys'], threshold=st['thr'], chosen=b['path'], score=b['total'],
                          passed=b['total'] is not None and b['total'] >= st['thr'] if st['keys'] else None,
                          takes=st['takes'])
        if st['keys'] and not g['_cast']['passed']:
            log(f"    {st['s']['id']}: cast identity still below {st['thr']} after re-rolls (best {b['total']}): "
                f"kept the best take - check it, add a closer reference, or restage the shot (bigger face)")


# ------------------------------------------------------------------ QC report
def _clip_frames(path, n=4):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for q in np.linspace(0.1, 0.9, n):
        if total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(q * (total - 1)))
        ok, fr = cap.read()
        if ok:
            out.append((round(float(q), 2), fr))
    cap.release()
    return out


@memo('global', version='cast-clipqc-1')
def clip_scores(clip_path, ref_paths):
    """[(position, [score per ref set])] for a few sampled frames; ref_paths: {key: [ref paths]}."""
    refs = {k: ref_feats(dict(refs=v)) for k, v in ref_paths.items()}
    out = []
    for q, fr in _clip_frames(clip_path):
        f = _embed_faces(fr)
        out.append((q, {k: (None if score_feats(f, r) is None else round(score_feats(f, r), 3))
                        for k, r in refs.items()}))
    return out


def clip_gate(spec, s, clip, cast=None):
    """Identity of a generated clip: (median of sampled frames, threshold), or (None, thr) when no face is measured
    or the shot has no cast. Used by stages.generate to re-roll clips whose face drifted below the threshold."""
    cast = sheet(spec) if cast is None else cast
    keys = shot_cast(spec, s, cast)
    if not keys or (s.get('gen') or {}).get('cast_refine') is False or             (s.get('cast_gate') or (s.get('gen') or {}).get('cast_gate')) == 'final':
        return None, None  # 'final': the face is composited later (stamp): the raw clip can't be judged
    thr = max(cast[k]['threshold'] for k in keys)
    fr = clip_scores(os.path.abspath(clip), {k: cast[k]['refs'] for k in keys})
    per = [min(v.values()) for _, v in fr if v and all(x is not None for x in v.values())]
    return (round(float(np.median(per)), 3) if per else None), thr


def _shot_media(s):
    g = s.get('gen') or {}
    kf = g.get('_image') or g.get('image')
    clip = None
    for L in s.get('layers') or []:
        if L.get('type') == 'video':
            clip = L.get('src')
            break
    if not clip:
        clip = (g.get('_take') or {}).get('clip')
    return kf, clip


def report(spec, log=print, clips=True, takes=None):
    """Identity scores of every cast shot (keyframe + sampled clip frames) -> out/cast_qc.json; returns the dict."""
    from .spec import resolve
    cast = sheet(spec)
    shots, flagged = {}, []
    for s in spec['shots']:
        if not s.get('gen'):
            continue
        keys = shot_cast(spec, s, cast)
        if not keys:
            continue
        kf, clip = _shot_media(s)
        if takes and s['id'] in takes and not clip:
            clip = (takes[s['id']] or {}).get('clip')
        kf = resolve(spec, kf) if kf else None
        thr = max(cast[k]['threshold'] for k in keys)
        rec = dict(cast=keys, threshold=thr, keyframe=kf)
        if kf and os.path.exists(kf):
            rec['keyframe_scores'] = {k: (None if v is None else round(v, 3)) for k, v in score(kf, keys, cast).items()}
        if s['gen'].get('_cast'):
            rec['gate'] = {k: v for k, v in s['gen']['_cast'].items() if k != 'takes'}
            rec['candidates'] = [dict(path=t['path'], klein=t['klein'], total=t['total'])
                                 for t in s['gen']['_cast']['takes']]
        if clips and clip and os.path.exists(clip):
            fr = clip_scores(os.path.abspath(clip), {k: cast[k]['refs'] for k in keys})
            rec['clip'] = clip
            rec['clip_frames'] = [dict(at=q, scores=v) for q, v in fr]
            per = [min(v.values()) for _, v in fr if v and all(x is not None for x in v.values())]
            rec['clip_min'] = round(min(per), 3) if per else None
            rec['clip_median'] = round(float(np.median(per)), 3) if per else None
        ks = list((rec.get('keyframe_scores') or {}).values())
        rec['keyframe_min'] = None if not ks or any(v is None for v in ks) else min(ks)
        why = []
        if (s.get('cast_gate') or s['gen'].get('cast_gate')) in ('clip', 'final'):
            pass  # the face is composited later (e.g. a stamped photo): only the clip counts
        elif rec['keyframe_min'] is None:
            why.append('no face measured in keyframe')
        elif rec['keyframe_min'] < thr:
            why.append(f"keyframe {rec['keyframe_min']} < {thr}")
        if rec.get('clip_median') is not None and rec['clip_median'] < thr:
            why.append(f"clip median {rec['clip_median']} < {thr}")
        rec['flag'] = why
        if why:
            flagged.append(s['id'])
        shots[s['id']] = rec
    rep = dict(version=VERSION, lock=enabled(spec), cast={k: dict(name=c['name'], refs=c['refs'], threshold=c['threshold'])
                                                          for k, c in cast.items()},
               shots=shots, flagged=flagged)
    out = os.path.join(spec['root'], 'out', 'cast_qc.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=1, default=str)
    log(f'  cast QC ({len(shots)} cast shot(s), {len(flagged)} flagged) -> {out}')
    for sid, r in shots.items():
        log(f"    {sid:16s} {','.join(r['cast']):18s} keyframe {r.get('keyframe_min')}  clip median "
            f"{r.get('clip_median', '-')} min {r.get('clip_min', '-')}  {'FLAG: ' + '; '.join(r['flag']) if r['flag'] else 'ok'}")
    return rep


# ------------------------------------------------------------------ CLI
def main(argv=None):
    import argparse
    from . import spec as S
    ap = argparse.ArgumentParser(prog='python -m vf.cast')
    ap.add_argument('cmd', choices=['plan', 'qc'])
    ap.add_argument('episode')
    a = ap.parse_args(argv)
    spec = S.load(a.episode)
    cast = sheet(spec)
    if a.cmd == 'plan':
        print(f"cast lock: {'ON' if enabled(spec) else 'off'} for {os.path.basename(spec['root'])}")
        for k, c in cast.items():
            print(f"  {k}: {c['name']}  refs={[os.path.relpath(p, _ROOT) for p in c['refs']]}  thr={c['threshold']}")
        for s in spec['shots']:
            if s.get('gen'):
                g = s['gen']
                src = 'image' if g.get('image') else 'keyframe' if g.get('keyframe') else 'other'
                print(f"  {s['id']:18s} {src:8s} cast={shot_cast(spec, s, cast)}")
        return
    takes = {}
    p = os.path.join(spec['root'], 'out', 'clip_qc.json')
    if os.path.exists(p):
        takes = json.load(open(p, encoding='utf-8')) or {}
    report(spec, takes=takes)


if __name__ == '__main__':
    main()
