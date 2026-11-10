"""Fixed review harness (docs/review_workflow.md): the same checks every round, no per-round scripts.

    python -m vf.qa <episode.yaml> <video.mp4> --base <last approved.mp4> --focus shot1,shot2
                    [--timeline this_build.yaml] [--base-timeline approved.yaml] [--intent "shot: aspect; ..."]
    python -m vf.qa replay [qa/cases.yaml]     # re-run every calibration case, report agreement with the owner

All local, CPU + OpenCV + ffmpeg. Wall time is logged in qa.json.
  0. preflight asserts: frame size, fps, duration vs the spec; audio stream; every mask's aspect matches the frame.
  1. one sequential, frame-exact decode at native resolution: largest face per frame (YuNet), SFace identity on every
     2nd frame, thumbnails, cut-edge frames.
  2. tracked face and lip crops for focus shots AND their cut neighbours. Metrics:
       identity median/min (gated), lip/face jitter = per-pixel 2nd temporal difference (gated where the mouth is
       synthesised, relative to base elsewhere), face Lab at the shot edges (cut continuity), native-resolution face
       sharpness/over-sharpening/smear (INFO ONLY: on owner-judged builds they did not rank rejected below approved,
       qa/calibration.md, so they are never presented as evidence of quality).
  3. cuts: luma jump (warn), same-character face delta E (warn/fail), same-axis jump cut (fail).
  4. base comparison: untouched shots pixel-identical (fail), changed shots: identity drop, jitter rise (fail).
     A failure the base already had is KNOWN only for absolute gates and only if the value is not worse than base.
  5. audio: loudness, true peak, clipped samples, dropouts (silence inside the programme), lip-sync offset proxy.
  6. fail closed: a focus shot whose face/identity the base could measure but this build cannot = FAIL.
  7. critic packet: both cuts (new + base) of the focus shots under random A_/B_ names, 3 timecoded 8-frame windows
     per shot (worst, median, seeded random), native face crops, cut frames; written to a separate folder with no
     metrics, no history and no labels saying which cut is new or approved. Key in <out>/ab_key.json.
"""
import os, re, sys, json, time, hashlib, pickle, random, subprocess, shutil
import numpy as np
import cv2
import yaml

QA_VERSION = 'qa-11'  # qa-11: face image quality (VisualQuality-R1-7B, vf/gen/iqa.py) of every focus shot, new vs base;
# a drop of more than 0.3 (1-5 scale) FAILs (rejected q8/p43/p41/old pier_wide -0.41..-1.25, approved >= -0.02).
# qa-10: qa-10 (owner 2026-11-05 "let it vote"): a director model (Qwen3.5-9B, vf/gen/director.py) rates
# every focus shot, new and base take, one take at a time; its notes go in the report and a "worse" call turns CHECK
# (your call) into BLOCK. It never changes SHIP or BLOCK. Missing director on a would-be CHECK = INCOMPLETE.
# qa-9: the relative lip/face flicker gate uses motion-compensated residual from an optical-flow
# model (RAFT, vf/gen/flowqc.py) instead of the 2nd temporal difference, which grew with motion and sharpness and
# failed the owner-approved pier3. The old lip/face jitter numbers stay as INFO; frame jitter and the absolute
# lip-sync gate are unchanged.
# qa-8 (audit 2026-11-05): brief moved out of the A/B prompt into its own new-cut-only read
# (delivered? looks real?), realism-weighted score that actually decides (SHIP >= 8.0), median only on disputed
# classes, one prompt + answer folder per read, KNOWN gates listed with values, identity min = rolling median of 3,
# version + scoring hash enforced/logged, cache key covers cast refs/code, replay compares with the owner.
# qa-7: full-episode audit (no --focus) no longer fails unchanged shots; qa-6: tone judged absolutely
_QA_PREV = 'qa-4'   # QA_VERSION is frozen for held-out predictions: any change to gates/prompt bumps it
GATES = dict(
    identity_median=0.60,      # owner rule (2026-11-03): below .60 = regenerate + flag
    identity_min=0.50,         # rolling median of 3 samples below this = a momentary face swap (qa-8; was the raw min)
    identity_known_tol=0.02,   # a pre-existing identity fail stays KNOWN only if not worse than base by more than this
    lip_jitter_L=22.0,         # lip-synced shots, absolute: owner-caught flicker p43 31.2 vs approved q6c 16.3
    lip_jitter_a=10.0,         # same, colour: p43 14.0 vs q6c 7.1
    jitter_rise=1.3,           # changed shot vs base, any face: owner-rejected q8 mara_pier x1.80
    face_quality_drop=0.3,     # qa-11: VisualQuality-R1 face score, new vs base (in-sample: rejected <= -0.41, approved >= -0.02)
    director_worse=-0.02,      # qa-10: mean change of the director's 6 per-take scores (new - base) below this = worse
                               # (band set after seeing the past cases: in-sample)
    flow_rise=1.3,             # qa-9: changed face shot vs base, RAFT residual p99 (mouth / face): rejected p43 x1.43/x1.53,
                               # every approved build <= x1.05 (pier3 x0.57). Same 1.3 as the old gate, not re-tuned.
    identity_drop=0.05,        # changed shot vs base: q8 -0.076 (rejected), p43 mara_cu -0.103 (flicker build)
    cut_face_dE=10.0,          # same character across a cut -> WARN
    cut_face_dE_fail=15.0,     # -> FAIL (an approved close-up -> wide cut = 12.0)
    cut_luma_jump=35.0,        # WARN only: location changes are intended
    unchanged_block=1.0,       # untouched shots (renders are deterministic: measured 0.0 on all pinned cases): worst 10x10 block (on 90x160 thumbnails) over ALL frames, + equal frame count
    audio_db=1.0,              # untouched shots: per-shot audio level vs base (WARN)
    sharp_drop_warn=0.25,      # native face sharpness drop vs base (WARN: INFO metric, did not rank owner verdicts)
    min_face_px=60,            # faces smaller than this are not judged for jitter
    lufs=(-14.0, 1.5),
    true_peak=-1.0,            # dBTP; +0.05 dB tolerance for ffmpeg's rounding
    clipped_samples=0,
    dropout_s=0.30,            # silence (< -60 dBFS RMS) longer than this inside the programme
    lipsync_ms=None,           # measured (INFO) until validated against owner verdicts
)
FACE_W, FACE_H = 128, 160
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(_ROOT, 'cache', 'qa')


# ------------------------------------------------------------------ inputs
def timeline(yaml_path):
    """[(id, start, end)] from an episode yaml without loading media (works on backup copies)."""
    ep = yaml.safe_load(open(yaml_path, encoding='utf-8')) or {}
    out, t = [], 0.0
    for i, s in enumerate(ep.get('shots', [])):
        d = float(s.get('dur', 0))
        out.append((s.get('id', f'shot{i + 1:02d}'), t, t + d))
        t += d
    return out


def probe(video):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height,r_frame_rate,nb_frames',
                        '-show_entries', 'format=duration', '-of', 'json', video], capture_output=True, text=True,
                       timeout=60)
    j = json.loads(r.stdout or '{}')
    v = next((s for s in j.get('streams', []) if s.get('codec_type') == 'video'), {})
    num, den = (v.get('r_frame_rate') or '0/1').split('/')
    return dict(W=int(v.get('width', 0)), H=int(v.get('height', 0)), fps=float(num) / float(den or 1),
                frames=int(v.get('nb_frames') or 0), dur=float(j.get('format', {}).get('duration', 0)),
                audio=any(s.get('codec_type') == 'audio' for s in j.get('streams', [])))


def preflight(spec, video, fmt, tl):
    from .spec import FORMATS, resolve
    p, res = probe(video), []
    F = FORMATS[fmt]

    def chk(ok, what):
        res.append(dict(ok=bool(ok), check=what))
    chk((p['W'], p['H']) == (F['W'], F['H']), f"frame {p['W']}x{p['H']} == {F['W']}x{F['H']}")
    chk(abs(p['fps'] - spec['fps']) < 0.01, f"fps {p['fps']:.3f} == {spec['fps']}")
    chk(abs(p['dur'] - tl[-1][2]) < 0.15, f"duration {p['dur']:.2f} s == timeline {tl[-1][2]:.2f} s")
    chk(p['audio'], 'audio stream present')
    aspect = F['W'] / F['H']
    for s in spec['shots']:
        for L in s.get('layers', []):
            m = L.get('mask')
            if not m:
                continue
            im = cv2.imread(resolve(spec, m), cv2.IMREAD_UNCHANGED)
            if im is None:
                chk(False, f"{s['id']}: mask {m} readable")
                continue
            a = im.shape[1] / im.shape[0]
            chk(abs(a / aspect - 1) < 0.005, f"{s['id']}: mask {m} aspect {im.shape[1]}x{im.shape[0]} matches frame")
    return p, res


# ------------------------------------------------------------------ decode
def _largest(f, scale=0.5):
    from .identity import _det
    s = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, fa = _det(s.shape[1], s.shape[0]).detect(s)
    if fa is None or not len(fa):
        return None
    a = fa[np.argmax(fa[:, 2] * fa[:, 3])].copy()
    a[:14] /= scale
    return a


def _crop(f, cx, cy, w, h, ow, oh):
    M = np.float32([[ow / w, 0, ow / 2 - cx * ow / w], [0, oh / h, oh / 2 - cy * oh / h]])
    return cv2.warpAffine(f, M, (ow, oh), flags=cv2.INTER_AREA if w > ow else cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def decode(video, tl, fps, want_id):
    from .identity import _rec
    cap = cv2.VideoCapture(video)
    shots = {sid: dict(t0=t0, t1=t1, i0=int(round(t0 * fps)), i1=int(round(t1 * fps)), boxes=[], thumbs=[],
                       luma=[], ids=[], edge_small=[]) for sid, t0, t1 in tl}
    order = [(shots[sid]['i0'], shots[sid]['i1'], sid) for sid, _, _ in tl]
    i, k = 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        while k < len(order) and i >= order[k][1]:
            k += 1
        if k >= len(order):
            break
        i0, i1, sid = order[k]
        if i < i0:
            i += 1
            continue
        S = shots[sid]
        a = _largest(f)
        S['boxes'].append(a)
        th = cv2.resize(f, (90, 160), interpolation=cv2.INTER_AREA)
        S['thumbs'].append(th)
        S['luma'].append(float(cv2.cvtColor(th, cv2.COLOR_BGR2LAB)[..., 0].mean()))
        if i - i0 < 3 or i1 - i <= 3:
            S['edge_small'].append((i, cv2.resize(f, (360, int(360 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)))
        if a is not None and want_id(sid) and (i - i0) % 2 == 0:
            try:
                S['ids'].append((i - i0, _rec().feature(_rec().alignCrop(f, a)).astype(np.float32).ravel()))
            except cv2.error:
                pass
        i += 1
    cap.release()
    return shots


def _smooth_boxes(boxes):
    n = len(boxes)
    t = np.full((n, 7), np.nan)
    for j, a in enumerate(boxes):
        if a is None:
            continue
        lm = a[4:14].reshape(5, 2)
        t[j] = [a[0] + a[2] / 2, a[1] + a[3] / 2, a[2], a[3], (lm[3, 0] + lm[4, 0]) / 2, (lm[3, 1] + lm[4, 1]) / 2,
                np.linalg.norm(lm[4] - lm[3])]
    ok = ~np.isnan(t[:, 0])
    if n == 0 or ok.sum() < max(3, 0.3 * n):
        return None, ok
    idx = np.arange(n)
    for c in range(7):
        t[:, c] = np.interp(idx, idx[ok], t[ok, c])
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(t, size=5, axis=0, mode='nearest'), ok


def face_series(video, S):
    """Tracked crops for one shot: normalised face (128x160) + lips (96x48) for temporal metrics, and a native-size
    face crop every 4th frame for the INFO sharpness numbers."""
    tr, ok = _smooth_boxes(S['boxes'])
    if tr is None:
        return None
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, S['i0'])
    faces, lips, native = [], [], []
    for j in range(len(S['boxes'])):
        okr, f = cap.read()
        if not okr:
            break
        cx, cy, w, h, mx, my, mw = tr[j]
        faces.append(_crop(f, cx, cy + 0.05 * h, 1.1 * w, 1.1 * w * FACE_H / FACE_W, FACE_W, FACE_H))
        lips.append(_crop(f, mx, my, 1.6 * mw, 0.8 * mw, 96, 48))
        if j % 4 == 0:
            s = 0.55 * max(w, h)
            native.append(f[int(max(0, cy - s)):int(cy + s), int(max(0, cx - 0.8 * s)):int(cx + 0.8 * s)].copy())
    cap.release()
    return dict(track=tr, detected=ok, faces=faces, lips=lips, native=native)


# ------------------------------------------------------------------ metrics
def _native_sharp(c):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = g.mean() + 1.0
    b1, b2 = cv2.GaussianBlur(g, (0, 0), 1.0), cv2.GaussianBlur(g, (0, 0), 2.0)
    hf, mf = np.abs(g - b1).mean() / m, np.abs(b1 - b2).mean() / m
    hp = np.abs(g - b1)
    H, W = (hp.shape[0] // 16) * 16, (hp.shape[1] // 16) * 16
    if H == 0 or W == 0:
        return np.nan, np.nan, np.nan
    bl = hp[:H, :W].reshape(H // 16, 16, W // 16, 16).mean((1, 3))
    return np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean() / m, hf / (mf + 1e-4), float((bl < 0.2 * np.percentile(bl, 90)).mean())


def _jitter(crops):
    """per-frame p95 of the per-pixel 2nd temporal difference (lag 1) for L and a*; returns (perL, perA)."""
    L = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32) for x in crops])
    if len(L) < 5:
        return None, None
    d = np.abs(L[2:] - 2 * L[1:-1] + L[:-2])
    return (np.percentile(d[..., 0].reshape(len(d), -1), 95, axis=1),
            np.percentile(d[..., 1].reshape(len(d), -1), 95, axis=1))


def shot_metrics(S, fs, cast_refs, fps, lipsynced):
    m = dict(dur=round(S['t1'] - S['t0'], 2), frames=len(S['thumbs']))
    if cast_refs is not None:
        m['identity_samples'] = len(S['ids'])
        if S['ids']:
            from .cast import _cos
            sc = [float(_cos(e[None], cast_refs).max(axis=0).mean()) for _, e in S['ids']]
            # qa-8: the gated minimum is over a rolling median of 3 consecutive samples (every 2nd frame): one noisy
            # SFace sample (+-0.06 spread) no longer decides a FAIL (whereAB3's 0.434 was noise), while a swap that
            # lasts 2+ samples (~4 frames, the shortest a viewer can see) still drags it down
            r3 = [float(np.median(sc[max(0, k - 1):k + 2])) for k in range(len(sc))] if len(sc) >= 3 else sc
            m['identity_median'], m['identity_min'] = round(float(np.median(sc)), 3), round(float(min(r3)), 3)
            m['identity_min_raw'] = round(float(min(sc)), 3)
            m['identity_min_t'] = round(S['t0'] + S['ids'][int(np.argmin(r3))][0] / fps, 2)
    if not fs:
        return m
    m['face_detected'] = round(float(fs['detected'].mean()), 2)
    m['face_px'] = int(np.median(fs['track'][:, 3]))
    sh = np.array([_native_sharp(c) for c in fs['native'] if c.size])
    if len(sh):
        m['info_sharp'], m['info_oversharp'], m['info_smear'] = (round(float(np.nanmedian(sh[:, k])), 4) for k in range(3))
    # temporal: lips where the mouth is synthesised or the face is big, else the face crop
    src = 'lips' if (lipsynced or m['face_px'] >= 120) else 'faces'
    for key, crops in (('lip', fs['lips']), ('face', fs['faces'])):
        pL, pA = _jitter(crops)
        if pL is None:
            continue
        m[f'{key}_jitter_L'] = round(float(np.percentile(pL, 99)), 1)
        m[f'{key}_jitter_a'] = round(float(np.percentile(pA, 99)), 1)
        if (key == 'lip') == (src == 'lips'):
            m['_per'] = pL
    if '_per' in m:
        per = m['_per']
        rng = random.Random(f"{S['i0']}-{len(per)}")
        worst = int(np.argmax(per)) + 1
        med = int(np.argsort(per)[len(per) // 2]) + 1
        rnd = rng.randrange(4, max(5, len(per) - 4))
        m['_windows'] = [('worst', worst), ('median', med), ('random', rnd)]
        m['jitter_worst_t'] = round(S['t0'] + worst / fps, 2)
        m['_src'] = 'faces' if src == 'faces' else 'lips'
    lab = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[40:120, 32:96].reshape(-1, 3).mean(0) for x in fs['faces']])
    m['face_lab_start'] = [round(float(v), 1) for v in lab[:6].mean(0)]
    m['face_lab_end'] = [round(float(v), 1) for v in lab[-6:].mean(0)]
    return m


def _pcm(video):
    """Native-rate interleaved samples -> (frames, channels). Never downmix with -ac 1: ffmpeg sums the channels and
    a clean -1.3 dBTP master read as 12k 'clipped' samples (calibration 2026-11-03)."""
    pr = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate,channels',
                         '-of', 'csv=p=0', video], capture_output=True, text=True, timeout=60).stdout.strip().split(',')
    sr, ch = int(pr[0]), int(pr[1])
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', video, '-vn', '-f', 'f32le', '-'], capture_output=True, timeout=300)
    x = np.frombuffer(r.stdout, np.float32)
    return x[:len(x) // ch * ch].reshape(-1, ch), sr


def audio_metrics(video, dur):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', video, '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                       capture_output=True, text=True, timeout=300)
    I = re.findall(r'I:\s+(-?[\d.]+) LUFS', r.stderr)
    P = re.findall(r'Peak:\s+(-?[\d.]+) dBFS', r.stderr)
    out = dict(lufs=float(I[-1]) if I else None, true_peak=float(P[-1]) if P else None)
    X, sr = _pcm(video)
    out['clipped_samples'] = int((np.abs(X) >= 0.999).sum())
    out['sample_peak'] = round(float(np.abs(X).max()), 4) if X.size else None
    x = np.abs(X).max(1)                     # loudest channel, for dropouts
    hop = int(0.05 * sr)
    n = len(x) // hop
    rms = 20 * np.log10(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    t = np.arange(n) * 0.05
    quiet = (rms < -60) & (t > 0.3) & (t < dur - 2.0)   # the final fade-out is allowed to be silent
    runs, cur, start = [], 0, 0
    for j, q in enumerate(quiet):
        if q:
            if not cur:
                start = j
            cur += 1
        elif cur:
            runs.append((round(t[start], 2), round(cur * 0.05, 2))); cur = 0
    if cur:
        runs.append((round(t[start], 2), round(cur * 0.05, 2)))
    out['dropouts'] = [r_ for r_ in runs if r_[1] > GATES['dropout_s']]
    out['_pcm'] = (X.mean(1), sr)
    return out


def lipsync_offset(S, fs, pcm, fps):
    """Proxy: cross-correlate lip motion energy with the programme's loudness envelope (+ = picture late)."""
    if not fs or pcm is None:
        return None
    x, sr = pcm
    L = np.array([cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) for c in fs['lips']])
    mot = np.r_[0, np.abs(np.diff(L, axis=0)).mean((1, 2))]
    hop = sr / fps
    env = np.array([np.sqrt((x[int((S['i0'] + j) * hop):int((S['i0'] + j + 1) * hop)] ** 2).mean() + 1e-12)
                    for j in range(len(mot))])
    if len(mot) < 30 or env.std() == 0 or mot.std() == 0:
        return None
    a, b = (mot - mot.mean()) / mot.std(), (env - env.mean()) / env.std()
    lags = range(-15, 16)
    cc = [float(np.mean(a[max(0, l):len(a) + min(0, l)] * b[max(0, -l):len(b) - max(0, l)])) for l in lags]
    k = int(np.argmax(cc))
    return dict(offset_ms=round(list(lags)[k] * 1000 / fps), corr=round(cc[k], 2))


# ------------------------------------------------------------------ gates
def gates(tl, M, cuts, base_cmp, audio, pre, cast_of, lipsynced, focus):
    G, out = GATES, []

    def add(ok, sev, where, metric, what, value=None, kind='abs'):
        out.append(dict(ok=bool(ok), severity=sev, where=where, metric=metric, check=what, value=value, kind=kind))
    for r in pre:
        add(r['ok'], 'FAIL', 'preflight', 'preflight', r['check'])
    for sid, _, _ in tl:
        m = M[sid]
        if cast_of(sid) and 'identity_samples' in m:
            if m['identity_samples'] == 0:
                add(False, 'FAIL', sid, 'identity_measurable', 'identity not measurable (no face found on a cast shot)')
            else:
                add(m['identity_median'] >= G['identity_median'], 'FAIL', sid, 'identity_median',
                    f"identity median {m['identity_median']} >= {G['identity_median']}", m['identity_median'])
                add(m['identity_min'] >= G['identity_min'], 'FAIL', sid, 'identity_min',
                    f"identity min {m['identity_min']} >= {G['identity_min']} (worst at {m.get('identity_min_t')} s)", m['identity_min'])
        if sid in lipsynced and m.get('face_px', 0) >= G['min_face_px']:
            for k in ('lip_jitter_L', 'lip_jitter_a'):
                if m.get(k) is not None:
                    add(m[k] <= G[k], 'FAIL', sid, k, f"{k} {m[k]} <= {G[k]} (worst at {m.get('jitter_worst_t')} s)", m[k])
        if sid in lipsynced and m.get('lipsync') and G['lipsync_ms'] is not None:
            add(abs(m['lipsync']['offset_ms']) <= G['lipsync_ms'], 'FAIL', sid, 'lipsync', f"lip-sync offset {m['lipsync']['offset_ms']} ms")
    for c in cuts:
        if c.get('face_dE') is not None:
            add(c['face_dE'] <= G['cut_face_dE_fail'], 'FAIL', c['cut'], 'cut_face_dE', f"same-character face delta E {c['face_dE']} <= {G['cut_face_dE_fail']}", c['face_dE'])
            add(c['face_dE'] <= G['cut_face_dE'], 'WARN', c['cut'], 'cut_face_dE_warn', f"same-character face delta E {c['face_dE']} <= {G['cut_face_dE']}", c['face_dE'])
        if c.get('jump_cut'):
            add(False, 'FAIL', c['cut'], 'jump_cut', f"jump cut: same character, same framing (face size ratio {c['size_ratio']})")
        add(abs(c['luma_jump']) <= G['cut_luma_jump'], 'WARN', c['cut'], 'cut_luma', f"luma jump {c['luma_jump']} <= {G['cut_luma_jump']}", c['luma_jump'])
    # relative-to-base/anchor gates live in relative_gates()
    if audio:
        if audio.get('lufs') is not None:
            t, tol = G['lufs']
            add(abs(audio['lufs'] - t) <= tol, 'FAIL', 'audio', 'lufs', f"loudness {audio['lufs']} LUFS within {t}+-{tol}", audio['lufs'])
        if audio.get('true_peak') is not None:
            add(audio['true_peak'] <= G['true_peak'] + 0.05, 'FAIL', 'audio', 'true_peak', f"true peak {audio['true_peak']} <= {G['true_peak']} dBTP", audio['true_peak'])
        add(audio.get('clipped_samples', 0) <= G['clipped_samples'], 'FAIL', 'audio', 'clipping', f"clipped samples {audio.get('clipped_samples')}", audio.get('clipped_samples'))
        add(not audio.get('dropouts'), 'FAIL', 'audio', 'dropouts', f"silent dropouts > {G['dropout_s']} s: {audio.get('dropouts')}")
    return out


def mark_known(G, bG, base_vals):
    """Absolute failures that the base build already had are KNOWN (listed, not blocking) only if not worse."""
    bad = {(g['where'], g['metric']) for g in bG if not g['ok']}
    for g in G:
        if g['ok'] or g['severity'] != 'FAIL' or g['kind'] != 'abs' or (g['where'], g['metric']) not in bad:
            continue
        bv, v = base_vals.get((g['where'], g['metric'])), g['value']
        if bv is None or v is None:      # pass/fail gates (not measurable, jump cut): same failure as base
            if bv is None and v is None:
                g['severity'] = 'KNOWN'
            continue
        worse = (v < bv - GATES['identity_known_tol']) if g['metric'].startswith('identity') else (v > bv * 1.1)
        if not worse:
            g['severity'] = 'KNOWN'


# ------------------------------------------------------------------ artifacts
def _frame(video, i):
    c = cv2.VideoCapture(video)
    c.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = c.read()
    c.release()
    return f if ok else None


def _stamp(img, txt):
    cv2.rectangle(img, (0, 0), (8 + 9 * len(txt), 20), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def artifacts(video, tl, shots, FS, M, out, shots_wanted, fps, prefix=''):
    """Per wanted shot: 3 frames, native face crop, three timecoded 8-frame windows; cuts touching wanted shots."""
    os.makedirs(out, exist_ok=True)
    made = []
    for sid, t0, t1 in tl:
        if sid not in shots_wanted or sid not in shots:
            continue
        S, fs, m = shots[sid], FS.get(sid), M[sid]
        n = len(S['thumbs'])
        fr = []
        for k in (0.05, 0.5, 0.95):
            f = _frame(video, S['i0'] + int(k * (n - 1)))
            if f is not None:
                fr.append(_stamp(cv2.resize(f, (540, 960), interpolation=cv2.INTER_AREA), f"{sid} +{k * (t1 - t0):.2f}s"))
        if fr:
            p = os.path.join(out, f'{prefix}{sid}_frames.jpg'); cv2.imwrite(p, np.hstack(fr), [cv2.IMWRITE_JPEG_QUALITY, 90]); made.append(p)
        if fs:
            tr = fs['track']; j = len(tr) // 2
            f = _frame(video, S['i0'] + j)
            if f is not None:
                cx, cy, w, h = tr[j][:4]; s = 1.6 * max(w, h)
                c = f[int(max(0, cy - s / 2)):int(cy + s / 2), int(max(0, cx - s / 2)):int(cx + s / 2)].copy()
                p = os.path.join(out, f'{prefix}{sid}_face_100pct.jpg'); cv2.imwrite(p, _stamp(c, f"{sid} +{j / fps:.2f}s"), [cv2.IMWRITE_JPEG_QUALITY, 95]); made.append(p)
            src = fs[m.get('_src', 'faces')]
            sc = 3 if m.get('_src') == 'lips' else 2
            for wn, (label, jc) in enumerate(m.get('_windows', [])):
                seq = src[max(0, jc - 4):max(0, jc - 4) + 8]
                tiles = [_stamp(cv2.resize(x, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC), f"+{(max(0, jc - 4) + q) / fps:.3f}s")
                         for q, x in enumerate(seq)]
                if tiles:
                    p = os.path.join(out, f'{prefix}{sid}_8frames_{wn + 1}.jpg'); cv2.imwrite(p, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92]); made.append(p)
        else:   # no face: 6-frame full-frame windows; with a base, at the frames that CHANGED most (calibration:
            # fixed positions missed a light's dying moment), else spread over the shot
            rng = random.Random(f'{sid}-{n}')
            starts = m.get('_change_starts') or [n // 6, n // 2, rng.randrange(0, max(1, n - 6))]
            for wn, st in enumerate(starts):
                tiles = []
                for q in range(6):
                    f = _frame(video, S['i0'] + min(n - 1, st + q))
                    if f is not None:
                        tiles.append(_stamp(cv2.resize(f, (270, 480), interpolation=cv2.INTER_AREA), f"+{(st + q) / fps:.3f}s"))
                if tiles:
                    p = os.path.join(out, f'{prefix}{sid}_6frames_{wn + 1}.jpg'); cv2.imwrite(p, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88]); made.append(p)
    # cuts named by position relative to the focus shot (never by the neighbour's id: after a reorder the neighbour
    # names alone revealed which cut was new - v3 audit)
    # both builds produce the SAME file set even after a reorder (v4 audit: differing cut file sets leaked the new cut)
    ids = [x for x, _, _ in tl]
    for f in sorted(shots_wanted):
        if f not in ids:
            continue
        i = ids.index(f)
        for name, (a, b) in ((f'cut_into_{f}', (ids[i - 1] if i > 0 else None, f)),
                             (f'cut_out_of_{f}', (f, ids[i + 1] if i + 1 < len(ids) else None))):
            if a is None or b is None:
                continue
            L = [x for _, x in shots[a]['edge_small'][-2:]] + [x for _, x in shots[b]['edge_small'][:2]]
            if len(L) == 4:
                L = [_stamp(x.copy(), 'before cut' if q < 2 else 'after cut') for q, x in enumerate(L)]
                p = os.path.join(out, f'{prefix}{name}.jpg'); cv2.imwrite(p, np.hstack(L), [cv2.IMWRITE_JPEG_QUALITY, 90]); made.append(p)
    return made


CRITIC_PROMPT = """You are a blind quality reviewer for {what}. You know nothing about how these images were made.
Read every image listed below with the Read tool. Your only job is to find defects a viewer would see on a phone.

Images (folder {folder}):
{images}

There are two cuts of the same shots, A_ and B_. Review each cut, and compare them at the same places. Some shots
may differ between the cuts on purpose: a difference between the cuts is not a defect by itself. Judge what a viewer
sees in each cut.

Intended look of this series (these are NOT defects): {style_note}

Image types: *_frames = start/middle/end of a shot; *_face_100pct = native-resolution face crop; *_8frames_N and
*_6frames_N = CONSECUTIVE frames (timecode on each tile): anything that jumps, changes colour or shape for one frame
is flicker; cut_into_X / cut_out_of_X = last two frames before a cut, then the first two after it.

Rules:
1. List every defect with: cut (A or B), file, position, timecode (strips), class, sev (2 = a viewer notices it on
   the first watch on a phone; 1 = only on a second look or a paused frame), worse_here ("yes" if the same
   file/position/timecode in the OTHER cut is clean or clearly milder, else "no"; matching files share a name after
   the A_/B_ prefix and strips show the same shot-relative timecodes, +s from the shot start, in both cuts; if the other cut has no file of that name, "yes").
   Classes, with one example each:
   - smear: skin or features melt, blur into blobs, or smudge (an eye that turns into a dark smear)
   - flicker: something changes for one or two frames in a strip (lip colour pops on one tile)
   - blur: the shot is soft compared with what the framing needs (a close-up with no skin texture)
   - identity: a character looks like a different person between images (different face shape across a cut)
   - costume/set: clothes, props or the set change between images of the same moment (stains move, a lamp jumps)
   - cut: continuity across a cut (eyeline flips, same framing repeated = jump cut, light jumps within one location)
   - compositing: something looks pasted, has a halo or mask edge, or light from the wrong direction
   - grade: crushed, washed out, or a colour cast that the style note does not explain
   - text/anatomy: garbled text, extra or missing fingers, impossible anatomy
   - tone: the shot reads as a different genre than the intended look (horror or fantasy effects in a grounded crime
     look), or something happens with no visible in-world cause (a light dies in one place only, for no reason)
2. A defect without a cut, file and position does not count. No praise, no story notes.
3. Do not open any other folder. Do not edit any file except your answer file.
4. Write your answer as JSON with the Write tool to {answer} , replacing <model> with opus, sonnet or haiku:
   {{"defects": [{{"cut": "A", "file": "...", "position": "...", "timecode": "...", "class": "flicker", "sev": 2,
     "worse_here": "yes"}}],
    "class_scores": {{"A": {{"smear": 0, "flicker": 0, "blur": 0, "identity": 0, "costume/set": 0, "cut": 0,
     "compositing": 0, "grade": 0, "text/anatomy": 0, "tone": 0}}, "B": {{"smear": 0}}}}}}
   class_scores = the highest sev you listed for that class in that cut (0 if none). Then reply "done"."""
CRITIC_PROMPT_HASH = hashlib.sha1(CRITIC_PROMPT.encode()).hexdigest()[:10]
CLASSES = ('smear', 'flicker', 'blur', 'identity', 'costume/set', 'cut', 'compositing', 'grade', 'text/anatomy', 'tone')

# qa-8 brief read: the NEW cut alone (no A/B, nothing to give away), with the owner's brief. Why: the A/B prompt used
# to carry the brief, which told critics which cut was new and to "judge execution, not existence"; both held-out SHIP
# misses (q12b, whereAB "looks bad") were changes the critics rated as requested instead of asking whether they look
# real. The brief now lives only here, and the A/B read never sees it.
BRIEF_PROMPT = """You are a blind quality reviewer for {what}. You know nothing about how these images were made.
Read every image listed below with the Read tool.

Images (folder {folder}):
{images}

Intended look of this series (not defects in themselves): {style_note}

The owner asked for these shots to show the following:
{aspects}

Image types: *_frames = start/middle/end of a shot; *_face_100pct = native-resolution face crop; *_8frames_N and
*_6frames_N = consecutive frames (timecode on each tile); cut_into_X / cut_out_of_X = two frames before a cut, then
two after it.

Answer two questions, from what the images show (not from what was probably intended):
1. For each numbered request: is it delivered? "yes" (clearly, as asked), "partly" (only some of it, or it is there
   but undercut by something else in the image), "no". Name the file that shows it.
2. For each shot: would a viewer on a phone believe it is real filmed footage in this style?
   "yes"; "mostly" (something looks artificial on a second look or a paused frame); "no" (on the first watch
   something looks pasted in, cut out, painted, CGI-like, melted, or lit differently from its surroundings).
   For "mostly" and "no" name the file, the position and what gives it away.
Rules: do not open any other folder; do not edit any file except your answer file; no praise, no story notes.
Write your answer as JSON with the Write tool to {answer} , replacing <model> with opus, sonnet or haiku:
  {{"brief": [{{"n": 1, "delivered": "yes", "file": "...", "why": "..."}}],
   "real": [{{"shot": "...", "verdict": "mostly", "file": "...", "position": "...", "why": "..."}}]}}
Then reply "done"."""
BRIEF_PROMPT_HASH = hashlib.sha1(BRIEF_PROMPT.encode()).hexdigest()[:10]
READS = 3   # one prompt + one empty answer folder per read: a third reader never sees the first two answers


def _aspects(intent):
    """'shot: a; b, shot2: c' -> numbered aspect lines (the owner's brief, split on ';')."""
    return [a.strip() for a in re.split(r'[;\n]', intent or '') if a.strip()]


def _write_prompts(pk, template, fmt, kind):
    """One prompt file + one answer folder per read (PROMPT_1..3 / BRIEF_1..3); returns the answer folders."""
    dirs = []
    for r in range(1, READS + 1):
        ans = os.path.join(_ROOT, 'cache', 'qa_answers', os.path.basename(pk), f'{kind}{r}')
        os.makedirs(ans, exist_ok=True)
        txt = template.format(answer=os.path.join(ans, f'answer_<model>_{kind}{r}.json').replace('\\', '/'), **fmt)
        open(os.path.join(pk, f"{'PROMPT' if kind == 'ab' else 'BRIEF'}_{r}.txt"), 'w', encoding='utf-8').write(txt)
        dirs.append(ans)
    return dirs


def critic_packet(new_made, base_made, out, style_note='', intent='', what='a short film', seed=None):
    """Random A/B names; the packet lives in cache/qa_packets/<id> (a different tree from the metrics and the key).
    With an intent, a second packet (new cut only, neutral names) carries the brief read."""
    rng = random.Random(seed)
    tag = hashlib.sha1(f'{out}{time.time()}{seed}'.encode()).hexdigest()[:10]
    pk = os.path.join(_ROOT, 'cache', 'qa_packets', tag)
    os.makedirs(pk, exist_ok=True)
    flip = rng.random() < 0.5
    key = dict(A='base' if flip else 'new', B='new' if flip else 'base', packet=pk, prompt_hash=CRITIC_PROMPT_HASH,
               version=QA_VERSION)
    names = []
    for label, lst in (('A', base_made if flip else new_made), ('B', new_made if flip else base_made)):
        for p in lst:
            n = f'{label}_' + os.path.basename(p).replace('base_', '')
            shutil.copyfile(p, os.path.join(pk, n)); names.append(n)
    for n in names:   # one timestamp for every file: copy order (base first) must not reveal which cut is new
        os.utime(os.path.join(pk, n), (1700000000, 1700000000))
    fmt = dict(what=what, folder=pk.replace('\\', '/'), images='\n'.join(f'- {n}' for n in sorted(names)),
               style_note=style_note or 'none given')
    key['answers'] = _write_prompts(pk, CRITIC_PROMPT, fmt, 'ab')
    asp = _aspects(intent)
    if asp and new_made:
        btag = hashlib.sha1(f'{tag}brief{time.time()}'.encode()).hexdigest()[:10]
        bk = os.path.join(_ROOT, 'cache', 'qa_packets', btag)
        os.makedirs(bk, exist_ok=True)
        bn = []
        for p in new_made:
            n = os.path.basename(p)
            shutil.copyfile(p, os.path.join(bk, n)); os.utime(os.path.join(bk, n), (1700000000, 1700000000)); bn.append(n)
        bfmt = dict(what=what, folder=bk.replace('\\', '/'), images='\n'.join(f'- {n}' for n in sorted(bn)),
                    style_note=style_note or 'none given', aspects='\n'.join(f'{i + 1}. {a}' for i, a in enumerate(asp)))
        key.update(brief_packet=bk, brief_answers=_write_prompts(bk, BRIEF_PROMPT, bfmt, 'brief'),
                   brief_prompt_hash=BRIEF_PROMPT_HASH, aspects=asp)
    json.dump(key, open(os.path.join(out, 'ab_key.json'), 'w'), indent=1)
    return pk, key


# Classes judged on the NEW cut alone (worse_here ignored), with their blocking severity. Why: relative scoring only
# counts defects worse than the base, so a defect the approved base ALSO has can never block. q12b (owner REJECTED,
# "more horror than neo noir"): both critics rated the new cut's tone 1, the base's tone 2 -> nothing counted -> SHIP.
# Threshold 1 (not 2): the owner rejected a cut both critics rated tone 1. In-sample on q12b and endB only.
ABSOLUTE_CLASSES = {'tone': 1}
# qa-8: the owner rejects on realism ("pasted", "looks bad", "doesn't look like her"), so a severity-1 defect in these
# classes costs 1.0 (others 0.5) and SHIP needs >= 8.0: two realism flaws, or one plus two minor ones, block.
# Before qa-8 the number never decided anything (it only blocked at 7 sev-1 classes; audit 2026-11-05 Q2).
# The weights are a judgement, not fitted: on the stored answers they move whereAB (owner rejected) from 9.0 to
# below 8, but those answers came from the leaky qa-7 prompt, so that is in-sample, not evidence.
REALISM_CLASSES = ('smear', 'identity', 'compositing', 'tone')
SHIP_AT = 8.0


def _block_at(c):
    return ABSOLUTE_CLASSES.get(c, 2)


def score_critics(new_defects, harness_passed=()):
    """Derived score (docs/review_workflow.md step 4). new_defects = merged [(class, sev, shot)] of the NEW cut with
    worse_here == yes (ABSOLUTE_CLASSES: any new-cut defect).
    score = 10 - 2.5 per blocking class - 1.0 per other realism class at sev 1 - 0.5 per other class at sev 1.
    SHIP needs no blocking class AND score >= SHIP_AT.
    - UNCERTAIN (owner decides, not blocking): sev-2 flicker on a lip-synced shot whose calibrated jitter gate passed
      (strips cannot see sub-visible jitter and flagged the approved q6c's accepted snap)"""
    worse, uncertain = {}, []
    for c, sv, shot in new_defects:
        if sv >= 2 and c == 'flicker' and shot in harness_passed:
            uncertain.append((shot, c)); continue
        worse[c] = max(worse.get(c, 0), sv)
    blocking = sorted(c for c, v in worse.items() if v >= _block_at(c))   # sev 2, or an absolute class at its level
    minor = [c for c, v in worse.items() if c not in blocking and v >= 1]
    n2, n1r = len(blocking), sum(1 for c in minor if c in REALISM_CLASSES)
    n1 = len(minor) - n1r
    rel = max(1.0, 10 - 2.5 * n2 - 1.0 * n1r - 0.5 * n1)
    return dict(relative=rel, counted=worse, sev2=n2, sev1_realism=n1r, sev1=n1, blocking=blocking, taste=[],
                uncertain=uncertain, ship=(n2 == 0 and rel >= SHIP_AT))


def _shot_of(fname):
    f = re.sub(r'^[AB]_', '', os.path.basename(fname or ''))
    m = re.match(r'cut_(?:into|out_of)_(.+?)\.jpg$', f)
    if m:
        return m.group(1)
    return re.split(r'_(?:frames|face_100pct|8frames_\d+|6frames_\d+)', f)[0]


def _cls(c):
    c = str(c or '').lower()
    return {'costume': 'costume/set', 'set': 'costume/set', 'text': 'text/anatomy', 'anatomy': 'text/anatomy'}.get(c, c)


def merge_answers(answers, key):
    """Per class, for the NEW cut and worse_here=yes (ABSOLUTE_CLASSES: any new-cut defect, worse_here ignored): one
    value per critic (its worst sev, 0 if none), merged by MAX. A class whose critics sit on opposite sides of its
    blocking level (2, or the ABSOLUTE_CLASSES level) is DISPUTED: with 2 reads it needs a third read, with 3+ reads it
    takes the median. qa-8: the median applies ONLY to the disputed class (qa-7 switched every class to the median
    once a third read existed, which erased defects that 2 of 3 readers saw; audit Q4).
    Returns (rows [(class, sev, shot)], disputes [class], per_critic {class: [values]})."""
    new = 'A' if key['A'] == 'new' else 'B'
    per = {c: [] for c in CLASSES}
    detail = {c: [] for c in CLASSES}
    for ans in answers:
        best = {c: 0 for c in CLASSES}
        for d in ans.get('defects', []):
            c = _cls(d.get('class'))
            if str(d.get('cut')).upper() != new or c not in per or (
                    c not in ABSOLUTE_CLASSES and str(d.get('worse_here')).lower() != 'yes'):
                continue
            sv = int(d.get('sev', 0))
            best[c] = max(best[c], sv)
            detail[c].append((sv, _shot_of(d.get('file')), d.get('timecode') or d.get('position')))
        for c in CLASSES:
            per[c].append(best[c])
    rows, disputes = [], []
    for c, vals in per.items():
        if not vals or max(vals) == 0:
            continue
        split = max(vals) >= _block_at(c) > min(vals)
        if split and len(vals) >= 3:
            v = int(np.median(vals))
        else:
            v = max(vals)
            if split:
                disputes.append(c)
        if v == 0:
            continue
        top = [x for x in detail[c] if x[0] >= v] or detail[c]
        for shot in sorted({sh for _, sh, _ in top}) or ['?']:
            rows.append((c, v, shot))
    return rows, disputes, per


_BRIEF_V = {'yes': 2, 'partly': 1, 'no': 0}
_REAL_V = {'yes': 0, 'mostly': 1, 'no': 2}


def merge_brief(answers, aspects):
    """Brief read (new cut only). Per request: delivered yes/partly/no; per shot: real yes/mostly/no.
    Two reads -> the worst; a split across the blocking line ('no' against anything else, for delivery and realism)
    is DISPUTED: a third read decides by median. A request no reader answered is 'unanswered' and blocks (fail
    closed: re-read). 'partly' never blocks and never ships: it makes the prediction CHECK (owner's call), because
    on 8 past cases critics said 'partly' about approved builds (q6c, q11) as often as about rejected ones."""
    n = len(aspects)
    dv = {i: [] for i in range(1, n + 1)}
    per_read, why = [], {}
    for a in answers:
        got = {}
        for b in a.get('brief', []) or []:
            try:
                i = int(b.get('n'))
            except (TypeError, ValueError):
                continue
            v = str(b.get('delivered', '')).lower()
            if i in dv and v in _BRIEF_V and i not in got:
                got[i] = _BRIEF_V[v]
                if v != 'yes':
                    why.setdefault(f'request {i}', []).append(f"{v}: {b.get('why', '')} ({b.get('file', '')})")
        for i, v in got.items():
            dv[i].append(v)
        mine = {}
        for r in a.get('real', []) or []:
            s, v = (str(r['shot']) if r.get('shot') else _shot_of(r.get('file'))), str(r.get('verdict', '')).lower()
            if v in _REAL_V:
                mine[s] = max(mine.get(s, 0), _REAL_V[v])
                if v != 'yes':
                    why.setdefault(f'real {s}', []).append(f"{v}: {r.get('why', '')} ({r.get('file', '')} {r.get('position', '')})")
        per_read.append(mine)
    # a reader who did not list a shot found nothing artificial in it (0)
    rv = {s: [m.get(s, 0) for m in per_read] for s in sorted({s for m in per_read for s in m})}
    delivered, real, disputes, unanswered = {}, {}, [], []
    for i, vals in dv.items():
        if not vals:
            unanswered.append(i); continue
        split = max(vals) > 0 == min(vals)   # only 'no' blocks, so only a 'no' against a yes/partly needs a third read
        if split and len(vals) >= 3:
            delivered[i] = int(np.median(vals))
        else:
            delivered[i] = min(vals)
            if split:
                disputes.append(f'request {i}')
    for s, vals in rv.items():
        split = max(vals) == 2 > min(vals)
        if split and len(vals) >= 3:
            real[s] = int(np.median(vals))
        else:
            real[s] = max(vals)
            if split:
                disputes.append(f'real {s}')
    inv = {v: k for k, v in _BRIEF_V.items()}
    return dict(delivered={i: inv[v] for i, v in delivered.items()}, real={s: {v: k for k, v in _REAL_V.items()}[x] for s, x in real.items()},
                unanswered=unanswered, disputes=disputes, why=why)


def _chain_append(path, rec):
    """Append-only, hash-chained prediction log (tamper-evident: every line carries the hash of the previous line)."""
    prev = '0' * 64
    if os.path.exists(path):
        lines = [l for l in open(path, encoding='utf-8').read().splitlines() if l.strip()]
        if lines:
            prev = hashlib.sha256(lines[-1].encode()).hexdigest()
    rec = dict(rec, prev=prev, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    line = json.dumps(rec, sort_keys=True, default=str)
    open(path, 'a', encoding='utf-8').write(line + '\n')
    return hashlib.sha256(line.encode()).hexdigest()


def verify_log(path):
    """Internal links only. A rewritten-and-rechained or truncated log still verifies, so the HEAD hash is also written
    into qa/verdicts.log with every owner verdict (and quoted to the owner): compare heads to detect a rewrite."""
    prev = '0' * 64
    for i, l in enumerate(x for x in open(path, encoding='utf-8').read().splitlines() if x.strip()):
        if json.loads(l).get('prev') != prev:
            return False, i, None
        prev = hashlib.sha256(l.encode()).hexdigest()
    return True, None, prev


def _scoring_hash():
    """Hash of the code and constants that turn critic answers into a prediction (logged with every prediction)."""
    import inspect
    src = ''.join(inspect.getsource(f) for f in (score_critics, merge_answers, merge_brief, score, _block_at))
    src += repr((ABSOLUTE_CLASSES, REALISM_CLASSES, SHIP_AT, CLASSES, _BRIEF_V, _REAL_V))
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def _held_out_done(logpath, video_sha):
    """A held-out prediction is logged once per video; only a NEEDS THIRD READ / INCOMPLETE line may be followed."""
    if not os.path.exists(logpath):
        return None
    last = None
    for l in open(logpath, encoding='utf-8').read().splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get('video_sha') == video_sha:
                last = r.get('prediction', '')
    if last is None or last.startswith('NEEDS THIRD READ') or last.startswith('INCOMPLETE'):
        return None
    return last


def _file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def score(out, answer_paths, item='', log=print, record=True, logfile='predictions.log', brief_paths=()):
    """Merge critic answers in code, route, write report.md, append the prediction to qa/predictions.log.
    answer_paths = A/B reads; brief_paths = brief reads (new cut only; required when the packet has a brief)."""
    q = json.load(open(os.path.join(out, 'qa.json'), encoding='utf-8'))
    key = json.load(open(os.path.join(out, 'ab_key.json'), encoding='utf-8'))
    held_out = record and logfile == 'predictions.log'
    if held_out and q.get('version') != QA_VERSION:   # qa-8: the frozen version is enforced, not just recorded
        raise SystemExit(f"qa.json was made by {q.get('version')}, this is {QA_VERSION}: re-run the harness "
                         f"(or score with --calibration, which is not a held-out prediction)")
    vsha = _file_sha(q['video']) if record else None
    if held_out:
        prev = _held_out_done(os.path.join(_ROOT, 'qa', logfile), vsha)
        if prev:
            raise SystemExit(f"a held-out prediction for this video is already logged ({prev}); "
                             f"re-scores go to --calibration")
    answers = [json.load(open(a, encoding='utf-8')) for a in answer_paths]
    rows, disputes, per = merge_answers(answers, key)
    ran = {}
    for g in q['gates']:
        if g['metric'] in ('lip_jitter_L', 'lip_jitter_a'):
            ran.setdefault(g['where'], []).append(g['ok'])
    passed = {s for s, oks in ran.items() if oks and all(oks)}   # gate ran AND passed (not merely absent)
    sc = score_critics(rows, passed)
    routed = {c for _, c in sc['uncertain']}
    disputes = [c for c in disputes if c not in routed]   # already the owner's call: no third read needed
    fails = [g for g in q['gates'] if not g['ok'] and g['severity'] == 'FAIL']
    near = [g for g in q['gates'] if g['severity'] == 'NEAR']
    known = [g for g in q['gates'] if not g['ok'] and g['severity'] == 'KNOWN']
    aspects = key.get('aspects') or []
    br = merge_brief([json.load(open(a, encoding='utf-8')) for a in brief_paths], aspects) if aspects and brief_paths else None
    brief_bad, partly = [], []
    if br:
        partly = [f"request {i} partly ('{aspects[i - 1]}')" for i, v in br['delivered'].items() if v == 'partly']
        brief_bad = [f'request {i} {v}' for i, v in br['delivered'].items() if v == 'no'] + \
                    [f'request {i} unanswered' for i in br['unanswered']] + \
                    [f'{s} does not look real' for s, v in br['real'].items() if v == 'no']
        brief_bad = [b for b in brief_bad if not any(b.startswith(d.replace('real ', '')) or b.startswith(d) for d in br['disputes'])]
    # a disputed class is undecided: test the verdict with it at the LOWER critic's value; if that would ship, the
    # third read decides (never the harsher or the milder critic alone)
    low = [(c, (min(per[c]) if c in disputes else v), sh) for c, v, sh in rows]
    sc_low = score_critics(low, passed)
    all_disputes = disputes + (br['disputes'] if br else [])
    dr = q.get('director') or {}
    director_err = [f"{s}: {d['error']}" for s, d in dr.items() if d.get('error')]
    director_ok = bool(dr) and not director_err
    director_worse = [(s, d) for s, d in dr.items() if d.get('verdict') == 'worse']
    if q['verdict'] != 'PASS' or not sc_low['ship'] or brief_bad:
        pred = 'BLOCK'
    elif aspects and not brief_paths:
        pred = 'INCOMPLETE (brief read missing)'
    elif all_disputes:
        pred = 'NEEDS THIRD READ'
    elif not sc['ship']:
        pred = 'BLOCK'
    elif partly and not director_ok:   # qa-10: the director votes on exactly this case: it must have run
        pred = 'INCOMPLETE (director missing: ' + '; '.join(director_err or ['not run']) + ')'
    elif partly and director_worse:      # qa-10 (owner: "let it vote"): the director breaks the tie
        pred = 'BLOCK (director: ' + ', '.join(f"{s} worse ({d['delta']:+.3f})" for s, d in director_worse) + \
               '; critics: ' + '; '.join(partly) + ')'
    elif partly:   # qa-8: the critics cannot tell; say so instead of guessing (reasons are in report line 5)
        pred = 'CHECK (your call: ' + '; '.join(partly) + ')'
    else:
        checks = [f'{c}@{s}' for s, c in sc['uncertain']] + ([f'{s} looks real only "mostly"' for s, v in br['real'].items() if v == 'mostly'] if br else [])
        pred = 'SHIP' + (' (owner checks: ' + ', '.join(checks) + (', near-gate items' if near else '') + ')' if (checks or near) else '')
    res = dict(prediction=pred, harness=q['verdict'], critics=sc, director=dr, disputes=all_disputes, per_critic=per, brief=br,
               answers=[os.path.abspath(a) for a in answer_paths], brief_answers=[os.path.abspath(a) for a in brief_paths],
               prompt_hash=key.get('prompt_hash'), brief_prompt_hash=key.get('brief_prompt_hash'), scoring=_scoring_hash())
    json.dump(res, open(os.path.join(out, 'critics.json'), 'w'), indent=1, default=str)
    q['prediction'] = pred
    json.dump(q, open(os.path.join(out, 'qa.json'), 'w'), indent=1, default=str)

    def kn(g):   # every KNOWN gate by name and value; the owner's 0.60 identity rule is called out, never just counted
        flag = ' (below your 0.60 rule; accepted in the approved build)' if g['metric'] == 'identity_median' else ''
        return f"{g['where']} {g['metric']} {g['value']}{flag}"
    L = [f"1. Item: {item or ', '.join(q['focus'])}",
         f"2. Harness {q['verdict']}: " + ('; '.join(f"{g['where']} {g['check']}" for g in fails) or 'no fails')
         + (" | KNOWN (the approved build had it, not worse): " + '; '.join(kn(g) for g in known) if known else '')
         + (" | NEAR gate: " + '; '.join(f"{g['where']} {g['check']}" for g in near) if near else ''),
         "3. Critic defects in the new cut, worse than the approved cut (tone: judged on its own): "
         + (', '.join(f"{c} sev{v} @ {sh}" for c, v, sh in rows) or 'none')
         + (f" | UNCERTAIN (your call): {sc['uncertain']}" if sc['uncertain'] else '')
         + (f" | DISPUTE, third read needed: {all_disputes}" if all_disputes else ''),
         f"4. Score {sc['relative']} = 10 - 2.5 x {sc['sev2']} blocking {sc['blocking']} - 1.0 x {sc['sev1_realism']} "
         f"realism sev1 - 0.5 x {sc['sev1']} other sev1; SHIP needs no blocking class and >= {SHIP_AT}. Per critic "
         + str({c: v for c, v in per.items() if any(v)}),
         "5. Brief read (new cut alone): " + (
             ('; '.join(f"request {i} '{aspects[i - 1]}': {v}" for i, v in br['delivered'].items())
              + ' | real: ' + (', '.join(f'{s} {v}' for s, v in br['real'].items()) or 'all yes')
              + (f" | why: {br['why']}" if br['why'] else '')) if br else ('MISSING' if aspects else 'no brief given'))
         + " | Not covered: lip-sync timing, sound by ear, story/acting, non-focus shots beyond pixel identity",
         "5b. Director (Qwen3.5-9B, one take at a time; votes only on CHECK): " + (
             ' | '.join((f"{s}: ERROR {d['error']}" if d.get('error') else
                         f"{s}: new {d['verdict']} ({d['delta']:+.3f}; " + ', '.join(f"{k} {v:+.2f}" for k, v in d['change'].items())
                         + f") notes: {d.get('notes') or '-'}") for s, d in dr.items()) or 'not run'),
         "6. Attachments: worst crop or strip, the A/B, a preview with sound",
         f"7. Predicted verdict{(' (logged before you see it)' if held_out else ' (calibration re-score, not held-out)') if record else ' (re-score, not logged)'}: {pred}",
         "8. Waiting for your OK"]
    open(os.path.join(out, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    if record:
        res['log_hash'] = _chain_append(os.path.join(_ROOT, 'qa', logfile),
                                        dict(video=q['video'], video_sha=vsha, base=q.get('base'),
                                             focus=q['focus'], harness=q['verdict'], prediction=pred,
                                             relative=sc['relative'], uncertain=sc['uncertain'], brief=br and dict(
                                                 delivered=br['delivered'], real=br['real']),
                                             disputes=all_disputes, prompt_hash=key.get('prompt_hash'),
                                             brief_prompt_hash=key.get('brief_prompt_hash'), version=q.get('version'),
                                             code=q.get('code'), scoring=res['scoring'], reads=[len(answer_paths), len(brief_paths)],
                                             item=item))
    for l in L:
        log(l)
    return res


# ------------------------------------------------------------------ analysis
def analyse(spec, video, tl, fps, wanted, lipsynced):
    from . import cast as C
    sheet = C.sheet(spec)
    by_id = {s['id']: s for s in spec['shots']}

    def cast_of(sid):
        s = by_id.get(sid)
        try:
            return C.shot_cast(spec, s, sheet) if s else []
        except ValueError:
            return []
    refs = {k: C.ref_feats(c) for k, c in sheet.items()}
    shots = decode(video, tl, fps, lambda sid: bool(cast_of(sid)))
    FS, M, jobs = {}, {}, []
    for sid, _, _ in tl:
        S = shots[sid]
        ch = cast_of(sid)
        rf = refs.get(ch[0]) if ch else None
        fs = face_series(video, S) if sid in wanted else None
        FS[sid] = fs
        M[sid] = shot_metrics(S, fs, rf, fps, sid in lipsynced)
        if fs:
            jobs.append((sid, dict(video=os.path.abspath(video), i0=S['i0'], track=fs['track'][:, :4].round(2).tolist())))
        th = np.array([cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in S['thumbs']])
        if len(th) > 4:   # whole-frame temporal jitter (backgrounds, non-face motion), on thumbnails
            d = np.abs(th[2:] - 2 * th[1:-1] + th[:-2]).reshape(len(th) - 2, -1)
            M[sid]['frame_jitter'] = round(float(np.percentile(np.percentile(d, 95, axis=1), 99)), 2)
    _flow(jobs, M)
    return shots, FS, M, cast_of


def _flow(jobs, M):
    """qa-9: RAFT motion-compensated face/mouth residual (vf/gen/flowqc.py, gen venv). Per shot: p99 over frames of
    the per-frame p99 residual. A failed run is recorded and fails the shot's gates closed, never passes it."""
    if not jobs:
        return
    try:
        from .gen.bridge import run_gen
        res = run_gen('flowqc', [j for _, j in jobs], persistent=False, check=False)
    except Exception as e:
        res = [dict(ok=False, error=f'{type(e).__name__}: {e}')] * len(jobs)
    for (sid, _), r in zip(jobs, res):
        if r.get('ok'):
            M[sid]['flow_face'] = round(float(np.percentile(r['face'], 99)), 2)
            M[sid]['flow_mouth'] = round(float(np.percentile(r['mouth'], 99)), 2)
            M[sid]['flow_version'] = r.get('version')
        else:
            M[sid]['flow_error'] = str(r.get('error'))[:200]


_CODE_HASH = hashlib.sha1(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:10]


def _deps_hash(spec, lipsynced):
    """qa-8 (audit Q7): what else decides the cached metrics: the cast/identity code, the cast reference images
    (paths + size + mtime) and which shots are lip-synced. Before, a new cast ref left the base's identity stale."""
    from . import cast as C
    h = hashlib.sha1()
    for m in ('cast.py', 'identity.py', os.path.join('gen', 'flowqc.py')):
        h.update(open(os.path.join(_ROOT, 'vf', m), 'rb').read())
    try:
        for k, c in sorted(C.sheet(spec).items()):
            for r in c.get('refs', []) or []:
                p = r if os.path.isabs(r) else os.path.join(spec.get('root', ''), r)
                st = os.stat(p) if os.path.exists(p) else None
                h.update(f"{k}|{r}|{st and st.st_size}|{st and int(st.st_mtime)}".encode())
    except Exception as e:   # never silently share a key: an unreadable sheet gets its own key
        h.update(repr(e).encode())
    h.update(json.dumps(sorted(lipsynced)).encode())
    return h.hexdigest()[:12]


def _cache_key(video, tl, wanted, deps=''):
    st = os.stat(video)
    return hashlib.sha1(json.dumps([os.path.abspath(video), st.st_size, st.st_mtime, QA_VERSION, _CODE_HASH, tl,
                                    sorted(wanted), deps]).encode()).hexdigest()[:20]


def analyse_cached(spec, video, tl, fps, wanted, lipsynced):
    os.makedirs(CACHE, exist_ok=True)
    # pickle is safe here: only this function writes cache/qa/*.pkl (local, never downloaded or shared); the key
    # includes this file's hash, so a code change never reuses stale results
    p = os.path.join(CACHE, _cache_key(video, tl, wanted, _deps_hash(spec, lipsynced)) + '.pkl')
    if os.path.exists(p):
        try:
            shots, FS, M = pickle.load(open(p, 'rb'))
            return shots, FS, M, True
        except Exception:
            pass
    shots, FS, M, _ = analyse(spec, video, tl, fps, wanted, lipsynced)
    pickle.dump((shots, FS, M), open(p, 'wb'))
    return shots, FS, M, False


def cut_metrics(tl, shots, M, cast_of):
    cuts = []
    for (a, _, _), (b, _, _) in zip(tl, tl[1:]):
        A, B = shots[a], shots[b]
        c = dict(cut=f'{a}->{b}', luma_jump=round(float(np.mean(B['luma'][:3]) - np.mean(A['luma'][-3:])), 1))
        same = set(cast_of(a)) & set(cast_of(b))
        if same and M[a].get('face_lab_end') and M[b].get('face_lab_start'):
            c['face_dE'] = round(float(np.linalg.norm(np.subtract(M[a]['face_lab_end'], M[b]['face_lab_start']) * [100 / 255, 1, 1])), 1)
            ba = next((x for x in reversed(A['boxes']) if x is not None), None)
            bb = next((x for x in B['boxes'] if x is not None), None)
            if ba is not None and bb is not None:
                r = float(bb[3] / ba[3]); c['size_ratio'] = round(r, 2)
                shift = abs((bb[0] + bb[2] / 2) - (ba[0] + ba[2] / 2)) / ba[2]
                c['jump_cut'] = bool(0.75 < r < 1.33 and shift < 0.8)
        cuts.append(c)
    return cuts


def _block_max(a, b, k=10):
    d = np.abs(a.astype(np.int16) - b).mean(2)
    H, W = (d.shape[0] // k) * k, (d.shape[1] // k) * k
    return float(d[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3)).max())


def compare_base(tl, shots, M, base_shots, base_M, focus, audio=None):
    out = {}
    for sid, t0, t1 in tl:
        if sid not in base_shots:
            continue
        A, B = shots[sid]['thumbs'], base_shots[sid]['thumbs']
        n = min(len(A), len(B))
        per = [np.abs(A[k].astype(np.int16) - B[k]).mean() for k in range(n)]
        blk = [_block_max(A[k], B[k]) for k in range(n)]
        r = dict(diff=round(float(np.mean(per)), 2) if n else 99.0, frame_max=round(float(max(per)), 2) if n else 99.0,
                 block_max=round(float(max(blk)), 2) if n else 99.0, frames=[len(A), len(B)])
        r['changed'] = (len(A) != len(B)) or r['block_max'] > GATES['unchanged_block']
        r['unchanged_expected'] = sid not in focus
        a, b = M[sid], base_M.get(sid, {})
        r['lost'] = [k for k in ('identity_median', 'face_px', 'lip_jitter_L', 'flow_face', 'flow_mouth') if b.get(k) is not None and a.get(k) is None]
        r['flow_error'] = a.get('flow_error') or b.get('flow_error')
        d = {}
        if a.get('identity_median') is not None and b.get('identity_median') is not None:
            d['identity'] = round(a['identity_median'] - b['identity_median'], 3)
        for k, src in (('lip_jitter', 'lip_jitter_L'), ('face_jitter', 'face_jitter_L'), ('frame_jitter', 'frame_jitter')):
            if a.get(src) and b.get(src):
                d[k] = round(a[src] / b[src], 2)
        for k in ('flow_mouth', 'flow_face'):
            if a.get(k) and b.get(k):
                d[k] = round(a[k] / b[k], 2)
        for k in ('info_sharp', 'info_oversharp', 'info_smear'):
            if a.get(k) and b.get(k):
                d[k] = round(a[k] / b[k] - 1, 3)
        if audio and audio.get('_new') is not None and audio.get('_base') is not None:
            (x, sr), (y, sr2) = audio['_new'], audio['_base']
            rx, ry = x[int(t0 * sr):int(t1 * sr)], y[int(t0 * sr2):int(t1 * sr2)]
            if len(rx) and len(ry):
                d['audio_db'] = round(float(20 * np.log10((np.sqrt((rx ** 2).mean()) + 1e-9) / (np.sqrt((ry ** 2).mean()) + 1e-9))), 2)
        r['delta'] = d
        out[sid] = r
    return out


def relative_gates(cmp, label, focus, near_frac=0.9, audit=False):
    """Gates against a reference build: the base (last approved) or the ANCHOR (first approved), so per-round
    tolerances can never compound into a slow drift ('worse every iteration'). `audit` = full-episode run (no
    --focus): every shot is checked, an unchanged shot is fine, changed shots get the relative gates."""
    G, out = GATES, []

    def add(ok, sev, where, metric, what, value=None):
        out.append(dict(ok=bool(ok), severity=sev, where=where, metric=metric, check=what, value=value, kind='rel'))
    for sid, b in (cmp or {}).items():
        if b.get('unchanged_expected'):
            if label == 'base':
                add(b['block_max'] <= G['unchanged_block'] and b['frames'][0] == b['frames'][1], 'FAIL', sid, 'untouched',
                    f"untouched shot identical to base (worst 10x10 block diff over all frames {b['block_max']}, frames {b['frames']})", b['block_max'])
                if 'audio_db' in b['delta']:
                    add(abs(b['delta']['audio_db']) <= G['audio_db'], 'WARN', sid, 'audio_untouched',
                        f"audio level vs base {b['delta']['audio_db']:+} dB", b['delta']['audio_db'])
            continue
        if label == 'base' and not b.get('changed') and not audit:
            add(False, 'FAIL', sid, 'focus_unchanged', 'focus shot did not change (the intended fix is not in this build)')
            continue
        if not b.get('changed'):
            continue
        for k in b.get('lost', []):
            add(False, 'FAIL', sid, f'lost_{k}', f"{k} measurable in {label} but not in this build (fail closed)")
        if b.get('flow_error'):
            add(False, 'FAIL', sid, 'flow_unmeasured', f"face flicker (RAFT) not measured: {b['flow_error']} (fail closed)")
        d = b['delta']
        if d.get('identity') is not None:
            v, t = d['identity'], -G['identity_drop']
            add(v >= t, 'FAIL', sid, f'identity_drop_{label}', f"identity vs {label} {v:+.3f} (>= {t})", v)
            if t <= v < t * near_frac:
                add(False, 'NEAR', sid, f'identity_drop_{label}_near', f"identity vs {label} {v:+.3f} is within 10 % of the gate", v)
        # qa-9: faces are judged on the RAFT motion-compensated residual (flow_mouth / flow_face); the old
        # lip/face 2nd-difference jitter grew with motion and sharpness (pier3 false alarm) and is INFO now
        for k, t in (('flow_mouth', G['flow_rise']), ('flow_face', G['flow_rise']), ('frame_jitter', G['jitter_rise'])):
            if d.get(k) is not None:
                v = d[k]
                add(v <= t, 'FAIL', sid, f'{k}_rise_{label}', f"{k} vs {label} x{v:.2f} (<= x{t})", v)
                if 1 + (t - 1) * near_frac < v <= t:
                    add(False, 'NEAR', sid, f'{k}_rise_{label}_near', f"{k} vs {label} x{v:.2f} is within 10 % of the gate x{t}", v)
                if 1 + (t - 1) * near_frac < v <= t:
                    add(False, 'NEAR', sid, f'{k}_rise_{label}_near', f"{k} vs {label} x{v:.2f} is within 10 % of the gate x{t}", v)
        if label == 'base' and d.get('info_sharp') is not None:
            add(d['info_sharp'] >= -G['sharp_drop_warn'], 'WARN', sid, 'sharp_drop',
                f"native face sharpness vs base {d['info_sharp']:+.0%} (INFO metric, not calibrated)", d['info_sharp'])
    return out


def _neighbours(tl, focus):
    ids = [s for s, _, _ in tl]
    w = set(focus)
    for i, s in enumerate(ids):
        if s in focus:
            w.update(ids[max(0, i - 1):i + 2])
    return w


# ------------------------------------------------------------------ run
def run(episode, video, timeline_yaml=None, base=None, base_timeline=None, focus=(), fmt='short', out=None,
        intent='', no_audio=False, seed=None, anchor=None, anchor_timeline=None, log=print):
    from . import spec as Sp
    t_start = time.time()
    spec = Sp.load(episode)
    tl = timeline(timeline_yaml or episode)
    audit = not focus   # full-episode run: every shot is a focus shot, unchanged ones are fine
    focus = set(focus or [s for s, _, _ in tl])
    lipsynced = {s['id'] for s in spec['shots'] if (s.get('gen') or {}).get('lipsync')}
    out = out or os.path.join(spec['root'], 'out', 'qa', os.path.splitext(os.path.basename(video))[0])
    os.makedirs(out, exist_ok=True)
    p, pre = preflight(spec, video, fmt, tl)
    fps = p['fps']
    wanted = _neighbours(tl, focus)
    shots, FS, M, cast_of = analyse(spec, video, tl, fps, wanted, lipsynced)
    audio = {} if no_audio else audio_metrics(video, p['dur'])
    pcm = audio.pop('_pcm', None)
    for sid in lipsynced & wanted:
        if sid in M:
            M[sid]['lipsync'] = lipsync_offset(shots[sid], FS.get(sid), pcm, fps)
    cuts = cut_metrics(tl, shots, M, cast_of)
    base_cmp, anchor_cmp, base_made, cached, director, quality = None, None, [], None, None, None
    G = gates(tl, M, cuts, None, audio, pre, cast_of, lipsynced, focus)
    if base:
        btl = timeline(base_timeline or timeline_yaml or episode)
        bshots, bFS, bM, cached = analyse_cached(spec, base, btl, probe(base)['fps'], _neighbours(btl, focus), lipsynced)
        aud = {}
        if not no_audio and pcm is not None:
            bx, bsr = _pcm(base)
            aud = dict(_new=pcm, _base=(bx.mean(1), bsr))
        base_cmp = compare_base(tl, shots, M, bshots, bM, focus, aud)
        G += relative_gates(base_cmp, 'base', focus, audit=audit)
        bG = gates(btl, bM, cut_metrics(btl, bshots, bM, cast_of), None, {}, [], cast_of, lipsynced, focus)
        mark_known(G, bG, {(g['where'], g['metric']): g['value'] for g in bG})
        if anchor:
            atl = timeline(anchor_timeline or base_timeline or timeline_yaml or episode)
            ashots, _, aM, _ = analyse_cached(spec, anchor, atl, probe(anchor)['fps'], _neighbours(atl, focus), lipsynced)
            anchor_cmp = compare_base(tl, shots, M, ashots, aM, focus)
            G += relative_gates(anchor_cmp, 'anchor', focus, audit=audit)
        # aligned windows: BOTH cuts at the SAME timecodes (union of each build's worst/median/random)
        for sid in focus:
            if sid in M and sid in bM:
                n = min(len(shots[sid]['thumbs']), len(bshots[sid]['thumbs']))
                js = []
                for lab, j in M[sid].get('_windows', []) + [('base_' + l, j) for l, j in bM[sid].get('_windows', [])]:
                    j = min(max(4, j), max(4, n - 4))
                    if all(abs(j - k) >= 8 for _, k in js):
                        js.append((lab, j))
                M[sid]['_windows'] = bM[sid]['_windows'] = js[:6]
                if M[sid].get('_src') or bM[sid].get('_src'):
                    M[sid]['_src'] = bM[sid]['_src'] = M[sid].get('_src') or bM[sid].get('_src')
        for sid in focus:   # face-less focus shots: windows at the largest per-frame changes vs base
            if sid in M and sid in bM and not FS.get(sid):
                A, B = shots[sid]['thumbs'], bshots[sid]['thumbs']
                n = min(len(A), len(B))
                if n > 12:
                    d = np.array([np.abs(A[k].astype(np.int16) - B[k]).mean() for k in range(n)])
                    st = []
                    for k in np.argsort(-d):
                        k0 = int(min(max(0, k - 3), n - 6))
                        if all(abs(k0 - q) >= 6 for q in st):
                            st.append(k0)
                        if len(st) == 4:
                            break
                    M[sid]['_change_starts'] = bM[sid]['_change_starts'] = sorted(st)
        base_made = artifacts(base, btl, bshots, bFS, bM, os.path.join(out, 'base_art'), focus, fps, prefix='base_')
        director = run_director(video, tl, base, btl, focus, timeline_yaml or episode, log=log)
        quality = run_quality(video, tl, base, btl, focus, log=log)
        for sid, d in quality.items():
            if d.get('error'):
                G.append(dict(ok=False, severity='FAIL', where=sid, metric='face_quality_unmeasured',
                              check=f"face quality model did not run: {d['error']} (fail closed)", value=None, kind='rel'))
            elif d.get('delta') is not None:
                G.append(dict(ok=d['delta'] >= -GATES['face_quality_drop'], severity='FAIL', where=sid,
                              metric='face_quality_drop_base', kind='rel', value=d['delta'],
                              check=f"face quality (VisualQuality-R1, 1-5) vs base {d['delta']:+.2f} (>= -{GATES['face_quality_drop']}; new {d['new']}, base {d['base']})"))
    # passing absolute gates within 10 % of their threshold -> NEAR (shown to the owner, not blocking)
    for g in list(G):
        if g['ok'] and g['kind'] == 'abs' and isinstance(g['value'], (int, float)) and g['metric'] in ('identity_median', 'lip_jitter_L', 'lip_jitter_a'):
            t = GATES[g['metric']]
            if (g['metric'].startswith('identity') and g['value'] < t * 1.1) or (g['metric'].startswith('lip') and g['value'] > t * 0.9):
                G.append(dict(ok=False, severity='NEAR', where=g['where'], metric=g['metric'] + '_near',
                              check=g['check'] + ' (within 10 % of the gate)', value=g['value'], kind='abs'))
    made = artifacts(video, tl, shots, FS, M, os.path.join(out, 'new_art'), focus, fps)
    se = spec.get('series') or {}
    pk, key = critic_packet(made, base_made, out, style_note=se.get('qa_style_note', ''), intent=intent,
                            what=se.get('qa_what') or f"a {fmt} video", seed=seed)
    fails = [g for g in G if not g['ok'] and g['severity'] == 'FAIL']
    res = dict(version=QA_VERSION, code=_CODE_HASH, prompt_hash=CRITIC_PROMPT_HASH, video=video, base=base, anchor=anchor,
               focus=sorted(focus), lipsynced=sorted(lipsynced), verdict='PASS' if not fails else 'FAIL', probe=p,
               wall_s=round(time.time() - t_start, 1), base_cache_hit=cached,
               shots={k: {x: y for x, y in v.items() if not x.startswith('_')} for k, v in M.items()},
               cuts=cuts, base_compare=base_cmp, anchor_compare=anchor_cmp, audio=audio, gates=G, critic_packet=pk,
               director=director, quality=quality)
    json.dump(res, open(os.path.join(out, 'qa.json'), 'w'), indent=1, default=str)
    order = {'FAIL': 0, 'NEAR': 1, 'KNOWN': 2, 'WARN': 3}
    bad = sorted([g for g in G if not g['ok']], key=lambda g: order.get(g['severity'], 9))
    log(f"[qa] {res['verdict']}: " + ', '.join(f"{sum(1 for g in bad if g['severity'] == s)} {s.lower()}" for s in order)
        + f", {sum(g['ok'] for g in G)} ok, {res['wall_s']} s -> {out}")
    for g in bad:
        log(f"  {g['severity']:5s} {g['where']:28s} {g['check']}")
    log(f"  critic packet: {pk}")
    return res


def _captions(yaml_path):
    """shot id -> what the shot should show (its gen motion prompt, else its prompt): the director's caption."""
    try:
        ep = yaml.safe_load(open(yaml_path, encoding='utf-8')) or {}
    except Exception:
        return {}
    out = {}
    for sh in ep.get('shots', []) or []:
        g = sh.get('gen') or {}
        out[sh.get('id')] = (g.get('motion') or g.get('prompt') or sh.get('prompt') or sh.get('caption') or sh.get('id') or '').strip()
    return out


def run_director(video, tl, base, btl, focus, new_yaml, log=print):
    """qa-10: the director model rates every focus shot's new and base take (vf/gen/director.py, one take at a time,
    6 yes/no questions + notes on the new take). Takes are cached in cache/qa/director by file identity + range +
    caption + DIRECTOR_VERSION (base takes repeat across reviews). -> {shot: dict(new, base, delta, verdict, notes)}
    or {shot: dict(error=...)}. A failed run is recorded, never skipped silently."""
    from .gen import director as D
    caps = _captions(new_yaml)
    tn, tb = {s: (a, b) for s, a, b in tl}, {s: (a, b) for s, a, b in btl}
    shots = [s for s in sorted(focus) if s in tn and s in tb]
    cdir = os.path.join(CACHE, 'director')
    os.makedirs(cdir, exist_ok=True)

    def ck(v, t, cap, crit):
        st = os.stat(v)
        return os.path.join(cdir, hashlib.sha1(json.dumps([os.path.abspath(v), st.st_size, st.st_mtime, t, cap, crit,
                                                           D.DIRECTOR_VERSION]).encode()).hexdigest()[:20] + '.json')
    todo, got = [], {}
    for sid in shots:
        for side, v, t in (('new', video, tn[sid]), ('base', base, tb[sid])):
            p = ck(v, t, caps.get(sid, sid), side == 'new')
            if os.path.exists(p):
                got[(sid, side)] = json.load(open(p, encoding='utf-8'))
            else:
                todo.append(((sid, side), p, dict(mode='rate', clip=dict(video=os.path.abspath(v), t0=t[0], t1=t[1]),
                                                   caption=caps.get(sid, sid), frames=8, critique=(side == 'new'))))
    if todo:
        log(f"[qa] director: rating {len(todo)} take(s) (Qwen3.5-9B, ~40 s load + ~15 s each)")
        try:
            from .gen.bridge import run_gen
            rs = run_gen('director', [j for _, _, j in todo], persistent=False, check=False, timeout=3600)
        except Exception as e:
            rs = [dict(ok=False, error=f'{type(e).__name__}: {e}')] * len(todo)
        for (k, p, _), r in zip(todo, rs):
            got[k] = r
            if r.get('ok'):
                json.dump(r, open(p, 'w', encoding='utf-8'), indent=1)
    out = {}
    for sid in shots:
        n, b = got.get((sid, 'new')) or {}, got.get((sid, 'base')) or {}
        if not (n.get('ok') and b.get('ok')):
            out[sid] = dict(error=str(n.get('error') or b.get('error') or 'no result')[:200])
            continue
        dd = {k: round(n['scores'][k] - b['scores'][k], 3) for k in n['scores']}
        delta = round(sum(dd.values()) / len(dd), 3)
        out[sid] = dict(new=n['scores'], base=b['scores'], change=dd, delta=delta, notes=n.get('notes'),
                        verdict='worse' if delta < GATES['director_worse'] else 'better' if delta > -GATES['director_worse'] else 'same',
                        version=n.get('version'))
    return out


def run_quality(video, tl, base, btl, focus, log=print):
    """qa-11: VisualQuality-R1-7B face-crop quality (median over 8 frames) of each focus shot's new and base take.
    Cached per take in cache/qa/quality. Shots without a face on either side get no check. -> {shot: dict(new, base,
    delta)} or {shot: dict(error)}."""
    from .gen import iqa as Q
    tn, tb = {s: (a, b) for s, a, b in tl}, {s: (a, b) for s, a, b in btl}
    shots = [s for s in sorted(focus) if s in tn and s in tb]
    cdir = os.path.join(CACHE, 'quality')
    os.makedirs(cdir, exist_ok=True)

    def ck(v, t):
        st = os.stat(v)
        return os.path.join(cdir, hashlib.sha1(json.dumps([os.path.abspath(v), st.st_size, st.st_mtime, t,
                                                           Q.IQA_VERSION]).encode()).hexdigest()[:20] + '.json')
    todo, got = [], {}
    for sid in shots:
        for side, v, t in (('new', video, tn[sid]), ('base', base, tb[sid])):
            p = ck(v, t)
            if os.path.exists(p):
                got[(sid, side)] = json.load(open(p, encoding='utf-8'))
            else:
                todo.append(((sid, side), p, dict(clip=dict(video=os.path.abspath(v), t0=t[0], t1=t[1]), frames=8)))
    if todo:
        log(f"[qa] face quality: scoring {len(todo)} take(s) (VisualQuality-R1-7B, ~70 s each)")
        try:
            from .gen.bridge import run_gen
            rs = run_gen('iqa', [j for _, _, j in todo], persistent=False, check=False, timeout=7200)
        except Exception as e:
            rs = [dict(ok=False, error=f'{type(e).__name__}: {e}')] * len(todo)
        for (k, p, _), r in zip(todo, rs):
            got[k] = r
            if r.get('ok'):
                json.dump(r, open(p, 'w', encoding='utf-8'), indent=1)
    out = {}
    for sid in shots:
        n, b = got.get((sid, 'new')) or {}, got.get((sid, 'base')) or {}
        if not (n.get('ok') and b.get('ok')):
            out[sid] = dict(error=str(n.get('error') or b.get('error') or 'no result')[:200])
            continue
        fn, fb = (n.get('summary') or {}).get('face'), (b.get('summary') or {}).get('face')
        out[sid] = dict(new=fn, base=fb, delta=None if fn is None or fb is None else round(fn - fb, 3))
    return out


def replay(cases_path, log=print):
    """Re-run every pinned case. Harness verdict must match `expect` AND every `expect_gates` entry (shot:metric) must
    be among the failing gates (right verdict for the right reason). Stored critic answers are re-scored in code and
    compared with `expect_system`."""
    C = yaml.safe_load(open(cases_path, encoding='utf-8'))
    base_dir = os.path.dirname(os.path.abspath(cases_path))
    rp = lambda x: x if (x is None or os.path.isabs(x)) else os.path.join(base_dir, x)
    rows = []
    for c in C['cases']:
        out = os.path.join(base_dir, 'replay', c['name'])
        r = run(rp(C['episode']), rp(c['video']), rp(c.get('timeline')), rp(c.get('base')), rp(c.get('base_timeline')),
                c.get('focus', []), out=out, intent=c.get('intent', ''), no_audio=c.get('no_audio', False), seed=c['name'], anchor=rp(c.get('anchor')),
                anchor_timeline=rp(c.get('anchor_timeline')), log=lambda *_: None)
        failing = {f"{g['where']}:{g['metric']}" for g in r['gates'] if not g['ok'] and g['severity'] == 'FAIL'}
        missing = [e for e in c.get('expect_gates', []) if e not in failing]
        ok_h = r['verdict'] == c['expect'] and not missing
        # qa-8 (audit Q9): the system prediction is compared with the OWNER's verdict, never with a stored
        # expectation (cases.yaml used to record known misses as expected, so replay counted them as agreement)
        sysv, owner = None, c.get('verdict')
        crit = c.get('critics') or {}
        if crit.get('key'):
            shutil.copy2(rp(crit['key']), os.path.join(out, 'ab_key.json'))
            sysv = score(out, [rp(a) for a in crit['answers']], item=c['name'], log=lambda *_: None, record=False,
                         brief_paths=[rp(a) for a in crit.get('brief', [])])['prediction']
        elif crit:
            sysv = 'NO KEY'   # answers stored without their A/B key: cannot be re-scored
        want = {'approved': 'SHIP', 'rejected': 'BLOCK'}.get(owner)
        head = (sysv or '').split(' ')[0]
        # CHECK / NEEDS THIRD READ / INCOMPLETE hand the call to the owner: neither agreement nor a wrong call
        ok_s = None if (sysv in (None, 'NO KEY') or want is None) else ('owner' if head not in ('SHIP', 'BLOCK') else head == want)
        judged = c.get('judged', True)
        rows.append(dict(name=c['name'], ok=ok_h, ok_system=ok_s, judged=judged, harness=r['verdict'], system=sysv,
                         owner=owner, missing=missing))
        sys_txt = (f" | system {sysv} vs owner {owner}: "
                   f"{'your call' if ok_s == 'owner' else 'agree' if ok_s else ('n/a' if ok_s is None else 'WRONG')}") if sysv else ''
        log(f"{'OK ' if ok_h else 'MISS'} {c['name']:22s} harness {r['verdict']} (expect {c['expect']}){sys_txt}"
            f"{'' if judged else '  [not judged]'}{'  MISSING ' + str(missing) if missing else ''}")
    J = [x for x in rows if x['judged']]
    S = [x for x in rows if x['ok_system'] is not None]
    n_own = sum(1 for x in S if x['ok_system'] == 'owner')
    log(f"[qa replay] harness: {sum(1 for x in J if x['ok'])}/{len(J)} owner-judged cases as expected; system vs owner: "
        f"{sum(1 for x in S if x['ok_system'] is True)} agree, {sum(1 for x in S if x['ok_system'] is False)} wrong, "
        f"{n_own} your call, of {len(S)}. IN-SAMPLE (thresholds and prompts were tuned on these): "
        f"not evidence. Held-out accuracy comes only from qa/predictions.log vs qa/verdicts.csv.")
    return rows


def main(argv=None):
    import argparse
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'replay':
        rows = replay(argv[1] if len(argv) > 1 else os.path.join(_ROOT, 'qa', 'cases.yaml'))
        return 0 if all(r['ok'] for r in rows) else 1
    if argv and argv[0] == 'verify-log':
        ok, bad, head = verify_log(os.path.join(_ROOT, 'qa', argv[1] if len(argv) > 1 else 'predictions.log'))
        print(f'prediction log intact, head {head[:16]} (compare with the last head in qa/verdicts.log)' if ok
              else f'prediction log BROKEN at line {bad}')
        return 0 if ok else 1
    if argv and argv[0] == 'score':
        ap = argparse.ArgumentParser(prog='python -m vf.qa score')
        ap.add_argument('cmd'); ap.add_argument('out'); ap.add_argument('answers', nargs='+', help='A/B read answers')
        ap.add_argument('--brief', nargs='*', default=[], help='brief read answers (new cut only)')
        ap.add_argument('--item', default='')
        ap.add_argument('--calibration', action='store_true', help='in-sample re-scoring: log to predictions_calibration.log')
        a = ap.parse_args(argv)
        r = score(a.out, a.answers, a.item, logfile='predictions_calibration.log' if a.calibration else 'predictions.log',
                  brief_paths=a.brief)
        return 0 if r['prediction'].startswith('SHIP') else 1
    ap = argparse.ArgumentParser(prog='python -m vf.qa')
    ap.add_argument('episode'); ap.add_argument('video')
    ap.add_argument('--timeline', help='episode yaml this video was built from (default: episode)')
    ap.add_argument('--base', help='the last build the owner approved')
    ap.add_argument('--base-timeline')
    ap.add_argument('--anchor', help='the FIRST approved build of these shots: tolerances never compound past it')
    ap.add_argument('--anchor-timeline')
    ap.add_argument('--fmt', default='short')
    ap.add_argument('--focus', help='comma list of shots meant to change (all others must be pixel-identical)')
    ap.add_argument('--out')
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--intent', help="aspects the owner asked to change, e.g. 'det_speak: lighting; mara_pier: edit position'")
    a = ap.parse_args(argv)
    r = run(a.episode, a.video, a.timeline, a.base, a.base_timeline, a.focus.split(',') if a.focus else (), a.fmt,
            a.out, a.intent or '', a.no_audio, anchor=a.anchor, anchor_timeline=a.anchor_timeline)
    return 0 if r['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
