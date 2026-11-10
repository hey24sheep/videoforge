"""Automatic QC + repair for generated (image-to-video) clips. Render python, CPU only (OpenCV + NumPy + SciPy):
~1-2 s per 5 s clip, so it needs no GPU lock and can run while the next clip generates.

    from vf.clipqc import analyze, repair, best_window
    rep = analyze(clip, hints=dict(locked=True, people=False, faces_speaking=False, smoke=False, water=True))
    if rep['fail']: ...retry another seed, or:
    fixed = repair(clip, rep, hints)          # patches 1-3 frame smears, stabilises locked shots (cached mp4)
    s, e = best_window(rep, n_frames)          # cleanest contiguous span of n_frames (trim long clips)

Detectors (all at half resolution unless noted; px values are reported at the clip's native resolution):
  jitter     LK feature tracks + RANSAC similarity per frame pair -> camera path; residual against a 0.6 s
             Savitzky-Golay smooth = high-frequency shake (dock/rail/road wobble).
  transient  pixels that differ from BOTH neighbours (1- or 2-frame pops), L + 2x chroma, blurred, largest
             blob as % of the frame; event when above max(abs floor, k x clip median) so smoke/rain/puddles that
             change every frame raise their own baseline (decoder block smears are isolated spikes).
  sudden     area of large frame-to-frame change, robust z vs the clip (pop-ins, teleports, hard morphs).
  drift      correlation of each frame with frame 0 at 1/8 res (identity / scene drift; reported, not failed).
  mouth      OpenCV YuNet face detector (opencv_zoo face_detection_yunet_2023mar.onnx, MIT licence, 230 KB,
             downloaded once to cache/models/yunet) gives mouth corners + nose; dense flow inside the mouth box
             minus the face's own motion -> lip opening speed. Faces narrower than MIN_FACE px are skipped.
Thresholds live in TH (tuned on early test clips, 576x1024 Wan2.2-TI2V-5B-Turbo, 24 fps).
"""
import os, json, time, subprocess
import numpy as np
import cv2
from .memo import memo, memo_file, tier_dir

VERSION = 'clipqc-1+skipstart'
WORK_W = 288                  # analysis width for portrait clips (long side 512); landscape uses 512
YUNET_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/'
             'face_detection_yunet_2023mar.onnx')
MIN_FACE = 60                 # px (native) face width below which mouths are not judged

TH = dict(
    jitter_rms=1.2, jitter_peak=3.0,               # free camera / no hint     (clean clips: 0.2-0.5 / 0.4-1.0)
    jitter_rms_locked=0.7, jitter_peak_locked=2.0,  # structures, static camera (pier_coat shake: 0.98 / 3.15)
    jitter_rms_people=1.5, jitter_peak_people=3.5,  # close-ups: head motion dominates the tracks (0.5-0.6 / 1.2-1.3)
    jitter_event=1.0,
    trans_floor=1.5, trans_k=3.0,                  # % of frame; x clip median
    trans_floor_busy=2.5, trans_k_busy=4.0,        # smoke / water hints
    trans_fail_pct=3.0, trans_fail_pct_busy=4.5,   # an event this big fails the clip (smaller ones: repair only)
    trans_patch_len=5, trans_patch_area=0.45,      # events up to this long / this much of the frame (bbox) and
    trans_patch_nd=14.0,                           # whose clean neighbours agree (mean |diff| in the bbox, 0-255)
                                                   # are patched by repair()
    sudden_z=12.0, sudden_area=3.0,                # robust z and % area for a sudden-change event
    mouth_speed=0.035, mouth_osc=3, mouth_span_min=8,  # opening speed (fraction of mouth width per frame)
)


def _version():
    """cache version = VERSION + the thresholds, so tuning TH re-checks clips (bump VERSION for code changes)."""
    import hashlib
    return VERSION + ':' + hashlib.blake2b(repr(sorted(TH.items())).encode(), digest_size=6).hexdigest()


# ------------------------------------------------------------------ helpers
def _read(path, full=False):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    small, big = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if full:
            big.append(f)
        h, w = f.shape[:2]
        sw = WORK_W if w < h else 512
        small.append(cv2.resize(f, (sw, int(round(h * sw / w))), interpolation=cv2.INTER_AREA))
    cap.release()
    return small, big, fps


def _norm_hints(h):
    h = dict(h or {})
    out = {k: bool(h.get(k, d)) for k, d in
           (('locked', False), ('people', True), ('faces_speaking', False), ('smoke', False), ('water', False))}
    out['skip_start'] = int(h.get('skip_start', 0))  # frames the edit drops anyway (k6 start pop): never fail on them
    return out


def _groups(idx, gap=1):
    """consecutive frame indices -> [(start, end_inclusive)]"""
    out = []
    for i in idx:
        if out and i - out[-1][1] <= gap + 1:   # up to `gap` missing frames inside one group
            out[-1][1] = i
        else:
            out.append([i, i])
    return [tuple(g) for g in out]


def _camera(G):
    """per-pair similarity (2x3, work-res) via LK + RANSAC; identity when tracking fails."""
    Ms = []
    for i in range(1, len(G)):
        a, b = G[i - 1], G[i]
        M = None
        p0 = cv2.goodFeaturesToTrack(a, 400, 0.01, 5, blockSize=5)
        if p0 is not None and len(p0) > 15:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=3)
            st = st.ravel() == 1
            if st.sum() > 15:
                M, _ = cv2.estimateAffinePartial2D(p0[st], p1[st], method=cv2.RANSAC, ransacReprojThreshold=0.7)
        Ms.append(M if M is not None else np.float64([[1, 0, 0], [0, 1, 0]]))
    return Ms


def _path(Ms, c):
    """cumulative camera path (frame 0 -> frame i) as centre-based (tx, ty, log s, rot): C = T(c+t) L T(-c)."""
    C = np.eye(3)
    par = [(0.0, 0.0, 0.0, 0.0)]
    for M in Ms:
        C = np.vstack([M, [0, 0, 1]]) @ C
        s = np.sqrt(abs(np.linalg.det(C[:2, :2])))
        t = C[:2, :2] @ c + C[:2, 2] - c
        par.append((t[0], t[1], np.log(max(s, 1e-6)), np.arctan2(C[1, 0], C[0, 0])))
    return np.array(par)


def _mat(p, c):
    """centre-based (tx, ty, log s, rot) -> 3x3"""
    tx, ty, ls, rot = p
    s = np.exp(ls)
    L = s * np.float64([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = c + np.float64([tx, ty]) - L @ c
    return M


def _smooth(x, win):
    from scipy.signal import savgol_filter
    n = len(x)
    w = min(win, n if n % 2 else n - 1)
    return savgol_filter(x, w, 2, axis=0) if w >= 5 else x.copy()


_YUNET = {}


def _yunet_path():
    d = os.path.join(tier_dir('global'), 'models', 'yunet')
    p = os.path.join(d, 'face_detection_yunet_2023mar.onnx')
    if not os.path.exists(p):
        os.makedirs(d, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(YUNET_URL, p + '.tmp')
        os.replace(p + '.tmp', p)
    return p


def _faces(img):
    h, w = img.shape[:2]
    det = _YUNET.get('det')
    if det is None:
        det = _YUNET['det'] = cv2.FaceDetectorYN.create(_yunet_path(), '', (w, h), 0.7, 0.3, 50)
    det.setInputSize((w, h))
    _, f = det.detect(img)
    return f if f is not None else np.zeros((0, 15), np.float32)


# ------------------------------------------------------------------ mouth
def _mouth(small, S, G, fps):
    """Mouth-opening speed per frame for the largest face. Returns dict (per_frame, face_frames, spans)."""
    n = len(small)
    det = {}
    for i in range(0, n, 3):
        f = _faces(small[i])
        if len(f):
            f = f[np.argmax(f[:, 2] * f[:, 3])]
            if f[2] * S >= MIN_FACE:
                det[i] = f
    speed = np.zeros(n)
    if len(det) < max(3, n // 12):
        return dict(checked=True, face_frames=len(det) * 3, per_frame=speed.round(4).tolist(), spans=[],
                    detector='yunet-2023mar')
    keys = np.array(sorted(det))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for i in range(1, n):
        k = keys[np.argmin(np.abs(keys - i))]
        if abs(k - i) > 4:
            continue
        f = det[k]
        x, y, w, h = f[:4]
        nose = f[8:10]
        mr, ml = f[10:12], f[12:14]
        mc = (mr + ml) / 2
        mw = max(np.linalg.norm(ml - mr), 0.3 * w)
        drop = max(mc[1] - nose[1], 0.12 * h)
        x0, x1 = int(mc[0] - 0.8 * mw), int(mc[0] + 0.8 * mw)
        y0, y1 = int(mc[1] - 0.55 * drop), int(mc[1] + 0.75 * drop)
        fx0, fx1, fy0, fy1 = int(x), int(x + w), int(y), int(y + h)
        H, W = G[i].shape
        x0, y0, fx0, fy0 = max(x0, 0), max(y0, 0), max(fx0, 0), max(fy0, 0)
        x1, y1, fx1, fy1 = min(x1, W), min(y1, H), min(fx1, W), min(fy1, H)
        if x1 - x0 < 6 or y1 - y0 < 6 or fx1 - fx0 < 12:
            continue
        # flow on the face box (upsampled so the mouth spans ~48 px), mouth = sub-box
        sc = 48.0 / max(x1 - x0, 1)
        A = cv2.resize(G[i - 1][fy0:fy1, fx0:fx1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        B = cv2.resize(G[i][fy0:fy1, fx0:fx1], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        fl = dis.calc(A, B, None)
        vy = fl[..., 1] - np.median(fl[..., 1])     # remove head motion
        my0, my1 = int((y0 - fy0) * sc), int((y1 - fy0) * sc)
        mx0, mx1 = int((x0 - fx0) * sc), int((x1 - fx0) * sc)
        mid = int((mc[1] - fy0) * sc)
        up, lo = vy[my0:mid, mx0:mx1], vy[mid:my1, mx0:mx1]
        if up.size < 20 or lo.size < 20:
            continue
        speed[i] = (np.median(lo) - np.median(up)) / (mw * sc)   # + opening, - closing (mouth widths / frame)
    a = np.abs(speed)
    act = a > TH['mouth_speed']
    spans = []
    for s, e in _groups(np.nonzero(act)[0].tolist(), gap=6):
        seg = speed[s:e + 1]
        osc = int(np.sum(np.abs(np.diff(np.sign(seg[np.abs(seg) > TH['mouth_speed']]))) > 0))
        if e - s + 1 >= TH['mouth_span_min'] and osc >= TH['mouth_osc']:
            spans.append([int(s), int(e)])
    return dict(checked=True, face_frames=len(det) * 3, per_frame=speed.round(4).tolist(), spans=spans,
                detector='yunet-2023mar')


# ------------------------------------------------------------------ analyze
def analyze(clip_path, hints=None):
    """QC report for one clip (dict, JSON-serialisable). Cached in the global tier (keyed on file + hints)."""
    return dict(_analyze(os.path.abspath(clip_path), _norm_hints(hints)))


@memo('global', version=_version)
def _analyze(clip_path, hints):
    t0 = time.time()
    small, _, fps = _read(clip_path)
    n = len(small)
    if n < 5:
        return dict(version=VERSION, path=clip_path, frames=n, fps=fps, score=99.0, fail=True,
                    reasons=['clip unreadable or too short'])
    cap = cv2.VideoCapture(clip_path)
    Wn, Hn = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    h, w = small[0].shape[:2]
    S = Wn / w
    G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in small]
    LAB = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.int16) for f in small]
    reasons = []

    # 1) camera jitter
    Ms = _camera(G)
    par = _path(Ms, np.float64([w / 2, h / 2]))
    hf = par - _smooth(par, 15)
    jit = np.hypot(hf[:, 0], hf[:, 1]) * S
    jrms, jpk = float(np.sqrt(np.mean(jit ** 2))), float(jit.max())
    suf = '_locked' if hints['locked'] else ('_people' if hints['people'] else '')
    lim_rms, lim_pk = TH['jitter_rms' + suf], TH['jitter_peak' + suf]
    if jrms > lim_rms:
        reasons.append(f'camera jitter {jrms:.2f} px rms > {lim_rms}')
    if jpk > lim_pk:
        reasons.append(f'camera jolt {jpk:.1f} px at frame {int(jit.argmax())} > {lim_pk}')

    # 2) transient smears (1-2 frame pops)
    trans = np.zeros(n)
    boxes = [None] * n
    for i in range(1, n - 1):
        B = LAB[i]
        dA, dC = np.abs(B - LAB[i - 1]), np.abs(B - LAB[i + 1])
        r = np.minimum(dA, dC)
        if i >= 2:
            r = np.maximum(r, np.minimum(np.abs(B - LAB[i - 2]), dC))
        if i + 2 < n:
            r = np.maximum(r, np.minimum(dA, np.abs(B - LAB[i + 2])))
        r = (r[..., 0] + 2 * (r[..., 1] + r[..., 2])).astype(np.float32)
        r = cv2.GaussianBlur(r, (0, 0), 2.0)
        m = (r > 18).astype(np.uint8)
        k, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if k > 1:
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            trans[i] = st[j, cv2.CC_STAT_AREA] / (h * w) * 100
            x, y, bw, bh = st[j, :4]
            boxes[i] = [int(x * S), int(y * S), int((x + bw) * S), int((y + bh) * S)]
    busy = hints['smoke'] or hints['water']
    base = float(np.median(trans[1:-1]))
    thr = max(TH['trans_floor_busy' if busy else 'trans_floor'], TH['trans_k_busy' if busy else 'trans_k'] * base)
    ev = []
    for s, e in _groups(np.nonzero(trans > thr)[0].tolist(), gap=1):
        bb = [b for b in boxes[s:e + 1] if b]
        box = [min(b[0] for b in bb), min(b[1] for b in bb), max(b[2] for b in bb), max(b[3] for b in bb)]
        area = (box[2] - box[0]) * (box[3] - box[1]) / (Wn * Hn)
        pk = float(trans[s:e + 1].max())
        # blending the clean neighbours is only safe where they agree (static region, no ghosting)
        nd = 99.0
        if s > 1 and e < n - 1:
            x0, y0, x1, y1 = (int(v / S) for v in box)
            nd = float(cv2.absdiff(G[s - 1][y0:y1, x0:x1], G[e + 1][y0:y1, x0:x1]).mean())
        ev.append(dict(start=int(s), end=int(e), peak_pct=round(pk, 2), bbox=box, neighbour_diff=round(nd, 1),
                       patchable=bool(e - s + 1 <= TH['trans_patch_len'] and area < TH['trans_patch_area']
                                      and nd < TH['trans_patch_nd'])))
    lim = max(TH['trans_fail_pct_busy' if busy else 'trans_fail_pct'], thr)
    big = [e for e in ev if e['peak_pct'] >= lim and e['start'] > max(1, hints.get('skip_start', 0))]   # frame 0-1 pops: trim, don't fail
    if any(e['start'] <= 1 for e in ev):
        reasons_soft = 'first-frame pop (trim the first 2 frames)'
    else:
        reasons_soft = None
    if big:
        reasons.append(f'{len(big)} transient smear(s), worst {max(e["peak_pct"] for e in big):.1f}% '
                       f'at frame {max(big, key=lambda e: e["peak_pct"])["start"]}')

    # 3) sudden change
    chg = np.zeros(n)
    for i in range(1, n):
        d = cv2.GaussianBlur(cv2.absdiff(G[i], G[i - 1]), (0, 0), 2.5)
        chg[i] = float((d > 22).mean()) * 100
    med = np.median(chg[1:])
    z = (chg - med) / (np.median(np.abs(chg[1:] - med)) * 1.48 + 0.05)
    z[0] = 0
    sud = [dict(frame=int(i), z=round(float(z[i]), 1), area_pct=round(float(chg[i]), 1))
           for i in np.nonzero((z > TH['sudden_z']) & (chg > TH['sudden_area']))[0] if i > hints.get('skip_start', 0)]
    if sud:
        reasons.append(f'sudden change at frame(s) {[s["frame"] for s in sud][:5]}')

    # 4) drift (report only)
    t8 = [cv2.resize(g, (w // 8, h // 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() for g in G]
    corr = [float(np.corrcoef(t8[0], t)[0, 1]) if t8[0].std() > 0 and t.std() > 0 else 1.0 for t in t8]

    # 5) mouths
    mouth = dict(checked=False, spans=[], per_frame=[])
    if hints['people'] and not hints['faces_speaking']:
        mouth = _mouth(small, S, G, fps)
        if mouth['spans']:
            reasons.append(f'mouth moving (no lip-sync intended) frames {mouth["spans"]}')
    mouth['moving'] = bool(mouth['spans'])

    # per-frame badness (for best_window) and score
    bad = np.zeros(n)
    bad += np.clip(jit / lim_pk, 0, 3)
    bad += np.clip(trans / max(thr, 1e-3), 0, 5) * (trans > thr)
    for s in sud:
        bad[s['frame']] += 3
    for s, e in mouth['spans']:
        bad[s:e + 1] += 1.5
    score = (jrms / lim_rms) * (jrms > 0.6 * lim_rms) + sum(e['peak_pct'] / thr for e in ev) + 2 * len(sud) \
        + 2 * len(mouth['spans'])
    rep = dict(
        version=VERSION, path=clip_path, frames=n, fps=fps, width=Wn, height=Hn, hints=hints,
        jitter=dict(rms_px=round(jrms, 2), peak_px=round(jpk, 2), limit_rms=lim_rms, limit_peak=lim_pk,
                    per_frame=jit.round(2).tolist(), path=par.round(5).tolist(), path_hf=hf.round(5).tolist(),
                    scale=S, work_size=[w, h],
                    events=[[int(i), round(float(v), 1)] for i, v in enumerate(jit) if v > TH['jitter_event']]),
        transient=dict(baseline_pct=round(base, 2), threshold_pct=round(thr, 2),
                       per_frame=trans.round(2).tolist(), events=ev),
        sudden=dict(events=sud, per_frame_z=z.round(1).tolist()),
        drift=dict(corr_to_first=np.round(corr, 3).tolist(), min=round(min(corr), 3), end=round(corr[-1], 3)),
        mouth=mouth, per_frame_bad=bad.round(3).tolist(),
        score=round(float(score), 2), fail=bool(reasons), reasons=reasons,
        notes=[reasons_soft] if reasons_soft else [], secs=round(time.time() - t0, 2))
    return rep


# ------------------------------------------------------------------ best window
def best_window(report, n_frames):
    """(start, end_exclusive) of the cleanest n_frames span. Slight preference for early frames (i2v identity
    drifts with time)."""
    bad = np.asarray(report.get('per_frame_bad') or [], float)
    N = len(bad)
    n_frames = int(n_frames)
    if N == 0 or n_frames >= N:
        return 0, min(N, n_frames) if N else n_frames
    cost = bad + 0.002 * np.arange(N)
    c = np.concatenate([[0], np.cumsum(cost)])
    sums = c[n_frames:] - c[:-n_frames]
    s = int(np.argmin(sums + 1e-9 * np.arange(len(sums))))
    return s, s + n_frames


# ------------------------------------------------------------------ repair
def repair(clip_path, report, hints=None, stabilize=None):
    """Patched/stabilised copy (cached mp4, same size + fps), or clip_path when nothing applies.
    stabilize: None = only for hints.locked shots whose jitter exceeds 0.6x the limit."""
    hints = _norm_hints(hints or report.get('hints'))
    smears = [dict(start=e['start'], end=e['end'], bbox=e['bbox'])
              for e in report.get('transient', {}).get('events', []) if e.get('patchable')]
    j = report.get('jitter', {})
    if stabilize is None:
        stabilize = hints['locked'] and j.get('rms_px', 0) > 0.6 * j.get('limit_rms', 1.0) or \
            hints['locked'] and j.get('peak_px', 0) > 0.6 * j.get('limit_peak', 3.0)
    if not smears and not stabilize:
        return clip_path
    plan = dict(smears=smears, stab=bool(stabilize),
                path=j.get('path') if stabilize else None, scale=j.get('scale', 1.0), work=j.get('work_size'))
    return _repair(os.path.abspath(clip_path), plan)


def _write(frames, fps, out):
    from .render import ffmpeg
    h, w = frames[0].shape[:2]
    p = subprocess.Popen([ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{w}x{h}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '14',
                          '-preset', 'medium', '-pix_fmt', 'yuv420p', '-f', 'mp4', out], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('clipqc: ffmpeg encode failed')


@memo_file('global', '.mp4', version=VERSION + '-repair')
def _repair(clip_path, plan, out=None):
    _, F, fps = _read(clip_path, full=True)
    n = len(F)
    H, W = F[0].shape[:2]
    F = [f.astype(np.float32) for f in F]
    bad = set()
    for e in plan['smears']:
        bad.update(range(e['start'], e['end'] + 1))
    for e in plan['smears']:
        p = e['start'] - 1
        while p in bad and p > 0:
            p -= 1
        q = e['end'] + 1
        while q in bad and q < n - 1:
            q += 1
        p, q = max(p, 0), min(q, n - 1)
        x0, y0, x1, y1 = e['bbox']
        pw, ph = int(0.15 * (x1 - x0)) + 8, int(0.15 * (y1 - y0)) + 8
        x0, y0, x1, y1 = max(x0 - pw, 0), max(y0 - ph, 0), min(x1 + pw, W), min(y1 + ph, H)
        m = np.zeros((H, W), np.float32)
        m[y0:y1, x0:x1] = 1
        sig = 0.15 * min(x1 - x0, y1 - y0) + 4
        m = cv2.GaussianBlur(m, (0, 0), sig)
        m = np.clip(m / max(m.max(), 1e-6), 0, 1)[..., None]
        for k in range(e['start'], e['end'] + 1):
            a = (k - p) / max(q - p, 1)
            patch = (1 - a) * F[p] + a * F[q]
            F[k] = F[k] * (1 - m) + patch * m
    if plan['stab'] and plan['path']:
        par = np.asarray(plan['path'], float)
        sm = _smooth(par, 15)
        S = float(plan['scale'])
        cw = np.float64(plan['work']) / 2
        D, Di = np.diag([S, S, 1.0]), np.diag([1 / S, 1 / S, 1.0])
        cx, cy = W / 2, H / 2
        corners = np.float64([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]])
        mats, disp = [], 0.0
        for p_, s_ in zip(par, sm):
            # frame i -> where it would be on the smoothed path: C~ C^-1 (work res), then to native px
            M = (D @ _mat(s_, cw) @ np.linalg.inv(_mat(p_, cw)) @ Di)[:2]
            mats.append(M)
            disp = max(disp, float(np.abs(corners @ M.T - corners[:, :2]).max()))
        z = 1 + min(0.03, 2 * disp / min(W, H))
        Z = np.float64([[z, 0, cx - z * cx], [0, z, cy - z * cy], [0, 0, 1]])
        for k, M in enumerate(mats):
            A = (Z @ np.vstack([M, [0, 0, 1]]))[:2]
            F[k] = cv2.warpAffine(F[k], A, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    _write([np.clip(f, 0, 255).astype(np.uint8) for f in F], fps, out)


if __name__ == '__main__':  # python -m vf.clipqc clip.mp4 [locked,people,smoke,water,faces_speaking]
    import sys
    hs = {k: True for k in (sys.argv[2].split(',') if len(sys.argv) > 2 else [])}
    if hs and 'people' not in hs:
        hs['people'] = False
    r = analyze(sys.argv[1], hs or None)
    print(json.dumps({k: v for k, v in r.items() if k not in ('per_frame_bad',)}, default=str)[:4000])
