"""Face tone match: `gen: {face_tone: {stop, skin_ab, lip_ab, feather}}` on a finished AI clip.

Matches one shot's face to a neighbour of the same character (continuity across a cut) without regenerating:
  - the face is found every frame (YuNet), its box and landmarks smoothed over time (no mask jitter);
  - inside a soft ellipse over the face, lightness is scaled by 2**stop and skin chroma shifted by skin_ab (Lab a*, b*);
  - inside a smaller ellipse over the lips, only lip-coloured pixels (a* above the skin's) get lip_ab added, so the
    lipstick hue follows the neighbour shot (e.g. warm red -> cool magenta) without tinting the skin around them.
Measured targets come from the neighbour shot (vf/facetone.py has no opinion about colours). 2026-11-03, Q8:
mara_pier vs mara_cu: skin L* ~160 vs ~87, skin b* +8 warmer, lips a*/b* 147/133 vs 154/120 (OpenCV 8-bit Lab).
"""
import numpy as np
import cv2
from .memo import memo_file

VERSION = 'facetone-1'


def _track(src, scale=0.5):
    """Per-frame (cx, cy, w, h, mouth_cx, mouth_cy, mouth_w) of the largest face, smoothed over time; NaN-filled gaps
    are interpolated (a frame without a detection keeps the neighbours' mask)."""
    from .identity import _det
    from .platelock import _frames
    rows = []
    for f in _frames(src):
        s = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, fa = _det(s.shape[1], s.shape[0]).detect(s)
        if fa is None or not len(fa):
            rows.append([np.nan] * 7)
            continue
        a = fa[np.argmax(fa[:, 2] * fa[:, 3])] / scale
        lm = a[4:14].reshape(5, 2)
        ml, mr = lm[3], lm[4]
        rows.append([a[0] + a[2] / 2, a[1] + a[3] / 2, a[2], a[3], (ml[0] + mr[0]) / 2, (ml[1] + mr[1]) / 2,
                     float(np.linalg.norm(mr - ml))])
    t = np.array(rows, np.float64)
    if not len(t) or np.isnan(t[:, 0]).all():
        return None
    idx = np.arange(len(t))
    for c in range(t.shape[1]):
        ok = ~np.isnan(t[:, c])
        t[:, c] = np.interp(idx, idx[ok], t[ok, c])
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(t, size=9, axis=0, mode='nearest')


@memo_file('global', '.mp4', version=VERSION)
def _face_tone(src, stop, skin_ab, lip_ab, feather, out=None):
    from .platelock import _frames, _x264
    from .proc import writer
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    tr = _track(src)
    if tr is None:
        raise RuntimeError(f'facetone: no face found in {src}')
    gain = 2.0 ** float(stop)
    with writer(_x264(out, W, H, fps), out) as p:
        for i, f in enumerate(_frames(src)):
            cx, cy, fw, fh, mx, my, mw = tr[min(i, len(tr) - 1)]
            # work only in a box around the face (fast at 1152x2048)
            pad = int(0.9 * max(fw, fh))
            x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
            x1, y1 = min(W, int(cx + pad)), min(H, int(cy + pad))
            roi = f[y0:y1, x0:x1]
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
            h, w = roi.shape[:2]
            face = np.zeros((h, w), np.float32)
            cv2.ellipse(face, (int(cx - x0), int(cy - y0 + 0.05 * fh)), (int(0.62 * fw), int(0.78 * fh)), 0, 0, 360,
                        1.0, -1)
            face = cv2.GaussianBlur(face, (0, 0), max(2.0, feather * fw))
            lips = np.zeros((h, w), np.float32)
            cv2.ellipse(lips, (int(mx - x0), int(my - y0)), (int(0.62 * mw), int(0.36 * mw)), 0, 0, 360, 1.0, -1)
            lips = cv2.GaussianBlur(lips, (0, 0), max(1.5, 0.08 * mw))
            # lip pixels: a* clearly above the skin's (skin = face mask outside the lip ellipse)
            skin_sel = (face > 0.8) & (lips < 0.05)
            skin_a = float(np.median(lab[..., 1][skin_sel])) if skin_sel.any() else 135.0
            lipw = lips * np.clip((lab[..., 1] - skin_a - 4) / 8.0, 0, 1)
            lab[..., 0] = lab[..., 0] * (1 - face) + lab[..., 0] * gain * face
            lab[..., 1] += face * skin_ab[0] + lipw * lip_ab[0]
            lab[..., 2] += face * skin_ab[1] + lipw * lip_ab[1]
            f[y0:y1, x0:x1] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            p.stdin.write(f.tobytes())


def face_tone(src, stop=-0.6, skin_ab=(0.0, 0.0), lip_ab=(0.0, 0.0), feather=0.12):
    """Returns the cached path of `src` with the face tone match applied (see module doc)."""
    import os
    return _face_tone(os.path.abspath(src), round(float(stop), 3), tuple(float(x) for x in skin_ab),
                      tuple(float(x) for x in lip_ab), round(float(feather), 3))
