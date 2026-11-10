"""Plate lock: freeze everything in a clip except one region (CPU, cached mp4).

Video models move the whole scene when asked to move one thing (2026-11-03: "the whole payphone dangles" when only
the receiver should swing). `gen: {lock: {keep: [x0, y0, x1, y1], anchor: [x0, y0, x1, y1], feather: 0.05}}`
(fractions of the frame):
  - plate  = the first frame the edit uses (`start` s into the clip), so tone and finish match exactly;
  - anchor = a rigid area (the payphone body) used to measure the clip's sway; each frame is shifted back onto the
    plate, so what moves inside `keep` stays attached to the static parts;
  - keep   = the only area taken from the clip (the cord + receiver), feathered into the plate.
"""
import os
import subprocess
import numpy as np
import cv2
from .memo import memo_file

VERSION = 'platelock-1'


def _box(b, W, H):
    return int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)


def _mask(boxes, W, H, feather):
    m = np.zeros((H, W), np.float32)
    for b in boxes:
        x0, y0, x1, y1 = _box(b, W, H)
        m[y0:y1, x0:x1] = 1
    return cv2.GaussianBlur(m, (0, 0), max(1.0, feather * min(W, H)))[..., None]


@memo_file('global', '.mp4', version=VERSION + '-swing')
def _swing(src, keep, pivot, amp, period, decay, feather, start, matte=None, out=None):
    """Damped pendulum: the plate's `keep` boxes (cord + receiver) rotate about `pivot` (where the cord leaves the
    phone); the rest of the frame is the still plate. Sharp every frame - AI receivers melted mid-swing.
    matte = optional PNG silhouette of the swinging part (keyframe coordinates, white = swings): only those pixels
    rotate, and the spot they leave is filled from the plate around it (a box would swing its background too)."""
    from .render import ffmpeg
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    ok, plate = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'platelock: no plate frame in {src}')
    c = (pivot[0] * W, pivot[1] * H)
    if matte:
        mt = cv2.resize(cv2.imread(matte, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_AREA)
        hole = cv2.dilate((mt > 40).astype(np.uint8), np.ones((9, 9), np.uint8))
        clean = cv2.inpaint(plate, hole * 255, 9, cv2.INPAINT_TELEA).astype(np.float32)
        m = cv2.GaussianBlur(mt.astype(np.float32) / 255.0, (0, 0), 1.0)[..., None]
    plate = plate.astype(np.float32)
    if not matte:
        clean = plate
        m = _mask(keep, W, H, feather)
    p = subprocess.Popen([ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{W}x{H}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '12',
                          '-preset', 'medium', '-pix_fmt', 'yuv420p', '-f', 'mp4', out], stdin=subprocess.PIPE)
    for k in range(n):  # same length as the source so start/speed in the edit still line up
        t = max(0.0, k / fps - start)
        th = amp * np.exp(-t / decay) * np.cos(2 * np.pi * t / period)
        R = cv2.getRotationMatrix2D(c, th, 1.0)
        rot = cv2.warpAffine(plate, R, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        mr = cv2.warpAffine(m, R, (W, H))[..., None]  # the swung part carries its own soft edge
        if matte:  # the part swings over a background with the part removed
            o = clean * (1 - mr) + rot * mr
        else:
            mm = np.maximum(m, mr)
            o = plate * (1 - mm) + rot * mm
        p.stdin.write(np.clip(o, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('platelock: ffmpeg encode failed')


def _plate(src, k0):
    """Frame k0 (or the last one) of a clip, decoded forward, without holding the clip in RAM (2026-11-03 audit:
    _lock/_stamp kept every frame: 291 frames at 768x1344 = 0.9 GB per call)."""
    cap = cv2.VideoCapture(src)
    plate, i = None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        plate = f
        if i >= k0:
            break
        i += 1
    cap.release()
    if plate is None:
        raise RuntimeError(f'platelock: no frames in {src}')
    return plate


def _frames(src):
    cap = cv2.VideoCapture(src)
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f
    finally:
        cap.release()


def _x264(out, W, H, fps):
    from .render import ffmpeg
    return [ffmpeg(), '-y', '-loglevel', 'error', '-threads', '4', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{W}x{H}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '12', '-preset', 'medium',
            '-pix_fmt', 'yuv420p', '-f', 'mp4', out]


@memo_file('global', '.mp4', version=VERSION)
def _lock(src, keep, anchor, feather, start, out=None):
    from .proc import writer
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    plate = _plate(src, int(round(start * fps))).astype(np.float32)
    m = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = _box(keep, W, H)
    m[y0:y1, x0:x1] = 1
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, feather * min(W, H)))[..., None]
    ax0, ay0, ax1, ay1 = _box(anchor, W, H) if anchor else (0, 0, W, H)
    ref = cv2.cvtColor(plate[ay0:ay1, ax0:ax1], cv2.COLOR_BGR2GRAY)
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    with writer(_x264(out, W, H, fps), out) as p:
        for f in _frames(src):
            f = f.astype(np.float32)
            cur = cv2.cvtColor(f[ay0:ay1, ax0:ax1], cv2.COLOR_BGR2GRAY)
            (dx, dy), _ = cv2.phaseCorrelate(ref, cur, win)  # sway of the rigid anchor vs the plate
            f = cv2.warpAffine(f, np.float32([[1, 0, -dx], [0, 1, -dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
            o = plate * (1 - m) + f * m
            p.stdin.write(np.clip(o, 0, 255).astype(np.uint8).tobytes())


def lock(src, keep, anchor=None, feather=0.05, start=0.0, mode='clip', pivot=None, amp=7.0, period=1.6, decay=2.5,
         matte=None):
    """mode 'clip': AI motion inside `keep` (one box), the rest frozen. mode 'swing': `keep` = list of boxes rotated
    as a damped pendulum about `pivot` (degrees `amp`, seconds `period`/`decay`); no AI motion used."""
    if mode == 'swing':
        boxes = keep if isinstance(keep[0], (list, tuple)) else [keep]
        return _swing(os.path.abspath(src), [[float(x) for x in b] for b in boxes], [float(x) for x in pivot],
                      float(amp), float(period), float(decay), float(feather), round(float(start), 3),
                      **({'matte': os.path.abspath(matte)} if matte else {}))
    return _lock(os.path.abspath(src), [float(x) for x in keep], [float(x) for x in anchor] if anchor else None,
                 float(feather), round(float(start), 3))


def _clean_plate(plate, box, W, H):
    """Fill `box` row by row between its left and right edge colours (out-of-focus backgrounds only)."""
    x0, y0, x1, y1 = _box(box, W, H)
    c = plate.copy()
    l = plate[y0:y1, max(0, x0 - 3):x0].mean(1)
    r = plate[y0:y1, x1:x1 + 3].mean(1)
    w = np.linspace(0, 1, x1 - x0, dtype=np.float32)[None, :, None]
    c[y0:y1, x0:x1] = l[:, None] * (1 - w) + r[:, None] * w
    # restore the soft vertical structure of the surroundings (blur hides the seams)
    c[y0:y1, x0:x1] = cv2.GaussianBlur(c, (0, 0), max(1.0, 0.004 * min(W, H)))[y0:y1, x0:x1]
    return c


@memo_file('global', '.mp4', version=VERSION + '-erase')
def _erase(src, box, ref, feather, start, grain, out=None):
    """Remove a static unwanted object inside an out-of-focus `box`: a cleaned plate replaces it, scaled every frame
    by the brightness of `ref` (light flicker survives) plus the frame's own fine detail (rain, grain)."""
    from .render import ffmpeg
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    ok, plate = cap.read()
    if not ok:
        raise RuntimeError(f'platelock: no plate frame in {src}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    plate = plate.astype(np.float32)
    clean = _clean_plate(plate, box, W, H)
    pad = int(0.03 * min(W, H))
    x0, y0, x1, y1 = _box(box, W, H)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    m = _mask([box], W, H, feather)[Y0:Y1, X0:X1]
    rx0, ry0, rx1, ry1 = _box(ref, W, H)
    ref0 = plate[ry0:ry1, rx0:rx1].mean() + 1e-3
    sig = max(1.0, 0.006 * min(W, H))
    p = subprocess.Popen([ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{W}x{H}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '12',
                          '-preset', 'medium', '-pix_fmt', 'yuv420p', '-f', 'mp4', out], stdin=subprocess.PIPE)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = f[ry0:ry1, rx0:rx1].mean() / ref0
        crop = f[Y0:Y1, X0:X1].astype(np.float32)
        hp = crop - cv2.GaussianBlur(crop, (0, 0), sig)
        fill = clean[Y0:Y1, X0:X1] * g + hp * grain
        f[Y0:Y1, X0:X1] = np.clip(crop * (1 - m) + fill * m, 0, 255).astype(np.uint8)
        p.stdin.write(f.tobytes())
    cap.release()
    p.stdin.close()
    if p.wait():
        raise RuntimeError('platelock: ffmpeg encode failed')


def erase(src, box=None, ref=None, feather=0.01, start=0.0, grain=0.6, plate=None, mask=None, detail=1.0):
    """`gen: {erase: {box, ref}}` (frame fractions): paint a static object out of a blurred background area.
    `gen: {erase: {plate: <clean image>, mask: <grey image>}}`: a SHARP background (railings, boards) - the object is
    removed by a prepared clean plate (e.g. an image-editor fill of the plate frame, same framing); see _erase_plate."""
    if plate:
        return _erase_plate(os.path.abspath(src), os.path.abspath(plate), os.path.abspath(mask), round(float(start), 3),
                            float(detail))
    return _erase(os.path.abspath(src), [float(x) for x in box], [float(x) for x in ref], float(feather),
                  round(float(start), 3), float(grain))


@memo_file('global', '.mp4', version=VERSION + '-eraseplate3')
def _erase_plate(src, plate, mask, start, detail, out=None):
    """Inside `mask` (white = remove) every frame becomes clean_plate x light change + detail x the clip's own finest
    change (thin moving rain streaks, |d| clamped). The light change is measured on a ring AROUND the mask, never on the
    object (v1 took both from the object: its per-frame shimmer left a ghost outline). Static camera clips only."""
    from .render import ffmpeg
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    ok, p0 = cap.read()
    if not ok:
        raise RuntimeError(f'platelock: no plate frame in {src}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    p0 = p0.astype(np.float32)
    clean = cv2.resize(cv2.imread(plate), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(cv2.imread(mask, cv2.IMREAD_GRAYSCALE), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    ys, xs = np.nonzero(m > 0.002)
    Y0, Y1, X0, X1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = m[Y0:Y1, X0:X1, None]
    c, q = clean[Y0:Y1, X0:X1], p0[Y0:Y1, X0:X1]
    ring = ((cv2.dilate((m[..., 0] > 0.02).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & (m[..., 0] <= 0.02))
    rq = q[ring].mean(0) + 2.0
    fsig = max(1.0, 0.002 * min(W, H))
    # the object's own edges shimmer frame to frame (v2 left its sign lettering as a ghost): no detail on them
    gq = cv2.GaussianBlur(cv2.cvtColor(q.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    edges = cv2.magnitude(cv2.Sobel(gq, cv2.CV_32F, 1, 0), cv2.Sobel(gq, cv2.CV_32F, 0, 1))
    keep = 1 - cv2.GaussianBlur(cv2.dilate((edges > 18).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32),
                                (0, 0), 2)[..., None]
    vk = max(3, int(0.008 * H) | 1)   # rain falls vertically: smooth the detail along y (streaks stay, edges/dots fade)
    pr = subprocess.Popen([ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                           '-s', f'{W}x{H}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '12',
                           '-preset', 'medium', '-pix_fmt', 'yuv420p', '-f', 'mp4', out], stdin=subprocess.PIPE)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        crop = f[Y0:Y1, X0:X1].astype(np.float32)
        gain = np.clip((crop[ring].mean(0) + 2.0) / rq, 0.5, 2.0)   # light flicker from the surroundings
        d = crop - q
        d = np.clip(d - cv2.GaussianBlur(d, (0, 0), fsig), -25, 25)   # thin streaks only (rain), not the object
        d = cv2.blur(d, (1, vk)) * keep
        fill = c * gain + d * detail
        f[Y0:Y1, X0:X1] = np.clip(crop * (1 - m) + fill * m, 0, 255).astype(np.uint8)
        pr.stdin.write(f.tobytes())
    cap.release()
    pr.stdin.close()
    if pr.wait():
        raise RuntimeError('platelock: ffmpeg encode failed')


@memo_file('global', '.mp4', version=VERSION + '-retex')
def _retex(src, img, key, box, feather, start, flat=0.0, maxr=4.0, out=None):
    """Carry an edit made on the keyframe (e.g. lettering stamped in by an image editor) onto every frame of a clip:
    inside `box` each frame is multiplied by edited/original (per channel), so the clip's own light, flicker and
    grain stay and the edit sits in the surface. Static subjects only (desk, badge)."""
    from .render import ffmpeg
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    ok, plate = cap.read()
    if not ok:
        raise RuntimeError(f'platelock: no plate frame in {src}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    k0 = cv2.imread(key).astype(np.float32)
    e0 = cv2.imread(img).astype(np.float32)
    e0 = cv2.resize(e0, (k0.shape[1], k0.shape[0]), interpolation=cv2.INTER_AREA)
    g = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    win = cv2.createHanningWindow((k0.shape[1], k0.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(g(k0), g(e0), win)  # the editor may nudge the frame by a pixel or two
    e0 = cv2.warpAffine(e0, np.float32([[1, 0, -dx], [0, 1, -dy]]), (k0.shape[1], k0.shape[0]),
                        borderMode=cv2.BORDER_REFLECT)
    R = (cv2.GaussianBlur(e0, (0, 0), 0.6) + 4) / (cv2.GaussianBlur(k0, (0, 0), 0.6) + 4)
    R = np.clip(R, 0.15, maxr)  # maxr: bright raised lettering on a dark plate needs > 4x
    if flat:  # keep only detail finer than `flat` px: the editor's broad tint shifts would show as a box
        R = np.clip(R / np.maximum(cv2.GaussianBlur(R, (0, 0), flat), 1e-3), 0.15, maxr)
    # keyframe -> clip framing: scale to the clip, then align on the clip's own plate frame
    R = cv2.resize(R, (W, H), interpolation=cv2.INTER_CUBIC)
    kc = cv2.resize(k0, (W, H), interpolation=cv2.INTER_AREA)
    win2 = cv2.createHanningWindow((W, H), cv2.CV_32F)
    (sx, sy), _ = cv2.phaseCorrelate(g(kc), g(plate.astype(np.float32)), win2)
    R = cv2.warpAffine(R, np.float32([[1, 0, sx], [0, 1, sy]]), (W, H), borderValue=(1, 1, 1))
    m = _mask([box], W, H, feather)
    R = 1 + m * (R - 1)
    x0, y0, x1, y1 = _box(box, W, H)
    pad = int(feather * min(W, H) * 3) + 2
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    Rc = R[Y0:Y1, X0:X1]
    p = subprocess.Popen([ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{W}x{H}', '-r', f'{fps:.6f}', '-i', '-', '-c:v', 'libx264', '-crf', '12',
                          '-preset', 'medium', '-pix_fmt', 'yuv420p', '-f', 'mp4', out], stdin=subprocess.PIPE)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f[Y0:Y1, X0:X1] = np.clip(f[Y0:Y1, X0:X1].astype(np.float32) * Rc, 0, 255).astype(np.uint8)
        p.stdin.write(f.tobytes())
    cap.release()
    p.stdin.close()
    if p.wait():
        raise RuntimeError('platelock: ffmpeg encode failed')


def retex(src, img, key, box, feather=0.01, start=0.0, flat=0.0, maxr=4.0):
    """`gen: {retex: {img: edited keyframe, key: original keyframe, box}}` (frame fractions)."""
    return _retex(os.path.abspath(src), os.path.abspath(img), os.path.abspath(key), [float(x) for x in box],
                  float(feather), round(float(start), 3), float(flat),
                  **({'maxr': float(maxr)} if float(maxr) != 4.0 else {}))


@memo_file('global', '.mp4', version=VERSION + '-stamp6')
def _stamp(src, img, key, box, start, sigma=12.0, feather=0.006, follow=False, cover=False, ink=False, out=None):
    """Paste a keyframe edit (e.g. numerals stamped into a badge plate) onto a clip whose model redrew the scene a bit
    differently (2026-11-03: the light-ratio `retex` turned the plate cyan because the clip's badge is warmer and
    shifted). The edit is warped onto the clip's plate frame with a SIFT homography; per frame its broad colour/light
    is taken from the clip (low-pass clip / low-pass keyframe), its fine detail (the lettering) from the edit; small
    per-frame drift can be followed (`follow: true`) by phase correlation inside the box - OFF by default: on the
    rainy badge it jumped +-130 px per frame (the plate shook) while the badge itself moved < 1.3 px."""
    from .proc import writer
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    plate = _plate(src, int(round(start * fps)))
    kc = cv2.resize(cv2.imread(key), (W, H), interpolation=cv2.INTER_AREA)
    ec = cv2.resize(cv2.imread(img), (W, H), interpolation=cv2.INTER_AREA)
    g = lambda a: cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = _box(box, W, H)
    bw, bh = x1 - x0, y1 - y0
    roi = np.zeros((H, W), np.uint8)
    roi[max(0, y0 - 2 * bh):min(H, y1 + 2 * bh), max(0, x0 - bw // 2):min(W, x1 + bw // 2)] = 255
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(g(kc), roi)
    kb, db = sift.detectAndCompute(g(plate), roi)
    Hm = None
    if da is not None and db is not None and len(ka) > 10 and len(kb) > 10:
        mt = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
        if len(mt) >= 12:
            A = np.float32([ka[m.queryIdx].pt for m in mt]); B = np.float32([kb[m.trainIdx].pt for m in mt])
            Hm, inl = cv2.findHomography(A, B, cv2.RANSAC, 3.0)
    if Hm is None:
        Hm = np.eye(3)
    # refine on the plate itself (SIFT locks onto the emblem texture; the clip drew its plate a few px lower and the
    # old plate edge showed under the stamped one): affine ECC of the warped original keyframe vs the clip's plate
    # frame, edges only (blurred gradient magnitude), inside the box grown by half its size
    kW0 = cv2.warpPerspective(kc, Hm, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    edge = lambda a: cv2.GaussianBlur(cv2.magnitude(cv2.Sobel(cv2.GaussianBlur(g(a).astype(np.float32), (0, 0), 1.5),
                                                              cv2.CV_32F, 1, 0),
                                                    cv2.Sobel(cv2.GaussianBlur(g(a).astype(np.float32), (0, 0), 1.5),
                                                              cv2.CV_32F, 0, 1)), (0, 0), 2)
    em = np.zeros((H, W), np.uint8)
    em[max(0, y0 - bh // 2):min(H, y1 + bh // 2), max(0, x0 - bw // 4):min(W, x1 + bw // 4)] = 255
    try:
        A2 = np.eye(2, 3, dtype=np.float32)
        _, A2 = cv2.findTransformECC(edge(plate), edge(kW0), A2, cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), em, 5)
        Hm = np.vstack([cv2.invertAffineTransform(A2), [0, 0, 1]]) @ Hm
    except cv2.error:
        pass
    eW = cv2.warpPerspective(ec, Hm, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    kW = cv2.warpPerspective(kc, Hm, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    m0 = np.zeros((H, W), np.float32); m0[y0:y1, x0:x1] = 1
    # only where the edit changed the keyframe (the stamped plate), never the rest of the box
    dif = (np.abs(ec.astype(np.float32) - kc.astype(np.float32)).max(2) > 6).astype(np.uint8)
    dif = cv2.morphologyEx(dif, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m0 *= cv2.dilate(dif, np.ones((3, 3), np.uint8)).astype(np.float32)
    m0 = cv2.warpPerspective(m0, Hm, (W, H))
    ring = cv2.dilate((m0 > 0.5).astype(np.uint8), np.ones((25, 25), np.uint8)) - (m0 > 0.01).astype(np.uint8)
    rb = ring.astype(bool)
    # the clip's own (old) plate can be bigger/lower than the stamped one: cover the dark old plate too (filled with
    # the keyframe's metal from the warped edit), so no second plate edge shows under the new one
    if cover:
        gp = cv2.GaussianBlur(g(plate).astype(np.float32), (0, 0), 2)
        inner = (m0 > 0.5)
        # a tight band around the new plate only (a box-wide search took in the emblem and the badge's edge)
        grow = cv2.dilate(inner.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, bw // 12) | 1, max(3, bh // 3) | 1)))
        thr = np.percentile(gp[inner], 60) if inner.any() else 60
        old = ((gp < thr) & (grow > 0)).astype(np.uint8)
        old = cv2.morphologyEx(old, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(old)
        keep = np.zeros_like(old)
        for i in range(1, n):  # only dark blobs touching the new plate (the old plate), not shadows elsewhere
            blob = lab == i
            if (blob & cv2.dilate(inner.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)).any():
                keep |= blob.astype(np.uint8)
        keep = cv2.dilate(keep, np.ones((7, 7), np.uint8))
        m0 = np.maximum(m0, keep.astype(np.float32))
        cov = cv2.GaussianBlur((keep & ~inner).astype(np.float32), (0, 0), 3)[..., None]  # covered metal, not plate
    else:
        cov = None
    m0 = cv2.GaussianBlur(m0, (0, 0), max(1.0, feather * min(W, H)))[..., None]
    lk = cv2.GaussianBlur(kW, (0, 0), sigma if sigma > 0 else 12.0) + 4
    Rk = None
    if ink:  # per-channel ink ratio, but only where the edit is ink (darker); paper tone differences stay the clip's own
        Rc = np.clip((cv2.GaussianBlur(eW, (0, 0), 0.6) + 6) / (cv2.GaussianBlur(kW, (0, 0), 0.6) + 6), 0.05, 1.0)
        Rl = Rc @ np.float32([0.114, 0.587, 0.299])
        wi = np.clip((0.93 - Rl) / 0.12, 0, 1)[..., None]
        Rk = 1 - wi + Rc * wi
    ref = g(plate)[y0:y1, x0:x1].astype(np.float32)
    win = cv2.createHanningWindow((bw, bh), cv2.CV_32F)
    with writer(_x264(out, W, H, fps), out) as p:
        for f in _frames(src):
            (dx, dy), _ = cv2.phaseCorrelate(ref, g(f)[y0:y1, x0:x1].astype(np.float32), win) if follow else ((0.0, 0.0), 0)
            T = np.float32([[1, 0, dx], [0, 1, dy]])
            e, m = (cv2.warpAffine(eW, T, (W, H), borderMode=cv2.BORDER_REFLECT),
                    cv2.warpAffine(m0, T, (W, H))[..., None])
            lkk = cv2.warpAffine(lk, T, (W, H), borderMode=cv2.BORDER_REFLECT)
            ff = f.astype(np.float32)
            if ink:  # ink on paper: only darken/tint the clip's own page (rain shadows + lamp flicker pass over the text)
                Rw = cv2.warpAffine(Rk, T, (W, H), borderValue=(1, 1, 1))
                o = ff * (1 - m) + ff * Rw * m
                p.stdin.write(np.clip(o, 0, 255).astype(np.uint8).tobytes())
                continue
            if sigma > 0:
                tone = np.clip((cv2.GaussianBlur(ff, (0, 0), sigma) + 4) / lkk, 0.2, 5.0)
            else:  # one gain per channel from the metal around the plate: no local tint, keeps the frame's flicker
                rr = cv2.warpAffine(rb.astype(np.uint8), T, (W, H)).astype(bool)
                tone = ((ff[rr].mean(0) + 4) / (cv2.warpAffine(kW, T, (W, H), borderMode=cv2.BORDER_REFLECT)[rr].mean(0) + 4))
                tone = np.clip(tone, 0.3, 3.0)[None, None, :]
                if cov is not None:  # covered badge metal: the clip's local light (no plate there, so no tint risk)
                    loc = np.clip((cv2.GaussianBlur(ff, (0, 0), 12.0) + 4) / lkk, 0.2, 5.0)
                    tone = tone * (1 - cov) + loc * cov
            o = ff * (1 - m) + e * tone * m
            p.stdin.write(np.clip(o, 0, 255).astype(np.uint8).tobytes())


def stamp(src, img, key, box, start=0.0, sigma=12.0, feather=0.006, follow=False, cover=False, ink=False):
    """`gen: {stamp: {img: edited keyframe, key: original keyframe, box}}` (keyframe fractions): see _stamp."""
    return _stamp(os.path.abspath(src), os.path.abspath(img), os.path.abspath(key), [float(x) for x in box],
                  round(float(start), 3), float(sigma), float(feather), **({'follow': True} if follow else {}),
                  **({'cover': True} if cover else {}), **({'ink': True} if ink else {}))
