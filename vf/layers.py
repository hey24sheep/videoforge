"""Layer types. Each draw_* takes (canvas, layer settings for this format, shot-local time u, frame ctx)
and returns a NEW canvas (device float32 RGB). Scene layers work in linear light; text overlays are drawn on the
graded display-encoded frame (ctx['space'] == 'display'). All RGBA on the device is premultiplied."""
import glob, math, os
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from . import gpu
from .gpu import xp
from .spec import kf, resolve

FONT_DIRS = [r'C:\Windows\Fonts', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')]


# ------------------------------------------------------------------ sources (decoded once, cached)
def _nbytes(v):
    if isinstance(v, tuple):
        return sum(_nbytes(x) for x in v)
    return int(getattr(v, 'nbytes', 0) or 0)


class _LRU:
    """Bounded by entry count and (optionally) by bytes: interpolated 60 fps clips are ~0.5 GB each in RAM."""
    def __init__(self, n, max_bytes=None):
        self.d, self.n, self.max_bytes, self.bytes = OrderedDict(), n, max_bytes, 0

    def get(self, key, make):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = make()
        self.d[key] = v
        self.bytes += _nbytes(v)
        while len(self.d) > 1 and (len(self.d) > self.n or (self.max_bytes and self.bytes > self.max_bytes)):
            _, old = self.d.popitem(last=False)
            self.bytes -= _nbytes(old)
        return v

    def clear(self):
        self.d.clear()
        self.bytes = 0


# decoded clips / images in RAM, bounded by count and bytes (long episodes must not grow without limit)
_HOST = _LRU(8, max_bytes=int(float(os.environ.get('VF_HOST_CACHE_GB', '2.5')) * 2**30))
DEV = _LRU(96, max_bytes=int(float(os.environ.get('VF_DEV_CACHE_GB', '1.5')) * 2**30))  # frames on the GPU


def _mt(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def load_image(path):
    """uint8 RGB(A) (host)."""
    def make():
        im = Image.open(path)
        return np.asarray(im.convert('RGBA' if im.mode in ('RGBA', 'LA', 'P') else 'RGB'))
    return _HOST.get(('img', path, _mt(path)), make)


def _frame_files(path):
    if os.path.isdir(path):
        fs = sorted(glob.glob(os.path.join(path, 'rgba_*.png'))) or sorted(glob.glob(os.path.join(path, '*.png'))) \
            or sorted(glob.glob(os.path.join(path, '*.jpg')))
        return fs
    if any(c in path for c in '*?'):
        return sorted(glob.glob(path))
    return None


def load_video(path, max_frames=2000):
    """uint8 frames (N,H,W,C) from a frame folder/glob or a video file, plus its fps (None for folders)."""
    def make():
        fs = _frame_files(path)
        if fs is not None:
            return np.stack([np.asarray(Image.open(f).convert('RGBA' if f.endswith('.png') else 'RGB'))
                             for f in fs]), None
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        out = []
        while len(out) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if not out:
            raise FileNotFoundError(f'no frames decoded from {path}')
        return np.stack(out), fps
    return _HOST.get(('vid', path, _mt(path)), make)


def _dev_frame(path, idx):
    return DEV.get((path, _mt(path), idx), lambda: gpu.upload_rgba_u8(load_video(path)[0][idx]))


def _dev_image(path):
    return DEV.get((path, _mt(path), 'img'), lambda: gpu.upload_rgba_u8(load_image(path)))


# ------------------------------------------------------------------ helpers
def _placement(hs, ws, L, u, H, W):
    """(cx, cy, scale, rot) placing a (hs, ws) source: cover/contain fit * zoom, focus point at frame centre."""
    s = gpu.fit_scale(hs, ws, H, W, L.get('fit', 'cover')) * float(kf(L.get('zoom', 1.0), u))
    fx, fy = kf(L.get('focus', [0.5, 0.5]), u)
    if L.get('fit', 'cover') == 'cover':  # clamp the focus so a cover fit never shows past the source edge
        hw, hh = W / 2 / s, H / 2 / s
        fx = min(max(fx * ws, hw), ws - hw) / ws if ws > 2 * hw else 0.5
        fy = min(max(fy * hs, hh), hs - hh) / hs if hs > 2 * hh else 0.5
    return fx * ws, fy * hs, s, math.radians(float(kf(L.get('rotate', 0), u)))


def _place_full(canvas, src, L, u, H, W, ctx=None):
    cx, cy, s, rot = _placement(src.shape[0], src.shape[1], L, u, H, W)
    out = gpu.sample(src, H, W, cx, cy, s, rot=rot)
    if ctx is not None and (L.get('mask') or L.get('radial')):  # limit an image/clip layer to part of the frame
        m = _layer_mask(L, u, ctx)
        if m is not None:
            out = out * m
    return gpu.over(canvas, out, float(kf(L.get('opacity', 1.0), u)), L.get('blend', 'normal'))


def _layer_mask(L, u, ctx):
    """(H, W, 1) device mask from `mask:` (grayscale image, white = keep, stretched to the frame) times
    `radial: {center, radius, invert, falloff}` (soft gaussian spot, frame fractions; radius of the short side)."""
    H, W = ctx['H'], ctx['W']
    m = None
    if L.get('mask'):
        mp = resolve(ctx['spec'], L['mask'])

        def _load():
            import cv2
            g = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise FileNotFoundError(mp)
            return gpu.to_dev(np.ascontiguousarray(cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
                                                   .astype(np.float32) / 255.0))
        m = DEV.get(('mask', mp, os.path.getmtime(mp), W, H), _load)[..., None]
    rad = L.get('radial')
    if rad:
        X, Y = gpu._coords(H, W)
        cx, cy = kf(rad.get('center', [0.5, 0.5]), u)
        r = float(kf(rad.get('radius', 0.25), u)) * min(H, W)
        g = xp.exp(-((((X - cx * W) ** 2 + (Y - cy * H) ** 2) / (r * r)) ** float(rad.get('falloff', 1.0))))[..., None]
        if rad.get('invert'):
            g = 1 - g
        m = g if m is None else m * g
    return m


def _depth_canvas(L, u, ctx):
    """Depth map of a clip layer (`depth:` 16-bit png, near = bright) placed exactly like the clip, once per shot
    and format: the camera uses it for 2.5D parallax."""
    key = (ctx.get('_shot_id'), ctx.get('fmt'))
    store = ctx.setdefault('_depth', {})
    if key in store or not L.get('depth'):
        return
    p = resolve(ctx['spec'], L['depth'])
    im = np.asarray(Image.open(p)).astype(np.float32)
    im = im / (65535.0 if im.max() > 255 else 255.0)
    if im.ndim == 3:
        im = im[..., 0]
    d = gpu.to_dev(np.ascontiguousarray(im))
    src = resolve(ctx['spec'], L['src'])
    if L.get('type') == 'image':
        hs, ws = load_image(src).shape[:2]
    else:
        hs, ws = load_video(src)[0].shape[1:3]
    if d.shape != (hs, ws):  # depth may be at another resolution: map it in source pixel units
        sy, sx = d.shape[0] / hs, d.shape[1] / ws
        cx, cy, s, rot = _placement(hs, ws, L, u, ctx['H'], ctx['W'])
        store[key] = gpu.sample(d, ctx['H'], ctx['W'], cx * sx, cy * sy, s / sx, rot=rot)
    else:
        cx, cy, s, rot = _placement(hs, ws, L, u, ctx['H'], ctx['W'])
        store[key] = gpu.sample(d, ctx['H'], ctx['W'], cx, cy, s, rot=rot)


def _tint(rgba, L):
    """Recolour towards a tint, keeping luminance detail (premultiplied-safe)."""
    if not L.get('tint'):
        return rgba
    c = gpu.dev_hex(L['tint'])
    k = float(L.get('tint_amount', 0.5))
    out = rgba.copy()
    out[..., :3] = rgba[..., :3] * (1 - k) + rgba[..., :3].mean(-1, keepdims=True) * c * 2 * k
    return out


# ------------------------------------------------------------------ layer types
def draw_solid(canvas, L, u, ctx):
    """Flat colour; `opacity` (keyframable) fades it over the layers below (e.g. lights dying, fades to black).
    `radial: {center: [x, y], radius: r}` limits it to a soft gaussian spot (frame fractions; r of the short side);
    a bulb going out darkens the bulb and its glow, not the whole frame. `invert: true` = everything BUT the spot.
    `blend: light` makes the spot LIGHT the scene (canvas x (1 + colour x opacity)) instead of painting over it:
    spill from a practical (a neon sign, a lamp) - lit surfaces pick up the colour, black stays black.
    `mask: <grayscale image>` (episode-relative) limits it to the white parts of that image instead (soft edges kept)."""
    c = gpu.dev_hex(L.get('color', '#000000'))
    op = float(kf(L.get('opacity', 1.0), u))
    if L.get('mask'):  # grayscale image (white = affected), stretched to the frame: e.g. only a sign's light dies
        if op <= 0.001:
            return canvas
        H, W = ctx['H'], ctx['W']
        mp = resolve(ctx['spec'], L['mask'])

        def _load():
            import cv2
            g = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise FileNotFoundError(mp)
            return gpu.to_dev(np.ascontiguousarray(cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
                                                   .astype(np.float32) / 255.0))
        m = DEV.get(('mask', mp, os.path.getmtime(mp), W, H), _load)[..., None]
        if L.get('blend') == 'light':
            return canvas * (1 + c * (op * m))
        return canvas * (1 - op * m) + c * (op * m)
    rad = L.get('radial')
    if rad:
        if op <= 0.001:
            return canvas
        H, W = ctx['H'], ctx['W']
        X, Y = gpu._coords(H, W)
        cx, cy = kf(rad.get('center', [0.5, 0.5]), u)
        r = float(kf(rad.get('radius', 0.25), u)) * min(H, W)
        m = xp.exp(-(((X - cx * W) ** 2 + (Y - cy * H) ** 2) / (r * r)))[..., None]
        if rad.get('invert'):
            m = 1 - m
        if L.get('blend') == 'light':  # spill/practical light: lit surfaces pick up the colour, black stays black
            return canvas * (1 + c * (op * m))
        return canvas * (1 - op * m) + c * (op * m)
    if op >= 0.999:
        return xp.broadcast_to(c, canvas.shape).copy()
    return canvas * (1 - op) + c * op if op > 0.001 else canvas


def draw_gradient(canvas, L, u, ctx):
    H, W = ctx['H'], ctx['W']
    a, b = gpu.dev_hex(L.get('top', '#101820')), gpu.dev_hex(L.get('bottom', '#000000'))
    t = xp.linspace(0, 1, H, dtype=xp.float32)[:, None, None]
    out = xp.broadcast_to(a * (1 - t) + b * t, (H, W, 3)).copy()
    if L.get('glow'):
        gx, gy = kf(L.get('glow_at', [0.5, 0.4]), u)
        X, Y = gpu._coords(H, W)
        r = xp.sqrt(((X / W - gx) * W / H) ** 2 + (Y / H - gy) ** 2)[..., None]
        out += gpu.dev_hex(L['glow']) * xp.exp(-(r / float(L.get('glow_size', 0.35))) ** 2)
    return out


def draw_image(canvas, L, u, ctx):
    if L.get('depth'):
        _depth_canvas(L, u, ctx)
    src = _dev_image(resolve(ctx['spec'], L['src']))
    k = float(L.get('sharpen', 0.0))
    if k > 0:
        src = DEV.get(('sharp', resolve(ctx['spec'], L['src']), k), lambda: _unsharp(src, k, 1.0))
    return _place_full(canvas, _tint(src, L), L, u, ctx['H'], ctx['W'], ctx)


def _unsharp(a, k, sigma):
    b = gpu.blur_sep(a[..., :3], sigma)
    a = a.copy()
    a[..., :3] = xp.maximum(a[..., :3] + k * (a[..., :3] - b), 0)
    return a


_RAIN = {}
RAIN_TILE = 512


def _rain_tile(seed, density, length, width, angle):
    """Tileable RGBA rain-streak texture (straight alpha, white), RAIN_TILE px square, made once per setting."""
    key = (seed, density, length, width, angle)
    if key not in _RAIN:
        rng = np.random.default_rng(seed)
        S = RAIN_TILE
        im = Image.new('L', (S, S), 0)
        d = ImageDraw.Draw(im)
        n = int(density * 900 * (S / 1024) ** 2)
        dx = math.tan(math.radians(angle))
        for _ in range(n):
            x, y = rng.uniform(0, S), rng.uniform(0, S)
            ln = length * S * rng.uniform(0.5, 1.2)
            a = int(255 * rng.uniform(0.25, 1.0))
            for ox in (-S, 0, S):  # wrap so the tile repeats seamlessly
                for oy in (-S, 0, S):
                    d.line([(x + ox, y + oy), (x + ox + dx * ln, y + oy + ln)], fill=a, width=max(1, int(width)))
        im = im.filter(ImageFilter.GaussianBlur(0.6))
        al = np.asarray(im).astype(np.float32) / 255
        rgba = np.dstack([np.ones_like(al)] * 3 + [al])
        _RAIN[key] = gpu.to_dev(np.ascontiguousarray((rgba * 255).astype(np.uint8)))
    return _RAIN[key]


def draw_rain(canvas, L, u, ctx):
    """Procedural rain over a shot (for camera-moved stills, or to thicken weather): `layers` depth planes of
    streaks falling at different speeds/sizes (near = bigger, faster, brighter, slightly blurred).
    {type: rain, intensity: 0.7, angle: 8, speed: 1.0, opacity: 0.35, layers: 3, seed: 1, color: '#cfd8e0'}"""
    H, W = ctx['H'], ctx['W']
    inten = float(kf(L.get('intensity', 0.7), u))
    op = float(kf(L.get('opacity', 0.35), u))
    if inten <= 0 or op <= 0:
        return canvas
    ang = float(L.get('angle', 8))
    col = gpu.dev_hex(L.get('color', '#cfd8e0'))
    nl = int(L.get('layers', 3))
    out = canvas
    for j in range(nl):
        z = (j + 1) / nl                      # 0..1 far -> near
        tile = _rain_tile(int(L.get('seed', 1)) * 7 + j, round(inten * (1.0 - 0.55 * z), 3),
                          0.035 + 0.05 * z, 1, ang)  # thin 1-px streaks: real rain reads as fine lines
        S = RAIN_TILE
        # premultiplied RGBA tile repeated 3x3: any view centre in [S, 2S) stays inside it at these scales
        big = DEV.get(('rainbig', id(tile)), lambda: xp.tile(
            gpu.upload_rgba_u8(gpu.to_host(tile), linear=False), (3, 3, 1)))
        scale = (0.9 + 1.1 * z) * max(H, W) / S / 2.4
        v = float(L.get('speed', 1.0)) * (1.1 + 2.2 * z) * H  # output px per second
        oy = u * v / scale
        ox = u * v * math.tan(math.radians(ang)) / scale
        cx, cy = S + (-ox) % S, S + (-oy) % S          # falling: the texture moves down/sideways
        lay = gpu.sample(big, H, W, cx, cy, scale)
        a = lay[..., 3:4] * op * (0.45 + 0.55 * z)
        out = out * (1 - a * 0.15) + col * a  # light streaks, slight occlusion
    return out


def _video_pos(path, L, u):
    """(i0, i1, w): frame i0 blended with i1 by weight w. Stepped styles hold drawings (w = 0). Smooth styles blend
    the two neighbouring frames instead of holding the nearest one (nearest-frame holds at 24->60 fps gave an
    uneven 2:3 cadence that read as shake); retimed (RIFE) clips land on whole frames anyway."""
    frames, file_fps = load_video(path)
    fps = float(L.get('fps') or file_fps or 24)
    t = float(L.get('start', 0)) + u * float(L.get('speed', 1.0))
    x = t * fps
    n = len(frames)
    step = int(L.get('step', 1))  # hold each drawing `step` frames ("on twos" stop-motion / limited animation)
    if step > 1 or not L.get('blend_frames', True):
        i = int(math.floor(x + 1e-6))
        if step > 1:
            i = i // step * step
        i = i % n if L.get('loop', False) else min(max(i, 0), n - 1)
        return i, i, 0.0
    i0 = int(math.floor(x + 1e-4))
    w = x - i0
    if w < 0.02:
        w = 0.0
    elif w > 0.98:
        i0, w = i0 + 1, 0.0
    if L.get('loop', False):
        return i0 % n, (i0 + 1) % n, w
    i0 = min(max(i0, 0), n - 1)
    i1 = min(i0 + 1, n - 1)
    return i0, i1, (w if i1 != i0 else 0.0)


def _video_index(path, L, u):
    """Hashable frame state (prefix cache key): equal values draw identical pixels."""
    i0, i1, w = _video_pos(path, L, u)
    return (i0, i1, round(w, 3)) if w else i0


def _video_frame(path, L, u):
    i0, i1, w = _video_pos(path, L, u)
    a = _dev_frame(path, i0)
    a = a if not w else a * (1 - w) + _dev_frame(path, i1) * w
    k = float(L.get('sharpen', 0.0))
    if k > 0:  # unsharp mask at SOURCE resolution (cheap): restores detail lost when 480p clips are scaled up
        b = gpu.blur_sep(a[..., :3], float(L.get('sharpen_sigma', 1.0)))
        a = a.copy()
        a[..., :3] = xp.maximum(a[..., :3] + k * (a[..., :3] - b), 0)
    return a


def draw_video(canvas, L, u, ctx):
    path = resolve(ctx['spec'], L['src'])
    if L.get('depth'):
        _depth_canvas(L, u, ctx)
    return _place_full(canvas, _tint(_video_frame(path, L, u), L), L, u, ctx['H'], ctx['W'], ctx)


def draw_sprite(canvas, L, u, ctx):
    """Cut-out (RGBA png or frame folder). x,y = frame fractions of the anchor point; h = height fraction."""
    H, W = ctx['H'], ctx['W']
    path = resolve(ctx['spec'], L['src'])
    src = _video_frame(path, L, u) if (os.path.isdir(path) or '*' in path) else _dev_image(path)
    hs, ws = src.shape[:2]
    s = float(kf(L.get('h', 0.6), u)) * H / hs
    ax, ay = L.get('anchor', [0.5, 1.0])
    x, y = float(kf(L.get('x', 0.5), u)) * W, float(kf(L.get('y', 1.0), u)) * H
    mirror = L.get('mirror', False)
    if mirror:
        src = src[:, ::-1]
        ax = 1 - ax
    out = gpu.sample(src, H, W, ax * ws, ay * hs, s, rot=math.radians(float(kf(L.get('rotate', 0), u))), ox=x, oy=y)
    out = _tint(out, L)
    if L.get('shadow', True):  # soft contact shadow under the anchor
        X, Y = gpu._coords(H, W)
        sw = ws * s * 0.35
        sh_ = xp.exp(-(((X - x) / max(sw, 1)) ** 2 + ((Y - y) / max(sw * 0.12, 1)) ** 2))[..., None]
        canvas = canvas * (1 - 0.45 * sh_ * float(kf(L.get('opacity', 1.0), u)))
    return gpu.over(canvas, out, float(kf(L.get('opacity', 1.0), u)), L.get('blend', 'normal'))


def _font(name, px):
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    return ImageFont.truetype(name, px)


_TXT = _LRU(256)


def text_bitmap(*args):
    k = ('txt',) + args
    return _TXT.get(k, lambda: _text_bitmap(*args))


def _text_bitmap(text, font, px, color, stroke, stroke_color, shadow, max_w, align='center', spacing=1.15, plate=0,
                 soft=0.0):
    im = _text_bitmap0(text, font, px, color, stroke, stroke_color, shadow, max_w, align, spacing, plate)
    if soft > 0:  # in-scene props: writing in fog, ink bleeding into paper, a stamp under glass
        pim = Image.fromarray(im)
        pad = int(soft * 3) + 2
        big = Image.new('RGBA', (pim.width + 2 * pad, pim.height + 2 * pad), (0, 0, 0, 0))
        big.paste(pim, (pad, pad))
        im = np.asarray(big.filter(ImageFilter.GaussianBlur(soft)))
    return im


def _text_bitmap0(text, font, px, color, stroke, stroke_color, shadow, max_w, align='center', spacing=1.15, plate=0):
    """uint8 RGBA (straight alpha, display-encoded colour) of wrapped text."""
    f = _font(font, px)
    d0 = ImageDraw.Draw(Image.new('L', (8, 8)))
    lines = []
    for para in text.split('\n'):
        cur = ''
        for w in para.split(' '):
            t = (cur + ' ' + w).strip()
            if cur and d0.textlength(t, font=f) > max_w:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
    lh = int(px * spacing)
    tw = int(max(d0.textlength(l, font=f) for l in lines)) + 2 * stroke + 8
    th = lh * len(lines) + 2 * stroke + int(px * 0.4)
    pad = int(px * 0.3) if shadow else 4
    im = Image.new('RGBA', (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    if shadow:
        sh = Image.new('RGBA', im.size, (0, 0, 0, 0))
        ds = ImageDraw.Draw(sh)
    d = ImageDraw.Draw(im)
    if plate:  # soft dark backing so text reads on any background
        pl = Image.new('RGBA', im.size, (0, 0, 0, 0))
        ImageDraw.Draw(pl).rounded_rectangle([pad // 2, pad // 2, im.width - pad // 2, im.height - pad // 2],
                                             radius=int(px * 0.35), fill=(0, 0, 0, int(255 * plate)))
        im = Image.alpha_composite(im, pl.filter(ImageFilter.GaussianBlur(px * 0.08)))
        d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        lw = d.textlength(l, font=f)
        x = pad + stroke + ((tw - 2 * stroke - lw) / 2 if align == 'center' else 0)
        y = pad + stroke + i * lh
        if shadow:
            ds.text((x + px * 0.04, y + px * 0.06), l, font=f, fill=(0, 0, 0, 200), stroke_width=stroke,
                    stroke_fill=(0, 0, 0, 200))
        d.text((x, y), l, font=f, fill=color, stroke_width=stroke, stroke_fill=stroke_color)
    if shadow:
        sh = sh.filter(ImageFilter.GaussianBlur(px * 0.06))
        im = Image.alpha_composite(sh, im)
    return np.asarray(im)


def _hex8(h, a=255):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


# ------------------------------------------------------------------ realistic in-scene text (2026-11-03)
# User: "the text looks like a digital block placed on things". `style:` on a text layer:
#   ink   - handwriting: per-glyph jitter (rotation, baseline, size), pen pressure, ink bleed; use blend: multiply
#           so the paper's own texture and light show through the ink.
#   typed - typewriter: baseline/darkness jitter per key strike, slight bleed, no rotation; blend: multiply.
#   fog   - traced by a finger in condensation: the strokes WIPE the fog (glass behind shows darker and sharper),
#           water collects in a beaded rim, drips grow down from the strokes after the writing.
def _noise(rng, h, w, cell):
    g = rng.random((max(2, int(h / cell) + 2), max(2, int(w / cell) + 2))).astype(np.float32)
    return np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255


def _glyph_mask(text, font, px, max_w, align, spacing, jit, rot_deg, seed, typed=False):
    """float32 (h,w) alpha of text drawn one glyph at a time with jitter, plus a per-glyph darkness map."""
    rng = np.random.default_rng(seed)
    f = _font(font, px)
    lines = []
    d0 = ImageDraw.Draw(Image.new('L', (8, 8)))
    for para in text.split('\n'):
        cur = ''
        for w in para.split(' '):
            t = (cur + ' ' + w).strip()
            if cur and d0.textlength(t, font=f) > max_w:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
    lh = int(px * spacing)
    pad = int(px * 0.5)
    tw = int(max(d0.textlength(l, font=f) for l in lines) * 1.08) + 2 * pad
    th = lh * len(lines) + int(px * 0.5) + 2 * pad
    A = np.zeros((th, tw), np.float32)
    D = np.ones((th, tw), np.float32)
    for i, l in enumerate(lines):
        lw = d0.textlength(l, font=f) * 1.04
        x = pad + ((tw - 2 * pad - lw) / 2 if align == 'center' else 0)
        drift = 0.0
        for ch in l:
            adv = f.getlength(ch)
            if ch.strip():
                sz = max(6, int(px * (1 + rng.normal(0, 0.045 * jit))))
                fc = _font(font, sz)
                g = Image.new('L', (int(sz * 2), int(sz * 2)), 0)
                ImageDraw.Draw(g).text((sz * 0.5, sz * 0.35), ch, font=fc, fill=255)
                if rot_deg:
                    g = g.rotate(rng.normal(0, rot_deg * jit), resample=Image.BICUBIC, center=(sz, sz))
                drift += rng.normal(0, 0.015 * jit) * px
                gy = int(pad + i * lh - sz * 0.35 + rng.normal(0, 0.04 * jit) * px + drift)
                gx = int(x - sz * 0.5 + rng.normal(0, 0.03 * jit) * px)
                a = np.asarray(g, np.float32) / 255
                y0, x0 = max(0, gy), max(0, gx)
                y1, x1 = min(th, gy + a.shape[0]), min(tw, gx + a.shape[1])
                if y1 > y0 and x1 > x0:
                    sub = a[y0 - gy:y1 - gy, x0 - gx:x1 - gx]
                    A[y0:y1, x0:x1] = np.maximum(A[y0:y1, x0:x1], sub)
                    dk = rng.uniform(0.72, 1.0) if typed else rng.uniform(0.88, 1.0)
                    D[y0:y1, x0:x1] = np.where(sub > 0.05, dk, D[y0:y1, x0:x1])
            x += adv * (1.04 + rng.normal(0, 0.03 * jit))
    return A, D


def _ink_bitmap(text, font, px, color, max_w, align, spacing, jit, seed, typed):
    """uint8 straight-alpha RGBA of handwritten (or typewritten) ink."""
    A, D = _glyph_mask(text, font, px, max_w, align, spacing, jit, 0 if typed else 4.0, seed, typed)
    rng = np.random.default_rng(seed + 1)
    h, w = A.shape
    press = 0.62 + 0.38 * _noise(rng, h, w, px * 0.45)  # pen pressure / ribbon wear along the strokes
    core = A * press * D
    bleed = np.asarray(Image.fromarray((A * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(0.6, px * (0.035 if not typed else 0.02)))), np.float32) / 255
    fib = 0.75 + 0.25 * _noise(rng, h, w, max(2, px * 0.06))  # feathering along paper fibres
    a = np.clip(np.maximum(core, bleed * 0.38 * fib), 0, 1)
    a = np.clip(a * 1.05, 0, 1) ** 0.9
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = color[:3]
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def _fog_maps(text, font, px, max_w, align, spacing, seed, drip_len):
    """float32 (h,w,4): [wiped stroke, beaded rim, drip trail, drip progress 0..1]."""
    A, _ = _glyph_mask(text, font, px, max_w, align, spacing, 0.6, 3.0, seed)
    rng = np.random.default_rng(seed + 7)
    pad_b = int(drip_len * px)
    h0, w = A.shape
    A = np.pad(A, ((0, pad_b), (0, 0)))
    h = A.shape[0]
    im = Image.fromarray((A * 255).astype(np.uint8))
    # a fingertip is wider and rounder than a font stroke
    fat = im.filter(ImageFilter.MaxFilter(max(3, int(px * 0.06) | 1))).filter(ImageFilter.GaussianBlur(px * 0.025))
    S = np.clip(np.asarray(fat, np.float32) / 255 * 1.3, 0, 1)
    wide = np.asarray(Image.fromarray((S * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(max(3, int(px * 0.05) | 1))).filter(ImageFilter.GaussianBlur(px * 0.02)),
        np.float32) / 255
    rim = np.clip(wide - S, 0, 1)
    rim = rim * (0.55 + 0.45 * _noise(rng, h, w, px * 0.08))
    beads = Image.new('L', (w, h), 0)
    db = ImageDraw.Draw(beads)
    ys, xs = np.nonzero(rim > 0.25)
    if len(ys):
        for k in rng.choice(len(ys), size=min(len(ys), int(len(text) * 14)), replace=False):
            r = rng.uniform(0.008, 0.02) * px
            db.ellipse([xs[k] - r, ys[k] - r, xs[k] + r, ys[k] + r], fill=int(rng.uniform(150, 255)))
    rim = np.maximum(rim, np.asarray(beads.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255)
    # drips: water gathers at the lowest points of strokes and runs down
    trail = Image.new('L', (w, h), 0)
    prog = np.zeros((h, w), np.float32)
    dt = ImageDraw.Draw(trail)
    cols = np.nonzero(S[:h0].max(0) > 0.5)[0]
    n_drip = max(2, min(7, len(text.strip()) + 1))
    picks = rng.choice(cols, size=min(n_drip, len(cols)), replace=False) if len(cols) and drip_len > 0 else []
    for x in picks:
        y0 = int(np.nonzero(S[:h0, x] > 0.5)[0].max())
        ln = rng.uniform(0.35, 1.0) * drip_len * px
        wd = rng.uniform(0.018, 0.03) * px
        xx, y = float(x), y0
        for t in range(int(ln)):
            xx += rng.normal(0, 0.15)
            y = y0 + t
            if y >= h:
                break
            dt.ellipse([xx - wd, y - wd, xx + wd, y + wd], fill=255)
            x0i, x1i = max(0, int(xx - wd - 1)), min(w, int(xx + wd + 2))
            seg = prog[y, x0i:x1i]
            prog[y, x0i:x1i] = np.where(seg > 0, seg, t / (drip_len * px) + 1e-3)
        dt.ellipse([xx - wd * 1.8, y - wd * 1.8, xx + wd * 1.8, y + wd * 1.8], fill=255)  # bead at the tip
        prog[max(0, int(y - wd * 2)):int(y + wd * 2) + 1, max(0, int(xx - wd * 2)):int(xx + wd * 2) + 1] = ln / (drip_len * px)
    T = np.asarray(trail.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
    prog = np.asarray(Image.fromarray(prog).filter(ImageFilter.MaxFilter(5)), np.float32)
    return np.stack([S, rim, T, prog], -1).astype(np.float32)


def _emboss_map(text, font, px, max_w, align, spacing, light_deg, depth):
    """float32 (h,w,4): [side-light shading -1..1 (lit bevels +, grooves -), raised face 0..1, 0, 0] for digits
    stamped up out of a metal plate."""
    A, _ = _glyph_mask(text, font, px, max_w, align, spacing, 0.0, 0.0, 1)
    A = np.pad(A, int(px * 0.15))
    hgt = np.asarray(Image.fromarray((A * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(px * 0.035)),
                     np.float32) / 255 * depth
    gy, gx = np.gradient(hgt)
    th = math.radians(light_deg)
    sh = -(gx * math.cos(th) + gy * math.sin(th)) * px * 0.09
    sh = np.clip(sh, -1, 1)
    # soft contact shadow just outside the digits, away from the light
    sx, sy = int(round(-math.cos(th) * px * 0.03)), int(round(-math.sin(th) * px * 0.03))
    drop = np.roll(np.roll(hgt, sy, 0), sx, 1)
    sh = sh - 0.35 * np.clip(drop - hgt, 0, 1)
    face = np.clip(A, 0, 1)
    z = np.zeros_like(sh)
    return np.stack([sh, face, z, z], -1).astype(np.float32)


def _breath(L, u, H, W, x, y, px):
    """Breath condensation mask (H,W,1) on the GPU: a misty bloom that spreads fast from (x, y), holds, then
    evaporates from its rim inwards (`anim: breath`, user 2026-11-03: the ghost's breath reveals the number)."""
    at = float(L.get('at', 0))
    t = u - at
    grow = min(1.0, max(0.0, t / max(0.05, float(L.get('in', 0.8)))))
    r = (1 - (1 - grow) ** 3) * float(L.get('breath_r', 3.2)) * px
    ev_at, ev = float(L.get('evap_at', 2.4)), float(L.get('evap', 1.6))
    if t > ev_at:
        r *= max(0.0, 1 - ((t - ev_at) / ev) ** 1.3)
    if r <= 1:
        return None
    key = ('breath_noise', H, W)
    nz = DEV.get(key, lambda: gpu.to_dev(np.asarray(Image.fromarray(
        (np.random.default_rng(5).random((H // 24 + 2, W // 24 + 2)) * 255).astype(np.uint8)).resize(
        (W, H), Image.BICUBIC), np.float32) / 255))
    yy, xx = xp.ogrid[0:H, 0:W]
    d = xp.sqrt(((xx - x) / 1.25) ** 2 + (yy - y) ** 2) / r + (nz - 0.5) * 0.35  # wider than tall, ragged rim
    return xp.clip((1.0 - d) / 0.35, 0, 1)[..., None] ** 1.5


def _draw_fog(canvas, L, u, ctx, text, font, px, x, y, rot, op, k_in):
    H, W = ctx['H'], ctx['W']
    if L.get('anim') == 'breath':
        # breath on glass that was written on earlier: the fresh mist beads everywhere except where the finger left
        # grease, so the strokes show up clear inside the bloom and vanish again as it evaporates
        B = _breath(L, u, H, W, x, y, px)
        if B is None:
            return canvas
        args = ('fog', text, font, px, int(float(L.get('max_w', 0.86)) * W), L.get('align', 'center'),
                float(L.get('spacing', 1.15)), int(L.get('seed', 3)), 0.0)
        src = DEV.get(args, lambda: gpu.to_dev(_fog_maps(*args[1:])))
        hs, ws = src.shape[:2]
        m = gpu.sample(src, H, W, ws / 2, hs / 2, 1.0, rot=rot, ox=x, oy=y)
        small = gpu._down(canvas, 8)
        hz = gpu.blur_sep(small, 3.0)
        haze = gpu.sample(hz, H, W, hz.shape[1] / 2, hz.shape[0] / 2, H / hz.shape[0])
        mist = float(L.get('mist', 0.55))
        fogged = canvas * (1 - mist * B) + (haze * 1.3 + 0.03) * mist * B  # milky, softened glass
        # soft stroke edge (a fingertip, not a pen): blur the wipe mask a little at 1/2 res
        w0 = m[..., 0:1]
        ws_ = gpu.blur_sep(gpu._down(w0, 2), max(1.0, px / 90))
        w0 = gpu.sample(ws_, H, W, ws_.shape[1] / 2, ws_.shape[0] / 2, H / ws_.shape[0])
        wipe = w0 * B * op
        # 2026-11-03 review: the old 'canvas - haze' made the strokes near-black = marker pen. A cleared stroke is
        # just clear glass: the lit interior seen sharp, a little darker than the milky fog around it
        a = float(L.get('clear', 0.6))
        behind = canvas * (1 - 0.45 * a)
        out = fogged * (1 - wipe) + behind * wipe
        return out + m[..., 1:2] * B * op * float(L.get('rim', 0.35)) * (haze + 0.02)
    drip_len = float(L.get('drip', 0.9))
    args = ('fog', text, font, px, int(float(L.get('max_w', 0.86)) * W), L.get('align', 'center'),
            float(L.get('spacing', 1.15)), int(L.get('seed', 3)), drip_len)
    src = DEV.get(args, lambda: gpu.to_dev(_fog_maps(*args[1:])))
    hs, ws = src.shape[:2]
    pad_b = int(drip_len * px)
    # anchor = centre of the text block (the drip padding hangs below it)
    m = gpu.sample(src, H, W, ws / 2, (hs - pad_b) / 2, 1.0, rot=rot, ox=x, oy=y)
    t_drip = u - float(L.get('at', 0)) - float(L.get('in', 0.3)) - 0.2
    f = max(0.0, min(1.0, t_drip / float(L.get('drip_time', 2.2)))) ** 0.75
    trail = m[..., 2] * ((m[..., 3] > 0) & (m[..., 3] <= f)) if t_drip > 0 else 0 * m[..., 2]
    wipe = xp.clip(m[..., 0] + 0.9 * trail, 0, 1)[..., None] * op
    # local haze level from a low-res blur of the fogged frame
    k = 8
    small = gpu._down(canvas, k)
    hz = gpu.blur_sep(small, 3.0)
    haze = gpu.sample(hz, H, W, hz.shape[1] / 2, hz.shape[0] / 2, H / hz.shape[0])
    a = float(L.get('clear', 0.6))
    # behind the wipe: the fog's glow is gone (darker, same hue) and the detail is sharp
    behind = canvas * (1 - a) + (canvas - haze) * 0.8
    out = canvas * (1 - wipe) + xp.clip(behind, 0, None) * wipe
    rim = m[..., 1:2] * op * k_in
    return out + rim * float(L.get('rim', 0.35)) * (haze + 0.02)



def draw_text(canvas, L, u, ctx):
    H, W = ctx['H'], ctx['W']
    at, until = float(L.get('at', 0)), float(L.get('until', 1e9))
    if u < at or u > until:
        return canvas
    m = min(W, H)
    px = max(8, int(float(L.get('size', 0.06)) * m))
    fonts = ctx['spec']['fonts']
    font = fonts.get(L.get('font', 'title'), L.get('font', 'title'))
    text = L['text']
    anim = L.get('anim', 'fade')
    ti, to = float(L.get('in', 0.3)), float(L.get('out', 0.3))
    k_in = min(1.0, (u - at) / ti) if ti > 0 else 1.0
    k_out = min(1.0, (until - u) / to) if to > 0 and until < 1e8 else 1.0
    if anim == 'type':
        n = int(len(text) * min(1.0, (u - at) / max(0.05, float(L.get('type_time', 0.04)) * len(text))))
        text = text[:n] or ' '
    args = (text, font, px, _hex8(L.get('color', '#ffffff')), int(px * float(L.get('stroke', 0.0))),
            _hex8(L.get('stroke_color', '#000000')), bool(L.get('shadow', True)), int(float(L.get('max_w', 0.86)) * W),
            L.get('align', 'center'), float(L.get('spacing', 1.15)), float(L.get('plate', 0)),
            float(L.get('soft', 0.0)) * px)
    linear = ctx.get('space', 'linear') == 'linear'
    style = L.get('style')
    if style == 'fog':
        x, y = float(kf(L.get('x', 0.5), u)) * W, float(kf(L.get('y', 0.5), u)) * H
        rot = math.radians(float(kf(L.get('rotate', 0.0), u)))
        op = float(kf(L.get('opacity', 1.0), u)) * k_out
        return _draw_fog(canvas, L, u, ctx, text, font, px, x, y, rot, op, k_in)
    if style == 'emboss':
        x, y = float(kf(L.get('x', 0.5), u)) * W, float(kf(L.get('y', 0.5), u)) * H
        rot = math.radians(float(kf(L.get('rotate', 0.0), u)))
        op = float(kf(L.get('opacity', 1.0), u)) * k_in * k_out
        eargs = ('emboss', text, font, px, int(float(L.get('max_w', 0.86)) * W), L.get('align', 'center'),
                 float(L.get('spacing', 1.15)), float(L.get('light', 135.0)), float(L.get('depth', 1.0)))
        src = DEV.get(eargs, lambda: gpu.to_dev(_emboss_map(*eargs[1:])))
        hs, ws = src.shape[:2]
        s = gpu.sample(src, H, W, ws / 2, hs / 2, 1.0, rot=rot, ox=x, oy=y)
        # shading MULTIPLIES the plate: the digits inherit the metal's own colour, grain and light
        k = float(L.get('strength', 0.9)) * op
        return canvas * xp.clip(1 + k * s[..., 0:1] + float(L.get('face', 0.12)) * op * s[..., 1:2], 0.05, None)
    if style in ('ink', 'typed'):
        iargs =(text, font, px, _hex8(L.get('color', '#1e1a17')), int(float(L.get('max_w', 0.86)) * W),
                 L.get('align', 'center'), float(L.get('spacing', 1.15)), float(L.get('jitter', 1.0)),
                 int(L.get('seed', 3)), style == 'typed')
        src = DEV.get(('ink', linear) + iargs, lambda: gpu.upload_rgba_u8(_ink_bitmap(*iargs), linear=linear))
    else:
        # texture keyed by its arguments (an id() key could be recycled and draw the wrong text)
        src = DEV.get(('txt', linear) + args, lambda: gpu.upload_rgba_u8(text_bitmap(*args), linear=linear))
    hs, ws = src.shape[:2]
    scale = 1.0
    dy = 0.0
    if anim == 'pop':
        e = 1 - (1 - k_in) ** 3
        scale = 0.6 + 0.4 * e + 0.08 * math.sin(e * math.pi)
    elif anim == 'slide':
        dy = (1 - (1 - (1 - k_in) ** 3)) * 0.04 * H
    x, y = float(kf(L.get('x', 0.5), u)) * W, float(kf(L.get('y', 0.5), u)) * H + dy
    # `anchor: [ax, ay]` = which point of the text box sits at (x, y): [0.5, 0.5] centre (default), [1, 1] bottom-right
    ax, ay = L.get('anchor', (0.5, 0.5))
    x, y = x - (float(ax) - 0.5) * ws, y - (float(ay) - 0.5) * hs
    rot = math.radians(float(kf(L.get('rotate', 0.0), u)))
    if scale == 1.0 and not rot:  # land texels on whole pixels: no half-pixel softening of text
        x, y = round(x - ws / 2) + ws / 2, round(y - hs / 2) + hs / 2
    out = gpu.sample(src, H, W, ws / 2, hs / 2, scale, rot=rot, ox=x, oy=y)
    op = float(kf(L.get('opacity', 1.0), u)) * (k_in if anim in ('fade', 'slide', 'pop') else 1.0) * k_out
    return gpu.over(canvas, out, op, L.get('blend', 'normal'))


def draw_glitch(canvas, L, u, ctx):
    """Analog picture tear (a failing video/phone line), applied to everything below: `strength` (keyframable,
    0 = off) shifts horizontal bands sideways, rolls the picture a little, adds scan noise and lifts black a touch.
    Monochrome on purpose (analog tape/line), no RGB blocks. `seed` picks the band pattern; it changes every frame.
    Tuning (defaults = the strong tear): `bands` (count scale), `shift` (max band shift, frame widths),
    `wobble` (per-line sideways ripple, frame widths), `roll` (vertical slip, frame heights), `noise` (scan noise)."""
    k = float(kf(L.get('strength', 0.0), u))
    if k <= 0.002:
        return canvas
    H, W = ctx['H'], ctx['W']
    fps = float(ctx.get('fps', 60) or 60)
    rng = np.random.default_rng(int(L.get('seed', 0)) * 100003 + int(u * fps))
    shift = np.zeros(H, np.float32)
    nb = float(L.get('bands', 1.0))
    for _ in range(int((3 + 6 * k) * nb)):  # torn bands: big slow ones plus thin fast slivers
        y0 = int(rng.uniform(0, H)); h = int(rng.uniform(0.004, 0.09 + 0.12 * k) * H)
        shift[y0:y0 + h] += rng.uniform(-1, 1) * k * float(L.get('shift', 0.22)) * W
    shift += np.sin(np.arange(H) / H * rng.uniform(4, 14) + rng.uniform(0, 6.3)).astype(np.float32) * k * float(L.get('wobble', 0.012)) * W
    roll = int(rng.uniform(-1, 1) * k * k * float(L.get('roll', 0.06)) * H)
    X = xp.arange(W, dtype=xp.int32)[None, :]
    sh = xp.asarray(np.roll(shift, roll).astype(np.int32))[:, None]
    rows = xp.asarray(np.roll(np.arange(H), roll))
    out = canvas[rows[:, None], (X - sh) % W]
    noise = xp.asarray(rng.uniform(0, 1, (H, 1)).astype(np.float32))
    nz = float(L.get('noise', 1.0))
    lines = (noise > 1 - 0.08 * k).astype(xp.float32) * 0.35 * k * nz
    grain = xp.asarray(rng.normal(0, 0.05 * k * nz, (H // 4, W // 4)).astype(np.float32))
    grain = xp.repeat(xp.repeat(grain, 4, 0), 4, 1)[:H, :W, None]
    out = out * (1 - 0.25 * k) + lines[..., None] + grain + 0.02 * k
    return xp.maximum(out, 0)


_DROPS = {}


def _drop_list(L, ctx):
    """Deterministic raindrop impacts (t, x, y) for 0..120 s, positions sampled inside `mask` (white = surface)."""
    H, W = ctx['H'], ctx['W']
    key = (L.get('seed', 0), L.get('mask'), float(L.get('rate', 20)), float(L.get('life', 0.45)), W, H)
    if key in _DROPS:
        return _DROPS[key]
    rng = np.random.default_rng(int(L.get('seed', 0)))
    if L.get('mask'):
        import cv2
        g = cv2.imread(resolve(ctx['spec'], L['mask']), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (W // 4, H // 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    else:
        g = np.ones((H // 4, W // 4), np.float32)
    ys, xs = np.nonzero(g > 0.5)
    life = float(L.get('life', 0.45))
    n = rng.poisson(float(L.get('rate', 20)) * (120 + life))
    ts = np.sort(rng.uniform(-life, 120, n))
    idx = rng.integers(0, max(1, len(ys)), n)
    pts = np.stack([ts, (xs[idx] + rng.uniform(0, 1, n)) * 4, (ys[idx] + rng.uniform(0, 1, n)) * 4,
                    rng.uniform(0.6, 1.0, n)], 1) if len(ys) else np.zeros((0, 4))
    _DROPS[key] = pts
    return pts


def draw_ripples(canvas, L, u, ctx):
    """Raindrops landing on a wet surface: a bright pin-prick splash, then a thin expanding ring (a light crest and a
    dark trough) that fades. `mask` = where drops land (white); `rate` drops/s; `life` s; `radius` max ring radius
    (frame widths); `tilt` ring height/width (a surface seen at an angle); `perspective` [far, near] radius scale from
    the top of the frame to the bottom; `rings: false` = splashes only (beads hitting metal). Rings pick up the local
    brightness (they are reflections), so they read on lit wet wood and vanish in shadow."""
    H, W = ctx['H'], ctx['W']
    op = float(kf(L.get('opacity', 1.0), u))
    if op <= 0.001:
        return canvas
    life = float(L.get('life', 0.45))
    R = float(L.get('radius', 0.03)) * W
    tilt = float(L.get('tilt', 0.4))
    far, near = L.get('perspective', [0.5, 1.2])
    wpx = max(1.0, float(L.get('width', 0.0022)) * W)
    rings = L.get('rings', True)
    col = gpu.dev_hex(L.get('color', '#ffffff'))
    out = canvas
    pts = _drop_list(L, ctx)
    act = pts[(pts[:, 0] <= u) & (pts[:, 0] > u - life)]
    for t0, x, y, k in act:
        a = (u - t0) / life
        sc = (far + (near - far) * y / H) * k
        r = R * sc * a ** 0.6
        amp = (1 - a) ** 1.5 * op * k
        ext = int(r + 4 * wpx + 6)
        x0, x1 = max(0, int(x) - ext), min(W, int(x) + ext + 1)
        y0, y1 = max(0, int(y - ext * tilt) - 2), min(H, int(y + ext * tilt) + 3)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = xp.mgrid[y0:y1, x0:x1].astype(xp.float32)
        d = xp.sqrt((xx - x) ** 2 + ((yy - y) / tilt) ** 2)
        patch = out[y0:y1, x0:x1]
        lum = patch.mean(-1, keepdims=True)
        if rings and a > 0.04:
            crest = xp.exp(-((d - r) / wpx) ** 2)[..., None]
            trough = xp.exp(-((d - r + 1.8 * wpx) / (1.2 * wpx)) ** 2)[..., None]
            hi = crest * amp * (0.02 + 1.6 * lum)
            new = patch * (1 - 0.45 * amp * trough) + col * hi
        else:
            new = patch
        if a < 0.12:   # the impact: a tiny bright splash
            sp = xp.exp(-(d / (1.6 * wpx * sc + 0.5)) ** 2)[..., None] * (1 - a / 0.12) * op * k
            new = new + col * sp * (0.08 + 0.9 * lum)
        if out is canvas:
            out = canvas.copy()
        out[y0:y1, x0:x1] = new
    return out


def draw_adjust(canvas, L, u, ctx):
    """Local correction inside `mask`/`radial` (white = affected): `despeckle` k clamps isolated bright specks to
    k x their blurred surroundings (upscaler 'sugar' on wet surfaces), `smooth` mixes in that blur (0..1),
    `saturation` (1 = unchanged), `gain` [r, g, b] colour balance, `exposure` (x)."""
    op = float(kf(L.get('opacity', 1.0), u))
    if op <= 0.001:
        return canvas
    out = canvas
    sig = float(L.get('sigma', 2.5)) * ctx['H'] / 1920
    if L.get('despeckle') or L.get('smooth'):
        b = gpu.blur_sep(canvas, sig)
        if L.get('despeckle'):
            out = xp.minimum(out, b * float(L['despeckle']))
        if L.get('smooth'):
            k = float(L['smooth'])
            out = out * (1 - k) + b * k
    if 'saturation' in L:
        lum = out.mean(-1, keepdims=True)
        out = lum + (out - lum) * float(L['saturation'])
    if L.get('gain'):
        out = out * xp.asarray(np.array(L['gain'], np.float32))
    if 'exposure' in L:
        out = out * float(L['exposure'])
    m = _layer_mask(L, u, ctx)
    m = op if m is None else m * op
    return canvas * (1 - m) + out * m


DRAW = dict(ripples=draw_ripples, adjust=draw_adjust, glitch=draw_glitch, solid=draw_solid, gradient=draw_gradient, image=draw_image, video=draw_video, sprite=draw_sprite,
            text=draw_text, rain=draw_rain)
