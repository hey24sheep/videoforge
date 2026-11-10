"""GPU image ops (CuPy; NumPy fallback with identical maths).

Colour pipeline per frame:
  scene   linear-light RGB, layers composited with PREMULTIPLIED alpha (no dark fringes at edges)
  grade   one kernel: exposure, bloom add, highlight knee (or ACES), split-tone, saturation, vignette -> display
          (sRGB-encoded) float in [0,1]. Display-referred sources keep their values: white stays white.
  overlay captions / titles drawn on the graded frame (crisp, not graded, not shaken)
  encode  one kernel: RGB -> NV12 (BT.709 limited range) + hashed film grain + +-0.5 LSB dither (no banding)

`sample` is the one placement primitive (cover/contain fit, Ken Burns, sprites, text): bilinear, transparent
outside the source.
"""
import math, os
import numpy as np

try:
    if os.environ.get('VF_GPU', '1') == '0':
        raise ImportError
    import cupy as xp
    xp.zeros(1).sum()
    GPU = True
except Exception:
    xp = np
    GPU = False


def to_dev(a):
    return xp.asarray(a, dtype=xp.float32)


def to_host(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)


def hexrgb(h, linear=True):
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)
    return c ** 2.2 if linear else c


_HEX = {}


def dev_hex(h, linear=True):
    """Device colour, cached (a fresh host->device copy per call cost 30 us and a sync)."""
    k = (h, linear)
    if k not in _HEX:
        _HEX[k] = to_dev(hexrgb(h, linear))
    return _HEX[k]


def upload_rgba_u8(a_u8, linear=True):
    """uint8 (H,W,3|4) -> device float32 premultiplied RGBA (linear or display), conversion done on the GPU."""
    a = xp.asarray(a_u8).astype(xp.float32) / 255
    if a.shape[2] == 3:
        a = xp.concatenate([a, xp.ones(a.shape[:2] + (1,), xp.float32)], -1)
    if linear:
        a[..., :3] = a[..., :3] ** 2.2
    a[..., :3] *= a[..., 3:4]
    return a


_grid = {}


def _coords(H, W):
    k = (H, W)
    if k not in _grid:
        yy, xx = xp.mgrid[0:H, 0:W].astype(xp.float32)
        _grid[k] = (xx + 0.5, yy + 0.5)
    return _grid[k]


_K = {}


def _kernel(name, src):
    if name not in _K:
        _K[name] = xp.RawKernel(src, name)
    return _K[name]


def _grid1(n):
    return ((n + 255) // 256,), (256,)


_SAMPLE_SRC = r'''
extern "C" __global__ void sample_k(const float* __restrict__ src, float* __restrict__ out, int hs, int ws, int C,
    int H, int W, float cx, float cy, float inv_s, float cs, float sn, float ox, float oy) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= H * W) return;
  int X = idx % W, Y = idx / W;
  float dx = (X + 0.5f - ox) * inv_s, dy = (Y + 0.5f - oy) * inv_s;
  float sx = cx + dx * cs - dy * sn - 0.5f, sy = cy + dx * sn + dy * cs - 0.5f;
  float* o = out + (long long)idx * C;
  if (sx <= -1.f || sx >= ws || sy <= -1.f || sy >= hs) { for (int c = 0; c < C; c++) o[c] = 0.f; return; }
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  float fx = sx - x0, fy = sy - y0;
  /* outside texels count as transparent (premultiplied zeros): edges fade instead of smearing */
  bool vx0 = x0 >= 0, vx1 = x0 + 1 < ws, vy0 = y0 >= 0, vy1 = y0 + 1 < hs;
  int xa = min(max(x0, 0), ws - 1), xb = min(max(x0 + 1, 0), ws - 1);
  int ya = min(max(y0, 0), hs - 1), yb = min(max(y0 + 1, 0), hs - 1);
  float w00 = (1.f - fx) * (1.f - fy) * (vx0 && vy0), w01 = fx * (1.f - fy) * (vx1 && vy0);
  float w10 = (1.f - fx) * fy * (vx0 && vy1), w11 = fx * fy * (vx1 && vy1);
  const float* p00 = src + ((long long)ya * ws + xa) * C; const float* p01 = src + ((long long)ya * ws + xb) * C;
  const float* p10 = src + ((long long)yb * ws + xa) * C; const float* p11 = src + ((long long)yb * ws + xb) * C;
  for (int c = 0; c < C; c++) o[c] = p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11;
}'''


def sample(src, H, W, cx, cy, scale, rot=0.0, ox=None, oy=None):
    """(H,W[,C]) view of `src`: source point (cx,cy) appears at output (ox,oy) (default centre), magnified by
    `scale` (output px per source px), rotated `rot` rad. Use premultiplied RGBA for anything with alpha."""
    hs, ws = src.shape[:2]
    ox = W / 2 if ox is None else ox
    oy = H / 2 if oy is None else oy
    if GPU:
        s = xp.ascontiguousarray(src, dtype=xp.float32)
        C = 1 if s.ndim == 2 else s.shape[2]
        out = xp.empty((H, W) + (() if s.ndim == 2 else (C,)), xp.float32)
        _kernel('sample_k', _SAMPLE_SRC)(*_grid1(H * W), (
            s, out, np.int32(hs), np.int32(ws), np.int32(C), np.int32(H), np.int32(W), np.float32(cx), np.float32(cy),
            np.float32(1.0 / scale), np.float32(math.cos(-rot)), np.float32(math.sin(-rot)), np.float32(ox),
            np.float32(oy)))
        return out
    X, Y = _coords(H, W)
    dx, dy = (X - ox) / scale, (Y - oy) / scale
    c, s_ = math.cos(-rot), math.sin(-rot)
    sx, sy = cx + dx * c - dy * s_ - 0.5, cy + dx * s_ + dy * c - 0.5
    x0, y0 = np.floor(sx).astype(np.int32), np.floor(sy).astype(np.int32)
    fx, fy = sx - x0, sy - y0
    s2 = src if src.ndim == 3 else src[..., None]
    acc = 0
    for ddx, ddy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + ddx, y0 + ddy
        ok = (xi >= 0) & (xi < ws) & (yi >= 0) & (yi < hs)
        acc = acc + s2[np.clip(yi, 0, hs - 1), np.clip(xi, 0, ws - 1)] * (w * ok)[..., None]
    return acc if src.ndim == 3 else acc[..., 0]


def fit_scale(hs, ws, H, W, mode='cover'):
    return max(W / ws, H / hs) if mode == 'cover' else min(W / ws, H / hs)


def over(dst, rgba, opacity=1.0, mode='normal'):
    """Composite PREMULTIPLIED RGBA onto RGB dst."""
    a = rgba[..., 3:4] * opacity
    c = rgba[..., :3] * opacity
    if mode == 'add':
        return dst + c
    if mode == 'screen':
        return dst + c - dst * c
    if mode == 'multiply':
        return dst * (1 - a) + dst * c
    return dst * (1 - a) + c


def blur_sep(a, sigma):
    """Separable gaussian (keep sigma small: use it at reduced resolution)."""
    if GPU:
        from cupyx.scipy.ndimage import convolve1d
    else:
        from scipy.ndimage import convolve1d
    r = max(1, int(sigma * 3))
    g = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    g = xp.asarray(g / g.sum())
    return convolve1d(convolve1d(a, g, axis=0, mode='reflect'), g, axis=1, mode='reflect')


def _down(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(a.shape[0] // k, k, a.shape[1] // k, k, *a.shape[2:]).mean((1, 3))


def bloom(img, look):
    """Glow pyramid: tight at 1/4 res, wide at 1/16 res (large-sigma gaussians at full res cost ~70 ms)."""
    H, W = img.shape[:2]
    br = xp.maximum(_down(img, 4) - look.get('bloom_threshold', 0.9), 0)
    s = H / 1080
    near = blur_sep(br, 1.5 * s)
    tiny = _down(br, 4)
    far = blur_sep(tiny, 1.75 * s)
    far = sample(far, br.shape[0], br.shape[1], far.shape[1] / 2, far.shape[0] / 2, 4)
    b = near * 0.5 + far * 0.8
    return sample(b, H, W, b.shape[1] / 2, b.shape[0] / 2, 4)


_GRADE_SRC = r'''
extern "C" __global__ void grade_k(const float* __restrict__ img, const float* __restrict__ bl,
    const float* __restrict__ vig, float* __restrict__ out, int N, float exposure, float bloomk, int tmode,
    float knee, float shr, float shg, float shb, float hir, float hig, float hib, float split, float sat,
    int hb, int hv, float har, float hag, float hab, float blk, float con, float lift) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= N) return;
  float c[3], ha[3] = {har, hag, hab};   /* halation: bloom tint per channel (film: red-orange glow) */
  for (int k = 0; k < 3; k++) {
    float x = img[i * 3 + k];
    x = isfinite(x) ? fmaxf(x, 0.f) : 0.f;
    x = x * exposure + (hb ? bl[i * 3 + k] * bloomk * ha[k] : 0.f);
    if (tmode == 1) x = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);       /* ACES */
    else if (tmode == 2 && x > knee) x = knee + (1.f - knee) * (1.f - expf(-(x - knee) / (1.f - knee)));  /* knee */
    c[k] = fminf(fmaxf(x, 0.f), 1.f);
  }
  float lum = 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
  float sh[3] = {shr, shg, shb}, hi[3] = {hir, hig, hib};
  float v = hv ? vig[i] : 1.f;
  for (int k = 0; k < 3; k++) {
    float x = c[k] + sh[k] * (1.f - lum) * (1.f - lum) * split;
    x = x * (1.f + (hi[k] - 1.f) * lum * split * 1.5f);
    x = lum + (x - lum) * sat;
    x = fminf(fmaxf(x * v, 0.f), 1.f);
    float y = powf(x, 1.f / 2.2f);   /* display encoding */
    y = fmaxf(y - blk, 0.f) / (1.f - blk);            /* black point: milky blacks -> true black */
    y = y + con * (y * y * (3.f - 2.f * y) - y);      /* S-curve contrast (0 = off, 1 = smoothstep) */
    y = lift + y * (1.f - lift);                       /* black floor: v9-like dark grey, not dead 0 */
    out[i * 3 + k] = fminf(fmaxf(y, 0.f), 1.f);
  }
}'''

_post = {}


def grade(img, look):
    """Linear scene -> display-encoded float [0,1] (device)."""
    H, W = img.shape[:2]
    bl = float(look.get('bloom', 0.4))
    b = bloom(img, look) if bl > 0 else None
    v = float(look.get('vignette', 0.3))
    vk = ('vig', H, W, v)
    if v > 0 and vk not in _post:
        X, Y = _coords(H, W)
        r = xp.sqrt(((X - W / 2) / (W / 2)) ** 2 + ((Y - H / 2) / (H / 2)) ** 2 * 0.8)
        _post[vk] = xp.ascontiguousarray(1 - v * xp.clip(r - 0.35, 0, None) ** 1.6).astype(xp.float32)
    tmode = {'none': 0, 'aces': 1, 'knee': 2}[look.get('tonemap', 'knee')]
    sh = tuple(hexrgb(look['shadows'])) if look.get('shadows') else (0.0, 0.0, 0.0)
    hi = tuple(hexrgb(look['highlights'])) if look.get('highlights') else (1.0, 1.0, 1.0)
    if not GPU:
        return _grade_np(img, b, _post.get(vk) if v > 0 else None, look, tmode, sh, hi, bl)
    out = xp.empty((H, W, 3), xp.float32)
    dummy = _post.setdefault('dummy', xp.zeros(1, xp.float32))
    _kernel('grade_k', _GRADE_SRC)(*_grid1(H * W), (
        xp.ascontiguousarray(img, dtype=xp.float32), b if b is not None else dummy, _post[vk] if v > 0 else dummy,
        out, np.int32(H * W), np.float32(look.get('exposure', 1.0)), np.float32(bl), np.int32(tmode),
        np.float32(look.get('knee', 0.8)), *[np.float32(x) for x in sh], *[np.float32(x) for x in hi],
        np.float32(look.get('split', 0.12)), np.float32(look.get('saturation', 1.0)), np.int32(b is not None),
        np.int32(v > 0), *[np.float32(x) for x in _halation(look)], np.float32(look.get('black', 0.0)),
        np.float32(look.get('contrast', 0.0)), np.float32(look.get('lift', 0.0))))
    return _selective(out, look)


_HUES = dict(red=0.0, orange=30.0, yellow=55.0, green=120.0, cyan=185.0, blue=230.0, magenta=300.0, pink=335.0)


def _selective(y, look):
    """Optional colourist pass on the display-encoded frame (only when the look asks for it, so other shots are
    untouched): look['temp'] -1..1 (negative = cooler white balance) and look['hue_sat'] = {band: factor}, a
    saturation factor per hue band (red, orange, yellow, green, cyan, blue, magenta, pink; smooth +-30 deg falloff).
    A key may also be a hue in degrees (e.g. '145': 0.45) for colours that sit between two named bands."""
    t, hs = float(look.get('temp', 0.0)), look.get('hue_sat')
    if not t and not hs:
        return y
    if t:
        y = y * xp.asarray([1 + 0.1 * t, 1.0, 1 - 0.1 * t], xp.float32)
    if hs:
        r, g, b = y[..., 0], y[..., 1], y[..., 2]
        mx, mn = xp.maximum(xp.maximum(r, g), b), xp.minimum(xp.minimum(r, g), b)
        d = mx - mn + 1e-6
        h = xp.where(mx == r, ((g - b) / d) % 6, xp.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60.0
        num = xp.zeros_like(r); den = xp.zeros_like(r)
        for band, k in hs.items():
            dh = xp.abs((h - (_HUES[band] if band in _HUES else float(band)) + 180.0) % 360.0 - 180.0)
            w = xp.clip(1 - dh / 30.0, 0, 1) ** 2
            num += w * float(k); den += w
        f = xp.where(den > 1, num / xp.maximum(den, 1e-6), num + (1 - den))  # bands overlap: blend; else ease to 1
        lum = (y @ xp.asarray([0.2126, 0.7152, 0.0722], xp.float32))[..., None]
        y = lum + (y - lum) * f[..., None]
    return xp.clip(y, 0, 1).astype(xp.float32)


def _halation(look):
    """look['halation']: 0..1 strength of the red-orange tint on the glow around bright lights (film halation)."""
    h = float(look.get('halation', 0.0))
    return (1.0 + 0.25 * h, 1.0 - 0.35 * h, 1.0 - 0.55 * h)


def _grade_np(img, b, vig, look, tmode, sh, hi, bl):
    x = np.nan_to_num(np.maximum(img, 0)) * look.get('exposure', 1.0)
    if b is not None:
        x = x + b * bl * np.array(_halation(look), np.float32)
    if tmode == 1:
        x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    elif tmode == 2:
        kn = look.get('knee', 0.8)
        x = np.where(x > kn, kn + (1 - kn) * (1 - np.exp(-(x - kn) / (1 - kn))), x)
    x = np.clip(x, 0, 1)
    lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    sp = look.get('split', 0.12)
    x = x + np.array(sh, np.float32) * (1 - lum) ** 2 * sp
    x = x * (1 + (np.array(hi, np.float32) - 1) * lum * sp * 1.5)
    x = lum + (x - lum) * look.get('saturation', 1.0)
    if vig is not None:
        x = x * vig[..., None]
    y = np.clip(x, 0, 1) ** (1 / 2.2)
    blk, con = float(look.get('black', 0.0)), float(look.get('contrast', 0.0))
    y = np.maximum(y - blk, 0) / (1 - blk)
    y = np.clip(y + con * (y * y * (3 - 2 * y) - y), 0, 1)
    lift = float(look.get('lift', 0.0))
    return _selective((lift + y * (1 - lift)).astype(np.float32), look)


_NV12_SRC = r'''
__device__ float hash01(unsigned int x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return (x & 0xffffff) / 16777216.0f;
}
extern "C" __global__ void nv12_k(const float* __restrict__ rgb, unsigned char* __restrict__ out, int H, int W,
    unsigned int frame, float grain, int gluma) {
  /* one thread per 2x2 block: 4 luma samples + 1 chroma pair (BT.709, limited range) */
  int bx = blockDim.x * blockIdx.x + threadIdx.x;
  int nbx = W / 2, nby = H / 2;
  if (bx >= nbx * nby) return;
  int X = (bx % nbx) * 2, Y = (bx / nbx) * 2;
  float cb = 0.f, cr = 0.f;
  for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
    int x = X + i, y = Y + j;
    const float* p = rgb + ((long long)y * W + x) * 3;
    unsigned int h = (unsigned int)(y * W + x) * 9781u + frame * 6271u;
    /* grain in 2x2 clumps (film-like, and ~3x more compressible than per-pixel noise) */
    unsigned int hg = (unsigned int)((y >> 1) * W + (x >> 1)) * 2654435761u + frame * 40503u;
    float g = grain > 0.f ? (hash01(hg) + hash01(hg ^ 0x9e3779b9u) - 1.f) * grain * 1.7f : 0.f;
    if (gluma) {  /* film: grain strongest in the mids, faint in deep blacks and clipped whites */
      float l = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
      g *= 0.25f + 3.0f * l * (1.f - l);
    }
    float r = fminf(fmaxf(p[0] + g, 0.f), 1.f), gg = fminf(fmaxf(p[1] + g, 0.f), 1.f), b = fminf(fmaxf(p[2] + g, 0.f), 1.f);
    float Yv = 0.2126f * r + 0.7152f * gg + 0.0722f * b;
    float d = hash01(h ^ 0x85ebca6bu) - 0.5f;   /* +-0.5 LSB dither: no gradient banding */
    out[(long long)y * W + x] = (unsigned char)fminf(fmaxf(16.f + 219.f * Yv + d + 0.5f, 0.f), 255.f);
    cb += (b - Yv) / 1.8556f; cr += (r - Yv) / 1.5748f;
  }
  float d2 = hash01((unsigned int)bx * 7919u + frame * 104729u) - 0.5f;
  unsigned char* uv = out + (long long)H * W + (long long)(Y / 2) * W + X;
  uv[0] = (unsigned char)fminf(fmaxf(128.f + 224.f * cb * 0.25f + d2 + 0.5f, 0.f), 255.f);
  uv[1] = (unsigned char)fminf(fmaxf(128.f + 224.f * cr * 0.25f - d2 + 0.5f, 0.f), 255.f);
}'''


def to_nv12(rgb_disp, frame, grain, luma=False):
    """Display RGB float (H,W,3) -> NV12 uint8 flat (H*W*3/2) on the device."""
    H, W = rgb_disp.shape[:2]
    if not GPU:
        return _nv12_np(rgb_disp, frame, grain, luma)
    out = xp.empty(H * W * 3 // 2, xp.uint8)
    _kernel('nv12_k', _NV12_SRC)(*_grid1((H // 2) * (W // 2)), (
        xp.ascontiguousarray(rgb_disp, dtype=xp.float32), out, np.int32(H), np.int32(W), np.uint32(frame),
        np.float32(grain), np.int32(bool(luma))))
    return out


def _nv12_np(x, frame, grain, luma=False):
    H, W = x.shape[:2]
    rng = np.random.default_rng(frame)
    if grain > 0:
        n = rng.standard_normal((H, W, 1)).astype(np.float32) * grain
        if luma:
            l = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
            n = n * (0.25 + 3.0 * l * (1 - l))
        x = np.clip(x + n, 0, 1)
    Y = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    cb = (x[..., 2] - Y) / 1.8556
    cr = (x[..., 0] - Y) / 1.5748
    d = rng.random((H, W), np.float32) - 0.5
    y8 = np.clip(16 + 219 * Y + d + 0.5, 0, 255).astype(np.uint8)
    cb2 = cb.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    cr2 = cr.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    uv = np.stack([128 + 224 * cb2, 128 + 224 * cr2], -1)
    uv = np.clip(uv + 0.5, 0, 255).astype(np.uint8).reshape(H // 2, W)
    return np.concatenate([y8.ravel(), uv.ravel()])
