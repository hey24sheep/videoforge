"""Virtual cinema camera, applied to each composed shot (after the scene layers, before grade and overlays).

Why here and not in the AI model: when the video model moves the camera itself, piers, rails and buildings wobble.
So AI clips are generated with a steady camera and the move is made here: exact, eased, motion-blurred, and with
2.5D parallax when a depth map exists (near things move more than far things, like a real lens).

shot:
  camera:
    move: push_in        # push_in pull_out pan_left pan_right tilt_up tilt_down truck_left truck_right drift
                         # crash_zoom dolly_zoom orbit_left orbit_right rise settle none
    amount: 0.06         # size of the move (fraction of frame; zoom moves: +6 %)
    ease: io             # io (default, S-curve), out (decelerate), in, lin
    start: 0 / end: dur  # when the move runs (seconds in the shot); holds before/after keep drifting gently
    handheld: 0.2        # organic operator sway 0..1 (low-frequency, never jittery); 0 = tripod
    blur: 1.0            # motion-blur strength (180-degree shutter = 1.0); 0 = off
    parallax: 0.6        # depth-based parallax strength (needs a depth map on the clip layer); 0 = flat
    focus: [0.5, 0.5]    # point the camera pushes toward (frame fractions)
    # legacy / explicit keyframes still work and stack on top: zoom, roll, shake (impact jitter), focus
  camera: false          # locked off
The episode/series/style `camera:` is the default for every shot without one (spec['camera']).
"""
import math, zlib
import numpy as np
from . import gpu
from .gpu import xp
from .spec import kf, ease

MOVES = ('push_in', 'pull_out', 'pan_left', 'pan_right', 'tilt_up', 'tilt_down', 'truck_left', 'truck_right',
         'drift', 'crash_zoom', 'dolly_zoom', 'orbit_left', 'orbit_right', 'rise', 'settle', 'none')


def settings(spec, shot):
    """The shot's camera dict (shot overrides the episode default), or None when locked off."""
    c = shot.get('camera', None)
    if c is False:
        return None
    base = spec.get('camera') or {}
    if c is None:
        return dict(base) if base else None
    return dict(base, **c) if isinstance(c, dict) else None


def _noise(t, seed, octaves=((0.23, 1.0), (0.51, 0.55), (1.13, 0.25), (2.3, 0.08))):
    """Smooth organic 1-D noise in about [-1, 1]: incommensurate low frequencies with seeded phases."""
    r = np.random.default_rng(seed)
    v, norm = 0.0, 0.0
    for f, a in octaves:
        ph = r.uniform(0, 2 * math.pi)
        f2 = f * r.uniform(0.85, 1.15)
        v += a * math.sin(2 * math.pi * f2 * t + ph)
        norm += a
    return v / norm


def state(cam, u, dur, sid):
    """Camera at shot time u -> dict(zoom, tx, ty, roll, cx, cy, par). tx/ty are frame fractions of the image
    shift (+x = image moves right), zoom >= 1 includes overscan so no edge is ever revealed."""
    move = cam.get('move', 'none') or 'none'
    if move not in MOVES:
        raise ValueError(f'camera move {move!r}; one of {", ".join(MOVES)}')
    a = float(cam.get('amount', 0.06))
    t0 = float(cam.get('start', 0.0))
    t1 = float(cam.get('end', dur))
    k_lin = (u - t0) / max(1e-6, t1 - t0)
    e = cam.get('ease', 'out' if move in ('crash_zoom', 'settle') else 'io')
    k = ease(e, k_lin)
    z, tx, ty, roll, par_dir = 1.0, 0.0, 0.0, 0.0, 0.0
    if move == 'push_in':
        z = 1 + a * k
    elif move == 'pull_out':
        z = 1 + a * (1 - k)
    elif move == 'crash_zoom':
        k = ease('out', min(1.0, k_lin * 3.0))  # fast in the first third, then settles
        z = 1 + a * 2.5 * k
    elif move in ('pan_left', 'pan_right', 'truck_left', 'truck_right'):
        sgn = 1 if move.endswith('left') else -1  # camera pans left -> image slides right
        tx = sgn * a * (k - 0.5)
        par_dir = 1.0 if move.startswith('truck') else 0.25
    elif move in ('tilt_up', 'tilt_down'):
        ty = (1 if move == 'tilt_up' else -1) * a * (k - 0.5)
    elif move == 'rise':
        ty = a * (k - 0.5)
        z = 1 + 0.3 * a * k
        par_dir = 1.0
    elif move == 'drift':  # slow lateral float with a hint of push: never a dead still
        tx = a * 0.5 * (k - 0.5)
        z = 1 + 0.5 * a * k
        par_dir = 0.6
    elif move in ('orbit_left', 'orbit_right'):
        sgn = 1 if move == 'orbit_left' else -1
        tx = sgn * a * (k - 0.5)
        roll = sgn * 0.6 * (k - 0.5)
        z = 1 + 0.4 * a
        par_dir = 1.4
    elif move == 'dolly_zoom':
        z = 1 + a * k  # foreground stays, background swells (sign flipped in the parallax term below)
        par_dir = -1.0
    elif move == 'settle':  # lands from a small offset (end of a whip / a held breath)
        tx = a * 0.4 * (1 - k)
        z = 1 + 0.5 * a * (1 - k)
    # explicit keyframes stack on top
    z *= float(kf(cam.get('zoom', 1.0), u))
    roll += float(kf(cam.get('roll', 0.0), u))
    seed = zlib.crc32(str(sid).encode())
    hh = float(cam.get('handheld', 0.0))
    if hh:
        tx += hh * 0.006 * _noise(u, seed)
        ty += hh * 0.006 * _noise(u, seed + 1)
        roll += hh * 0.35 * _noise(u, seed + 2)
    sh = float(kf(cam.get('shake', 0.0), u))  # impact jitter (legacy): fast, decaying in the keyframes
    if sh:
        tx += sh * (math.sin(u * 37.1) * 0.6 + math.sin(u * 59.7) * 0.4)
        ty += sh * (math.sin(u * 43.3) * 0.6 + math.sin(u * 71.9) * 0.4)
    fx, fy = kf(cam.get('focus', [0.5, 0.5]), u)
    return dict(zoom=z, tx=tx, ty=ty, roll=roll, cx=float(fx), cy=float(fy), sh=sh, hh=hh,
                par=float(cam.get('parallax', 0.6)) * (par_dir if par_dir else 0.35), a=a, move=move)


def _overscan(cam, dur, sid, steps=24):
    """Smallest extra zoom that keeps every frame edge covered for the whole shot (incl. parallax and roll)."""
    need = 1.0
    for i in range(steps + 1):
        s = state(cam, dur * i / steps, dur, sid)
        m = 1 + 2 * (abs(s['tx']) + abs(s['ty'])) * (1 + abs(s['par'])) + abs(math.radians(s['roll'])) * 1.2 \
            + 2 * s['sh']
        need = max(need, m / max(1e-6, min(1.0, s['zoom'])))
    return need


_WARP_SRC = r'''
extern "C" __global__ void camwarp_k(const float* __restrict__ src, const float* __restrict__ depth, float* __restrict__ out,
    int H, int W, int has_depth, float cx, float cy, float zoom, float tx, float ty, float cs, float sn,
    float par, float d0, float wmul) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= H * W) return;
  int X = idx % W, Y = idx / W;
  float dw = 1.f;
  if (has_depth) dw = 1.f + par * (depth[idx] - d0);  /* near (1) moves more than far (0) */
  float z = 1.f + (zoom - 1.f) * dw;
  if (z < 0.2f) z = 0.2f;
  float qx = X + 0.5f - cx - tx * dw, qy = Y + 0.5f - cy - ty * dw;
  qx /= z; qy /= z;
  float sx = cx + qx * cs - qy * sn - 0.5f, sy = cy + qx * sn + qy * cs - 0.5f;
  sx = fminf(fmaxf(sx, 0.f), W - 1.001f); sy = fminf(fmaxf(sy, 0.f), H - 1.001f);
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  float fx = sx - x0, fy = sy - y0;
  const float* p00 = src + ((long long)y0 * W + x0) * 3; const float* p01 = p00 + 3;
  const float* p10 = p00 + (long long)W * 3; const float* p11 = p10 + 3;
  float* o = out + (long long)idx * 3;
  for (int c = 0; c < 3; c++)
    o[c] += wmul * ((p00[c] * (1.f - fx) + p01[c] * fx) * (1.f - fy) + (p10[c] * (1.f - fx) + p11[c] * fx) * fy);
}'''


def _warp_into(out, canvas, depth, s, over, H, W, wmul):
    zoom = s['zoom'] * over
    rot = math.radians(s['roll'])
    cx, cy = s['cx'] * W, s['cy'] * H
    if gpu.GPU:
        d = depth if depth is not None else canvas  # dummy pointer when unused
        gpu._kernel('camwarp_k', _WARP_SRC)(*gpu._grid1(H * W), (
            canvas, d, out, np.int32(H), np.int32(W), np.int32(depth is not None), np.float32(cx), np.float32(cy),
            np.float32(zoom), np.float32(s['tx'] * W), np.float32(s['ty'] * H), np.float32(math.cos(-rot)),
            np.float32(math.sin(-rot)), np.float32(s['par']), np.float32(0.5), np.float32(wmul)))
        return out
    # CPU fallback (tests only): flat warp, no parallax
    img = gpu.sample(canvas, H, W, cx - s['tx'] * W / zoom, cy - s['ty'] * H / zoom, zoom, rot=rot, ox=cx, oy=cy)
    out += wmul * img
    return out


def apply(canvas, cam, u, dur, sid, fps, depth=None, cache=None):
    """Camera-moved (and motion-blurred) canvas. `cache` (dict per shot) keeps the overscan computation."""
    H, W = canvas.shape[:2]
    cache = cache if cache is not None else {}
    if 'over' not in cache:
        cache['over'] = _overscan(cam, dur, sid)
    over = cache['over']
    s = state(cam, u, dur, sid)
    canvas = xp.ascontiguousarray(canvas, dtype=xp.float32)
    blur = float(cam.get('blur', 1.0))
    n = 1
    if blur > 0:  # samples from the image motion across a 180-degree shutter (half a frame interval)
        dt = 0.5 * blur / fps
        a, b = state(cam, u - dt / 2, dur, sid), state(cam, u + dt / 2, dur, sid)
        disp = (abs(a['tx'] - b['tx']) * W + abs(a['ty'] - b['ty']) * H
                + abs(a['zoom'] - b['zoom']) * over * 0.5 * math.hypot(W, H)
                + abs(math.radians(a['roll'] - b['roll'])) * 0.5 * math.hypot(W, H)) * (1 + abs(s['par']) * 0.5)
        n = int(min(12, max(1, math.ceil(disp / 1.25))))
    out = xp.zeros_like(canvas)
    if n == 1:
        return _warp_into(out, canvas, depth, s, over, H, W, 1.0)
    for i in range(n):
        ui = u + (i / (n - 1) - 0.5) * (0.5 * blur / fps)
        _warp_into(out, canvas, depth, state(cam, ui, dur, sid), over, H, W, 1.0 / n)
    return out
