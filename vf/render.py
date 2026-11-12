"""Shot renderer. Per frame: scene layers (linear, premultiplied) -> camera -> transition -> grade (display)
-> overlays (captions/titles, crisp & ungraded) -> NV12 (BT.709, grain, dither) -> async copy -> NVENC.
One cached segment per (shot, format); QC results are stored next to each segment."""
import functools, hashlib, inspect, json, math, os, queue, shutil, subprocess, threading
import numpy as np
from . import camera, gpu, layers
from . import enc as encoder
from .gpu import xp
from .memo import memo_file
from .spec import FORMATS, for_format, kf, resolve


@functools.lru_cache(None)
def ffmpeg():
    p = shutil.which('ffmpeg')
    if p:
        return p
    base = os.path.expandvars(r'%LOCALAPPDATA%\Microsoft\WinGet\Packages')
    for root, _, files in os.walk(base):
        if 'ffmpeg.exe' in files:
            return os.path.join(root, 'ffmpeg.exe')
    raise RuntimeError('ffmpeg not found')


@functools.lru_cache(None)
def renderer_code():
    """Hash of every renderer module's source: changing how pixels are made re-renders all segments."""
    from . import gpu as g, layers as l, render as r, captions as c, camera as cm
    h = hashlib.blake2b(digest_size=12)
    for m in (g, l, r, c, cm, encoder):
        h.update(inspect.getsource(m).encode())
    return h.hexdigest() if encoder.nvenc() else h.hexdigest() + '.cpu'   # CPU-encoded files differ


def is_overlay(L):
    """Text is drawn on the graded frame (crisp, white stays white, unaffected by camera) unless in_scene: true."""
    return L.get('type') == 'text' and not L.get('in_scene', False)


def shot_deps(spec, i):
    """Asset files shot i reads (and the previous shot's, when it transitions in)."""
    ids = [i] + ([i - 1] if i > 0 and (spec['shots'][i].get('transition') or {}).get('dur', 0) > 0 else [])
    out = set()
    for j in ids:
        for L in spec['shots'][j].get('layers', []):
            for v in [L] + [L.get(f) or {} for f in FORMATS]:
                if v.get('src'):
                    p = resolve(spec, v['src'])
                    if os.path.exists(p):
                        out.add(os.path.abspath(p))
    return sorted(out)


WATERMARK = 'hey24sheep.com/videoforge'


def watermark(spec):
    """The small corner mark drawn on every render, or None when it is off.
    Off: `vf build --no-watermark`, VF_WATERMARK=0, or `watermark: false` in series.yaml / episode.yaml (MCP: watermark=False).
    `watermark: {text, size, opacity, margin}` changes it. Please keep it on: it is as small as it can be and helps the
    project grow (README: Watermark)."""
    w = spec.get('watermark', True)
    if os.environ.get('VF_WATERMARK', '1') == '0' or w is False or w is None:
        return None
    w = dict(text=w) if isinstance(w, str) else (dict(w) if isinstance(w, dict) else {})
    m = float(w.get('margin', 0.02))
    return dict(type='text', text=str(w.get('text', WATERMARK)), font=w.get('font', 'body'), size=float(w.get('size', 0.017)),
                color='#ffffff', opacity=float(w.get('opacity', 0.6)), shadow=True, anim='none', **{'in': 0.0, 'out': 0.0},
                anchor=(1.0, 1.0), x=1 - m, y=1 - m, margin=m)


def shot_key(spec, i, fmt):
    """Everything that decides shot i's pixels in format fmt (the previous shot too when it transitions in)."""
    s = spec['shots'][i]
    caps = spec['_captions'].get(fmt, {})
    k = dict(shot=s, look=spec['look'], fonts=spec['fonts'], captions=caps.get(s['id']))
    wm = watermark(spec)
    if wm:
        k['watermark'] = wm
    if i > 0 and (s.get('transition') or {}).get('dur', 0) > 0:
        k['prev'] = spec['shots'][i - 1]
    return k


# ------------------------------------------------------------------ hash-keyed reuse of static layer stacks
def _animated(v):
    return isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2 and isinstance(v[0][0], (int, float))


def layer_state(L, u, ctx):
    """Hashable state of a scene layer at time u: equal states draw identical pixels. None = always redraw."""
    t = L['type']
    keyed = repr(tuple(sorted((k, u if _animated(v) else None) for k, v in L.items())))
    if t in ('video', 'sprite'):
        path = resolve(ctx['spec'], L['src'])
        if t == 'sprite' and not (os.path.isdir(path) or '*' in path):
            return ('img', keyed)
        return ('vid', keyed, layers._video_index(path, L, u))  # held drawings share an index
    if t in ('text', 'rain'):  # in-scene text (animated in/out) and particles change every frame: redraw
        return None
    return ('st', keyed)


class PrefixCache:
    """signature of the bottom k layers -> composed canvas (static plates / held drawings drawn once)."""
    def __init__(self, n=6):
        from collections import OrderedDict
        self.d, self.n = OrderedDict(), n

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]

    def put(self, k, v):
        self.d[k] = v
        while len(self.d) > self.n:
            self.d.popitem(last=False)


def _stack(shot, fmt, overlay):
    st = [for_format(L, fmt) for L in shot.get('layers', [])]
    return [L for L in st if L.get('enabled', True) is not False and is_overlay(L) == overlay]


def compose(spec, shot, u, fmt, ctx):
    """Scene of one shot at time u (device float32 RGB, linear light), camera applied."""
    H, W = ctx['H'], ctx['W']
    stack = _stack(shot, fmt, overlay=False)
    pc = ctx.setdefault('_prefix', {}).setdefault((shot['id'], fmt), PrefixCache())
    sig, keys = shot['id'], []
    for L in stack:
        s = layer_state(L, u, ctx)
        if s is None:
            break
        sig = hash((sig, s))
        keys.append(sig)
    canvas, start = None, 0
    for i in range(len(keys) - 1, -1, -1):
        c = pc.get(keys[i])
        if c is not None:
            canvas, start = c, i + 1
            break
    if canvas is None:
        canvas = xp.zeros((H, W, 3), xp.float32)
    ctx['_shot_id'] = shot['id']
    for i in range(start, len(stack)):
        canvas = layers.DRAW[stack[i]['type']](canvas, stack[i], u, ctx)
        if i < len(keys):
            pc.put(keys[i], canvas)  # draw functions return new arrays: cached canvases are never mutated
    cam = camera.settings(spec, shot)
    if cam:  # virtual camera: eased move + handheld + motion blur + depth parallax (vf/camera.py)
        cache = ctx.setdefault('_cam', {}).setdefault((shot['id'], fmt), {})
        depth = ctx.get('_depth', {}).get((shot['id'], fmt))
        canvas = camera.apply(canvas, cam, u, shot['dur'], shot['id'], ctx.get('fps', spec.get('fps', 60)),
                              depth=depth, cache=cache)
    for t0, k in (shot.get('flash') or []):  # impact flashes
        if u >= t0:
            canvas = canvas + float(k) * math.exp(-(u - t0) * 14)
    return canvas


def frame(spec, i, u, fmt, ctx):
    """Scene of shot i at local time u including its incoming transition (linear)."""
    shot = spec['shots'][i]
    tr = shot.get('transition') or {}
    d = float(tr.get('dur', 0.0))
    if not (i > 0 and d > 0 and u < d):
        return compose(spec, shot, u, fmt, ctx)
    prev = spec['shots'][i - 1]
    k, typ = u / d, tr.get('type', 'fade')
    if typ == 'fade':
        return compose(spec, prev, prev['dur'] + u, fmt, ctx) * (1 - k) + compose(spec, shot, u, fmt, ctx) * k
    if typ in ('dip', 'dip_white'):
        a = 1 - abs(2 * k - 1)
        img = compose(spec, prev, prev['dur'] + u, fmt, ctx) if k < 0.5 else compose(spec, shot, u, fmt, ctx)
        return img * (1 - a) + (1.0 if typ == 'dip_white' else 0.0) * a
    if typ == 'flash':
        return compose(spec, shot, u, fmt, ctx) + 3.0 * (1 - k) ** 2
    return compose(spec, shot, u, fmt, ctx)


def overlay(img, spec, i, u, fmt, ctx):
    """Titles + captions on the graded (display-encoded) frame."""
    shot = spec['shots'][i]
    items = _stack(shot, fmt, overlay=True) + [for_format(L, fmt) for L in (ctx['captions'].get(shot['id']) or [])]
    wm = watermark(spec)
    if wm:
        H, W = ctx['H'], ctx['W']   # equal margins in pixels on both edges
        wm = dict(wm, x=1 - wm['margin'] * min(W, H) / W, y=1 - wm['margin'] * min(W, H) / H)
        items = items + [wm]
    if not items:
        return img
    octx = dict(ctx, space='display')
    for L in items:
        img = layers.draw_text(img, L, u, octx)
    return img


# ------------------------------------------------------------------ encoder
class Writer:
    """NV12 frames -> ffmpeg (NVENC, or libx264 without it: vf/enc.py). Device->host copies go through a ring of
    pinned buffers on their own stream; a thread feeds ffmpeg, so GPU rendering, the copy and encoding all overlap."""
    NBUF = 4

    def __init__(self, path, W, H, fps, cq=18, preset='p4'):
        self.size = W * H * 3 // 2
        enc = encoder.master(W, H, fps, cq, preset)
        self.p = subprocess.Popen([ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'nv12',
                                   '-s', f'{W}x{H}', '-r', str(fps), '-i', '-'] + enc + ['-an', path],
                                  stdin=subprocess.PIPE)
        self.free, self.q = queue.Queue(), queue.Queue()
        if gpu.GPU:
            import cupyx
            self.bufs = [cupyx.empty_pinned((self.size,), np.uint8) for _ in range(self.NBUF)]
            self.stream = xp.cuda.Stream(non_blocking=True)
        else:
            self.bufs = [np.empty(self.size, np.uint8) for _ in range(self.NBUF)]
        for i in range(self.NBUF):
            self.free.put(i)
        self.err = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                idx, ev, _keep = item
                if ev is not None:
                    ev.synchronize()
                self.p.stdin.write(memoryview(self.bufs[idx]))
                self.free.put(idx)
        except Exception as e:  # broken pipe etc.: surface it on close()
            self.err = e

    def write(self, nv12):
        while True:  # if the encoder died, the feeder thread stops returning buffers: fail instead of hanging forever
            try:
                idx = self.free.get(timeout=5)
                break
            except queue.Empty:
                if self.err is not None or not self.t.is_alive() or self.p.poll() is not None:
                    raise RuntimeError(f'ffmpeg encoder stopped ({self.err or self.p.poll()})')
        if gpu.GPU:
            ready = xp.cuda.Event()
            ready.record()  # rendering of this frame on the default stream
            self.stream.wait_event(ready)
            xp.cuda.runtime.memcpyAsync(self.bufs[idx].ctypes.data, nv12.data.ptr, self.size, 2, self.stream.ptr)
            ev = self.stream.record()
            self.q.put((idx, ev, nv12))  # keep the device array alive until the copy finished
        else:
            self.bufs[idx][:] = nv12
            self.q.put((idx, None, None))

    def close(self):
        self.q.put(None)
        self.t.join(timeout=600)
        try:
            self.p.stdin.close()
        except OSError:
            pass
        rc = self.p.wait(timeout=600)
        if self.err or rc != 0:
            raise RuntimeError(f'ffmpeg encode failed ({self.err or rc})')

    def kill(self):
        try:
            self.p.kill()
        except Exception:
            pass


@memo_file('project', '.mp4')
def render_segment(key, i, fmt, fps, scale, code, deps, out=None, _spec=None, _qc=None):
    """Encode shot i for format fmt into `out` (+ `out`.qc.json). Cache key: (key, i, fmt, fps, scale, code, deps)."""
    spec = _spec
    F = FORMATS[fmt]
    W, H = int(F['W'] * scale) // 2 * 2, int(F['H'] * scale) // 2 * 2
    ctx = dict(spec=spec, W=W, H=H, fmt=fmt, captions=spec['_captions'].get(fmt, {}), space='linear', fps=fps)
    shot = spec['shots'][i]
    n = max(1, int(round(shot['dur'] * fps)))
    look = dict(spec['look'], **(shot.get('look') or {}))
    grain = float(look.get('grain', 0.0))
    gf = int(round(shot['start'] * fps))
    wr = Writer(out, W, H, fps)
    means = []
    try:
        for k in range(n):
            u = k / fps
            disp = gpu.grade(frame(spec, i, u, fmt, ctx), look)
            disp = overlay(disp, spec, i, u, fmt, ctx)
            means.append(disp.mean())  # stays on the device: one sync per shot, not per frame
            wr.write(gpu.to_nv12(disp, gf + k, grain, look.get('grain_luma', False)))
        wr.close()
    except BaseException:
        wr.kill()
        raise
    lum = [float(x) * 255 for x in (xp.stack(means).get() if gpu.GPU else means)]
    items = _qc.check_shot(i, lum, fps) if _qc is not None else []
    json.dump(items, open(out + '.qc.json', 'w'))
