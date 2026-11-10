"""`profile: k6` video path (render-side orchestration; heavy work runs in the gen venv as separate processes so
each model's RAM is returned before the next starts - running them together hung the GPU driver on 2026-11-02).

  generate:  Kandinsky 6.0 Lite distill (MIT), fp8 DiT resident + SageAttention, 6 steps, TAEHV decode
             384x672 (~1.3 min / 5 s clip) or `gen: {hires: true}` 480x864 (~4.3 min) for close-ups of faces
  finish:    [lip-sync] -> denoise (RTX VSR 1x DENOISE_MEDIUM) -> exposure match -> NVIDIA frame generation to
             fps/speed -> RTX VSR 4x ULTRA (1536x2688: camera push-ins then DOWN-sample, staying sharp)

Both stages are content-cached (global tier, cache/k6 and cache/k6fin): a shot regenerates only when its first frame,
prompt, seed, size, frames, steps or finishing options change. Measured numbers: memory/film-pipeline-performance.
shot gen options: hires, steps, lipsync: {engine: musetalk|latentsync, mask: lips|face}, expo: false | target L.
"""
import hashlib, json, os, shutil, subprocess, tempfile, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_PY = os.path.join(_AI, 'tools', 'envs', 'gen', 'Scripts', 'python.exe')
K6_VERSION = 'k6lite-distill|fp8|sage1|vae-tiled-cap6.5|prelift2|v3'  # 2026-11-02: full VAE decode (TAEHV caused blocky skin)
# 2026-11-10 A/B (docs/model_optimization_plan.md): the DiT in full bf16, block-offloaded, is FASTER than the fp8-stored
# resident one (13.3 vs 16.9 s/step: layerwise fp8 casting costs more than streaming) and it IS the model's own precision
# (fp8store vs bf16: SSIM 0.959-0.984). New clips use bf16; a clip already cached under the fp8 key is reused (approved
# episodes stay identical). VF_K6_PREC=fp8store brings the old path back.
K6_PREC = os.environ.get('VF_K6_PREC', 'bf16')
K6_VERSION_BF16 = 'k6lite-distill|bf16-offload|sage1|vae-tiled-cap6.5|prelift2|v3'
EXPO_MODE = os.environ.get('VF_EXPO', 'filmic')  # filmic | linear (v9)
SIZE = {False: (384, 672), True: (480, 864)}  # portrait (W, H); landscape swaps
# `gen: {engine: wan}`: Wan 2.2 TI2V 5B (fast profile, vf/gen/video.py) makes the raw clip instead; everything after
# (QC, start trim, lip-sync, finishing) is shared. Measured 2026-11-02: ~32 s/clip at 480x832, sharper faces.
WAN_SIZE = (480, 832)
WAN_VERSION = 'wan22-ti2v5b-fast|v1'
UP = int(os.environ.get('VF_UP', 2))  # 2x HIGH beat 4x ULTRA on faces (2026-11-02: pores vs paint)  # final RTX VSR scale
UPQ = os.environ.get('VF_UPQ', 'HIGH')  # RTX VSR quality for that pass
FIN_VERSION = f'fin|denoise-med|expo-{EXPO_MODE}|vfg-high|vsr{UP}-{UPQ.lower()}|v3'


_FIN_SRC = ('vf/gen/rtx_vsr.py', 'vf/gen/framegen.py')
_LS_SRC = {'musetalk': 'vf/gen/lipsync.py', 'latentsync': 'vf/gen/latentsync_run.py'}
# Source hash of the finishing code when this was introduced (2026-11-03). Finishes made before then were made by
# this code, so they keep their keys; any later edit of these files or of _expo/_grade changes the key (the audit
# found k6fin keyed on FIN_VERSION only: code edits silently served stale finishes). Note: finishes written before
# 2026-11-03 13:00 are lossless h264, later ones near-lossless cq13 (rtx_vsr final pass) under the same keys: visually
# identical after the compositor's own re-encode, so they were not invalidated.
# 2026-11-03 (Q5): the lip-sync scripts only key clips that use them (editing lipsync.py re-finished all 15 clips).
# _FIN_BASE = core hash (rtx_vsr + framegen + _expo/_grade helpers) at that split, so existing keys are unchanged.
_FIN_BASE = '7cbeb4189b6a7aeb'


def _src_hash(files, fns=()):
    import inspect
    h = hashlib.sha1()
    for f in files:
        try:
            h.update(open(os.path.join(ROOT, f), 'rb').read().replace(bytes([13, 10]), bytes([10])))  # CRLF-proof
        except OSError:
            pass
    for fn in fns:
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()[:16]


# 2026-11-10: lipsync.py / latentsync_run.py got the VF_AI_ROOT path change (same default paths, no behaviour change);
# their new file hashes map back to the old ones so finished lip-sync clips stay cached.
_LS_SAME = {'5ae0136235a4a682': '57fb48deb476cd83', '5ccb33d3659b6379': 'bc79a69324d53ee7'}


def _fin_code(engine=None):
    d = _src_hash(_FIN_SRC, (_expo, _grade, _mid_frame, _stream_lossless))
    core = () if d == _FIN_BASE else (d,)
    if engine not in _LS_SRC:
        return core
    ls = _src_hash([_LS_SRC[engine]])
    return core + (_LS_SAME.get(ls, ls),)


def _fhash(p):
    st = os.stat(p)
    return f'{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}'


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _gen(args, timeout=1800, **env):
    """One finishing step in the gen venv. Holds the GPU lock (2026-11-03 audit: finishing ran next to the judge and
    the next K6 batch -> spill) and a timeout (a hung step used to block the build forever)."""
    from .gpulock import gpu_lock
    from .proc import run
    e = dict(os.environ, **{k: str(v) for k, v in env.items()})
    with gpu_lock('vf finish ' + args[0]):
        run([GEN_PY, '-m'] + args, timeout=timeout, cwd=ROOT, env=e)


def clips(jobs, log=print, on_clip=None):
    """jobs: [{image, prompt, seed, frames, width, height, steps?}] -> [(path, fps)] (cached).
    on_clip(i, path) is called as soon as each clip exists (cached ones first), so finishing can overlap with the
    generation of the next clips (finishing needs <1 GB VRAM and ~2 GB RAM)."""
    d = os.path.join(ROOT, 'cache', 'k6')
    os.makedirs(d, exist_ok=True)
    todo, outs = [], []
    for j in jobs:
        if j.get('engine') == 'wan':
            k = _key(WAN_VERSION, _fhash(j['image']), j['prompt'], j['seed'], j['frames'], j['width'], j['height'])
        else:
            args = (_fhash(j['image']), j['prompt'], j['seed'], j['frames'], j['width'], j['height'], j.get('steps', 6))
            k = _key(K6_VERSION, *args)
            if K6_PREC == 'bf16' and not os.path.exists(os.path.join(d, k + '.mp4')):
                k = _key(K6_VERSION_BF16, *args)
        out = os.path.join(d, k + '.mp4')
        outs.append(out)
        from .memo import touch
        touch(out)  # vf/gc.py: raw takes of this build (re-roll decisions replay from them) are kept
        if not os.path.exists(out):
            todo.append(dict(j, id=k, out=out))
    told = set()

    def tell():
        for i, o in enumerate(outs):
            if i not in told and os.path.exists(o):
                told.add(i)
                if on_clip:
                    on_clip(i, o)
    wan = [j for j in todo if j.get('engine') == 'wan']
    if not todo:
        tell()
    todo = [j for j in todo if j.get('engine') != 'wan']
    if wan:
        log(f'  wan: generating {len(wan)} clip(s) (Wan 2.2 TI2V 5B, {WAN_SIZE[0]}x{WAN_SIZE[1]})')
        tdir = tempfile.mkdtemp(prefix='vf_wan_')
        jf = os.path.join(tdir, 'jobs.json')
        json.dump([dict(image=j['image'], prompt=j['prompt'], seed=j['seed'], frames=j['frames'], width=j['width'],
                        height=j['height'], profile='fast') for j in wan], open(jf, 'w'))
        from .gpulock import gpu_lock
        from .proc import _kill_tree
        with gpu_lock('vf wan'):
          p = subprocess.Popen([GEN_PY, '-m', 'vf.gen.video', jf], cwd=ROOT, stdout=subprocess.PIPE, text=True)
          import threading
          dog = threading.Timer(900 * len(wan) + 900, _kill_tree, [p])  # a hung child must not hold the GPU lock forever
          dog.daemon = True
          dog.start()
          try:
            for line in p.stdout:  # one JSON line per finished clip
                if line.startswith('{'):
                    r = json.loads(line)
                    if r.get('ok'):
                        shutil.copyfile(r['out'], wan[r['i']]['out'])
                        log(f"    wan clip {r['i'] + 1}/{len(wan)} {r.get('seconds', 0):.0f}s")
                        # no tell() here: Wan uses ~7-8 GB, so finishing a clip on the GPU at the same time
                        # overflows the 8 GB card and both crawl (2026-11-02: build stalled for an hour)
                    else:
                        log(f"    wan clip {r['i'] + 1} failed: {r.get('error')}")
            if p.wait(timeout=600):
                raise subprocess.CalledProcessError(p.returncode, 'vf.gen.video')
          except BaseException:
            _kill_tree(p)  # never leave a 7 GB Wan child running after the lock is released
            raise
          finally:
            dog.cancel()
            shutil.rmtree(tdir, ignore_errors=True)
        tell()
    if todo:
        log(f'  k6: generating {len(todo)} clip(s) ({sum(1 for j in todo if j["width"] > 400)} hi-res)')
        tdir = tempfile.mkdtemp(prefix='vf_k6_')
        jf = os.path.join(tdir, 'jobs.json')
        json.dump(todo, open(jf, 'w'))
        from .gpulock import gpu_lock
        from .proc import run
        # 2026-11-06 (an outro build): the keyframe stage left the gen worker parked in RAM (~8 GB), so the K6
        # encode pass for a new prompt was refused (14.2 GB free, needs 20). The worker reloads its models in ~6 s:
        # stop it outright before every K6 batch instead of parking it.
        try:
            from .gen import bridge as _br
            _br.stop_gen()
        except Exception as e:
            log(f'  k6: could not stop the gen worker ({type(e).__name__}: {e})')
        try:
            with gpu_lock('vf k6'):
                # no tell() while generating: the full-VAE decode reserves ~6.3 GB, so finishing a clip (VSR + frame
                # generation) on the GPU at the same time overflows the 8 GB card and both stall (2026-11-02, 30 min)
                run([GEN_PY, '-m', 'vf.gen.k6_batch', jf], timeout=900 * len(todo) + 2700, cwd=ROOT,
                    env=dict(os.environ, K6_PREC=K6_PREC))  # hi-res ~10 min/clip + an encode pass
        finally:
            shutil.rmtree(tdir, ignore_errors=True)
        tell()
    return [(o, 24) for o in outs]


def _expo(src, out, target_l=0.36, strength=0.6):
    """Shift the clip's median luminance toward target_l (Lab L/100), keeping contrast and colour: one dark shot
    no longer reads as 'murky' next to its neighbours. Measured on the clip's middle frame, applied uniformly."""
    import cv2
    mid, fps = _mid_frame(src)
    L = cv2.cvtColor(mid, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255
    cur = float(np.median(L))
    if EXPO_MODE == 'linear':  # v9: linear-light gain, clipped at white (amplifies the noise floor)
        gain = float(np.clip((target_l / max(cur, 0.02)) ** strength, 0.8, 1.8))
        lut = (np.clip((np.arange(256) / 255.0) ** 2.2 * gain, 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)
    else:  # filmic lift y = (1+k)x/(1+kx) after a small black point: mids up, highlights roll off, never clip,
        # and the noise floor under the black point stays black (2026-11-02 Phase 1)
        black = 0.012
        mb = max((cur - black) / (1 - black), 1e-3)
        gain = float(np.clip((target_l - mb) / (mb * (1 - target_l)), 0, 3.0))  # k; y(mb) = target_l
        x = np.clip((np.arange(256) / 255.0 - black) / (1 - black), 0, 1)
        lut = ((1 + gain) * x / (1 + gain * x) * 255).round().astype(np.uint8)
    _stream_lossless(src, out, mid.shape, fps, lambda f: cv2.LUT(f, lut))
    return cur, gain


def _mid_frame(src):
    """Middle frame + fps without holding the clip in RAM (291 frames at 960x1728 = 1.4 GB as uint8)."""
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = [None]
    for i in range(n // 2 + 1):  # decode forward (seeking lossless NVENC h264 is not frame-exact)
        ok, f = cap.read()
        if not ok:
            break
        fr[0] = f
    cap.release()
    return fr[0], fps


def _stream_lossless(src, out, shape, fps, fn):
    import cv2
    from .render import ffmpeg
    from .proc import writer
    H, W = shape[:2]
    cap = cv2.VideoCapture(src)
    cmd = [ffmpeg(), '-v', 'error', '-y', '-threads', '4', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
           '-r', f'{fps}', '-i', '-', '-c:v', 'h264_nvenc', '-tune', 'lossless', '-pix_fmt', 'yuv420p', out]
    try:
        with writer(cmd, out) as p:
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                p.stdin.write(np.ascontiguousarray(fn(f)).tobytes())
    finally:
        cap.release()


def _lab_stats(img_bgr):
    import cv2
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = lab.reshape(-1, 3)
    lum = m[:, 0]
    keep = (lum > 8) & (lum < 247)  # ignore crushed blacks and clipped practicals (bulbs, neon cores)
    m = m[keep] if keep.sum() > 1000 else m
    return m.mean(0), m.std(0) + 1e-3


def _grade(src, out, ref, strength=0.35):
    """Shot matching: move the clip's Lab mean/std (measured on its middle frame) toward the reference frame's,
    `strength` of the way (Reinhard transfer). One palette across the episode, each shot keeps its own key light."""
    import cv2
    mid, fps = _mid_frame(src)
    rm, rs = _lab_stats(cv2.imread(ref))
    sm, ss = _lab_stats(mid)
    # partial move of the colour cast; only half as far on the spread (matching a muted reference's spread fully
    # desaturated every shot and read as darker)
    tm, ts = sm + (rm - sm) * strength, ss * (rs / ss) ** (strength * 0.5)

    def fn(f):  # colour only (a, b): brightness is the exposure step's job (matching L darkened every shot)
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 1:] = (lab[..., 1:] - sm[1:]) * (ts[1:] / ss[1:]) + tm[1:]
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    _stream_lossless(src, out, mid.shape, fps, fn)


def finish(raw, fps_in, fps_out, portrait=True, lipsync=None, expo=0.36, grade=None, log=print):
    """raw clip -> [lipsync] -> deblur -> exposure -> frame generation (fps_out) -> 4x upscale [-> lipsync]. Cached.
    lipsync `late: true` (MuseTalk only) runs it LAST, on the upscaled fps_out clip: frame generation otherwise blends
    two mouth shapes into a doubled "ghost" mouth, and the 4x upscale magnifies a mouth drawn at 480p (2026-11-03)."""
    d = os.path.join(ROOT, 'cache', 'k6fin')
    os.makedirs(d, exist_ok=True)
    ls = None
    if lipsync:
        ls = dict(lipsync, wav=_fhash(lipsync['wav']))
    gr = (_fhash(grade['ref']), float(grade.get('strength', 0.35))) if grade else None
    k = _key(FIN_VERSION, _fhash(raw), fps_in, round(fps_out, 4), ls, expo, portrait, gr,
             *_fin_code(lipsync.get('engine', 'musetalk') if lipsync else None))
    out = os.path.join(d, k + '.mp4')
    from .memo import touch
    touch(out)
    if os.path.exists(out):
        return out
    t = time.time()
    work = tempfile.mkdtemp(prefix='vf_fin_')
    try:
        src = raw
        late = bool(lipsync and lipsync.get('late'))
        if lipsync and not late:
            p = os.path.join(work, 'ls.mp4')
            if lipsync.get('engine', 'musetalk') == 'latentsync':
                _gen(['vf.gen.latentsync_run', src, lipsync['wav'], str(lipsync.get('offset', 0)), p, '8', '1.0',
                      str(lipsync.get('rate', 1.0))])
                fps_in = 25.0  # LatentSync resamples to 25 fps
            else:
                _gen(['vf.gen.lipsync', src, lipsync['wav'], str(lipsync.get('offset', 0)), p, str(lipsync.get('rate', 1.0))],
                     LS_MASK=lipsync.get('mask', ''))
            src = p
        import cv2
        c = cv2.VideoCapture(src)
        w, h = int(c.get(3)), int(c.get(4))
        c.release()
        p = os.path.join(work, 'deblur.mp4')
        # DENOISE_MEDIUM, not DEBLUR: measured 2026-11-02 on 3 clips, deblur + exposure gain doubled the
        # high-frequency energy (Laplacian 3.7 -> 7.3) = crunchy, posterised "pixelated" faces; denoise kept detail
        _gen(['vf.gen.rtx_vsr', src, p, str(w), str(h), 'DENOISE_MEDIUM'])
        src = p
        if expo:
            p = os.path.join(work, 'expo.mp4')
            cur, gain = _expo(src, p, target_l=float(expo))
            src = p
        if grade:
            p = os.path.join(work, 'grade.mp4')
            _grade(src, p, grade['ref'], float(grade.get('strength', 0.35)))
            src = p
        p = os.path.join(work, 'fg.mp4')
        _gen(['vf.gen.framegen', src, str(fps_in), str(fps_out), p, 'HIGH'])
        src = p
        if late:
            p = os.path.join(work, 'up.mp4')
            _gen(['vf.gen.rtx_vsr', src, p, str(w * UP), str(h * UP), UPQ])
            _gen(['vf.gen.lipsync', p, lipsync['wav'], str(lipsync.get('offset', 0)), out + '.part.mp4',
                  str(lipsync.get('rate', 1.0))],
                 LS_MASK=lipsync.get('mask', ''))
        else:
            _gen(['vf.gen.rtx_vsr', src, out + '.part.mp4', str(w * UP), str(h * UP), UPQ])
        os.replace(out + '.part.mp4', out)
        log(f'    finished {os.path.basename(raw)[:10]} in {time.time() - t:.0f}s'
            + (f' (exposure x{gain:.2f})' if expo else ''))
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if os.path.exists(out + '.part.mp4'):  # a failed step must not leave a 100+ MB partial in k6fin
            try:
                os.remove(out + '.part.mp4')
            except OSError:
                pass
