"""Talking shots: real audio-driven acting (InfiniteTalk) with the owner's sync rule built in (2026-11-07).

A shot opts in with `gen: {talk: {line: <dialogue text prefix> | [prefixes], prompt?, seed?, audio_cfg?, steps?}}`.
It replaces the old K6 clip + LatentSync paste for that shot (the owner: "still pictures with lips on top").

Per talking shot (runs after keyframes, BEFORE the K6 clips, because it moves shot boundaries):
  1. the line's take gets its interior pauses removed (InfiniteTalk never closes the mouth in a short pause) and is
     loudness-normalised; the episode mix uses this same wav, so picture and sound stay identical
  2. InfiniteTalk (tools/talk_video.py, talk venv: Wan2.1-I2V-14B-480P int8 + InfiniteTalk + lightx2v 4 steps) acts the
     keyframe over a window = shot +- 0.8 s, driven by an audio track that holds only this character's line(s), placed
     SHIFT later than the timeline (measured on 3 shots: the mouth leads the voice by 2-3 frames; SyncNet -2..-3 -> 0)
  3. tools/silent_mouth.py --analyse: per frame, is the mouth open, is the line sounding (timeline-aligned audio)
  4. cut points: the shot may only start where no open-mouth-without-voice frame lies before the first word, and must
     end before the mouth runs on after the last word. Shot boundaries move inside the window to satisfy that; the
     neighbour shot absorbs the change, so every later shot and every dialogue `at:` stays where it was
  5. finishing = the K6 chain (denoise, exposure, grade, NVIDIA frame generation, RTX VSR)
Report: out/talk.json (window, cuts, worst open-and-silent run inside the shot, flags).
Licences: InfiniteTalk + Wan2.1 Apache-2.0; Wan2GP runner = WanGP Community License 2.0 (free use, commercial outputs OK).
"""
import hashlib, json, math, os, subprocess, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALK_PY = os.path.join(_AI, 'tools', 'envs', 'talk', 'Scripts', 'python.exe')
CACHE = os.path.join(ROOT, 'cache', 'talk')
VERSION = 'talk|infinitetalk-wan2gp|v1'
FPS = 25
SHIFT = 0.08        # audio fed to the model this much later than the timeline (mouth leads the voice by 2-3 frames)
MARGIN = 0.8        # the generated window may reach this far outside the planned shot (room to move cut points)
TOL = 1             # open-and-silent frames tolerated next to a word (one frame = 40 ms)
MIN_DUR = 0.5
REROLL_AT = 3       # re-roll a take with more open-mouth-without-voice frames than this within 1 s of its line


def _fh(p):
    st = os.stat(p)
    return f'{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}'


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _ff(args):
    from .render import ffmpeg
    subprocess.run([ffmpeg(), '-v', 'error', '-y'] + args, check=True, timeout=600)


def _dur(p):
    from .audio import decode
    return decode(p).shape[-1] / 48000.0   # decode -> (2, n)


def tighten(wav):
    """Interior pauses out, loudness -18 LUFS. Cached by the source take."""
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, 'tight_' + _key('tight|v1', _fh(wav)) + '.wav')
    if not os.path.exists(out):
        _ff(['-i', wav, '-af', 'aformat=sample_rates=48000:channel_layouts=mono,loudnorm=I=-18:TP=-2:LRA=7,'
             'silenceremove=start_periods=1:start_threshold=-50dB,'
             'silenceremove=stop_periods=-1:stop_duration=0.05:stop_threshold=-40dB:stop_silence=0.03',
             '-ar', '48000', out + '.part.wav'])
        os.replace(out + '.part.wav', out)
    return out


def _track(lines, w0, w1, shift, out):
    """Mono 48 kHz track of length w1-w0 with each (wav, at) placed at at - w0 + shift."""
    from .audio import decode
    import soundfile as sf
    n = int(round((w1 - w0) * 48000))
    buf = np.zeros(n, np.float32)
    for wav, at in lines:
        x = decode(wav)   # (2, n) at 48 kHz
        x = x.mean(0) if x.ndim > 1 else x
        i = int(round((at - w0 + shift) * 48000))
        j = min(n, i + len(x))
        if j > max(i, 0):
            buf[max(i, 0):j] += x[max(0, -i):j - i]
    sf.write(out, buf, 48000)
    return out


def _lines_of(spec, g):
    want = g['talk']['line']
    want = want if isinstance(want, list) else [want]
    out = []
    for w in want:
        d = next((x for x in spec['dialogue'] if x['text'].startswith(w)), None)
        if d is None or not d.get('_wav'):
            raise ValueError(f"talk line '{w}' has no dialogue entry with a voice take")
        out.append(d)
    return out


def _runs(mask):
    r, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            r.append((i, j))
            i = j + 1
        else:
            i += 1
    return r


def _limits(a, w0):
    """From per-frame arrays -> timeline limits: earliest allowed start, first word, last word, latest allowed end."""
    op, sp = np.array(a['open']), np.array(a['speech'])
    idx = np.flatnonzero(sp)
    on, off = int(idx[0]), int(idx[-1])
    bad = op & ~sp
    s_min = 0
    for i, j in _runs(bad[:on]):
        if j - i + 1 > TOL or j < on - 1:   # any open-silent run before the first word (a lone 1-frame one touching it is OK)
            s_min = j + 1
    e_max = len(sp) - 1
    for i, j in _runs(bad[off + 1:]):
        i, j = i + off + 1, j + off + 1
        if j - i + 1 > TOL or i > off + 1:
            e_max = i - 1
            break
    t = lambda f: w0 + f / FPS
    return dict(s_min=t(s_min), onset=t(on), offset=t(off + 1), e_max=t(e_max + 1),
                inner=[(round(t(i), 2), j - i + 1) for i, j in _runs(bad[on:off + 1]) if j - i + 1 > TOL])


def _edge_bad(an):
    """Open-mouth-without-voice frames within 1 s before the first word and 1 s after the last word (what a cut next
    to the line can show). Decides re-rolls."""
    op, sp = np.array(an['open']), np.array(an['speech'])
    idx = np.flatnonzero(sp)
    on, off = int(idx[0]), int(idx[-1])
    bad = op & ~sp
    return int(bad[max(0, on - FPS):on].sum() + bad[off + 1:off + 1 + FPS].sum())


def _generate(spec, s, g, lines, log, w0_max=None, seed_add=0, plan_only=False):
    """-> (clip path, window start, window end, analysis dict). w0_max: the window must start no later than this
    (set when the previous shot is a talking shot whose mouth runs on after its line: this shot's closed-mouth lead
    has to cover that silence, 2026-11-07 ep02: 760 ms gaps nobody's clip covered)."""
    os.makedirs(CACHE, exist_ok=True)
    t0, t1 = s['start'], s['start'] + s['dur']
    last = max(float(d['at']) + _dur(d['_wav']) for d in lines)
    # the window hugs the line (generation time grows with every 72 frames): up to 1.0 s before the first word and
    # 0.6 s after the last, never more than MARGIN outside the planned shot. A shot longer than that is shortened
    # (its cut points must lie inside the window) and the neighbour shot absorbs the rest.
    first = min(float(d['at']) for d in lines)
    w0 = max(0.0, t0 - MARGIN, first - 1.0)
    if w0_max is not None:
        w0 = max(0.0, min(w0, w0_max))
    w0 = math.floor(w0 * FPS) / FPS
    w1 = min(spec['duration'], t1 + MARGIN, last + 0.6)
    tk = g['talk']
    prompt = tk.get('prompt') or g.get('motion') or 'a person speaking'
    gen_track = os.path.join(CACHE, 'gen_' + _key(VERSION, w0, w1, [(_fh(d['_wav']), d['at']) for d in lines]) + '.wav')
    if not os.path.exists(gen_track):   # never rewrite: the clip key hashes this file's mtime (a rewrite on every
        _track([(d['_wav'], float(d['at'])) for d in lines], w0, w1, SHIFT, gen_track)   # build re-acted every shot)
    seed = int(tk.get('seed', g.get('seed', 7))) + seed_add
    sc = tk.get('step_cache') or ''   # opt-in WanGP step cache (mag/tea): only keyed when set (old keys unchanged)
    k = _key(VERSION, _fh(g['_image']), _fh(gen_track), prompt, seed, int(tk.get('steps', 4)),
             float(tk.get('audio_cfg', 2.0)), *([sc, float(tk.get('step_cache_mult', 1.5))] if sc else []))
    clip = os.path.join(CACHE, k + '.mp4')
    if plan_only:   # batch prefetch: what the run below would generate
        return dict(image=os.path.abspath(g['_image']), audio=gen_track, prompt=prompt, out=clip + '.part.mp4',
                    seed=seed, steps=int(tk.get('steps', 4)), audio_cfg=float(tk.get('audio_cfg', 2.0)), clip=clip,
                    **({'step_cache': sc, 'step_cache_mult': float(tk.get('step_cache_mult', 1.5))} if sc else {}))
    from .memo import touch
    touch(clip)
    if not os.path.exists(clip):
        log(f"    {s['id']}: InfiniteTalk {w1 - w0:.1f}s window ({len(lines)} line(s))")
        from .proc import run
        try:
            from .gen import bridge
            bridge.stop_gen()   # its parked weights hold RAM the 14B talk model needs
        except Exception:
            pass
        # no gpu_lock here: tools/talk_video.py takes the same cross-process lock itself (holding it here deadlocked
        # the first ep02 build: the child waited for its own parent, 2026-11-07)
        if True:
            run([TALK_PY, os.path.join(ROOT, 'tools', 'talk_video.py'), '--image', os.path.abspath(g['_image']),
                 '--audio', gen_track, '--prompt', prompt, '--out', clip + '.part.mp4', '--seed',
                 str(seed), '--steps', str(int(tk.get('steps', 4))),
                 '--size', '480x832', '--audio-cfg', str(float(tk.get('audio_cfg', 2.0))), '--pad', '0',
                 '--ws-cap', '9'] + (['--step-cache', sc, '--step-cache-mult', str(float(tk.get('step_cache_mult', 1.5)))]
                                    if sc else []), timeout=3600, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        os.replace(clip + '.part.mp4', clip)
    an = clip[:-4] + '.mouth.json'
    if not os.path.exists(an):
        tl = clip[:-4] + '.timeline.wav'
        _track([(d['_wav'], float(d['at'])) for d in lines], w0, w1, 0.0, tl)
        from .proc import run
        run([TALK_PY, os.path.join(ROOT, 'tools', 'silent_mouth.py'), '--analyse', clip, tl, an], timeout=900, cwd=ROOT)
    return clip, w0, w1, json.load(open(an))


def _start_rng(L):
    """Allowed start: after the last open-mouth-without-voice frame, at most 1 frame before the first word."""
    hi = L['onset'] - 1.0 / FPS
    return min(L['s_min'], hi), hi


def _end_rng(L):
    """Allowed end: at least 1 frame after the last word, before the mouth runs on in silence."""
    lo = L['offset'] + 1.0 / FPS
    return lo, max(L['e_max'], lo)


def _set_starts(spec):
    t = 0.0
    for s in spec['shots']:
        s['start'] = t
        t += s['dur']


def stage(spec, shots, formats, log=print):
    """Generate every talking shot, move cut points, finish clips, add layers. Returns the talk shots handled."""
    talk = [s for s in shots if (s.get('gen') or {}).get('talk')]
    if not talk:
        return []
    t_all = time.time()
    for s in talk:  # 1. pause-free takes, shared with the mix
        for d in _lines_of(spec, s['gen']):
            if not d.get('_tight'):
                d['_wav'] = tighten(d['_wav'])
                d['_tight'] = True
    info = {}
    ids = [s['id'] for s in spec['shots']]
    talk_ids = {s['id'] for s in talk}
    # 2a. batch prefetch (2026-11-09): every first take whose window does not depend on another talking shot (the shot
    # before it is not a talk shot) is generated in ONE talk_video.py process, so the 14B model loads once instead of
    # once per shot (~90 s each; ep02 = 16 talk shots). The loop below then finds them cached.
    plans = []
    for s in talk:
        i = ids.index(s['id'])
        if i and ids[i - 1] in talk_ids:
            continue
        pl = _generate(spec, s, s['gen'], _lines_of(spec, s['gen']), log, None, 0, plan_only=True)
        if not os.path.exists(pl['clip']):
            plans.append((s['id'], pl))
    if len(plans) > 1:
        log(f"  talk: batch of {len(plans)} take(s) in one model load: {', '.join(x[0] for x in plans)}")
        from .proc import run
        try:
            from .gen import bridge
            bridge.stop_gen()
        except Exception:
            pass
        jf = os.path.join(CACHE, 'batch_' + _key(VERSION, [p['out'] for _, p in plans]) + '.json')
        json.dump([{k: v for k, v in p.items() if k != 'clip'} for _, p in plans], open(jf, 'w'))
        try:
            run([TALK_PY, os.path.join(ROOT, 'tools', 'talk_video.py'), '--jobs', jf, '--size', '480x832', '--pad', '0',
                 '--ws-cap', '9'], timeout=3600 * len(plans), cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        except Exception as e:   # whatever finished is kept; the loop below generates the rest one by one
            log(f'  talk: batch stopped ({type(e).__name__}): the remaining takes run one by one')
        for _, p in plans:
            if os.path.exists(p['out']) and not os.path.exists(p['clip']):
                os.replace(p['out'], p['clip'])
    for s in talk:  # 2-3. act + measure (in timeline order: a shot's window depends on the talking shot before it)
        g = s['gen']
        i = ids.index(s['id'])
        prev = info.get(ids[i - 1]) if i else None
        w0_max = (prev['lim']['e_max'] - 2.0 / FPS) if prev else None
        clip, w0, w1, an = _generate(spec, s, g, _lines_of(spec, g), log, w0_max)
        info[s['id']] = dict(clip=clip, w0=w0, w1=w1, an=an, lim=_limits(an, w0), w0_max=w0_max, seed_add=0)
    by_id = {s['id']: s for s in talk}

    def solve():
        """4. cut points: every boundary next to a talking shot goes where the fewest open-mouth-without-voice frames
        show (left shot after its last word + right shot before its first word), never cutting a word, never outside a
        clip, every shot >= MIN_DUR; ties -> closest to the planned cut. -> (boundaries, flags, bad cuts)"""
        B = [s['start'] for s in spec['shots']] + [spec['duration']]
        flags, badcuts = {}, []
        bad = {k: np.array(v['an']['open']) & ~np.array(v['an']['speech']) for k, v in info.items()}
        fr = lambda k, t: int(round((t - info[k]['w0']) * FPS))
        for bi in range(1, len(spec['shots'])):
            A, Bs = ids[bi - 1], ids[bi]
            if A not in info and Bs not in info:
                continue
            lo, hi = B[bi - 1] + MIN_DUR, B[bi + 1] - MIN_DUR
            if A in info:
                LA = info[A]['lim']
                lo, hi = max(lo, LA['offset'] + 1.0 / FPS), min(hi, info[A]['w1'])
            if Bs in info:
                LB = info[Bs]['lim']
                lo, hi = max(lo, info[Bs]['w0']), min(hi, LB['onset'] - 1.0 / FPS)
            if hi + 1e-6 < lo <= hi + 2.0 / FPS + 1e-6:   # a 1-2 frame gap: cut where the next clip starts (2026-11-08:
                hi = lo                                       # it fell back to the planned cut, 2 s past the left clip's end)
            if lo > hi + 1e-6:
                flags.setdefault(A if A in info else Bs, []).append(
                    f'cut {A}|{Bs}: no frame covers the gap ({lo:.2f} > {hi:.2f}); kept {B[bi]:.2f}s')
                badcuts.append((99, 99 if A in info else 0, 99 if Bs in info else 0, A, Bs))
                continue
            best = None
            for n in range(int(math.floor(lo * FPS + 1e-6)), int(math.ceil(hi * FPS - 1e-6)) + 1):
                t = n / FPS
                if t < lo - 1e-6 or t > hi + 1e-6:
                    continue
                ca = int(bad[A][max(0, fr(A, LA['offset'])):max(0, fr(A, t))].sum()) if A in info else 0
                cb = int(bad[Bs][max(0, fr(Bs, t)):max(0, fr(Bs, LB['onset']))].sum()) if Bs in info else 0
                key = (ca + cb, abs(t - B[bi]))
                if best is None or key < best[0]:
                    best = (key, t, ca, cb)
            B[bi] = best[1]
            if best[0][0] > TOL:
                flags.setdefault(A if A in info else Bs, []).append(
                    f'cut {A}|{Bs} at {B[bi]:.2f}s still shows {best[0][0]} open-mouth frame(s) without voice')
                badcuts.append((best[0][0], best[2], best[3], A, Bs))
        return B, flags, badcuts

    B, flags, badcuts = solve()
    # re-roll the shot that shows most of a bad cut (new seed, same window); keep the take only if the whole edit
    # shows fewer open-mouth-without-voice frames. gen.talk.rerolls / episode gen.talk_rerolls (default 1).
    # each shot may re-roll up to its own `rerolls` times (a `tried` set capped it at one per build, 2026-11-08)
    for _ in range(6):
        todo = []
        for tot, ca, cb, A, Bs in sorted(badcuts, reverse=True):
            who = A if ca >= cb else Bs
            if who in info and who not in todo:
                rr = int(by_id[who]['gen']['talk'].get('rerolls', (spec.get('gen') or {}).get('talk_rerolls', 1)))
                if info[who]['seed_add'] // 1000 < rr:
                    todo.append(who)
        if not todo:
            break
        for who in todo:
            I = info[who]
            sa = I['seed_add'] + 1000
            log(f"    {who}: cut shows open mouth without voice -> re-roll (seed +{sa})")
            sh = by_id[who]
            c2, a0, a1, an2 = _generate(spec, sh, sh['gen'], _lines_of(spec, sh['gen']), log, I['w0_max'], seed_add=sa)
            before = sum(c[0] for c in badcuts)
            old = info[who]
            info[who] = dict(clip=c2, w0=a0, w1=a1, an=an2, lim=_limits(an2, a0), w0_max=I['w0_max'], seed_add=sa)
            B2, f2, bc2 = solve()
            if sum(c[0] for c in bc2) < before:
                B, flags, badcuts = B2, f2, bc2
                log(f"    {who}: re-roll kept ({before} -> {sum(c[0] for c in bc2)} bad frames at cuts)")
            else:
                info[who] = dict(old, seed_add=sa)   # keep the old take; this seed is spent
                log(f"    {who}: re-roll not better ({sum(c[0] for c in bc2)} vs {before}); kept the first take")
    moved = []
    for j, sh in enumerate(spec['shots']):
        nd = round(B[j + 1] - B[j], 4)
        if abs(nd - sh['dur']) > 1e-3:
            moved.append(f"{sh['id']} {sh['dur']:.2f}->{nd:.2f}s")
            sh['dur'] = nd
    _set_starts(spec)
    if moved:
        log('  talk: cut points moved: ' + ', '.join(moved))
    # 5. finish + layer
    from . import k6
    from .interp import target_fps
    from .stages import _k6_finish_opts
    report = []
    for s in talk:
        g, I = s['gen'], info[s['id']]
        a, sp = I['an'], np.array(I['an']['speech'])
        f0 = int(round((s['start'] - I['w0']) * FPS))
        f1 = int(round((s['start'] + s['dur'] - I['w0']) * FPS))
        bad = (np.array(a['open']) & ~sp)[max(0, f0):f1]
        worst = max([j - i + 1 for i, j in _runs(bad)], default=0)
        ft = round(target_fps(spec.get('fps', 60), 1.0), 4)
        fin = k6.finish(I['clip'], FPS, ft, log=log, **_k6_finish_opts(spec, s, formats))
        layer = dict(type='video', src=fin, fps=ft, fit=g.get('fit', 'cover'), start=s['start'] - I['w0'], speed=1.0,
                     sharpen=float(g.get('sharpen', 0.0)))
        for k in ('focus', 'zoom', 'tint', 'tint_amount'):
            if k in g:
                layer[k] = g[k]
        s.setdefault('layers', []).insert(0, layer)
        if spec.get('talk_look') and not s.get('look'):
            # InfiniteTalk clips come out already exposed (the K6 exposure lift adds ~0), so the episode look's black
            # point crushed the faces to silhouettes (2026-11-08): talking shots get their own look overrides
            s['look'] = dict(spec['talk_look'])
        fl = flags.get(s['id'], []) + ([f'mouth open without voice for {worst} frames ({worst * 40} ms)'] if worst > TOL else []) \
            + [f'open mouth in a pause inside the line at {t}s ({n} frames)' for t, n in I['lim']['inner']]
        g['_take'] = dict(talk=True, clip=I['clip'], window=[I['w0'], I['w1']], start=s['start'], dur=s['dur'],
                          worst_open_silent_frames=worst, flags=fl)
        report.append(dict(shot=s['id'], start=round(s['start'], 3), dur=round(s['dur'], 3), window=[I['w0'], I['w1']],
                           limits={k: (round(v, 3) if isinstance(v, float) else v) for k, v in I['lim'].items()},
                           worst_open_silent_frames=worst, flags=fl))
        log(f"    {s['id']}: {'FLAG ' + '; '.join(fl) if fl else 'ok'} (worst open-silent run {worst} frame(s))")
    os.makedirs(os.path.join(spec['root'], 'out'), exist_ok=True)
    json.dump(dict(moved=moved, shots=report), open(os.path.join(spec['root'], 'out', 'talk.json'), 'w'), indent=1)
    log(f'  talk: {len(talk)} shot(s) in {time.time() - t_all:.0f}s')
    return talk
