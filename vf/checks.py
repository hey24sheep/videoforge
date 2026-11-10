"""Gates that were hand-run scratch scripts until 2026-11-06 and are now pipeline steps (see docs/workflow.md).

    python -m vf check <episode.yaml> [--video out/x_short.mp4] [--no-sync]      post-render playback + lip-sync gates
    python -m vf script-qa <script.md ...> --bible bible.md [--ref REF.md]       script gate (local model rubric)

check:
  - luma_dip     a shot's brightness drops > 18 % within 14 frames and recovers (a "bulb stutter": the owner read two of
                 them in a test episode's handset shot as a video glitch). A shot that really wants it says `allow_dip: true`.
  - frozen       >= 6 identical frames in a row inside a shot (not on `allow_black` shots). Exact facts, so formulas.
  - lipsync      SyncNet (vf/gen/syncnet.py) audio-video offset of every lip-synced or talking shot: |offset| > 4 frames
                 (160 ms) or confidence < 4 fails. Calibrated on three test takes: approved -3, bad +11, fixed +2.
  - silent_mouth tools/silent_mouth.py on every talking shot of the FINAL video: the mouth open while no voice plays
                 for more than 100 ms fails (the owner's #1 complaint, 2026-11-07; whole-clip SyncNet passed such shots).
  - props        (WARN, review) per series `check_props: [lantern, ...]` (default: none): Qwen3-VL asks per TALK shot
                 whether the prop is visible, held by a hand, or hung/standing on something; visible + neither = it may
                 float (owner, 2026-11-09: a lantern hung in mid-air at a witness's chest). Tested on 9 test shots: caught
                 the floating one, 3 false alarms (wall lantern, lantern on a crate), so it flags for a human look:
                 out/prop_review.jpg holds a frame of every flagged shot.
  - loudness     (WARN) integrated LUFS and true peak of the whole video (ffmpeg ebur128): true peak > -1 dBTP warns.
  Shot times come from out/timeline.json (the build writes it; vf/talk.py moves cut points), else the yaml.
script-qa: Qwen3.5-9B answers a yes/no rubric about each script (vf/gen/scriptqa.py); a script must reach the mean of the
  reference script minus 0.25 (0-10 scale) to go on. Blind Opus + Sonnet critics (prompts/script_critic.md) are the second gate.
"""
import glob, json, os, subprocess, sys, tempfile
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

import numpy as np
import yaml

GATES = dict(dip_drop=0.18, dip_frames=14, frozen_run=6, frozen_diff=0.03, sync_frames=4, sync_conf=4.0,
             script_margin=0.25, silent_mouth_ms=100)
TALK_PY = os.path.join(_AI, 'tools', 'envs', 'talk', 'Scripts', 'python.exe')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _series_yaml(ep_root):
    """series.yaml next to the episode folder (series/<name>/series.yaml), {} if none."""
    p = os.path.join(os.path.dirname(ep_root), 'series.yaml')
    return (yaml.safe_load(open(p, encoding='utf-8')) or {}) if os.path.exists(p) else {}


def shots_of(yaml_path):
    ep = yaml.safe_load(open(yaml_path, encoding='utf-8')) or {}
    out, t = [], 0.0
    for i, s in enumerate(ep.get('shots', [])):
        d = float(s.get('dur', 0))
        g = s.get('gen') or {}
        out.append(dict(id=s.get('id', f'shot{i + 1:02d}'), start=t, end=t + d, allow_black=bool(s.get('allow_black')),
                        allow_dip=bool(s.get('allow_dip')), lipsync=bool(g.get('lipsync') or g.get('talk')),
                        talk=bool(g.get('talk'))))
        t += d
    tl = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), 'out', 'timeline.json')
    if os.path.exists(tl):
        T = {r['id']: r for r in json.load(open(tl))}
        if set(T) == {s['id'] for s in out}:
            for s in out:
                s['start'], s['end'] = T[s['id']]['start'], T[s['id']]['start'] + T[s['id']]['dur']
    return out


CHECK_CACHE = os.path.join(ROOT, 'cache', 'check')   # per-shot results keyed on the shot's own frames + audio


def _ckey(gate, *files):
    """sha1 over the extracted shot files: an unchanged shot (same render) is never measured twice (2026-11-09: a
    cached release spent ~7 of its 10.5 min re-measuring identical talk shots)."""
    import hashlib
    h = hashlib.sha1(gate.encode())
    for f in files:
        with open(f, 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 20), b''):
                h.update(b)
    return os.path.join(CHECK_CACHE, f'{gate}_{h.hexdigest()[:24]}.json')


def _fkey(gate, video, start, dur):
    """Key on the DECODED frames of a segment (md5 of a small decode): stable even when a re-encode is not."""
    r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', video, '-an',
                        '-vf', 'scale=96:-2', '-f', 'md5', '-'], capture_output=True, text=True)
    return os.path.join(CHECK_CACHE, f"{gate}_{r.stdout.strip().split('=')[-1][:24]}.json")


def _cget(path):
    try:
        return json.load(open(path)) if os.path.exists(path) else None
    except Exception:
        return None


def _cput(path, obj):
    os.makedirs(CHECK_CACHE, exist_ok=True)
    json.dump(obj, open(path, 'w'))


def silent_mouth(video, shots, log=print):
    """Open mouth without voice inside every talking shot of the final video (frame level)."""
    todo = [s for s in shots if s.get('talk')]
    out, tmp = [], tempfile.mkdtemp(prefix='vf_sm_')
    stem = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(video))), 'out', 'dialogue_stem.wav')
    stem = stem if os.path.exists(stem) else None   # the mix's music bed reads as 'sound' in every frame
    for s in todo:
        p = os.path.join(tmp, s['id'] + '.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f"{s['start']:.3f}", '-t', f"{s['end'] - s['start']:.3f}",
                        '-i', video, '-c:v', 'libx264', '-crf', '14'] + (['-an'] if stem else ['-c:a', 'aac']) + [p], check=True)
        args = [p]
        if stem:
            a = os.path.join(tmp, s['id'] + '.wav')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f"{s['start']:.3f}", '-t', f"{s['end'] - s['start']:.3f}",
                            '-i', stem, a], check=True)
            args = ['--audio', a, p]
        else:
            log('  silent_mouth: no out/dialogue_stem.wav, judging against the full mix (music hides silences)')
        ck = _ckey('silent_mouth_v1', *[x for x in args if x != '--audio'])
        m = _cget(ck)
        if m is None:
            r = subprocess.run([TALK_PY, os.path.join(ROOT, 'tools', 'silent_mouth.py')] + args, capture_output=True,
                               text=True, timeout=900)
            line = next((l for l in r.stdout.splitlines() if l.startswith('{')), None)
            if not line:
                out.append(dict(gate='silent_mouth', shot=s['id'], t=round(s['start'], 2), ok=False,
                                detail='unmeasured: ' + (r.stderr or '')[-200:]))
                continue
            m = json.loads(line)
            _cput(ck, m)
        else:
            log(f"  silent_mouth {s['id']}: unchanged shot, cached result")
        ms = m['worst_run_frames'] * 1000.0 / m['fps']
        out.append(dict(gate='silent_mouth', shot=s['id'], t=round(s['start'], 2), ok=ms <= GATES['silent_mouth_ms'],
                        worst_ms=round(ms), runs=m['open_silent_runs'],
                        detail=f"mouth open without voice for {ms:.0f} ms at worst"))
    return out


def _series(video):
    """Per-frame mean luma + a 54x96 blurred grey copy. ffmpeg decodes and shrinks to 108x192 grey itself (NVDEC when
    available): ~4x faster than decoding full 1080x1920 frames into Python (2026-11-09)."""
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    cap.release()
    W, H = 108, 192
    raw = None
    for hw in (['-hwaccel', 'cuda'], []):
        r = subprocess.run(['ffmpeg', '-v', 'error'] + hw + ['-i', video, '-an', '-vf', f'scale={W}:{H}:flags=area,format=gray',
                            '-f', 'rawvideo', '-'], capture_output=True)
        if r.returncode == 0 and r.stdout:
            raw = r.stdout
            break
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
    luma = fr.mean((1, 2))
    low = np.array([cv2.GaussianBlur(cv2.resize(g, (54, 96), interpolation=cv2.INTER_AREA), (0, 0), 1.0) for g in fr])
    return fps, luma, low


def playback(video, shots):
    """-> list of dict(gate, shot, t, detail) for luma dips and frozen runs inside shots."""
    fps, L, G = _series(video)
    flags = []
    d = np.abs(np.diff(G, axis=0)).mean((1, 2)) if len(G) > 1 else np.zeros(0)
    for s in shots:
        a, b = int(round((s['start'] + 0.1) * fps)), int(round((s['end'] - 0.1) * fps))
        b = min(b, len(L))
        if b - a < 30:
            continue
        if not (s['allow_black'] or s['allow_dip']):
            i = a + 6
            while i < b - GATES['dip_frames'] - 1:
                base = float(np.median(L[i - 6:i]))
                if base > 3:
                    w = L[i:i + GATES['dip_frames']]
                    k = int(np.argmin(w))
                    after = L[min(i + GATES['dip_frames'], b - 1)]
                    if w[k] < (1 - GATES['dip_drop']) * base and abs(after - base) < 0.12 * base:
                        flags.append(dict(gate='luma_dip', shot=s['id'], t=round(i / fps, 2),
                                          detail=f'brightness {base:.1f} -> {w[k]:.1f} ({(1 - w[k] / base) * 100:.0f} % dip) and back'))
                        i += GATES['dip_frames']
                        continue
                i += 1
        if not s['allow_black']:
            run = 0
            for i in range(a, min(b, len(d))):
                run = run + 1 if d[i] < GATES['frozen_diff'] else 0
                if run == GATES['frozen_run']:
                    flags.append(dict(gate='frozen', shot=s['id'], t=round((i - run + 1) / fps, 2),
                                      detail=f'{GATES["frozen_run"]}+ identical frames in a row'))
    return flags


def sync(video, shots, log=print):
    """SyncNet offset per lip-synced shot -> list of dict(gate, shot, t, detail, ok)."""
    todo = [s for s in shots if s['lipsync']]
    if not todo:
        return []
    from .gen.bridge import run_gen
    ff = 'ffmpeg'
    jobs, tmp = [], tempfile.mkdtemp(prefix='vf_check_')
    for s in todo:
        p = os.path.join(tmp, s['id'] + '.mp4')
        subprocess.run([ff, '-v', 'error', '-y', '-ss', f"{s['start']:.3f}", '-t', f"{s['end'] - s['start']:.3f}", '-i', video,
                        '-c:v', 'libx264', '-crf', '14', '-c:a', 'aac', p], check=True)
        jobs.append(dict(video=p))
    keys = [_ckey('syncnet_v1', j['video']) for j in jobs]
    res = [_cget(k) for k in keys]
    miss = [i for i, r in enumerate(res) if r is None]
    log(f'[check] SyncNet on {len(jobs)} lip-synced shot(s) ({len(jobs) - len(miss)} unchanged, cached)')
    if miss:
        got = run_gen('syncnet', [jobs[i] for i in miss], persistent=False, check=False, timeout=1800)
        for i, r in zip(miss, got):
            res[i] = r
            if r.get('ok'):
                _cput(keys[i], r)
    out = []
    for s, r in zip(todo, res):
        if not r.get('ok'):
            out.append(dict(gate='lipsync', shot=s['id'], t=round(s['start'], 2), ok=False,
                            detail='unmeasured: ' + str(r.get('error'))))
            continue
        bad = abs(r['offset']) > GATES['sync_frames'] or r['conf'] < GATES['sync_conf']
        out.append(dict(gate='lipsync', shot=s['id'], t=round(s['start'], 2), ok=not bad, offset_frames=r['offset'],
                        conf=round(r['conf'], 2),
                        detail=f"mouth {'lags' if r['offset'] > 0 else 'leads'} the voice by {abs(r['offset']) * 40} ms, "
                               f"confidence {r['conf']:.1f}"))
    return out


def props(video, shots, names, out_dir, log=print):
    """Floating-prop review: [dict(gate, shot, t, detail, ok)], ok=False means 'look at it' (a WARN, never a block)."""
    from .gen.bridge import run_gen
    # talk shots only: their hands are out of frame, so that is where props float. On every shot the judge counted
    # street lamp posts as lanterns (9 false alarms, 2026-11-09)
    todo = [s for s in shots if s.get('talk') and s['end'] - s['start'] > 0.4]
    tmp = tempfile.mkdtemp(prefix='vf_props_')
    jobs = []
    for s in todo:
        p = os.path.join(tmp, s['id'] + '.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f"{s['start'] + 0.05:.3f}",
                        '-t', f"{max(0.4, s['end'] - s['start'] - 0.1):.3f}", '-i', video, '-an', '-vf', 'scale=540:-2',
                        '-c:v', 'libx264', '-crf', '20', p], check=True)
        qs = []
        for n in names:
            qs += [dict(q=f'Is a {n} visible in the frames?', expect='no'),
                   dict(q=f'Is a hand gripping the handle or body of the {n}?', expect='yes'),
                   dict(q=f'Is the {n} hanging from a hook or bracket on a wall, or standing on a surface?', expect='yes')]
        jobs.append(dict(clip=p, questions=qs, frames=8, describe=False))
    keys = [_fkey('props_v1_' + '-'.join(names), video, s['start'] + 0.05, max(0.4, s['end'] - s['start'] - 0.1))
            for s in todo]
    res = [_cget(k) for k in keys]
    miss = [i for i, r in enumerate(res) if r is None]
    log(f'[check] prop review ({", ".join(names)}) on {len(jobs)} shot(s) ({len(jobs) - len(miss)} unchanged, cached)')
    if miss:
        got = run_gen('judge', [jobs[i] for i in miss], check=False)
        for i, r in zip(miss, got):
            res[i] = r
            if r.get('ok'):
                _cput(keys[i], r)
    out, sheet = [], []
    for s, r, j in zip(todo, res, jobs):
        if not r.get('ok'):
            continue
        A = [a['p_yes'] for a in r['answers']]
        for k, n in enumerate(names):
            vis, held, sup = A[3 * k:3 * k + 3]
            if vis >= 0.5:
                bad = held < 0.5 and sup < 0.5
                out.append(dict(gate='props', shot=s['id'], t=round(s['start'], 2), ok=not bad, warn=True,
                                detail=f"{n} visible ({vis:.2f}), held {held:.2f}, hung/standing {sup:.2f}"
                                       + (' -> may FLOAT: look at out/prop_review.jpg' if bad else '')))
                if bad:
                    sheet.append((s['id'], j['clip']))
    if sheet:
        ims = []
        for sid, clip in sheet:
            f = os.path.join(tmp, sid + '.png')
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '0.6', '-i', clip, '-frames:v', '1', '-vf', 'scale=270:-2', f])
            if os.path.exists(f):
                ims.append(f)
        if ims:
            args = sum([['-i', f] for f in ims], [])
            fc = ['-filter_complex', f'hstack=inputs={len(ims)}'] if len(ims) > 1 else []
            subprocess.run(['ffmpeg', '-v', 'error', '-y'] + args + fc + [os.path.join(out_dir, 'prop_review.jpg')])
    return out


def loudness(video):
    """Integrated loudness + true peak of the whole video (a WARN gate: true peak above -1 dBTP)."""
    import re
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', video, '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                       capture_output=True, text=True, encoding='utf-8', errors='replace').stderr
    tail = r[r.rfind('Summary:'):]
    I = re.search(r'I:\s+(-?[\d.]+) LUFS', tail)
    P = re.search(r'Peak:\s+(-?[\d.]+|-inf) dBFS', tail)
    lufs = float(I.group(1)) if I else None
    peak = float(P.group(1)) if P and P.group(1) != '-inf' else None
    ok = peak is None or peak <= -1.0
    return dict(gate='loudness', shot='(all)', t=0.0, ok=ok, warn=True, lufs=lufs, true_peak=peak,
                detail=f'{lufs} LUFS, true peak {peak} dBTP' + ('' if ok else ' (> -1 dBTP)'))


def check(yaml_path, video=None, do_sync=True, log=print):
    root = os.path.dirname(os.path.abspath(yaml_path))
    if not video:
        c = [p for p in glob.glob(os.path.join(root, 'out', '*.mp4')) if 'phone' not in p]
        if not c:
            raise SystemExit('no rendered video under out/: pass --video')
        video = max(c, key=os.path.getmtime)
    shots = shots_of(yaml_path)
    flags = playback(video, shots)
    syncs = (sync(video, shots, log) + silent_mouth(video, shots, log)) if do_sync else []
    os.makedirs(os.path.join(root, 'out'), exist_ok=True)
    names = (_series_yaml(root).get('check_props') or []) if do_sync else []
    warns = (props(video, shots, names, os.path.join(root, 'out'), log) if names else []) + [loudness(video)]
    log(f'[check] {os.path.basename(video)}: {len(shots)} shots')
    for f in flags:
        log(f"  FAIL {f['gate']:9s} {f['shot']:14s} {f['t']:6.2f}s  {f['detail']}")
    for s in syncs:
        log(f"  {'ok  ' if s['ok'] else 'FAIL'} {s['gate']:9s} {s['shot']:14s} {s['t']:6.2f}s  {s['detail']}")
    for s in warns:
        log(f"  {'ok  ' if s['ok'] else 'WARN'} {s['gate']:9s} {s['shot']:14s} {s['t']:6.2f}s  {s['detail']}")
    fails = len(flags) + sum(not s['ok'] for s in syncs)
    nw = sum(not s['ok'] for s in warns)
    log(f"[check] {'PASS' if not fails else 'BLOCK'}: {fails} failing check(s), {nw} warning(s) to review")
    res = dict(video=video, pass_=not fails, playback=flags, sync=syncs, review=warns, gates=GATES)
    json.dump(res, open(os.path.join(root, 'out', 'check.json'), 'w'), indent=1)
    return res


def script_qa(files, bible, ref=None, log=print, out_json=None):
    """Score script files (and an optional reference script scored in the same run). -> list of result dicts."""
    from .gen.bridge import run_gen
    bible_txt = open(bible, encoding='utf-8').read()
    paths = ([ref] if ref else []) + [f for f in files if f != ref]
    jobs = [dict(script=open(p, encoding='utf-8').read(), bible=bible_txt, label=os.path.basename(p)) for p in paths]
    res = run_gen('scriptqa', jobs, persistent=False, check=False, timeout=3600)
    ok = [r for r in res if r.get('ok')]
    for r in res:
        if not r.get('ok'):
            log(f"{r.get('label')}: ERROR {r.get('error')}")
    if not ok:
        raise SystemExit('script scoring failed')
    keys = list(ok[0]['scores'])
    log('script                         ' + ' '.join(f'{k[:7]:>7s}' for k in keys) + '   mean/10  words  spoken s')
    mean = lambda r: 10 * sum(r['scores'].values()) / len(r['scores'])
    base = mean(ok[0]) if ref else None
    for r in ok:
        flag = ''
        if ref and r['label'] != os.path.basename(ref):
            flag = '  PASS' if mean(r) >= base - GATES['script_margin'] else '  BLOCK (below the reference)'
        timing = ''
        if r.get('shots_s') and not (54.0 <= r["shots_s"] <= 120.0 and r['last_line'] <= r['shots_s']):
            timing = f"  TIMING: shots add up to {r['shots_s']} s, last line at {r['last_line']} s (want 54-120 s, last line inside)"
        log(f"{r['label'][:30]:30s} " + ' '.join(f"{r['scores'][k]:7.2f}" for k in keys) +
            f"   {mean(r):6.2f}  {r['words']:5d}  {r['spoken_s']:6.1f}{flag}{timing}")
    for r in ok:
        log(f"\n[{r['label']}] editor notes: " + (r['notes'] or '').replace('\n', ' '))
    if out_json:
        json.dump(res, open(out_json, 'w'), indent=1)
    return res
