"""vf archive: keep what an episode was made from, so a lost render can be made again with the same people and voices.

    vf archive <episode.yaml> [--out DIR]     (default DIR: <episode folder>/archive)

Writes DIR/lock.json and copies next to it:
  voices/   the kept take of every dialogue line (the exact audio in the episode) + each speaker's reference voice
  keyframes/ the first frame of every AI shot (the exact faces and sets the clips were animated from)
  refs/     the cast reference faces
lock.json lists, per line and per shot, the files above plus everything that picked them: voice settings, seeds,
keyframe seed, prompts, the kept clip take and its seed, the cast sheet, the model settings (VF_*) and the videoforge
commit. Run it after the final build (out/voices.json, out/shots.json, out/talk.json must exist).

To remake a shot exactly: pin `gen.image` to keyframes/<shot>.png and `gen.seed` to the kept seed (a clip is the same
when the model files and settings are the same). A voice line: the take in voices/ is the audio itself.
"""
import json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _copy(src, dst_dir, name):
    if not src or not os.path.isfile(src) or os.path.getsize(src) == 0:
        return None
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name)
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy2(src, dst)
    return os.path.relpath(dst, os.path.dirname(dst_dir)).replace('\\', '/')


def _safe(s):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(s))[:40]


def archive(yaml_path, out_dir=None, log=print):
    from . import spec as S
    spec = S.load(yaml_path)
    ep = spec['root']
    out = os.path.join(ep, 'out')
    dst = os.path.abspath(out_dir or os.path.join(ep, 'archive'))
    if not os.path.exists(os.path.join(out, 'voices.json')):
        raise SystemExit('archive: run the final build first (missing out/voices.json)')
    voices = json.load(open(os.path.join(out, 'voices.json'), encoding='utf-8'))
    if os.path.exists(os.path.join(out, 'shots.json')):
        shots = json.load(open(os.path.join(out, 'shots.json'), encoding='utf-8'))
    else:   # built before shots.json existed: episode.yaml gen blocks + clip_qc takes + cast_qc keyframes
        takes = json.load(open(os.path.join(out, 'clip_qc.json'), encoding='utf-8')) if os.path.exists(
            os.path.join(out, 'clip_qc.json')) else {}
        cq = json.load(open(os.path.join(out, 'cast_qc.json'), encoding='utf-8')) if os.path.exists(
            os.path.join(out, 'cast_qc.json')) else {}
        kfs = {sid: (r.get('gate') or {}).get('chosen') or r.get('keyframe')
               for sid, r in ((cq.get('shots') or {}).items() if isinstance(cq.get('shots'), dict) else [])}
        shots = dict(takes=takes, shots={s['id']: dict(s['gen'], _image=kfs.get(s['id']))
                                         for s in spec['shots'] if s.get('gen')})
        log('archive: no out/shots.json (built before 2026-11-14): keyframes only for cast shots; rebuild once for all')
    talk = json.load(open(os.path.join(out, 'talk.json'), encoding='utf-8')) if os.path.exists(
        os.path.join(out, 'talk.json')) else {}

    lines = []
    for i, v in enumerate(voices):
        take = _copy(v.get('wav'), os.path.join(dst, 'voices'), f"{i:02d}_{_safe(v.get('who'))}.wav")
        lines.append(dict({k: v[k] for k in v if k != 'wav'}, take=take))
    refs = {}
    for who, cfg in (spec.get('voices') or {}).items():
        ref = cfg.get('ref') or (cfg.get('direct') or {}).get('anchor')
        if ref:
            path = ref if os.path.isabs(ref) else os.path.normpath(os.path.join(ep, ref))
            if not os.path.exists(path):
                path = os.path.normpath(os.path.join(os.path.dirname(ep), ref))
            refs[who] = _copy(path, os.path.join(dst, 'voices'), f'ref_{_safe(who)}.wav')
    shot_rows = []
    for sid, g in (shots.get('shots') or {}).items():
        kf = _copy(g.get('_image'), os.path.join(dst, 'keyframes'), f'{_safe(sid)}.png')
        take = (shots.get('takes') or {}).get(sid) or {}
        shot_rows.append(dict(id=sid, keyframe=kf, kept_seed=take.get('seed'), kept_score=take.get('score'),
                              gen={k: v for k, v in g.items() if not k.startswith('_')}))
    cast = {}
    cq_path = os.path.join(out, 'cast_qc.json')
    sheet = (json.load(open(cq_path, encoding='utf-8')).get('cast') if os.path.exists(cq_path) else None) or {}
    if not sheet:
        try:
            from .cast import sheet as cast_sheet   # the series cast sheet merged with the episode's
            sheet = cast_sheet(spec) or {}
        except Exception:
            sheet = spec.get('cast') if isinstance(spec.get('cast'), dict) else {}
    for key, c in sheet.items():
        files = [_copy(r if os.path.isabs(r) else os.path.normpath(os.path.join(os.path.dirname(ep), r)),
                       os.path.join(dst, 'refs'), f'{_safe(key)}_{j}{os.path.splitext(r)[1]}')
                 for j, r in enumerate((c or {}).get('refs') or [])]
        cast[key] = dict({k: v for k, v in (c or {}).items() if k != 'refs'}, refs=[f for f in files if f])
    try:
        commit = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ''
    from . import config
    lock = dict(episode=spec.get('episode'), title=spec.get('title'), made=time.strftime('%Y-%m-%d %H:%M'),
                videoforge=commit, settings={r[0]: r[1] for r in config.rows()},
                voices=spec.get('voices'), voice_refs=refs, lines=lines,
                cast=cast, shots=shot_rows, talk=talk.get('shots') if isinstance(talk, dict) else talk,
                style=spec.get('style'), look=spec.get('look'), gen=spec.get('gen'))
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, 'lock.json'), 'w', encoding='utf-8') as f:
        json.dump(lock, f, indent=1, default=str, ensure_ascii=False)
    n_k = sum(1 for s in shot_rows if s['keyframe'])
    n_v = sum(1 for l in lines if l['take'])
    log(f"archive: {os.path.join(dst, 'lock.json')}  ({n_v}/{len(lines)} voice takes, {len(refs)} voice refs, {n_k}/{len(shot_rows)} "
        f"keyframes, {sum(len(c['refs']) for c in cast.values())} cast refs)")
    gaps = [l['text'][:40] for l in lines if not l['take']] + [s['id'] for s in shot_rows if not s['keyframe']]
    if gaps:
        log('archive: MISSING (not recorded or gone from the cache; one cached rebuild records them): ' + ', '.join(gaps))
    return dst


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='vf archive', description=__doc__.split('\n\n')[0])
    ap.add_argument('episode')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    archive(a.episode, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
