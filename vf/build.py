"""videoforge CLI.

    python -m vf build <episode.yaml> --formats yt,short [--preview] [--force] [--no-cache]
    python -m vf formats                      list output formats

Final renders go to <episode>/out/<title>_<fmt>.mp4 (previews: _<fmt>_preview.mp4, half-res 30 fps).
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from . import memo, spec as S
from .render import render_segment, renderer_code, shot_key, shot_deps, ffmpeg
from . import captions, audio, qc as QCmod


def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_') or 'episode'


def voice_of(spec, d):
    """series/episode `voices:` entry for a line: a ref path / 'default', or a dict (ref, exaggeration, cfg_weight,
    temperature, seed, fx, gain_db)."""
    v = spec['voices'].get(d.get('who')) if d.get('who') else None
    if v is None:
        return None
    return {'ref': v} if isinstance(v, str) else dict(v)


def _tts_job(spec, d, v):
    ref = v.get('ref', 'default')
    job = dict(text=d.get('say', d['text']),  # `say:` = spoken wording when it differs from the caption
               voice=ref if ref == 'default' else os.path.abspath(S.resolve(spec, ref)))
    for k in ('exaggeration', 'cfg_weight', 'temperature', 'seed'):
        val = d.get(k, v.get(k))
        if val is not None:  # only pass what's set: the cache keys on the arguments actually given
            job[k] = val
    return job


def dialogue_files(spec):
    """Resolve voice files. Lines with `wav:` use it; lines with only text are spoken by TTS (vf.gen.tts, checked by
    ASR and re-taken when garbled) when the speaker has a voice, else they become timed captions without audio."""
    todo, directed = [], []
    for d in spec['dialogue']:
        if d.get('wav'):
            d['_wav'] = S.resolve(spec, d['wav'])
        elif voice_of(spec, d) is not None and voice_of(spec, d).get('direct'):
            directed.append((d, voice_of(spec, d)))   # vf/voicedir.py: acted takes + strict judge pass
        elif voice_of(spec, d) is not None:
            todo.append(d)
    if directed:
        from .voicedir import direct
        rows = direct(spec, directed, lambda p: S.resolve(spec, p))
        os.makedirs(os.path.join(spec['root'], 'out'), exist_ok=True)
        json.dump(rows, open(os.path.join(spec['root'], 'out', 'voices.json'), 'w'), indent=1)
    if todo:
        from .gen.bridge import run_gen
        res = run_gen('tts', [_tts_job(spec, d, voice_of(spec, d)) for d in todo])
        for d, r in zip(todo, res):
            d['_wav'] = r['out']
            if not r.get('checked', True):
                print(f"  QC warning: line at {d['at']}s still mis-heard after retakes: \"{r.get('heard')}\"")
    # a line that runs into the next one is a script/timing error
    lines = sorted(captions.line_timings(spec))
    for (a0, a1, ta, _), (b0, _, tb, _) in zip(lines, lines[1:]):
        if a1 > b0 + 0.05:
            print(f'  QC warning: line at {a0:.2f}s ends {a1:.2f}s, overlapping the line at {b0:.2f}s: "{tb[:40]}"')
    over = [l for l in lines if l[1] > spec['duration']]
    for l in over:
        print(f'  QC warning: line at {l[0]:.2f}s runs past the end ({l[1]:.2f}s > {spec["duration"]:.2f}s)')


def concat(parts, wav, out):
    lst = tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False)
    try:
        for p in parts:
            lst.write(f"file '{p}'\n")
        lst.close()
        cmd = [ffmpeg(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst.name]
        if wav:
            cmd += ['-i', wav, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '320k', '-shortest']
        cmd += ['-c:v', 'copy', '-movflags', '+faststart', out]
        subprocess.run(cmd, check=True, timeout=600)
    finally:
        lst.close()
        os.unlink(lst.name)


def _record_manifest(spec, results, wav, preview=False):
    """Remember every cache file the latest build of each format read or wrote (vf/gc.py keeps exactly these).
    Keyed per format AND preview/full (a preview build used to overwrite the full build's entry, so `vf clean` then
    deleted the full-resolution segments)."""
    import json
    p = os.path.join(spec['root'], '.vfcache', 'manifest.json')
    m = json.load(open(p)) if os.path.exists(p) else {}
    used = set(memo.TOUCHED)
    for fmt, r in results.items():
        m[fmt + ('_preview' if preview else '')] = sorted(
            used | {os.path.abspath(x) for x in r['parts']} | {os.path.abspath(x) + '.qc.json' for x in r['parts']})
    m['audio'] = [os.path.abspath(wav)] if wav else []
    m['_v'] = 2  # lists every touched cache path (v1 manifests listed only render segments: gc must not trust them)
    tmp = p + '.tmp'
    json.dump(m, open(tmp, 'w'), indent=1)
    os.replace(tmp, p)


def clean(path=None, apply=True):
    """Old `vf clean` (permanent os.remove of project-cache files) now goes through vf/gc.py: every episode's
    manifest is honoured and files go to the Recycle Bin."""
    from .gc import collect
    collect(apply=apply)


BUILDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', 'builds')


def _register_build():
    """cache/builds/<pid>.json while a build runs: vf/gc.py refuses to collect while any OTHER build is live (its new
    files are in no manifest yet)."""
    os.makedirs(BUILDS, exist_ok=True)
    json.dump(dict(pid=os.getpid(), since=time.time()), open(os.path.join(BUILDS, f'{os.getpid()}.json'), 'w'))


def _unregister_build():
    try:
        os.remove(os.path.join(BUILDS, f'{os.getpid()}.json'))
    except OSError:
        pass


def build(path, formats, preview=False, force=False):
    from .proc import job_guard
    _register_build()
    job_guard()  # every child (ffmpeg, gen steps, the gen worker) dies with this build, even on a crash
    try:
        return _build(path, formats, preview, force)
    finally:
        # the gen worker held torch + a CUDA context + 5-7 GB of parked weights in RAM for up to an hour after builds
        try:
            from .gen import bridge
            if bridge.STARTED[0]:  # only the worker this build started (another agent may be using its own)
                bridge.stop_gen()
        except Exception:
            pass
        _unregister_build()


def _build(path, formats, preview=False, force=False):
    t_all = time.time()
    spec = S.load(path)
    memo.set_project(spec['root'])
    fps = 30 if preview else spec['fps']
    scale = 0.5 if preview else 1.0
    from .lint import report as lint_report
    lerrs = lint_report(spec)  # authoring mistakes that cause artifacts: stop before spending GPU time
    if lerrs and not force:
        sys.exit(f'{len(lerrs)} lint error(s): fix the episode (or rebuild with --force)')
    dialogue_files(spec)  # before the AI stage: lip-synced shots need the voice takes
    if any(s.get('gen') for s in spec['shots']):
        from .stages import generate
        t = time.time()
        n = generate(spec, formats)
        print(f'ai stage: {n} shot(s) in {time.time() - t:.1f}s (cached results return instantly)')
        takes = {s['id']: s['gen'].get('_take') for s in spec['shots'] if s.get('gen')}
        os.makedirs(os.path.join(spec['root'], 'out'), exist_ok=True)
        with open(os.path.join(spec['root'], 'out', 'clip_qc.json'), 'w', encoding='utf-8') as f:
            json.dump(takes, f, indent=1, default=str)
    # the timeline as rendered (vf/talk.py moves cut points): `vf check` reads it instead of the yaml durations
    os.makedirs(os.path.join(spec['root'], 'out'), exist_ok=True)
    with open(os.path.join(spec['root'], 'out', 'timeline.json'), 'w', encoding='utf-8') as f:
        json.dump([dict(id=s['id'], start=round(s['start'], 4), dur=round(s['dur'], 4),
                        talk=bool((s.get('gen') or {}).get('talk'))) for s in spec['shots']], f, indent=1)
    spec['_captions'] = captions.build(spec)
    for fmt in formats:  # layout errors (text outside the safe area) stop the build before any rendering
        q = QCmod.QC(spec, fmt)
        q.layout()
        errs = [i for i in q.items if i['severity'] == 'error']
        if errs and not force:
            for it in errs:
                print(f"  QC error: {it['where']}: {it['msg']}")
            sys.exit(f'{len(errs)} layout error(s): fix the episode or rebuild with --force')
    t = time.time()
    a = spec['audio']
    from . import synth
    src = lambda p: synth.resolve(p) if str(p).startswith('synth:') else S.resolve(spec, p)
    beds = a.get('beds') or []  # extra looping layers (ambience) mixed like music but never ducked
    music = dict(a['music'], src=src(a['music']['src'])) if a.get('music') else None
    sfx = [(src(x['src']), float(x.get('at', 0)), float(x.get('gain_db', 0))) for x in a.get('sfx', []) + beds]
    dlg = []
    for d in spec['dialogue']:
        if d.get('_wav'):
            v = voice_of(spec, d) or {}
            dlg.append((d['_wav'], float(d['at']), float(d.get('gain_db', v.get('gain_db', 0))),
                        d.get('fx', v.get('fx'))))
    # `audio: {lufs: -18}` = own loudness target (a music-only intro at the dialogue target of -14 is too loud);
    # omitted = mix default, same cache key as before
    lk = {'target_lufs': float(a['lufs'])} if a.get('lufs') is not None else {}
    wav = audio.mix(spec['duration'], dlg, music, sfx, **lk) if (dlg or music or sfx) else None
    if dlg:   # dialogue-only stem: `vf check` judges open-mouth-without-voice against it (the music bed reads as sound)
        os.makedirs(os.path.join(spec['root'], 'out'), exist_ok=True)
        import shutil   # audio.mix is memo-cached: it picks its own output path and returns it
        shutil.copyfile(audio.mix(spec['duration'], dlg, None, []), os.path.join(spec['root'], 'out', 'dialogue_stem.wav'))
    print(f'audio {time.time() - t:.1f}s')
    outdir = os.path.join(spec['root'], 'out')
    os.makedirs(outdir, exist_ok=True)
    code = renderer_code()
    results = {}
    for fmt in formats:
        q = QCmod.QC(spec, fmt)
        q.layout()
        q.sources()
        t = time.time()
        parts, fresh = [], 0
        from . import ui
        with ui.progress(len(spec['shots']), f'render {fmt}') as step:
            for i, shot in enumerate(spec['shots']):
                before = memo.STATS['miss']
                ts = time.time()
                p = render_segment(shot_key(spec, i, fmt), i, fmt, fps, scale, code, shot_deps(spec, i),
                                   _spec=spec, _qc=q)
                if memo.STATS['miss'] > before:
                    fresh += 1
                    print(f'  {fmt} {shot["id"]}: rendered {shot["dur"]:.1f}s in {time.time() - ts:.1f}s')
                parts.append(p)
                step(note=shot['id'])
        items = q.report(outdir, parts)
        errs = [i for i in items if i['severity'] == 'error']
        for it in items[:12]:
            print(f"  QC {it['severity']}: {it['where']}: {it['msg']}")
        name = f"{slug(spec['title'])}_{fmt}{'_preview' if preview else ''}.mp4"
        out = os.path.join(outdir, name)
        if errs and not force:
            print(f'{fmt}: {len(errs)} QC error(s), final file NOT written: fix them or rebuild with --force')
            out = None
        else:
            concat(parts, wav, out)
            print(f'{fmt}: {fresh}/{len(parts)} shots rendered, {time.time() - t:.1f}s -> {out}')
        results[fmt] = dict(out=out, qc=items, parts=parts)
    _record_manifest(spec, results, wav, preview)
    print(f'total {time.time() - t_all:.1f}s for {spec["duration"]:.1f}s of video '
          f'(cache hits {memo.STATS["hit"]}, misses {memo.STATS["miss"]})')
    if os.environ.get('VF_GC', '1') == '1' and not any(r['out'] is None for r in results.values()):
        from .gc import collect  # regeneration cleans up after itself: superseded takes/finishes -> Recycle Bin
        try:
            collect(apply=True, quiet=True)
        except Exception as e:
            print(f'gc: skipped ({type(e).__name__}: {e})')
    return results


def main(argv=None):
    from . import ui
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        ui.overview()
        return
    passthru = {'status': 'status', 'config': 'config', 'judge': 'judge', 'framecheck': 'framecheck',
                'script-judge': 'scriptjudge'}
    if argv[0] in passthru:   # these parse their own arguments, so `vf judge -h` shows judge's help
        from importlib import import_module
        rc = import_module('.' + passthru[argv[0]], 'vf').main(argv[1:])
        sys.exit(rc if isinstance(rc, int) else 0)
    fmt = ui.help_formatter()
    ap = argparse.ArgumentParser(prog='vf', formatter_class=fmt,
                                 description='videoforge: cached local GPU pipeline for AI series episodes. '
                                             'Run vf with no arguments for the grouped command list.')
    ap.add_argument('--version', action='version', version=f'videoforge {ui.version()}')
    sub = ap.add_subparsers(dest='cmd', required=True, metavar='<command>')
    _add = sub.add_parser
    sub.add_parser = lambda *x, **k: _add(*x, formatter_class=fmt, **k)   # colour help on every subcommand
    st = sub.add_parser('status', help='running and recent jobs, stage progress, GPU (vf/status.py)')
    st.add_argument('rest', nargs=argparse.REMAINDER)
    cf = sub.add_parser('config', help='every VF_* setting: value, source, default; get / set / unset (vf/config.py)')
    cf.add_argument('rest', nargs=argparse.REMAINDER)
    b = sub.add_parser('build', help='render an episode.yaml to one or more formats')
    b.add_argument('episode')
    b.add_argument('--formats', default=None, help='comma list: ' + ','.join(S.FORMATS))
    b.add_argument('--preview', action='store_true')
    b.add_argument('--force', action='store_true')
    b.add_argument('--no-cache', action='store_true')
    b.add_argument('--no-watermark', action='store_true', help='leave out the small hey24sheep.com/videoforge corner mark (please keep it: README Watermark)')
    enc_help = ('allow text encoders to load for new or changed prompts (off by default: a build whose prompt '
                'embeddings are not all cached stops before anything heavy loads; vf/gen/guard.py)')
    b.add_argument('--allow-encode', action='store_true', help=enc_help)
    sub.add_parser('formats', help='output formats')
    sub.add_parser('styles', help='visual styles')
    c = sub.add_parser('clean', help='same as gc --apply (kept for old scripts)')
    c.add_argument('episode', nargs='?')
    g = sub.add_parser('gc', help='list (or --apply: move to the Recycle Bin) cache files no latest build uses')
    g.add_argument('--apply', action='store_true')
    k = sub.add_parser('keyframes', help='generate only the first frames of AI shots + a review contact sheet')
    k.add_argument('episode')
    k.add_argument('--formats', default=None)
    k.add_argument('--allow-encode', action='store_true', help=enc_help)
    v = sub.add_parser('voices', help='generate (and ASR-check) only the dialogue lines; print their timings')
    v.add_argument('episode')
    li = sub.add_parser('lint', help='check an episode for authoring mistakes that cause artifacts (no GPU)')
    li.add_argument('episode')
    ck = sub.add_parser('check', help='post-render gates: brightness dips, frozen frames, SyncNet lip-sync offset')
    ck.add_argument('episode')
    ck.add_argument('--video', default=None, help='default: newest non-phone mp4 under <episode>/out')
    ck.add_argument('--no-sync', action='store_true', help='skip the SyncNet model run (no GPU)')
    ru = sub.add_parser('run', help='whole path: gen -> lint -> voices -> keyframes -> cut (vf/runner.py)')
    ru.add_argument('target', help='<ep>_gen.py or episode.yaml')
    ru.add_argument('--until', default='framecheck', choices=['gen', 'lint', 'voices', 'keyframes', 'cut', 'framecheck'])
    ru.add_argument('--preview', action='store_true')
    ru.add_argument('--detach', action='store_true', help='run under tools/jobwatch.py (survives the session)')
    ru.add_argument('--no-watermark', action='store_true', help='leave out the small hey24sheep.com/videoforge corner mark (please keep it: README Watermark)')
    sub.add_parser('doctor', help='check interpreters, venvs, GPU, NVENC, ffmpeg, models, disk')
    nw = sub.add_parser('new', help='scaffold series/<series>/<ep>/<ep>_gen.py')
    nw.add_argument('series')
    nw.add_argument('ep')
    nw.add_argument('--no', type=int, required=True)
    nw.add_argument('--title', required=True)
    cu = sub.add_parser('cut', help='release cut: series intro (episode name filled in) + episode + outro, check, join, phone copy')
    cu.add_argument('episode')
    cu.add_argument('--skip-build', action='store_true', help='join the newest renders on disk')
    cu.add_argument('--no-check', action='store_true')
    cu.add_argument('--phone-mb', type=float, default=28.0)
    cu.add_argument('--no-watermark', action='store_true', help='leave out the small hey24sheep.com/videoforge corner mark (please keep it: README Watermark)')
    sq = sub.add_parser('script-qa', help='score script drafts with the local model rubric (vf/gen/scriptqa.py)')
    sq.add_argument('scripts', nargs='+')
    sq.add_argument('--bible', required=True, help='series bible text file (series/<name>/bible.md)')
    sq.add_argument('--ref', default=None, help='an approved script scored alongside: drafts must reach its mean')
    sq.add_argument('--json', default=None)
    sj = sub.add_parser('script-judge', help='judging v2: persona prompts + score table (vf/scriptjudge.py)')
    sj.add_argument('rest', nargs=argparse.REMAINDER)
    jd = sub.add_parser('judge', help='every judge in one place: script, dialogue, music, clip, image, video, critics (vf/judge.py)')
    jd.add_argument('rest', nargs=argparse.REMAINDER)
    fc = sub.add_parser('framecheck', help='learned frame checker: defect frames, predicted shot scores, fix plan (vf/framecheck.py)')
    fc.add_argument('rest', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if getattr(a, 'no_watermark', False):
        os.environ['VF_WATERMARK'] = '0'   # read by vf/render.py; inherited by every child build
    if a.cmd == 'status':
        from .status import main as st_main
        sys.exit(st_main(a.rest))
    if a.cmd == 'config':
        from .config import main as cf_main
        sys.exit(cf_main(a.rest))
    if a.cmd == 'judge':
        from .judge import main as j_main
        return j_main(a.rest)
    if a.cmd == 'framecheck':
        from .framecheck import main as fc_main
        return fc_main(a.rest)
    if a.cmd == 'script-judge':
        from .scriptjudge import main as sj_main
        return sj_main(a.rest)
    if getattr(a, 'allow_encode', False):
        os.environ['VF_ALLOW_ENCODE'] = '1'  # read by vf/gen/guard.py in every gen child and the gen worker
    if a.cmd == 'check':
        from .checks import check
        sys.exit(0 if check(a.episode, a.video, not a.no_sync)['pass_'] else 1)
    if a.cmd in ('run', 'doctor', 'new'):
        from . import runner
        if a.cmd == 'run':
            sys.exit(runner.run(a.target, a.until, a.preview, a.detach))
        if a.cmd == 'doctor':
            sys.exit(runner.doctor(log=ui.doctor_line))
        sys.exit(runner.new(a.series, a.ep, a.no, a.title))
    if a.cmd == 'cut':
        from .cut import cut
        r = cut(a.episode, a.skip_build, a.no_check, a.phone_mb)
        sys.exit(0 if r['ok'] else 1)
    if a.cmd == 'script-qa':
        from .checks import script_qa
        script_qa(a.scripts, a.bible, a.ref, out_json=a.json)
        return
    if a.cmd == 'lint':
        from .lint import lint
        items = lint(S.load(a.episode))
        for lvl, where, msg in items:
            print(f'[{lvl}] {where}: {msg}')
        print(f"{sum(i[0] == 'error' for i in items)} error(s), {sum(i[0] == 'warn' for i in items)} warning(s)")
        return
    if a.cmd == 'keyframes':
        from .stages import keyframes, contact_sheet
        if not a.formats:
            sys.exit('which formats? pass --formats (clip aspect follows it)')
        spec = S.load(a.episode)
        got = keyframes(spec, a.formats.split(','))
        out = os.path.join(spec['root'], 'out', 'keyframes_sheet.jpg')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        contact_sheet([(s['id'], p) for s, p in got], out)
        for s, p in got:
            print(f"{s['id']:14s} {p}")
        print('sheet:', out)
        return
    if a.cmd == 'voices':
        spec = S.load(a.episode)
        dialogue_files(spec)
        for (t0, t1, text, who) in captions.line_timings(spec):
            print(f'{t0:6.2f}-{t1:6.2f} ({t1 - t0:4.2f}s) {who}: {text}')
        return
    if a.cmd == 'clean':
        clean(a.episode)
        return
    if a.cmd == 'gc':
        from .gc import collect
        collect(apply=a.apply)
        return
    if a.cmd == 'styles':
        from .styles import listing
        for k, name, risk in listing():
            print(f'{k:12s} {name}   [8 GB artifact risk: {risk}]')
        return
    if a.cmd == 'formats':
        for k, v in S.FORMATS.items():
            print(f'{k:6s} {v["W"]}x{v["H"]}  {v["label"]}')
        return
    if a.no_cache:
        os.environ['VF_NO_CACHE'] = '1'
    if not a.formats:
        sys.exit('which formats? pass --formats (e.g. yt,short). Options: ' + ', '.join(S.FORMATS))
    build(a.episode, a.formats.split(','), a.preview, a.force)


if __name__ == '__main__':
    main()
