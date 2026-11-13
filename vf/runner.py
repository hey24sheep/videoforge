"""One-command tools for the whole package (2026-11-09, owner: "videoforge as a complete package and run tool").

  vf run <ep_gen.py | episode.yaml> [--until STAGE] [--preview] [--detach]
      the full path from script to release, each stage a subprocess, stops at the first failure:
        gen       run the episode generator (if a *_gen.py is given) -> episode.yaml
        lint      vf lint (errors stop the run)
        voices    vf voices (directed voices, out/voices.json); then gen again so shots follow the real takes
        keyframes vf keyframes --formats short (+ out/keyframes_sheet.jpg)
        cut       vf cut: intro + episode + outro, vf check, join, phone copy (or a --preview build)
        framecheck  the learned frame checker (vf/framecheck.py) on the new build: out/framecheck_fixes.md. It
                  never stops the run (skipped with a note when no model is trained yet).
      --until keyframes stops for a keyframe review; --detach starts the same run under tools/jobwatch.py (survives
      the session) and prints the job id to watch.
  vf doctor   checks interpreters, venvs, GPU/VRAM, NVENC, ffmpeg, model folders, fonts, free disk
  vf new <series> <ep> --no N --title "Title"   scaffolds series/<series>/<ep>/<ep>_gen.py on vf/autolayout.py
"""
import os, shutil, subprocess, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ['gen', 'lint', 'voices', 'keyframes', 'cut', 'framecheck']


def _py():
    return sys.executable


def _sh(args, log):
    log('$ ' + ' '.join(args))
    t = time.time()
    r = subprocess.run(args, cwd=ROOT)
    log(f'  -> exit {r.returncode} in {time.time() - t:.0f}s')
    return r.returncode


def run(target, until='framecheck', preview=False, detach=False, log=print):
    target = os.path.abspath(target)
    if detach:
        name = 'run_' + os.path.basename(os.path.dirname(target))
        args = [_py(), os.path.join(ROOT, 'tools', 'jobwatch.py'), 'run', '--name', name, '--env', 'VF_GC=0',
                '--env', 'PYTHONIOENCODING=utf-8', '--', 'python', '-m', 'vf', 'run', target, '--until', until] + \
               (['--preview'] if preview else [])
        return subprocess.run(args, cwd=ROOT).returncode
    gen = target if target.endswith('.py') else None
    ep = os.path.join(os.path.dirname(target), 'episode.yaml') if gen else target
    if preview and gen:
        ep = os.path.join(os.path.dirname(target), 'episode_preview.yaml')
    stop = STAGES.index(until)

    def regen():
        return _sh([_py(), gen] + (['--preview'] if preview else []), log) if gen else 0

    steps = [
        ('gen', regen),
        ('lint', lambda: _lint(ep, log)),
        ('voices', lambda: _sh([_py(), '-m', 'vf', 'voices', ep], log) or regen()),
        ('keyframes', lambda: _sh([_py(), '-m', 'vf', 'keyframes', ep, '--formats', 'short', '--allow-encode'], log)),
        ('cut', lambda: _sh([_py(), '-m', 'vf', 'build', ep, '--formats', 'short', '--preview', '--allow-encode'], log)
         if preview else _sh([_py(), '-m', 'vf', 'cut', ep], log)),
        ('framecheck', lambda: _framecheck(ep, log) if not preview else 0),
    ]
    for i, (name, fn) in enumerate(steps):
        if i > stop:
            break
        from . import ui
        if ui.live() and log is print:   # terminal: a coloured rule; logs keep the line tools/jobwatch.py and vf status read
            ui.CON.rule(f'[bold magenta][{i + 1}/{stop + 1}] {name}[/]', align='left', style='magenta')
        else:
            log(f'=== [{i + 1}/{stop + 1}] {name}')
        rc = fn()
        if rc:
            log(f'=== run stopped at {name} (exit {rc})')
            return rc
    log('=== run done' + (f' (stopped after {until} for review)' if until not in ('cut', 'framecheck') else ''))
    return 0


def _framecheck(ep, log):
    """Warn-only stage: a missing model or a failure is logged, never an exit code."""
    from . import framecheck
    if not os.path.exists(framecheck._p('model.pkl')):
        log('  framecheck: no trained model yet (vf framecheck learn + train); skipped')
        return 0
    try:
        framecheck.check(ep, log=log)
    except Exception as e:   # the checker must never cost a finished build
        log(f'  framecheck failed (ignored): {type(e).__name__}: {e}')
    return 0


def _lint(ep, log):
    from . import spec as S
    from .lint import lint
    items = lint(S.load(ep))
    for lvl, where, msg in items:
        log(f'  [{lvl}] {where}: {msg}')
    errs = sum(i[0] == 'error' for i in items)
    log(f'  lint: {errs} error(s), {len(items) - errs} warning(s)')
    return 1 if errs else 0


# ------------------------------------------------------------------ doctor
def doctor(log=print):
    ok = True

    def item(good, what, detail=''):
        nonlocal ok
        ok &= bool(good)
        log(f"  {'ok  ' if good else 'FAIL'} {what}" + (f'  ({detail})' if detail else ''))

    log('videoforge doctor')
    item(sys.version_info >= (3, 11), 'render python', sys.version.split()[0] + ' ' + sys.executable)
    for mod in ('cupy', 'cv2', 'yaml', 'scipy', 'PIL', 'numpy'):
        try:
            __import__(mod)
            item(True, f'import {mod}')
        except Exception as e:
            item(False, f'import {mod}', str(e)[:80])
    try:
        from .gen.bridge import GEN_PY
    except Exception:
        GEN_PY = None
    from .checks import TALK_PY
    venvs = dict(gen=GEN_PY, talk=TALK_PY,
                 tts=os.path.join(_AI, 'tools', 'envs', 'tts', 'Scripts', 'python.exe'),
                 music=os.path.join(_AI, 'tools', 'envs', 'music', 'Scripts', 'python.exe'))
    for k, p in venvs.items():
        item(p and os.path.exists(p), f'{k} venv', p)
    for exe in ('ffmpeg', 'ffprobe', 'nvidia-smi'):
        item(shutil.which(exe), exe, shutil.which(exe) or 'not on PATH')
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,temperature.gpu',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=20).stdout.strip()
        item(bool(q), 'GPU', q)
    except Exception as e:
        item(False, 'GPU', str(e)[:80])
    try:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=256x256:d=0.2', '-c:v',
                            'hevc_nvenc', '-f', 'null', '-'], capture_output=True, text=True, timeout=60)
        item(r.returncode == 0, 'NVENC (hevc)', (r.stderr or '').strip()[:80])
    except Exception as e:
        item(False, 'NVENC', str(e)[:80])
    from . import enc
    log(f'  info build/cut encoder: {enc.name()}  (VF_ENCODER={os.environ.get("VF_ENCODER", "auto")})')
    models = os.path.join(_AI, 'models', 'hf')
    item(os.path.isdir(models), 'model folder', models)
    for sub in ('flux2-klein-4b', 'ace-step-1.5'):
        item(os.path.isdir(os.path.join(models, sub)), f'model {sub}')
    item(os.path.exists(r'C:\Windows\Fonts\Inkfree.ttf'), 'ink font (Inkfree.ttf)')
    free = shutil.disk_usage(ROOT).free / 2 ** 30
    item(free > 40, 'free disk', f'{free:.0f} GB on {os.path.splitdrive(ROOT)[0]}')
    jw = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'jobwatch.py'), 'check'], capture_output=True,
                        text=True, encoding='utf-8', errors='replace')
    running = [l for l in jw.stdout.splitlines() if l.startswith('[running]')]
    log(f"  info {len(running)} job(s) running" + (': ' + '; '.join(running) if running else ''))
    log('doctor: ' + ('all ok' if ok else 'PROBLEMS above'))
    return 0 if ok else 1


# ------------------------------------------------------------------ new episode scaffold
TEMPLATE = '''"""{series} {ep}: "{title}". episode.yaml generator (vf/autolayout.py); edit this file, not episode.yaml.
python {ep}_gen.py [--preview]   (or: vf run series/{series}/{ep}/{ep}_gen.py)
"""
import argparse, os, sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
from vf.autolayout import L, insert, talk_shot, layout, voice_durs  # noqa: E402

# who: keys of series.yaml voices; directions are acting notes for vf/voicedir.py
BEATS = [
    insert('opening',
           "A wide establishing view ..., nobody in frame.",       # keyframe prompt (props/states live here)
           "Slow ambient motion only. No people.",                 # motion prompt (no camera words, no mouths)
           lines=[L('narrator', "First line.", "Inner voice, low and slow.")], mindur=3.0),
    # talk_shot('speaker_cu', 'narrator', 'detective', "{{detective}}, close-up ..., lips closed.", "Line said on screen.",
    #           "Acting note.", "Close-up ... speaking quietly, his lips moving as he talks.", style=None),
]


def build(durs, preview):
    shots, dialogue, total, starts = layout(BEATS, durs, preview=preview)
    ep = dict(title="{title}", episode=dict(no={no}, title="{title}"), gen=dict(profile='k6', retries=1),
              audio=dict(), voices={{}}, dialogue=dialogue, shots=shots)
    return ep, total


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    ep, total = build(voice_durs(os.path.join(HERE, 'out', 'voices.json')), a.preview)
    out = os.path.join(HERE, 'episode_preview.yaml' if a.preview else 'episode.yaml')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# GENERATED by {ep}_gen.py. Edit that file.\\n')
        yaml.safe_dump(ep, f, sort_keys=False, allow_unicode=True, width=200)
    print(out, 'shots', len(ep['shots']), 'lines', len(ep['dialogue']), 'total %.1fs' % total)
'''


def new(series, ep, no, title, log=print):
    d = os.path.join(ROOT, 'series', series, ep)
    p = os.path.join(d, f'{ep}_gen.py')
    if os.path.exists(p):
        log(f'exists: {p}')
        return 1
    os.makedirs(d, exist_ok=True)
    open(p, 'w', encoding='utf-8').write(TEMPLATE.format(series=series, ep=ep, title=title, no=no))
    log(f'wrote {p}\nnext: write the beats, then  vf run {os.path.relpath(p, ROOT)} --until keyframes')
    return 0
