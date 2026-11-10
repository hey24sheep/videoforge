"""`vf judge <kind> ...`: one front door to every judge in the repo. Each judge is an open model run locally, or a
blind Claude critic whose prompt this module writes and whose answers it averages. Nothing here is story-specific.

  vf judge script   a.md b.md [--context ctx.md] [--ref approved.md]   rubric model, 12 yes/no questions per draft
  vf judge dialogue take1.wav ... --text "the line" [--direction "quiet, tired"] [--who name]
                    words heard + word error rate, same voice across takes; acting judge when the judge venv exists
  vf judge music    a.wav b.wav ... --target "what it should sound like" [--avoid "..."] [--key]
                    text-audio match (CLAP), loudness per second; --key adds key / bpm / caption (ACE-Step LM, GPU)
  vf judge clip     shot.mp4 [--q "Is there a person?=no" ...] [--frames 8]   VLM yes/no answers + description
  vf judge image    still.png [--q ...]                                      same VLM on one picture
  vf judge video    episode.yaml [--video cut.mp4]                           post-render gates (`vf check`)
  vf judge critics  script|dialogue|video --entry NAME=path ... --context ctx.md --out DIR
                    blind critic prompts: entries relabelled A, B, C..., a different order per critic (Opus, Sonnet;
                    never Fable). The session runs one fresh subagent per DIR/<critic>.prompt.txt and saves its reply
                    as DIR/<critic>.answer.md.
  vf judge table    DIR                                                      per-entry mean over critics and criteria

Rules every judge follows (docs/review_workflow.md): no leading questions, blind labels, both orders where a model
compares, and scores derived from criteria, never typed as one number.
"""
import argparse, functools, json, os, random, re, string, subprocess, sys, tempfile

print = functools.partial(print, flush=True)   # headers before the child tools' output
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)


def _venv(name):
    return os.path.join(_AI, 'tools', 'envs', name, 'Scripts', 'python.exe')


def _run(py, args):
    if not os.path.exists(py):
        raise SystemExit(f'missing venv python: {py} (README: Setup step 3)')
    r = subprocess.run([py] + args, cwd=ROOT)
    if r.returncode:
        raise SystemExit(r.returncode)


# ------------------------------------------------------------------ local model judges
def script(a):
    from .checks import script_qa
    ctx = a.context
    if not ctx:
        fd, ctx = tempfile.mkstemp(suffix='.md')
        os.write(fd, b'No series bible: judge each script on its own.')
        os.close(fd)
    script_qa(a.files, ctx, a.ref, out_json=a.json)


def dialogue(a):
    jobs = [dict(wav=os.path.abspath(w), text=a.text, who=a.who) for w in a.files]
    fd, jf = tempfile.mkstemp(suffix='.json')
    os.write(fd, json.dumps(jobs).encode())
    os.close(fd)
    print('== words heard + same voice (faster-whisper, ECAPA)')
    _run(_venv('tts'), ['tools/voice_qc.py', '--jobs', jf] + (['--out', a.json] if a.json else []))
    if os.path.exists(_venv('judge')):
        print('== acting (Step-Audio-2-mini, blind passes)')
        args = ['tools/judge_voice.py', '--text', a.text, '--direction', a.direction or 'natural, believable']
        for w in a.files:
            args += ['--wav', os.path.abspath(w)]
        _run(_venv('judge'), args)
    else:
        print('(acting judge skipped: no judge venv at', _venv('judge') + ')')


def music(a):
    import numpy as np
    import soundfile as sf
    print('== text-audio match (CLAP; higher = closer to --target, minus --avoid)')
    args = ['tools/clap_rank.py', '--target', a.target]
    for v in a.avoid or []:
        args += ['--avoid', v]
    _run(_venv('gen'), args + [os.path.abspath(w) for w in a.files])
    print('== loudness per second (dBFS RMS): where it builds, peaks and ends')
    for w in a.files:
        x, sr = sf.read(w)
        x = x.mean(1) if x.ndim > 1 else x
        r = [20 * np.log10(np.sqrt(np.mean(x[i * sr:(i + 1) * sr] ** 2)) + 1e-9) for i in range(int(len(x) / sr))]
        print(f'{os.path.basename(w)}: ' + ' '.join(f'{v:.0f}' for v in r))
    if a.key:
        print('== key / bpm / caption (ACE-Step 5 Hz LM)')
        _run(_venv('music'), ['tools/music_understand.py'] + [os.path.abspath(w) for w in a.files])


def _vlm(clips, questions, frames):
    from .gen.bridge import run_gen
    qs = []
    for q in questions or []:
        text, _, exp = q.rpartition('=') if re.search(r'=(yes|no)$', q) else (q, '', '')
        qs.append(dict(q=text, expect=exp) if exp else dict(q=text))
    jobs = [dict(clip=os.path.abspath(c), questions=qs, frames=frames, describe=True) for c in clips]
    for r in run_gen('judge', jobs, persistent=False, check=False):
        print(json.dumps(r, ensure_ascii=False))


def clip(a):
    _vlm(a.files, a.q, a.frames)


def image(a):
    clips = []
    for p in a.files:   # the clip judge samples frames: a 1 s still clip shows it the picture
        out = os.path.join(tempfile.gettempdir(), 'vfjudge_' + os.path.splitext(os.path.basename(p))[0] + '.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-i', p, '-t', '1', '-r', '8', '-vf',
                        'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p', out], check=True)
        clips.append(out)
    _vlm(clips, a.q, 1)


def video(a):
    args = [sys.executable, '-m', 'vf', 'check', a.files[0]] + (['--video', a.video] if a.video else [])
    raise SystemExit(subprocess.run(args, cwd=ROOT).returncode)


# ------------------------------------------------------------------ blind Claude critics
CRITICS = {'opus': 'opus', 'sonnet': 'sonnet'}   # never Fable


def critics(a):
    tpl = open(os.path.join(ROOT, 'prompts', 'judges', f'{a.kind}.md'), encoding='utf-8').read()
    ctx = open(a.context, encoding='utf-8').read() if a.context else '(none given)'
    names = [e.split('=', 1)[0] for e in a.entry]
    paths = {e.split('=', 1)[0]: e.split('=', 1)[1] for e in a.entry}
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    key = {}
    for critic in CRITICS:
        order = names[:]
        rng.shuffle(order)
        while len(order) > 1 and any(order == list(k.values()) for k in key.values()):
            rng.shuffle(order)   # every critic reads the entries in a different order
        labels = list(string.ascii_uppercase[:len(order)])
        key[critic] = dict(zip(labels, order))
        parts = []
        for lab, n in zip(labels, order):
            p = paths[n]
            if a.kind == 'video':   # Claude reads the frame sheet itself
                body = os.path.abspath(p)
            else:
                body = open(p, encoding='utf-8').read().strip()
            parts.append(f'=== {lab} ===\n{body}')
        prompt = tpl   # plain replace: the rubric's JSON example has braces
        for k, v in dict(n=str(len(order)), labels=', '.join(labels), context=ctx, entries='\n\n'.join(parts)).items():
            prompt = prompt.replace('{' + k + '}', v)
        open(os.path.join(a.out, f'{critic}.prompt.txt'), 'w', encoding='utf-8').write(prompt)
    json.dump(dict(key=key, models=CRITICS, kind=a.kind), open(os.path.join(a.out, 'key.json'), 'w'), indent=1)
    print(f'wrote {len(CRITICS)} prompts to {a.out}: run one fresh subagent per prompt ({", ".join(CRITICS.values())}),'
          f' save each reply as <critic>.answer.md, then `vf judge table {a.out}`')


def table(a):
    meta = json.load(open(os.path.join(a.dir, 'key.json')))
    per = {}
    for critic, key in meta['key'].items():
        f = os.path.join(a.dir, f'{critic}.answer.md')
        if not os.path.exists(f):
            print(f'missing {f}')
            continue
        m = re.findall(r'```json\s*(\{.*?\})\s*```', open(f, encoding='utf-8').read(), re.S)
        if not m:
            print(f'{f}: no json block')
            continue
        ans = json.loads(m[-1])
        for lab, sc in ans.get('scores', {}).items():
            name = key.get(lab, lab)
            vals = [float(v) for v in sc.values()]
            per.setdefault(name, {})[critic] = (sum(vals) / len(vals), min(vals))
    rows = sorted(per.items(), key=lambda kv: -sum(v[0] for v in kv[1].values()) / len(kv[1]))
    print(f'{"entry":24s} ' + ' '.join(f'{c:>12s}' for c in meta['key']) + '      mean   lowest criterion')
    for name, d in rows:
        cells = ' '.join(f'{d[c][0]:12.2f}' if c in d else f'{"-":>12s}' for c in meta['key'])
        mean = sum(v[0] for v in d.values()) / len(d)
        print(f'{name:24s} {cells} {mean:9.2f} {min(v[1] for v in d.values()):10.0f}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='vf judge', description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='kind', required=True)
    s = sub.add_parser('script'); s.add_argument('files', nargs='+'); s.add_argument('--context'); s.add_argument('--ref')
    s.add_argument('--json')
    d = sub.add_parser('dialogue'); d.add_argument('files', nargs='+'); d.add_argument('--text', required=True)
    d.add_argument('--direction'); d.add_argument('--who', default='speaker'); d.add_argument('--json')
    m = sub.add_parser('music'); m.add_argument('files', nargs='+'); m.add_argument('--target', required=True)
    m.add_argument('--avoid', action='append'); m.add_argument('--key', action='store_true')
    for k in ('clip', 'image'):
        c = sub.add_parser(k); c.add_argument('files', nargs='+')
        c.add_argument('--q', action='append', help='question, optionally ending =yes or =no (the expected answer)')
        c.add_argument('--frames', type=int, default=8)
    v = sub.add_parser('video'); v.add_argument('files', nargs=1); v.add_argument('--video')
    cr = sub.add_parser('critics'); cr.add_argument('kind_', choices=['script', 'dialogue', 'video'])
    cr.add_argument('--entry', action='append', required=True, help='NAME=path (text file, or a frame sheet image)')
    cr.add_argument('--context'); cr.add_argument('--out', required=True); cr.add_argument('--seed', type=int, default=0)
    t = sub.add_parser('table'); t.add_argument('dir')
    a = ap.parse_args(argv)
    if a.kind == 'critics':
        a.kind = a.kind_
        return critics(a)
    dict(script=script, dialogue=dialogue, music=music, clip=clip, image=image, video=video, table=table)[a.kind](a)


if __name__ == '__main__':
    main()
