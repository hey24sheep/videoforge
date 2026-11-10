"""Directed voices with a strict pass (built 2026-11-07 after the owner rejected flat, over-dramatic and inconsistent
the first series' voices). Every line is acted several times, every take is measured by open models, and only a take that
passes every gate is used.

A voice opts in with `direct:` in its `voices:` entry (series or episode):

  narrator:
    direct:
      anchor: voices/hale_tired.wav          # the character's voice: every take must sound like it (ECAPA >= 0.6)
      character: "Male, late forties, tired homicide detective; low, quiet, restrained."   # judge + design prompt
      banned: [excited, happy, surprise, joy, panic]   # emotions the judge must NOT hear (blind, pass 1)
      takes:                                  # each entry x every seed = one take
        - {engine: cbox, exaggeration: 0.35, cfg_weight: 0.45}        # Chatterbox (MIT) cloning `ref` (default anchor)
        - {engine: xvec, xvec_ref: voices/walt_xvec.wav}              # Qwen3-TTS VoiceDesign + speaker embedding
        - {engine: qclone}                                           # Qwen3-TTS Base clone of `ref`
      seeds: [1, 2, 3]
  dialogue line: `direction:` (how to act it; the judge scores the take against it blind) and `emotion: [fear, ...]`
  (optional: the judge's emotion word must be one of these).

Gates per take (all must hold): words exact (faster-whisper, WER 0), naturalness >= 9.3 (Step-Audio-2-mini judge),
emotion not banned (and in `emotion` if given), same voice as the anchor >= 0.6. Pick = passing take with the highest
judge score; if none pass, the take with the fewest failed gates is used and the line is FLAGGED in out/voices.json.

Models: Qwen3-TTS 1.7B (Apache-2.0, voice venv), Chatterbox (MIT, tts venv via the gen bridge), faster-whisper small.en
(MIT) + SpeechBrain ECAPA (Apache-2.0) in tools/voice_qc.py, Step-Audio-2-mini (Apache-2.0, 4-bit) in tools/judge_voice.py.
Cached per line in cache/voicedir/<key>.wav (+ .json with the scores of every take).
"""
import hashlib, json, os, shutil, subprocess, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = os.path.join(_AI, 'tools', 'envs')
PY = {'voice': os.path.join(ENVS, 'voice', 'Scripts', 'python.exe'),
      'tts': os.path.join(ENVS, 'tts', 'Scripts', 'python.exe'),
      'judge': os.path.join(ENVS, 'judge', 'Scripts', 'python.exe')}
CACHE = os.path.join(ROOT, 'cache', 'voicedir')
VERSION = 'voicedir|v1'
GATES = dict(naturalness=9.3, anchor=0.6, wer=0.0)
QWEN_MODE = {'xvec': 'design_xvec', 'qclone': 'clone', 'design': 'design', 'qclone_instruct': 'clone_instruct'}


def _fh(p):
    st = os.stat(p)
    return f'{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}'


def _key(*parts):
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _run(args, log, timeout=3600):
    """One stage in its own venv/process (the models never share a process: switching Qwen modes in one run OOMed)."""
    from .gpulock import gpu_lock
    from .proc import run
    with gpu_lock('vf voicedir ' + os.path.basename(args[1])):
        run(args, timeout=timeout, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8'))


def _ref_text(path):
    j = path + '.txt'
    return open(j, encoding='utf-8').read().strip() if os.path.exists(j) else None


def plan(spec, lines, resolve):
    """lines: [(dialogue dict, voice dict)] -> [(d, key, takes)] with takes = list of job dicts (not yet made)."""
    out = []
    for d, v in lines:
        dv = dict(v['direct'], **(d.get('direct') or {}))   # a line may override seeds/takes (more tries for a hard line)
        anchor = resolve(dv['anchor'])
        ref = resolve(dv.get('ref') or v.get('ref') or dv['anchor'])
        text = d.get('say', d['text'])
        direction = d.get('direction') or dv.get('direction') or ''
        takes = []
        for t in dv.get('takes') or [{'engine': 'cbox'}]:
            for seed in dv.get('seeds') or [1, 2, 3]:
                tj = dict(t, seed=int(seed), text=text, direction=direction)
                if t['engine'] == 'cbox':
                    tj['voice'] = os.path.abspath(resolve(t.get('ref') or ref))
                elif t['engine'] in QWEN_MODE:
                    tj['mode'] = QWEN_MODE[t['engine']]
                    tj['character_voice'] = dv.get('character_voice', '') if t['engine'] in ('xvec', 'design') else ''
                    r = resolve(t.get('xvec_ref') or t.get('ref') or ref)
                    tj['ref_wav'] = os.path.abspath(r)
                    if t['engine'] == 'qclone':
                        tj['ref_text'] = t.get('ref_text') or _ref_text(r)
                else:
                    raise ValueError(f"voices.{d.get('who')}.direct: unknown engine {t['engine']}")
                takes.append(tj)
        key = _key(VERSION, text, direction, _fh(anchor), sorted(d.get('emotion') or []),
                   sorted(dv.get('banned') or []), dv.get('character', ''),
                   [{k: (_fh(x) if k in ('voice', 'ref_wav') else x) for k, x in tj.items()} for tj in takes])
        out.append((d, key, takes, dict(anchor=os.path.abspath(anchor), banned=[b.lower() for b in dv.get('banned') or []],
                                          avoid=[a.lower() for a in (dv.get('avoid') or []) + (d.get('avoid') or [])],
                                          emotion=[e.lower() for e in d.get('emotion') or []],
                                          character=dv.get('character', ''), direction=direction, text=text)))
    return out


_ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '         'seventeen eighteen nineteen'.split()
_TENS = 'x x twenty thirty forty fifty sixty seventy eighty ninety'.split()


def _n2w(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ('' if n % 10 == 0 else _ONES[n % 10])
    if n < 10000:   # clock / badge style: 320 -> three twenty, 1205 -> twelve oh five
        hi, lo = divmod(n, 100)
        return _n2w(hi) + ('hundred' if lo == 0 else ('oh' + _ONES[lo] if lo < 10 else _n2w(lo)))
    return str(n)


def _compact(t):
    """Spelling-independent form: digits as words, no spaces/punctuation ('Tideboard ... 320' == 'Tide board ... three-twenty')."""
    import re
    t = re.sub(r'\d+', lambda m: _n2w(int(m.group(0))), str(t).lower())
    t = t.replace("'s", 's').replace('kohl', 'cole')
    return re.sub(r'[^a-z]', '', t)


def _fails(r, meta):
    """Gate a take from its stored measurements (re-run on every build: changing `avoid` re-picks, never re-acts)."""
    emo = str(r.get('emotion', '')).lower().strip('. ')
    no = meta['banned'] + meta.get('avoid', [])
    fails = []
    words_ok = r.get('wer') is not None and r['wer'] <= GATES['wer']
    if not words_ok and r.get('heard') and _compact(r['heard']) == _compact(meta['text']):
        words_ok = True
    if not words_ok:
        fails.append(f"words (heard '{r.get('heard')}')")
    if (r.get('naturalness') or 0) < GATES['naturalness']:
        fails.append(f"naturalness {r.get('naturalness')}")
    if emo in no or any(b in emo for b in no):
        fails.append(f'emotion {emo} (not wanted)')
    if meta['emotion'] and not any(e in emo for e in meta['emotion']):
        fails.append(f"emotion {emo} not in {meta['emotion']}")
    if (r.get('anchor') or 0) < GATES['anchor']:
        fails.append(f"voice match {r.get('anchor')}")
    return fails


def _rank(r):
    """Fewest failed gates, then how well it plays the written direction + how human it sounds. NOT intensity: ranking
    on intensity picked the most dramatic take every time (2026-11-07, the narrator came out angry/urgent on quiet lines)."""
    # a wrong voice is the worst failure (a stranger speaking), then wrong words, then a wrong emotion / naturalness
    w = sum(3 if f.startswith('voice match') else 2 if f.startswith('words') else 1 for f in r['fails'])
    return (w, -((r.get('matches_direction') or 0) + (r.get('naturalness') or 0)))


def direct(spec, lines, resolve, log=print):
    """Make, measure and pick a take for every directed line. Sets d['_wav']; returns the report rows."""
    os.makedirs(os.path.join(CACHE, 'takes'), exist_ok=True)
    P = plan(spec, lines, resolve)
    todo = [(d, k, takes, meta) for d, k, takes, meta in P if not os.path.exists(os.path.join(CACHE, k + '.json'))]
    rows = []
    if todo:
        log(f'  voicedir: {len(todo)} line(s), {sum(len(t) for _, _, t, _ in todo)} take(s) to act and judge')
        work = os.path.join(CACHE, 'work_' + time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(work)
        qwen, cbox, judge = [], [], []
        for d, k, takes, meta in todo:
            for i, tj in enumerate(takes):
                wav = os.path.join(CACHE, 'takes', f'{k}_{i:02d}.wav')
                tj['out'] = wav
                if os.path.exists(wav):
                    pass
                elif tj['engine'] == 'cbox':
                    cbox.append(tj)
                else:
                    qwen.append({x: tj[x] for x in ('text', 'character_voice', 'direction', 'seed', 'mode', 'ref_wav',
                                                    'ref_text', 'out') if tj.get(x) is not None})
                judge.append(dict(wav=wav, text=tj['text'], direction=meta['direction'] or meta['character'],
                                  who=d.get('who'), anchor=meta['anchor']))
        t0 = time.time()
        for mode in sorted({q['mode'] for q in qwen}):   # one Qwen model per process
            jf = os.path.join(work, f'qwen_{mode}.json')
            json.dump([q for q in qwen if q['mode'] == mode], open(jf, 'w'), indent=1)
            _run([PY['voice'], os.path.join(ROOT, 'tools', 'act_voice.py'), '--jobs', jf, '--skip-existing'], log)
        if cbox:
            from .gen.bridge import run_gen, stop_gen
            res = run_gen('tts', [{x: tj[x] for x in ('text', 'voice', 'exaggeration', 'cfg_weight', 'temperature', 'seed')
                                   if tj.get(x) is not None} for tj in cbox])
            for tj, r in zip(cbox, res):
                if r.get('ok', True) and r.get('out'):
                    shutil.copyfile(r['out'], tj['out'])
            try:
                stop_gen()   # the judge needs ~6.7 GB of the card next
            except Exception:
                pass
        judge = [j for j in judge if os.path.exists(j['wav'])]
        jf = os.path.join(work, 'judge.json')
        json.dump(judge, open(jf, 'w'), indent=1)
        _run([PY['tts'], '-I', os.path.join(ROOT, 'tools', 'voice_qc.py'), '--jobs', jf, '--out',
              os.path.join(work, 'qc.json')], log)
        _run([PY['judge'], os.path.join(ROOT, 'tools', 'judge_voice.py'), '--jobs', jf, '--out',
              os.path.join(work, 'judge_out.json')], log)
        qc = {os.path.abspath(r['wav']): r for r in json.load(open(os.path.join(work, 'qc.json')))}
        jd = {os.path.abspath(r['wav']): r for r in json.load(open(os.path.join(work, 'judge_out.json')))}
        log(f'  voicedir: acted + judged in {time.time() - t0:.0f}s')
        for d, k, takes, meta in todo:
            scored = []
            for tj in takes:
                w = os.path.abspath(tj['out'])
                if w not in jd:
                    continue
                q, j = qc.get(w, {}), jd[w]
                scored.append(dict(wav=w, engine=tj['engine'], seed=tj['seed'], fails=[],
                                   score=j.get('judge_score', 0), emotion=j.get('emotion_word'),
                                   intensity=j.get('intensity'), naturalness=j.get('naturalness'),
                                   matches_direction=j.get('matches_direction'), wer=q.get('wer'),
                                   anchor=q.get('anchor_sim'), heard=q.get('heard')))
            if not scored:
                raise RuntimeError(f"voicedir: no take was made for line '{meta['text']}'")
            json.dump(dict(text=meta['text'], who=d.get('who'), direction=meta['direction'], takes=scored),
                      open(os.path.join(CACHE, k + '.json'), 'w'), indent=1)
    for d, k, takes, meta in P:
        r = json.load(open(os.path.join(CACHE, k + '.json')))
        for t in r['takes']:
            t['fails'] = _fails(t, meta)
        r['takes'].sort(key=_rank)
        p = r['pick'] = r['takes'][0]
        pick_wav = os.path.join(CACHE, f"{k}_pick_{os.path.basename(p['wav'])}")   # name changes with the pick
        if not os.path.exists(pick_wav):
            shutil.copyfile(p['wav'], pick_wav)
        d['_wav'] = pick_wav
        json.dump(r, open(os.path.join(CACHE, k + '.json'), 'w'), indent=1)
        flag = bool(p['fails'])
        rows.append(dict(at=d.get('at'), who=d.get('who'), text=meta['text'], wav=d['_wav'], flagged=flag, direction=meta['direction'],
                         matches_direction=p.get('matches_direction'),
                         fails=p['fails'], emotion=p['emotion'], intensity=p['intensity'], naturalness=p['naturalness'],
                         anchor=p['anchor'], wer=p['wer'], takes=len(r['takes']),
                         passing=sum(1 for t in r['takes'] if not t['fails'])))
        log(f"    {'FLAG' if flag else 'pass'} {d.get('who', ''):9s} {meta['text'][:44]:44s} {p['emotion']} "
            f"nat {p['naturalness']} voice {p['anchor']} ({rows[-1]['passing']}/{len(r['takes'])} takes pass)"
            + (f" -> {'; '.join(p['fails'])}" if flag else ''))
    return rows
