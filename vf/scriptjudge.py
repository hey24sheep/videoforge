"""`vf script-judge`: judging v2 for episode scripts (2026-11-10). Craft basis: the series' own CRAFT.md.

The old gate (blind critics list every logic gap -> patch -> re-score) plateaued at 6.1-7.1 for 10 test-episode
drafts and produced "a radio play of facts". v2 judges DRAMA (70 %) with persona judges ranking several drafts against
an owner-approved anchor episode; logic only counts the 3 worst "a one-watch viewer is lost here" points (30 %).

The judges are Claude subagents run by the session (Opus/Sonnet/Haiku, never Fable); this module only writes their
prompts and aggregates their JSON answers, so every run uses the same wording and the same maths.

  python -m vf script-judge transcript <episode.yaml> [--out f.md]   viewer transcript of a built episode
  python -m vf script-judge prompts --premise P.md --entry NAME=file.md ... --out DIR
        writes DIR/<persona>.prompt.txt (+ DIR/jobs.json: persona -> model); the session runs one subagent per prompt
        and saves each answer as DIR/<persona>.answer.md (the answer ends with one ```json block)
  python -m vf script-judge table DIR       per-entry scores (drama 70 % + logic 30 %), ranks, Borda total
"""
import json, os, re, sys

DRAMA = {
    'hook': 'first 3 s: a conflict, threat or impossible image on screen, no backstory needed',
    'stakes': 'personal and immediate: this person, this night',
    'hero_choice': 'the lead makes a choice and it visibly costs him something',
    'antagonist': 'the antagonist or his proxy ACTS on screen (watches, moves, takes, threatens), not only described',
    'suspense': 'the audience knows a danger the hero does not, and/or a visible clock',
    'turns': 'scenes turn a value (safe->unsafe, trust->betrayal); nothing ends where it started',
    'subtext': 'characters want and hide things; nobody recites facts both know; no voice-over recapping the picture',
    'image': 'one signature image people would remember or put in a trailer',
    'pull': 'must watch the next episode right now',
}
INTRO = {
    'identity': 'unmistakably THIS series: its own recurring motifs, places and objects, not generic genre stock',
    'mood': 'dread and suspense that match a noir mystery; the score and the picture feel like one thing',
    'tease': 'hints at the central mystery without explaining it; makes you want the episode',
    'motion': 'every shot really moves (no stills), each shot readable on a phone (not too short to see)',
    'rhythm': 'cuts sit on the music; it builds to the title hit; no empty stretches',
    'bridge': 'works placed right AFTER a cold open (carries the tension on, does not reset it) and at the very start',
    'title': 'the series name and the episode name land clearly and stay readable',
    'repeat': 'still watchable on the 10th episode: right length, not annoying, you would not reach for skip',
    'image': 'one iconic image people associate with the show',
}
KINDS = {'episode': DRAMA, 'intro': INTRO}
PERSONAS = {
    'showrunner': dict(model='opus', who='the showrunner of a prestige noir/thriller series giving development notes '
                                         '(notes on problems, never rewrites)'),
    'exec_opus': dict(model='opus', who='a network drama development executive deciding what goes into production'),
    'exec_sonnet': dict(model='sonnet', who='a network drama development executive deciding what goes into production'),
    'binger': dict(model='sonnet', who='a 35-year-old who binge-watches crime series and decides after every episode '
                                       'whether to keep going or quit the show'),
    'viewer': dict(model='haiku', who='a 24-year-old who watches crime/thriller shorts on a phone, low patience, '
                                      'swipes away the second something drags or confuses'),
}
DRAMA_W, LOGIC_W = 0.7, 0.3
VOICES = {'narrator': 'NARRATOR (voice-over)'}   # speaker labels in transcripts


def load_rubric(path):
    """A series' own judge wording (kept with the series, e.g. series/<s>/judge.yaml):
        episode: {criterion: text}   intro: {criterion: text}   voices: {who: LABEL}   personas: {name: {who, model}}
    Keys present replace the defaults; new criteria are added."""
    import yaml
    r = yaml.safe_load(open(path, encoding='utf-8')) or {}
    for kind, crit in KINDS.items():
        crit.update(r.get(kind) or {})
    VOICES.update(r.get('voices') or {})
    for n, v in (r.get('personas') or {}).items():
        PERSONAS[n] = dict(PERSONAS.get(n, {}), **v)
    return r


def _first_sentence(t):
    t = re.sub(r'\s+', ' ', t or '').strip()
    m = re.match(r'(.+?[.;])(\s|$)', t)
    return (m.group(1) if m else t)[:220]


def transcript(episode_yaml):
    """Viewer transcript of an episode spec: one line per shot = [what moves on screen] + the lines heard in it.
    The picture text is the first sentence of the shot's motion prompt (stamped/in-scene text is not in the spec:
    add it by hand when it matters)."""
    import yaml
    ep = yaml.safe_load(open(episode_yaml, encoding='utf-8'))
    lines, t = [], 0.0
    dl = sorted(ep.get('dialogue') or [], key=lambda d: float(d.get('at', 0)))
    voices = VOICES
    for i, s in enumerate(ep.get('shots') or []):
        dur = float(s.get('dur', 0))
        g = s.get('gen') or {}
        what = _first_sentence(g.get('motion') or g.get('keyframe') or s.get('id'))
        txt = [L.get('text') for L in s.get('layers') or [] if L.get('type') == 'text' and L.get('text')]
        said = [d for d in dl if t <= float(d.get('at', 0)) < t + dur]
        heard = ' '.join(f'{voices.get(d["who"], d["who"].upper())}: "{d["text"]}"' for d in said)
        shown = f' On screen: {" / ".join(txt)}.' if txt else ''
        lines.append(f'{i + 1}. [{what}{shown}] {heard}'.rstrip())
        t += dur
    return '\n'.join(lines) + '\n'


def _criteria_text(crit=None):
    return '\n'.join(f'- {k}: {v}' for k, v in (crit or DRAMA).items())


def prompt(persona, premise, entries, kind='episode'):
    """entries: [(name, path)]; the judge reads only these files."""
    p = PERSONAS[persona]
    crit = KINDS[kind]
    files = '\n'.join(f'- {n}: {os.path.abspath(f)}' for n, f in entries)
    names = ', '.join(n for n, _ in entries)
    schema = ('{"entries": {"<NAME>": {' + ', '.join(f'"{k}": 1-10' for k in crit) +
              ', "lost": ["<at most 3 moments a one-watch viewer is truly lost, not open mysteries the show means to '
              'keep>"], "swipe": "<first second you would swipe, or none>", "watch_next": true/false, '
              '"reveal_over": true/false}}, "rank": ["<NAME best>", "..."]}')
    nxt = ('would you sit through it every episode instead of skipping it (that is "watch_next")' if kind == 'intro'
           else 'would you watch the next episode right now')
    return f"""You are {p['who']}. Be direct and honest; do not be generous.

{premise.strip()}

Read these files and no others (formats may differ; judge the drama, not the formatting):
{files}

For EACH of {names}: score 1-10 with one line of evidence each on:
{_criteria_text(crit)}
Also: the 3 worst moments where a one-watch viewer is truly lost (open mysteries the show keeps on purpose are NOT
gaps), the first second you would swipe away (or none), {nxt}, and does it spend
too many of the series' big secrets for its place in the season.
Then rank them best to worst with one line of reason each.

End your answer with ONE fenced ```json block, exactly this shape:
{schema}
Under 800 words before the json."""


def write_prompts(premise_file, entries, out, kind='episode'):
    os.makedirs(out, exist_ok=True)
    premise = open(premise_file, encoding='utf-8').read()
    jobs = {}
    for name in PERSONAS:
        f = os.path.join(out, f'{name}.prompt.txt')
        open(f, 'w', encoding='utf-8').write(prompt(name, premise, entries, kind))
        jobs[name] = dict(model=PERSONAS[name]['model'], prompt=f, answer=os.path.join(out, f'{name}.answer.md'))
    json.dump(dict(entries=entries, jobs=jobs, kind=kind), open(os.path.join(out, 'jobs.json'), 'w'), indent=1)
    return jobs


def _parse(answer_text):
    m = re.findall(r'```json\s*(\{.*?\})\s*```', answer_text, re.S)
    if not m:
        raise ValueError('no ```json block in the answer')
    return json.loads(m[-1])


def score_entry(e, crit=None):
    drama = [float(e[k]) for k in (crit or DRAMA) if isinstance(e.get(k), (int, float))]
    d = sum(drama) / len(drama) if drama else 0.0
    logic = max(4.0, 10.0 - 2.0 * len((e.get('lost') or [])[:3]))
    return round(DRAMA_W * d + LOGIC_W * logic, 2), round(d, 2), logic


def table(out):
    meta = json.load(open(os.path.join(out, 'jobs.json')))
    crit = KINDS[meta.get('kind', 'episode')]
    names = [n for n, _ in meta['entries']]
    per, borda, rows = {}, {n: 0 for n in names}, []
    for persona, job in meta['jobs'].items():
        if not os.path.exists(job['answer']):
            print(f'[missing] {persona}: {job["answer"]}')
            continue
        r = _parse(open(job['answer'], encoding='utf-8').read())
        for n in names:
            e = r['entries'].get(n)
            if e:
                per.setdefault(n, {})[persona] = (score_entry(e, crit), e)
        for i, n in enumerate(r.get('rank', [])):
            if n in borda:
                borda[n] += len(names) - 1 - i
    personas = [p for p in meta['jobs'] if any(p in per.get(n, {}) for n in names)]
    head = '| entry | ' + ' | '.join(personas) + ' | mean | drama | Borda | watch next |'
    rows.append(head)
    rows.append('|' + '---|' * (len(personas) + 5))
    summary = {}
    for n in names:
        s = per.get(n, {})
        tot = [s[p][0][0] for p in personas if p in s]
        dr = [s[p][0][1] for p in personas if p in s]
        wn = sum(bool(s[p][1].get('watch_next')) for p in personas if p in s)
        mean = round(sum(tot) / len(tot), 2) if tot else None
        summary[n] = dict(mean=mean, drama=round(sum(dr) / len(dr), 2) if dr else None, borda=borda[n], watch_next=wn,
                          per={p: s[p][0][0] for p in personas if p in s},
                          criteria={k: round(sum(float(s[p][1].get(k, 0)) for p in personas if p in s) / max(1, len(tot)), 1)
                                    for k in crit})
        rows.append(f'| {n} | ' + ' | '.join(str(s[p][0][0]) if p in s else '-' for p in personas) +
                    f' | {mean} | {summary[n]["drama"]} | {borda[n]} | {wn}/{len(tot)} |')
    crit = '| criterion | ' + ' | '.join(names) + ' |\n|' + '---|' * (len(names) + 1) + '\n' + '\n'.join(
        f'| {k} | ' + ' | '.join(str(summary[n]['criteria'][k]) for n in names) + ' |' for k in crit)
    text = '\n'.join(rows) + '\n\n' + crit + '\n'
    open(os.path.join(out, 'table.md'), 'w', encoding='utf-8').write(text)
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    return text


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='vf script-judge')
    sub = ap.add_subparsers(dest='c', required=True)
    t = sub.add_parser('transcript')
    t.add_argument('episode')
    t.add_argument('--out')
    p = sub.add_parser('prompts')
    p.add_argument('--premise', required=True, help='text file: the series premise the judges get')
    p.add_argument('--entry', action='append', required=True, help='NAME=path (repeat)')
    p.add_argument('--out', required=True)
    p.add_argument('--kind', default='episode', choices=list(KINDS))
    tb = sub.add_parser('table')
    tb.add_argument('dir')
    for sp in (t, p, tb):
        sp.add_argument('--rubric', help='series judge.yaml: its own criteria wording, speaker labels, personas')
    a = ap.parse_args(argv)
    rub = a.rubric
    if not rub and a.c == 'transcript':   # a series may keep judge.yaml next to series.yaml
        cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.episode))), 'judge.yaml')
        rub = cand if os.path.exists(cand) else None
    if rub:
        load_rubric(rub)
    if a.c == 'transcript':
        s = transcript(a.episode)
        if a.out:
            open(a.out, 'w', encoding='utf-8').write(s)
        print(s)
    elif a.c == 'prompts':
        jobs = write_prompts(a.premise, [tuple(e.split('=', 1)) for e in a.entry], a.out, a.kind)
        for n, j in jobs.items():
            print(f'{n:12s} {j["model"]:7s} {j["prompt"]}')
    else:
        print(table(a.dir))


if __name__ == '__main__':
    main(sys.argv[1:])
