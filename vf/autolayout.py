"""Episode layout from a script: beats (shots with their lines) + real voice lengths -> shot durations and dialogue
`at:` times. Shared by every episode generator (2026-11-09).

A generator script builds a list of beats with `insert()` / `talk_shot()` (or plain shot dicts carrying `lines`,
`mindur`, `kind`) and calls `layout(beats, durs)`. Voice lengths come from `voice_durs(<episode>/out/voices.json)`
after `vf voices` (a line not voiced yet is estimated from its word count). Re-run the generator after voicing so
every shot is cut to its real line lengths.

Layout rules (owner-approved on ep02): a shot's first line starts `lead` (0.2 s) after the cut, lines follow each
other with `gap` (0.12 s), the shot ends `tail` (0.1 s) after its last line, never shorter than its `mindur`.
"""
import copy, json, os, wave

LEAD, GAP, TAIL = 0.2, 0.12, 0.1
# talk shots (InfiniteTalk, vf/talk.py): what keeps the generated face clean. "hands out of frame" means NOTHING may
# be held in the keyframe: a held prop then floats (owner 2026-11-09, a lantern at a witness's chest).
NOBREATH = "only a faint thin wisp of breath at most, the mouth always clearly visible, the hands out of frame, no wristwatch"


def L(who, text, direction, **kw):
    """A dialogue line: who, text, acting direction (voicedir), extra keys (fx, say, emotion...)."""
    d = dict(who=who, text=text, direction=direction)
    d.update(kw)
    return d


def talk_shot(id, who, face, kf, line, direction, motion, style, cam=None, seed=7, extra_lines=(), rerolls=None,
              image=None, nobreath=NOBREATH, extra=None):
    """An on-screen speaking shot: keyframe (prompt `kf` or a supplied `image`) acted over the line by InfiniteTalk."""
    pre = [t.split('.')[0].split(',')[0][:24].strip() for t in [line] + [x['text'] for x in extra_lines]]
    g = dict(**({'style': style} if style else {}), motion=motion, seed=seed, hires=True, k6_size=[576, 1024], retries=0,
             # every line of the shot is acted on screen (extra_lines too: talk.line takes a list of prefixes)
             talk=dict(line=pre if len(pre) > 1 else pre[0], prompt=motion + " " + nobreath + ".",
                       **({'rerolls': rerolls} if rerolls else {})))
    if image:
        g['image'] = image
    else:
        g['keyframe'] = kf
    d = dict(id=id, kind='talk', cast=[face], lines=[L(who, line, direction)] + list(extra_lines),
             camera=cam or dict(move='push_in', amount=0.05, handheld=0.04, ease='lin'), gen=g)
    if extra:
        d.update(extra)
    return d


def insert(id, kf=None, motion='', lines=(), mindur=3.0, cam=None, seed=7, image=None, cast=None, extra=None):
    """A shot without on-screen speech (voice-over lines allowed)."""
    g = dict(motion=motion, seed=seed)
    if image:
        g['image'] = image
    else:
        g['keyframe'] = kf
    d = dict(id=id, kind='insert', cast=cast or [], lines=list(lines), mindur=mindur,
             camera=cam or dict(move='push_in', amount=0.06, handheld=0.04, ease='lin'), gen=g)
    if extra:
        d.update(extra)
    return d


def est(text, who):
    rate = 2.35 if who == 'narrator' else 2.5
    return round(len(text.split()) / rate + 0.35, 2)


def voice_durs(voices_json):
    """{line text: seconds} from a `vf voices` / build report (out/voices.json); {} if not voiced yet."""
    if not os.path.exists(voices_json):
        return {}
    out = {}
    for r in json.load(open(voices_json, encoding='utf-8')):
        w = r.get('wav')
        if w and os.path.exists(w):
            with wave.open(w) as f:
                out[r['text']] = round(f.getnframes() / f.getframerate(), 2)
    return out


def preview_talk(s, who_words='person'):
    """--preview: a talk shot becomes a plain K6 clip with closed lips (fast animatic, no lip-sync)."""
    s['gen'].pop('talk', None)
    s['gen'].pop('hires', None)
    s['gen'].pop('k6_size', None)
    s['gen']['motion'] = (f"Close-up of the {who_words} holding still with the lips closed, a slow calm breath, "
                          "the eyes alive and thinking.")
    return s


def layout(beats, durs, lead=LEAD, gap=GAP, tail=TAIL, preview=False, preview_words=None):
    """beats: shot dicts with `lines`, `mindur`, `kind` (+ optional `gen_extra` merged into gen).
    Returns (shots, dialogue, total_seconds, starts{id: t})."""
    shots, dialogue, starts, t = [], [], {}, 0.0
    for b in beats:
        s = copy.deepcopy(b)
        lines = s.pop('lines', [])
        mind = s.pop('mindur', 2.2)
        kind = s.pop('kind', 'insert')
        extra = s.pop('gen_extra', None)
        if extra:
            s.setdefault('gen', {}).update(extra)
        if preview and kind == 'talk':
            preview_talk(s, (preview_words or {}).get(s.get('cast', [''])[0], 'person'))
        s.setdefault('layers', [])
        cur = t + lead
        for ln in lines:
            d = durs.get(ln.get('say') or ln['text'], durs.get(ln['text'], est(ln.get('say') or ln['text'], ln['who'])))   # voices.json keys the spoken wording
            ln2 = dict(who=ln['who'], at=round(cur, 2), text=ln['text'], direction=ln['direction'])
            for k, v in ln.items():
                if k not in ('who', 'text', 'direction'):   # fx: None = explicitly no voice effect
                    ln2[k] = v
            dialogue.append(ln2)
            cur += d + gap
        dur = max(mind, cur - t + tail) if lines else mind   # cur already holds the gap after the last line
        s['dur'] = round(dur, 2)
        starts[s['id']] = round(t, 2)
        shots.append(s)
        t += dur
    return shots, dialogue, round(t, 2), starts
