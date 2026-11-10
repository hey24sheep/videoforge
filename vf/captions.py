"""Auto captions from the dialogue track, per format: 'subtitle' (one line, bottom) or 'shorts' (big 1-3 word
chunks that pop, placed above the Shorts/Reels UI safe zone), or 'none'."""
import os, re, wave


def line_timings(spec):
    """[(start, end, text, who)] using the real wav length when a voice file exists."""
    out = []
    for d in spec['dialogue']:
        t0 = float(d['at'])
        dur = d.get('dur')
        wav = d.get('_wav') or d.get('wav')
        if wav and os.path.exists(wav):
            with wave.open(wav) as w:
                dur = w.getnframes() / w.getframerate()
        if dur is None:
            dur = 0.35 * len(d['text'].split()) + 0.3
        out.append((t0, t0 + float(dur), d['text'], d.get('who')))
    return out


def word_times(d):
    """[(start, end)] per caption word from the TTS word timings (<wav>.words.json), absolute seconds; None when
    unknown or when the spoken words don't line up with the caption words."""
    import json
    wav = d.get('_wav') or d.get('wav')
    if not wav or d.get('say') or not os.path.exists(wav + '.words.json'):
        return None
    ws = json.load(open(wav + '.words.json', encoding='utf-8')).get('words') or []
    if len(ws) != len(d['text'].split()):
        return None
    t0 = float(d['at'])
    return [(t0 + w['start'], t0 + w['end']) for w in ws]


def _chunks(text, max_words=3, max_chars=18):
    words = text.split()
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(' '.join(cur + [w])) > max_chars or re.search(r'[.,!?;:…]$', cur[-1])):
            out.append(' '.join(cur))
            cur = []
        cur.append(w)
    if cur:
        out.append(' '.join(cur))
    return out


MIN_TAIL = 0.4  # a caption piece shorter than this after a cut is an orphan ("AROUND." over the next shot): drop it


def _to_shots(spec, t0, t1, layer):
    """Split one absolute-time caption across the shots it overlaps (shot-local at/until). A short spill past a cut
    is dropped and the caption fades out on the cut instead."""
    res = []
    for s in spec['shots']:
        a, b = max(t0, s['start']), min(t1, s['start'] + s['dur'])
        if b > a:
            if a > t0 and b - a < MIN_TAIL and res:  # orphan tail: end the previous piece at its cut
                res[-1][1]['out'] = layer.get('out', 0.08) or 0.08
                continue
            L = dict(layer, at=a - s['start'], until=b - s['start'])
            if s.get('caption_y') is not None:  # a shot whose clue sits where captions go moves them
                L['y'] = float(s['caption_y'])
            if a > t0:
                L['in'] = 0.0  # continues across the cut: no re-fade
            if b < t1:
                L['out'] = 0.0
            res.append((s['id'], L))
    return res


def build(spec):
    """{fmt: {shot_id: [text layers]}}"""
    lines = line_timings(spec)
    out = {}
    for fmt, style in spec['captions'].items():
        per = {}
        if style == 'none' or not lines:
            out[fmt] = per
            continue
        for (t0, t1, text, who), dl in zip(lines, spec['dialogue']):
            if style == 'shorts':
                ch = _chunks(text)
                total = sum(len(c) for c in ch) or 1
                wt = word_times(dl)
                starts = None
                if wt:  # each chunk appears on its first spoken word
                    k, starts = 0, []
                    for c in ch:
                        starts.append(wt[k][0] if k else t0)
                        k += len(c.split())
                t = t0
                for j, c in enumerate(ch):
                    d = (t1 - t0) * len(c) / total
                    if starts:
                        d = (starts[j + 1] if j + 1 < len(ch) else t1) - starts[j]
                        t = starts[j]
                    # slightly left of centre and narrow: clears the like/comment/share column on the right
                    # cinematic, not meme: smaller, thin outline + soft shadow (user 2026-11-02: "text looks bad")
                    L = dict(type='text', text=c.upper(), font='caption', size=0.058, y=0.74, x=0.47, color='#f4f1ea',
                             stroke=0.07, stroke_color='#000000', shadow=True, anim='fade', max_w=0.74)
                    # the next chunk replaces this one on the same frame: no fade-out, so nothing blinks
                    L['in'], L['out'] = 0.08, (0.08 if j == len(ch) - 1 else 0.0)
                    for sid, l2 in _to_shots(spec, t, t + d, L):
                        per.setdefault(sid, []).append(l2)
                    t += d
            else:
                L = dict(type='text', text=text, font='body', size=0.042, y=0.9, x=0.5, color='#f2f2f2',
                         stroke=0.06, stroke_color='#000000', shadow=True, anim='fade', max_w=0.8, plate=0.45)
                L['in'], L['out'] = 0.12, 0.15
                for sid, l2 in _to_shots(spec, t0 - 0.05, t1 + 0.2, L):
                    per.setdefault(sid, []).append(l2)
        out[fmt] = per
    return out
