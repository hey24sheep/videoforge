"""Episode spec: series.yaml (shared look/fonts/voices) + episode.yaml (shots, dialogue, audio). See README.md.

Any layer value may be a constant or keyframes [[t, value], ...] (seconds from shot start, eased in/out).
Any layer may carry per-format overrides: `yt: {...}` / `short: {...}` merged over its base settings.
"""
import copy, math, os
import yaml

FORMATS = {
    'yt': dict(W=1920, H=1080, label='YouTube 16:9 1080p'),
    'short': dict(W=1080, H=1920, label='Shorts/Reels 9:16 1080x1920'),
    'yt4k': dict(W=3840, H=2160, label='YouTube 16:9 4K'),
}
DEFAULT_LOOK = dict(exposure=1.0, bloom=0.5, bloom_threshold=0.9, tonemap='knee', saturation=1.0, vignette=0.3,
                    grain=0.018, split=0.12)


def _merge(a, b):
    out = copy.deepcopy(a) if a is not None else {}
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load(episode_path):
    episode_path = os.path.abspath(episode_path)
    root = os.path.dirname(episode_path)
    ep = yaml.safe_load(open(episode_path, encoding='utf-8')) or {}
    series_path = ep.get('series') or os.path.join(os.path.dirname(root), 'series.yaml')
    if not os.path.isabs(series_path):
        series_path = os.path.normpath(os.path.join(root, series_path))
    se = yaml.safe_load(open(series_path, encoding='utf-8')) if os.path.exists(series_path) else {}
    sdir = os.path.dirname(series_path)
    style_key = ep.get('style') or se.get('style')
    style = {}
    if style_key:
        from .styles import get
        style = get(style_key)
    spec = dict(
        root=root, path=episode_path, series=se, series_dir=sdir, style_key=style_key, style=style,
        title=ep.get('title', os.path.basename(root)),
        fps=int(ep.get('fps', se.get('fps', 60))),
        look=_merge(_merge(_merge(DEFAULT_LOOK, style.get('look')), se.get('look')), ep.get('look')),
        talk_look=_merge(se.get('talk_look'), ep.get('talk_look')) or None,   # look overrides for gen.talk shots (vf/talk.py)
        motion=_merge(_merge(style.get('motion'), se.get('motion')), ep.get('motion')),
        models=_merge(_merge(style.get('models'), se.get('models')), ep.get('models')),
        fonts=_merge({'title': 'bahnschrift.ttf', 'caption': 'segoeuib.ttf', 'body': 'segoeui.ttf'},
                     _merge(se.get('fonts'), ep.get('fonts'))),
        captions=_merge(_merge({'yt': 'subtitle', 'short': 'shorts', 'yt4k': 'subtitle'}, style.get('captions')),
                        _merge(se.get('captions'), ep.get('captions'))),
        audio=ep.get('audio', {}) or {},
        dialogue=ep.get('dialogue', []) or [],
        shots=[],
        voices=_merge(se.get('voices'), ep.get('voices')),
        # defaults merged under every shot's `gen:` block (e.g. size, upscale, profile)
        gen=_merge(_merge(style.get('gen'), se.get('gen')), ep.get('gen')),
        # default virtual camera for shots without their own `camera:` (vf/camera.py); `camera: false` = locked
        camera=_merge(_merge(style.get('camera'), se.get('camera')), ep.get('camera')),
        # the small corner mark on every render (vf/render.py watermark; README: Watermark). Episode beats series.
        watermark=ep.get('watermark', se.get('watermark', True)),
    )
    t = 0.0
    problems = []
    for i, s in enumerate(ep.get('shots', [])):
        s = copy.deepcopy(s)
        s.setdefault('id', f'shot{i + 1:02d}')
        s['start'] = t
        s['dur'] = float(s.get('dur', 0))
        if s['dur'] <= 0:
            problems.append(f"shot {s['id']}: dur must be > 0")
        t += s['dur']
        spec['shots'].append(s)
    spec['duration'] = t
    # fail early with the shot/layer named, instead of a decoder error in the middle of an encode
    for s in spec['shots']:
        for j, L in enumerate(s.get('layers', [])):
            for v in [L] + [L.get(f) or {} for f in FORMATS]:
                p = v.get('src')
                if p and not any(c in p for c in '*?') and not os.path.exists(resolve(spec, p)):
                    problems.append(f"shot {s['id']} layer {j + 1} ({L.get('type')}): missing file {p}")
    if problems:
        raise ValueError('episode problems:\n  ' + '\n  '.join(problems))
    return spec


def resolve(spec, p):
    """Paths in specs are relative to the episode folder, then the series folder."""
    if p is None or os.path.isabs(p):
        return p
    for base in (spec['root'], spec['series_dir']):
        q = os.path.normpath(os.path.join(base, p))
        if os.path.exists(q) or any(c in p for c in '*%'):
            return q
    return os.path.normpath(os.path.join(spec['root'], p))


def for_format(layer, fmt):
    """Layer settings for one format: base, then `yt:` (also used by yt4k), then the format's own overrides."""
    base = {k: v for k, v in layer.items() if k not in FORMATS}
    if fmt == 'yt4k':
        base = _merge(base, layer.get('yt'))
    return _merge(base, layer.get(fmt))


# ------------------------------------------------------------------ keyframes
def ease(e, k):
    k = min(1.0, max(0.0, k))
    if e == 'lin':
        return k
    if e == 'in':
        return k * k
    if e == 'out':
        return 1 - (1 - k) ** 2
    return k * k * (3 - 2 * k)  # 'io'


def kf(v, t, e='io'):
    """Constant or [[t, value], ...] (value scalar or list) -> value at t."""
    if not (isinstance(v, list) and v and isinstance(v[0], list) and len(v[0]) == 2 and
            isinstance(v[0][0], (int, float))):
        return v
    if t <= v[0][0]:
        return v[0][1]
    for (t0, a), (t1, b) in zip(v, v[1:]):
        if t <= t1:
            k = ease(e, (t - t0) / max(1e-9, t1 - t0))
            if isinstance(a, list):
                return [x + (y - x) * k for x, y in zip(a, b)]
            return a + (b - a) * k
    return v[-1][1]
