"""Style profiles (styles/styles.yaml): prompt wrapping, negatives, grade, motion and model defaults per look.

    series.yaml:   style: papercraft        # any key of styles.yaml (episode.yaml may set its own `style:`)
Precedence for every setting: built-in default < style profile < series.yaml < episode.yaml.
"""
import os, re
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_S = {}


def _merge(a, b):
    from .spec import _merge as m
    return m(a, b)


def all_styles():
    if 'all' not in _S:
        st = yaml.safe_load(open(os.path.join(ROOT, 'styles', 'styles.yaml'), encoding='utf-8'))
        local = os.path.join(ROOT, 'styles', 'local')   # your own styles, kept out of git (a folder or a junction)
        if os.path.isdir(local):
            for f in sorted(os.listdir(local)):
                if f.endswith(('.yaml', '.yml')):
                    st.update(yaml.safe_load(open(os.path.join(local, f), encoding='utf-8')) or {})
        _S['all'] = st
    return _S['all']


def get(key):
    st = all_styles()
    if key not in st or key.startswith('_'):
        raise KeyError(f'unknown style {key!r}; choose from: ' + ', '.join(k for k in st if not k.startswith('_')))
    return _merge(st['_defaults'], st[key])


def shot_style(spec_style, g):
    """A shot's `gen: {style: <key>}` overrides the episode style for its keyframe prompt only (2026-11-06)."""
    k = (g or {}).get('style')
    return get(k) if isinstance(k, str) else spec_style


def wrap_prompt(style, text, characters=None, kind='keyframe'):
    """Style prefix + prompt (with `{name}` replaced by the series character sheet, verbatim) + style suffix.
    kind='keyframe' uses prompt_prefix/suffix (a still: lighting, lens, set dressing). kind='motion' uses
    motion_prefix/suffix (how things move: steady camera, physical motion) so still-only words ("cinema still",
    "rain-slick asphalt") don't leak into indoor or motion prompts; falls back to a neutral motion wrapper."""
    chars = characters or {}
    text = re.sub(r'\{(\w+)\}', lambda m: str(chars.get(m.group(1), m.group(0))).strip(), text)
    st = style or {}
    if kind == 'motion':
        pre = st.get('motion_prefix', 'cinematic live-action footage, ')
        suf = st.get('motion_suffix', ', steady locked-off camera, natural physically plausible motion, '
                                      'consistent identity, stable anatomy')
        return f'{pre}{text}{suf}'
    return f"{st.get('prompt_prefix', '')}{text}{st.get('prompt_suffix', '')}"


def listing():
    st = all_styles()
    return [(k, v['name'], v.get('eight_gb_risk', '?')) for k, v in st.items() if not k.startswith('_')]


TEMP_NEGATIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'temp_negative_keywords.txt')


def temp_negative(path=TEMP_NEGATIVE):
    """The owner's temporary negative keyword list ('' when the file is missing or its `active:` line says no).
    Only models run with real CFG (> 1) can use negatives; the distilled guidance-1.0 models ignore them."""
    if not os.path.exists(path):
        return ''
    on, words = False, []
    for line in open(path, encoding='utf-8'):
        t = line.strip()
        if not t or t.startswith('#'):
            continue
        if t.lower().startswith('active:'):
            on = t.split(':', 1)[1].strip().lower() in ('yes', 'true', 'on', '1')
            continue
        words.append(t)
    return ', '.join(w.strip().rstrip(',') for w in words) if on else ''
