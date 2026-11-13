"""videoforge settings: every VF_* variable in one table, plus a local settings file.

    vf config                     table: setting, value, where it came from (env / file / default), what it does
    vf config get VF_ENCODER      one value
    vf config set VF_ENCODER cpu  save to vf.config.json (repo root, not committed)
    vf config unset VF_ENCODER    remove from the file
    vf config --json              machine-readable

Order: a real environment variable wins, then vf.config.json, then the default in the code. `apply()` runs in
vf/__init__.py, before any module reads its variable, and every child process inherits the result.
Standard library only: the gen venvs import the vf package too.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = os.path.join(ROOT, 'vf.config.json')
_AI = '<AI_ROOT>'

# (name, default as shown, group, what it sets). Defaults mirror the code; the code stays the source of truth.
SETTINGS = [
    ('VF_AI_ROOT', r'D:\claude_work\hero_film', 'paths', 'root of the AI venvs (tools/envs/), tools and models'),
    ('VF_GEN_PYTHON', _AI + r'\tools\envs\gen\Scripts\python.exe', 'paths', 'interpreter of the gen venv'),
    ('VF_TTS_PYTHON', _AI + r'\tools\envs\tts\Scripts\python.exe', 'paths', 'interpreter of the tts venv'),
    ('VF_HF_HOME', _AI + r'\models\hf', 'paths', 'Hugging Face model cache'),
    ('VF_GLOBAL_CACHE', r'.\cache', 'paths', 'shared cache of AI clips, voices, stills'),
    ('VF_FRAMECHECK_DIR', r'.\cache\framecheck', 'paths', 'learned frame checker: features, labels, model'),
    ('VF_FFMPEG', 'a local ffmpeg build', 'paths', 'ffmpeg used by the clip generator'),
    ('VF_RIFE', _AI + r'\tools\rife', 'paths', 'frame interpolation binary'),
    ('VF_ESRGAN', _AI + r'\tools\esrgan', 'paths', 'anime upscaler binary'),
    ('VF_WATERMARK', '1', 'output', '0 = no corner mark (please keep it: README Watermark)'),
    ('VF_ENCODER', 'auto', 'output', 'auto = NVENC when it works, else CPU; nvenc; cpu'),
    ('VF_GPU', '1', 'output', '0 = NumPy instead of CUDA in the compositor (very slow)'),
    ('VF_HOST_CACHE_GB', '2.5', 'output', 'compositor frame cache in RAM'),
    ('VF_DEV_CACHE_GB', '1.5', 'output', 'compositor frame cache on the GPU'),
    ('VF_NO_CACHE', 'off', 'cache', '1 = ignore the disk caches'),
    ('VF_GC', '1', 'cache', '0 = skip the cache clean-up after a build'),
    ('VF_ALLOW_ENCODE', 'off', 'models', '1 = allow text encoders to load for new prompts (~20 GB free RAM)'),
    ('VF_ENCODE_MIN_RAM', '20', 'models', 'free RAM (GB) a text encoder needs before it loads'),
    ('VF_ENCODE_VRAM_CAP', '6.0', 'models', 'VRAM (GB) a text encoder may use'),
    ('VF_PROC_VRAM_GB', '7.2', 'models', 'VRAM cap (GB) per model process'),
    ('VF_LTX_MODEL', 'ltx2_25_22B_distilled', 'models', 'LTX video model'),
    ('VF_LTX_STEPS', '8', 'models', 'LTX sampling steps'),
    ('VF_GEN_PERSIST', '1', 'worker', '0 = one-shot model processes instead of the warm worker'),
    ('VF_GEN_PORT', '47219', 'worker', 'port of the warm gen worker'),
    ('VF_GEN_IDLE', '600', 'worker', 'seconds before an idle gen worker exits'),
    ('VF_GEN_PARK_IDLE', '120', 'worker', 'seconds before an idle worker parks its weights in RAM'),
    ('VF_GEN_TIMEOUT', '3600', 'worker', 'seconds per gen child run / worker reply'),
]
NAMES = {s[0] for s in SETTINGS}


def load_file():
    try:
        with open(FILE, encoding='utf-8') as f:
            d = json.load(f)
        return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d):
    tmp = FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(d.items())), f, indent=1)
    os.replace(tmp, FILE)


_FROM_ENV = set()   # VF_* names the environment set before apply() read the file
_APPLIED = [False]


def apply():
    """Fill os.environ from vf.config.json; a variable already set in the environment wins."""
    _FROM_ENV.update(k for k in os.environ if k.startswith('VF_'))
    _APPLIED[0] = True
    for k, v in load_file().items():
        os.environ.setdefault(k, v)


def rows():
    """[(name, value, source, default, group, help)] for every known setting, plus unknown VF_* in env or file."""
    f = load_file()
    env = _FROM_ENV if _APPLIED[0] else {k for k in os.environ if k.startswith('VF_')}

    def where(name):
        if name in env:
            return os.environ.get(name, ''), 'env'
        if name in f:
            return f[name], 'file'
        return None, 'default'
    out = []
    for name, default, group, what in SETTINGS:
        val, src = where(name)
        out.append((name, default if val is None else val, src, default, group, what))
    for name in sorted((set(f) | set(env)) - NAMES):
        val, src = where(name)
        out.append((name, val or '', src, '', 'other', '(not in the table: see the code)'))
    return out


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='vf config', description=__doc__.split('\n\n')[0])
    ap.add_argument('action', nargs='?', default='show', choices=['show', 'get', 'set', 'unset', 'path'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('value', nargs='?')
    ap.add_argument('--json', action='store_true', help='print the table as JSON')
    ap.add_argument('--group', default=None, help='only this group: ' + ', '.join(sorted({s[2] for s in SETTINGS})))
    a = ap.parse_args(argv)
    from . import ui
    if a.action == 'path':
        print(FILE)
        return 0
    if a.action in ('get', 'set', 'unset'):
        if not a.name:
            ap.error(f'{a.action} needs a setting name, e.g. VF_ENCODER')
        name = a.name.upper()
        if not name.startswith('VF_'):
            name = 'VF_' + name
        if a.action == 'get':
            r = [x for x in rows() if x[0] == name]
            print(r[0][1] if r else '')
            return 0 if r else 1
        d = load_file()
        if a.action == 'set':
            if a.value is None:
                ap.error('set needs a value')
            if name not in NAMES:
                ui.warn(f'{name} is not a known setting (saved anyway)')
            d[name] = a.value
            _save(d)
            ui.ok(f'{name} = {a.value}  (saved to {os.path.basename(FILE)})')
            if name in _FROM_ENV and os.environ.get(name) != a.value:
                ui.warn(f'the environment sets {name}={os.environ[name]}, which wins over the file')
        else:
            if d.pop(name, None) is None:
                ui.warn(f'{name} was not in {os.path.basename(FILE)}')
                return 1
            _save(d)
            ui.ok(f'{name} removed: back to the default')
        return 0
    r = [x for x in rows() if not a.group or x[4] == a.group]
    if a.json:
        print(json.dumps([dict(name=n, value=v, source=s, default=d, group=g, help=h) for n, v, s, d, g, h in r],
                         indent=1))
        return 0
    ui.config_table(r, FILE)
    return 0
