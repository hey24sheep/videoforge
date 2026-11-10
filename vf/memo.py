"""Three-tier memoisation: if a function's code and inputs didn't change, its output didn't either.

    @memo('mem')      in-process LRU only (per-frame helpers: text bitmaps, resized sprites, gradients)
    @memo('project')  on disk inside the episode folder (.vfcache): shot segments, audio stems, layouts
    @memo('global')   on disk shared by every series/episode (videoforge/cache): AI clips, upscales, mattes,
                      TTS lines, key art — anything whose inputs (prompt, seed, model, source file) recur

Key = BLAKE2(function qualname + source of the function and of every videoforge function it reaches
            + fingerprint of the arguments). Arrays hash by content; existing file/dir paths hash by
            (path, size, mtime) of every file inside; dicts/lists recursively.

Functions that produce FILES use @memo_file(tier, suffix): the decorator passes `out=<cache path>` and returns
that path; the call is skipped when the path already exists for the same key. Writes are atomic (tmp + rename),
so a crashed run never leaves a half-written cache entry.
"""
import functools, hashlib, inspect, os, pickle, sys, types, threading
from collections import OrderedDict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBAL_DIR = os.environ.get('VF_GLOBAL_CACHE', os.path.join(ROOT, 'cache'))
_project = {'dir': None}
_mem = OrderedDict()
_MEM_MAX = 512
_lock = threading.Lock()
STATS = {'hit': 0, 'miss': 0}
# Every disk-cache path this process read or wrote (vf/gc.py keeps exactly what the latest build of each episode
# touched; 2026-11-03: superseded takes/finishes/stamps were never removed and grew to ~14 GB).
TOUCHED = set()


def touch(p):
    TOUCHED.add(os.path.abspath(p))
    return p
SALT = 'vf1'  # bump to invalidate every cache entry after a semantic change of this module


def set_project(path):
    """Episode folder: project-tier entries live in <path>/.vfcache."""
    _project['dir'] = os.path.abspath(path)


def tier_dir(tier):
    if tier == 'global':
        d = GLOBAL_DIR
    elif tier == 'project':
        if not _project['dir']:
            raise RuntimeError('memo: set_project() before using project-tier caches')
        d = os.path.join(_project['dir'], '.vfcache')
    else:
        raise ValueError(tier)
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ code fingerprint
_code_h = {}


def _vf_module(m):
    f = getattr(m, '__file__', None)
    return bool(f) and os.path.abspath(f).startswith(os.path.join(ROOT, 'vf'))


def _is_vf_obj(o):
    return _vf_module(sys.modules.get(getattr(o, '__module__', None) or ''))


def _const_repr(o):
    """repr of a module constant, or None when it isn't stable data (objects print their address)."""
    if isinstance(o, (int, float, str, tuple, bool)):
        return repr(o)
    if isinstance(o, (dict, list)):
        r = repr(o)
        return None if ' at 0x' in r or len(r) > 20000 else r
    return None


def _module_by_name(name):
    """Import a vf module by absolute name; None when it can't be imported in this interpreter (e.g. a torch-only
    module seen from the render venv): the caller then hashes its file text, so the key stays deterministic."""
    m = sys.modules.get(name)
    if m is not None:
        return m
    try:
        import importlib
        return importlib.import_module(name)
    except Exception:
        return None


def _referenced_modules(c, g, fn_module):
    """vf modules a code object can reach: globals bound to modules + its own (function-local) imports.
    Only these are searched for `mod.attr` / `from .mod import attr` names. (The old walk searched EVERY loaded
    vf module for a matching name, so keys changed with whatever else happened to be imported.)"""
    import dis, importlib.util
    mods, files = [], []
    for name in c.co_names:
        o = g.get(name)
        if isinstance(o, types.ModuleType) and _vf_module(o):
            mods.append(o)
    pkg = fn_module.rpartition('.')[0] if fn_module else ''
    level = 0
    for ins in dis.get_instructions(c):
        if ins.opname == 'LOAD_CONST' and isinstance(ins.argval, int):
            level = ins.argval
        elif ins.opname == 'IMPORT_NAME':
            try:
                full = importlib.util.resolve_name('.' * level + ins.argval if level else ins.argval, pkg)
            except Exception:
                full = ins.argval
            level = 0
            if not (full == 'vf' or full.startswith('vf.')):
                continue
            m = _module_by_name(full)
            if m is not None and _vf_module(m):
                mods.append(m)
            elif m is None:
                files.append(os.path.join(ROOT, *full.split('.')) + '.py')
    return mods, files


def _walk(fn, h, seen):
    if id(fn) in seen:
        return
    seen.add(id(fn))
    try:
        h.update(inspect.getsource(fn).encode())
    except (OSError, TypeError):
        h.update(getattr(getattr(fn, '__code__', None), 'co_code', b''))
    co = getattr(fn, '__code__', None)
    if co is None:
        return
    g = getattr(fn, '__globals__', {})

    def visit(c):
        for name in c.co_names:
            o = g.get(name)
            if isinstance(o, types.ModuleType):
                continue  # attribute access is resolved below through the referenced modules
            o = getattr(o, '__wrapped__', o)
            if isinstance(o, types.FunctionType) and _is_vf_obj(o):
                _walk(o, h, seen)
            elif isinstance(o, type) and _is_vf_obj(o):
                try:
                    h.update(inspect.getsource(o).encode())
                except (OSError, TypeError):
                    pass
            elif not name.startswith('_') and name in g:
                r = _const_repr(o)
                if r is not None:
                    h.update(f'{name}={r}'.encode())  # module constants it reads (incl. dicts such as PROFILES)
        for k in c.co_consts:
            if isinstance(k, types.CodeType):
                visit(k)
        mods, files = _referenced_modules(c, g, getattr(fn, '__module__', ''))
        for m in mods:  # module.attr and function-local `from .m import attr`
            for name in c.co_names:
                if name in m.__dict__:
                    o = getattr(m.__dict__[name], '__wrapped__', m.__dict__[name])
                    if isinstance(o, types.FunctionType) and _is_vf_obj(o):
                        _walk(o, h, seen)
        for p in files:
            if p not in seen:
                seen.add(p)
                try:
                    h.update(open(p, 'rb').read())
                except OSError:
                    pass
    visit(co)


def code_hash(fn):
    k = id(fn)
    if k not in _code_h:
        h = hashlib.blake2b(digest_size=16)
        h.update(SALT.encode())
        _walk(fn, h, set())
        _code_h[k] = h.hexdigest()
    return _code_h[k]


# ------------------------------------------------------------------ argument fingerprint
_PFP = {}


def _path_fp(p, h):
    """(size, mtime) of a file or of every file in a folder; computed once per process per path (assets don't
    change during a build, and walking a 120-frame folder each call cost ~4 ms)."""
    p = os.path.abspath(p)
    if p not in _PFP:
        g = hashlib.blake2b(digest_size=16)
        _path_fp_raw(p, g)
        _PFP[p] = g.digest()
    h.update(_PFP[p])


def _path_fp_raw(p, h):
    if os.path.isdir(p):
        for root, _, files in sorted(os.walk(p)):
            for f in sorted(files):
                q = os.path.join(root, f)
                st = os.stat(q)
                h.update(f'{os.path.relpath(q, p)}|{st.st_size}|{st.st_mtime_ns}'.encode())
    else:
        st = os.stat(p)
        h.update(f'{os.path.abspath(p)}|{st.st_size}|{st.st_mtime_ns}'.encode())


def fingerprint(x, h=None):
    top = h is None
    h = h or hashlib.blake2b(digest_size=16)
    if isinstance(x, np.ndarray):
        h.update(f'nd{x.dtype}{x.shape}'.encode())
        h.update(np.ascontiguousarray(x).tobytes())
    elif hasattr(x, '__cuda_array_interface__'):
        import cupy as cp
        fingerprint(cp.asnumpy(x), h)
    elif isinstance(x, (str, os.PathLike)):
        s = os.fspath(x)
        if len(s) < 400 and (os.sep in s or '/' in s) and os.path.exists(s):
            h.update(b'p' + os.path.abspath(s).encode())  # absolute: keys don't depend on the working directory
            _path_fp(s, h)
        else:
            h.update(b's' + s.encode())
    elif isinstance(x, dict):
        h.update(b'd')
        for k in sorted(x, key=repr):
            h.update(repr(k).encode())
            fingerprint(x[k], h)
    elif isinstance(x, (list, tuple)):
        h.update(f'l{len(x)}'.encode())
        for v in x:
            fingerprint(v, h)
    elif isinstance(x, float):
        h.update(repr(round(x, 9)).encode())
    elif x is None or isinstance(x, (int, bool, bytes)):
        h.update(repr(x).encode())
    elif callable(x):
        h.update(code_hash(x).encode())
    else:
        h.update(repr(x).encode())
    return h.hexdigest() if top else h


_VERSIONS = {}


def _code_part(fn):
    v = _VERSIONS.get(fn)
    if v is None:
        return code_hash(fn)
    return 'v:' + (v() if callable(v) else str(v))


def key_for(fn, args, kwargs):
    """Keyword arguments named `_xxx` are passed through but NOT part of the key (context objects, progress
    callbacks, QC collectors): everything that decides the output must be a normal argument.
    Functions decorated with `version=` key on that explicit version instead of their code (AI generators: their
    loaders/speed code change often but must not throw away minutes of generated media; bump the version when
    the OUTPUT would change)."""
    kw = {k: v for k, v in kwargs.items() if not k.startswith('_')}
    return hashlib.blake2b(f'{fn.__module__}.{fn.__qualname__}|{_code_part(fn)}|'
                           f'{fingerprint([list(args), kw])}'.encode(), digest_size=16).hexdigest()


# ------------------------------------------------------------------ decorators
def _mem_get(k):
    with _lock:
        if k in _mem:
            _mem.move_to_end(k)
            return True, _mem[k]
    return False, None


def _mem_put(k, v):
    with _lock:
        _mem[k] = v
        while len(_mem) > _MEM_MAX:
            _mem.popitem(last=False)


def memo(tier='mem', version=None):
    """Cache the RETURN VALUE. Disk tiers store numpy arrays as .npy (memory-mapped on read), else pickle.
    version: explicit key version (str or zero-arg callable) instead of the code fingerprint."""
    def deco(fn):
        if version is not None:
            _VERSIONS[fn] = version
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            k = key_for(fn, args, kwargs)
            ok, v = _mem_get(k)  # VF_NO_CACHE skips only the disk tiers: in-process reuse is always safe
            if ok:
                STATS['hit'] += 1
                return v
            if tier != 'mem' and os.environ.get('VF_NO_CACHE') != '1':
                d = os.path.join(tier_dir(tier), fn.__qualname__)
                for ext in ('.npy', '.pkl'):
                    p = os.path.join(d, k + ext)
                    if os.path.exists(p):
                        touch(p)
                        v = np.load(p, mmap_mode='r') if ext == '.npy' else pickle.load(open(p, 'rb'))
                        _mem_put(k, v)
                        STATS['hit'] += 1
                        return v
            STATS['miss'] += 1
            v = fn(*args, **kwargs)
            if tier != 'mem':
                d = os.path.join(tier_dir(tier), fn.__qualname__)
                os.makedirs(d, exist_ok=True)
                ext = '.npy' if isinstance(v, np.ndarray) else '.pkl'
                p = os.path.join(d, k + ext)
                tmp = p + f'.{os.getpid()}.tmp'
                with open(tmp, 'wb') as f:
                    np.save(f, v) if ext == '.npy' else pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, p)
                touch(p)
            _mem_put(k, v)
            return v
        wrap.memo_tier = tier
        return wrap
    return deco


def memo_file(tier='global', suffix='', version=None):
    """Cache a function that WRITES its result to `out` (a file path, or a directory if suffix == '/').
    version: explicit key version (str or zero-arg callable) instead of the code fingerprint."""
    def deco(fn):
        if version is not None:
            _VERSIONS[fn] = version
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            k = key_for(fn, args, kwargs)
            d = os.path.join(tier_dir(tier), fn.__qualname__)
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, k + (suffix if suffix != '/' else ''))
            no_cache = os.environ.get('VF_NO_CACHE') == '1'
            if os.path.exists(out) and not no_cache:
                STATS['hit'] += 1
                return touch(out)
            STATS['miss'] += 1
            tmp = out + f'.{os.getpid()}.tmp' + (suffix if suffix not in ('/', '') else '')
            if suffix == '/':
                os.makedirs(tmp, exist_ok=True)
            import glob as _g, shutil

            def _rm(q):
                shutil.rmtree(q, ignore_errors=True) if os.path.isdir(q) else (os.path.exists(q) and os.remove(q))
            try:
                fn(*args, out=tmp, **kwargs)
            except BaseException:  # a failed/aborted step must not leave its partial output in the cache
                _rm(tmp)
                for s in _g.glob(_g.escape(tmp) + '.*'):
                    _rm(s)
                raise
            sidecars = _g.glob(_g.escape(tmp) + '.*')  # e.g. <segment>.qc.json travels with its segment
            if os.path.exists(out) and not no_cache:  # another process finished first
                _rm(tmp)
                for s in sidecars:
                    os.remove(s)
            else:
                if os.path.exists(out):  # --no-cache: the fresh result REPLACES the old one (it used to be thrown away)
                    _rm(out)
                os.replace(tmp, out)
                for s in sidecars:
                    os.replace(s, out + s[len(tmp):])
            return touch(out)
        wrap.memo_tier = tier
        return wrap
    return deco
