"""Cache garbage collection (2026-11-03: every re-roll, version bump and finishing change left its old outputs in the
cache forever: ~14 GB of superseded clips, finishes and stamps).

    python -m vf gc            dry run: what would go, and how much
    python -m vf gc --apply    move it to the Recycle Bin (never a permanent delete)

Every build records the cache files it read or wrote (memo.TOUCHED) in <episode>/.vfcache/manifest.json. A file in a
collected folder is junk when NO episode's latest build touched it. Safety rules:
  - only the folders in GC_DIRS (written by the build process itself, so its manifest is complete). Folders the gen
    worker writes (t2i keyframes, i2v, depth_maps, say, embeddings) and hand-made assets are never collected;
  - an episode with AI shots but no manifest yet blocks the whole run (its files would look unused);
  - files newer than the newest manifest are skipped (a build may be writing them), and nothing runs while another
    process holds the GPU lock;
  - any path written in a series.yaml / episode.yaml is kept;
  - removal = Recycle Bin, after checking the bin has room (Windows deletes over-quota files for good). If it hasn't,
    files are moved to videoforge/_trash/<date>/ instead.
Each build runs it automatically at the end (build.py), so regeneration cleans up after itself.
"""
import glob, json, os, re, shutil, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'cache')
# build-process writers only (memo qualname folders + vf/k6.py's own). _resmoke: removed tool, no producer left.
GC_DIRS = ('k6', 'k6fin', '_stamp', '_lock', '_swing', '_erase', '_retex', '_repair', '_analyze', 'retime',
           '_resmoke', '_face_tone')
STRAY = re.compile(r'(\.\d+\.tmp(\.\w+)?|\.part\.mp4)$')  # crashed-run leftovers anywhere in GC_DIRS


def _episodes():
    return sorted(glob.glob(os.path.join(ROOT, 'series', '*', '*', 'episode.yaml')))


def _yaml_paths():
    keep = set()
    for y in glob.glob(os.path.join(ROOT, 'series', '**', '*.yaml'), recursive=True):
        txt = open(y, encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r'[A-Za-z]:[\\/][^\s\'",}\]]+', txt):
            keep.add(os.path.normcase(os.path.abspath(m.group(0))))
    return keep


def plan(log=print):
    """-> (junk [(path, bytes)], reason-if-refused or None)"""
    from .gpulock import owner, _alive
    for b in glob.glob(os.path.join(CACHE, 'builds', '*.json')):  # other builds running: their new files are unlisted
        try:
            r = json.load(open(b))
            if int(r['pid']) != os.getpid() and _alive(int(r['pid']), r.get('since')):
                return [], f'another build is running (pid {r["pid"]})'
            if int(r['pid']) != os.getpid():
                os.remove(b)  # dead build's registration
        except (OSError, ValueError, KeyError):
            pass
    o = owner()
    if o and int(o.get('pid', -1)) != os.getpid() and _alive(int(o.get('pid', -1)), o.get('since')):
        return [], f'GPU lock held by pid {o.get("pid")} ({o.get("what")}): a build is running'
    keep, newest, proj = set(), 0.0, []
    for y in _episodes():
        ep = os.path.dirname(y)
        mp = os.path.join(ep, '.vfcache', 'manifest.json')
        if not os.path.exists(mp):
            if re.search(r'^\s*gen:', open(y, encoding='utf-8', errors='ignore').read(), re.M):
                return [], f'{os.path.relpath(ep, ROOT)} has AI shots but no manifest yet: build it once first'
            continue
        m = json.load(open(mp, encoding='utf-8'))
        if m.get('_v') != 2 and re.search(r'^\s*gen:', open(y, encoding='utf-8', errors='ignore').read(), re.M):
            # v1 manifests list only render segments: trusting one would recycle every AI clip of that episode
            return [], f'{os.path.relpath(ep, ROOT)} has an old-format manifest: rebuild it once first'
        for k, v in m.items():
            if isinstance(v, list):
                keep |= {os.path.normcase(os.path.abspath(x)) for x in v}
        newest = max(newest, os.path.getmtime(mp))
        proj.append((ep, m))
    keep |= _yaml_paths()
    junk = []

    def consider(p):
        n = os.path.normcase(os.path.abspath(p))
        if n in keep or any(n.startswith(k + '.') for k in _prefixes(n, keep)):
            return  # kept, or a sidecar (<file>.qc.json ...) of a kept file
        try:
            st = os.stat(p)
        except OSError:
            return
        if st.st_mtime >= newest - 5 and not (STRAY.search(p) and st.st_mtime < time.time() - 3600):
            return  # written after the newest manifest (maybe a build in progress); crash leftovers after 1 h
        junk.append((p, st.st_size))
    for d in GC_DIRS:
        base = os.path.join(CACHE, d)
        if not os.path.isdir(base):
            continue
        for e in os.scandir(base):
            if e.is_dir():  # memo_file('/') outputs are folders: one unit
                n = os.path.normcase(os.path.abspath(e.path))
                if n not in keep:
                    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(e.path) for f in fs)
                    if os.path.getmtime(e.path) < newest - 5:
                        junk.append((e.path, size))
            else:
                consider(e.path)
    for ep, m in proj:  # project tier: shot segments / mixes no current build of this episode uses
        d = os.path.join(ep, '.vfcache')
        for r, _, fs in os.walk(d):
            for f in fs:
                p = os.path.join(r, f)
                if f == 'manifest.json' or f.endswith('.npy') or f.endswith('.pkl'):
                    continue  # small memo values: cheap to keep, costly to recompute
                consider(p)
    return junk, None


def _prefixes(n, keep):
    """kept paths that n could be a sidecar of (n = kept + '.something'): check by stripping extensions."""
    out, q = [], n
    while True:
        q2, ext = os.path.splitext(q)
        if not ext:
            return out
        if q2 in keep:
            out.append(q2)
        q = q2


def _bin_room(drive):
    """Bytes the drive's Recycle Bin can still take, or None if unknown."""
    try:
        import ctypes, winreg
        from ctypes import wintypes
        buf = ctypes.create_unicode_buffer(64)
        if not ctypes.windll.kernel32.GetVolumeNameForVolumeMountPointW(drive + '\\', buf, 64):
            return None
        guid = re.search(r'\{[^}]+\}', buf.value).group(0)
        k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                           rf'Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket\Volume\{guid}')
        cap = winreg.QueryValueEx(k, 'MaxCapacity')[0] * 2**20
        nuke = winreg.QueryValueEx(k, 'NukeOnDelete')[0]
        if nuke:
            return 0

        class Q(ctypes.Structure):
            _fields_ = [('cbSize', wintypes.DWORD), ('i64Size', ctypes.c_longlong), ('i64NumItems', ctypes.c_longlong)]
        q = Q()
        q.cbSize = ctypes.sizeof(Q)
        if ctypes.windll.shell32.SHQueryRecycleBinW(drive + '\\', ctypes.byref(q)) != 0:
            return None
        return cap - q.i64Size
    except Exception:
        return None


def _recycle(paths):
    """Send to the Recycle Bin (undoable). Returns True on success."""
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [('hwnd', wintypes.HWND), ('wFunc', ctypes.c_uint), ('pFrom', wintypes.LPCWSTR),
                    ('pTo', wintypes.LPCWSTR), ('fFlags', ctypes.c_ushort), ('fAnyOperationsAborted', wintypes.BOOL),
                    ('hNameMappings', ctypes.c_void_p), ('lpszProgressTitle', wintypes.LPCWSTR)]
    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 4, 16, 64, 1024
    op = SHFILEOPSTRUCTW(wFunc=FO_DELETE, pFrom=''.join(os.path.abspath(p) + '\0' for p in paths) + '\0',
                         fFlags=FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return rc == 0 and not op.fAnyOperationsAborted


def collect(apply=False, log=print, quiet=False):
    junk, why = plan(log)
    if why:
        log(f'gc: skipped - {why}')
        return 0
    total = sum(b for _, b in junk)
    if not junk:
        if not quiet:
            log('gc: nothing to remove')
        return 0
    if not apply:
        by = {}
        for p, b in junk:
            d = os.path.relpath(p, ROOT).split(os.sep)
            k = os.sep.join(d[:2]) if d[0] == 'cache' else os.sep.join(d[:4])
            by[k] = (by.get(k, (0, 0))[0] + 1, by.get(k, (0, 0))[1] + b)
        for k, (n, b) in sorted(by.items(), key=lambda x: -x[1][1]):
            log(f'  {k:40s} {n:5d} file(s) {b / 2**30:7.2f} GB')
        log(f'gc: {len(junk)} unused file(s), {total / 2**30:.2f} GB (dry run; --apply moves them to the Recycle Bin)')
        return total
    room = _bin_room(os.path.splitdrive(ROOT)[0]) if os.name == 'nt' else None
    paths = [p for p, _ in junk]
    if os.name == 'nt' and room is not None and room > total * 1.1 and _recycle(paths):
        left = [p for p in paths if os.path.exists(p)]
        log(f'gc: moved {len(paths) - len(left)} unused file(s), {total / 2**30:.2f} GB to the Recycle Bin'
            + (f' ({len(left)} in use, kept)' if left else ''))
        return total
    dst = os.path.join(ROOT, '_trash', time.strftime('%Y%m%d_%H%M%S'))
    for p in paths:  # bin full / unknown: same-drive move, nothing deleted
        q = os.path.join(dst, os.path.relpath(p, ROOT))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        try:
            shutil.move(p, q)
        except OSError:
            pass
    log(f'gc: Recycle Bin has no room for {total / 2**30:.2f} GB: moved to {dst} instead (delete it when sure)')
    return total
