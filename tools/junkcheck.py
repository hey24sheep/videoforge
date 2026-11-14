"""Block a git push that adds junk: working files, review sheets, logs, build outputs, test files, big unreleased media.

Owner rule (2026-11-14): "Do not push junk, always remove junk before push."

    python tools/junkcheck.py [--repo DIR] [--range A..B] [--allow GLOB ...] [--max-mb 25]

Checks every file ADDED or MODIFIED in the commits about to be pushed (default range: @{u}..HEAD, or every tracked
file when there is no upstream yet). Exit 1 and a list when something matches; the fix is `git rm --cached <file>` +
a .gitignore rule (the file stays on disk), then amend or add a commit.

Installed as .git/hooks/pre-push by `python tools/junkcheck.py --install [--allow GLOB ...]` (per clone; hooks are
not versioned). `git push --no-verify` skips it: only when the owner says so.
"""
import argparse, fnmatch, os, re, subprocess, sys

JUNK = [   # (regex on the repo-relative path, why)
    (r'(^|/)(jobs|_trash|scratch|scratchpad|tmp|temp)/', 'working folder'),
    (r'(^|/)__pycache__/|\.py[co]$', 'Python cache'),
    (r'\.(log|tmp|part|bak|orig|swp)$|\.part\.mp4$|~$', 'log / temp / partial file'),
    (r'(^|/)review\d*/|(^|/)v\d+_review/|(^|/)grids?/|(^|/)strips?\d*/', 'review sheets / critic working files'),
    (r'(^|/)(labels|review)_[a-z]\.(json|md)$|(^|/)prompt_[a-z]\.md$|(^|/)intent\.txt$', 'critic output'),
    (r'(^|/)ep\d+/out/', 'episode build output (releases go in releases/)'),
    (r'_build\d+\.py$|_build\d+_[a-z0-9]+\.py$|_kf\d+\.py$', 'build snapshot of a generator'),
    (r'(^|/)[^/]*(_test|test_)[^/]*\.(yaml|mp4|png|jpg|wav)$', 'test file'),
    (r'(^|/)(Thumbs\.db|\.DS_Store|desktop\.ini)$', 'OS file'),
    (r'(^|/)vf\.config\.json$|(^|/)\.env$', 'local settings / secrets'),
]


def git(repo, *args):
    return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True).stdout


def changed(repo, rng):
    if rng is None:
        up = git(repo, 'rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}').strip()
        rng = f'{up}..HEAD' if up else None
    if rng is None:
        return git(repo, 'ls-files').splitlines()
    out = git(repo, 'diff', '--name-only', '--diff-filter=AM', rng)
    return [p for p in out.splitlines() if p]


MEDIA = r'\.(png|jpe?g|webp|gif|bmp|tiff?|exr|wav|mp3|flac|ogg|m4a|mp4|mov|mkv|webm|avi|safetensors|ckpt|pt|pth|bin|npy|npz|onnx|gguf)$'


CODE = r'\.(ya?ml|py|json)$'   # files whose mention of a media file means a build uses it


def referenced(repo):
    """Basenames of media files named in the repo's tracked yaml / py / json (the build inputs that are used)."""
    names = set()
    for p in git(repo, 'ls-files').splitlines():
        if re.search(CODE, p, re.I):
            try:
                txt = open(os.path.join(repo, p), encoding='utf-8', errors='ignore').read()
            except OSError:
                continue
            names.update(m.group(0).lower() for m in re.finditer(r'[\w.\-]+' + MEDIA.replace('$', r'\b'), txt, re.I))
    return names


def broken(full):
    """Why a media file is unusable (empty, unreadable, no frames), or None. Uses PIL / ffprobe when present."""
    if not os.path.isfile(full):
        return None
    if os.path.getsize(full) == 0:
        return 'empty file'
    ext = full.rsplit('.', 1)[-1].lower()
    head = open(full, 'rb').read(200)
    if head.startswith(b'version https://git-lfs'):
        return None   # LFS pointer in the work tree: the object itself is checked where it was made
    if ext in ('png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tif', 'tiff'):
        try:
            from PIL import Image
            with Image.open(full) as im:
                im.verify()
        except ImportError:
            if ext == 'png' and not head.startswith(b'\x89PNG') or ext in ('jpg', 'jpeg') and not head.startswith(b'\xff\xd8'):
                return 'not a valid image'
        except Exception as e:
            return f'broken image ({str(e)[:60]})'
    elif ext in ('mp4', 'mov', 'mkv', 'webm', 'avi', 'wav', 'mp3', 'flac', 'ogg', 'm4a'):
        try:
            r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', full],
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        try:
            if r.returncode != 0 or float(r.stdout.strip() or 0) <= 0:
                return 'unreadable or zero-length media'
        except ValueError:
            return 'unreadable media'
    return None


def check(repo, files, allow, max_mb, media_only=(), use_refs=False):
    bad = []
    refs = referenced(repo) if use_refs else set()
    for p in files:
        full = os.path.join(repo, p)
        is_media = re.search(MEDIA, p, re.I)
        if is_media:
            why = broken(full)
            if why:
                bad.append((p, why))
                continue
        elif os.path.isfile(full) and os.path.getsize(full) == 0 and not re.search(r'(__init__\.py|\.gitkeep)$', p):
            bad.append((p, 'empty file'))
            continue
        if any(fnmatch.fnmatch(p, g) for g in allow):
            continue
        if media_only and is_media and not any(fnmatch.fnmatch(p, g) for g in media_only) \
                and os.path.basename(p).lower() not in refs:
            bad.append((p, 'media that is neither a final render (' + ', '.join(media_only) + ')'
                           + (' nor used by a tracked yaml/py/json' if use_refs else '') + ''))
            continue
        why = next((w for rx, w in JUNK if re.search(rx, p, re.I)), None)
        full = os.path.join(repo, p)
        if why is None and os.path.isfile(full) and os.path.getsize(full) > max_mb * 1e6:
            why = f'over {max_mb:g} MB (put releases under an allowed folder)'
        if why:
            bad.append((p, why))
    return bad


HOOK = """#!/bin/sh
# videoforge junk check (owner rule: never push junk). Installed by tools/junkcheck.py --install.
"{py}" "{tool}" --repo "$(git rev-parse --show-toplevel)"{allow}{media}{refs} || exit 1
# then Git LFS's own pre-push (it uploads the LFS objects; replacing it broke LFS pushes on 2026-11-14)
if command -v git-lfs >/dev/null 2>&1; then
  exec git lfs pre-push "$@"
fi
"""


def install(repo, allow, media_only=(), use_refs=False):
    hooks = git(repo, 'rev-parse', '--git-path', 'hooks').strip()
    path = os.path.join(repo, hooks, 'pre-push') if not os.path.isabs(hooks) else os.path.join(hooks, 'pre-push')
    allow_args = ''.join(f' --allow "{g}"' for g in allow)
    with open(path, 'w', newline='\n') as f:
        f.write(HOOK.format(py=sys.executable.replace('\\', '/'), tool=os.path.abspath(__file__).replace('\\', '/'),
                            allow=allow_args, media=''.join(f' --media-only "{g}"' for g in media_only),
                            refs=' --referenced' if use_refs else ''))
    print(f'pre-push hook installed: {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='junkcheck', description=__doc__.split('\n\n')[0])
    ap.add_argument('--repo', default='.')
    ap.add_argument('--range', default=None, help='commit range, default @{u}..HEAD')
    ap.add_argument('--all', action='store_true', help='check every tracked file, not only the new commits')
    ap.add_argument('--allow', action='append', default=[], help='glob that is never junk (repeatable)')
    ap.add_argument('--max-mb', type=float, default=25)
    ap.add_argument('--media-only', action='append', default=[], metavar='GLOB',
                    help='images, audio, video and weights are allowed ONLY here (repeatable), e.g. final renders')
    ap.add_argument('--referenced', action='store_true',
                    help='with --media-only: also allow media that a tracked yaml/py/json names (build inputs: cast '
                         'refs, voice refs, source stills)')
    ap.add_argument('--install', action='store_true', help='install as this repo\'s pre-push hook')
    a = ap.parse_args(argv)
    repo = os.path.abspath(a.repo)
    if a.install:
        install(repo, a.allow, a.media_only, a.referenced)
        return 0
    files = git(repo, 'ls-files').splitlines() if a.all else changed(repo, a.range)
    bad = check(repo, files, a.allow, a.max_mb, a.media_only, a.referenced)
    if not bad:
        print(f'junkcheck: {len(files)} file(s) clean')
        return 0
    print(f'junkcheck: {len(bad)} junk file(s) - push blocked. Untrack them (git rm --cached), add a .gitignore rule:')
    for p, why in bad:
        print(f'  {p}   ({why})')
    return 1


if __name__ == '__main__':
    sys.exit(main())
