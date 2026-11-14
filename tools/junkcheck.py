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


def check(repo, files, allow, max_mb, media_only=()):
    bad = []
    for p in files:
        if any(fnmatch.fnmatch(p, g) for g in allow):
            continue
        if media_only and re.search(MEDIA, p, re.I) and not any(fnmatch.fnmatch(p, g) for g in media_only):
            bad.append((p, 'media outside the final-render folders (' + ', '.join(media_only) + ')'))
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
exec "{py}" "{tool}" --repo "$(git rev-parse --show-toplevel)"{allow}{media}
"""


def install(repo, allow, media_only=()):
    hooks = git(repo, 'rev-parse', '--git-path', 'hooks').strip()
    path = os.path.join(repo, hooks, 'pre-push') if not os.path.isabs(hooks) else os.path.join(hooks, 'pre-push')
    allow_args = ''.join(f' --allow "{g}"' for g in allow)
    with open(path, 'w', newline='\n') as f:
        f.write(HOOK.format(py=sys.executable.replace('\\', '/'), tool=os.path.abspath(__file__).replace('\\', '/'),
                            allow=allow_args, media=''.join(f' --media-only "{g}"' for g in media_only)))
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
    ap.add_argument('--install', action='store_true', help='install as this repo\'s pre-push hook')
    a = ap.parse_args(argv)
    repo = os.path.abspath(a.repo)
    if a.install:
        install(repo, a.allow, a.media_only)
        return 0
    files = git(repo, 'ls-files').splitlines() if a.all else changed(repo, a.range)
    bad = check(repo, files, a.allow, a.max_mb, a.media_only)
    if not bad:
        print(f'junkcheck: {len(files)} file(s) clean')
        return 0
    print(f'junkcheck: {len(bad)} junk file(s) - push blocked. Untrack them (git rm --cached), add a .gitignore rule:')
    for p, why in bad:
        print(f'  {p}   ({why})')
    return 1


if __name__ == '__main__':
    sys.exit(main())
