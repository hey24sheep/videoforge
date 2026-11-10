"""videoforge as an MCP server: any MCP client (Claude Code, Claude Desktop, other agents) can plan, write, render,
check and judge episodes through tools, without knowing the CLI.

    python -m vf.mcp_server                 stdio (what Claude Code / Claude Desktop launch; see .mcp.json)
    python -m vf.mcp_server --http 8765     streamable HTTP on 127.0.0.1 (a remote or cloud agent through a tunnel)

Design:
- Every tool wraps an existing `vf` command, so the server and the CLI never disagree.
- Quick tools (lint, styles, read, judge table) answer directly. GPU work (key frames, builds, cuts, checks, judges
  that load a model) starts as a detached job under tools/jobwatch.py and returns its id at once; the client polls
  `job_status` / `job_log`. A job survives the client disconnecting.
- Paths are relative to the repo root and may not leave it; files are written only under series/ and examples/.
"""
import argparse, json, os, re, subprocess, sys

from mcp.server.mcpserver import MCPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
WRITABLE = ('series', 'examples')

mcp = MCPServer(
    name='videoforge',
    instructions=(
        'videoforge turns a script into finished 16:9 and 9:16 videos with open models on a local GPU. '
        'Typical flow: list_styles -> new_episode or write_file (series/<s>/<ep>/episode.yaml) -> lint -> '
        'keyframes (review the sheet image) -> build (preview first) -> check -> release_cut. '
        'GPU tools return a job id: poll job_status / job_log until it ends; run one GPU job at a time. '
        'Read the resource videoforge://docs/workflow before writing an episode. Judge with judge_* tools and '
        'blind critics (critic_prompts -> run each prompt in a fresh subagent -> critic_table).'),
)


# ------------------------------------------------------------------ helpers
def _path(p, write=False):
    full = os.path.realpath(os.path.join(ROOT, p))
    if os.path.commonpath([full, os.path.realpath(ROOT)]) != os.path.realpath(ROOT):
        raise ValueError(f'path leaves the repo: {p}')
    if write and os.path.relpath(full, ROOT).replace('\\', '/').split('/')[0] not in WRITABLE:
        raise ValueError(f'writes are allowed only under {", ".join(WRITABLE)}/')
    return full


def _vf(*args, timeout=600):
    """Run `python -m vf ...` and return its output (stderr warnings from CuPy dropped)."""
    r = subprocess.run([PY, '-m', 'vf'] + [str(a) for a in args], cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout, encoding='utf-8', errors='replace')
    err = '\n'.join(l for l in r.stderr.splitlines() if 'warn' not in l.lower())
    return (r.stdout + ('\n' + err if err.strip() else '')).strip() + f'\n[exit {r.returncode}]'


def _job(name, argv, env=None):
    """Start a detached job under tools/jobwatch.py; return its id and log path."""
    cmd = [PY, 'tools/jobwatch.py', 'run', '--name', re.sub(r'[^\w-]', '_', name)]
    for k, v in (env or {}).items():
        cmd += ['--env', f'{k}={v}']
    out = subprocess.run(cmd + ['--'] + argv, cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r'started (\S+)', out)
    if not m:
        return dict(ok=False, output=out)
    return dict(ok=True, job=m.group(1), log=f'jobs/{m.group(1)}.log',
                next='poll job_status(job) until it is ok / FAILED; job_log(job) shows the tail')


def _wm(on):
    """Job env for the watermark switch (vf/render.py reads VF_WATERMARK)."""
    return {} if on else {'VF_WATERMARK': '0'}


def _ep(episode):
    p = _path(episode)
    if os.path.isdir(p):
        p = os.path.join(p, 'episode.yaml')
    if not os.path.exists(p):
        raise ValueError(f'no episode file: {episode}')
    return os.path.relpath(p, ROOT)


# ------------------------------------------------------------------ discover
@mcp.tool()
def doctor() -> str:
    """Check the machine: Python, venvs, GPU, NVENC (or the CPU encoder), ffmpeg, model folders, free disk."""
    return _vf('doctor', timeout=300)


@mcp.tool()
def list_styles() -> str:
    """Every visual style (series or per-shot `style:`), with its look and how risky it is on an 8 GB GPU."""
    return _vf('styles')


@mcp.tool()
def list_series() -> dict:
    """Series folders, their episodes and the finished videos in each episode's out/ folder."""
    res = {}
    base = os.path.join(ROOT, 'series')
    for s in sorted(os.listdir(base)):
        sd = os.path.join(base, s)
        if not os.path.isdir(sd):
            continue
        eps = {}
        for e in sorted(os.listdir(sd)):
            if os.path.exists(os.path.join(sd, e, 'episode.yaml')):
                od = os.path.join(sd, e, 'out')
                eps[e] = sorted(f for f in os.listdir(od) if f.endswith('.mp4')) if os.path.isdir(od) else []
        res[s] = eps
    return res


@mcp.tool()
def read_file(path: str, max_chars: int = 20000) -> str:
    """Read a text file in the repo (episode.yaml, series.yaml, a script draft, a QC json, a log)."""
    return open(_path(path), encoding='utf-8', errors='replace').read()[:max_chars]


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write a text file under series/ or examples/ (an episode.yaml, a generator script, a script draft).
    YAML files are parsed first and refused if invalid."""
    full = _path(path, write=True)
    if full.endswith(('.yaml', '.yml')):
        import yaml
        yaml.safe_load(content)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, 'w', encoding='utf-8').write(content)
    return f'wrote {os.path.relpath(full, ROOT)} ({len(content)} chars)'


@mcp.tool()
def new_episode(series: str, episode: str, number: int, title: str) -> str:
    """Scaffold series/<series>/<episode>/<episode>_gen.py (a generator that writes episode.yaml)."""
    return _vf('new', series, episode, '--no', number, '--title', title)


# ------------------------------------------------------------------ author + render
@mcp.tool()
def lint(episode: str) -> str:
    """Check an episode for authoring mistakes that cause artifacts (no GPU, seconds)."""
    return _vf('lint', _ep(episode))


@mcp.tool()
def keyframes(episode: str, formats: str = 'yt', allow_encode: bool = True) -> dict:
    """GPU job: the first frame of every AI shot + out/keyframes_sheet.jpg to review before any video is made.
    formats: yt (16:9), short (9:16), or both 'yt,short'."""
    ep = _ep(episode)
    return _job('mcp_kf', [PY, '-m', 'vf', 'keyframes', ep, '--formats', formats] +
                (['--allow-encode'] if allow_encode else []), env=dict(VF_GC='0'))


@mcp.tool()
def build(episode: str, formats: str = 'yt', preview: bool = True, allow_encode: bool = True, watermark: bool = True) -> dict:
    """GPU job: voices, AI clips (cached, QC'd, re-rolled), finishing, grade, captions, mix and encode.
    preview=True renders at half size / 30 fps for review; set False for the final.
    watermark=False leaves out the small hey24sheep.com/videoforge corner mark; please keep it on unless the owner
    asks (it is as small as it can be and helps the project grow)."""
    ep = _ep(episode)
    argv = [PY, '-m', 'vf', 'build', ep, '--formats', formats] + (['--preview'] if preview else []) + \
           (['--allow-encode'] if allow_encode else [])
    return _job('mcp_build', argv, env=dict(VF_GC='0', **_wm(watermark)))


@mcp.tool()
def run_pipeline(target: str, until: str = 'cut', preview: bool = False, watermark: bool = True) -> dict:
    """GPU job: the whole path for a <ep>_gen.py or episode.yaml: gen -> lint -> voices -> keyframes -> cut.
    until: gen | lint | voices | keyframes | cut.
    watermark=False leaves out the small hey24sheep.com/videoforge corner mark; please keep it on unless the owner
    asks (it is as small as it can be and helps the project grow)."""
    t = os.path.relpath(_path(target), ROOT)
    return _job('mcp_run', [PY, '-m', 'vf', 'run', t, '--until', until] + (['--preview'] if preview else []),
                env=_wm(watermark))


@mcp.tool()
def check(episode: str, video: str = '') -> dict:
    """GPU job: post-render gates on the newest render (or `video`): brightness dips, frozen frames, lip-sync,
    loudness. Exit 1 in the log = BLOCK."""
    argv = [PY, '-m', 'vf', 'check', _ep(episode)] + (['--video', _path(video)] if video else [])
    return _job('mcp_check', argv)


@mcp.tool()
def release_cut(episode: str, phone_mb: float = 28.0, watermark: bool = True) -> dict:
    """GPU job: series intro + episode + outro, checked and joined, plus a phone copy under phone_mb.
    watermark=False leaves out the small hey24sheep.com/videoforge corner mark; please keep it on unless the owner
    asks (it is as small as it can be and helps the project grow)."""
    return _job('mcp_cut', [PY, '-m', 'vf', 'cut', _ep(episode), '--phone-mb', phone_mb], env=_wm(watermark))


@mcp.tool()
def render_titles(spec: str, watermark: bool = True) -> dict:
    """CPU job: a motion-graphics title sequence (intro / outro) from a vf.mograph spec yaml.
    watermark=False leaves out the small hey24sheep.com/videoforge corner mark; please keep it on unless the owner
    asks (it is as small as it can be and helps the project grow)."""
    return _job('mcp_titles', [PY, '-m', 'vf.mograph', 'render', os.path.relpath(_path(spec), ROOT)], env=_wm(watermark))


# ------------------------------------------------------------------ jobs
@mcp.tool()
def job_status(job: str = '') -> str:
    """Status of recent jobs (running / ok / FAILED / STALLED / KILLED), or of one job id."""
    out = subprocess.run([PY, 'tools/jobwatch.py', 'check'], cwd=ROOT, capture_output=True, text=True).stdout
    if not job:
        return out
    keep, on = [], False
    for line in out.splitlines():
        if line.startswith('['):
            on = job in line
        if on:
            keep.append(line)
    return '\n'.join(keep) or f'no job {job} in the last 48 h'


@mcp.tool()
def job_log(job: str, lines: int = 60) -> str:
    """Last lines of a job's log (progress bars and CuPy warnings dropped)."""
    p = _path(os.path.join('jobs', f'{job}.log'))
    txt = open(p, encoding='utf-8', errors='replace').read().replace('\r', '\n').splitlines()
    txt = [l for l in txt if l.strip() and 'warn' not in l.lower() and '%|' not in l]
    return '\n'.join(txt[-lines:])


# ------------------------------------------------------------------ judges
@mcp.tool()
def judge_music(files: list[str], target: str, avoid: list[str] | None = None) -> str:
    """Rank music takes: CLAP match to `target` minus `avoid`, plus each take's loudness per second (CPU)."""
    args = ['judge', 'music'] + [_path(f) for f in files] + ['--target', target]
    for a in avoid or []:
        args += ['--avoid', a]
    return _vf(*args)


@mcp.tool()
def judge_clip(file: str, questions: list[str] | None = None, frames: int = 8) -> dict:
    """GPU job: a vision-language model describes a clip or still and answers yes/no questions.
    Write a question as 'Is there a person in frame?=no' to mark the expected answer."""
    kind = 'image' if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) else 'clip'
    argv = [PY, '-m', 'vf', 'judge', kind, _path(file), '--frames', str(frames)]
    for q in questions or []:
        argv += ['--q', q]
    return _job('mcp_judge', argv)


@mcp.tool()
def judge_script(files: list[str], context: str = '') -> dict:
    """GPU job: score script drafts with the local rubric model (12 yes/no craft questions each)."""
    argv = [PY, '-m', 'vf', 'judge', 'script'] + [_path(f) for f in files] + (['--context', _path(context)] if context else [])
    return _job('mcp_script', argv)


@mcp.tool()
def judge_dialogue(files: list[str], text: str, direction: str = '') -> dict:
    """Job: voice takes of one line: words heard + error rate, same voice across takes, acting (if installed)."""
    argv = [PY, '-m', 'vf', 'judge', 'dialogue'] + [_path(f) for f in files] + ['--text', text]
    return _job('mcp_dialogue', argv + (['--direction', direction] if direction else []))


@mcp.tool()
def critic_prompts(kind: str, entries: dict[str, str], out_dir: str, context: str = '') -> dict:
    """Blind critics. kind: script | dialogue | video. entries: {name: path} (text files, or frame-sheet images for
    video). Writes one prompt per critic (Opus, Sonnet; never Fable) with relabelled, reordered entries.
    Run each prompt in a FRESH subagent, save its reply to <out_dir>/<critic>.answer.md, then call critic_table."""
    args = ['judge', 'critics', kind, '--out', _path(out_dir, write=True)]
    for n, p in entries.items():
        args += ['--entry', f'{n}={_path(p)}']
    if context:
        args += ['--context', _path(context)]
    _vf(*args)
    d = _path(out_dir)
    return {f[:-11]: os.path.relpath(os.path.join(d, f), ROOT) for f in os.listdir(d) if f.endswith('.prompt.txt')}


@mcp.tool()
def critic_table(out_dir: str) -> str:
    """Scores from the critics' saved answers: per-entry mean over critics and criteria, and the lowest criterion."""
    return _vf('judge', 'table', _path(out_dir))


# ------------------------------------------------------------------ docs
for _uri, _rel, _desc in [('videoforge://docs/readme', 'README.md', 'Setup, commands, styles, timings'),
                          ('videoforge://docs/workflow', 'docs/workflow.md', 'Idea to delivered video, with every gate'),
                          ('videoforge://docs/reference', 'docs/reference.md', 'Episode fields, camera, caching, models'),
                          ('videoforge://styles', 'styles/styles.yaml', 'Style profiles')]:
    def _make(rel):
        return lambda: open(os.path.join(ROOT, rel), encoding='utf-8').read()
    mcp.resource(_uri, name=_rel, description=_desc)(_make(_rel))


@mcp.prompt()
def make_video(idea: str, format: str = '16:9', seconds: int = 30) -> str:
    """Start a new video from an idea."""
    return (f'Make a {seconds}-second {format} video with videoforge from this idea: {idea}\n'
            'Read videoforge://docs/workflow first. Write 2-3 script options and judge them with critic_prompts / '
            'critic_table; pick a style with list_styles; write the episode with write_file; lint; keyframes and '
            'show me the sheet; then build a preview, check it, and only then the final.')


def main():
    ap = argparse.ArgumentParser(prog='python -m vf.mcp_server')
    ap.add_argument('--http', type=int, default=0, help='serve streamable HTTP on 127.0.0.1:<port> instead of stdio')
    a = ap.parse_args()
    if a.http:
        mcp.run('streamable-http', host='127.0.0.1', port=a.http)
    else:
        mcp.run()


if __name__ == '__main__':
    main()
