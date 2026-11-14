"""Terminal look of the vf CLI: overview screen, tables, ok/warn lines and progress bars (rich).

Without `rich`, or when output goes to a file or pipe (jobwatch logs), everything prints plain text and progress bars
draw nothing, so log lines that tools/jobwatch.py matches stay exactly as before.
"""
import contextlib, os, sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH = True
except ImportError:   # plain-text fallback
    RICH = False

GROUPS = [
    ('Make', [('run', 'whole path: gen, lint, voices, keyframes, cut, framecheck'),
              ('new', 'scaffold series/<series>/<ep>/<ep>_gen.py'),
              ('build', 'render an episode.yaml to one or more formats'),
              ('keyframes', 'first frames of AI shots + a review contact sheet'),
              ('voices', 'dialogue takes only, with timings'),
              ('cut', 'release cut: intro + episode + outro, check, join, phone copy')]),
    ('Check', [('lint', 'authoring mistakes that cause artifacts (no GPU)'),
               ('check', 'post-render gates: dips, frozen frames, lip-sync'),
               ('framecheck', 'learned frame checker: defect frames, shot scores, fix plan'),
               ('judge', 'every judge: script, dialogue, music, clip, image, video'),
               ('script-qa', 'score script drafts with the local rubric'),
               ('script-judge', 'persona prompts + score table')]),
    ('Look after', [('status', 'running and recent jobs, stage progress, GPU'),
                    ('config', 'every VF_* setting: value, source, default; set / unset'),
                    ('thermal', 'CPU / GPU / RAM temperatures; jobs pause at VF_TEMP_MAX (80 C)'),
                    ('doctor', 'interpreters, venvs, GPU, NVENC, ffmpeg, models, disk'),
                    ('gc', 'cache files no latest build uses (--apply: Recycle Bin)'),
                    ('formats', 'output formats'),
                    ('styles', 'visual styles')]),
]


def _console():
    if not RICH:
        return None
    return Console(highlight=False, soft_wrap=False)


CON = _console()


def live():
    """True when progress bars and colour make sense: rich is installed and stdout is a terminal."""
    return bool(CON and CON.is_terminal and os.environ.get('VF_PLAIN') != '1')


def _line(style, mark, msg):
    if CON and live():
        CON.print(f'[{style}]{mark}[/] {msg}')
    else:
        print(f'{mark} {msg}')


def ok(msg):
    _line('green', 'ok  ', msg)


def warn(msg):
    _line('yellow', 'warn', msg)


def fail(msg):
    _line('bold red', 'FAIL', msg)


def info(msg):
    _line('cyan', 'info', msg)


def version():
    import re   # the repo's pyproject.toml first: installed metadata goes stale on a source checkout
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        m = re.search(r'^version\s*=\s*"([^"]+)"', open(os.path.join(root, 'pyproject.toml'), encoding='utf-8').read(), re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version as v
        return v('videoforge')
    except Exception:
        return '?'


def overview():
    """`vf` with no command: what it is, the commands by group, where to look next."""
    if not live():
        print(f'videoforge {version()}: cached local GPU pipeline for AI series episodes')
        for g, cmds in GROUPS:
            print(f'\n{g}:')
            for c, h in cmds:
                print(f'  {c:13s} {h}')
        print('\nvf <command> -h for options.  vf status: jobs.  vf config: settings.')
        return
    CON.print(Panel.fit(f'[bold]videoforge[/] [dim]{version()}[/]\n'
                        '[dim]cached local GPU pipeline: script -> voices -> keyframes -> clips -> checks -> release[/]',
                        border_style='magenta', box=box.ROUNDED))
    for g, cmds in GROUPS:
        t = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
        t.add_column(style='bold cyan', no_wrap=True, min_width=13)
        t.add_column()
        for c, h in cmds:
            t.add_row(c, h)
        CON.print(f'\n[bold magenta]{g}[/]')
        CON.print(t)
    CON.print('\n[dim]vf <command> -h for options   ·   vf status: jobs   ·   vf config: settings[/]')


def config_table(rows, path):
    """rows: [(name, value, source, default, group, help)]."""
    if not live():
        for n, v, s, d, g, h in rows:
            print(f'{n:20s} {v:42s} [{s}]  {h}')
        print(f'file: {path}')
        return
    colour = {'env': 'yellow', 'file': 'green', 'default': 'dim'}
    t = Table(box=box.SIMPLE_HEAD, header_style='bold magenta', expand=False)
    for c in ('setting', 'value', 'from', 'what it sets'):
        t.add_column(c)
    last = None
    for n, v, s, d, g, h in rows:
        if g != last and last is not None:
            t.add_section()
        last = g
        val = f'[{colour[s]}]{v}[/]' + (f'\n[dim]default {d}[/]' if s != 'default' and d else '')
        t.add_row(f'[bold cyan]{n}[/]', val, f'[{colour[s]}]{s}[/]', f'[dim]{g}[/] {h}')
    CON.print(t)
    CON.print(f'[dim]env wins over file, file over default · file: {path}\n'
              f'vf config set VF_ENCODER cpu  ·  vf config unset VF_ENCODER[/]')


@contextlib.contextmanager
def progress(total, label):
    """Progress bar over `total` steps; yields advance(n=1, note=''). Plain output: a no-op."""
    if not live() or total <= 0:
        yield lambda n=1, note='': None
        return
    from rich.progress import (Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
                               MofNCompleteColumn, SpinnerColumn)
    p = Progress(SpinnerColumn(), TextColumn('[bold cyan]{task.description}'), BarColumn(),
                 MofNCompleteColumn(), TimeElapsedColumn(), TextColumn('eta'), TimeRemainingColumn(),
                 TextColumn('[dim]{task.fields[note]}'), console=CON, transient=False)
    with p:
        tid = p.add_task(label, total=total, note='')
        yield lambda n=1, note='': p.update(tid, advance=n, note=note)


def help_formatter():
    """argparse formatter class: colour help with rich-argparse when installed."""
    try:
        from rich_argparse import RichHelpFormatter
        if not live():
            raise ImportError
        RichHelpFormatter.styles['argparse.prog'] = 'bold magenta'
        RichHelpFormatter.styles['argparse.groups'] = 'bold magenta'
        return RichHelpFormatter
    except ImportError:
        import argparse
        return argparse.HelpFormatter


def doctor_line(s):
    """log= for runner.doctor: same text, coloured by its ok / FAIL / info prefix."""
    if not live():
        print(s)
        return
    from rich.markup import escape
    t = s.strip()
    style = ('green' if t.startswith('ok') else 'bold red' if t.startswith('FAIL') else 'cyan' if t.startswith('info')
             else 'bold magenta' if t.startswith(('videoforge', 'doctor')) else None)
    CON.print(f'[{style}]{escape(s)}[/]' if style else escape(s))
