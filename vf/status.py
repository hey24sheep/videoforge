"""vf status: running and recent jobs (tools/jobwatch.py records), their `vf run` stage, last log line, and the GPU.

    vf status [--hours 24] [--all] [--watch [SECONDS]] [--json]
"""
import importlib.util, json, os, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = re.compile(r'^=== \[(\d+)/(\d+)\] (\S+)', re.M)


def _jobwatch():
    spec = importlib.util.spec_from_file_location('jobwatch', os.path.join(ROOT, 'tools', 'jobwatch.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _gpu():
    import subprocess
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15).stdout
        n, u, used, tot, t = [x.strip() for x in q.strip().splitlines()[0].split(',')]
        return dict(name=n, util=int(u), used_mb=int(used), total_mb=int(tot), temp=int(t))
    except Exception:
        return None


def _ago(sec):
    sec = max(0, int(sec))
    return f'{sec // 3600}h{sec % 3600 // 60:02d}m' if sec >= 3600 else f'{sec // 60}m{sec % 60:02d}s'


def collect(hours=24, show_all=False):
    jw = _jobwatch()
    rows = []
    for j in reversed(jw._recent(hours)):
        c = jw.classify(j, 12)
        if not show_all and c['status'] == 'ok' and len(rows) >= 8:
            continue
        tail = jw._tail(j['log'], 200000)
        st = STAGE.findall(tail)
        stage = dict(i=int(st[-1][0]), n=int(st[-1][1]), name=st[-1][2]) if st else None
        end = j.get('ended') or time.time()
        rows.append(dict(id=j['id'], name=j.get('name', ''), status=c['status'], stage=stage,
                         elapsed=end - j.get('started', end), log_age_min=c['log_age_min'],
                         last=(c['last'][-1] if c['last'] else '').strip(), cause=(c['cause'] or {}).get('label'),
                         log=j['log']))
    return dict(jobs=rows, gpu=_gpu())


COLOUR = {'running': 'green', 'ok': 'dim green', 'STALLED': 'bold yellow', 'FAILED': 'bold red', 'KILLED': 'red'}


def render(d):
    from . import ui
    g = d['gpu']
    if not ui.live():
        if g:
            print(f"GPU {g['name']}: {g['util']}% busy, {g['used_mb']}/{g['total_mb']} MB, {g['temp']} C")
        for r in d['jobs']:
            s = f"{r['stage']['i']}/{r['stage']['n']} {r['stage']['name']}" if r['stage'] else '-'
            print(f"[{r['status']}] {r['id']}  stage {s}  {_ago(r['elapsed'])}  {r['cause'] or r['last'][:100]}")
        if not d['jobs']:
            print('no jobs in this window')
        return
    from rich.table import Table
    from rich.progress_bar import ProgressBar
    from rich.console import Group
    from rich import box
    parts = []
    if g:
        bar = ProgressBar(total=g['total_mb'], completed=g['used_mb'], width=24)
        t = Table.grid(padding=(0, 1))
        t.add_row(f"[bold]GPU[/] {g['name']}", f"[cyan]{g['util']}%[/] busy", 'VRAM', bar,
                  f"{g['used_mb'] / 1024:.1f}/{g['total_mb'] / 1024:.1f} GB", f"{g['temp']} °C")
        parts.append(t)
    t = Table(box=box.SIMPLE_HEAD, header_style='bold magenta')
    for c in ('job', 'status', 'stage', 'time', 'last line'):
        t.add_column(c, no_wrap=c != 'last line', overflow='fold' if c == 'last line' else 'ellipsis')
    for r in d['jobs']:
        if r['stage']:
            s = r['stage']
            done = s['i'] - (1 if r['status'] == 'running' else 0)
            stage = f"{'█' * done}{'░' * (s['n'] - done)} {s['i']}/{s['n']} {s['name']}"
        else:
            stage = '[dim]-[/]'
        col = COLOUR.get(r['status'], 'white')
        quiet = f" [dim](quiet {r['log_age_min']:.0f} min)[/]" if r['status'] in ('running', 'STALLED') and r[
            'log_age_min'] else ''
        t.add_row(f"[bold]{r['name'] or r['id']}[/]\n[dim]{r['id']}[/]", f'[{col}]{r["status"]}[/]', stage,
                  _ago(r['elapsed']) + quiet, f"[red]{r['cause']}[/]" if r['cause'] else f"[dim]{r['last'][:160]}[/]")
    parts.append(t if d['jobs'] else '[dim]no jobs in this window[/]')
    ui.CON.print(Group(*parts))


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='vf status', description='running and recent jobs, stage progress, GPU')
    ap.add_argument('--hours', type=float, default=24)
    ap.add_argument('--all', action='store_true', help='every job in the window (default: running/failed + 8 newest)')
    ap.add_argument('--watch', nargs='?', type=float, const=5, default=None, metavar='SECONDS',
                    help='refresh until Ctrl+C (default every 5 s)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    if a.json:
        print(json.dumps(collect(a.hours, a.all), indent=1, default=str))
        return 0
    from . import ui
    if a.watch and ui.live():
        from rich.live import Live
        try:
            with Live(console=ui.CON, auto_refresh=False) as lv:
                while True:
                    with ui.CON.capture() as cap:
                        render(collect(a.hours, a.all))
                    from rich.text import Text
                    lv.update(Text.from_ansi(cap.get() + f'\nrefresh every {a.watch:g} s · Ctrl+C to stop'),
                              refresh=True)
                    time.sleep(a.watch)
        except KeyboardInterrupt:
            return 0
    render(collect(a.hours, a.all))
    return 0
