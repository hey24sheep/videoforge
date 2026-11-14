"""Long-job runner + restart checker for videoforge (stdlib only, Windows).

Why (2026-11-06): builds were started from the Claude Code session's shell. When that session's process
exited at 18:34:51, Windows killed the whole process tree: ffmpeg died first, the K6 clip writer hit
`OSError: [Errno 22] Invalid argument` on the dead pipe, and the build crashed with nothing left to restart it.
2.7 h passed before anyone noticed.

  run     start a command fully detached (own process group, breaks away from the caller's job object), log to a file,
          record pid/exit code in jobs/<id>.json. The job survives the session, the terminal and the app closing.
  check   report every recent job: running / STALLED / ok / FAILED (with the known cause and the fix) / KILLED (died
          without an exit code: session end, reboot, user). Prints the exact resume command. Exit 1 when something
          needs action. Run by the SessionStart hook (startup, resume, compact) in ~/.claude/settings.json.
  restart re-launch KILLED or transient-FAILED resumable jobs (builds are cache-backed, so a rerun only redoes the
          missing work), only when the GPU is clear and no other job is running.

  python tools/jobwatch.py run --name ep02_build --resumable -- python -m vf build series/.../episode.yaml --formats short
  python tools/jobwatch.py check [--hours 48] [--stall-min 12]
  python tools/jobwatch.py restart [--id ID]
  python tools/jobwatch.py watch --id ID   (run in the background after every launch: exits with one EVENT line on an
          error, abort, FLAG, QC warning, stage milestone, deadlock, stall (log quiet + GPU idle 5 min), end, or the
          30-min heartbeat; the session handles it, tells the owner, and starts the next watch)
"""
import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS = os.path.join(ROOT, 'jobs')
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000

# (regex on the log tail, label, transient (safe to just rerun), fix hint)
CAUSES = [
    (r'OSError: \[Errno 22\] Invalid argument[\s\S]{0,400}|stdin\.write[\s\S]{0,200}Errno 22|BrokenPipeError',
     'encoder pipe died (ffmpeg was killed or crashed mid-clip)', True,
     'rerun: finished clips are cached. If no session ended at that time, read the ffmpeg log next to the clip.'),
    (r'did not fit in the VRAM cap|does not fit in the [0-9.]+ GB sampling',
     'clip too big for the 8 GB card (576x1024 holds ~93 frames)', False,
     'lower gen.speed so dur*speed <= 3.35 s, or a smaller k6_size, then rerun'),
    (r'EncodeRefused[^\n]*RAM free',
     'not enough free RAM for the text encoder (an idle gen worker holds ~8 GB)', True,
     'stop the gen worker (python -c "from vf.gen.bridge import stop_gen; stop_gen()") and rerun'),
    (r'EncodeRefused', 'new prompts need the text encoder', False, 'rerun with --allow-encode'),
    (r'OutOfMemoryError|CUDA out of memory', 'GPU out of memory (another process on the card?)', True,
     'check nvidia-smi for other GPU users, then rerun'),
    (r'\[talk\] ABORT', 'talk_video RAM guard aborted the InfiniteTalk run', True,
     'check what else holds RAM (gen worker, other python), then rerun'),
    (r'=== THERMAL KILL', 'thermal emergency stop (too hot, or no working thermal guard)', True,
     'let the machine cool (vf thermal), improve airflow, then rerun: finished work is cached'),
    (r'TimeoutExpired', 'a child process timed out', False, 'read the log: a model run hung; check GPU temp/driver'),
    (r'K6 encode pass failed|k6 encode pass failed', 'K6 prompt-encode pass failed', False, 'read the log above it'),
]


def _alive(pid):
    if not pid:
        return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def _load(path):
    try:
        return json.load(open(path, encoding='utf-8'))
    except Exception:
        return None


def _save(job):
    os.makedirs(JOBS, exist_ok=True)
    tmp = os.path.join(JOBS, job['id'] + '.json.tmp')
    json.dump(job, open(tmp, 'w', encoding='utf-8'), indent=1)
    os.replace(tmp, os.path.join(JOBS, job['id'] + '.json'))


def _tail(path, n=40000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode('utf-8', 'replace').replace('\r', '\n')
    except OSError:
        return ''


def _spawn(argv, log=None):
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    out = open(log, 'ab') if log else subprocess.DEVNULL
    kw = dict(stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, close_fds=True, cwd=ROOT)
    try:  # leave the caller's job object so its teardown cannot kill us
        return subprocess.Popen(argv, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **kw)
    except OSError:  # the caller's job forbids breakaway: still detached from the console/process group
        return subprocess.Popen(argv, creationflags=flags, **kw)


def cmd_run(a):
    argv = a.argv[1:] if a.argv and a.argv[0] == '--' else a.argv
    if not argv:
        sys.exit('nothing to run')
    jid = time.strftime('%Y%m%d_%H%M%S') + '_' + re.sub(r'[^A-Za-z0-9_-]', '_', a.name)
    log = os.path.abspath(a.log or os.path.join(JOBS, jid + '.log'))
    env = {k: v for k, v in (e.split('=', 1) for e in a.env)}
    job = dict(id=jid, name=a.name, argv=argv, cwd=ROOT, log=log, env=env, resumable=a.resumable,
               started=time.time(), status='starting', runner_pid=None, pid=None, rc=None, ended=None, restarts=0)
    _save(job)
    p = _spawn([sys.executable, os.path.abspath(__file__), '_runner', jid])
    job['runner_pid'] = p.pid
    _save(job)
    print(f'started {jid} (runner pid {p.pid})\n  log: {log}\n  check: python tools/jobwatch.py check')


def _vf_module(name):
    """Load vf/<name>.py by path (no package import: the runner stays light and stdlib-only)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('vf_' + name, os.path.join(ROOT, 'vf', name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cmd_runner(a):
    """Detached supervisor: run the job, wait, record the exit code (a killed runner leaves rc None = KILLED)."""
    path = os.path.join(JOBS, a.id + '.json')
    job = _load(path)
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env.update(job.get('env', {}))  # was dict(..., PYTHONUNBUFFERED='1', **env): TypeError when env also set it
    try:   # vf.config.json settings (VF_TEMP_MAX ...) for the thermal guard below; the child applies them itself
        os.environ.update(job.get('env', {}))
        _vf_module('config').apply()
    except Exception:
        pass
    with open(job['log'], 'ab') as out:
        out.write(f"\n=== jobwatch start {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(job['argv'])}\n".encode())
        out.flush()

        def note(s):
            out.write(('\n' + s + '\n').encode())
            out.flush()
        # thermal guard (vf/thermal.py): load caps, throttle, pause, emergency kill. Fail closed: no guard, no job
        # (unless VF_THERMAL=0 says so on purpose).
        thermal = None
        if os.environ.get('VF_THERMAL', '1') != '0':
            try:
                thermal = _vf_module('thermal')
            except Exception as e:
                note(f'=== THERMAL KILL: guard could not load ({e}); job not started (VF_THERMAL=0 runs unguarded)')
                job.update(rc=98, ended=time.time(), status='failed')
                _save(job)
                return
            thermal.wait_cool(log=note, what='job start')   # 2. start only on a cool machine
        p = subprocess.Popen(job['argv'], cwd=job['cwd'], env=env, stdin=subprocess.DEVNULL, stdout=out,
                             stderr=subprocess.STDOUT)
        job.update(pid=p.pid, status='running')
        _save(job)
        guard = None
        if thermal:
            try:
                guard = thermal.Guard(p.pid, note).start()
            except Exception as e:   # fail closed: an unguarded job is stopped
                note(f'=== THERMAL KILL: guard failed to start ({e}); job stopped')
                p.kill()
        try:
            rc = p.wait()
        finally:
            if guard:
                guard.close()
        out.write(f"\n=== jobwatch end {time.strftime('%Y-%m-%d %H:%M:%S')} rc={rc}\n".encode())
    job = _load(path) or job
    job.update(rc=rc, ended=time.time(), status='ok' if rc == 0 else 'failed')
    _save(job)


def _boot_time():
    """Unix time of the last Windows boot."""
    ctypes.windll.kernel32.GetTickCount64.restype = ctypes.c_ulonglong
    return time.time() - ctypes.windll.kernel32.GetTickCount64() / 1000.0


def classify(job, stall_min):
    # a job started before the last boot cannot be alive: after a reboot Windows hands its old pids to other
    # processes (2026-11-08: audiodg/svchost made two dead jobs look 'running' and blocked the restart)
    # a recorded exit code + end time is final: Windows reuses pids, so a new process could make it look alive
    finished = job.get('rc') is not None and job.get('ended')
    alive = not finished and job.get('started', 0) > _boot_time() and (
        _alive(job.get('pid')) or _alive(job.get('runner_pid')))
    tail = _tail(job['log'])
    age = (time.time() - os.path.getmtime(job['log'])) / 60 if os.path.exists(job['log']) else None
    cause = None
    if job.get('rc') not in (None, 0) or (not alive and job.get('rc') is None):
        for rx, label, transient, hint in CAUSES:
            if re.search(rx, tail):
                cause = dict(label=label, transient=transient, hint=hint)
                break
    if alive:
        st = 'STALLED' if age is not None and age > stall_min else 'running'
    elif job.get('rc') == 0:
        st = 'ok'
    elif job.get('rc') is None:
        st = 'KILLED'  # no exit code recorded: the runner itself died (session end, reboot, user)
    else:
        st = 'FAILED'
    last = [ln for ln in tail.splitlines() if ln.strip() and 'it/s]' not in ln][-3:]
    return dict(status=st, alive=alive, log_age_min=round(age, 1) if age is not None else None, cause=cause, last=last)


def _recent(hours):
    if not os.path.isdir(JOBS):
        return []
    out = []
    for f in sorted(os.listdir(JOBS)):
        if f.endswith('.json'):
            j = _load(os.path.join(JOBS, f))
            if j and time.time() - j.get('started', 0) < hours * 3600:
                out.append(j)
    return out


def _orphan_builds():
    """vf builds not started through jobwatch (e.g. from a shell): list them so the checker sees every heavy job."""
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match ' -m vf (build|keyframes|voices)' } "
                            "| ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"],
                           capture_output=True, text=True, timeout=20)
        return [ln for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def cmd_check(a):
    jobs = _recent(a.hours)
    need = 0
    print(f'== jobwatch: {len(jobs)} job(s) in the last {a.hours} h ==')
    for j in jobs:
        c = classify(j, a.stall_min)
        tag = c['status']
        print(f"[{tag}] {j['id']}  log {j['log']} (last write {c['log_age_min']} min ago)")
        if tag in ('STALLED', 'FAILED', 'KILLED'):
            need += 1
            if c['cause']:
                print(f"   cause: {c['cause']['label']}\n   fix:   {c['cause']['hint']}")
            elif tag == 'KILLED':
                print('   cause: died without an exit code (session/app exit, reboot or a manual kill)')
            elif tag == 'STALLED':
                print(f'   no log output for {c["log_age_min"]} min: check GPU (nvidia-smi), temps, newest cache files')
            for ln in c['last']:
                print('   | ' + ln[:200])
            if j.get('resumable') and tag != 'STALLED':
                print('   RESTART NEEDED: python tools/jobwatch.py restart --id ' + j['id'])
    others = _orphan_builds()
    known = {str(j.get('pid')) for j in jobs}
    others = [o for o in others if o.split('\t')[0] not in known and 'jobwatch' not in o]
    if others:
        print('vf jobs running outside jobwatch (they die if the session that started them ends):')
        for o in others:
            print('   ' + o[:200])
    if not jobs and not others:
        print('nothing running, nothing to restart')
    sys.exit(1 if need else 0)


def _gpu_used_mib():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def cmd_restart(a):
    jobs = _recent(a.hours)
    if any(classify(j, 10 ** 9)['alive'] for j in jobs):
        sys.exit('a job is still running: not restarting anything (one heavy GPU job at a time)')
    gpu = _gpu_used_mib()
    if gpu is None or gpu > 600:
        sys.exit(f'GPU not clear ({gpu} MiB used): not restarting')
    for j in jobs:
        if a.id and j['id'] != a.id:
            continue
        c = classify(j, 10 ** 9)
        ok = c['status'] == 'KILLED' or (c['status'] == 'FAILED' and c['cause'] and c['cause']['transient'])
        if not (j.get('resumable') and ok):
            continue
        if c['cause'] and 'gen worker' in c['cause']['hint']:
            subprocess.run([sys.executable, '-c', 'from vf.gen.bridge import stop_gen; stop_gen()'], cwd=ROOT)
        j.update(status='restarting', restarts=j.get('restarts', 0) + 1, rc=None, ended=None, started=time.time())
        _save(j)
        p = _spawn([sys.executable, os.path.abspath(__file__), '_runner', j['id']])
        j['runner_pid'] = p.pid
        _save(j)
        print(f"restarted {j['id']} (runner pid {p.pid}); log {j['log']}")
        return
    print('nothing to restart')


EVENTS = [  # (regex on NEW log lines, kind) - the first match ends `watch` so the session wakes up and acts
    (r'Traceback \(most recent call last\)', 'ERROR'),
    (r'\[talk\] ABORT|ABORT:', 'ABORT'),
    (r'=== THERMAL (KILL|still paused)', 'THERMAL'),   # routine pause/resume lines are not events
    (r'\bFLAG\b', 'FLAG'),
    (r'QC (error|warning)', 'QC'),
    (r'voicedir: acted|talk: \d+ shot\(s\) in|k6: generating|ai stage:|^short: ', 'MILESTONE'),
]


def _gpu_util():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _parents(pid):
    """pid and its ancestors (Windows), to spot a child waiting on a lock its own parent holds."""
    out = set()
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command',
                            'Get-CimInstance Win32_Process | ForEach-Object { "$($_.ProcessId) $($_.ParentProcessId)" }'],
                           capture_output=True, text=True, timeout=30)
        par = dict(tuple(map(int, l.split())) for l in r.stdout.splitlines() if l.strip())
        while pid and pid not in out:
            out.add(pid)
            pid = par.get(pid)
    except Exception:
        pass
    return out


def cmd_watch(a):
    """Block until something needs the session: job ended, error/abort/flag/QC line, a stage milestone, a deadlock
    (waiting for a GPU lock held by its own ancestor), a stall (log quiet AND GPU idle), or the heartbeat. Prints one
    'EVENT <kind>: ...' line and exits. Added 2026-11-07: the owner had to tell me twice that the build was stuck."""
    path = os.path.join(JOBS, a.id + '.json')
    j = _load(path)
    off = j.get('watch_offset', 0) if not a.from_start else 0
    t0 = time.time()
    lock_seen = None

    def out(kind, msg):
        j2 = _load(path) or j
        if os.path.exists(j2['log']):
            j2['watch_offset'] = os.path.getsize(j2['log'])
        _save(j2)
        print(f'EVENT {kind}: {msg}')
        sys.exit(0)
    while True:
        j = _load(path) or j
        log = j['log']
        if os.path.exists(log):
            with open(log, 'rb') as f:
                f.seek(off)
                new = f.read().decode('utf-8', 'replace')
            hits = []
            for raw in new.replace('\r', '\n').splitlines():
                for rx, kind in EVENTS:
                    if re.search(rx, raw):
                        hits.append((kind, raw.strip()))
                        break
            # a FLAG line the build prints again at every stage (cast table, voice report) is reported once per job
            seen_flags = set(j.get('seen_flags', []))
            hits = [(k, l) for k, l in hits if k != 'FLAG' or re.sub(r'\s+', ' ', l) not in seen_flags]
            j['seen_flags'] = sorted(seen_flags | {re.sub(r'\s+', ' ', l) for k, l in hits if k == 'FLAG'})[-500:]
            if hits:   # report EVERY event in the new output (the first version reported one and skipped the rest)
                j['watch_offset'] = off + len(new.encode('utf-8', 'replace'))
                _save(j)
                order = [k for _, k in EVENTS]
                kinds = sorted({k for k, _ in hits}, key=order.index)
                print(f'EVENT {kinds[0]}: {len(hits)} event(s) ({", ".join(kinds)})')
                seen = set()
                for k, l in hits:
                    if l not in seen:
                        seen.add(l)
                        print(f'   {k:9s} {l[:300]}')
                sys.exit(0)
            off += len(new.encode('utf-8', 'replace'))
            age = (time.time() - os.path.getmtime(log)) / 60
            last = [l for l in _tail(log, 4000).replace('\r', '\n').splitlines() if l.strip()][-1:] or ['']
            m = re.search(r'\[gpu-lock\] waiting for pid (\d+)', last[0])
            if m and age > 1.0:
                if lock_seen is None:
                    lock_seen = _parents(j.get('pid'))
                if int(m.group(1)) in lock_seen:
                    out('DEADLOCK', f'waits for GPU lock held by its own parent pid {m.group(1)}: {last[0][:200]}')
            if age > a.stall_min and (_gpu_util() or 0) < 10:
                out('STALL', f'no log output for {age:.0f} min and GPU idle: {last[0][:200]}')
        c = classify(j, 10 ** 9)
        if not c['alive']:
            out('DONE', f"{c['status']} rc={j.get('rc')}" + (f" cause: {c['cause']['label']}" if c['cause'] else ''))
        if a.heartbeat and time.time() - t0 > a.heartbeat * 60:
            last = [l for l in _tail(j['log'], 6000).replace('\r', '\n').splitlines() if l.strip()][-2:]
            out('HEARTBEAT', ' | '.join(l.strip()[:160] for l in last))
        time.sleep(15)


def main():
    try:   # log lines carry tqdm bar characters; the Windows console codec (cp1252) crashed `watch` on them
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog='jobwatch')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--name', required=True)
    r.add_argument('--log', default=None)
    r.add_argument('--env', action='append', default=[], help='KEY=VALUE (repeatable)')
    r.add_argument('--resumable', action='store_true', help='safe to rerun after a crash (cache-backed vf builds)')
    r.add_argument('argv', nargs=argparse.REMAINDER)
    rn = sub.add_parser('_runner')
    rn.add_argument('id')
    c = sub.add_parser('check')
    c.add_argument('--hours', type=float, default=48)
    c.add_argument('--stall-min', type=float, default=12)
    rs = sub.add_parser('restart')
    rs.add_argument('--id', default=None)
    rs.add_argument('--hours', type=float, default=48)
    w = sub.add_parser('watch')
    w.add_argument('--id', required=True)
    w.add_argument('--stall-min', type=float, default=5)
    w.add_argument('--heartbeat', type=float, default=30, help='minutes; 0 = off')
    w.add_argument('--from-start', action='store_true')
    a = ap.parse_args()
    dict(run=cmd_run, _runner=cmd_runner, check=cmd_check, restart=cmd_restart, watch=cmd_watch)[a.cmd](a)


if __name__ == '__main__':
    main()
