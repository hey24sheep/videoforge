"""Thermal guard: keep CPU, GPU and RAM under a hard limit (default 80 C) with margin for a hot room (45 C ambient).

    vf thermal                 current temperatures, their sensors, the limits
    vf thermal --watch [SEC]   refresh until Ctrl+C
    vf thermal --json

Layers (tools/jobwatch.py runs every job under `Guard`; settings in `vf config`, group `thermal`):
  1. load caps     the job's whole process tree runs at below-normal priority on VF_CPU_CORES logical cores (default a
                   quarter: GPU jobs only feed the card; half let one talk run spike the CPU 69 -> 88 C in 2 s), and the
                   runner sets OMP/MKL/torch thread counts to match, so the CPU never runs flat out. Re-applied every poll, so new children get it too.
  2. cool start    a job starts only when every sensor is at or below VF_TEMP_RESUME.
  (hold) every limit acts only after the reading stays at or above it for VF_TEMP_HOLD s in a row (10; the emergency
                   stop VF_TEMP_CRIT_HOLD, 3): the ACPI zone jumps 15 C for a second or two and settles again.
  3. soft limit    from VF_TEMP_SOFT (72 C) the tree is duty-cycled: suspended for a rest that grows from 0 to 100 %
                   of each poll period as the temperature climbs to the hard limit. Logged as `[thermal] throttle`.
  4. hard limit    reads every VF_TEMP_FAST_POLL s (2) once within 6 C of the soft limit. At VF_TEMP_MAX (80 C) the
                   tree is suspended until every sensor is at or below VF_TEMP_RESUME (70 C):
                   `=== THERMAL pause` / `=== THERMAL resume` (a jobwatch watch event).
                   After a pause the job runs at half speed for VF_TEMP_RECOVER s (60) before full speed.
  5. emergency     at VF_TEMP_CRIT (88 C), or over VF_TEMP_MAX for VF_TEMP_HOT_MIN minutes while paused, the tree is
                   killed: `=== THERMAL KILL` (jobwatch reports FAILED with the cause; builds are cached, rerun when cool).
  6. fail closed   when no sensor can be read for VF_TEMP_BLIND polls in a row, the tree is paused until one reads again.
gpu_lock (vf/gpulock.py) also waits for a cool machine (below VF_TEMP_SOFT) before a GPU step, so runs outside jobwatch
are covered between steps.

Sensors (Windows, no admin rights):
  gpu   nvidia-smi temperature.gpu (and the VRAM temperature when the card reports it)
  cpu   the ACPI thermal zone + VF_TEMP_CPU_OFFSET (default 8 C): the zone reads lower than the cores under load, so it
        is shifted up to stay on the safe side (UNVERIFIED margin: no core sensor to calibrate against without admin)
  ram   no sensor without a driver. Exact CPU core and DIMM temperatures: run LibreHardwareMonitor with its web server on
        (port 8085); VF_TEMP_LHM is read when it answers and replaces the ACPI zone (no offset).
Nothing here changes system, power or driver settings: it only slows, pauses or stops videoforge's own processes.
Standard library only: the runner and the gen venvs import it.
"""
import json, os, re, subprocess, sys, time, urllib.request

NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def limits():
    e = os.environ.get
    mx = float(e('VF_TEMP_MAX', 80))
    soft = min(float(e('VF_TEMP_SOFT', 72)), mx - 1)
    cores = os.cpu_count() or 2
    return dict(on=e('VF_THERMAL', '1') != '0', max=mx, soft=soft,
                resume=min(float(e('VF_TEMP_RESUME', 70)), soft),
                crit=max(float(e('VF_TEMP_CRIT', 88)), mx + 1), hot_min=float(e('VF_TEMP_HOT_MIN', 20)),
                poll=max(2.0, float(e('VF_TEMP_POLL', 10))), blind=int(e('VF_TEMP_BLIND', 3)),
                cpu_offset=float(e('VF_TEMP_CPU_OFFSET', 8)),
                cores=max(2, min(cores, int(e('VF_CPU_CORES', 0)) or cores // 4)),
                recover=float(e('VF_TEMP_RECOVER', 60)),
                hold=max(0.0, float(e('VF_TEMP_HOLD', 10))), crit_hold=max(0.0, float(e('VF_TEMP_CRIT_HOLD', 3))),
                priority=e('VF_JOB_PRIORITY', 'below_normal'))


def _run(argv, timeout=15):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW).stdout
    except Exception:
        return ''


def _gpu():
    out = {}
    q = _run(['nvidia-smi', '--query-gpu=temperature.gpu,temperature.memory', '--format=csv,noheader,nounits'])
    for i, line in enumerate(l for l in q.splitlines() if l.strip()):
        parts = [p.strip() for p in line.split(',')]
        tag = 'gpu' if i == 0 else f'gpu{i}'
        try:
            out[tag] = (float(parts[0]), 'nvidia-smi')
        except (ValueError, IndexError):
            pass
        try:
            out[tag.replace('gpu', 'vram')] = (float(parts[1]), 'nvidia-smi memory')
        except (ValueError, IndexError):
            pass   # consumer cards report [N/A]
    return out


def _acpi_cpu(offset):
    ps = ('Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation | '
          'ForEach-Object { "$($_.Name)=$($_.HighPrecisionTemperature)" }')
    vals = []
    for line in _run(['powershell', '-NoProfile', '-Command', ps]).splitlines():
        m = re.match(r'(.+)=(\d+)$', line.strip())
        if m and int(m.group(2)) > 0:
            vals.append(int(m.group(2)) / 10.0 - 273.15)   # tenths of a kelvin
    if not vals:
        return {}
    return {'cpu': (round(max(vals) + offset, 1), f'ACPI thermal zone {max(vals):.0f} C + {offset:g} C margin')}


def _lhm():
    """LibreHardwareMonitor web server: exact CPU core + DIMM temperatures when it runs."""
    url = os.environ.get('VF_TEMP_LHM', 'http://localhost:8085/data.json')
    if not url or url == '0':
        return {}
    try:
        with urllib.request.urlopen(url, timeout=1.5) as r:
            tree = json.load(r)
    except Exception:
        return {}
    found = {'cpu': [], 'ram': []}

    def walk(n, hw):
        text = str(n.get('Text', ''))
        if re.search(r'ryzen|intel|core i|cpu', text, re.I) and n.get('Children'):
            hw = 'cpu'
        elif re.search(r'dimm|memory|ram', text, re.I) and n.get('Children') and hw != 'cpu':
            hw = 'ram'
        v = re.match(r'\s*([\d.]+)\s*°C', str(n.get('Value', '')))
        if v and hw in found:
            found[hw].append(float(v.group(1)))
        for c in n.get('Children', []) or []:
            walk(c, hw)
    walk(tree, None)
    return {k: (max(v), 'LibreHardwareMonitor') for k, v in found.items() if v}


def read():
    """{sensor: (celsius, source)}; sensors that cannot be read are absent. CPU includes the safety margin."""
    t = _gpu()
    lhm = _lhm()
    if 'cpu' not in lhm:
        t.update(_acpi_cpu(limits()['cpu_offset']))
    t.update(lhm)
    return t


def hottest(temps):
    return max((v[0] for v in temps.values()), default=None)


def hot(temps=None, limit=None):
    """[(sensor, celsius)] at or above the limit."""
    lim = limits()['max'] if limit is None else limit
    temps = read() if temps is None else temps
    return [(k, v[0]) for k, v in sorted(temps.items()) if v[0] >= lim]


def describe(temps):
    return ', '.join(f'{k} {v[0]:.0f} C' for k, v in sorted(temps.items())) or 'no sensors'


def wait_cool(log=None, what='', start_at=None):
    """Block while any sensor is at/above `start_at` (default VF_TEMP_SOFT), or while no sensor reads, until every
    sensor is at/below VF_TEMP_RESUME. Returns seconds waited."""
    L = limits()
    if not L['on']:
        return 0.0
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    gate = L['soft'] if start_at is None else start_at
    t = read()
    if t and not hot(t, gate):
        return 0.0
    t0 = last = time.time()
    log(f"[thermal] {what or 'next step'} waits: {describe(t)} (start below {gate:.0f} C, resume at {L['resume']:.0f} C)")
    while True:
        time.sleep(L['poll'])
        t = read()
        if t and not hot(t, L['resume'] + 1e-6):
            log(f"[thermal] cool again after {time.time() - t0:.0f} s: {describe(t)}")
            return time.time() - t0
        if time.time() - last > 300:
            log(f"[thermal] still waiting ({time.time() - t0:.0f} s): {describe(t)}")
            last = time.time()


# ---------------------------------------------------------------- process tree control (Windows)
def _tree(root):
    """root pid and all its descendants (Toolhelp snapshot; works for DETACHED children, they keep their parent id)."""
    import ctypes
    from ctypes import wintypes

    class PE(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD), ('th32ProcessID', wintypes.DWORD),
                    ('th32DefaultHeapID', ctypes.c_void_p), ('th32ModuleID', wintypes.DWORD),
                    ('cntThreads', wintypes.DWORD), ('th32ParentProcessID', wintypes.DWORD),
                    ('pcPriClassBase', ctypes.c_long), ('dwFlags', wintypes.DWORD), ('szExeFile', ctypes.c_char * 260)]
    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    kids = {}
    try:
        e = PE()
        e.dwSize = ctypes.sizeof(PE)
        ok = k32.Process32First(snap, ctypes.byref(e))
        while ok:
            if e.th32ProcessID != e.th32ParentProcessID:
                kids.setdefault(e.th32ParentProcessID, []).append(e.th32ProcessID)
            ok = k32.Process32Next(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    # Windows reuses pids and keeps a dead parent's pid in its children's records: a new process can get the pid an
    # unrelated older process names as its parent (2026-11-14: the guard counted its own jobwatch runner as a child
    # and suspended itself, freezing the whole job). A real child starts after its parent, and never is this process.
    me = os.getpid()
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        if p in out or p == me:
            continue
        out.append(p)
        born = _born(p)
        todo.extend(c for c in kids.get(p, []) if born is None or (_born(c) or 0) >= born)
    return out


def _born(pid):
    """Process creation time (100 ns ticks), or None when unreadable."""
    import ctypes
    from ctypes import wintypes
    k32 = _api()[0]
    h = k32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return None
    try:
        t = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
            return None
        return (t[0].dwHighDateTime << 32) | t[0].dwLowDateTime
    finally:
        k32.CloseHandle(h)


_API = None


def _api():
    """kernel32 / ntdll with 64-bit handle types. Without argtypes ctypes passes a HANDLE as a 32-bit int: a resume
    could silently fail and leave a job suspended with the GPU idle (2026-11-14: 29 of 31 talk threads stayed
    suspended after a pause)."""
    global _API
    if _API is None:
        import ctypes
        from ctypes import wintypes
        k32, nt = ctypes.windll.kernel32, ctypes.windll.ntdll
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(ctypes.c_size_t)]
        for f in (nt.NtSuspendProcess, nt.NtResumeProcess):
            f.argtypes, f.restype = [wintypes.HANDLE], ctypes.c_long
        _API = (k32, nt)
    return _API


def _each(pids, access, fn):
    k32 = _api()[0]
    done = []
    for pid in pids:
        h = k32.OpenProcess(access, False, int(pid))
        if h:
            try:
                if fn(h):
                    done.append(pid)
            finally:
                k32.CloseHandle(h)
    return done


def suspend_tree(root):
    return _each(_tree(root), 0x0800, lambda h: _api()[1].NtSuspendProcess(h) == 0)


def resume_tree(pids, log=None):
    """Resume every pid suspend_tree returned; retry a failed one, and say which could not be resumed."""
    pids = list(pids)
    done = _each(pids, 0x0800, lambda h: _api()[1].NtResumeProcess(h) == 0)
    for _ in range(3):
        left = [p for p in pids if p not in done and _alive(p)]
        if not left:
            break
        time.sleep(0.2)
        done += _each(left, 0x0800, lambda h: _api()[1].NtResumeProcess(h) == 0)
    left = [p for p in pids if p not in done and _alive(p)]
    if left and log:
        log(f'=== THERMAL resume FAILED for pid(s) {left}: the job may stay suspended')
    return done


def _alive(pid):
    k32 = _api()[0]
    h = k32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    k32.CloseHandle(h)
    return True


def kill_tree(root):
    return _each(list(reversed(_tree(root))), 0x0001, lambda h: _api()[0].TerminateProcess(h, 99) != 0)


PRIORITY = {'idle': 0x40, 'below_normal': 0x4000, 'normal': 0x20}


def cap_tree(root, cores, priority='below_normal', seen=None):
    """Below-normal priority + the first `cores` logical cores for every process of the tree not capped yet."""
    import ctypes
    k32 = _api()[0]
    mask = (1 << cores) - 1
    pri = PRIORITY.get(priority, 0x4000)
    seen = set() if seen is None else seen
    new = [p for p in _tree(root) if p not in seen]

    def cap(h):
        k32.SetPriorityClass(h, pri)
        cur, sysm = ctypes.c_size_t(), ctypes.c_size_t()
        if k32.GetProcessAffinityMask(h, ctypes.byref(cur), ctypes.byref(sysm)) and cur.value and not cur.value & ~mask:
            return True   # already inside the cap (a stage that set fewer cores itself, e.g. the talk stage): keep it
        k32.SetProcessAffinityMask(h, mask)
        return True
    done = _each(new, 0x0200 | 0x0400, cap)   # PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION
    seen.update(new)
    return done


class Guard:
    """Background thread for tools/jobwatch.py: load caps, soft throttle, hard pause, emergency kill, fail closed."""

    def __init__(self, pid, log):
        import threading
        self.pid, self.log, self.stop = pid, log, threading.Event()
        self.paused = None          # (since, pids, reason)
        self.over_since = None      # first time over the hard limit in this pause
        self.blind = 0
        self.capped = set()
        self.killed = None
        self.recover_until = 0.0
        self.th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        L = limits()
        if L['on'] and os.name == 'nt':
            try:
                cap_tree(self.pid, L['cores'], L['priority'], self.capped)
                self.log(f"[thermal] guard on: {L['cores']} of {os.cpu_count()} cores, {L['priority']} priority; "
                         f"acts after {L['hold']:.0f} s high (critical {L['crit_hold']:.0f} s); "
                         f"throttle from {L['soft']:.0f} C, pause at {L['max']:.0f} C, resume at {L['resume']:.0f} C, "
                         f"kill at {L['crit']:.0f} C or {L['hot_min']:.0f} min over {L['max']:.0f} C")
            except Exception as e:
                self.log(f'[thermal] load caps failed: {e}')
            self.th.start()
        return self

    def _pause(self, reason):
        pids = suspend_tree(self.pid)
        self.paused = (time.time(), pids, reason)
        self.log(f"=== THERMAL pause: {reason}; {len(pids)} process(es) suspended")

    def _resume(self, t):
        since, pids, _ = self.paused
        resume_tree(pids, self.log)
        self.paused, self.over_since = None, None
        self.recover_until = time.time() + limits()['recover']   # come back at half speed, not flat out
        self.log(f"=== THERMAL resume after {time.time() - since:.0f} s: {describe(t)}; half speed for "
                 f"{limits()['recover']:.0f} s")

    def _loop(self):
        last_note = throttle_s = 0.0
        since = {}    # level -> first time of the current unbroken run at or above it (spikes shorter than the hold
                      # are ignored: the ACPI zone jumps 15 C for a second or two)

        def held(level, secs, hi, now):
            if hi < level:
                since.pop(level, None)
                return False
            since.setdefault(level, now)
            return now - since[level] >= secs
        while not self.stop.is_set():
            L = limits()
            try:
                cap_tree(self.pid, L['cores'], L['priority'], self.capped)
            except Exception:
                pass
            try:
                t = read()
            except Exception:
                t = {}
            hi = hottest(t)
            now = time.time()
            # 6. fail closed: no sensor -> pause until one reads again
            if hi is None:
                self.blind += 1
                if self.blind >= L['blind'] and self.paused is None:
                    self._pause(f"no temperature sensor readable for {self.blind} polls (fail closed)")
                self.stop.wait(L['poll'])
                continue
            self.blind = 0
            # warm: read every VF_TEMP_FAST_POLL s (a 10 s poll let a CPU spike pass 80 C between two reads)
            fast = hi >= L['soft'] - 6
            poll = min(L['poll'], float(os.environ.get('VF_TEMP_FAST_POLL', 2))) if fast else L['poll']
            crit_held = held(L['crit'], L['crit_hold'], hi, now)
            max_held = held(L['max'], L['hold'], hi, now)
            soft_held = held(L['soft'], L['hold'], hi, now)
            # 5. emergency kill (crit for VF_TEMP_CRIT_HOLD s, or over the limit for VF_TEMP_HOT_MIN while paused)
            if crit_held or (self.over_since and now - self.over_since > L['hot_min'] * 60):
                why = (f"{describe(t)} >= {L['crit']:.0f} C" if hi >= L['crit'] else
                       f"over {L['max']:.0f} C for {(now - self.over_since) / 60:.1f} min while paused ({describe(t)})")
                if self.paused:
                    resume_tree(self.paused[1])   # a suspended process cannot die cleanly on every Windows build
                n = kill_tree(self.pid)
                self.killed = why
                self.log(f"=== THERMAL KILL: {why}; {len(n)} process(es) stopped. Let the machine cool, then rerun "
                         f"(builds are cached)")
                return
            if self.paused:
                last_note = max(last_note, self.paused[0])
                if hi >= L['max'] and self.over_since is None:
                    self.over_since = now
                if hi <= L['resume']:
                    self._resume(t)
                elif now - last_note > 300:
                    last_note = now
                    self.log(f"=== THERMAL still paused ({now - self.paused[0]:.0f} s): {describe(t)}")
                self.stop.wait(poll)
                continue
            # 4. hard limit: at or above VF_TEMP_MAX for VF_TEMP_HOLD s in a row
            if max_held:
                self.over_since = now
                self._pause(f"{', '.join(f'{k} {c:.0f} C' for k, c in hot(t, L['max']))} >= {L['max']:.0f} C for "
                            f"{now - since[L['max']]:.0f} s ({describe(t)}), resume at {L['resume']:.0f} C")
                self.stop.wait(poll)
                continue
            # 3. soft limit: duty cycle, rest share grows from 0 to 1 between soft and max
            recovering = now < self.recover_until
            if soft_held or recovering:
                share = min(1.0, max(0.5 if recovering else 0.0,
                                     (hi - L['soft']) / max(1.0, L['max'] - L['soft']) + 0.2))
                rest = poll * share
                pids = suspend_tree(self.pid)
                self.stop.wait(rest)
                resume_tree(pids, self.log)
                throttle_s += rest
                if now - last_note > 120:
                    last_note = now
                    self.log(f"[thermal] throttle: {describe(t)}, resting {share * 100:.0f} % "
                             f"({throttle_s:.0f} s of rest so far)")
                self.stop.wait(max(0.0, poll - rest))
                continue
            self.stop.wait(poll)

    def close(self):
        self.stop.set()
        if self.paused:
            resume_tree(self.paused[1])
            self.paused = None


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='vf thermal', description='CPU / GPU / RAM temperatures and the thermal limits')
    ap.add_argument('--watch', nargs='?', type=float, const=5, default=None, metavar='SECONDS')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    while True:
        t, L = read(), limits()
        if a.json:
            print(json.dumps(dict(limits=L, temps={k: dict(c=v[0], source=v[1]) for k, v in t.items()}), indent=1))
            return 0
        print(f"thermal guard {'on' if L['on'] else 'OFF'}: throttle {L['soft']:.0f} C, pause {L['max']:.0f} C, "
              f"resume {L['resume']:.0f} C, kill {L['crit']:.0f} C; jobs use {L['cores']} of {os.cpu_count()} cores")
        for k in ('cpu', 'gpu', 'vram', 'ram'):
            if k in t:
                c, src = t[k]
                flag = 'HOT ' if c >= L['max'] else 'warm' if c >= L['soft'] else 'ok  '
                print(f'  {flag} {k:5s} {c:5.1f} C   {src}')
            elif k in ('cpu', 'gpu', 'ram'):
                print(f'  --   {k:5s}   n/a    no readable sensor' + (' (LibreHardwareMonitor web server gives RAM)'
                                                                       if k == 'ram' else ''))
        if a.watch is None:
            return 1 if hot(t, L['max']) else 0
        time.sleep(a.watch)
        print()


if __name__ == '__main__':
    sys.exit(main())
