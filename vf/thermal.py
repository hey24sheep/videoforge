"""Thermal guard: CPU, GPU and RAM temperatures must stay under a limit (default 80 C) while videoforge works.

    vf thermal                 current temperatures, their sensors, the limit
    vf thermal --watch [SEC]   refresh until Ctrl+C
    vf thermal --json

Two guards use it:
  * tools/jobwatch.py run: the runner polls every VF_TEMP_POLL s. When any sensor reaches VF_TEMP_MAX it SUSPENDS the
    job's whole process tree (every child: model workers, ffmpeg) and logs `=== THERMAL pause: ...`; it resumes when
    every sensor is back at VF_TEMP_RESUME or below (`=== THERMAL resume ...`). This stops the heat mid-run, not only
    between steps. A pause counts against child timeouts (VF_GEN_TIMEOUT), so long pauses can time a model run out.
  * gpu_lock (vf/gpulock.py): a step that takes the GPU first waits until the machine is cool (runs outside jobwatch too).

Sensors (Windows, no admin rights):
  gpu   nvidia-smi temperature.gpu (and the VRAM temperature when the card reports it)
  cpu   the ACPI thermal zones (Win32_PerfFormattedData_Counters_ThermalZoneInformation). On laptops this is the
        board/CPU zone, not the exact core temperature: it reads lower than the core under load.
  ram   no sensor without a driver. Exact CPU core and RAM (DIMM) temperatures: run LibreHardwareMonitor with its
        web server on (Options > Remote Web Server, port 8085); VF_TEMP_LHM (default http://localhost:8085/data.json)
        is read when it answers and replaces the ACPI zone.
A sensor that cannot be read is reported as missing, never as cool.

Settings (vf config): VF_THERMAL (1 = on), VF_TEMP_MAX (80), VF_TEMP_RESUME (70), VF_TEMP_POLL (10 s).
Standard library only: the runner and the gen venvs import it.
"""
import json, os, re, subprocess, sys, time, urllib.request

NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def limits():
    mx = float(os.environ.get('VF_TEMP_MAX', 80))
    return dict(on=os.environ.get('VF_THERMAL', '1') != '0', max=mx,
                resume=min(float(os.environ.get('VF_TEMP_RESUME', 70)), mx - 1),
                poll=max(2.0, float(os.environ.get('VF_TEMP_POLL', 10))))


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


def _acpi_cpu():
    ps = ('Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation | '
          'ForEach-Object { "$($_.Name)=$($_.HighPrecisionTemperature)" }')
    vals = []
    for line in _run(['powershell', '-NoProfile', '-Command', ps]).splitlines():
        m = re.match(r'(.+)=(\d+)$', line.strip())
        if m and int(m.group(2)) > 0:
            vals.append(int(m.group(2)) / 10.0 - 273.15)   # tenths of a kelvin
    return {'cpu': (round(max(vals), 1), 'ACPI thermal zone (board/CPU, not the core)')} if vals else {}


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
    """{sensor: (celsius, source)}; sensors that cannot be read are absent."""
    t = _gpu()
    lhm = _lhm()
    if 'cpu' not in lhm:
        t.update(_acpi_cpu())
    t.update(lhm)
    return t


def hot(temps=None, limit=None):
    """[(sensor, celsius)] at or above the limit."""
    lim = limits()['max'] if limit is None else limit
    temps = read() if temps is None else temps
    return [(k, v[0]) for k, v in sorted(temps.items()) if v[0] >= lim]


def describe(temps):
    return ', '.join(f'{k} {v[0]:.0f} C' for k, v in sorted(temps.items())) or 'no sensors'


def wait_cool(log=None, what=''):
    """Block while any sensor is at/above VF_TEMP_MAX, until all are at/below VF_TEMP_RESUME. Returns seconds waited."""
    L = limits()
    if not L['on']:
        return 0.0
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    t = read()
    h = hot(t, L['max'])
    if not h:
        return 0.0
    t0 = time.time()
    log(f"[thermal] {what or 'next step'} waits: {describe(t)} (limit {L['max']:.0f} C, resume at {L['resume']:.0f} C)")
    last = t0
    while True:
        time.sleep(L['poll'])
        t = read()
        if not hot(t, L['resume'] + 1e-6):
            log(f"[thermal] cool again after {time.time() - t0:.0f} s: {describe(t)}")
            return time.time() - t0
        if time.time() - last > 300:
            log(f"[thermal] still waiting ({time.time() - t0:.0f} s): {describe(t)}")
            last = time.time()


# ---------------------------------------------------------------- suspend / resume a process tree (Windows)
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
            kids.setdefault(e.th32ParentProcessID, []).append(e.th32ProcessID)
            ok = k32.Process32Next(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def _nt(pids, fn):
    import ctypes
    k32, nt = ctypes.windll.kernel32, ctypes.windll.ntdll
    done = []
    for pid in pids:
        h = k32.OpenProcess(0x0800, False, int(pid))   # PROCESS_SUSPEND_RESUME
        if h:
            if getattr(nt, fn)(h) == 0:
                done.append(pid)
            k32.CloseHandle(h)
    return done


def suspend_tree(root):
    return _nt(_tree(root), 'NtSuspendProcess')


def resume_tree(pids):
    return _nt(pids, 'NtResumeProcess')


class Guard:
    """Background thread for tools/jobwatch.py: suspend the job's process tree while too hot."""

    def __init__(self, pid, log):
        import threading
        self.pid, self.log, self.paused, self.stop = pid, log, None, threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if limits()['on'] and os.name == 'nt':
            self.th.start()
        return self

    def _loop(self):
        t_last = 0.0
        while not self.stop.wait(limits()['poll']):
            L = limits()
            try:
                t = read()
            except Exception:
                continue
            if self.paused is None:
                h = hot(t, L['max'])
                if h:
                    pids = suspend_tree(self.pid)
                    self.paused = (time.time(), pids)
                    t_last = time.time()
                    self.log(f"=== THERMAL pause: {', '.join(f'{k} {c:.0f} C' for k, c in h)} >= {L['max']:.0f} C "
                             f"({describe(t)}); {len(pids)} process(es) suspended until <= {L['resume']:.0f} C")
            elif not hot(t, L['resume'] + 1e-6):
                t0, pids = self.paused
                resume_tree(pids)
                self.paused = None
                self.log(f"=== THERMAL resume after {time.time() - t0:.0f} s: {describe(t)}")
            elif time.time() - t_last > 300:
                t_last = time.time()
                self.log(f"=== THERMAL still paused ({time.time() - self.paused[0]:.0f} s): {describe(t)}")

    def close(self):
        self.stop.set()
        if self.paused:
            resume_tree(self.paused[1])
            self.paused = None


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='vf thermal', description='CPU / GPU / RAM temperatures and the thermal limit')
    ap.add_argument('--watch', nargs='?', type=float, const=5, default=None, metavar='SECONDS')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)
    while True:
        t, L = read(), limits()
        if a.json:
            print(json.dumps(dict(limits=L, temps={k: dict(c=v[0], source=v[1]) for k, v in t.items()}), indent=1))
            return 0
        print(f"thermal guard {'on' if L['on'] else 'OFF'}: pause at {L['max']:.0f} C, resume at {L['resume']:.0f} C")
        for k in ('cpu', 'gpu', 'vram', 'ram'):
            if k in t:
                c, src = t[k]
                flag = 'HOT ' if c >= L['max'] else 'ok  '
                print(f'  {flag}{k:5s} {c:5.1f} C   {src}')
            elif k in ('cpu', 'gpu', 'ram'):
                print(f'  --  {k:5s}   n/a    no readable sensor' + (' (LibreHardwareMonitor web server gives RAM)'
                                                                      if k == 'ram' else ''))
        if a.watch is None:
            return 1 if hot(t, L['max']) else 0
        time.sleep(a.watch)
        print()


if __name__ == '__main__':
    sys.exit(main())
