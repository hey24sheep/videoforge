"""Call the torch-side generators (vf.gen.<module>) from the main, torch-free interpreter.

    from vf.gen.bridge import run_gen, stop_gen
    res = run_gen('video', [dict(image=r'C:\\abs\\a.png', prompt='slow push-in, hair moving in the wind', seed=1)])
    res[0]['out'], res[0]['fps']

By default jobs go to a persistent gen worker (vf.gen.server, started on demand) so models load once per session
instead of once per call. `persistent=False` (or VF_GEN_PERSIST=0) runs a one-shot child process instead.
  park_gen()  : free the worker's VRAM now (weights wait in CPU RAM; the next job resumes in ~1 s). Use before GPU
                rendering / TTS. The worker also parks itself after VF_GEN_PARK_IDLE s idle (default 120) and exits
                after VF_GEN_IDLE s (default 3600).
  stop_gen()  : end the worker (frees its RAM too; the next job pays ~15 s imports + ~20 s per model load).
  gen_status(): resident models, RSS, VRAM (None if the worker isn't running).
"""
import json, os, socket, subprocess, sys, tempfile, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN_PY = os.environ.get('VF_GEN_PYTHON', os.path.join(_AI, 'tools', 'envs', 'gen', 'Scripts', 'python.exe'))
TTS_PY = os.environ.get('VF_TTS_PYTHON', os.path.join(_AI, 'tools', 'envs', 'tts', 'Scripts', 'python.exe'))
# modules that live in another venv run as one-shot children of that venv (the persistent worker is the gen venv)
OTHER_PY = {'tts': TTS_PY, 'delivery': TTS_PY}   # delivery: speech models live in the tts venv
HF_HOME = os.environ.get('VF_HF_HOME', os.path.join(_AI, 'models', 'hf'))
PORT = int(os.environ.get('VF_GEN_PORT', '47219'))
IDLE = os.environ.get('VF_GEN_IDLE', '600')  # builds stop the worker they started; this is the fallback
STARTED = [False]  # this process started the worker (build.py stops only its own, never another client's)
LOG = os.path.join(ROOT, 'cache', 'gen_server.log')


def _env():
    env = dict(os.environ, HF_HOME=HF_HOME, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', VF_GEN_IDLE=IDLE)
    env.setdefault('VF_GLOBAL_CACHE', os.path.join(ROOT, 'cache'))
    return env


CHILD_TIMEOUT = float(os.environ.get('VF_GEN_TIMEOUT', 3600))  # per child run / per worker reply line


def _request(req, timeout=None):
    s = socket.create_connection(('127.0.0.1', PORT), timeout=5)
    s.settimeout(timeout)
    f = s.makefile('rw', encoding='utf-8', newline='\n')
    f.write(json.dumps(req) + '\n')
    f.flush()
    return s, f


def _ping():
    try:
        s, f = _request({'cmd': 'ping'}, timeout=5)
        with s:
            return json.loads(f.readline()).get('pong', False)
    except OSError:
        return False


def start_gen(wait=120):
    """Start the persistent worker if it isn't running. Returns True when it answers."""
    if _ping():
        return True
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    flags = 0
    if os.name == 'nt':
        # a hidden console, not DETACHED: the venv python.exe is a launcher, and its real-python child opened a new,
        # empty console window on the owner's desktop when the launcher had no console at all (2026-11-14)
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with open(LOG, 'a', encoding='utf-8') as log:
        subprocess.Popen([GEN_PY, '-m', 'vf.gen.server', '--port', str(PORT)], cwd=ROOT, env=_env(),
                         stdout=log, stderr=log, stdin=subprocess.DEVNULL, creationflags=flags)
    STARTED[0] = True
    t = time.time()
    while time.time() - t < wait:
        if _ping():
            return True
        time.sleep(0.5)
    raise RuntimeError(f'gen worker did not start; see {LOG}')


def stop_gen():
    try:
        s, f = _request({'cmd': 'stop'}, timeout=10)
        with s:
            f.readline()
    except OSError:
        pass


def _cmd(cmd, timeout=60):
    try:
        s, f = _request({'cmd': cmd}, timeout=timeout)
        with s:
            return json.loads(f.readline())
    except (OSError, ValueError):
        return None  # worker not running: nothing to park


def park_gen():
    """Move the worker's models to CPU RAM (or free them when RAM is short) so the GPU is free. Returns the worker's
    answer ({'parked': {...}, 'vram_reserved_gb': ...}) or None when no worker runs."""
    return _cmd('park')


def gen_status():
    return _cmd('status', timeout=10)


def _check(module, out, rc=0):
    bad = [r for r in out if not r.get('ok')]
    if rc != 0 or bad:
        raise RuntimeError(f'vf.gen.{module} failed (exit {rc}): {bad[:3]}')


def run_gen(module, jobs, timeout=None, check=True, persistent=None):
    """Run jobs through vf.gen.<module>.run_jobs; return one result dict per job (in order).
    Paths in jobs should be absolute (the worker's cwd is the videoforge root)."""
    jobs = list(jobs)
    if not jobs:
        return []
    if persistent is None:
        persistent = os.environ.get('VF_GEN_PERSIST', '1') != '0'
    if module in OTHER_PY:
        park_gen()  # the other venv's model needs the VRAM; the gen models wait in RAM (~1 s to resume)
        return _run_child(module, jobs, timeout, check, OTHER_PY[module])
    if persistent:
        start_gen()
        results = {}
        s, f = _request({'module': module, 'jobs': jobs, 'allow_encode': os.environ.get('VF_ALLOW_ENCODE') == '1'},
                        timeout=timeout or CHILD_TIMEOUT)  # was None: a hung worker blocked the build forever
        with s:
            for line in f:
                r = json.loads(line)
                if r.get('done'):
                    if r.get('error'):
                        results.setdefault('_error', r['error'])
                    break
                results[r['i']] = r
        err = results.pop('_error', None)
        out = [results.get(i, dict(i=i, ok=False, error=err or 'no result')) for i in range(len(jobs))]
        if check:
            _check(module, out)
        return out
    return _run_child(module, jobs, timeout, check)


def _run_child(module, jobs, timeout, check, py=None):
    fd, path = tempfile.mkstemp(prefix=f'vf_{module}_', suffix='.json')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(jobs, f)
    results, rc = {}, None
    try:
        p = subprocess.Popen([py or GEN_PY, '-m', f'vf.gen.{module}', path], cwd=ROOT, env=_env(), stdout=subprocess.PIPE,
                             text=True, encoding='utf-8', errors='replace')
        # the old p.wait(timeout) only ran after stdout closed, so a hung child blocked the build forever
        import threading
        from vf.proc import _kill_tree
        dog = threading.Timer(timeout or CHILD_TIMEOUT, _kill_tree, [p])
        dog.daemon = True
        dog.start()
        for line in p.stdout:
            s = line.strip()
            if s.startswith('{'):
                try:
                    r = json.loads(s)
                    if isinstance(r, dict) and 'i' in r:
                        results[r['i']] = r
                        continue
                except ValueError:
                    pass
            sys.stderr.write(line)
        rc = p.wait(timeout=60)
        dog.cancel()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    out = [results.get(i, dict(i=i, ok=False, error=f'no result (child exit {rc})')) for i in range(len(jobs))]
    if check:
        _check(module, out, rc)
    return out
