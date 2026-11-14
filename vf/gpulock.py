"""Cross-process GPU lock (8 GB card: two model loads at once spill to system RAM and stall both).

    from vf.gpulock import gpu_lock
    with gpu_lock('bench i2v'):
        ...

Command line (for ad-hoc scripts): python -m vf.gpulock status
A lock older than `stale` seconds whose owner process is gone is broken automatically.
"""
import json, os, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'cache', 'gpu.lock')


def _ctime(pid):
    """Process creation time (Windows reuses pids: a live pid alone doesn't prove the lock owner is alive)."""
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _alive(pid, since=None):
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        c = _ctime(pid)
        return c is None or since is None or c <= since + 1.0  # a process born after the lock was taken is a reused pid
    except ImportError:
        if os.name == 'nt':
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def owner():
    try:
        with open(PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# Threads of one process share the file lock but must not use the GPU at the same time (2026-11-03 audit: the
# speculative finishing thread ran RTX VSR/frame generation next to the judge VLM and the next K6 batch -> spill).
# So: a per-process RLock serialises threads (re-entrant for nested use in one thread), and the cross-process file
# lock is taken by the outermost holder only.
_tlock = threading.RLock()
_depth = [0]


class gpu_lock:
    def __init__(self, what='', wait=3600, poll=2.0):
        self.what, self.wait, self.poll = what, wait, poll

    def __enter__(self):
        if not _tlock.acquire(timeout=self.wait):
            raise TimeoutError(f'GPU busy in this process ({self.what})')
        try:
            if _depth[0] == 0:
                self._file_acquire()
        except BaseException:
            _tlock.release()
            raise
        _depth[0] += 1
        return self

    def _file_acquire(self):
        try:   # a step that takes the GPU first waits until CPU/GPU/RAM are under VF_TEMP_MAX (vf/thermal.py)
            from .thermal import wait_cool
            wait_cool(what=self.what)
        except ImportError:
            pass
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        t0 = time.time()
        said = False
        while True:
            try:
                fd = os.open(PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(dict(pid=os.getpid(), what=self.what, since=time.time()), f)
                return
            except FileExistsError:
                o = owner()
                if o is None or not _alive(int(o.get('pid', -1)), o.get('since')):
                    try:
                        os.remove(PATH)  # owner died without releasing (or its pid was reused)
                    except OSError:
                        pass
                    continue
                if int(o.get('pid', -1)) == os.getpid():
                    os.remove(PATH)  # our own stale entry (an earlier holder in this process crashed mid-release)
                    continue
                if not said:
                    print(f'[gpu-lock] waiting for pid {o.get("pid")} ({o.get("what")})', file=sys.stderr, flush=True)
                    said = True
                if time.time() - t0 > self.wait:
                    raise TimeoutError(f'GPU busy: {o}')
                time.sleep(self.poll)

    def __exit__(self, *exc):
        _depth[0] -= 1
        try:
            if _depth[0] == 0:
                o = owner()
                if o and int(o.get('pid', -1)) == os.getpid():
                    try:
                        os.remove(PATH)
                    except OSError:
                        pass
        finally:
            _tlock.release()
