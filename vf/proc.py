"""Child-process hygiene (2026-11-03 audit: stray python/ffmpeg after builds, encoders left running on exceptions,
waits with no timeout).

    job_guard()          put this process in a Windows Job Object with KILL_ON_JOB_CLOSE: every child started
                         afterwards (ffmpeg, gen venv steps, the gen worker) dies with the build, even on a crash.
    run(args, timeout)   subprocess.run with check + timeout; the child tree is killed on timeout.
    writer(cmd, out)     context manager for an ffmpeg rawvideo writer: yields the Popen; on an exception the encoder
                         is killed and reaped and the partial file removed; on success waits (timeout) and checks rc.
"""
import contextlib, os, subprocess

_JOB = {}


def job_guard():
    """Idempotent. No-op off Windows or when the job can't be created (then children just aren't auto-killed)."""
    if os.name != 'nt' or _JOB.get('h'):
        return bool(_JOB.get('h'))
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)

        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ('r', 'w', 'o', 'rb', 'wb', 'ob')]

        class BASIC(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                        ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class EXT(ctypes.Structure):
            _fields_ = [('Basic', BASIC), ('Io', IO), ('ProcessMemoryLimit', ctypes.c_size_t),
                        ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t),
                        ('PeakJobMemoryUsed', ctypes.c_size_t)]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        h = k32.CreateJobObjectW(None, None)
        if not h:
            return False
        info = EXT()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(wintypes.HANDLE(h), 9, ctypes.byref(info), ctypes.sizeof(info)):
            return False
        if not k32.AssignProcessToJobObject(wintypes.HANDLE(h), wintypes.HANDLE(k32.GetCurrentProcess())):
            return False  # e.g. a parent job that forbids nesting: carry on without it
        _JOB['h'] = h  # keep the handle open for the life of the process (closing it kills the job)
        return True
    except Exception:
        return False


def _kill_tree(p):
    try:
        import psutil
        for c in psutil.Process(p.pid).children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass
    try:
        p.wait(timeout=30)
    except Exception:
        pass


def run(args, timeout=1800, **kw):
    p = subprocess.Popen(args, **kw)
    try:
        rc = p.wait(timeout=timeout)
    except BaseException:
        _kill_tree(p)
        raise
    if rc:
        raise subprocess.CalledProcessError(rc, args)
    return rc


@contextlib.contextmanager
def writer(cmd, out=None, timeout=900):
    """cmd: full ffmpeg argv reading rawvideo from stdin. out: the file it writes (removed on failure)."""
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        yield p
        p.stdin.close()
        rc = p.wait(timeout=timeout)
        if rc:
            raise RuntimeError(f'ffmpeg failed (exit {rc}): {" ".join(map(str, cmd[:6]))} ...')
    except BaseException:
        _kill_tree(p)
        if out and os.path.exists(out):
            try:
                os.remove(out)
            except OSError:
                pass
        raise
