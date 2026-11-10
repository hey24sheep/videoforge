"""Persistent gen worker: keeps vf.gen.* modules (and their loaded models) alive across jobs and builds.

    python -m vf.gen.server [--port 47219] [--idle 3600] [--park-idle 120]

Protocol (localhost TCP, newline-delimited JSON):
    request : {"module": "video", "jobs": [ {...}, ... ]}
              {"cmd": "ping"} / {"cmd": "stop"} / {"cmd": "park"} / {"cmd": "release"} / {"cmd": "status"}
    response: one JSON line per job result (same dicts the CLI prints), then {"done": true}
              (commands answer with one line)
Jobs run one request at a time (GPU is serial).
  park    : every loaded model's weights go to CPU RAM (RAM permitting, else freed) and the CUDA cache is emptied, so
            the renderer / TTS get the whole GPU; the next job moves them back in ~1 s (a reload from disk is ~20 s
            per model plus ~15 s of imports for a fresh worker).
  release : free every model (RAM too); the worker stays up.
The worker parks itself after `park_idle` seconds without a request (VRAM never sits taken by an idle worker) and
exits after `idle` seconds. vf.gen.bridge.run_gen starts it on demand.
"""
import argparse, importlib, json, os, socket, sys, threading, time, traceback

os.environ['VF_IN_GEN_SERVER'] = '1'  # vf/gen/guard.py: never park the worker from inside it

PORT = int(os.environ.get('VF_GEN_PORT', '47219'))


def _mem():
    out = {}
    try:
        import psutil
        mi = psutil.Process().memory_info()
        out['rss_gb'] = round(mi.rss / 2 ** 30, 2)
        if hasattr(mi, 'private'):  # Windows: committed private bytes (RSS also counts mapped model files)
            out['private_gb'] = round(mi.private / 2 ** 30, 2)
        out['avail_ram_gb'] = round(psutil.virtual_memory().available / 2 ** 30, 2)
    except Exception:
        pass
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        if torch.cuda.is_available():
            out['vram_alloc_gb'] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
            out['vram_reserved_gb'] = round(torch.cuda.memory_reserved() / 2 ** 30, 2)
    return out


def _park():
    from . import park_all
    states = park_all()
    if 'torch' in sys.modules:
        import gc
        gc.collect()
        torch = sys.modules['torch']
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return states


def _status():
    from . import status
    out = dict(owners=status(), pid=os.getpid(), **_mem())
    for name in ('video', 'image'):
        m = sys.modules.get(f'vf.gen.{name}')
        if m is not None and hasattr(m, 'status') and 'torch' in sys.modules:
            try:
                out[name] = m.status()
            except Exception as e:
                out[name] = f'{type(e).__name__}: {e}'
    return out


def serve(port=PORT, idle=3600, park_idle=120):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(('127.0.0.1', port))
    srv.listen(4)
    srv.settimeout(5)
    last = [time.time()]
    parked = [True]  # nothing loaded yet
    print(f'[gen-server] listening on 127.0.0.1:{port} (park after {park_idle}s idle, exit after {idle}s)',
          file=sys.stderr, flush=True)
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            quiet = time.time() - last[0]
            if quiet > idle:
                print('[gen-server] idle, exiting', file=sys.stderr, flush=True)
                return
            if not parked[0] and quiet > park_idle:
                try:
                    print(f'[gen-server] idle {int(quiet)}s, parking: {_park()}', file=sys.stderr, flush=True)
                except Exception:
                    traceback.print_exc()
                parked[0] = True
            continue
        with conn:
            f = conn.makefile('rw', encoding='utf-8', newline='\n')
            try:
                req = json.loads(f.readline())
                cmd = req.get('cmd')
                if cmd == 'ping':
                    f.write(json.dumps({'pong': True, 'pid': os.getpid()}) + '\n')
                elif cmd == 'stop':
                    f.write(json.dumps({'stopping': True}) + '\n'); f.flush()
                    return
                elif cmd == 'park':
                    t0 = time.time()
                    st = _park()
                    parked[0] = True
                    f.write(json.dumps({'parked': st, 'seconds': round(time.time() - t0, 2), **_mem()}) + '\n')
                elif cmd == 'release':
                    from . import release_all
                    release_all()
                    _park()
                    parked[0] = True
                    f.write(json.dumps({'released': True, **_mem()}) + '\n')
                elif cmd == 'status':
                    f.write(json.dumps(_status()) + '\n')
                else:
                    mod = importlib.import_module(f"vf.gen.{req['module']}")
                    emit = lambda r: (f.write(json.dumps(r) + '\n'), f.flush())
                    parked[0] = False
                    # the worker outlives builds: take the encoder permission from each request, not from startup
                    os.environ['VF_ALLOW_ENCODE'] = '1' if req.get('allow_encode') else '0'
                    mod.run_jobs(req['jobs'], emit=emit)
                    f.write(json.dumps({'done': True}) + '\n')
                f.flush()
            except Exception as e:  # report and keep serving
                traceback.print_exc()
                try:
                    f.write(json.dumps({'done': True, 'error': f'{type(e).__name__}: {e}'}) + '\n')
                    f.flush()
                except OSError:
                    pass
            last[0] = time.time()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--idle', type=float, default=float(os.environ.get('VF_GEN_IDLE', '600')))  # builds stop it anyway
    ap.add_argument('--park-idle', type=float, default=float(os.environ.get('VF_GEN_PARK_IDLE', '120')))
    a = ap.parse_args()
    from .guard import process_cap
    process_cap()
    serve(a.port, a.idle, a.park_idle)
