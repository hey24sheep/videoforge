"""AI generators (run inside the gen/tts venvs through vf.gen.bridge). Torch-free at import (the main interpreter
imports this package for the bridge).

8 GB VRAM holds one big model at a time. Each generator calls `claim(name, release, park, size_gb)` before it puts
weights on the GPU. Every other generator holding the GPU is then PARKED (weights moved to CPU RAM, back on the GPU
in ~1 s on its next claim) when the box has RAM to spare, else RELEASED (freed; the next use reloads from disk,
~20 s). `park_all()` parks everything (the gen worker's `park` command: frees all VRAM for the renderer / TTS
while keeping the next AI job fast).
"""
HEADROOM_GB = 6.0   # keep at least this much system RAM available after parking (31 GB box: renderer, TTS, OS)
_OWNERS = {}        # name -> (release, park, size_gb)
_PARKED = set()


def avail_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return 0.0


def _park_or_release(k):
    release, park, gb = _OWNERS[k]
    if k in _PARKED:  # already in RAM: keep it only while the box still has headroom
        if avail_gb() < HEADROOM_GB:
            _OWNERS.pop(k)
            _PARKED.discard(k)
            release()
        return
    if park is not None and avail_gb() - gb >= HEADROOM_GB:
        park()
        _PARKED.add(k)
    else:
        _OWNERS.pop(k)
        release()


def claim(name, release, park=None, size_gb=0.0):
    """`name` is about to use the GPU: park (or release) every other generator first."""
    for k in [k for k in _OWNERS if k != name]:
        try:
            _park_or_release(k)
        except Exception:
            _OWNERS.pop(k, None)
            _PARKED.discard(k)
    _OWNERS[name] = (release, park, size_gb)
    _PARKED.discard(name)


def park_all():
    """Move every generator's weights off the GPU (RAM permitting, else free them). Returns {name: state}."""
    out = {}
    for k in list(_OWNERS):
        try:
            _park_or_release(k)
        except Exception:
            _OWNERS.pop(k, None)
            _PARKED.discard(k)
        out[k] = 'parked' if k in _PARKED else 'released'
    return out


def release_all():
    for k in list(_OWNERS):
        release = _OWNERS.pop(k)[0]
        _PARKED.discard(k)
        try:
            release()
        except Exception:
            pass


def status():
    return {k: ('parked' if k in _PARKED else 'gpu') for k in _OWNERS}
