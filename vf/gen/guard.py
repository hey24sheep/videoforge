"""Guardrails for text encoders and other big helper models (user rule, 2026-11-03).

A text encoder (Kandinsky Qwen2.5-VL 16 GB, Z-Image Qwen3 8.5 GB, Wan UMT5 11 GB, Klein Qwen3) may only load when:
  1. the run was explicitly allowed to encode: `vf build --allow-encode` (sets VF_ALLOW_ENCODE=1). A build whose
     prompts are all cached never needs one; a changed prompt without the flag stops before anything heavy loads.
  2. the machine has room: free RAM >= VF_ENCODE_MIN_RAM GB (default 20) and no other process holds more than
     0.5 GB of VRAM (the shared gen worker is parked first when the caller is not the worker itself).
  3. it runs under a GPU allocator cap (`encode_cap`), so an overflow raises an OOM instead of spilling into
     shared system memory.
On 2026-11-03 the Kandinsky encoder loaded by itself for 2 new prompts next to the DiT and the QC worker: 2.6 GB
spilled into shared GPU memory and RAM peaked at 30.6 of 31.3 GB.
"""
import contextlib, os, subprocess

MIN_FREE_RAM_GB = float(os.environ.get('VF_ENCODE_MIN_RAM', 20))
MAX_OTHER_VRAM_GB = float(os.environ.get('VF_ENCODE_MAX_OTHER_VRAM', 0.5))
VRAM_CAP_GB = float(os.environ.get('VF_ENCODE_VRAM_CAP', 6.0))


class EncodeRefused(RuntimeError):
    pass


def allowed():
    return os.environ.get('VF_ALLOW_ENCODE') == '1'


def free_ram_gb():
    import psutil
    return psutil.virtual_memory().available / 2**30


def _per_process_vram():
    """{pid: dedicated bytes} on the busiest GPU adapter, from Windows' GPU Process Memory counters (per-process
    numbers that nvidia-smi can't give under WDDM). The NVIDIA card is the adapter this process uses; before CUDA
    is initialised it is the adapter with the most dedicated memory in use."""
    ps = ("(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -EA SilentlyContinue).CounterSamples | "
          "% { $_.InstanceName + ' ' + [int64]$_.CookedValue }")
    out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True,
                         timeout=30).stdout
    rows = []
    for line in out.splitlines():
        try:
            inst, v = line.rsplit(' ', 1)
            parts = inst.split('_')
            rows.append((int(parts[1]), '_'.join(parts[3:5]), int(v)))
        except (ValueError, IndexError):
            continue
    if not rows:
        return None
    me = os.getpid()
    luids = {l for p, l, _ in rows if p == me}
    if not luids:
        tot = {}
        for _, l, v in rows:
            tot[l] = tot.get(l, 0) + v
        luids = {max(tot, key=tot.get)}
    got = {}
    for p, l, v in rows:
        if l in luids:
            got[p] = got.get(p, 0) + v
    return got


def other_vram_gb():
    """VRAM held on the card by other processes. Own process (incl. its CUDA context, which PyTorch's allocator
    doesn't count) and its parent (the venv launcher stub) are excluded: 2026-11-03 the worker's own 1.1 GB
    context was miscounted as 'other' and blocked a legitimate encode."""
    if os.name == 'nt':
        try:
            import torch
            if torch.cuda.is_available():
                torch.zeros(1, device='cuda')  # own context on the NVIDIA card marks which adapter to read
        except Exception:
            pass
        try:
            got = _per_process_vram()
            if got is not None:
                mine = {os.getpid(), os.getppid()}
                return sum(v for p, v in got.items() if p not in mine) / 2**30
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return max(0.0, (total - free - torch.cuda.memory_reserved()) / 2**30)
    except Exception:
        pass
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.split()[0]) / 1024
    except Exception:
        return 0.0


def allow_encoder(name, n_prompts, park=None, what=None):
    """Raise EncodeRefused unless `name` may load now for `n_prompts` uncached prompt(s). Call it BEFORE any weights
    load. The shared gen worker is parked first unless this code runs inside it."""
    if n_prompts <= 0:
        return
    shown = '; '.join(p[:70] for p in (what or [])[:3])
    if not allowed():
        raise EncodeRefused(
            f'{name}: {n_prompts} prompt(s) have no cached embedding ({shown}). Loading the text encoder is blocked. '
            f'If the prompt change is intended, rerun with `vf build --allow-encode` (or set VF_ALLOW_ENCODE=1).')
    if park is None:  # inside the shared worker, parking would wait on itself
        park = os.environ.get('VF_IN_GEN_SERVER') != '1'
    if park:
        try:
            from .bridge import park_gen
            park_gen()
        except Exception:
            pass
    ram = free_ram_gb()
    if ram < MIN_FREE_RAM_GB:
        raise EncodeRefused(f'{name}: only {ram:.1f} GB RAM free, need {MIN_FREE_RAM_GB:.0f} GB to load the text '
                            f'encoder without swapping. Close other programs or stop the gen worker first.')
    other = other_vram_gb()
    if other > MAX_OTHER_VRAM_GB:
        raise EncodeRefused(f'{name}: other processes hold {other:.1f} GB of VRAM; the encoder runs only on a clear '
                            f'GPU (limit {MAX_OTHER_VRAM_GB} GB). Stop them first.')
    print(f'  [guard] {name}: encoding {n_prompts} prompt(s); RAM free {ram:.1f} GB, other VRAM {other:.2f} GB',
          flush=True)


_BASE = [1.0]  # this process's standing allocator fraction (process_cap); encode_cap restores it, never 1.0


def process_cap(gb=float(os.environ.get('VF_PROC_VRAM_GB', 7.2))):  # Wan fast 832x480 measured ~7.1 GB (nvidia-smi)
    """Cap the whole process's CUDA allocator (2026-11-03 audit: the gen worker - judge, i2v, t2i - and Klein ran
    uncapped; on WDDM an uncapped allocator pages past the 8 GB card into shared RAM instead of raising OOM)."""
    import torch
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    _BASE[0] = min(1.0, gb / total)
    torch.cuda.set_per_process_memory_fraction(_BASE[0])


@contextlib.contextmanager
def encode_cap(gb=VRAM_CAP_GB):
    """Cap this process's CUDA allocator while an encoder runs; overflow becomes an OOM, never a spill."""
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total))
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(_BASE[0])
        torch.cuda.empty_cache()
        if hasattr(torch._C, '_host_emptyCache'):
            torch._C._host_emptyCache()
