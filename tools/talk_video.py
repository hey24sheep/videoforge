"""Audio-driven talking video: image + speech wav -> 9:16 clip (face, head, eyes, lips, shoulders move from the audio).

Engine: MeiGen InfiniteTalk (single speaker, Apache-2.0) on Wan2.1-I2V-14B-480P (Apache-2.0), run through DeepBeepMeep's
Wan2GP (WanGP Community License 2.0) with int8 quanto weights, offload profile 5 + a 9 GB working-set cap, the lightx2v I2V step-distill LoRA
(Apache-2.0, 4 steps, text CFG 1, audio CFG 2) and SageAttention 2.2 (sage2; was 1 until 2026-11-10). Fully offline (--offline): weights live in hero_film/models/wan2gp_ckpts.

Run with the talk venv:
  D:\\claude_work\\hero_film\\tools\\envs\\talk\\Scripts\\python.exe tools\\talk_video.py --image a.png --audio a.wav \
      --prompt "an old man speaking anxiously" --out out.mp4 [--seed N] [--steps N] [--size 480x832]

VRAM guard: PyTorch's caching allocator (not mmgp's spilling VMM allocator) + set_per_process_memory_fraction, so a run
that does not fit raises CUDA OOM instead of paging into Windows shared memory. Shared GPU memory is sampled during the
run (Windows perf counters) and reported.
"""
import argparse
import json
import math
import os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERO = Path(_AI)
WAN2GP = HERO / "tools" / "wan2gp"
CFG = HERO / "tools" / "wan2gp_cfg" / "wgp_config.json"
LORA_URL = ("https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/"
            "Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")
MODEL_URL = "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/wan2.1_image2video_480p_14B_quanto_mbf16_int8.safetensors"
FPS = 25  # InfiniteTalk native frame rate
VRAM_FRACTION = 0.88  # 0.88 * 8 GB = 7.2 GB hard cap for this process
RAM_ABORT_GB = 2.5     # abort if commit comes within this many GB of physical RAM (4.0 tripped inside a vf build at 27.3/31.3 GB with 7 GB still available, 2026-11-07)
RAM_START_GB = 8.0     # refuse to start below this much free RAM (the run pins ~10 GB + holds ~9 GB of weights)


def ffprobe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


class GpuSampler(threading.Thread):
    """Every ~3 s: device-wide VRAM used (NVML, includes other processes) and THIS process's Windows GPU counters
    (GPU Process Memory: Dedicated Usage = VRAM, Shared Usage = system RAM mapped for the GPU). Note: WanGP/mmgp pins
    model blocks in host RAM for fast DMA; that pinned RAM also shows up as process Shared Usage, so shared usage alone
    is not proof of spill. A spill = dedicated usage pressed against the card while shared grows beyond the pinned set."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_evt = threading.Event()
        self.peak_device_mb = 0
        self.peak_proc_dedicated_mb = 0
        self.peak_proc_shared_mb = 0
        self.pid = os.getpid()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nv = None

    def proc_counters(self):
        if os.name != "nt":
            return None, None
        try:
            ps = (f"$s=(Get-Counter '\\GPU Process Memory(pid_{self.pid}_*)\\Dedicated Usage',"
                  f"'\\GPU Process Memory(pid_{self.pid}_*)\\Shared Usage' -ErrorAction Stop).CounterSamples;"
                  "$d=($s | ? Path -like '*dedicated*' | Measure-Object CookedValue -Sum).Sum;"
                  "$h=($s | ? Path -like '*shared*' | Measure-Object CookedValue -Sum).Sum; \"$d $h\"")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                 timeout=20, creationflags=0x08000000).stdout.strip().split()
            return float(out[0]) / 2**20, float(out[1]) / 2**20
        except Exception:
            return None, None

    def run(self):
        import psutil
        phys_gb = psutil.virtual_memory().total / 2**30
        while not self.stop_evt.is_set():
            avail = psutil.virtual_memory().available / 2**30
            commit = commit_gb()
            self.min_free_ram_gb = min(getattr(self, "min_free_ram_gb", 99.0), avail)
            self.peak_commit_gb = max(getattr(self, "peak_commit_gb", 0.0), commit)
            # Measured 2026-11-07: Windows "available" falls to ~1.6 GB during sampling while this process holds only
            # 11.8 GB private / 14.4 GB working set and nothing else is big: the rest is clean file cache of the
            # streamed 17 GB model, which Windows drops on demand. Real pressure = private commit near physical RAM,
            # so the guard aborts when commit > physical - 4 GB, or available < 1 GB.
            me = psutil.Process().memory_info()
            self.ticks = getattr(self, "ticks", 0) + 1
            if self.ticks % 4 == 1 or avail < 4:
                print(f"[talk] mem: available {avail:.1f} GB, commit {commit:.1f} GB, this process private "
                      f"{me.private / 2**30:.1f} GB / working set {me.wset / 2**30:.1f} GB, system file cache "
                      f"{getattr(commit_gb, 'system_cache', 0):.1f} GB", flush=True)
            # 2026-11-07: with the 9 GB working-set cap the model's overflow goes to the pagefile BY DESIGN, so commit
            # climbs past physical RAM while available RAM stays flat (7 GB): the guard tripped twice inside a build.
            # Real danger = available RAM running out, or commit reaching the commit limit (RAM + pagefile).
            if avail < 2.0 or commit > getattr(commit_gb, 'limit', phys_gb + 8) - RAM_ABORT_GB:
                print(f"[talk] ABORT: commit {commit:.1f} GB of {phys_gb:.1f} GB physical, available {avail:.1f} GB",
                      flush=True)
                top = sorted(psutil.process_iter(["pid", "name", "memory_info"]),
                             key=lambda q: q.info["memory_info"].private if q.info["memory_info"] else 0, reverse=True)[:6]
                for q in top:
                    mi = q.info["memory_info"]
                    print(f"[talk]   pid {q.info['pid']} {q.info['name']}: private {mi.private / 2**30:.1f} GB, "
                          f"working set {mi.wset / 2**30:.1f} GB", flush=True)
                os._exit(3)
            if self.nv:
                try:
                    used = self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**20
                    self.peak_device_mb = max(self.peak_device_mb, used)
                except Exception:
                    pass
            d, s = self.proc_counters()
            if d is not None:
                self.peak_proc_dedicated_mb = max(self.peak_proc_dedicated_mb, d)
                self.peak_proc_shared_mb = max(self.peak_proc_shared_mb, s)
            self.stop_evt.wait(3.0)


def commit_gb():
    """System commit charge (private memory of all processes + pagefile-backed), GB. Windows only."""
    if os.name != "nt":
        return 0.0
    import ctypes
    from ctypes import wintypes

    class PI(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]
    pi = PI()
    pi.cb = ctypes.sizeof(PI)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    commit_gb.system_cache = pi.SystemCache * pi.PageSize / 2**30
    commit_gb.limit = pi.CommitLimit * pi.PageSize / 2**30   # physical RAM + pagefile
    return pi.CommitTotal * pi.PageSize / 2**30


def cap_working_set(gb):
    """Hard cap on this process's physical working set (Windows). WanGP memory-maps the 17 GB int8 model; without a cap
    the mapped pages stay in the working set and system 'available' RAM fell to 0.2 GB. With the cap, pages beyond it
    go to the standby list (still counted as available, re-read from RAM, not disk). Private memory must stay below
    the cap or it pages to disk."""
    if os.name != "nt":
        return False
    import ctypes
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x2, 0x4
    return bool(k.SetProcessWorkingSetSizeEx(ctypes.c_void_p(k.GetCurrentProcess()), ctypes.c_size_t(256 << 20),
                                             ctypes.c_size_t(int(gb * 2**30)),
                                             ctypes.c_uint(QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE)))


def prep_image(src, dst, w, h):
    """Center-crop to the target aspect, then resize to exactly WxH, so WanGP keeps the requested size
    (it otherwise fits its pixel budget to the image's own aspect, e.g. 576x1024 -> 480x848)."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    sw, sh = im.size
    ta = w / h
    if sw / sh > ta:
        cw = round(sh * ta)
        box = ((sw - cw) // 2, 0, (sw - cw) // 2 + cw, sh)
    else:
        ch = round(sw / ta)
        box = (0, (sh - ch) // 2, sw, (sh - ch) // 2 + ch)
    im.crop(box).resize((w, h), Image.LANCZOS).save(dst)
    return dst


def speech_bounds(path, floor_db=30.0, hop=0.01):
    """(onset, offset) seconds of speech in `path`: 10 ms RMS frames within `floor_db` of the loudest (exact fact)."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, dtype=np.float32)
    k = int(16000 * hop)
    n = len(a) // k
    db = 20 * np.log10(np.sqrt((a[:n * k].reshape(n, k) ** 2).mean(1)) + 1e-9)
    on = np.flatnonzero(db > db.max() - floor_db)
    return (float(on[0] * hop), float((on[-1] + 1) * hop)) if len(on) else (0.0, n * hop)


def prep_audio(src, dst, seconds):
    """Mono wav padded with silence to `seconds` (WanGP clamps the clip length to the audio length)."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-af", f"apad=whole_dur={seconds:.3f}",
                    "-t", f"{seconds:.3f}", str(dst)], check=True)
    return dst


def talk_cores(n):
    """Run this process on the first `n` logical cores with that many math threads (VF_TALK_CORES, default 2). The talk
    stage streams 14B weights through the CPU on every step: on 4 cores it reached 80-82 C each step and 89 C at decode
    (2026-11-14). The jobwatch thermal guard keeps a narrower mask it finds."""
    try:
        n = max(1, min(os.cpu_count() or 2, int(n)))
    except ValueError:
        return
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p   # pseudo handle -1: a 32-bit int return truncates it
        if not k32.SetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.c_size_t((1 << n) - 1)):
            print(f"[talk] CPU cap NOT set (error {k32.GetLastError()})", flush=True)
            return
    print(f"[talk] CPU: {n} of {os.cpu_count()} cores (VF_TALK_CORES)", flush=True)


def cool_before_decode():
    """Wait for a cool machine before every video decode (VF_TALK_COOL_DECODE=1): the decode right after the last step
    spiked the CPU to 89 C and the guard killed the take (2026-11-14). Blocks inside the generation thread, so the
    decode itself waits."""
    if os.environ.get("VF_TALK_COOL_DECODE", "1") == "0":
        return
    try:
        sys.path.append(r"D:\claude_work\videoforge")
        from vf.thermal import wait_cool
        from models.wan.modules import vae as wan_vae
    except Exception as e:
        print(f"[talk] cool-before-decode not set: {e}", flush=True)
        return
    orig = wan_vae.WanVAE.decode_to_cpu_uint8

    def decode(self, *a, **k):
        waited = wait_cool(log=lambda s: print(s, flush=True), what="talk video decode")
        if waited:
            time.sleep(5)   # let the reading settle below the resume level before the burst
        return orig(self, *a, **k)
    wan_vae.WanVAE.decode_to_cpu_uint8 = decode
    print("[talk] decode waits for a cool machine (VF_TALK_COOL_DECODE)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--audio")
    ap.add_argument("--prompt")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=4, help="denoising steps (lightx2v distill LoRA: 4-8)")
    ap.add_argument("--size", default="480x832", help="WxH; 480x832 is the 9:16 size InfiniteTalk 480p supports")
    ap.add_argument("--audio-cfg", type=float, default=2.0,
                    help="audio guidance (InfiniteTalk README: 4 without LoRA, 2 with a step-distill LoRA; 1 = off, "
                         "half the compute per step)")
    ap.add_argument("--pad", type=float, default=0.3, help="seconds of video after the audio ends")
    ap.add_argument("--trim-tail", type=float, default=None,
                    help="cut the clip this many seconds after the speech ends (InfiniteTalk keeps the mouth moving "
                         "in trailing silence, 2026-11-07); cover the silence with another shot")
    ap.add_argument("--profile", default=os.environ.get("VF_TALK_PROFILE", "5"), help="WanGP/mmgp offload profile: 5 = no reserved RAM, weights memory-mapped "
                    "(default; profile 4 pins part of the transformer in page-locked RAM, capped by --pin: the GPU copies those blocks by "
                    "DMA without the CPU; uncapped it pinned ~10 GB and drove free RAM to 2.4 GB on this 31 GB laptop)")
    ap.add_argument("--pin", type=float, default=float(os.environ.get("VF_TALK_PIN", 0.2)),
                    help="share of system RAM mmgp may pin (page-locked) for the model (WanGP --perc-reserved-mem-max)")
    ap.add_argument("--max-ram", default="16", help="WanGP RAM planning budget in GB (--max-ram)")
    ap.add_argument("--ws-cap", type=float, default=0, help="hard working-set cap in GB (default 9: kept free RAM >= 7 GB in the 2026-11-07 test; 16 let it fall to 0.2 GB)")
    ap.add_argument("--attention", default="auto", help="auto = sage2 when SageAttention 2.x is installed (2026-11-10), else sage; or sdpa")
    ap.add_argument("--jobs", default=None, help="JSON list of {image, audio, prompt, out, seed?, steps?, audio_cfg?}: one process, the model loads ONCE for all of them (2026-11-09: ~90 s load per shot)")
    ap.add_argument("--preload", default=os.environ.get("VF_TALK_PRELOAD", "3000"),
                    help="MB of the 14B model kept resident in VRAM (mmgp; fewer RAM->GPU transfers per step)")
    ap.add_argument("--step-cache", default="", help="WanGP step cache: '' (off) | mag | tea: reuses the transformer output when a step barely changes it (opt-in; A/B tested 2026-11-10)")
    ap.add_argument("--step-cache-mult", type=float, default=1.5)
    ap.add_argument("--compile", action="store_true", help="WanGP torch.compile of the transformer (A/B 2026-11-10; off by default)")
    ap.add_argument("--quant", default="", help="A/B only: transformer quantization for a temp config copy (bf16 | fp8 | int8); with --model-type vf_it_bf16src the weights come from the bf16 files")
    ap.add_argument("--model-type", default="infinitetalk", help="WanGP model type (A/B: vf_it_bf16src)")
    ap.add_argument("--negative", default="blurry, distorted face, deformed mouth, extra limbs, text, watermark")
    args = ap.parse_args()
    if args.attention == "auto":
        try:
            from importlib.metadata import version as _v
            args.attention = "sage2" if _v("sageattention").startswith("2") else "sage"
        except Exception:
            args.attention = "sdpa"
    print(f"[talk] attention: {args.attention}, preload {args.preload} MB")

    if args.jobs:
        jobs = json.load(open(args.jobs, encoding="utf-8"))
    else:
        if not (args.image and args.audio and args.prompt and args.out):
            sys.exit("--image, --audio, --prompt and --out are required (or --jobs)")
        jobs = [dict(image=args.image, audio=args.audio, prompt=args.prompt, out=args.out)]

    import psutil
    free_gb = psutil.virtual_memory().available / 2**30
    if free_gb < RAM_START_GB:
        sys.exit(f"[talk] only {free_gb:.1f} GB RAM free (< {RAM_START_GB}): close other heavy jobs first")
    ws_cap = args.ws_cap or 9.0
    print(f"[talk] RAM: {free_gb:.1f} GB available; working-set cap {ws_cap:.1f} GB: "
          f"{'set' if cap_working_set(ws_cap) else 'NOT set'}")
    talk_cores(os.environ.get("VF_TALK_CORES", "2"))
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[talk] VRAM cap {VRAM_FRACTION * total_gb:.2f} GB of {total_gb:.2f} GB (overflow -> OOM, not shared memory)")

    sys.path.insert(0, str(WAN2GP))
    os.chdir(WAN2GP)
    from shared.api import init
    cool_before_decode()

    sys.path.append(r"D:\claude_work\videoforge")
    from vf.gpulock import gpu_lock  # cross-process lock shared with the vf pipeline
    lock = gpu_lock("talk_video InfiniteTalk")
    lock.__enter__()
    sampler = GpuSampler()
    sampler.start()
    cli = ["--offline", "--attention", args.attention, "--profile", str(args.profile),
           "--vram-allocator", "default", "--perc-reserved-mem-max", str(args.pin), "--max-ram", str(args.max_ram), "--preload", str(args.preload),
           "--verbose", "1"] + (["--compile"] if args.compile else [])
    t_init = time.time()
    cfg = CFG
    if args.quant:   # A/B: a temp copy of the config with another transformer quantization (production config untouched)
        d = json.load(open(CFG, encoding="utf-8"))
        d["transformer_quantization"] = args.quant
        cfg = Path(tempfile.mkdtemp(prefix="talkcfg_", dir=str(HERO / "tools" / "wan2gp_cfg"))) / "wgp_config.json"
        json.dump(d, open(cfg, "w", encoding="utf-8"), indent=1)
        print(f"[talk] A/B: transformer_quantization={args.quant}, model_type={args.model_type}", flush=True)
    session = init(root=WAN2GP, config_path=cfg, output_dir=Path(tempfile.mkdtemp(prefix="talk_", dir=str(HERO / "tools" / "wan2gp_cfg"))), cli_args=cli)
    failed = 0
    try:
        for i, job in enumerate(jobs):
            print(f"[talk] job {i + 1}/{len(jobs)}: {Path(job['out']).name}", flush=True)
            ok = run_one(args, job, session, sampler, total_gb, ws_cap, torch)
            failed += 0 if ok else 1
    finally:
        lock.__exit__(None, None, None)
        sampler.stop_evt.set()
        sampler.join(timeout=30)
    print(f"[talk] batch: {len(jobs)} job(s), {failed} failed, {time.time() - t_init:.0f}s incl. one model load")
    if failed:
        sys.exit(1)


def run_one(args, job, session, sampler, total_gb, ws_cap, torch):
    """One clip in an already initialised WanGP session. Returns True on success."""
    t0 = time.time()
    image = Path(job["image"]).resolve()
    audio = Path(job["audio"]).resolve()
    out = Path(job["out"]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in args.size.lower().split("x"))

    dur = ffprobe_duration(audio)
    target = dur + args.pad
    n = math.ceil(target * FPS)
    frames = 4 * math.ceil((n - 1) / 4) + 1  # Wan needs 4k+1 frames
    print(f"[talk] audio {dur:.2f}s -> {frames} frames @ {FPS} fps ({frames / FPS:.2f}s), size {w}x{h}, steps {args.steps}")
    if frames > 81:
        print(f"[talk] >81 frames: InfiniteTalk sliding windows of 81 with 9 overlap ({math.ceil((frames - 9) / 72)} windows)")

    work = Path(tempfile.mkdtemp(prefix="talk_", dir=str(HERO / "tools" / "wan2gp_cfg")))
    gen_image = prep_image(image, work / "ref.png", w, h)
    gen_audio = prep_audio(audio, work / "voice_padded.wav", frames / FPS)
    t_load = time.time()

    settings = {
        "model_type": args.model_type,
        "model_filename": MODEL_URL,
        "prompt": job["prompt"],
        "negative_prompt": args.negative,
        "resolution": f"{w}x{h}",
        "video_length": frames,
        "batch_size": 1,
        "seed": int(job.get("seed", args.seed)),
        "num_inference_steps": int(job.get("steps", args.steps)),
        "guidance_scale": 1,
        "guidance_phases": 1,
        "audio_guidance_scale": float(job.get("audio_cfg", args.audio_cfg)),
        "flow_shift": 7,
        "sample_solver": "euler",
        "repeat_generation": 1,
        "skip_steps_cache_type": job.get("step_cache", args.step_cache) or "",
        "skip_steps_multiplier": float(job.get("step_cache_mult", args.step_cache_mult)),
        "skip_steps_start_step_perc": 0,
        "activated_loras": [LORA_URL],
        "loras_multipliers": "1",
        "image_prompt_type": "",
        "video_prompt_type": "KI",
        "image_refs": [str(gen_image)],
        "audio_prompt_type": "A",
        "audio_guide": str(gen_audio),
        "sliding_window_size": 81,
        "sliding_window_overlap": 9,
        "sliding_window_discard_last_frames": 0,
        "remove_background_images_ref": 0,
        "temporal_upsampling": "",
        "spatial_upsampling": "",
        "film_grain_intensity": 0,
        "prompt_enhancer": "",
        "force_fps": "",  # model default (InfiniteTalk = 25 fps)
    }
    if job.get("settings"):   # A/B / tuning: per-job WanGP setting overrides (flow_shift, sample_solver, loras_multipliers, ...)
        settings.update(job["settings"])
        print(f"[talk] settings override: {job['settings']}", flush=True)
    steps_seen = {"t_first": None, "t_last": None, "n": 0}

    task = session.submit_task(settings)
    for ev in task.events.iter(timeout=0.5):
        if ev.kind == "progress":
            p = ev.data
            cur = getattr(p, "current_step", None)
            phase = str(getattr(p, "phase", ""))
            raw = str(getattr(p, "raw_phase", "") or "")
            key = (phase, cur)
            if cur is not None and ("denois" in (phase + raw).lower()) and key != steps_seen.get("key"):
                now = time.time()
                steps_seen["key"] = key
                if steps_seen["t_first"] is None:
                    steps_seen["t_first"], steps_seen["c_first"] = now, cur
                steps_seen["t_last"], steps_seen["c_last"] = now, cur
                steps_seen["n"] += 1
                print(f"[talk] step {cur}/{getattr(p, 'total_steps', '?')} {phase} t={now - t0:.0f}s", flush=True)
    result = task.result()
    if not result.success:
        for e in result.errors:
            print("[talk] ERROR:", getattr(e, "message", e))
        return False
    videos = [f for f in result.generated_files if str(f).lower().endswith(".mp4")]
    if not videos:
        print(f"[talk] ERROR no video produced: {result.generated_files}")
        return False
    src = Path(videos[-1])

    # Mux the original audio (AAC), length = audio + pad. If WanGP returned fewer frames than that, hold the last
    # frame (re-encode, crf 12); otherwise stream-copy the video.
    raw_dur = ffprobe_duration(src)
    tmp = out.with_suffix(".tmp.mp4")
    vopts = ["-c:v", "copy"]
    if raw_dur < target - 0.5 / FPS:
        vopts = ["-vf", f"tpad=stop_mode=clone:stop_duration={target - raw_dur + 0.1:.3f}", "-c:v", "libx264",
                 "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p"]
        print(f"[talk] raw clip {raw_dur:.2f}s < {target:.2f}s: last frame held {target - raw_dur:.2f}s")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
                    *vopts, "-c:a", "aac", "-b:a", "192k", "-af", f"apad=whole_dur={target:.3f}",
                    "-t", f"{target:.3f}", "-movflags", "+faststart", str(tmp)], check=True)
    os.replace(tmp, out)
    shutil.copy2(src, work / "raw_wangp.mp4") if src.parent != work else None
    trim = None
    if args.trim_tail is not None:
        onset, offset = speech_bounds(audio)
        trim = min(target, offset + args.trim_tail)
        untrimmed = out.with_suffix(".untrimmed.mp4")
        os.replace(out, untrimmed)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(untrimmed), "-t", f"{trim:.3f}", "-c:v", "libx264",
                        "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(out)], check=True)
        print(f"[talk] speech {onset:.2f}-{offset:.2f}s: trimmed to {trim:.2f}s (untrimmed kept: {untrimmed.name})")

    elapsed = time.time() - t0
    peak_res = torch.cuda.max_memory_reserved(0) / 2**30
    peak_alloc = torch.cuda.max_memory_allocated(0) / 2**30
    # s/step from the progress stream (denoise events); fall back to whole-gen time / steps
    # step k is reported when step k starts (or ends); the spacing between distinct step reports is the step time.
    if steps_seen["n"] >= 2 and steps_seen["t_last"] > steps_seen["t_first"]:
        sps = (steps_seen["t_last"] - steps_seen["t_first"]) / max(1, steps_seen["n"] - 1)
    else:
        sps = float("nan")
    cap_gb = VRAM_FRACTION * total_gb
    ded_gb = sampler.peak_proc_dedicated_mb / 1024
    report = {
        "out": str(out), "trimmed_s": trim, "frames": frames, "raw_seconds": round(raw_dur, 2), "size": f"{w}x{h}", "steps": int(job.get("steps", args.steps)), "audio_cfg": float(job.get("audio_cfg", args.audio_cfg)),
        "seed": int(job.get("seed", args.seed)), "elapsed_s": round(elapsed, 1), "load_s": round(t_load - t0, 1),
        "gen_s": round(time.time() - t_load, 1), "s_per_step": round(sps, 2) if sps == sps else None,
        "peak_vram_reserved_gb": round(peak_res, 2), "peak_vram_allocated_gb": round(peak_alloc, 2),
        "peak_process_dedicated_gb": round(ded_gb, 2),
        "peak_process_shared_gb": round(sampler.peak_proc_shared_mb / 1024, 2),
        "peak_device_used_gb_all_processes": round(sampler.peak_device_mb / 1024, 2),
        "ws_cap_gb": round(ws_cap, 1),
        "min_available_ram_gb": round(getattr(sampler, "min_free_ram_gb", -1), 1),
        "peak_commit_gb": round(getattr(sampler, "peak_commit_gb", -1), 1),
        "spill_suspected": bool(peak_res > cap_gb - 0.05 or ded_gb > total_gb - 0.4),
        "raw": str(src),
    }
    print("[talk] RESULT " + json.dumps(report))
    print(f"[talk] elapsed {elapsed:.1f}s | s/step {report['s_per_step']} | peak VRAM reserved {peak_res:.2f} GB "
          f"(allocated {peak_alloc:.2f}, process dedicated {report['peak_process_dedicated_gb']}, device all-process "
          f"{report['peak_device_used_gb_all_processes']}) | process shared {report['peak_process_shared_gb']} GB "
          f"(includes pinned RAM) | spill suspected: {report['spill_suspected']}")
    return True


if __name__ == "__main__":
    main()
