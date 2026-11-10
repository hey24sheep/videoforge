"""Local text-to-music via ACE-Step 1.5 turbo (MIT). Run with the music venv:
D:/claude_work/hero_film/tools/envs/music/Scripts/python.exe make_music.py --prompt "..." --dur 100 --out x.wav [--seed N] [--lyrics ""]
DiT-only (no LM), bf16, CPU offload (DiT+VAE+text enc). Instrumental unless --lyrics given."""
import argparse, os, sys, shutil, time, tempfile
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
REPO = os.path.join(_AI, 'tools', 'ACE-Step-1.5')
CKPT = os.path.join(_AI, 'models', 'hf', 'ace-step-1.5', 'checkpoints')
os.environ["HF_HOME"] = os.path.join(_AI, 'models', 'hf')
os.environ["ACESTEP_CHECKPOINTS_DIR"] = CKPT
os.environ["ACESTEP_PROJECT_ROOT"] = REPO
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, REPO)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--dur", type=float, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", default=None, help="JSON list of {prompt, dur, out, seed}: model loads once (load ~3.5 min, 30 s audio ~11 s)")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--lyrics", default="")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    dev = a.device
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=REPO, config_path="acestep-v15-turbo", device=dev,
                                     use_flash_attention=False, compile_model=False,
                                     offload_to_cpu=True, offload_dit_to_cpu=True, quantization=None)
    if not ok:
        sys.exit("init failed: " + msg)
    import json
    jobs = json.load(open(a.jobs)) if a.jobs else [dict(prompt=a.prompt, dur=a.dur, out=a.out, seed=a.seed)]
    instr = not a.lyrics.strip()
    for j in jobs:
        seed = int(j.get("seed", -1))
        p = GenerationParams(caption=j["prompt"], lyrics="[Instrumental]" if instr else a.lyrics, instrumental=instr,
                             vocal_language="unknown" if instr else "en", duration=float(j["dur"]),
                             inference_steps=a.steps, seed=seed, thinking=False, use_cot_metas=False,
                             use_cot_caption=False, use_cot_language=False,
                             # optional audio-conditioned jobs (style match to an approved take); absent = text2music as before
                             **{k: j[k] for k in ("task_type", "reference_audio", "src_audio", "audio_cover_strength", "repainting_start", "repainting_end", "repaint_mode", "repaint_strength", "repaint_wav_crossfade_sec", "bpm", "keyscale", "timesignature")
                                if k in j})
        c = GenerationConfig(batch_size=1, use_random_seed=seed < 0, seeds=None if seed < 0 else [seed],
                             audio_format="wav")
        tmp = tempfile.mkdtemp(prefix="acemusic_")
        t = time.time()
        r = generate_music(dit, LLMHandler(), p, c, save_dir=tmp)
        if not r.success:
            print("FAILED " + j["out"] + ": " + str(r.error), flush=True)
            continue
        os.makedirs(os.path.dirname(os.path.abspath(j["out"])), exist_ok=True)
        shutil.copyfile(r.audios[0]["path"], j["out"])
        shutil.rmtree(tmp, ignore_errors=True)
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        print(f"OK {j['out']} dur={j['dur']}s elapsed={time.time()-t:.1f}s peakVRAM={peak:.2f}GiB "
              f"seed={r.audios[0]['params'].get('seed')}", flush=True)

if __name__ == "__main__":
    main()
