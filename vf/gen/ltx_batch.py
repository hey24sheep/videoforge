"""LTX-2.5 distilled (Lightricks, 22B, int8) image-to-video through WanGP, one process for a list of clips.
Runs in the talk venv (it has WanGP). Called by vf/k6.py for shots with `gen: {engine: ltx}`; the parent holds the
GPU lock, so this child must not take it.

    <talk venv python> -m vf.gen.ltx_batch jobs.json [--online]

jobs.json: [{image, prompt, seed, frames, width, height, out}]. Prints one JSON line per clip:
{"i", "ok", "out", "seconds", "frames"} or {"i", "ok": false, "error"}.
Frames are rounded up to 8k+1 (LTX needs it); sizes must be multiples of 64 (LTX snaps others down).
The first run downloads the model files (~30 GB) unless --offline is in force (VF_LTX_ONLINE=0).
Measured 2026-11-11 (RTX 4060 8 GB laptop, 32 GB RAM): 384x672 -> 97 frames in ~47 s, 8 steps at ~3.2 s/step,
peak ~2 GB VRAM (WanGP profile 4 streams the weights). Licence (model card, checked 2026-11-11): free, commercial use
allowed under $10M yearly revenue; read it before commercial use.
"""
import json, math, os, subprocess, sys, tempfile, time
from pathlib import Path

AI = Path(os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film'))
WAN2GP = AI / 'tools' / 'wan2gp'
CFG = AI / 'tools' / 'wan2gp_cfg' / 'wgp_config.json'
MODEL = os.environ.get('VF_LTX_MODEL', 'ltx2_25_22B_distilled')
STEPS = int(os.environ.get('VF_LTX_STEPS', 8))


def main():
    jobs = json.load(open(sys.argv[1]))
    online = '--online' in sys.argv or os.environ.get('VF_LTX_ONLINE', '1') == '1'
    import torch
    torch.cuda.set_per_process_memory_fraction(0.92, 0)
    sys.path.insert(0, str(WAN2GP))
    os.chdir(WAN2GP)
    from shared.api import init
    d = json.load(open(CFG, encoding='utf-8'))
    d['transformer_quantization'] = 'int8'
    work = Path(tempfile.mkdtemp(prefix='vf_ltx_'))
    cfg = work / 'wgp_config.json'
    json.dump(d, open(cfg, 'w', encoding='utf-8'), indent=1)
    cli = ([] if online else ['--offline']) + ['--attention', 'auto', '--profile', '4', '--max-ram', '20',
                                               '--perc-reserved-mem-max', '0.2', '--verbose', '0']
    session = init(root=WAN2GP, config_path=cfg, output_dir=work, cli_args=cli)
    for i, j in enumerate(jobs):
        t0 = time.time()
        try:
            n = int(j['frames'])
            n = 8 * math.ceil((n - 1) / 8) + 1
            s = dict(model_type=MODEL, prompt=j['prompt'], resolution=f"{int(j['width'])}x{int(j['height'])}",
                     video_length=n, batch_size=1, seed=int(j['seed']), num_inference_steps=STEPS,
                     repeat_generation=1, image_prompt_type='S', image_start=[str(Path(j['image']).resolve())],
                     video_prompt_type='', audio_prompt_type='', prompt_enhancer='', temporal_upsampling='',
                     spatial_upsampling='', film_grain_intensity=0)
            task = session.submit_task(s)
            for _ in task.events.iter(timeout=0.5):
                pass
            r = task.result()
            vids = [f for f in r.generated_files if str(f).lower().endswith('.mp4')] if r.success else []
            if not vids:
                raise RuntimeError('; '.join(str(getattr(e, 'message', e)) for e in r.errors) or 'no video')
            # video only (LTX also makes a soundtrack), 24 fps, lossless-ish for the finishing chain
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(vids[-1]), '-an', '-r', '24', '-c:v', 'libx264',
                            '-crf', '12', '-preset', 'fast', '-pix_fmt', 'yuv420p', j['out']], check=True)
            print(json.dumps(dict(i=i, ok=True, out=j['out'], seconds=round(time.time() - t0, 1), frames=n)), flush=True)
        except Exception as e:
            print(json.dumps(dict(i=i, ok=False, error=f'{type(e).__name__}: {e}')), flush=True)


if __name__ == '__main__':
    main()
