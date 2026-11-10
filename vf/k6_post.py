"""Post-process generated clips: RIFE retime at source resolution (cheap) -> NVIDIA RTX VSR to 1080x1920.
Usage (render python): python -m vf.k6_post <jobs.json> [id=speed ...]
Writes <out>_up.mp4 next to each finished clip; skips ones already done. Speed s -> 60/s fps (slow motion)."""
import json, os, subprocess, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
from .interp import retime, target_fps

GEN_PY = os.path.join(_AI, 'tools', 'envs', 'gen', 'Scripts', 'python.exe')


def main(path, speeds):
    for j in json.load(open(path)):
        src = j['out']
        up = src[:-4] + '_up.mp4'
        if not os.path.exists(src) or (os.path.exists(up) and os.path.getmtime(up) > os.path.getmtime(src)):
            continue
        t = time.time()
        r = retime(src, 24, target_fps(60, speeds.get(j['id'], 1.0)))
        subprocess.run([GEN_PY, '-m', 'vf.gen.rtx_vsr', r, up], check=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(f"POST {j['id']} {time.time() - t:.0f}s -> {up}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], {a.split('=')[0]: float(a.split('=')[1]) for a in sys.argv[2:]})
