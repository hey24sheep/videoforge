"""High-quality clip finishing for the compositor (render python):
  raw Kandinsky clip (384x672, 24 fps) -> [MuseTalk lip-sync at source res, so the mouth gets the same upscale as
  the face: no soft patch / seam] -> RIFE retime to 60/speed fps -> NVIDIA RTX VSR 4x (1536x2688).
4x keeps camera push-ins as DOWN-sampling in the compositor (a 1080 source zoomed 1.2x was soft).
Usage: python -m vf.k6_hq plan.json   plan = [{name, src, speed?, wav?, offset?}]; writes clips_k6/<name>_hq.mp4"""
import json, os, subprocess, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
from .interp import retime, target_fps

GEN_PY = os.path.join(_AI, 'tools', 'envs', 'gen', 'Scripts', 'python.exe')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1536, 2688


def main(plan_path):
    plan = json.load(open(plan_path))
    base = os.path.dirname(os.path.abspath(plan_path))
    for p in plan:
        src = os.path.join(base, p['src'])
        out = os.path.join(base, 'clips_k6', p['name'] + '_hq.mp4')
        if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(src) and not p.get('force'):
            continue
        t = time.time()
        if p.get('wav'):  # lip-sync the raw clip first
            ls = os.path.join(base, 'clips_k6', p['name'] + '_rawls.mp4')
            subprocess.run([GEN_PY, '-m', 'vf.gen.lipsync', src, os.path.join(ROOT, p['wav']), str(p.get('offset', 0)),
                            ls], check=True, cwd=ROOT)
            src = ls
        r = retime(src, 24, target_fps(60, p.get('speed', 1.0)))
        subprocess.run([GEN_PY, '-m', 'vf.gen.rtx_vsr', r, out, str(W), str(H)], check=True, cwd=ROOT)
        print(f"HQ {p['name']} {time.time() - t:.0f}s fps={target_fps(60, p.get('speed', 1.0)):.4f} -> {out}",
              flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
