"""Frame-rate conversion with NVIDIA Video Frame Generation (nvidia-vfx, free NVIDIA SDK, Ada+): the clip's n frames
at fps_in become round((n-1)/fps_in*fps_out)+1 frames on an exact fps_out grid (same contract as vf.interp.retime).
Usage (gen venv): python -m vf.gen.framegen src.mp4 fps_in fps_out out.mp4 [MODE]"""
import os, subprocess, sys, time
import cv2, numpy as np, torch
from nvvfx import VideoFrameGeneration
from .k6_batch import FFMPEG


def retime(src, fps_in, fps_out, out, mode='HIGH'):
    cap = cv2.VideoCapture(src)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    n = len(fr)
    H, W = fr[0].shape[:2]
    n_out = int(round((n - 1) / fps_in * fps_out)) + 1
    t0 = time.time()
    tmp = out + '.part.mp4'
    from vf.proc import writer
    cmd = [FFMPEG, '-v', 'error', '-y', '-threads', '4', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r',
           f'{fps_out}', '-i', '-', '-c:v', 'h264_nvenc', '-tune', 'lossless', '-preset', 'medium', '-pix_fmt',
           'yuv420p', tmp]
    gpu = lambda f: torch.from_numpy(f).cuda().permute(2, 0, 1).float().div_(255).contiguous()
    with writer(cmd, tmp) as p, VideoFrameGeneration(mode=getattr(VideoFrameGeneration.Mode, mode),
                              automatic_shot_change_detection_enabled=False) as vfg:
        vfg.input_width, vfg.input_height = W, H
        vfg.load()
        cache = {}
        for j in range(n_out):
            x = j * fps_in / fps_out  # position in source frames
            i = min(int(x), n - 2)
            t = x - i
            if t < 1e-3:
                o = fr[i]
            elif t > 1 - 1e-3:
                o = fr[i + 1]
            else:
                a = cache.get(i) or cache.setdefault(i, (gpu(fr[i]), gpu(fr[i + 1])))
                y = torch.from_dlpack(vfg.run_at_timestep(a[0], a[1], float(t)).image).clone()  # SDK owns the buffer
                o = (y.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
                cache.pop(i - 1, None)
            p.stdin.write(np.ascontiguousarray(o).tobytes())
    os.replace(tmp, out)
    return n, n_out, time.time() - t0


if __name__ == '__main__':
    a = sys.argv[1:]
    n, m, dt = retime(a[0], float(a[1]), float(a[2]), a[3], a[4] if len(a) > 4 else 'HIGH')
    print(f'FRAMEGEN {n} -> {m} frames in {dt:.1f}s -> {a[3]}')
