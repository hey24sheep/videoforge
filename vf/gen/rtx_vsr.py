"""NVIDIA RTX Video Super Resolution (nvidia-vfx, free NVIDIA SDK; Tensor-Core GPUs) on an mp4: AI upscale +
compression/AI-artifact cleanup, a few ms per frame. Usage (gen venv):
    python -m vf.gen.rtx_vsr src.mp4 out.mp4 [W H] [QUALITY]
Defaults 1080x1920 ULTRA. Writes h264 crf 12, same fps."""
import os, subprocess, sys, time
import cv2, numpy as np, torch
from nvvfx import VideoSuperRes, QualityLevel
from .k6_batch import FFMPEG


def upscale(src, out, W=1080, H=1920, quality='ULTRA', strength=1.0):
    from vf.proc import writer
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    t, n = time.time(), 0
    tmp = out + '.part.mp4'
    # Final-size output (>= 1 MP: the 2x pass, 768x1344 / 960x1728) is near-lossless NVENC cq 13 (~6x smaller than
    # lossless; the compositor re-samples, grains and re-encodes it anyway). Native-size intermediates (the 1x deblur
    # pass, read again by frame generation) stay lossless. (Was `W >= 1080`, so portrait finals were lossless:
    # 135 MB per clip, 12 GB of cache.)
    enc = (['-c:v', 'h264_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '13', '-b:v', '0'] if W * H >= 1_000_000
           else ['-c:v', 'h264_nvenc', '-tune', 'lossless'])
    cmd = [FFMPEG, '-v', 'error', '-y', '-threads', '4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r',
           f'{fps}', '-i', '-', *enc, '-pix_fmt', 'yuv420p', tmp]
    try:
        with writer(cmd, tmp) as p, VideoSuperRes(quality=getattr(QualityLevel, quality), strength=strength) as sr:
            sr.output_width, sr.output_height = W, H
            sr.load()
            while True:  # streamed: the whole clip is never held in RAM
                ok, f = cap.read()
                if not ok:
                    break
                x = torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1).float().div_(255).contiguous()
                y = torch.from_dlpack(sr.run(x).image)
                p.stdin.write((y.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).contiguous().cpu().numpy().tobytes())
                n += 1
    finally:
        cap.release()
    os.replace(tmp, out)
    return n, time.time() - t


if __name__ == '__main__':
    a = sys.argv[1:]
    W, H = (int(a[2]), int(a[3])) if len(a) >= 4 else (1080, 1920)
    q = a[4] if len(a) >= 5 else 'ULTRA'
    n, dt = upscale(a[0], a[1], W, H, q)
    print(f'RTX VSR {q} {W}x{H}: {n} frames in {dt:.1f}s ({n / dt:.0f} fps) -> {a[1]}')
