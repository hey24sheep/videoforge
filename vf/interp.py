"""Motion-compensated frame interpolation (RIFE v4.6, ncnn-vulkan), cached globally.

AI clips come out at 24 fps; the episode plays at 60 fps, often slowed (`speed: 0.79`). Holding the nearest
24 fps frame gives an uneven 2:3 cadence that reads as judder/shake, worse when slowed. `retime()` instead
interpolates the clip to exactly `fps_out / speed` frames per clip-second, so every output frame lands on a real
(interpolated) frame: smooth, film-like motion including slow motion.
"""
import glob, os, shutil, subprocess, tempfile
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
from .memo import memo_file

RIFE = os.environ.get('VF_RIFE', os.path.join(_AI, 'tools', 'rife', 'rife-ncnn-vulkan-20221029-windows',
                                              'rife-ncnn-vulkan.exe'))
MODEL = 'rife-v4.6'


def target_fps(fps_out, speed=1.0):
    return float(fps_out) / max(0.05, float(speed))


def _probe_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return n, fps


@memo_file('global', '.mp4', version=lambda: f'retime-1|{MODEL}')
def retime(src, fps_in, fps_target, out=None):
    """mp4 at fps_in -> mp4 holding round((n-1)/fps_in*fps_target)+1 frames (same time span). Play it with
    layer fps = fps_target. Frames go through PNG (RIFE's I/O); written near-lossless (x264 crf 12)."""
    from .render import ffmpeg
    from .upscale import nvidia_index
    n_in, _ = _probe_frames(src)
    n_out = int(round((n_in - 1) / float(fps_in) * float(fps_target))) + 1
    work = tempfile.mkdtemp(prefix='vf_rife_')
    try:
        fin, fout = os.path.join(work, 'in'), os.path.join(work, 'out')
        os.makedirs(fin)
        os.makedirs(fout)
        subprocess.run([ffmpeg(), '-v', 'error', '-i', src, os.path.join(fin, '%06d.png')], check=True)
        exe_dir = os.path.dirname(RIFE)
        subprocess.run([RIFE, '-i', fin, '-o', fout, '-n', str(n_out), '-m', os.path.join(exe_dir, MODEL),
                        '-g', str(nvidia_index()), '-j', '2:2:2', '-f', '%08d.png'], check=True, capture_output=True)
        got = len(glob.glob(os.path.join(fout, '*.png')))
        if got < 2:
            raise RuntimeError(f'RIFE produced {got} frames for {src}')
        # container rate is nominal (the layer carries the exact fps); keep it an integer for players/decoders
        subprocess.run([ffmpeg(), '-v', 'error', '-y', '-framerate', str(max(1, int(round(fps_target)))),
                        '-i', os.path.join(fout, '%08d.png'), '-c:v', 'libx264', '-preset', 'fast', '-crf', '12',
                        '-pix_fmt', 'yuv420p', out], check=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
