"""Clip upscaling with Real-ESRGAN (ncnn-vulkan), cached globally. 480p AI clips -> 2x/4x before compositing.

The ncnn build enumerates Vulkan devices; on laptops device 0 is often the iGPU, so the NVIDIA device is picked by
name once and remembered.
"""
import glob, os, re, shutil, subprocess, tempfile
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
from .memo import memo_file
from .render import ffmpeg

ESRGAN = os.environ.get('VF_ESRGAN', os.path.join(_AI, 'tools', 'esrgan', 'realesrgan-ncnn-vulkan.exe'))
_GPU = {}


def nvidia_index():
    if 'i' not in _GPU:
        idx = 0
        try:
            d = tempfile.mkdtemp()
            from PIL import Image
            Image.new('RGB', (16, 16)).save(os.path.join(d, 'a.png'))
            r = subprocess.run([ESRGAN, '-i', os.path.join(d, 'a.png'), '-o', os.path.join(d, 'b.png'), '-g', '0'],
                               capture_output=True, text=True, timeout=60)
            names = dict(re.findall(r'\[(\d+) ([^\]]+)\]', r.stderr + r.stdout))
            for k, v in names.items():
                if re.search('nvidia|geforce|rtx', v, re.I):
                    idx = int(k)
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        _GPU['i'] = int(os.environ.get('VF_ESRGAN_GPU', idx))
    return _GPU['i']


@memo_file('global', '.mp4')
def upscale_clip(src, scale=2, model='realesr-animevideov3', fps=16, out=None):
    """mp4 (or frame folder) -> upscaled mp4 (lossless-ish h264 at cq 12) at the same fps."""
    work = tempfile.mkdtemp(prefix='vf_up_')
    fin, fout = os.path.join(work, 'in'), os.path.join(work, 'out')
    os.makedirs(fin)
    os.makedirs(fout)
    if os.path.isdir(src):
        for i, f in enumerate(sorted(glob.glob(os.path.join(src, '*.png')))):
            shutil.copy(f, os.path.join(fin, f'{i:05d}.png'))
    else:
        subprocess.run([ffmpeg(), '-v', 'error', '-i', src, os.path.join(fin, '%05d.png')], check=True)
    subprocess.run([ESRGAN, '-i', fin, '-o', fout, '-n', model, '-s', str(scale), '-g', str(nvidia_index()),
                    '-f', 'png'], check=True, capture_output=True)
    subprocess.run([ffmpeg(), '-v', 'error', '-y', '-framerate', str(fps), '-i', os.path.join(fout, '%05d.png'),
                    '-c:v', 'h264_nvenc', '-preset', 'p6', '-rc', 'vbr', '-cq', '12', '-b:v', '0', '-pix_fmt',
                    'yuv444p', out], check=True)
    shutil.rmtree(work, ignore_errors=True)
