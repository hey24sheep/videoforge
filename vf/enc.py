"""Video encoder choice: NVIDIA NVENC when it works, else the CPU encoders (libx264 / libx265).

VF_ENCODER = auto (default) | nvenc | cpu. `auto` tries a tiny NVENC encode once per process.
This module is part of the renderer source hash (render.renderer_code). CPU encodes are slower and their files
differ from NVENC ones, so segments made with the CPU encoder get their own cache key.
"""
import functools, os, subprocess


@functools.lru_cache(None)
def nvenc():
    """True when segments and cuts use NVENC."""
    mode = os.environ.get('VF_ENCODER', 'auto').lower()
    if mode in ('cpu', 'x264', 'libx264'):
        return False
    if mode == 'nvenc':
        return True
    try:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=256x256:d=0.1',
                            '-c:v', 'h264_nvenc', '-f', 'null', '-'], capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def name():
    return 'nvenc' if nvenc() else 'cpu (libx264/libx265)'


def master(W, H, fps, cq=18, preset='p4'):
    """Shot segment: high quality H.264, BT.709 tagged, capped bitrate."""
    # cq 18 with a ceiling: YouTube recommends ~12 Mbps for 1080p60 and re-encodes anyway (4K scales up)
    maxrate = max(8, int(20 * W * H / (1920 * 1080)))
    rate = ['-maxrate', f'{maxrate}M', '-bufsize', f'{maxrate * 2}M', '-g', str(fps * 2)]
    color = ['-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']
    # both encoders can drop primaries/transfer from raw input: write all three into the H.264 VUI so players don't guess
    vui = ['-bsf:v', 'h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1']
    if nvenc():
        return (['-c:v', 'h264_nvenc', '-preset', preset, '-tune', 'hq', '-rc', 'vbr', '-cq', str(cq), '-b:v', '0']
                + rate + color + vui)
    return ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(cq), '-pix_fmt', 'yuv420p'] + rate + color + vui


def intermediate():
    """Near-lossless working clip (plate lock, stamps, face tone): NVENC on the GPU keeps the CPU (and its heat) free;
    libx264 crf 12 only without NVENC."""
    if nvenc():
        return ['-c:v', 'h264_nvenc', '-preset', 'p5', '-tune', 'hq', '-rc', 'vbr', '-cq', '12', '-b:v', '0',
                '-pix_fmt', 'yuv420p']
    return ['-c:v', 'libx264', '-crf', '12', '-preset', 'medium', '-pix_fmt', 'yuv420p']


def join():
    """Full cut (intro + episode + outro)."""
    if nvenc():
        return ['-c:v', 'h264_nvenc', '-preset', 'p6', '-cq', '19']
    return ['-c:v', 'libx264', '-preset', 'medium', '-crf', '19']


def phone(kbps):
    """Phone copy: HEVC at a target bitrate."""
    rate = ['-b:v', f'{kbps}k', '-maxrate', f'{int(kbps * 1.6)}k', '-bufsize', f'{int(kbps * 3.2)}k']
    if nvenc():
        return ['-c:v', 'hevc_nvenc', '-preset', 'p6'] + rate + ['-tag:v', 'hvc1']
    return ['-c:v', 'libx265', '-preset', 'fast', '-x265-params', 'log-level=error'] + rate + ['-tag:v', 'hvc1']


def phone_overshoot():
    """Share of the size budget to aim for: NVENC VBR overshoots ~15 % (measured 2026-11-09); x265 ABR stays close."""
    return 0.82 if nvenc() else 0.92
