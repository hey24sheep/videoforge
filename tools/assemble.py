"""Assemble a series episode for delivery: shared intro + episode + shared outro -> master + HEVC phone copy.

  python tools/assemble.py --intro <intro.mp4> --episode <ep_short.mp4> --outro <outro.mp4> --out <final.mp4>

Hard cuts between the parts (no transitions added). Audio: concatenated, then a two-pass linear loudnorm to
-14 LUFS / -1 dBTP (YouTube/Instagram). Phone copy: HEVC sized to stay under 30 MB (bitrate from the duration),
peak kept <= -1 dBFS. Prints duration, LUFS, peak, sizes and the phone copy's SSIM against the master.
"""
import argparse, json, os, re, subprocess, sys


def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def probe_dur(p):
    return float(run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p]).stdout)


def loud(p):
    e = run(['ffmpeg', '-hide_banner', '-i', p, '-af', 'ebur128=peak=true', '-f', 'null', '-']).stderr
    i = re.findall(r'I:\s+(-?[0-9.]+) LUFS', e)
    pk = re.findall(r'Peak:\s+(-?[0-9.]+) dBFS', e)
    return (float(i[-1]) if i else None, float(pk[-1]) if pk else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intro', required=True)
    ap.add_argument('--episode', required=True)
    ap.add_argument('--outro', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--phone-mb', type=float, default=26.0)  # NVENC vbr overshoots ~7 %: 29 gave 31.2 MB
    a = ap.parse_args()
    parts = [a.intro, a.episode, a.outro]
    tmp = a.out + '.concat.mp4'
    fc = ''.join(f'[{i}:v]fps=60,scale=1080:1920,setsar=1[v{i}];[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}];'
                 for i in range(3)) + '[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]'
    args = ['ffmpeg', '-v', 'error', '-y']
    for p in parts:
        args += ['-i', p]
    r = run(args + ['-filter_complex', fc, '-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc', '-preset', 'p5',
                    '-rc', 'vbr', '-cq', '18', '-b:v', '0', '-maxrate', '20M', '-bufsize', '40M', '-pix_fmt', 'yuv420p',
                    '-c:a', 'pcm_s16le', '-f', 'mov', tmp])
    if r.returncode:
        sys.exit('concat failed: ' + r.stderr[-800:])
    # two-pass linear loudnorm (pass 1 measures, pass 2 applies one gain so dynamics stay as mixed)
    m = run(['ffmpeg', '-hide_banner', '-i', tmp, '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-']).stderr
    js = json.loads(m[m.rindex('{'):m.rindex('}') + 1])
    ln = (f"loudnorm=I=-14:TP=-1.5:LRA=11:linear=true:measured_I={js['input_i']}:measured_TP={js['input_tp']}:"
          f"measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}")
    r = run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-c:v', 'copy', '-af', ln + ',aresample=48000', '-c:a', 'aac',
             '-b:a', '192k', '-movflags', '+faststart', a.out])
    if r.returncode:
        sys.exit('master failed: ' + r.stderr[-800:])
    os.remove(tmp)
    dur = probe_dur(a.out)
    phone = a.out[:-4] + '_phone.mp4'
    vb = max(1.2, (a.phone_mb * 8 / dur) - 0.14)  # Mbit/s for video after 128 kb/s audio
    r = run(['ffmpeg', '-v', 'error', '-y', '-i', a.out, '-c:v', 'hevc_nvenc', '-preset', 'p7', '-tune', 'hq', '-rc',
             'vbr', '-b:v', f'{vb:.2f}M', '-maxrate', f'{vb * 1.5:.2f}M', '-bufsize', f'{vb * 2.6:.2f}M', '-multipass',
             'fullres', '-spatial-aq', '1', '-temporal-aq', '1', '-tag:v', 'hvc1', '-af', 'volume=-0.3dB', '-c:a', 'aac',
             '-b:a', '128k', '-movflags', '+faststart', phone])
    if r.returncode:
        sys.exit('phone copy failed: ' + r.stderr[-800:])
    ss = re.findall(r'All:([0-9.]+)', run(['ffmpeg', '-hide_banner', '-i', a.out, '-i', phone, '-lavfi',
                                           '[0:v][1:v]ssim', '-f', 'null', '-']).stderr)
    for p in (a.out, phone):
        i, pk = loud(p)
        print(json.dumps(dict(file=p, seconds=round(probe_dur(p), 2), mb=round(os.path.getsize(p) / 1e6, 2),
                              lufs=i, peak_dbfs=pk)))
    print(json.dumps(dict(phone_ssim=float(ss[-1]) if ss else None, phone_video_mbps=round(vb, 2))))


if __name__ == '__main__':
    main()
