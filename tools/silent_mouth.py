"""Frame-level gate: is the mouth open while nobody speaks? (Owner's #1 complaint, 2026-11-07.)

Per frame: mouth opening (MediaPipe Face Landmarker, Apache-2.0, inner-lip gap / face height) and speech (audio RMS
within 35 dB of the clip's loudest 40 ms frame = exact fact). A frame is "open" when the opening is above
closed + 0.30 * (open - closed), where closed / open are the 10th / 90th percentiles of that face's opening.

  python tools/silent_mouth.py clip.mp4 [...]            report open-and-silent runs (talk venv)
  python tools/silent_mouth.py --analyse clip.mp4 timeline.wav out.json   per-frame arrays (vf/talk.py)
  python tools/silent_mouth.py --cut clip.mp4 out.mp4    write the clip cut so the head/tail open-silent frames are gone:
        start = first frame of the closed run before speech, else first speech frame - 1
        end   = last closed frame after speech (keeps closed pauses), else last speech frame + 1
GATE: no open-and-silent run longer than MAX_RUN frames anywhere in the clip.
"""
import json, os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import mouth_sync as M

MAX_RUN = 1


def frames(path, audio=None):
    """audio: a separate wav aligned to the clip's frame 0 (default: the clip's own audio track)."""
    m, fps = M.mouth_curve(path)
    tmp = tempfile.mktemp(suffix='.f32')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', audio or path, '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le',
                    tmp], check=True)
    a = np.fromfile(tmp, dtype=np.float32)
    os.remove(tmp)
    hop = 16000 / fps
    a = np.pad(a, (0, max(0, int(np.ceil(len(m) * hop)) + 1 - len(a))))
    db = np.array([20 * np.log10(np.sqrt(np.mean(a[int(k * hop):int((k + 1) * hop)] ** 2)) + 1e-9) for k in range(len(m))])
    speech = db > db.max() - 35
    seen = ~getattr(M.mouth_curve, 'nan', np.zeros(len(m), bool))   # frames where a face was found
    lo, hi = np.percentile(m[seen], 10), np.percentile(m[seen], 90)
    is_open = (m > lo + 0.30 * (hi - lo)) & seen   # no face found = not judged (the median fill read as 'open')
    return m, is_open, speech, db, fps


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def report(path, audio=None):
    m, op, sp, db, fps = frames(path, audio)
    bad = runs(op & ~sp)
    worst = max([b - a + 1 for a, b in bad], default=0)
    return dict(clip=path, frames=len(m), fps=fps, open_silent_runs=[(round(a / fps, 2), b - a + 1) for a, b in bad],
                worst_run_frames=worst, ok=worst <= MAX_RUN,
                timeline=''.join('S' if s and o else 's' if s else 'X' if o else '.' for o, s in zip(op, sp)))


def cut(path, out):
    m, op, sp, db, fps = frames(path)
    idx = np.flatnonzero(sp)
    on, off = int(idx[0]), int(idx[-1])
    s = on
    while s - 1 >= 0 and not sp[s - 1] and not op[s - 1]:
        s -= 1                       # closed silent lead: keep it
    if s == on:
        s = max(0, on - 1)
    elif op[s:on].any():             # open-silent frames between the closed lead and speech
        s = max(0, on - 1)
    e = off
    while e + 1 < len(m) and not sp[e + 1] and not op[e + 1]:
        e += 1
    if e == off or op[off + 1:e + 1].any():
        e = min(len(m) - 1, off + 1)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-vf', f'trim=start_frame={s}:end_frame={e + 1},setpts=PTS-STARTPTS',
                    '-af', f'atrim=start={s / fps:.4f}:end={(e + 1) / fps:.4f},asetpts=PTS-STARTPTS', '-c:v', 'libx264',
                    '-crf', '12', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                    '-ac', '2', out], check=True)
    return dict(src=path, out=out, start_frame=s, end_frame=e, speech=(on, off))


def analyse(clip, audio, out):
    """Per-frame arrays for vf/talk.py: mouth opening, open flag, speech flag (audio = timeline-aligned wav)."""
    m, op, sp, db, fps = frames(clip, audio)
    seen = ~getattr(M.mouth_curve, 'nan', np.zeros(len(m), bool))
    json.dump(dict(fps=fps, mouth=[round(float(x), 5) for x in m], open=[bool(x) for x in op],
                   speech=[bool(x) for x in sp], seen=[bool(x) for x in seen]), open(out, 'w'))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--analyse':
        analyse(a[1], a[2], a[3])
    elif a and a[0] == '--cut':
        print(json.dumps(cut(a[1], a[2])))
    elif a and a[0] == '--audio':   # --audio stem.wav clip.mp4: judge speech on a separate (dialogue-only) track
        print(json.dumps(report(a[2], a[1])))
    else:
        for p in a:
            print(json.dumps(report(p)))
