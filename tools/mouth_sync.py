"""Strict lip-sync check that matches what a viewer sees: does the mouth move when there is NO speech, and stay
still when there IS? (SyncNet only gives one global offset and passed clips the owner called out of sync.)

Models: MediaPipe Face Landmarker (Apache-2.0) -> inner-lip opening per frame (normalised by face size).
Audio: speech activity from the waveform energy envelope (exact fact, not a judgement).
Run with the talk venv:  python tools/mouth_sync.py <clip.mp4> [<clip2.mp4> ...] [--json out.json]

Reports per clip:
  lag_ms        best cross-correlation lag between mouth opening and audio energy (+ = mouth late)
  corr          correlation at that lag (talking-head clips with real sync: ~0.5+)
  talk_in_silence  share of SILENT frames where the mouth is moving/open (should be ~0)
  still_in_speech  share of SPEECH frames where the mouth is closed and still (should be low)
STRICT PASS: |lag| <= 80 ms, corr >= 0.45, talk_in_silence <= 0.15, still_in_speech <= 0.35.
"""
import json, subprocess, sys, tempfile, os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import numpy as np
import cv2
import mediapipe as mp

MODEL = os.path.join(_AI, 'models', 'mediapipe', 'face_landmarker.task')
UP, LO, L, R = 13, 14, 61, 291        # inner upper/lower lip, mouth corners
FT, CH = 10, 152                       # forehead, chin (face height)


def mouth_curve(path):
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL), running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1)
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    # medium shots (small face): retry a missed frame cropped to the top + upscaled 3x in IMAGE mode (2026-11-09, walt_turn had no face found)
    lm2 = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL), running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=1))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vals, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        r = lm.detect_for_video(img, int(i * 1000 / fps))
        if not r.face_landmarks:
            for crop in (f[:int(f.shape[0] * 0.55)], f):   # face in the upper half of a 9:16 medium shot; ratio is scale-free
                big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                r = lm2.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
                if r.face_landmarks:
                    break
        if r.face_landmarks:
            p = r.face_landmarks[0]
            face_h = abs(p[CH].y - p[FT].y) or 1e-6
            vals.append(abs(p[LO].y - p[UP].y) / face_h)
        else:
            vals.append(np.nan)
        i += 1
    lm.close()
    lm2.close()
    v = np.array(vals, dtype=float)
    if np.isnan(v).all():
        raise RuntimeError('no face found')
    mouth_curve.nan = np.isnan(v)   # frames with no face found (callers must not judge them; 2026-11-07)
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    return v, fps


def audio_env(path, fps, n):
    tmp = tempfile.mktemp(suffix='.f32')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', tmp],
                   check=True)
    a = np.fromfile(tmp, dtype=np.float32)
    os.remove(tmp)
    hop = 16000 / fps
    a = np.pad(a, (0, max(0, int(np.ceil(n * hop)) + 1 - len(a))))  # video longer than audio: empty windows gave NaN
    e = np.array([np.sqrt(np.mean(a[int(k * hop):int((k + 1) * hop)] ** 2) + 1e-12) for k in range(n)])
    db = 20 * np.log10(e)
    speech = db > (np.max(db) - 30)          # within 30 dB of the loudest frame = speech
    return db, speech


def check(path):
    m, fps = mouth_curve(path)
    db, speech = audio_env(path, fps, len(m))
    ms = np.convolve(m, np.ones(3) / 3, mode='same')
    mz = (ms - ms.mean()) / (ms.std() + 1e-9)
    az = (db - db.mean()) / (db.std() + 1e-9)
    best = (0, -9)
    for lag in range(-25, 26):                # +lag: mouth late (+-1 s at 25 fps)
        a2 = az[:len(az) - lag] if lag >= 0 else az[-lag:]
        m2 = mz[lag:] if lag >= 0 else mz[:len(mz) + lag]
        c = float(np.corrcoef(a2, m2)[0, 1])
        if c > best[1]:
            best = (lag, c)
    closed = np.percentile(ms, 20)
    open_thr = closed + 0.35 * (np.percentile(ms, 90) - closed)
    motion = np.abs(np.gradient(ms))
    mov_thr = np.percentile(motion, 60)
    talking = (ms > open_thr) | (motion > mov_thr)
    sil = ~speech
    tis = float(np.mean(talking[sil])) if sil.any() else 0.0
    sis = float(np.mean(~talking[speech])) if speech.any() else 0.0
    lag_ms = round(best[0] * 1000 / fps)
    ok = abs(lag_ms) <= 80 and best[1] >= 0.45 and tis <= 0.15 and sis <= 0.35
    return dict(clip=path, frames=len(m), fps=round(fps, 2), lag_ms=lag_ms, corr=round(best[1], 3),
                talk_in_silence=round(tis, 3), still_in_speech=round(sis, 3),
                silent_frames=int(sil.sum()), speech_frames=int(speech.sum()), strict_pass=bool(ok))


if __name__ == '__main__':
    args = sys.argv[1:]
    out = None
    if '--json' in args:
        k = args.index('--json')
        out = args[k + 1]
        args = args[:k] + args[k + 2:]
    res = [check(p) for p in args]
    for r in res:
        print(json.dumps(r))
    if out:
        json.dump(res, open(out, 'w'), indent=1)
