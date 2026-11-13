"""Lip-sync check with SyncNet (the LatentSync repo's eval code and its syncnet_v2 weights, ByteDance/LatentSync-1.6
auxiliary/, OpenRAIL++, same repo as the LatentSync weights the pipeline already uses; S3FD face detector from the
same folder). A trained audio-video model reads a clip with its audio and returns the audio-video offset in frames
at 25 fps and a confidence. Runs ONLY inside the gen venv; from the main interpreter use
vf.gen.bridge.run_gen('syncnet', jobs, persistent=False).

    job = dict(video=r'C:\\abs\\clip_with_audio.mp4')
    run_jobs(jobs) -> [dict(i, ok, offset, offset_ms, conf)]     offset > 0 = the mouth LAGS the voice

Measured 2026-11-06 on a test close-up: approved MuseTalk cut -3 frames (conf 13.4); LatentSync without the speed
correction +11 frames (conf 13.6, owner: "doesn't match the voice"); with the correction +2 (conf 13.1).
Needs `scenedetect==0.5.6.1 python_speech_features` (pip --no-deps into the gen venv) and the two weight files under
hero_film/tools/latentsync/checkpoints/auxiliary/.
"""
import json, os, shutil, sys, tempfile, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

LS = os.path.join(_AI, 'tools', 'latentsync')
SYNCNET_VERSION = 'syncnet-1'
_M = {}


def _model():
    if 'sn' not in _M:
        from eval.syncnet import SyncNetEval
        sn = SyncNetEval(device='cuda')
        sn.loadParameters(os.path.join(LS, 'checkpoints', 'auxiliary', 'syncnet_v2.model'))
        _M['sn'] = sn
    return _M['sn']


def check(video):
    from eval.syncnet_detect import SyncNetDetector
    work = tempfile.mkdtemp(prefix='vf_sync_')
    try:
        crops = []
        for mt in (50, 25):   # 25 = shots under 2 s (50 frames at 25 fps); a 50-frame track found first wins
            d = os.path.join(work, f'detect{mt}')
            SyncNetDetector(device='cuda', detect_results_dir=d)(video_path=video, min_track=mt)
            crops = [os.path.join(d, 'crop', c) for c in os.listdir(os.path.join(d, 'crop'))]
            if crops:
                break
        if not crops:
            return None, None
        best = None
        for c in crops:
            off, conf, _ = _model().evaluate(video_path=c,
                                             temp_dir=os.path.join(work, 'temp'))
            if best is None or conf > best[1]:
                best = (int(off), float(conf))
        return best
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    cwd = os.getcwd()
    os.chdir(LS)                       # eval.* imports and the detector's relative weight path
    sys.path.insert(0, LS)
    from ..gpulock import gpu_lock
    from .guard import process_cap
    import torch
    out = []
    try:
        with gpu_lock():
            process_cap()
            for i, j in enumerate(jobs):
                t0 = time.time()
                try:
                    off, conf = check(os.path.abspath(os.path.join(cwd, j['video'])))
                    r = dict(i=i, ok=off is not None, offset=off, offset_ms=None if off is None else off * 40,
                             conf=conf, version=SYNCNET_VERSION, seconds=round(time.time() - t0, 1),
                             **({} if off is not None else dict(error='no face track found')))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 1))
                out.append(r)
                emit(r)
            torch.cuda.empty_cache()
    finally:
        os.chdir(cwd)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.syncnet jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
