"""Dialogue delivery checks on the spoken lines (runs in the TTS venv, CPU; from the main interpreter use
vf.gen.bridge.run_gen('delivery', jobs) which routes to that venv).

    job = dict(lines=[dict(wav=r'C:\\abs\\line.wav', who='ada', text='...'), ...])
    -> dict(i, ok, lines=[{wav, who, mos, same_voice}], ...)

Models:
  - Distill-MOS v7 (Microsoft, MIT, weights inside the pip package): predicted listener naturalness 1-5.
  - speechbrain ECAPA-TDNN spkrec-ecapa-voxceleb (Apache-2.0): speaker embedding; same_voice = cosine similarity of
    a line to the mean of the SAME character's other lines (leave-one-out). A take that drifts into another voice
    scores low. Characters with one line get no score.
"""
import json, os, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
DELIVERY_VERSION = 'delivery-1'
_M = {}


def _load(path):
    import torch, torchaudio
    x, sr = torchaudio.load(path)
    x = x.mean(0, keepdim=True)
    if sr != 16000:
        x = torchaudio.functional.resample(x, sr, 16000)
    return x


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    import torch
    torch.set_num_threads(4)
    out = []
    for i, j in enumerate(jobs):
        t0 = time.time()
        try:
            if 'mos' not in _M:
                import distillmos
                from speechbrain.inference.speaker import EncoderClassifier
                from speechbrain.utils.fetching import LocalStrategy
                _M['mos'] = distillmos.ConvTransformerSQAModel().eval()
                from huggingface_hub import snapshot_download
                # speechbrain 1.0.3 (pinned: 1.1 needs torch 2.7, Chatterbox needs 2.6) passes an argument newer
                # huggingface_hub rejects: download first, then load from the local folder
                snap = snapshot_download('speechbrain/spkrec-ecapa-voxceleb')
                # local sources must contain the optional custom.py (speechbrain only skips it for remote fetches)
                src = os.path.join(os.environ['HF_HOME'], 'speechbrain', 'ecapa_src')
                if not os.path.exists(os.path.join(src, 'custom.py')):
                    import shutil
                    os.makedirs(src, exist_ok=True)
                    for fn in os.listdir(snap):
                        if os.path.isfile(os.path.join(snap, fn)):
                            shutil.copy(os.path.join(snap, fn), os.path.join(src, fn))
                    open(os.path.join(src, 'custom.py'), 'w').close()
                _M['spk'] = EncoderClassifier.from_hparams(
                    source=src, overrides={'pretrained_path': src.replace(os.sep, '/')},
                    savedir=os.path.join(os.environ['HF_HOME'], 'speechbrain', 'spkrec-ecapa-voxceleb'),
                    run_opts={'device': 'cpu'}, local_strategy=LocalStrategy.COPY)   # Windows: no symlink right
            res, emb = [], []
            with torch.inference_mode():
                for ln in j['lines']:
                    x = _load(ln['wav'])
                    mos = float(_M['mos'](x).mean())
                    e = _M['spk'].encode_batch(x).squeeze()
                    emb.append(e / e.norm())
                    res.append(dict(wav=ln['wav'], who=ln.get('who'), text=ln.get('text'), mos=round(mos, 3)))
            for k, r in enumerate(res):
                same = [emb[m] for m in range(len(res)) if m != k and res[m]['who'] == r['who']]
                if same:
                    c = torch.stack(same).mean(0)
                    r['same_voice'] = round(float(emb[k] @ (c / c.norm())), 3)
            r = dict(i=i, ok=True, lines=res, version=DELIVERY_VERSION, seconds=round(time.time() - t0, 1))
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 1))
        out.append(r)
        emit(r)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.delivery jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
