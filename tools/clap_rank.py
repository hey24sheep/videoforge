"""Rank music candidates with an open text-audio model (LAION CLAP `laion/clap-htsat-unfused`, Apache-2.0), CPU only.

score = cos(audio, target text) - cos(audio, unwanted text). The unwanted prompt catches what the owner never wants
under dialogue (vocals, drums, upbeat pop). Run with the gen venv:
  D:/claude_work/hero_film/tools/envs/gen/Scripts/python.exe tools/clap_rank.py --target "..." --avoid "..." a.wav b.wav
"""
import argparse, json, os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def _vec(o):  # newer transformers return a ModelOutput, older a tensor
    return o if hasattr(o, 'norm') else (o.pooler_output if getattr(o, 'pooler_output', None) is not None else o[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--avoid', action='append', default=None, help='repeatable; default: vocals, drums, happy pop')
    ap.add_argument('wavs', nargs='+')
    a = ap.parse_args()
    import numpy as np, torch, soundfile as sf
    from scipy.signal import resample_poly
    from transformers import ClapModel, ClapProcessor
    mid = 'laion/clap-htsat-unfused'  # larger_clap_music loads with a collapsed text tower here (all texts cos .999)
    model, proc = ClapModel.from_pretrained(mid).eval(), ClapProcessor.from_pretrained(mid)
    with torch.no_grad():
        avoid = a.avoid or ['a song with singing vocals', 'a drum kit beat', 'happy upbeat pop music']
        tin = proc(text=[a.target] + avoid, return_tensors='pt', padding=True)
        rows = []
        for w in a.wavs:
            x, sr = sf.read(w, always_2d=True)
            x = resample_poly(x.mean(1), 48000, sr) if sr != 48000 else x.mean(1)
            # score 10 s windows (CLAP's input length) and average, so a long cue is judged across its whole length
            win = [x[i:i + 480000] for i in range(0, max(1, len(x) - 240000), 480000)] or [x]
            ain = proc(audio=[w_.astype(np.float32) for w_ in win], sampling_rate=48000, return_tensors='pt')
            o = model(input_ids=tin['input_ids'], attention_mask=tin['attention_mask'], input_features=ain['input_features'],
                      **({'is_longer': ain['is_longer']} if 'is_longer' in ain else {}))
            # the model's own projected, normalised embeddings (get_*_features gave unprojected vectors: cos ~0)
            pr = o.logits_per_audio.softmax(-1).mean(0)  # target vs each unwanted text, averaged over 10 s windows
            rows.append(dict(wav=os.path.basename(w), score=round(float(pr[0]), 4),
                             avoid={t: round(float(v), 3) for t, v in zip(avoid, pr[1:])}))
    rows.sort(key=lambda r: -r['score'])
    for r in rows:
        print(json.dumps(r))


if __name__ == '__main__':
    main()
