"""Word accuracy + same-voice check for voice takes (run with the tts venv, CPU so it never fights the GPU).

  D:/claude_work/hero_film/tools/envs/tts/Scripts/python.exe tools/voice_qc.py --jobs takes.json --out qc.json

takes.json = [{wav, text, who}, ...]. Adds:
  heard, wer          faster-whisper small.en (MIT), word error rate after normalising punctuation/numbers-as-words
  same_voice          SpeechBrain spkrec-ecapa-voxceleb (Apache-2.0) cosine to the mean embedding of the other
                      takes of the same `who` (leave-one-out), plus `anchor_sim` to job["anchor"] wav if given.
"""
import argparse
import json
import os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import re

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '        'seventeen eighteen nineteen'.split()
TENS = 'x x twenty thirty forty fifty sixty seventy eighty ninety'.split()
SAME = {'kohls': 'coles', 'kohl': 'cole', 'colts': 'coles'}   # homophones whisper picks for the name Cole


def n2w(m):
    n = int(m.group(0))
    if n < 20:
        return ' ' + ONES[n] + ' '
    if n < 100:
        return ' ' + TENS[n // 10] + ('' if n % 10 == 0 else ' ' + ONES[n % 10]) + ' '
    return ' ' + m.group(0) + ' '   # '1,205' / '1205' stays a mismatch: the pause was not heard


def norm(s):
    s = s.lower().replace('—', ' ').replace('-', ' ')
    s = re.sub(r'\d+(,\d+)*', lambda m: n2w(m) if ',' not in m.group(0) else ' ' + m.group(0) + ' ', s)
    s = re.sub(r"[^a-z0-9,' ]", ' ', s).replace(',', '')
    s = s.replace("'s", 's').replace("'", '')
    return [SAME.get(w, w) for w in s.split()]


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return round(float(d[len(h)]) / max(1, len(r)), 3)


def load16(p):
    x, sr = sf.read(p, always_2d=True)
    return resample_poly(x.mean(1), 16000, sr).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    takes = json.load(open(a.jobs, encoding='utf-8'))
    from faster_whisper import WhisperModel
    asr = WhisperModel('small.en', device='cpu', compute_type='int8')
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    # speechbrain 1.0.3 + new huggingface_hub: load from the local copy vf/gen/delivery.py prepared
    src = os.path.join(os.environ['HF_HOME'], 'speechbrain', 'ecapa_src')
    ecapa = EncoderClassifier.from_hparams(
        source=src, overrides={'pretrained_path': src.replace(os.sep, '/')},
        savedir=os.path.join(os.environ['HF_HOME'], 'speechbrain', 'spkrec-ecapa-voxceleb'),
        run_opts={'device': 'cpu'}, local_strategy=LocalStrategy.COPY)

    def emb(p):
        e = ecapa.encode_batch(torch.from_numpy(load16(p))[None]).squeeze().numpy()
        return e / np.linalg.norm(e)

    for t in takes:
        segs, _ = asr.transcribe(load16(t['wav']), language='en', beam_size=5,
                                 initial_prompt=None, condition_on_previous_text=False)
        t['heard'] = ' '.join(s.text.strip() for s in segs).strip()
        t['wer'] = wer(t['text'], t['heard']) if t.get('text') else None
        t['_e'] = emb(t['wav'])
        if t.get('anchor'):
            t['anchor_sim'] = round(float(t['_e'] @ emb(t['anchor'])), 3)
    for t in takes:
        others = [o['_e'] for o in takes if o is not t and o.get('who') == t.get('who')]
        if others:
            m = np.mean(others, 0)
            t['same_voice'] = round(float(t['_e'] @ (m / np.linalg.norm(m))), 3)
    for t in takes:
        t.pop('_e')
    json.dump(takes, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    for t in takes:
        print(f"{os.path.basename(t['wav']):40s} wer {t['wer']!s:<5} same {t.get('same_voice')}  "
              f"anchor {t.get('anchor_sim')}  heard: {t['heard']}")


if __name__ == '__main__':
    main()
