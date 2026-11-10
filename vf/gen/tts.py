"""Voice stage: text -> spoken line. Runs ONLY inside the tts venv (Chatterbox + faster-whisper); from the main
interpreter use vf.gen.bridge.run_gen('tts', jobs).

    say(text, voice='default', exaggeration=0.5, cfg_weight=0.5, temperature=0.8, seed=0, out=None) -> wav path
    run_jobs(jobs, emit=print)  /  python -m vf.gen.tts jobs.json

voice: 'default' (Chatterbox's built-in speaker) or a path to a reference wav (5-15 s of clean speech). Use only
synthetic or self-recorded references, never a real person's voice without consent.
exaggeration: emotion intensity (0.3 flat .. 0.9 intense). cfg_weight: lower = slower, more deliberate delivery.

Auto-check (the user's "auto-check, regenerate" rule): every take is transcribed by faster-whisper; a take that
drops, adds or garbles words (word error > 15 %), or runs too fast or slow, is regenerated with the next seed
(up to TAKES). The best take is kept and its word timings are written to `<wav>.words.json` (captions use them).
Silence and Chatterbox's trailing breath/noise are trimmed so `at:` times land on the first word.
"""
import json, os, re, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
from difflib import SequenceMatcher

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
import numpy as np

from ..memo import memo_file, key_for, tier_dir

MODEL_ID = 'chatterbox-0.1.7 + faster-whisper small.en int8 check'
TAKES = 4
MAX_WER = 0.15
_M = {}


def _torch():
    import torch
    return torch


def _tts():
    if 'tts' not in _M:
        from chatterbox.tts import ChatterboxTTS
        _M['tts'] = ChatterboxTTS.from_pretrained(device='cuda' if _torch().cuda.is_available() else 'cpu')
    return _M['tts']


def _asr():
    if 'asr' not in _M:
        from faster_whisper import WhisperModel
        _M['asr'] = WhisperModel('small.en', device='cpu', compute_type='int8')
    return _M['asr']


# Numbers are spoken as words but transcribed as digits ("three seventeen" -> "3.17"): both sides drop them, so the
# check is about the other words (a garbled number still shows up as a timing/rate problem).
_NUMWORDS = set('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
                'sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred '
                'thousand oh'.split())


def norm_words(s):
    s = s.lower().replace("'", '').replace('’', '')
    return [w for w in re.findall(r'[a-z]+', re.sub(r'\d+', ' ', s)) if w not in _NUMWORDS]


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    if not r:
        return 0.0
    m = SequenceMatcher(None, r, h, autojunk=False)
    same = sum(b.size for b in m.get_matching_blocks())
    return (len(r) - same + max(0, len(h) - same)) / len(r)


def _cut_after_script(ref, words):
    """Chatterbox sometimes repeats the end of a short line ("Come alone. Come alone.") or babbles after it.
    If every scripted word was heard before the extra ones, return the time just after the last scripted word."""
    r = norm_words(ref)
    toks = [(t, i) for i, w in enumerate(words) for t in norm_words(w['w'])]
    h = [t for t, _ in toks]
    if not r or len(h) <= len(r):
        return None
    m = SequenceMatcher(None, r, h, autojunk=False)
    blocks = [b for b in m.get_matching_blocks() if b.size]
    if sum(b.size for b in blocks) < len(r) or blocks[-1].a + blocks[-1].size != len(r):
        return None
    last = blocks[-1].b + blocks[-1].size - 1  # token index of the last scripted word
    if last >= len(h) - 1:
        return None
    wi = toks[last][1]
    nxt = words[wi + 1]['start'] if wi + 1 < len(words) else words[wi]['end'] + 0.3
    return min(words[wi]['end'] + 0.22, (words[wi]['end'] + nxt) / 2)


def _trim(x, sr, words):
    """Cut leading silence and anything after the last spoken word (breaths, clicks, babble)."""
    a = np.abs(x)
    hop = int(0.01 * sr)
    env = np.array([a[i:i + hop].max() for i in range(0, len(a), hop)]) if len(a) else np.zeros(1)
    thr = max(1e-3, env.max() * 0.03)
    on = np.nonzero(env > thr)[0]
    if not len(on):
        return x, 0.0
    i0 = max(0, on[0] * hop - int(0.06 * sr))
    i1 = (on[-1] + 1) * hop + int(0.12 * sr)
    if words:  # never keep more than 0.45 s past the last recognised word
        i1 = min(i1, int((words[-1]['end'] + 0.45) * sr))
    y = x[i0:max(i0 + hop, i1)].copy()
    f = min(len(y), int(0.03 * sr))
    y[-f:] *= np.linspace(1, 0, f)
    return y, i0 / sr


def _take(text, voice, exaggeration, cfg_weight, temperature, seed):
    torch = _torch()
    m = _tts()
    torch.manual_seed(int(seed))
    kw = {} if voice in (None, '', 'default') else dict(audio_prompt_path=voice)
    w = m.generate(text, exaggeration=float(exaggeration), cfg_weight=float(cfg_weight),
                   temperature=float(temperature), **kw)
    return w.squeeze(0).float().cpu().numpy(), m.sr


def _transcribe(x, sr):
    import librosa
    y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
    segs, _ = _asr().transcribe(y.astype(np.float32), language='en', word_timestamps=True, beam_size=5,
                                vad_filter=False, condition_on_previous_text=False)
    words, text = [], []
    for s in segs:
        text.append(s.text)
        for w in s.words or []:
            words.append(dict(w=w.word.strip(), start=round(float(w.start), 3), end=round(float(w.end), 3)))
    return ' '.join(text).strip(), words


GEN_VERSION = 'say-1'  # bump when takes/trim/check logic changes what is written


@memo_file('global', '.wav', version=lambda: f'{GEN_VERSION}|{MODEL_ID}|{TAKES}|{MAX_WER}')
def say(text, voice='default', exaggeration=0.5, cfg_weight=0.5, temperature=0.8, seed=0, out=None):
    """Best of up to TAKES checked takes; writes <out>.words.json (word timings relative to the trimmed wav)."""
    import soundfile as sf
    assert MODEL_ID and TAKES and MAX_WER
    n_words = max(1, len(norm_words(text)))
    best = None
    for k in range(TAKES):
        x, sr = _take(text, voice, exaggeration, cfg_weight, temperature, int(seed) * 1000 + k)
        heard, words = _transcribe(x, sr)
        end = _cut_after_script(text, words)
        if end is not None:  # repair: drop the repeated/babbled tail, then re-check what is left
            x = x[:int(end * sr)]
            heard, words = _transcribe(x, sr)
        y, cut = _trim(x, sr, words)
        e = wer(text, heard)
        dur = len(y) / sr
        rate = n_words / max(0.1, dur)                    # words per second
        bad_rate = not (0.9 <= rate <= 4.2) if n_words >= 4 else dur > 0.9 * n_words + 1.5
        score = e + (0.5 if bad_rate else 0.0)
        cand = dict(y=y, sr=sr, words=[dict(w, start=round(w['start'] - cut, 3), end=round(w['end'] - cut, 3))
                                       for w in words], wer=round(e, 3), heard=heard, take=k, rate=round(rate, 2),
                    score=score)
        print(f'[tts] take {k}: wer {e:.2f} rate {rate:.2f} w/s  "{heard}"', file=sys.stderr, flush=True)
        if best is None or score < best['score']:
            best = cand
        if score <= MAX_WER:
            break
    sf.write(out, best['y'], best['sr'], subtype='PCM_16', format='WAV')
    with open(out + '.words.json', 'w', encoding='utf-8') as f:
        json.dump(dict(text=text, heard=best['heard'], wer=best['wer'], take=best['take'], rate=best['rate'],
                       ok=best['score'] <= MAX_WER, words=best['words']), f, indent=1)


def cached_path(job):
    fn = say.__wrapped__
    return os.path.join(tier_dir('global'), fn.__qualname__, key_for(fn, (), job) + '.wav')


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    out = []
    for i, j in enumerate(jobs):
        t0 = time.time()
        hit = os.path.exists(cached_path(j))
        try:
            p = say(**j)
            meta = json.load(open(p + '.words.json', encoding='utf-8'))
            r = dict(i=i, ok=True, out=p, cached=hit, seconds=round(time.time() - t0, 2), wer=meta['wer'],
                     heard=meta['heard'], checked=meta['ok'])
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = dict(i=i, ok=False, error=f'{type(e).__name__}: {e}', seconds=round(time.time() - t0, 2))
        out.append(r)
        emit(r)
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.tts jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
