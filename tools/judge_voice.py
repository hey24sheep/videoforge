"""Listen to voice takes with an open audio-language model and judge the acting.

Model: stepfun-ai/Step-Audio-2-mini (Apache-2.0, https://huggingface.co/stepfun-ai/Step-Audio-2-mini),
loaded 4-bit NF4 (bitsandbytes) so it fits the 8 GB card. Run with the judge venv:

  D:/claude_work/hero_film/tools/envs/judge/Scripts/python.exe tools/judge_voice.py \
      --wav take.wav --text "Forty-four... seventy." --direction "cold, quiet menace" [--wav ...] [--out res.json]

Several --wav share one --text/--direction, or pass --jobs file.json = [{wav, text, direction}, ...].

Protocol (fixed, non-leading):
  pass 1 (blind: no script, no direction): the model describes what it hears (emotion, intensity, pitch movement,
          pace/pauses, voice quality, naturalness), then rates intensity / pitch movement / naturalness.
  pass 2 (blind forced choice): "which description best matches how this line is delivered?" - the intended
          direction among generic distractors (neutral / cheerful / shouting / grieving / frightened / menacing,
          minus any that share words with the direction), both option orders; matches_direction = 1 + 9*P(intended).
          (Telling the model the direction and asking "how well does it match" was sycophantic: a flat Kokoro read
          got 8.7/10 for "blurted, frightened". That number is kept as match_told for reference only.)
  pass 3 (told the script + direction): one-sentence critique.
Each 1-10 score is the probability-weighted mean over the model's next-token distribution on the digits 1-9
(rescaled to 1-10), not just the argmax digit, so close takes still separate.
"""
import argparse
import json
import os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import re
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get('VF_JUDGE_MODEL', _AI.replace(chr(92), '/') + '/models/step-audio-2-mini')
SA2_SRC = _AI.replace(chr(92), '/') + '/tools/src/Step-Audio2'
sys.path.insert(0, SA2_SRC)
from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels  # noqa: E402

SYS = ("You are an expert listener for voice acting. Listen carefully to the speaker's delivery and describe it "
       "objectively.")
P1 = ("Describe only what you hear in this voice recording, in 3-4 sentences: the emotion conveyed, how intense or "
      "restrained it is, how much the pitch rises and falls, the pace and any pauses, the voice quality (age, "
      "rasp, breath), and whether it sounds like a natural human performance or synthetic/read. "
      "End with one word naming the main emotion, on its own line as: Emotion: <word>")
SCALES = {
    'intensity': "Emotional intensity of the delivery, from 1 (flat, no emotion) to 9 (extremely intense):",
    'pitch_movement': "Pitch movement, from 1 (monotone, flat pitch) to 9 (very wide, dynamic pitch changes):",
    'naturalness': "Naturalness, from 1 (clearly robotic or synthetic) to 9 (indistinguishable from a real actor):",
}
P2 = ("The script of this recording is: \"{text}\"\nThe actor was directed to deliver it like this: \"{direction}\"\n"
      "Listen again.")
MATCH = "How well the delivery you heard matches that direction, from 1 (not at all) to 9 (exactly):"
CRIT = "In one sentence, the most important thing that is wrong or right about this take versus the direction:"
# Forced choice (blind to which option is intended): P(intended direction) among generic distractors.
DISTRACTORS = ["calm, neutral, matter-of-fact reading with even pitch",
               "warm, friendly and cheerful",
               "loud, furious shouting",
               "sad, tearful, grieving",
               "frightened, panicked, voice shaking and cracking",
               "cold, quiet, controlled menace"]
MC = "Which description best matches how this line is delivered? Answer with one letter only.\n"
STOP = set("a an the and or of to in on with his he as it its is be no not like than before after very too".split())


def _words(t):
    return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2}


class Judge:
    """Step-Audio-2-mini with a hand-written KV-cached greedy loop: the repo's forward() returns no cache, so HF
    generate() would re-run the whole prompt + audio encoder every token. Token embeddings (1.1 GB) stay on the CPU."""

    def __init__(self, path=MODEL, fourbit=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(7.2 / 8.0)
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side='right')
        kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={'': 0})
        if fourbit:
            kw['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, llm_int8_skip_modules=['encoder', 'adapter', 'lm_head'])
        self.llm = AutoModelForCausalLM.from_pretrained(path, **kw)
        self.llm.eval()
        self.emb = self.llm.model.embed_tokens.to('cpu')   # 1.1 GB off the card; lookups are cheap on the CPU
        torch.cuda.empty_cache()
        self.eot = self.tok.convert_tokens_to_ids('<|EOT|>')
        self.digit_ids = [self.tok.convert_tokens_to_ids(str(d)) for d in range(1, 10)]

    # ---- prompt building (same chat format as Step-Audio2/stepaudio2.py) ----
    def _build(self, wav, user_parts, assistant_prefix):
        audio = load_audio(wav)
        mels, pieces = [], [f"<|BOT|>system\n{SYS}<|EOT|>", "<|BOT|>human\n"]
        for part in user_parts:
            if part == '<AUDIO>':
                for i in range(0, audio.shape[0], 16000 * 25):
                    mel = log_mel_spectrogram(audio[i:i + 16000 * 25], n_mels=128, padding=479)
                    mels.append(mel)
                    pieces.append("<audio_start>" + "<audio_patch>" * compute_token_num(mel.shape[1]) + "<audio_end>")
            else:
                pieces.append(part)
        pieces.append("<|EOT|>")
        pieces.append("<|BOT|>assistant\n" + assistant_prefix)
        ids = torch.cat([self.tok(text=p, return_tensors='pt')['input_ids'] for p in pieces], dim=-1)
        m, ml = padding_mels(mels)
        return ids, m.cuda(), ml.cuda()

    def _embed(self, ids, mels=None, mel_lens=None):
        h = self.emb(ids.cpu()).cuda().to(torch.bfloat16)
        if mels is not None:   # same splice as StepAudio2ForCausalLM.forward
            out, fl = self.llm.encoder(mels.bfloat16(), mel_lens)
            out = self.llm.adapter(out)
            fl = (fl - 1) // 2 + 1
            loc = torch.nonzero(ids == 151688)
            loc[:, 1] += 1
            for k in range(len(loc)):
                i, s0 = loc[k]
                h[i][s0:s0 + fl[k]] = out[k][:fl[k]]
        return h

    @torch.inference_mode()
    def _logits(self, ids, mels, ml, past=None):
        o = self.llm.model(inputs_embeds=self._embed(ids, mels, ml), past_key_values=past, use_cache=True)
        return self.llm.lm_head(o.last_hidden_state[:, -1]).float(), o.past_key_values

    @torch.inference_mode()
    def generate(self, wav, user_parts, prefix='', max_new=200):
        ids, m, ml = self._build(wav, user_parts, prefix)
        logits, past = self._logits(ids, m, ml)
        out, seen = [], []
        for _ in range(max_new):
            lg = logits[0]
            for t in set(seen):   # repetition_penalty 1.05
                lg[t] = lg[t] / 1.05 if lg[t] > 0 else lg[t] * 1.05
            keep = lg[self.eot].clone()
            lg[151688:] = -1e9   # text tokens only (ids >= 151688 are audio/special tokens)
            lg[self.eot] = keep
            if len(out) >= 3:   # no repeated 4-grams: greedy decoding looped a sentence until max_new
                tail = tuple(out[-3:])
                for i in range(len(out) - 3):
                    if tuple(out[i:i + 3]) == tail:
                        lg[out[i + 3]] = -1e9
            nxt = int(lg.argmax())
            if nxt == self.eot or nxt == self.tok.eos_token_id:
                break
            out.append(nxt)
            seen.append(nxt)
            logits, past = self._logits(torch.tensor([[nxt]]), None, None, past)
        return self.tok.decode(out, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def score(self, wav, user_parts, prefix):
        ids, m, ml = self._build(wav, user_parts, prefix)
        logits, _ = self._logits(ids, m, ml)
        p = torch.softmax(logits[0, self.digit_ids], -1)
        ev = float((p * torch.arange(1, 10, device=p.device, dtype=p.dtype)).sum())
        return round(1 + (ev - 1) * 9 / 8, 2), round(float(p.max()), 2)

    @torch.inference_mode()
    def choice(self, wav, direction, distractors=None):
        """P(direction) in a lettered forced choice, averaged over forward and reversed option order."""
        opts = [d for d in (distractors or DISTRACTORS) if not (_words(d) & _words(direction))][:5]
        ps = []
        for order in (opts + [direction], [direction] + opts[::-1]):
            letters = [chr(65 + i) for i in range(len(order))]
            q = MC + "\n".join(f"{l}) {o}" for l, o in zip(letters, order))
            ids, m, ml = self._build(wav, ['<AUDIO>', q], "Answer: ")
            logits, _ = self._logits(ids, m, ml)
            lid = [self.tok.convert_tokens_to_ids(l) for l in letters]
            p = torch.softmax(logits[0, lid], -1)
            ps.append(float(p[order.index(direction)]))
        return sum(ps) / 2, len(opts) + 1

    def judge(self, wav, text, direction):
        t0 = time.time()
        desc = self.generate(wav, ['<AUDIO>', P1])
        m = re.search(r'[Ee]motion(?: conveyed)?(?: is)?:?\s*<?([A-Za-z-]+)>?\.?\s*$', desc)
        word = m.group(1).lower() if m else ''
        heard = re.sub(r'\s*(The|the|Main|main)?\s*$', '', desc[:m.start()]).strip() if m else desc
        base = desc + "\n\n"
        res = {'wav': wav, 'text': text, 'direction': direction, 'emotion_heard': heard, 'emotion_word': word}
        for k, q in SCALES.items():
            res[k], res[k + '_conf'] = self.score(wav, ['<AUDIO>', P1], base + q + " ")
        p2 = ['<AUDIO>', P2.format(text=text, direction=direction)]
        res['match_told'], _ = self.score(wav, p2, MATCH + " ")   # told the direction: sycophantic, info only
        pc, n = self.choice(wav, direction)
        res['p_direction'], res['n_options'] = round(pc, 3), n
        res['matches_direction'] = round(1 + 9 * pc, 2)               # blind forced choice -> 1-10
        res['critique'] = self.generate(wav, p2 + [" " + CRIT], max_new=80).split('\n')[0]
        res['judge_score'] = round(res['matches_direction'] + res['intensity'] + res['naturalness'], 2)
        res['sec'] = round(time.time() - t0, 1)
        return res


def peak_rss_gib():
    """Peak working set of this process (Windows), GiB."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] +                    [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage',
                    'PeakPagefileUsage')]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return round(c.PeakWorkingSetSize / 2**30, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', action='append', default=[])
    ap.add_argument('--text', default='')
    ap.add_argument('--direction', default='')
    ap.add_argument('--jobs', help='json list of {wav, text, direction}')
    ap.add_argument('--out', help='write results json here')
    a = ap.parse_args()
    jobs = [dict(wav=w, text=a.text, direction=a.direction) for w in a.wav]
    if a.jobs:
        jobs += json.load(open(a.jobs, encoding='utf-8'))
    if not jobs:
        ap.error('no takes')
    t0 = time.time()
    j = Judge()
    print(f'[judge] loaded in {time.time() - t0:.0f}s, vram {torch.cuda.memory_allocated() / 2**30:.2f} GiB',
          file=sys.stderr, flush=True)
    out = []
    for jb in jobs:
        r = j.judge(jb['wav'], jb.get('text', ''), jb.get('direction', ''))
        for k in jb:
            r.setdefault(k, jb[k])
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        if a.out:
            json.dump(out, open(a.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[judge] peak vram reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB, peak RAM {peak_rss_gib()} GiB, '
          f'{(time.time() - t0) / len(jobs):.1f} s/take incl load', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
