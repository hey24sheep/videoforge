"""Directed, expressive dialogue takes with Qwen3-TTS (Apache-2.0, https://github.com/QwenLM/Qwen3-TTS).

Run with the voice venv:
  D:/claude_work/hero_film/tools/envs/voice/Scripts/python.exe tools/act_voice.py --jobs jobs.json

jobs.json = [{"text": "...", "character_voice": "<fixed voice description>", "direction": "<per-line acting>",
              "seed": 1, "out": "path.wav", "mode": "design"}, ...]

Modes:
  design (default) - Qwen3-TTS-12Hz-1.7B-VoiceDesign. instruct = character_voice + " " + direction, so the
                     performance follows the direction; the timbre comes from the same fixed description every line
                     (same person, but it can drift per seed: check with ECAPA, see tools/voice_qc.py).
  clone            - Qwen3-TTS-12Hz-1.7B-Base cloning a SYNTHETIC reference (job "ref_wav" + "ref_text", e.g. a
                     VoiceDesign take). Locks timbre, but Base takes no instruction: prosody follows the reference.
  clone_instruct   - EXPERIMENTAL: Base with only the ref's speaker embedding (x-vector) + the direction as instruct.
  design_xvec      - EXPERIMENTAL: VoiceDesign (character_voice + direction) plus the speaker embedding of ref_wav
                     (computed with Base's speaker encoder) to pin the timbre to one synthetic character ref.
Never use a recording of a real person as ref_wav.
"""
import argparse
import json
import os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import sys
import time

import numpy as np
import soundfile as sf
import torch

MODELS = _AI.replace(chr(92), '/') + '/models'
PATHS = {'design': f'{MODELS}/qwen3-tts-1.7b-voicedesign', 'clone': f'{MODELS}/qwen3-tts-1.7b-base',
         'clone_instruct': f'{MODELS}/qwen3-tts-1.7b-base', 'design_xvec': f'{MODELS}/qwen3-tts-1.7b-voicedesign'}


def load(mode):
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(PATHS[mode], device_map='cuda:0', dtype=torch.bfloat16,
                                         attn_implementation='sdpa')


def seed_all(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


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
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--temperature', type=float, default=None, help='override checkpoint default')
    ap.add_argument('--skip-existing', action='store_true')
    a = ap.parse_args()
    jobs = json.load(open(a.jobs, encoding='utf-8'))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(7.2 / 8.0)
    gen_kw = {} if a.temperature is None else {'temperature': a.temperature}
    models, prompts, log = {}, {}, []
    t_all = time.time()
    xrefs = sorted({jb['ref_wav'] for jb in jobs if jb.get('mode') == 'design_xvec'})
    xemb = {}
    if xrefs:   # speaker embeddings from the Base checkpoint's encoder (VoiceDesign ships none), then unload it
        base = load('clone')
        for r in xrefs:
            xemb[r] = base.create_voice_clone_prompt(ref_audio=r, x_vector_only_mode=True)[0].ref_spk_embedding.cpu()
        del base
        torch.cuda.empty_cache()
    for jb in jobs:
        out = jb['out']
        if a.skip_existing and os.path.exists(out):
            continue
        mode = jb.get('mode', 'design')
        if mode not in models:
            same = [m for m in models if PATHS[m] == PATHS[mode]]
            if same:
                models[mode] = models[same[0]]
        if mode not in models:
            for m in list(models):  # one model resident at a time
                del models[m]
            torch.cuda.empty_cache()
            t0 = time.time()
            models[mode] = load(mode)
            print(f'[act] loaded {mode} in {time.time() - t0:.0f}s', file=sys.stderr, flush=True)
        model = models[mode]
        seed_all(int(jb.get('seed', 0)))
        t0 = time.time()
        if mode in ('design', 'design_xvec'):
            instruct = (jb['character_voice'].strip() + ' ' + jb['direction'].strip()).strip()
            orig = model.model.generate
            if mode == 'design_xvec':
                vcp = dict(ref_code=[None], ref_spk_embedding=[xemb[jb['ref_wav']]], x_vector_only_mode=[True],
                           icl_mode=[False])
                model.model.generate = lambda **kw: orig(voice_clone_prompt=vcp, **kw)
            try:
                wavs, sr = model.generate_voice_design(text=jb['text'], language='English', instruct=instruct,
                                                       **gen_kw)
            finally:
                model.model.generate = orig
        else:
            xvec = mode == 'clone_instruct'
            key = (jb['ref_wav'], jb.get('ref_text', ''), xvec)
            if key not in prompts:
                prompts[key] = model.create_voice_clone_prompt(ref_audio=jb['ref_wav'], ref_text=jb.get('ref_text'),
                                                               x_vector_only_mode=xvec)
            instruct = None
            orig = model.model.generate
            if xvec:
                # EXPERIMENT: speaker embedding of the synthetic ref (timbre lock) + the acting instruction. The core
                # generate() accepts both; the Base checkpoint was not documented as instruction-trained.
                instruct = jb['direction'].strip()
                ids = model._tokenize_texts([model._build_instruct_text(instruct)])[0]
                model.model.generate = lambda **kw: orig(instruct_ids=[ids], **kw)
            try:
                wavs, sr = model.generate_voice_clone(text=jb['text'], language='English',
                                                      voice_clone_prompt=prompts[key], **gen_kw)
            finally:
                model.model.generate = orig
        torch.cuda.synchronize()
        dt = time.time() - t0
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        sf.write(out, wavs[0], sr)
        rec = dict(jb, instruct=instruct, sec=round(dt, 2), dur=round(len(wavs[0]) / sr, 2),
                   vram_peak_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2))
        log.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    meta = os.path.splitext(a.jobs)[0] + '.done.json'
    json.dump(log, open(meta, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[act] {len(log)} takes in {time.time() - t_all:.0f}s, peak reserved '
          f'{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB, peak RAM {peak_rss_gib()} GiB', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
