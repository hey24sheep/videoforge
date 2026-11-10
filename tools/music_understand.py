"""Read key / bpm / caption of audio files with ACE-Step 1.5's own 5Hz LM (open, MIT): the model-based check for
"does this ending stay in the intro's key". Music venv:
D:/claude_work/hero_film/tools/envs/music/Scripts/python.exe tools/music_understand.py a.wav b.wav ...
Prints one JSON line per file: {"wav", "keyscale", "bpm", "timesig", "caption"}. Runs each file `--votes` times
(sampling) and reports every answer so a shaky read is visible."""
import argparse, os, sys, json
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
REPO = os.path.join(_AI, 'tools', 'ACE-Step-1.5')
CKPT = os.path.join(_AI, 'models', 'hf', 'ace-step-1.5', 'checkpoints')
os.environ["HF_HOME"] = os.path.join(_AI, 'models', 'hf')
os.environ["ACESTEP_CHECKPOINTS_DIR"] = CKPT
os.environ["ACESTEP_PROJECT_ROOT"] = REPO
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--votes", type=int, default=3)
    a = ap.parse_args()
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import understand_music
    dit = AceStepHandler()
    msg, ok = dit.initialize_service(project_root=REPO, config_path="acestep-v15-turbo", device="cuda",
                                     use_flash_attention=False, compile_model=False,
                                     offload_to_cpu=True, offload_dit_to_cpu=True, quantization=None)
    if not ok:
        sys.exit("dit init failed: " + msg)
    lm = LLMHandler()
    msg, ok = lm.initialize(checkpoint_dir=CKPT, lm_model_path="acestep-5Hz-lm-1.7B", backend="pt",
                            device="cuda", offload_to_cpu=True)
    if not ok:
        sys.exit("lm init failed: " + str(msg))
    for w in a.wavs:
        codes = dit.convert_src_audio_to_codes(w)
        if isinstance(codes, tuple):
            codes = codes[0]
        reads = []
        for v in range(a.votes):
            r = understand_music(lm, audio_codes=codes, temperature=0.3 if v == 0 else 0.7)
            reads.append(dict(keyscale=r.keyscale, bpm=r.bpm, timesig=r.timesignature,
                              caption=(r.caption or "")[:160]) if r.success else dict(error=r.error))
        print(json.dumps(dict(wav=os.path.basename(w), reads=reads)), flush=True)


if __name__ == "__main__":
    main()
