"""Download (once) everything the Wan2.2-I2V-A14B GGUF path needs into the HF cache. Run in the gen venv.

    python -m vf.gen.fetch_models
"""
import os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))
from huggingface_hub import hf_hub_download, snapshot_download

BASE = 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
GGUF = 'QuantStack/Wan2.2-I2V-A14B-GGUF'
GGUF_FILES = {'high': 'HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf',
              'low': 'LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf'}
LORA = 'lightx2v/Wan2.2-Lightning'
LORA_FILES = {'high': 'Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors',
              'low': 'Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors'}
BASE_PATTERNS = ['model_index.json', 'scheduler/*', 'tokenizer/*', 'text_encoder/*', 'vae/*',
                 'transformer/config.json', 'transformer_2/config.json']


def _local_first(f, *a, **k):
    try:  # no network round-trip once everything is cached
        return f(*a, local_files_only=True, **k)
    except Exception:
        return f(*a, **k)


def base_dir():
    return _local_first(snapshot_download, BASE, allow_patterns=BASE_PATTERNS)


def gguf_path(which):
    return _local_first(hf_hub_download, GGUF, GGUF_FILES[which])


def lora_path(which):
    return _local_first(hf_hub_download, LORA, LORA_FILES[which])


FAST = 'FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers'  # DMD 3-step distill of Wan2.2-TI2V-5B (24 fps, 4x16x16 VAE)
FAST_PATTERNS = ['model_index.json', 'scheduler/*', 'transformer/*', 'vae/*']  # UMT5 shared with BASE


def fast_dir():
    return _local_first(snapshot_download, FAST, allow_patterns=FAST_PATTERNS)


TURBO = 'yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers'  # quanhaol's Self-Forcing/DMD 4-step I2V distill of TI2V-5B


def turbo_dir():  # transformer + scheduler only; VAE comes from FAST (same Wan2.2 VAE), UMT5 from BASE
    return _local_first(snapshot_download, TURBO, allow_patterns=['model_index.json', 'scheduler/*', 'transformer/*'])


def tae_path(name='taew2_2'):
    return _local_first(hf_hub_download, 'lightx2v/Autoencoders', f'{name}.safetensors')


if __name__ == '__main__':
    import sys
    what = sys.argv[1:] or ['base', 'fast', 'lora', 'gguf']
    if 'fast' in what:
        print('fast', fast_dir(), turbo_dir(), tae_path(), flush=True)
    if 'base' in what:
        print('base', base_dir(), flush=True)
    if 'lora' in what:
        for w in LORA_FILES:
            print('lora', w, lora_path(w), flush=True)
    if 'gguf' in what:
        for w in GGUF_FILES:
            print('gguf', w, gguf_path(w), flush=True)
