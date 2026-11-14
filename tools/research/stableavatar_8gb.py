"""Research record (2026-11-14, REJECTED, see docs/model_research.md).
StableAvatar 1.3B (MIT) on one talk job, sized for an 8 GB card: the 11 GB T5 encoder runs once on the CPU and is
dropped; the transformer sits on the GPU only while sampling; CLIP visits the GPU per call; the 3 guidance passes run
one at a time; VRAM capped at 7.2 GB. Prints [sa] lines with timings. Measured ~100-140 s/step at 480x832x81.

    <venv>/python tools/research/stableavatar_8gb.py --image KF.png --audio LINE.wav --prompt "..." --out DIR
           [--steps 25] [--seed 42] [--size 480x832]
"""
import argparse, os, sys, time

AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')
SA = os.path.join(AI, 'tools', 'StableAvatar')            # git clone https://github.com/Francis-Rings/StableAvatar
MODELS = os.path.join(AI, 'models', 'stableavatar')       # Hugging Face FrancisRing/StableAvatar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--audio', required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--size', default='480x832')
    ap.add_argument('--text-scale', type=float, default=3.0)
    ap.add_argument('--audio-scale', type=float, default=5.0)
    a = ap.parse_args()
    w, h = map(int, a.size.split('x'))
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, SA)
    os.chdir(SA)
    import torch, librosa
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
    from diffusers import FlowMatchEulerDiscreteScheduler
    from wan.models.cache_utils import get_teacache_coefficients  # noqa: F401 (import check)
    from wan.models.wan_fantasy_transformer3d_1B import WanTransformer3DFantasyModel
    from wan.models.wan_image_encoder import CLIPModel
    from wan.models.wan_text_encoder import WanT5EncoderModel
    from wan.models.wan_vae import AutoencoderKLWan
    from wan.pipeline.wan_inference_long_pipeline import WanI2VTalkingInferenceLongPipeline
    from wan.utils.utils import get_image_to_video_latent, save_videos_grid, filter_kwargs

    assert torch.__version__ >= '2.6', 'torch.load must default to weights_only=True (no pickle code on load)'
    torch.cuda.set_per_process_memory_fraction(7.2 / (torch.cuda.get_device_properties(0).total_memory / 2**30), 0)
    t0 = time.time()
    base = os.path.join(MODELS, 'Wan2.1-Fun-V1.1-1.3B-InP')
    cfg = OmegaConf.load(os.path.join(SA, 'deepspeed_config', 'wan2.1', 'wan_civitai.yaml'))
    dt = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(os.path.join(base, cfg['text_encoder_kwargs'].get('tokenizer_subpath')))
    te = WanT5EncoderModel.from_pretrained(os.path.join(base, cfg['text_encoder_kwargs'].get('text_encoder_subpath')),
                                           additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
                                           low_cpu_mem_usage=True, torch_dtype=dt).eval()
    vae = AutoencoderKLWan.from_pretrained(os.path.join(base, cfg['vae_kwargs'].get('vae_subpath')),
                                           additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs'])).to(dt)
    w2v_dir = os.path.join(MODELS, 'wav2vec2-base-960h')
    w2v_p, w2v = Wav2Vec2Processor.from_pretrained(w2v_dir), Wav2Vec2Model.from_pretrained(w2v_dir).to('cpu')
    clip = CLIPModel.from_pretrained(os.path.join(base, cfg['image_encoder_kwargs'].get('image_encoder_subpath'))).eval().to(dt)
    tr = WanTransformer3DFantasyModel.from_pretrained(
        os.path.join(base, cfg['transformer_additional_kwargs'].get('transformer_subpath')),
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        low_cpu_mem_usage=False, torch_dtype=dt)
    sd = torch.load(os.path.join(MODELS, 'StableAvatar-1.3B', 'transformer3d-square.pt'), map_location='cpu',
                    weights_only=True)
    sd = sd.get('state_dict', sd)
    m, u = tr.load_state_dict(sd, strict=False)
    print(f'[sa] transformer: missing {len(m)} unexpected {len(u)}', flush=True)
    sch = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler,
                                                          OmegaConf.to_container(cfg['scheduler_kwargs'])))
    pipe = WanI2VTalkingInferenceLongPipeline(tokenizer=tok, text_encoder=te, vae=vae, transformer=tr,
                                              clip_image_encoder=clip, scheduler=sch, wav2vec_processor=w2v_p,
                                              wav2vec=w2v)
    neg = ('色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，'
           '多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走')
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, neg, do_classifier_free_guidance=True, device='cpu', dtype=dt)
    print(f'[sa] prompt encoded on CPU, load+encode {time.time() - t0:.0f}s', flush=True)

    class Stub(torch.nn.Module):   # the pipeline reads text_encoder.dtype; the 11 GB encoder itself is gone
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1, dtype=dt), requires_grad=False)

        @property
        def dtype(self):
            return self.p.dtype

        @property
        def device(self):
            return self.p.device
    del te
    pipe.register_modules(text_encoder=Stub())
    import gc
    gc.collect()
    # accelerate's model offload only evicts the PREVIOUS model in its chain, so the 2.4 GB CLIP stayed on the card while
    # the VAE ran (OOM). Manual plan: transformer + VAE live on the GPU (about 3.5 GB); CLIP visits the GPU per call.
    vae.to('cuda'); w2v.to('cuda')   # wav2vec is small (0.4 GB); the transformer (2.8 GB) joins at its first step and leaves before the final decode
    tr_fwd, dec = tr.forward, vae.decode

    def tr_visit(*x, **k):
        if next(tr.parameters()).device.type != 'cuda':
            tr.to('cuda')
        b = k['x'].shape[0] if torch.is_tensor(k.get('x')) else len(k.get('x', []))
        if b <= 1 or os.environ.get('SA_SPLIT_CFG', '1') == '0':
            return tr_fwd(*x, **k)
        # the 3 guidance passes run one after another instead of as one batch of 3 (8 GB card): same maths, a third
        # of the activation memory
        def part(v, i):
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == b:
                return v[i:i + 1]
            if isinstance(v, list) and len(v) == b:
                return v[i:i + 1]
            return v
        outs = [tr_fwd(*x, **{n: part(v, i) for n, v in k.items()}) for i in range(b)]
        return torch.cat(outs, 0)

    def dec_alone(*x, **k):
        if next(tr.parameters()).device.type == 'cuda':
            tr.to('cpu'); torch.cuda.empty_cache()
        return dec(*x, **k)
    tr.forward, vae.decode = tr_visit, dec_alone
    clip_fwd = clip.forward

    def clip_visit(*x, **k):
        clip.to('cuda')
        try:
            return clip_fwd(*x, **k)
        finally:
            clip.to('cpu'); torch.cuda.empty_cache()
    clip.forward = clip_visit
    pipe._execution_device_override = 'cuda'
    type(pipe)._execution_device = property(lambda self: torch.device('cuda'))
    pipe.check_inputs = lambda *x, **k: None   # we pass the prompt (for batch size) AND its precomputed embeddings
    g = torch.Generator(device='cuda').manual_seed(a.seed)
    n = 81
    vid, mask, clip_img = get_image_to_video_latent(a.image, None, video_length=n, sample_size=[h, w])
    voc, sr = librosa.load(a.audio, sr=16000)
    print(f'[sa] VRAM before sampling {torch.cuda.memory_allocated() / 2**30:.2f} GB', flush=True)
    t1 = time.time()
    with torch.no_grad():
        out = pipe(a.prompt, num_frames=n, negative_prompt=None, prompt_embeds=[x.to('cuda') for x in pe],
                   negative_prompt_embeds=[x.to('cuda') for x in ne], height=h, width=w, guidance_scale=6.0, generator=g,
                   num_inference_steps=a.steps, video=vid, mask_video=mask, clip_image=clip_img,
                   text_guide_scale=a.text_scale, audio_guide_scale=a.audio_scale, vocal_input_values=voc,
                   motion_frame=25, fps=25, sr=16000, cond_file_path=a.image, seed=a.seed, overlap_window_length=5,
                   overlapping_weight_scheme='uniform', clip_length=n).videos
    gen = time.time() - t1
    path = os.path.join(a.out, 'video_without_audio.mp4')
    save_videos_grid(out, path, fps=25)
    print(f'[sa] RESULT gen {gen:.0f}s, {gen / a.steps:.1f} s/step, peak VRAM reserved '
          f'{torch.cuda.max_memory_reserved() / 2**30:.2f} GB, total {time.time() - t0:.0f}s -> {path}', flush=True)


if __name__ == '__main__':
    main()
