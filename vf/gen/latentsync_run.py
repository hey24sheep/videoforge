"""Lip-sync with LatentSync 1.6 (ByteDance; code Apache-2.0, weights OpenRAIL++), 512 px diffusion inpainting of the
mouth, SyncNet-supervised. InsightFace (non-commercial weights) is replaced by OpenCV YuNet: LatentSync's alignment
only needs 3 points (eyebrow centres + nose), estimated from YuNet's 5 landmarks.
Usage (gen venv): python -m vf.gen.latentsync_run clip.mp4 voice.wav offset_s out.mp4 [steps] [guidance]
Output is 25 fps (LatentSync's rate) with the voice track muxed in."""
import os, subprocess, sys, tempfile, types
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import cv2, numpy as np, soundfile as sf

LS = os.path.join(_AI, 'tools', 'latentsync')
VAE = os.path.join(_AI, 'tools', 'musetalk', 'models', 'sd-vae')   # same weights as stabilityai/sd-vae-ft-mse
YUNET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'cache',
                     'models', 'yunet', 'face_detection_yunet_2023mar.onnx')
from .k6_batch import FFMPEG


class YuNetDetector:
    """Drop-in for latentsync.utils.face_detector.FaceDetector: returns (bbox, fake 106-point array) where the
    indices LatentSync reads hold YuNet-derived eyebrow centres and nose centre."""
    def __init__(self, device='cuda'):
        self.det, self.last = None, (None, None)

    def __call__(self, frame, threshold=0.5):
        r = self._detect(frame)
        if r[0] is not None:
            self.last = r
        return self.last  # dark frames: hold the last good face

    def _detect(self, frame):
        f = np.asarray(frame)
        if f.shape[0] == 3:
            f = f.transpose(1, 2, 0)
        h, w = f.shape[:2]
        if self.det is None or self.size != (w, h):
            self.det, self.size = cv2.FaceDetectorYN.create(YUNET, '', (w, h), 0.3), (w, h)
        _, faces = self.det.detect(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR))
        if faces is None or not len(faces):
            return None, None
        r = max(faces, key=lambda r: r[2] * r[3])
        e1, e2 = np.array(r[4:6]), np.array(r[6:8])
        left, right = (e1, e2) if e1[0] < e2[0] else (e2, e1)
        nose = np.array(r[8:10])
        mouth = (np.array(r[10:12]) + np.array(r[12:14])) / 2
        E = (left + right) / 2
        d = mouth - E  # eye-centre -> mouth axis: sets the SCALE (eye spacing shrinks on 3/4 faces and zoomed the
        L = float(np.linalg.norm(d))  # alignment in so far that the mouth fell off the crop = a second mouth)
        u = d / max(L, 1e-6)
        perp = np.array([u[1], -u[0]])
        C = 0.5 * E + 0.5 * (nose - 0.33 * d)  # face midline between the eye centre and the nose-tip line
        brow_l, brow_r = C - 0.25 * d + 0.75 * L * perp, C - 0.25 * d - 0.75 * L * perp
        if brow_l[0] > brow_r[0]:
            brow_l, brow_r = brow_r, brow_l
        nose_c = C + 0.33 * d  # calibrated on the aligned crop (scratch align_test): brows-to-chin framing
        lmk = np.zeros((106, 2), np.float32)
        lmk[[43, 48, 49, 51, 50]] = brow_l
        lmk[101:106] = brow_r
        lmk[[74, 77, 83, 86]] = nose_c
        x, y, bw, bh = r[:4]
        return (int(x), int(y), int(x + bw), int(y + bh)), lmk


def run(clip, wav, offset, out, steps=8, guidance=1.0, rate=1.0):   # 20 / 1.5 (CFG) took ~3 h on the 8 GB card; 8 / 1.0 = 113 s, peak 3.8 GB
    sys.modules['insightface'] = types.ModuleType('insightface')  # never loaded: replaced below
    sys.modules['insightface.app'] = types.SimpleNamespace(FaceAnalysis=None)
    sys.path.insert(0, LS)
    cwd = os.getcwd()
    os.chdir(LS)
    os.environ['PATH'] = os.path.dirname(FFMPEG) + os.pathsep + os.environ['PATH']
    try:
        import torch
        from omegaconf import OmegaConf
        from diffusers import AutoencoderKL, DDIMScheduler
        import latentsync.utils.face_detector as fd
        fd.FaceDetector = YuNetDetector
        import latentsync.utils.image_processor as ip
        ip.FaceDetector = YuNetDetector
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature
        cfg = OmegaConf.load('configs/unet/stage2_512.yaml')
        # voice placed at its offset in a clip-long track
        cap = cv2.VideoCapture(clip)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        y, sr = sf.read(wav, dtype='float32')
        y = y.mean(1) if y.ndim > 1 else y
        if abs(rate - 1) > 1e-3:
            # the shot plays the clip at `rate` (<1 = slowed), so clip time = shot time x rate: the voice must be
            # time-compressed by `rate` here or the mouth drifts later and later (SyncNet: 440 ms off on mara_cu at .66).
            # Tempo change without a pitch change (the mouth model reads the audio features).
            t_in, t_out = tempfile.mktemp(suffix='.wav'), tempfile.mktemp(suffix='.wav')
            sf.write(t_in, y, sr)
            subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', t_in, '-filter:a', f'atempo={1 / rate:.5f}', '-ar', str(sr),
                            '-ac', '1', t_out], check=True)
            y, sr = sf.read(t_out, dtype='float32')
        tr = np.zeros(int(dur * sr), np.float32)
        s0 = int(offset * sr)
        tr[s0:s0 + len(y)] = y[:max(0, len(tr) - s0)]
        tmp = tempfile.mktemp(suffix='.wav')
        sf.write(tmp, tr, sr)
        dt = torch.float16
        sched = DDIMScheduler.from_pretrained('configs')
        enc = Audio2Feature(model_path='checkpoints/whisper/tiny.pt', device='cuda', num_frames=cfg.data.num_frames,
                            audio_feat_length=cfg.data.audio_feat_length)
        vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=dt)
        vae.config.scaling_factor, vae.config.shift_factor = 0.18215, 0
        unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model),
                                                       'checkpoints/latentsync_unet.pt', device='cpu')
        pipe = LipsyncPipeline(vae=vae, audio_encoder=enc, unet=unet.to(dtype=dt), scheduler=sched).to('cuda')
        pipe.enable_vae_slicing()   # 16 frames x 512 px encode/decode in one batch needs > 1 GB per layer: OOM under the 7.2 GB cap (spilled before)
        torch.manual_seed(1247)
        pipe(video_path=clip, audio_path=tmp, video_out_path=out, num_frames=cfg.data.num_frames,
             num_inference_steps=steps, guidance_scale=guidance, weight_dtype=dt, width=cfg.data.resolution,
             height=cfg.data.resolution, mask_image_path=cfg.data.mask_image_path, temp_dir=tempfile.mkdtemp())
        print(f'LATENTSYNC done -> {out}  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB', flush=True)
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    a = sys.argv[1:]
    run(os.path.abspath(a[0]), os.path.abspath(a[1]), float(a[2]), os.path.abspath(a[3]),
        int(a[4]) if len(a) > 4 else 8, float(a[5]) if len(a) > 5 else 1.0, float(a[6]) if len(a) > 6 else 1.0)
