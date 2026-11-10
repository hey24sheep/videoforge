"""Tiny video decoder for Wan latents (TAEHV architecture by Ollin Boer Bohan, MIT; weights lightx2v/Autoencoders).

Our own re-implementation of the decoder half (weights load strictly, so any structural mismatch fails loudly).
Input: NORMALISED Wan2.2 latents (the pipeline's sampling-space latents, before *std+mean), shape (N, 48, T, h, w).
Output: (N, 4T-3, 3, 16h, 16w) in [0, 1]. Streams over time in chunks, so VRAM stays ~1-2 GB at 480p.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = 'lightx2v/Autoencoders'
FILES = {'taew2_2': 'taew2_2.safetensors', 'lighttaew2_2': 'lighttaew2_2.safetensors',
         'taew2_1': 'taew2_1.safetensors', 'lighttaew2_1': 'lighttaew2_1.safetensors'}


def conv(n_in, n_out, bias=True):
    return nn.Conv2d(n_in, n_out, 3, padding=1, bias=bias)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out),
                                  nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):  # (T, C, H, W) -> (T*stride, C, H, W)
        _, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


class TinyWanDecoder(nn.Module):
    def __init__(self, latent_channels=48, patch_size=2, time_upscale=(True, True), n_f=(256, 128, 64, 64)):
        super().__init__()
        self.patch_size = patch_size
        self.frames_to_trim = 2 ** sum(time_upscale) - 1
        self.decoder = nn.Sequential(
            Clamp(), conv(latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2), TGrow(n_f[0], 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2), TGrow(n_f[1], 2 if time_upscale[0] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2), TGrow(n_f[2], 2 if time_upscale[1] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], 3 * patch_size ** 2))

    @classmethod
    def load(cls, name='taew2_2', device='cuda', dtype=torch.float16):
        from safetensors.torch import load_file
        from .fetch_models import tae_path
        p = tae_path(name)
        sd ={k: v for k, v in load_file(p).items() if k.startswith('decoder.')}
        m = cls(latent_channels=sd['decoder.1.weight'].shape[1])
        m.load_state_dict(sd, strict=True)
        return m.to(device, dtype).eval().requires_grad_(False)

    @torch.no_grad()
    def decode(self, latents, chunk=4):
        """latents (N, C, T, h, w) normalised -> frames (N, 4T-3, 3, H, W) float in [0, 1] (on the model device)."""
        dev, dt = next(self.parameters()).device, next(self.parameters()).dtype
        outs = []
        for n in range(latents.shape[0]):
            lat = latents[n].permute(1, 0, 2, 3)  # (T, C, h, w)
            mem = {}
            frames = []
            for t0 in range(0, lat.shape[0], chunk):
                x = lat[t0:t0 + chunk].to(dev, dt)
                for i, b in enumerate(self.decoder):
                    if isinstance(b, MemBlock):
                        prev = mem.get(i)
                        if prev is None:
                            prev = torch.zeros_like(x[:1])
                        past = torch.cat([prev, x[:-1]], 0)
                        mem[i] = x[-1:].clone()
                        x = b(x, past)
                    else:
                        x = b(x)
                x = x.clamp_(0, 1)
                if self.patch_size > 1:
                    x = F.pixel_shuffle(x, self.patch_size)
                frames.append(x)
            outs.append(torch.cat(frames, 0)[self.frames_to_trim:])
        return torch.stack(outs)
