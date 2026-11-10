"""W8A8 INT8 linears for the diffusers DiTs (K6, Z-Image, Klein), opt-in (2026-11-10 model-optimisation plan)."""
import gc, sys
import os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)


def _torch():
    import torch
    return torch


WAN2GP = os.path.join(_AI, 'tools', 'wan2gp')


def int8_linears(root, skip=('embed', 'proj_out', 'norm', 'modulation', 'adaLN', 'time', 'final')):
    """Replace every nn.Linear under `root` (dims multiple of 16, not in `skip`) by a W8A8 INT8 tensor-core matmul:
    the weight is int8 with one scale per output channel, the activation is quantised per row on the fly (WanGP's fused
    Triton kernel, imported from the Wan2GP checkout). Measured 2026-11-10 at DiT shapes on this card: 1.3x BF16 with
    relative error 0.011 (FP8 per-tensor: 1.45x, error 0.038). Sensitive layers in `skip` stay BF16.
    Quantises on the GPU one layer at a time. Returns the number of layers replaced."""
    torch = _torch()
    nn = torch.nn
    if WAN2GP not in sys.path:
        sys.path.append(WAN2GP)
    from shared.kernels import quanto_int8_triton as q8

    _mm = torch.compiler.disable(q8.fused_quant_scaled_mm)  # torch.compile cannot trace the WanGP Triton kernel

    class Int8Linear(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.in_features, self.out_features = lin.in_features, lin.out_features
            if lin.weight.is_meta:  # skeleton for load_baked
                self.register_buffer('w8', torch.empty(lin.weight.shape, dtype=torch.int8, device='meta'))
                self.register_buffer('ws', torch.empty(lin.out_features, dtype=torch.float32, device='meta'))
                self.register_buffer('bias', None if lin.bias is None else
                                     torch.empty(lin.bias.shape, dtype=torch.bfloat16, device='meta'))
                return
            w = lin.weight.data.to('cuda', torch.float32)
            s = (w.abs().amax(1) / 127.0).clamp(min=1e-12)
            self.register_buffer('w8', (w / s[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous())
            self.register_buffer('ws', s.contiguous())
            self.register_buffer('bias', None if lin.bias is None else lin.bias.data.to('cuda', torch.bfloat16))

        def forward(self, x):
            shp = x.shape
            x2 = x.reshape(-1, shp[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            y = _mm(x2, self.w8, self.ws, out_dtype=torch.bfloat16)
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*shp[:-1], self.out_features)

    n = 0
    for name, mod in list(root.named_modules()):
        for cn, ch in list(mod.named_children()):
            full = f'{name}.{cn}' if name else cn
            if (isinstance(ch, nn.Linear) and ch.in_features % 16 == 0 and ch.out_features % 16 == 0
                    and not any(k in full for k in skip)):
                setattr(mod, cn, Int8Linear(ch))
                n += 1
    gc.collect()
    torch.cuda.empty_cache()
    return n
