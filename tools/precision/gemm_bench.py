"""GEMM throughput on this GPU at DiT shapes: BF16 vs FP8 (_scaled_mm) vs torch INT8 (_int_mm) vs WanGP fused INT8 W8A8,
with relative error vs fp32. Run in the talk or gen venv (needs Triton + the Wan2GP checkout).
    python tools/precision/gemm_bench.py [--m 32768]
2026-11-10 (RTX 4060 Laptop, M=32768): bf16 1x | WanGP int8 1.27-1.35x relerr 0.011 | fp8 1.34-1.61x relerr 0.038."""
import argparse, sys, time
import os
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)
import torch
WAN2GP = os.path.join(_AI, 'tools', 'wan2gp')


def bench(fn, it=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, default=32768, help='tokens (480x832x81 Wan latent = ~32760)')
    a = ap.parse_args()
    sys.path.insert(0, WAN2GP)
    from shared.kernels import quanto_int8_triton as q
    M, dev = a.m, 'cuda'
    print(torch.cuda.get_device_name(), torch.__version__)
    for K, N, name in [(5120, 5120, 'attn 5120x5120'), (5120, 13824, 'ffn up'), (13824, 5120, 'ffn down')]:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        s = (w.abs().amax(1) / 127).float(); wi = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
        w8 = w.to(torch.float8_e4m3fn).t(); one = torch.tensor(1.0, device=dev)

        def f8():
            sa = x.abs().amax().float() / 448
            return torch._scaled_mm((x / sa).to(torch.float8_e4m3fn), w8, scale_a=sa, scale_b=one, out_dtype=torch.bfloat16)
        ref = x.float() @ w.float().t()
        err = lambda y: ((y.float() - ref).norm() / ref.norm()).item()
        tb = bench(lambda: x @ w.t()); ti = bench(lambda: q.fused_quant_scaled_mm(x, wi, s)); tf = bench(f8)
        print(f'{name:15s} bf16 {tb*1e3:6.1f} ms | wgp int8 {tb/ti:.2f}x err {err(q.fused_quant_scaled_mm(x, wi, s)):.4f}'
              f' | fp8 {tb/tf:.2f}x err {err(f8()):.4f}')


if __name__ == '__main__':
    main()
