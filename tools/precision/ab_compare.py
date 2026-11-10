"""SSIM / PSNR of precision variants vs a reference (normally the model's own bf16 output, same seed), plus a sheet.
    python tools/precision/ab_compare.py <dir> <ref> <variant,variant> <name,name> [--sheet out.jpg]
Files: <dir>/<variant>_<name>.mp4 or .png. Videos: all frames. Run with the gen venv (skimage)."""
import argparse, os
import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim


def frames(p):
    if p.endswith('.png'):
        return [cv2.imread(p)]
    c, out = cv2.VideoCapture(p), []
    while True:
        ok, f = c.read()
        if not ok:
            return out
        out.append(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('ref'); ap.add_argument('variants'); ap.add_argument('names')
    ap.add_argument('--sheet')
    a = ap.parse_args()
    rows = []
    for n in a.names.split(','):
        ext = '.mp4' if os.path.exists(f'{a.dir}/{a.ref}_{n}.mp4') else '.png'
        ref = frames(f'{a.dir}/{a.ref}_{n}{ext}')
        row = []
        for v in a.variants.split(',') + [a.ref]:
            fs = frames(f'{a.dir}/{v}_{n}{ext}')
            k = min(len(fs), len(ref))
            if v != a.ref:
                s = [ssim(cv2.cvtColor(x, 6), cv2.cvtColor(y, 6)) for x, y in zip(fs[:k], ref[:k])]
                p = [cv2.PSNR(x, y) for x, y in zip(fs[:k], ref[:k])]
                print(f'{n:8s} {v:9s} vs {a.ref}: SSIM mean {np.mean(s):.4f} min {np.min(s):.4f} | PSNR {np.mean(p):.2f} dB | {k} frame(s)')
            f = fs[len(fs) // 2].copy()
            cv2.putText(f, v, (8, 30), 0, 1, (0, 255, 255), 2)
            row.append(f)
        rows.append(np.hstack(row))
    if a.sheet:
        w = min(r.shape[1] for r in rows)
        cv2.imwrite(a.sheet, np.vstack([cv2.resize(r, (w, int(r.shape[0] * w / r.shape[1]))) for r in rows]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == '__main__':
    main()
