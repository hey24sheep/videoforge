"""Score talking-head takes for face stability: per frame SFace identity vs the source image and face sharpness
(Laplacian variance of the face crop, relative to frame 0). Catches the 'smear + different face' defect that SyncNet misses.
    python tools/precision/talk_score.py <source_image> <take.mp4> [<take.mp4> ...]      (main or gen venv; OpenCV only)"""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vf import identity as I  # noqa: E402


def face_feat(img):
    h, w = img.shape[:2]
    _, faces = I._det(w, h).detect(img)
    if faces is None or not len(faces):
        return None, None
    f = faces[np.argmax(faces[:, 2] * faces[:, 3])]
    x, y, fw, fh = [int(v) for v in f[:4]]
    crop = img[max(0, y):y + fh, max(0, x):x + fw]
    sharp = cv2.Laplacian(cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    return I._rec().feature(I._rec().alignCrop(img, f)).astype(np.float32).ravel(), sharp


def main():
    src = cv2.imread(sys.argv[1])
    for take in sys.argv[2:]:
        c = cv2.VideoCapture(take); fs = []
        while True:
            ok, f = c.read()
            if not ok:
                break
            fs.append(f)
        h, w = fs[0].shape[:2]
        ref, _ = face_feat(cv2.resize(src, (w, int(src.shape[0] * w / src.shape[1])))[:h] if src.shape[0] * w / src.shape[1] >= h else cv2.resize(src, (w, h)))
        sims, sh = [], []
        for f in fs:
            e, s = face_feat(f)
            sims.append(I.similarity(ref, e) if e is not None else 0.0)
            sh.append(s or 0.0)
        sims, sh = np.array(sims), np.array(sh) / max(sh[0], 1e-6)
        print(f'{os.path.basename(take):18s} identity mean {sims.mean():.3f} min {sims.min():.3f} (f{sims.argmin()}) '
              f'frames<0.5: {(sims < 0.5).sum():2d}/{len(sims)} | sharpness min {sh.min():.2f} (f{sh.argmin()}) '
              f'frames<0.6: {(sh < 0.6).sum():2d}')


if __name__ == '__main__':
    main()
