"""Character identity check for key frames (OpenCV YuNet face detector + SFace recogniser, both from opencv_zoo,
MIT / Apache-2.0; ~40 MB total, CPU, ~30 ms per image).

Z-Image-Turbo takes no reference image, so the same character drifts between shots (different face, beard, hat).
The keyframe stage therefore draws a few candidates per character shot and keeps the one whose face is closest to
the character's ANCHOR portrait (one fixed-seed portrait per character, generated once from the series sheet).
"""
import os
import numpy as np
import cv2
from .memo import memo, tier_dir

SFACE_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/'
             'face_recognition_sface_2021dec.onnx')
_M = {}


def _model(name, url):
    d = os.path.join(tier_dir('global'), 'models', name)
    p = os.path.join(d, os.path.basename(url))
    if not os.path.exists(p):
        os.makedirs(d, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, p + '.tmp')
        os.replace(p + '.tmp', p)
    return p


def _det(w, h):
    from .clipqc import _yunet_path
    det = _M.get('det')
    if det is None:
        det = _M['det'] = cv2.FaceDetectorYN.create(_yunet_path(), '', (w, h), 0.6, 0.3, 20)
    det.setInputSize((w, h))
    return det


def _rec():
    if 'rec' not in _M:
        _M['rec'] = cv2.FaceRecognizerSF.create(_model('sface', SFACE_URL), '')
    return _M['rec']


@memo('global', version='identity-1')
def embedding(image_path):
    """128-d SFace feature of the largest face, or None (no face / face < 40 px)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, 1280 / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
    _, faces = _det(small.shape[1], small.shape[0]).detect(small)
    if faces is None or not len(faces):
        return None
    f = faces[np.argmax(faces[:, 2] * faces[:, 3])].copy()
    if f[2] / s < 40:
        return None
    f[:14] /= s
    crop = _rec().alignCrop(img, f)
    return _rec().feature(crop).astype(np.float32).ravel()


def similarity(a, b):
    """Cosine similarity of two embeddings (SFace: >= 0.36 is usually the same person); None if either is None."""
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def pick(anchor_path, candidates):
    """Index of the candidate image most like the anchor face, plus scores. Falls back to 0 when no face is
    measurable (wide shots, faces from behind)."""
    ea = embedding(anchor_path)
    scores = [similarity(ea, embedding(c)) for c in candidates]
    valid = [(s, i) for i, s in enumerate(scores) if s is not None]
    return (max(valid)[1] if valid else 0), scores
