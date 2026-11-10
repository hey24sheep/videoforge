"""Text written INTO a prop (a page, a note, a label) instead of a text layer pasted over the frame.

Owner, 2026-11-09 ("the text should be part of the journal, not pasted on top of it, a huge bug"): the words are
drawn onto the shot's keyframe as ink (multiply blend, so the paper's light and grain stay), then carried onto the
finished clip by `platelock.stamp(ink=True)`, which tracks the page through the clip's motion.

Shot spec:  gen: {ink: {text: "11:40  DOCK 3\\nFOR ROOM #12", center: [0.70, 0.80], angle: -16,
                        size: 0.025, font: Inkfree.ttf, color: '#161c34', opacity: 0.88, spacing: 0.35}}
  center   text centre in keyframe fractions (x, y)
  angle    degrees, positive = counter-clockwise; match the page's ruled lines
  size     font size as a fraction of the keyframe height
  a list of such dicts writes several notes. Check the result on frames of the FINISHED clip: the text must sit
  fully on the page.
"""
import hashlib, json, os

from .memo import ROOT

FONT_DIRS = [r'C:\Windows\Fonts', os.path.join(ROOT, 'fonts')]


def _font(name, px):
    from PIL import ImageFont
    for d in [''] + FONT_DIRS:
        p = os.path.join(d, name) if d else name
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    raise FileNotFoundError(f'ink font {name} not found in {FONT_DIRS}')


def _rgb(c):
    if isinstance(c, str):
        c = c.lstrip('#')
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(int(x) for x in c)


def ink(key, notes):
    """key: keyframe path; notes: dict or list of dicts (see module doc). Returns (inked image path, box)."""
    from PIL import Image, ImageDraw, ImageFilter, ImageChops
    notes = notes if isinstance(notes, list) else [notes]
    h = hashlib.sha1(json.dumps([os.path.abspath(key), os.path.getmtime(key), notes], sort_keys=True).encode()).hexdigest()[:20]
    out = os.path.join(ROOT, 'cache', 'ink', h + '.png')
    meta = out + '.json'
    if os.path.exists(out) and os.path.exists(meta):
        return out, json.load(open(meta))['box']
    im = Image.open(key).convert('RGB')
    W, H = im.size
    ink_rgb = Image.new('RGB', im.size, (255, 255, 255))
    for n in notes:
        px = max(8, int(round(float(n.get('size', 0.0145)) * H)))
        f = _font(n.get('font', 'Inkfree.ttf'), px)
        txt = str(n['text'])
        probe = ImageDraw.Draw(Image.new('L', (8, 8)))
        sp = int(round(float(n.get('spacing', 0.35)) * px))
        bb = probe.multiline_textbbox((0, 0), txt, font=f, spacing=sp)
        layer = Image.new('L', (bb[2] + 8, bb[3] + 8), 0)
        ImageDraw.Draw(layer).multiline_text((4, 4), txt, font=f, fill=235, spacing=sp)
        layer = layer.rotate(float(n.get('angle', 0)), resample=Image.BICUBIC, expand=True).filter(ImageFilter.GaussianBlur(0.4))
        cx, cy = float(n['center'][0]) * W, float(n['center'][1]) * H
        mask = Image.new('L', im.size, 0)
        mask.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)))
        op = float(n.get('opacity', 0.88))
        ink_rgb = Image.composite(Image.new('RGB', im.size, _rgb(n.get('color', '#161c34'))), ink_rgb,
                                  mask.point(lambda v: int(v * op)))
    res = ImageChops.multiply(im, ink_rgb)
    diff = ImageChops.difference(im.convert('L'), res.convert('L')).point(lambda v: 255 if v > 6 else 0).getbbox()
    if not diff:
        raise ValueError(f'ink on {key}: nothing drawn (text off the frame?)')
    pad = 0.02
    box = [round(max(0, diff[0] / W - pad), 3), round(max(0, diff[1] / H - pad), 3),
           round(min(1, diff[2] / W + pad), 3), round(min(1, diff[3] / H + pad), 3)]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res.save(out)
    json.dump(dict(box=box, key=os.path.abspath(key), notes=notes), open(meta, 'w'))
    return out, box


def stamp_entry(key, notes):
    """The `gen.stamp` entry that carries the ink onto the clip."""
    img, box = ink(key, notes)
    return dict(img=img, key=os.path.abspath(key), box=box, ink=True, feather=0.004)
