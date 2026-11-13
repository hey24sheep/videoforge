"""Text written INTO a prop (a page, a note, a label) instead of a text layer pasted over the frame.

Owner, 2026-11-09 ("the text should be part of the journal, not pasted on top of it, a huge bug"): the words are
drawn onto the shot's keyframe as ink (multiply blend, so the paper's light and grain stay), then carried onto the
finished clip by `platelock.stamp(ink=True)`, which tracks the page through the clip's motion.

Shot spec:  gen: {ink: {text: "11:40  DOCK 3\\nFOR ROOM #12", center: [0.70, 0.80], angle: -16,
                        size: 0.025, font: Inkfree.ttf, color: '#161c34', opacity: 0.88, spacing: 0.35}}
  center   text centre in keyframe fractions (x, y)
  angle    degrees, positive = counter-clockwise; match the page's ruled lines
  size     font size as a fraction of the keyframe height
  style    ink (default: multiply on paper) | engrave (cut into the surface: groove shading, lit far wall) |
           emboss (raised: lit near edge, shadow far edge). engrave/emboss use `color` x `opacity` as the groove/
           letter tint (bare metal: a dark grey at ~0.3) and `depth` (default 1.0) for the relief strength
  light    direction the light comes FROM, degrees (default 135 = top left)
  bake     true on ANY note: the inked keyframe is the image the video model animates, so the text moves with the
           prop (no stamp after the clip; a tracked stamp can drift off a moving prop). Needs a still, simple motion.
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


def _relief(im, items):
    """Engraved / embossed letters: the letter mask is a height map lit by one light (Lambert-style slope shading).
    engrave = a groove: the edge nearest the light falls in shadow, the far wall catches light, the floor is tinted
    by color x opacity. emboss = raised: the near edge is lit, the far edge casts a thin shadow."""
    import numpy as np
    from PIL import Image, ImageFilter
    out = np.asarray(im, np.float32)
    for n, mask, px in items:
        m = np.asarray(mask, np.float32)[..., None] / 255.0
        r = np.asarray(mask.filter(ImageFilter.GaussianBlur(max(0.8, px * 0.05))), np.float32) / 255.0
        gy, gx = np.gradient(r)
        a = np.deg2rad(float(n.get('light', 135)))
        to_light = np.cos(a), -np.sin(a)                      # image y points down
        slope = gx * to_light[0] + gy * to_light[1]           # > 0: the mask rises towards the light (letter's far edge)
        shade = slope if n['style'] == 'engrave' else -slope  # groove: near edge dark; raised: near edge lit
        shade = np.clip(shade / (float(np.abs(shade).max()) or 1.0) * float(n.get('depth', 1.0)), -1, 1)[..., None]
        col = np.asarray(_rgb(n.get('color', '#2a2a2a')), np.float32) / 255.0
        fill = m * (1.0 if n['style'] == 'engrave' else 0.35) * float(n.get('opacity', 0.3))
        out = out * (1 - fill * (1 - col))
        out = np.where(shade > 0, out + (255 - out) * shade * 0.55, out * (1 + shade * 0.75))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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
    ink_rgb = Image.new('RGB', im.size, (255, 255, 255))   # plain ink notes: the original multiply blend
    relief = []
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
        style = n.get('style', 'ink')
        if style in ('engrave', 'emboss'):
            relief.append((n, mask, px))
            continue
        if style != 'ink':
            raise ValueError(f'ink style {style!r}: use ink, engrave or emboss')
        op = float(n.get('opacity', 0.88))
        ink_rgb = Image.composite(Image.new('RGB', im.size, _rgb(n.get('color', '#161c34'))), ink_rgb,
                                  mask.point(lambda v: int(v * op)))
    res = ImageChops.multiply(im, ink_rgb)
    if relief:
        res = _relief(res, relief)
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


def baked(notes):
    """True when the text goes into the keyframe the video model animates (no stamp afterwards)."""
    return any(n.get('bake') for n in (notes if isinstance(notes, list) else [notes]))


def stamp_entry(key, notes):
    """The `gen.stamp` entry that carries the ink onto the clip."""
    img, box = ink(key, notes)
    return dict(img=img, key=os.path.abspath(key), box=box, ink=True, feather=0.004)
