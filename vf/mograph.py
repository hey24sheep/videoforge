"""Motion-graphics title sequences: photo / document cards floating in 3D, a drifting camera, depth fog, dust,
grain and two-line titles (a small white line over a big condensed accent line). CPU (OpenCV), 9:16 or 16:9.

Owner 2026-11-10: intro/outro "like" a cold-case show title animation (reference watched, nothing copied): every
frame moves (camera dolly + sway, cards turning), no held stills. Series-agnostic: everything comes from a YAML spec.

  python -m vf.mograph render <spec.yaml>        -> <spec dir>/out/<name>.mp4 (+ music muxed if the spec has one)

Spec (all times in s, positions in px at the focus plane; x right, y down, z away from camera):
  name, size: [1080, 1920], fps: 60, dur: 12
  bg: '#07080a'  fog: '#1a1f24'  accent: '#c22f3f'  grain: 0.05  vignette: 0.55  fade_out: 0.6
  camera: {z0: 0, z1: 900, sway: 40, roll: 1.5}         dolly forward z0 -> z1 (eased), slow x/y sway, roll in degrees
  cards: [{img, x, y, z, w, rx, ry, rz, spin: [drx, dry, drz] per s, bw: true, border: 0.03, at: 0, until: dur}]
  titles: [{small, big, at, until, y: 0.45, big_size: 0.13, small_size: 0.035}]
  figure: {img, at, x: 0.5, y: 0.62, h: 0.75, rise: 60}  a cut-out on black (screen-blended), fades in, rises slowly
  house: [{name: ACME, sub: PICTURES, top: 'AN', at, until, y: 0.47, size: 0.075, color}]   production-house card
  watermark: {text, size: 0.022, opacity: 0.55, margin: 0.03}   default: hey24sheep.com/videoforge; false = off
  music: {src, gain_db}
"""
import math, os, subprocess, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT = 'bahnschrift.ttf'


def _rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _font(px, var='Bold Condensed'):
    f = ImageFont.truetype(FONT, int(px))
    try:
        f.set_variation_by_name(var)
    except Exception:
        pass
    return f


def _card_tex(c, W):
    """RGBA float texture for a card: photo (optionally B&W) with a thin paper border and a soft shadow edge."""
    im = Image.open(c['img']).convert('RGB')
    w = int(c.get('w', 420))
    im = im.resize((w, int(w * im.height / im.width)), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255.0
    if c.get('bw', True):
        g = a @ np.array([0.299, 0.587, 0.114], np.float32)
        g = np.clip((g - 0.5) * 1.25 + 0.47, 0, 1)                 # a little more contrast, like old prints
        a = np.repeat(g[..., None], 3, 2) * np.array([1.0, 0.98, 0.95], np.float32)
    b = int(w * c.get('border', 0.03))
    if b:
        paper = np.ones((a.shape[0] + 2 * b, a.shape[1] + 2 * b, 3), np.float32) * np.array([0.86, 0.84, 0.80])
        paper[b:-b, b:-b] = a
        a = paper
    alpha = np.ones(a.shape[:2] + (1,), np.float32)
    return np.concatenate([a, alpha], 2)


def _rot(rx, ry, rz):
    rx, ry, rz = (math.radians(v) for v in (rx, ry, rz))
    X = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Z @ Y @ X


def _text_layer(W, H, title, t, accent):
    """Two-line title: small white line over a big accent line; fades in with letters spreading, blur clearing."""
    at, until = float(title['at']), float(title['until'])
    if not at <= t <= until:
        return None
    k_in, k_out = _ease((t - at) / 0.7), _ease((until - t) / 0.5)
    op = min(k_in, k_out)
    if op <= 0.003:
        return None
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = float(title.get('y', 0.45)) * H
    track = 1.0 + 0.06 * (1 - k_in) + 0.03 * ((t - at) / max(until - at, 1e-3))   # slow letter spread
    rows = []
    if title.get('small'):
        rows.append((title['small'], _font(H * float(title.get('small_size', 0.035)), 'SemiBold Condensed'), (235, 232, 226)))
    if title.get('big'):
        rows.append((title['big'], _font(H * float(title.get('big_size', 0.13))), tuple(int(v * 255) for v in accent)))
    for text, f, col in rows:
        ws = [f.getlength(ch) for ch in text]
        sp = f.size * 0.08 * track
        full = sum(ws) + sp * (len(text) - 1)
        if full > W * 0.86:                                             # never run off the frame: shrink to fit
            f = ImageFont.truetype(FONT, int(f.size * W * 0.86 / full))
            try:
                f.set_variation_by_name('Bold Condensed' if f is not rows[0][1] else 'SemiBold Condensed')
            except Exception:
                pass
            ws = [f.getlength(ch) for ch in text]
            sp = f.size * 0.08 * track
        tw = sum(ws) + sp * (len(text) - 1)
        x = (W - tw) / 2
        asc, desc = f.getmetrics()
        for ch, cw in zip(text, ws):
            d.text((x, y), ch, font=f, fill=col + (255,))
            x += cw + sp
        y += (asc + desc) * 0.92
    blur = 6 * (1 - k_in)
    if blur > 0.3:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(img, np.float32) / 255.0
    a[..., 3] *= op
    return a


def _house_layer(W, H, h, t):
    """Production-house card (a production house name shown in intro and outro, like a studio
    name"): big Bodoni name, a spaced small line under it between two thin rules, warm silver; fades in while the letter
    spacing settles; one slow soft light sweep crosses the letters (no flash)."""
    at, until = float(h['at']), float(h['until'])
    if not at <= t <= until:
        return None
    k_in, k_out = _ease((t - at) / 1.0), _ease((until - t) / 0.7)
    op = min(k_in, k_out)
    if op <= 0.003:
        return None
    col = tuple(int(v * 255) for v in _rgb(h.get('color', '#e2d8c4')))
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = float(h.get('y', 0.47)) * H

    def spaced(text, f, y, sp):
        ws = [f.getlength(ch) for ch in text]
        tw = sum(ws) + sp * (len(text) - 1)
        x = (W - tw) / 2
        for ch, cw in zip(text, ws):
            d.text((x, y), ch, font=f, fill=col + (255,))
            x += cw + sp
        return tw

    if h.get('top'):
        ft = ImageFont.truetype(h.get('font_small', 'BOD_I.TTF'), int(H * 0.022))
        spaced(h['top'], ft, y - H * 0.05, H * 0.006)
    fb = ImageFont.truetype(h.get('font', 'BOD_R.TTF'), int(H * float(h.get('size', 0.075))))
    settle = 1 - k_in
    tw = spaced(h['name'], fb, y, H * (0.012 + 0.02 * settle))
    asc, desc = fb.getmetrics()
    ly = y + asc + desc + H * 0.012
    sub = h.get('sub', 'PRODUCTIONS')
    fs = ImageFont.truetype(h.get('font_small', 'BOD_I.TTF'), int(H * 0.024))
    sp = H * (0.010 + 0.006 * settle)
    full = sum(fs.getlength(ch) for ch in sub) + sp * (len(sub) - 1)
    if full > W * 0.8:                                                  # fit the spaced line inside the frame
        k = W * 0.8 / full
        fs, sp = ImageFont.truetype(h.get('font_small', 'BOD_I.TTF'), int(H * 0.024 * k)), sp * k
    sw = spaced(sub, fs, ly, sp)
    rule = max(tw, sw) * 0.5 * k_in
    for yy in (y - H * 0.014, ly + H * 0.042):
        d.line([(W / 2 - rule, yy), (W / 2 + rule, yy)], fill=col + (200,), width=max(1, H // 900))
    a = np.asarray(img, np.float32) / 255.0
    sweep = (t - at) / max(until - at, 1e-3) * 1.6 - 0.3                 # 0..1 across the card, once
    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    glow = np.exp(-((xs - sweep) / 0.06) ** 2) * 0.55
    a[..., :3] = np.clip(a[..., :3] + glow[..., None], 0, 1)
    a[..., 3] *= op
    return a


def render(spec_path):
    import yaml
    S = yaml.safe_load(open(spec_path, encoding='utf-8'))
    base = os.path.dirname(os.path.abspath(spec_path))
    W, H = S.get('size', [1080, 1920])
    fps, dur = int(S.get('fps', 60)), float(S['dur'])
    N = int(round(fps * dur))
    bg, fogc, accent = _rgb(S.get('bg', '#07080a')), _rgb(S.get('fog', '#1a1f24')), _rgb(S.get('accent', '#c22f3f'))
    cam = S.get('camera', {})
    F = float(cam.get('focal', 1400))
    cards = []
    for c in S.get('cards', []):
        c = dict(c)
        c['img'] = c['img'] if os.path.isabs(c['img']) else os.path.join(base, c['img'])
        c['_tex'] = _card_tex(c, W)
        cards.append(c)
    fig = S.get('figure')
    if fig:
        fi = Image.open(fig['img'] if os.path.isabs(fig['img']) else os.path.join(base, fig['img'])).convert('L')
        fh = int(H * float(fig.get('h', 0.75)))
        fi = fi.resize((int(fi.width * fh / fi.height), fh), Image.LANCZOS)
        a = np.asarray(fi, np.float32) / 255.0
        if fig.get('key') == 'white':   # figure shot on pure white (image-edit model): white keys out exactly
            m = np.clip((0.93 - cv2.GaussianBlur(a, (0, 0), 1.5)) / 0.10, 0, 1)
        else:                            # figure on black: only very dark pixels drop out (dark clothes may leak)
            m = np.clip((cv2.GaussianBlur(a, (0, 0), 6) - 0.05) / 0.15, 0, 1)
        fig['_a'], fig['_m'] = a, cv2.GaussianBlur(m, (0, 0), 4)
    rng = np.random.default_rng(int(S.get('seed', 7)))
    # fog: a big soft noise field panned slowly; dust: drifting specks in 3D
    fog = cv2.GaussianBlur(rng.random((H // 8 + 64, W // 8 + 64)).astype(np.float32), (0, 0), 9)
    fog = (fog - fog.min()) / (fog.max() - fog.min() + 1e-6)
    dust = rng.uniform([-W, -H, 200], [W, H, 2600], (int(S.get('dust', 260)), 3))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - float(S.get('vignette', 0.55)) * np.clip(((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2 * 0.6) * 2.2, 0, 1)
    wm = None
    m = S.get('watermark', {'text': 'hey24sheep.com/videoforge', 'size': 0.017 * min(W, H) / H, 'opacity': 0.6,
                            'margin': 0.02 * min(W, H) / W})   # same pixel size as the vf build mark
    if isinstance(m, str):
        m = {'text': m}
    if m and os.environ.get('VF_WATERMARK', '1') != '0':   # default corner mark; `watermark: false` or VF_WATERMARK=0 = off
        img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        f = _font(H * float(m.get('size', 0.022)), 'SemiBold')
        tw = f.getlength(m['text'])
        mg = float(m.get('margin', 0.03)) * W
        asc, desc = f.getmetrics()
        ImageDraw.Draw(img).text((W - mg - tw, H - mg - asc - desc), m['text'], font=f, fill=(255, 255, 255, 255))
        wm = np.asarray(img, np.float32) / 255.0
        wm[..., 3] *= float(m.get('opacity', 0.55))
    out_dir = os.path.join(base, 'out')
    os.makedirs(out_dir, exist_ok=True)
    silent = os.path.join(out_dir, S['name'] + '_video.mp4')
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
                          '-i', '-', '-c:v', 'libx264', '-crf', '16', '-preset', 'medium', '-pix_fmt', 'yuv420p', silent],
                         stdin=subprocess.PIPE)
    for i in range(N):
        t = i / fps
        k = _ease(t / dur)
        cz = float(cam.get('z0', 0)) + (float(cam.get('z1', 900)) - float(cam.get('z0', 0))) * k
        cx = float(cam.get('sway', 40)) * math.sin(t * 0.45)
        cy = float(cam.get('sway', 40)) * 0.5 * math.sin(t * 0.31 + 1.0)
        roll = math.radians(float(cam.get('roll', 1.5)) * math.sin(t * 0.27))
        fr = np.empty((H, W, 3), np.float32)
        fr[:] = bg
        fo = cv2.resize(fog[int(t * 3) % 64:int(t * 3) % 64 + H // 8, :W // 8], (W, H), interpolation=cv2.INTER_LINEAR)
        fr += (fogc - bg) * (0.35 + 0.65 * fo[..., None]) * 0.9
        # cards, far to near
        for c in sorted(cards, key=lambda c: -(float(c['z']) - cz)):
            if not float(c.get('at', 0)) <= t <= float(c.get('until', dur)):
                continue
            tex = c['_tex']
            th, tw = tex.shape[:2]
            sp = c.get('spin', [0, 0, 0])
            R = _rot(float(c.get('rx', 0)) + sp[0] * t, float(c.get('ry', 0)) + sp[1] * t, float(c.get('rz', 0)) + sp[2] * t)
            corners = np.array([[-tw / 2, -th / 2, 0], [tw / 2, -th / 2, 0], [tw / 2, th / 2, 0], [-tw / 2, th / 2, 0]]) @ R.T
            corners += [float(c['x']) - cx, float(c['y']) - cy, float(c['z']) - cz + F]
            if corners[:, 2].min() < 60:
                continue
            px = corners[:, 0] * F / corners[:, 2]
            py = corners[:, 1] * F / corners[:, 2]
            cr, sr = math.cos(roll), math.sin(roll)
            dst = np.stack([px * cr - py * sr + W / 2, px * sr + py * cr + H / 2], 1).astype(np.float32)
            src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
            x0, y0 = max(0, int(dst[:, 0].min()) - 2), max(0, int(dst[:, 1].min()) - 2)   # warp only the card's box
            x1, y1 = min(W, int(dst[:, 0].max()) + 3), min(H, int(dst[:, 1].max()) + 3)
            if x1 <= x0 or y1 <= y0:
                continue
            M = cv2.getPerspectiveTransform(src, dst - np.array([x0, y0], np.float32))
            warped = cv2.warpPerspective(tex, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            depth = float(c['z']) - cz + F
            haze = np.clip((depth - 900) / 2600, 0, 0.85)                  # depth fog
            fade = min(_ease((t - float(c.get('at', 0))) / 0.8), _ease((float(c.get('until', dur)) - t) / 0.8))
            col = warped[..., :3] * (1 - haze) + fogc * haze
            al = warped[..., 3:4] * fade
            fr[y0:y1, x0:x1] = fr[y0:y1, x0:x1] * (1 - al) + col * al
        # figure (screen blend: black = transparent)
        if fig and t >= float(fig.get('at', 0)):
            kf = _ease((t - float(fig['at'])) / 1.6)
            a = fig['_a']
            fh, fw = a.shape
            x0 = int(W * float(fig.get('x', 0.5)) - fw / 2)
            y0 = int(H * float(fig.get('y', 0.62)) - fh / 2 + float(fig.get('rise', 60)) * (1 - kf))
            ys, xs = max(0, y0), max(0, x0)
            ye, xe = min(H, y0 + fh), min(W, x0 + fw)
            if ye > ys and xe > xs:
                sub = a[ys - y0:ye - y0, xs - x0:xe - x0][..., None]
                mm = fig['_m'][ys - y0:ye - y0, xs - x0:xe - x0][..., None] * kf
                fr *= (1 - 0.55 * kf)                                    # the whole board sinks into the dark
                fr[ys:ye, xs:xe] = fr[ys:ye, xs:xe] * (1 - mm) + sub * mm
        # dust specks
        d = dust.copy()
        d[:, 1] = (d[:, 1] + 18 * t + H) % (2 * H) - H
        z = d[:, 2] - cz * 0.6 + F
        ok = z > 80
        sx = ((d[ok, 0] - cx) * F / z[ok] + W / 2).astype(int)
        sy = ((d[ok, 1] - cy) * F / z[ok] + H / 2).astype(int)
        br = np.clip(1.4 - z[ok] / 2600, 0.15, 0.9) * 0.35
        m = (sx >= 1) & (sx < W - 1) & (sy >= 1) & (sy < H - 1)
        fr[sy[m], sx[m]] += br[m, None]
        # titles
        for ti in S.get('titles', []):
            L = _text_layer(W, H, ti, t, accent)
            if L is not None:
                fr = fr * (1 - L[..., 3:4]) + L[..., :3] * L[..., 3:4]
        for h in S.get('house', []):
            L = _house_layer(W, H, h, t)
            if L is not None:
                fr *= 1 - 0.5 * float(L[..., 3].max())                    # dim the board so the name reads
                fr = fr * (1 - L[..., 3:4]) + L[..., :3] * L[..., 3:4]
        fr *= vig[..., None]
        fr += (rng.standard_normal((H // 2, W // 2)).astype(np.float32).repeat(2, 0).repeat(2, 1)[..., None]
               * float(S.get('grain', 0.05)))
        fo_t = float(S.get('fade_out', 0.6))
        if fo_t and t > dur - fo_t:
            fr *= _ease((dur - t) / fo_t)
        if wm is not None:                                              # watermark stays through the fade
            fr = fr * (1 - wm[..., 3:4]) + wm[..., :3] * wm[..., 3:4]
        p.stdin.write((np.clip(fr, 0, 1) * 255).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    final = os.path.join(out_dir, S['name'] + '.mp4')
    mu = S.get('music')
    if mu:
        mu = dict(mu, src=mu['src'] if os.path.isabs(mu['src']) else os.path.join(base, mu['src']))
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', silent, '-i', mu['src'], '-filter_complex',
                        f"[1:a]volume={float(mu.get('gain_db', 0))}dB,atrim=0:{dur},apad,"
                        f"afade=t=out:st={dur - 0.8}:d=0.8,loudnorm=I={float(S.get('lufs', -18))}:TP=-1.5[a]",
                        '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-t', str(dur), final],
                       check=True)
    else:
        os.replace(silent, final)
    return final


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'render':
        print(render(sys.argv[2]))
    else:
        print(__doc__)
