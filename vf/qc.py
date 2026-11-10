"""Automatic quality checks. Severity 'error' blocks the final build (override: --force), 'warn' is reported.

Per shot (from per-frame mean luminance gathered on the GPU): black / blown-out frames, flicker spikes that are
not cuts, transitions or flashes. Layout: every text box (its real rendered size) inside the platform safe area
(Shorts/Reels UI covers the bottom ~20% and the right edge). Sources: cut-outs whose alpha has long straight edges
(matte slabs / frame crops). Results are stored next to each cached segment, so cache hits still report them.
"""
import json, os
import numpy as np
from .spec import FORMATS, for_format, kf, resolve

SAFE = {'short': dict(x0=0.06, x1=0.88, y0=0.10, y1=0.78), 'yt': dict(x0=0.04, x1=0.96, y0=0.04, y1=0.96)}
SAFE['yt4k'] = SAFE['yt']


class QC:
    def __init__(self, spec, fmt):
        self.spec, self.fmt = spec, fmt
        self.items = []

    def add(self, sev, where, msg):
        self.items.append(dict(severity=sev, where=where, msg=msg))

    # ---- render-time
    def check_shot(self, i, lum, fps):
        """lum: mean display luminance (0-255) per frame. Returns this shot's findings (stored with its segment)."""
        s = self.spec['shots'][i]
        sid, out = s['id'], []
        tr = float((s.get('transition') or {}).get('dur', 0))
        flashes = [float(t) for t, _ in (s.get('flash') or [])]
        prev = None
        for k, m in enumerate(lum):
            u = k / fps
            near_event = u < tr + 0.05 or any(abs(u - t) < 0.15 for t in flashes)
            where = f'{self.fmt}:{sid}@{u:.2f}s'
            if m < 1.5 and not (s.get('allow_black') or near_event):
                out.append(dict(severity='warn', where=where, msg='black frame'))
            if m > 252 and not near_event:
                out.append(dict(severity='warn', where=where, msg='blown-out white frame'))
            if prev is not None and abs(m - prev) > 40 and not near_event:
                out.append(dict(severity='warn', where=where, msg=f'flicker: luminance jumped {prev:.0f}->{m:.0f}'))
            prev = m
        return out

    # ---- before rendering
    def layout(self):
        from .layers import text_bitmap, _hex8
        sa = SAFE.get(self.fmt)
        if not sa:
            return
        W, H = FORMATS[self.fmt]['W'], FORMATS[self.fmt]['H']
        m = min(W, H)
        for s in self.spec['shots']:
            for L in s.get('layers', []) + (self.spec['_captions'].get(self.fmt, {}).get(s['id']) or []):
                L = for_format(L, self.fmt)
                if L.get('type') != 'text':
                    continue
                px = max(8, int(float(L.get('size', 0.06)) * m))
                font = self.spec['fonts'].get(L.get('font', 'title'), L.get('font', 'title'))
                bm = text_bitmap(L['text'], font, px, _hex8(L.get('color', '#ffffff')),
                                 int(px * float(L.get('stroke', 0.0))), _hex8(L.get('stroke_color', '#000000')),
                                 bool(L.get('shadow', True)), int(float(L.get('max_w', 0.86)) * W),
                                 L.get('align', 'center'), 1.15, float(L.get('plate', 0)))
                bh, bw = bm.shape[:2]
                ys, xs = np.nonzero(bm[..., 3] > 48)  # visible glyphs, not the transparent shadow padding
                if not len(xs):
                    continue
                gx0, gx1, gy0, gy1 = xs.min() - bw / 2, xs.max() - bw / 2, ys.min() - bh / 2, ys.max() - bh / 2
                for t in sorted({0.0, float(L.get('at', 0)), min(s['dur'], float(L.get('until', s['dur'])))}):
                    x, y = float(kf(L.get('x', 0.5), t)), float(kf(L.get('y', 0.5), t))
                    x0, x1 = x + gx0 / W, x + gx1 / W
                    y0, y1 = y + gy0 / H, y + gy1 / H
                    if x0 < sa['x0'] - 0.02 or x1 > sa['x1'] + 0.02 or y0 < sa['y0'] - 0.02 or y1 > sa['y1'] + 0.02:
                        # in-scene props (writing on paper, numbers on a badge) are part of the picture: warn
                        self.add('warn' if L.get('in_scene') else 'error', f'{self.fmt}:{s["id"]}',
                                 f'text "{L["text"][:30]}" box '
                                 f'({x0:.2f}-{x1:.2f}, {y0:.2f}-{y1:.2f}) leaves the {self.fmt} safe area')
                        break

    def sources(self):
        """Matte slabs: an alpha edge that runs perfectly straight for >25% of the sprite height/width."""
        from PIL import Image
        from .layers import _frame_files
        seen = set()
        for s in self.spec['shots']:
            for L in s.get('layers', []):
                if L.get('type') != 'sprite' or not L.get('src'):
                    continue
                p = resolve(self.spec, L['src'])
                if p in seen:
                    continue
                seen.add(p)
                files = _frame_files(p) or [p]
                for f in files[::max(1, len(files) // 8)]:
                    a = np.asarray(Image.open(f).convert('RGBA'))[..., 3] > 128
                    h, w = a.shape
                    if (a[:, 1:] != a[:, :-1]).sum(0).max() > 0.25 * h or (a[1:] != a[:-1]).sum(1).max() > 0.25 * w:
                        self.add('warn', f'source:{os.path.basename(p)}', f'straight matte edge in '
                                 f'{os.path.basename(f)} (slab or frame crop): regenerate or re-matte')
                        break

    def report(self, out_dir, segments=()):
        """Merge pre-render findings with each segment's stored findings; collapse repeats; write qc_<fmt>.json."""
        items = list(self.items)
        for seg in segments:
            p = seg + '.qc.json'
            if os.path.exists(p):
                items += json.load(open(p))
        seen, out = set(), []
        for it in items:
            k = (it['severity'], it['where'].split('@')[0], it['msg'].split(':')[0])
            if k not in seen:
                seen.add(k)
                out.append(it)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(out, open(os.path.join(out_dir, f'qc_{self.fmt}.json'), 'w'), indent=1)
        return out
