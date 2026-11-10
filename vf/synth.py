"""Procedural sound: ambience beds, stingers and simple scores, generated in numpy and cached (global tier).

In episode.yaml any audio `src:` may be `synth:<kind>?key=value&...`, e.g.
    music: {src: "synth:drone?dur=60&root=A1&dark=0.7", gain_db: -10}
    sfx:   - {src: "synth:ring?dur=6", at: 0.4}
Kinds (all stereo 48 kHz; every kind takes dur= and seed=):
    rain      intensity=0.7  (noise bed + droplets + distant rumble)
    drone     root=A1 dark=0.6 (detuned low pad, slow filter movement; minor colour)
    piano     notes="A3@0;C4@2.5;E4@5" vel=0.6 room=0.5 (sparse felt-piano notes, name@seconds)
    ring      pattern=2,4  (old electromechanical phone bell: ring s, pause s)
    heartbeat bpm=60 rise=0  (low lub-dub; rise= bpm added by the end)
    thunder   (crack + long rumble)
    hit       (low boom / reveal stinger)
    riser     (noise + tone sweeping up into a cut)
    tick      bpm=60 (clock tick)
"""
import math, os
from urllib.parse import parse_qsl
import numpy as np
from .memo import memo_file

SR = 48000
NOTES = {'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6, 'E': -5, 'F': -4, 'F#': -3, 'Gb': -3, 'G': -2,
         'G#': -1, 'Ab': -1, 'A': 0, 'A#': 1, 'Bb': 1, 'B': 2}


def hz(name):
    """'A4' -> 440.0 ; numbers pass through."""
    try:
        return float(name)
    except ValueError:
        n, o = name[:-1], int(name[-1])
        return 440.0 * 2 ** ((NOTES[n] + 12 * (o - 4)) / 12)


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _bp(x, lo, hi, order=2):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(order, [lo, hi], 'bp', fs=SR, output='sos'), x, axis=-1)


def _lp(x, f, order=2):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(order, f, 'lp', fs=SR, output='sos'), x, axis=-1)


def _hp(x, f, order=2):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(order, f, 'hp', fs=SR, output='sos'), x, axis=-1)


def _pink(n, rng):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    f[1:] /= np.sqrt(np.arange(1, len(f)))
    return np.fft.irfft(f, n) / 30


def _reverb(x, room=0.5, rng=None):
    """Cheap stereo reverb: decaying-noise impulse response, convolved by FFT."""
    from scipy.signal import fftconvolve
    rng = rng or np.random.default_rng(7)
    n = int((0.6 + 2.4 * room) * SR)
    env = np.exp(-np.arange(n) / (0.25 * n))
    ir = np.stack([rng.standard_normal(n) * env, rng.standard_normal(n) * env])
    ir = _lp(ir, 5000)
    ir /= np.abs(ir).sum(1, keepdims=True) ** 0.5 * 12
    wet = np.stack([fftconvolve(x[c], ir[c])[:x.shape[1]] for c in range(2)])
    return x * (1 - 0.35 * room) + wet * room


def _fade(x, a=0.02, b=0.05):
    n = x.shape[-1]
    ia, ib = min(n, int(a * SR)), min(n, int(b * SR))
    if ia:
        x[..., :ia] *= np.linspace(0, 1, ia)
    if ib:
        x[..., -ib:] *= np.linspace(1, 0, ib)
    return x


# ------------------------------------------------------------------ kinds
def rain(dur, rng, intensity=0.7):
    n = int(dur * SR)
    bed = np.stack([_pink(n, rng), _pink(n, rng)])
    bed = _bp(bed, 350, 9000) * (0.6 + 0.6 * intensity)
    drops = np.zeros((2, n))
    k = int(dur * 40 * intensity)
    for _ in range(k):
        i = rng.integers(0, max(1, n - 800))
        L = rng.integers(120, 700)
        d = rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 5))
        pan = rng.random()
        g = rng.uniform(0.05, 0.4)
        drops[0, i:i + L] += d * g * (1 - pan)
        drops[1, i:i + L] += d * g * pan
    drops = _hp(drops, 1500)
    rum = _lp(np.stack([_pink(n, rng), _pink(n, rng)]), 120) * 1.5
    return bed + 0.5 * drops + rum


def drone(dur, rng, root='A1', dark=0.6):
    t = _t(dur)
    f0 = hz(root)
    out = np.zeros((2, len(t)))
    # root, octave, fifth and a soft minor third high up: unresolved, uneasy
    for mult, g in ((1, 1.0), (2, 0.5), (1.5, 0.35), (2.378, 0.12 * (1 - dark) + 0.06), (4, 0.08)):
        for c in range(2):
            det = 1 + rng.uniform(-0.003, 0.003)
            ph = rng.uniform(0, 2 * np.pi)
            lfo = 1 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.03, 0.09) * t + ph)
            # soft saw: a few harmonics
            s = sum(np.sin(2 * np.pi * f0 * mult * det * h * t + ph * h) / h for h in range(1, 6))
            out[c] += g * lfo * s
    cut = 300 + 900 * (1 - dark)
    out = _lp(out, cut, 2)
    swell = 0.75 + 0.25 * np.sin(2 * np.pi * t / max(8.0, dur / 3) - np.pi / 2)
    return _reverb(out * swell, 0.6, rng)


def piano(dur, rng, notes='A3@0', vel=0.6, room=0.5):
    t = _t(dur)
    out = np.zeros((2, len(t)))
    for tok in str(notes).split(';'):
        if not tok.strip():
            continue
        name, at = tok.split('@')
        f = hz(name.strip())
        i0 = int(float(at) * SR)
        if i0 >= len(t):
            continue
        tt = t[:len(t) - i0]
        B = 0.0004
        s = np.zeros_like(tt)
        for k in range(1, 12):
            fk = f * k * math.sqrt(1 + B * k * k)
            if fk > 9000:
                break
            dec = 2.5 / (1 + 0.35 * k) * (220 / f) ** 0.3
            s += (0.9 ** k) / k ** 0.6 * np.sin(2 * np.pi * fk * tt) * np.exp(-tt / dec)
        att = 1 - np.exp(-tt / 0.004)
        hammer = rng.standard_normal(min(len(tt), 600)) * np.exp(-np.arange(min(len(tt), 600)) / 80) * 0.05
        s = s * att
        s[:len(hammer)] += hammer
        s = _lp(s, 2500 + 2500 * float(vel)) * float(vel)
        pan = 0.5 + 0.3 * math.tanh((f - 260) / 300)
        out[0, i0:] += s * (1 - pan) * 1.4
        out[1, i0:] += s * pan * 1.4
    return _reverb(out, float(room), rng)


def ring(dur, rng, pattern='2,4'):
    on, off = [float(x) for x in str(pattern).split(',')]
    t = _t(dur)
    gate = ((t % (on + off)) < on).astype(float)
    gate = _lp(gate, 30, 1)
    clap = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 20 * t))      # 20 Hz striker
    strikes = np.diff(clap, prepend=0).clip(0, None)
    bell = np.zeros_like(t)
    for f, d, g in ((1180, 0.25, 1.0), (1790, 0.16, 0.6), (2650, 0.1, 0.35), (3900, 0.06, 0.2)):
        L = int(0.4 * SR)
        k = np.sin(2 * np.pi * f * np.arange(L) / SR) * np.exp(-np.arange(L) / (d * SR))
        from scipy.signal import fftconvolve
        bell += g * fftconvolve(strikes, k)[:len(t)]
    bell *= gate
    bell = _bp(bell, 600, 6000) * 0.5
    x = np.stack([bell, bell * 0.92])
    return _reverb(x, 0.25, rng)


def heartbeat(dur, rng, bpm=60, rise=0):
    t = _t(dur)
    out = np.zeros_like(t)
    tb = 0.0
    while tb < dur:
        for off, g in ((0.0, 1.0), (0.28, 0.65)):
            i = int((tb + off) * SR)
            L = min(int(0.25 * SR), len(t) - i)
            if L <= 0:
                continue
            tt = np.arange(L) / SR
            out[i:i + L] += g * np.sin(2 * np.pi * (48 + 30 * np.exp(-tt / 0.03)) * tt) * np.exp(-tt / 0.07)
        b = float(bpm) + float(rise) * tb / max(dur, 1e-6)
        tb += 60.0 / b
    out = np.tanh(2.2 * out)
    return np.stack([out, out])


def thunder(dur, rng):
    n = int(dur * SR)
    t = _t(dur)
    crack = _hp(rng.standard_normal(n), 800) * np.exp(-t / 0.18) * (t > 0.0)
    rum = _lp(_pink(n, rng) * 30, 160, 4) * np.exp(-t / (dur * 0.35)) * (1 - np.exp(-t / 0.25))
    rum = rum * (1 + 0.6 * _lp(rng.standard_normal(n), 3, 1) * 40)
    x = 0.5 * crack + 2.5 * rum
    return _reverb(np.stack([x, np.roll(x, 400)]), 0.7, rng)


def hit(dur, rng):
    t = _t(dur)
    f = 32 + 60 * np.exp(-t / 0.12)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.35 * dur + 0.2))
    tr = _bp(rng.standard_normal(len(t)), 200, 4000) * np.exp(-t / 0.03) * 0.4
    x = np.tanh(2.5 * (boom + tr))
    return _reverb(np.stack([x, x]), 0.5, rng)


def riser(dur, rng):
    t = _t(dur)
    k = t / dur
    n = rng.standard_normal(len(t))
    from scipy.signal import butter, sosfilt
    out = np.zeros_like(t)
    seg = int(0.05 * SR)
    for i in range(0, len(t), seg):  # sweeping band
        fc = 300 + 5000 * k[i] ** 2
        sos = butter(2, [fc * 0.8, min(SR / 2 - 100, fc * 1.25)], 'bp', fs=SR, output='sos')
        out[i:i + seg] = sosfilt(sos, n[i:i + seg])
    tone = np.sin(2 * np.pi * np.cumsum(110 * 2 ** (2 * k)) / SR) * 0.3
    x = (out * 0.8 + tone) * k ** 2
    return np.stack([x, np.roll(x, 300)])


def tick(dur, rng, bpm=60):
    t = _t(dur)
    out = np.zeros_like(t)
    step = 60.0 / float(bpm)
    for j, tb in enumerate(np.arange(0, dur, step)):
        i = int(tb * SR)
        L = min(int(0.03 * SR), len(t) - i)
        tt = np.arange(L) / SR
        out[i:i + L] += np.sin(2 * np.pi * (3200 if j % 2 else 2700) * tt) * np.exp(-tt / 0.004)
    return np.stack([out, out])


KINDS = dict(rain=rain, drone=drone, piano=piano, ring=ring, heartbeat=heartbeat, thunder=thunder, hit=hit,
             riser=riser, tick=tick)


@memo_file('global', '.wav')
def render(kind, params, out=None):
    """params: sorted [(key, value-string)]. Peak-normalised to -3 dBFS; level is set in the mix (gain_db)."""
    from scipy.io import wavfile
    p = {}
    for k, v in params:
        try:
            p[k] = float(v)
        except ValueError:
            p[k] = v
    dur = float(p.pop('dur', 4))
    rng = np.random.default_rng(int(p.pop('seed', 1)))
    if kind not in KINDS:
        raise ValueError(f'unknown synth kind {kind!r}: {", ".join(KINDS)}')
    x = KINDS[kind](dur, rng, **p)
    x = _fade(np.asarray(x, np.float64))
    x *= 10 ** (-3 / 20) / (np.abs(x).max() + 1e-9)
    wavfile.write(out, SR, (x.T * 32767).astype(np.int16))


def resolve(src):
    """'synth:rain?dur=60' -> cached wav path; anything else passes through."""
    if not (isinstance(src, str) and src.startswith('synth:')):
        return src
    body = src[6:]
    kind, _, q = body.partition('?')
    return render(kind, sorted(parse_qsl(q)))
