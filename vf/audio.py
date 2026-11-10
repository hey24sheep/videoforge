"""Episode mix: dialogue + music (auto-ducked under speech) + sfx, mastered to -14 LUFS / -1 dBTP (YouTube,
Instagram). Cached: the mix is rebuilt only when its inputs change."""
import math, os, subprocess
import numpy as np
from .memo import memo_file
from .render import ffmpeg

SR = 48000


def decode(path):
    """Any audio/video file -> float32 stereo (2, n) at 48 kHz."""
    p = subprocess.run([ffmpeg(), '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2).T.copy()


def db(x):
    return 10 ** (x / 20)


def lufs(x):
    """BS.1770 integrated loudness of (2, n)."""
    from scipy.signal import lfilter
    k = lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, 0.73248077421585], x, axis=-1)
    k = lfilter([1, -2, 1], [1, -1.99004745483398, 0.99007225036621], k, axis=-1)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    p = (k ** 2).sum(0)
    c = np.concatenate([[0], np.cumsum(p)])
    st = np.arange(0, max(1, len(p) - blk), hop)
    L = -0.691 + 10 * np.log10((c[np.minimum(st + blk, len(p))] - c[st]) / blk + 1e-12)
    g = L[L > -70]
    if not len(g):
        return -70.0
    rel = 10 * np.log10(np.mean(10 ** (g / 10))) - 10
    return float(10 * np.log10(np.mean(10 ** (g[g > rel] / 10))))


def true_peak_env(x, os_=4):
    from scipy.signal import resample_poly
    u = np.abs(resample_poly(x, os_, 1, axis=-1)).max(0)
    u = u[:len(u) // os_ * os_].reshape(-1, os_).max(1)
    out = np.zeros(x.shape[1])
    out[:len(u)] = u
    return np.maximum(out, np.abs(x).max(0))


def limit(x, ceil_db=-1.0, look=0.005, rel=0.08):
    """Look-ahead true-peak limiter (stereo linked)."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    ceil = db(ceil_db)
    for _ in range(3):
        pk = true_peak_env(x)
        g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
        if g.min() >= 0.999:
            break
        w = int(look * SR)
        g = minimum_filter1d(g, 2 * w + 1)
        b = int(0.001 * SR)
        nb = len(g) // b + 1
        gb = np.ones(nb)
        gb[:len(g) // b] = g[:len(g) // b * b].reshape(-1, b).min(1)
        kk = 1 - math.exp(-1 / (rel * 1000))
        y, prev = np.empty(nb), 1.0
        for i in range(nb):
            prev = min(gb[i], prev + (1 - prev) * kk)
            y[i] = prev
        g = np.minimum(g, np.repeat(y, b)[:len(g)])
        g = uniform_filter1d(g, w)
        x = x * g
    return x


def _place(bus, x, t, gain=1.0):
    i = int(round(t * SR))
    if i >= bus.shape[1]:
        return
    if i < 0:
        x, i = x[:, -i:], 0
    n = min(x.shape[1], bus.shape[1] - i)
    bus[:, i:i + n] += x[:, :n] * gain


def voice_fx(x, fx):
    """Per-line colour. 'phone': telephone band (300-3400 Hz) + light saturation; 'radio': narrower and grittier;
    'room': short early reflections (a voice in a small space)."""
    if not fx:
        return x
    from scipy.signal import butter, sosfilt
    if fx in ('phone', 'radio'):
        lo, hi, drive = (300, 3400, 2.0) if fx == 'phone' else (500, 2600, 3.5)
        m = x.mean(0)
        m = sosfilt(butter(4, [lo, hi], 'bp', fs=SR, output='sos'), m)
        pk = np.abs(m).max() + 1e-9
        m = np.tanh(drive * m / pk) / np.tanh(drive) * pk
        return np.stack([m, m])
    if fx == 'room':
        y = x.copy()
        for ms, g in ((11, 0.32), (23, 0.22), (37, 0.15), (53, 0.09)):
            k = int(ms / 1000 * SR)
            y[:, k:] += g * x[:, :-k]
        return y / 1.4
    raise ValueError(f'unknown voice fx {fx!r} (phone, radio, room)')


@memo_file('project', '.wav')
def mix(duration, dialogue, music, sfx, target_lufs=-14.0, out=None):
    """dialogue: [(path, start_s, gain_db[, fx])]; music: dict(src, gain_db, duck_db, fade_in, fade_out, start) or None;
    sfx: [(path, at, gain_db)]."""
    from scipy.io import wavfile
    N = int(math.ceil(duration * SR))
    dlg, mus, fx = np.zeros((2, N)), np.zeros((2, N)), np.zeros((2, N))
    talk = np.zeros(N)
    for item in dialogue:
        path, t0, g = item[:3]
        x = voice_fx(decode(path), item[3] if len(item) > 3 else None)
        _place(dlg, x, t0, db(g))
        i0, i1 = int(max(0, t0 - 0.12) * SR), int(min(duration, t0 + x.shape[1] / SR + 0.2) * SR)
        talk[i0:i1] = 1
    if music and music.get('src'):
        x = decode(music['src'])
        start = float(music.get('start', 0))
        x = x[:, int(start * SR):]
        if x.shape[1] < N and music.get('loop', True):
            x = np.tile(x, (1, N // max(1, x.shape[1]) + 1))
        x = x[:, :N]
        n = x.shape[1]
        fi, fo = int(float(music.get('fade_in', 0.5)) * SR), int(float(music.get('fade_out', 1.5)) * SR)
        env = np.ones(n)
        if fi:
            env[:fi] = np.linspace(0, 1, min(fi, n))[:n]
        if fo:
            env[-fo:] = np.minimum(env[-fo:], np.linspace(1, 0, min(fo, n)))
        _place(mus, x * env, 0, db(float(music.get('gain_db', -8))))
    for path, t0, g in sfx:
        _place(fx, decode(path), t0, db(g))
    # duck the music ~8 dB under speech with 120 ms attack/250 ms release
    if talk.any() and music:
        k = np.ones(int(0.15 * SR)) / int(0.15 * SR)
        tk = np.convolve(talk, k, 'same')
        mus *= (1 - (1 - db(float(music.get('duck_db', -8)))) * np.clip(tk, 0, 1))
    m = dlg + mus + fx
    from scipy.signal import butter, sosfilt
    m = sosfilt(butter(4, 30, 'hp', fs=SR, output='sos'), m, axis=-1)  # rumble out
    L = lufs(m)
    if L > -69:
        for _ in range(3):  # gain to target, limit, make up what the limiter took
            m = m * db(target_lufs - lufs(m))
            m = limit(m, -1.6)  # AAC encoding adds ~0.5 dB of overshoot: -1.6 here lands under -1 dBTP in the mp4
            if abs(lufs(m) - target_lufs) < 0.1:
                break
    fade = np.ones(N)
    fade[-int(0.05 * SR):] = np.linspace(1, 0, int(0.05 * SR))
    m *= fade
    wavfile.write(out, SR, (np.clip(m, -1, 1).T * 32767).astype(np.int16))
