"""Episode linter: catches the authoring mistakes that make small video models produce artifacts, BEFORE any GPU
time is spent. Run by `python -m vf lint <episode>` and at the start of every build (errors stop the build).

Each rule comes from a real defect (first episode review, 2026-11-02):
  smoke from nowhere      motion asks for smoke/steam with no source in the keyframe
  dry face in rain        motion asks for water running on a face the keyframe shows dry
  receiver teleports      motion asks a hand to lift/hand over/pick up an object (5B models teleport props)
  bad walking             gait verbs: legs skate and blur at 480p
  wobbling dock/road      camera moves written into the motion prompt: the model moves the camera and structures
                          wobble; camera moves belong in `camera:` (made exactly by the compositor)
  frozen tail             the shot needs more clip time than the model makes (5 s)
  off-centre title        title-card text not at x 0.5 (0.47 is the captions' Shorts safe-zone centre)
  floating prop           (2026-11-09) a talk shot keeps the hands out of frame, so a hand-held prop named in its
                          keyframe ("lantern glow below his face") hangs in mid-air
  in-scene text layer     (2026-11-09) a `text` layer with in_scene: true reads as pasted on top: use `gen.ink`
"""
import re

GAIT = r"\b(walk(s|ing)?|strid(e|es|ing)|run(s|ning)?|jog(s|ging)?|step(s|ping)? (forward|toward|out|into)|march(es|ing)?)\b"
NOT_GAIT_SUBJECTS = {'rain', 'water', 'drop', 'droplet', 'drops', 'condensation', 'tear', 'tears', 'it', 'cord', 'light',
                     'shadow', 'shadows', 'river', 'stream', 'blood', 'ink', 'wax', 'clock', 'beam', 'line', 'that'}
HAND_ACTIONS = (r"\b(lift(s|ing)?|pick(s|ing)? up|hand(s|ing)? (it |him |her )?over|put(s|ting)? down|hang(s|ing)? up|"
                r"rais(es|ing) .{0,20}to (his|her) ear|open(s|ing)? the (door|drawer|case)|turn(s|ing)? around|"
                r"reach(es|ing)? (for|out)|grab(s|bing)?|throw(s|ing)?|take(s)? off|put(s|ting)? on)\b")
CAMERA_WORDS = (r"\b(push(es|ing)?[- ]in|pull(s|ing)?[- ](out|back)|dolly|pan(s|ning)?|tilt(s|ing)?|zoom(s|ing)?|"
                r"track(s|ing)? (shot|alongside|behind)|crane|orbit(s|ing)?|the camera (follows|moves|circles|"
                r"pushes|pulls|rises|drifts|tracks))\b")
SMOKE = r"\b(smoke|smoking|steam|vapou?r)\b"
SMOKE_SOURCE = r"\b(cigarette|cigar|pipe|ashtray|chimney|fire|candle|kettle|manhole|vent|exhaust|incense|steam)\b"
WET_FACE = r"\b(water|rain(water)?|drops?|droplets) (runs?|running|streams?|streaming|drips?|dripping|trickles?) (down|over|across) (his|her|the) (face|cheeks?|skin)\b"
WET_WORDS = r"\b(wet|soaked|drenched|dripping|rain-?(soaked|streaked|beaded)|beads? of (rain|water))\b"
HELD = (r"\b(lantern|oil lamp|torch|flashlight|cup|mug|bottle|cigarette|cigar|pipe|handset|phone|gun|pistol|badge|"
        r"tie clip|photo(graph)?|letter|keys?)\b")
MOUTH = r"\b(speak(s|ing)?|talk(s|ing)?|say(s|ing)?|whisper(s|ing)?|shout(s|ing)?|mouth(s|ing)?|jaw (tightens|moves|works))\b"


def _shot_lines(spec, s):
    t0, t1 = s['start'], s['start'] + s['dur']
    return [d for d in spec.get('dialogue', []) if t0 - 0.2 < float(d.get('at', 0)) < t1]


def lint(spec):
    """[(level, where, message)]; level in ('error', 'warn')."""
    out = []
    chars = dict(((spec.get('series') or {}).get('cast') or {}), **((spec.get('series') or {}).get('characters') or {}))
    for s in spec['shots']:
        sid, g = s['id'], s.get('gen') or {}
        kfr = str(g.get('keyframe', '')).lower()
        mot = str(g.get('motion', '')).lower()
        if g:
            for name in re.findall(r'\{(\w+)\}', kfr + ' ' + mot):
                if name not in chars:
                    out.append(('error', sid, f'{{{name}}} is not in series.yaml characters or cast'))
            # "cigarette smoke hanging in the air" names smoke, not a cigarette anyone holds
            kfr_src = re.sub(r'\b(cigarette|cigar|pipe) smoke\b', 'smoke', kfr)
            given_image = bool(g.get('image')) and not kfr  # a supplied first frame: its content can't be read here
            if given_image:  # the motion prompt describes it instead ("from the burning cigarette in the ashtray")
                kfr = kfr_src = mot
            if re.search(SMOKE, mot) and not re.search(SMOKE_SOURCE, kfr_src):
                out.append(('error', sid, 'motion asks for smoke/steam but the keyframe shows no source (cigarette, '
                                          'ashtray, vent...): it will appear from nowhere. Put the source in the keyframe or drop the smoke'))
            if re.search(WET_FACE, mot) and not re.search(WET_WORDS, kfr):
                out.append(('error', sid, 'motion has water running on a face the keyframe shows dry: describe the wet '
                                          'face/hat in the keyframe (the video model cannot add wetness)'))
            m = None
            for mm in re.finditer(GAIT, mot):  # "rain runs down the glass" is not a gait
                prev = re.findall(r'[a-z]+', mot[:mm.start()])[-2:] or ['']
                if prev[-1] not in NOT_GAIT_SUBJECTS and 'not' not in prev and "doesn't" not in prev:  # "does not walk"
                    m = mm
                    break
            if m:
                out.append(('warn', sid, f'gait ("{m.group(0)}"): legs skate/blur at 480p. Restage: legs out of frame, '
                                         'a static wide silhouette, or the move done by `camera: truck_*`'))
            m = re.search(HAND_ACTIONS, mot)
            if m:
                out.append(('warn', sid, f'hand/prop action ("{m.group(0)}"): props teleport and hands melt in small '
                                         'models. Prefer a held pose (grip tightens, fingers shift) and cut on the action'))
            m = re.search(CAMERA_WORDS, mot)
            if m:
                out.append(('warn', sid, f'camera move in the motion prompt ("{m.group(0)}"): structures wobble when '
                                         'the model moves the camera. Put the move in `camera:` and let the AI camera stay steady'))
            if _shot_lines(spec, s) and re.search(MOUTH, mot) and not (g.get('lip_sync') or g.get('lipsync') or g.get('talk')):
                out.append(('warn', sid, 'motion asks for mouth/jaw movement during a voice line with no lip-sync: '
                                         'reads as out of sync. Ask for "lips pressed together"'))
            if g.get('talk'):
                for k in ('erase', 'lock', 'stamp', 'ink', 'retex', 'face_tone'):
                    if g.get(k):
                        out.append(('error', sid, f'`{k}` is a K6-clip step and is NOT applied to talk shots: fix the keyframe '
                                                  'image itself (2026-11-09: an erased figure came back in a talk take)'))
            if g.get('talk'):   # InfiniteTalk shots: hands out of frame -> anything held floats
                for mm in re.finditer(HELD, kfr):
                    before = re.findall(r"[a-z']+", kfr[max(0, mm.start() - 30):mm.start()])[-3:]
                    if 'no' in before or 'without' in before:
                        continue
                    if re.search(r'\b(hold(s|ing)?|grip(s|ping)?|clutch(es|ing)?|hanging|hangs|on (the|a) (wall|hook|crate|boards|table|desk|sill|shelf)|beside (him|her))\b', kfr):
                        continue   # a hand or a surface holds it in the keyframe
                    lvl = 'warn'
                    out.append((lvl, sid, f'talk shot names a hand-held prop ("{mm.group(0)}") but its hands are out of '
                                          'frame: the prop floats (owner 2026-11-09). Drop it, or write "no ' + mm.group(0) + ' in frame"'))
                    break
            need = s['dur'] * float(g.get('speed', 1.0))
            if need > 5.0 + 1e-6:
                out.append(('warn', sid, f'needs {need:.2f}s of clip but clips are 5 s: it will be slowed to '
                                         f'speed {5.0 / s["dur"]:.2f} automatically (or shorten the shot)'))
        if s['dur'] < 0.8:
            out.append(('warn', sid, f'{s["dur"]}s is very short for a cut on Shorts'))
        for Ly in s.get('layers', []):
            if Ly.get('type') == 'text' and Ly.get('in_scene'):
                out.append(('warn', sid, f'in-scene text layer "{str(Ly.get("text"))[:30]}" floats over the frame: write it '
                                         'into the prop with gen.ink (vf/inktext.py)'))
        if not g:  # title / card shots: text centred on the frame
            for L in s.get('layers', []):
                if L.get('type') == 'text' and abs(float(L.get('x', 0.5)) - 0.5) > 1e-3 and not L.get('anchor_left'):
                    out.append(('warn', sid, f'title text "{L.get("text")}" at x={L.get("x")} is off-centre (use 0.5)'))
    # dialogue: lines must not overlap or start after the end
    ds = sorted(spec.get('dialogue', []), key=lambda d: float(d.get('at', 0)))
    for d in ds:
        if float(d.get('at', 0)) >= spec['duration']:
            out.append(('error', 'dialogue', f'line at {d.get("at")}s starts after the episode ends: "{d["text"]}"'))
    return out


def report(spec, log=print):
    items = lint(spec)
    for lvl, where, msg in items:
        log(f'  [{lvl}] {where}: {msg}')
    return [i for i in items if i[0] == 'error']
