You review a finished AI-generated video episode frame by frame, strictly.

Folder: {DIR}
Each shot is one image <shot_id>.jpg: the shot sampled at 12 frames per second, tiles numbered from 1 (made by
`vf framecheck grids`). intent.txt says what each shot should show. Open only intent.txt and your shots' images.

Your shots: {SHOTS}

For each shot, look at every frame in order and report:
- every defect with frame numbers: melting/morphing faces, hands or bodies; extra/missing fingers or limbs; identity
  change (does the person still look like the same character?); objects appearing, vanishing or duplicating; flicker or
  brightness pumping; jumps; gibberish or unwanted text; impossible physics; a face visible where none should be; on
  talk shots, mouth open while it should be silent or a frozen mouth.
- does the shot show what intent.txt says?
- real motion or effectively a still?
- score 1-10 (10 = would pass in a real film), worst severity (high / medium / low).

Ignore the small corner watermark and burned-in captions.

Write the full report to {DIR}/review_{NAME}.md. Then write {DIR}/labels_{NAME}.json: a JSON list, one object per shot:
{"shot": "<id>", "score": <int>, "n_frames": <tile count>,
 "defects": [{"type": "<type>", "severity": "low|medium|high", "frames": [[a, b], ...], "note": "<= 15 words"}]}
Types: identity_drift, face_morph, hair_change, body_morph, extra_person, object_morph, object_vanish, object_jump,
text_garbled, text_cut, wrong_light, intent_breach, dark_frames, hard_cut, flicker, no_motion, mouth_silent, glow_eyes,
other. Frames = tile ranges where the defect is visible ("all frames" = [[1, n_frames]]); list only defects, never what
is fine. Check the file parses as JSON.

Reply with a table: shot | score | worst severity | worst defect (frames).
