# Blind critic: video takes
You are looking at frame sheets from {n} takes, labelled {labels}. Each sheet shows frames sampled in order. Context
(what the shot or cut should show, the intended look) is below. Open every image path listed under each take.

For EACH take give an integer 1-10 per criterion (10 = excellent) and one sentence of evidence naming a frame.
1. realism: physics, anatomy, light and materials look real (or true to the intended style)
2. intent: it shows what the context asks for, nothing missing, nothing extra (no stray people, text, objects)
3. composition: framing, depth and focus read like a deliberate film shot
4. continuity: no popping, melting, flicker, identity drift or objects changing between frames
5. look: grade and lighting match the intended look
6. impact: it would hold a stranger's attention for its length

Answer last with one ```json block: {"scores": {"<label>": {"<criterion>": <int>, ...}, ...}, "pick": "<label>"}

CONTEXT:
{context}

TAKES:
{entries}
