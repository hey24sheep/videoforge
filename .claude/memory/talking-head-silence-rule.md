---
name: talking-head-silence-rule
description: owner rejects any open/moving mouth without voice; review per frame, gate with tools/silent_mouth.py
metadata:
  type: feedback
---
Owner rejected 3 InfiniteTalk proofs (2026-11-07) for "mouth moves but no voice". Whole-clip metrics (SyncNet offset,
mouth_sync corr/lag) passed them. The cause was a few frames at line heads/tails and in pauses inside a line.

**Why:** a viewer sees even 4-6 frames of a gaping mouth in silence; InfiniteTalk never closes the mouth in short pauses and
leads the voice by 2-3 frames.

**How to apply:** every talking shot: no pauses inside a line's audio, audio shifted 80 ms earlier, cut with
`tools/silent_mouth.py --cut`, then gate the final with `silent_mouth.py` (open-silent run <= 1 frame) AND look at a per-frame
mouth strip myself before sending. 
