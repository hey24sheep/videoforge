---
name: cinematic-camera-rule
description: "User wants every shot to have real cinematic camera movement (pan, push, zoom, shift, motion blur), never static held stills"
metadata:
  node_type: memory
  type: feedback
  originSessionId: bcdc64f6-e189-45d4-91c7-0dd42090e64c
  modified: 2026-11-02T02:03:06.049Z
---

The user said (2026-11-02, after a first episode) that there should be no hold stills. Every shot should feel like a film or series, with camera movement such as pans, push-ins and zooms, lens shifts, and cinematic motion blur.

**Why:** static or held shots, and the frame-hold judder, read as cheap rather than cinematic.
**How to apply:**
- Never fall back to a frozen frame or a still image as a fix.
- Fake camera moves (Ken Burns style) are only OK when they are smooth, eased and motion-blurred.
- Preferred design:
  - Generate the AI clip with a locked camera, because i2v camera motion makes structures wobble.
  - Add the camera move in the compositor: eased pan, zoom or drift, optional subtle handheld sway, and motion blur scaled to speed.
  - Blend or interpolate frames, never use nearest-frame holds.

**Failure, 2026-11-02 (a first episode, v3):** I turned 10 of 15 shots into "2.5D stills" (a keyframe moved by the virtual camera, with overlay rain) to dodge AI artifacts. The user called it "much worse", and the reviewer scored it 4.0/10, down from 4.5.
- The camera pushes were 5–12% over a shot, which measures as nearly invisible. Real pushes are 10–20%+ with variety, and the camera has to motivate them.
- Rain over frozen subjects looks like a photo with a filter.
- Depth parallax creates its own defects: a doubled hat brim, a cutout figure sliding, a see-through shoulder.

Rule:
- The user said "camera move is considered still". An image with only a camera move (Ken Burns or 2.5D parallax) counts as a still, and stills are banned.
- Every shot must be a real AI motion clip with subject motion inside it: breathing, hair or cloth in wind, flicker, drips, smoke.
  - Pick motions the 5B model can do. Make the motion prompts small; never freeze the shot.
  - Compositor camera moves only go on top of moving footage.
- Title cards are the only exception.

Related: [[videoforge-pipeline]], [[procedural-film-lessons]].
