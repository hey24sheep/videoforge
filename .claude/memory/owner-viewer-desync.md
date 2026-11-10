---
name: owner-viewer-desync
description: owner's video viewer plays audio ~1 s ahead of picture; judge sync only in a native player/PC
metadata:
  type: project
---
2026-11-07: sync_test.mp4 (white box + beep each 1 s) showed the beep ~1 s BEFORE the box on the owner's viewer (they watch
files I send over Remote Control). The same files play in sync in Chromium. So the 3 lip-sync rejections may be partly the viewer.

**Why:** we lost several rounds blaming the model for a playback offset.

**How to apply:** before any lip-sync review, ask the owner to open the file in the phone's own player or on PC; resend
proof/sync_test.mp4 if unsure. Still keep the frame-level gate ([[talking-head-silence-rule]]).
