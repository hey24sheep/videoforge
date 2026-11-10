---
name: context-hygiene
description: User rule - keep STATE.md current, compact at 60%, verify after compaction, never invent context %
metadata:
  type: feedback
---
Keep a working-state file (videoforge/STATE.md) current and verify it after every compaction; never estimate context usage.

**Why:** 2026-11-02 the user saw output quality degrade over a very long session ("pixelated, smeared blobs, worse every iteration") and blamed context rot.

**How to apply:** update STATE.md after each meaningful change; user settings set auto-compact at 60% and a SessionStart(compact) hook re-prints STATE.md. After compaction, list what's missing. Context % is only visible to the user via /context. Related: [[videoforge-pipeline]].
