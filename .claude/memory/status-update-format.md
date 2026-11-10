---
name: status-update-format
description: "Every progress report needs a time estimate, the measured scores, and any failure/defect flagged with a score or severity"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 6c6e0b2f-a2cd-4326-bbf8-c5f262050ea1
  modified: 2026-11-06T06:45:13.896Z
---

Owner (2026-11-06): "give me a time estimate too, always show the scores too and if something is breaking or failing or you see defects then flag it with a score."

**Why:** owner cannot see the background jobs and wants the state at a glance ([[long-process-watchdog]], [[caveman-reply-style]]).
**How to apply:** each status update = (1) clock time now + estimated finish per step, marked UNVERIFIED when guessed; (2) measured scores (cast identity, QC, director, critics); (3) flags: FAIL/WARN with severity, and a measured number when one exists. Never invent a number for a by-eye defect: say "by eye" and give severity only.
