---
name: no-fable-model
description: User does not want the Fable model used for subagents/reviewers
metadata:
  node_type: memory
  type: feedback
  originSessionId: 639c3a2f-6370-4991-9b6f-7fa4f2ad8728
  modified: 2026-10-31T18:55:34.711Z
---

Do not use the Fable model for subagents or multi-model scoring passes; use Opus, Sonnet, Haiku instead.

**Why:** user said "do not use fable" mid-task on 2026-11-01 (resume review).
**How to apply:** when picking models for parallel reviewers/agents, never pass model "fable". Related: [[user-creative-workflow]].
