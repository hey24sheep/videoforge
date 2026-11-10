---
name: never-stop-other-sessions
description: Owner rule — never stop, kill or interrupt other Claude sessions; only my own tasks/processes
metadata:
  type: feedback
---

Never stop other Claude sessions (no stop_session, no killing their processes, no TaskStop on tasks I didn't start).
Only stop my own background tasks and jobs I launched.

**Why:** owner, 2026-11-10: "do not stop other sessions ever".
**How to apply:** if another session's job blocks the GPU or holds a lock, wait or tell the owner; never stop it myself.
Cleanup of watchers ([[long-process-watchdog]]) covers only my own watchers.
