---
name: long-process-watchdog
description: "Recheck any process running 30+ min; diagnose and fix if stuck or slow, don't just wait"
metadata:
  node_type: memory
  type: feedback
  originSessionId: bcdc64f6-e189-45d4-91c7-0dd42090e64c
  modified: 2026-11-06T15:53:04.886Z
---

When a background process (render, build, model run) has run for 30 minutes or more, recheck it: are output files still growing, GPU/VRAM/CPU state, are there competing processes, and is the log progressing? If it is stuck or slow, diagnose and fix it, then restart. Don't wait silently.

**Why:** On 2026-11-02 a Klein+Wan build sat stuck for over an hour (Wan and early finishing overlapped on the 8 GB GPU, which spilled to RAM). The log was empty because of stdout buffering, so the user couldn't tell it was stuck and had to ask.

**How to apply:** Whenever a long job starts, also start a background check at 30 min (`sleep 1800; <status cmd>` with run_in_background), and repeat every 30 min until the job ends. Run builds with PYTHONUNBUFFERED=1. Judge progress by the newest output/cache file times, not only the log. Every wait loop I start must have a timeout and must also exit when the watched process ends, and I stop leftover checks as soon as the job finishes. (A loop waiting for a log line that never came ran for 70 min and the user had to ask, 2026-11-02.) Prefer a stall detector (report when no new DONE line for ~8 min) over a fixed 30-min check: the 2026-11-02 Phase 1 build stalled after 3 min and was only caught at 30.

**Start long videoforge jobs detached** with `tools/jobwatch.py run ... --resumable` (2026-11-06: a build started from my session shell was killed when the session process exited; 2.7 h lost before the owner asked "is anything done?"). The SessionStart hook runs `jobwatch.py check` (the owner's "restart checker"): act on KILLED/FAILED/STALLED lines at once.

**After EVERY pass** (build, keyframe run, Klein edit, render), unasked (user rule 2026-11-03: "after each pass, check for stray processes. do not want memory spillage and heating"): list python/powershell/ffmpeg processes, stop leftovers (idle gen worker `vf.gen.server`, memwatch.ps1 - it does NOT exit while the worker lives), and report GPU MiB, shared-memory peak from memwatch and GPU temp in one line. Related: [[film-pipeline-performance]], [[context-hygiene]].

**2026-11-07 update (owner had to say "stuck again" twice):** after every launch, run `tools/jobwatch.py watch --id <id>`
in the background and re-arm it after every event; push-notify the owner on problems and milestones with an ETA. A
watcher that only fires on crash/long silence missed a lock deadlock and a RAM-guard abort for minutes.

2026-11-10 (owner: "why is the wait task running for 3 hours"): a `until grep "jobwatch end"` loop never ends when I kill the job (taskkill writes no end line). When I stop a job, stop its waiters too (TaskStop), and prefer `jobwatch watch` (it reports a dead runner) over bare grep loops.

2026-11-10 (owner: "remove the watchers which are useless always"): ONE waiter per job, no extras. Don't stack a Monitor
on top of a waiter that already reports at the end. Don't use `jobwatch watch` on jobs that write no log lines (downloads):
it fires a false STALL. Stop every watcher the moment it has no job left to watch.
**Why:** redundant/useless watchers clutter the owner's task list. **How to apply:** before arming a watcher, check if a
running task already covers that job; after any event, stop watchers whose job is done.
