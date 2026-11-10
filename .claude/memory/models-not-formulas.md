---
name: models-not-formulas
description: "Owner rule: every check/scoring/grade step uses an open model, not a hand-made formula; must fit 8 GB with no VRAM spill"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 6c6e0b2f-a2cd-4326-bbf8-c5f262050ea1
  modified: 2026-11-05T15:32:26.171Z
---

On 2026-11-05 the owner rejected my hand-made "flicker index" fix with "It's wrong fix use some other models". The same day they asked for a model behind everything I do:
- visual defects: flicker, motion/shake, smear, artifacts, colour grading;
- overall scoring: a "director model" that says whether a cut is cinematic and good enough;
- script and dialogue text;
- dialogue delivery: tone, pitch, emotion, naturalness;
- lip-sync.

**Why:** hand-made formulas (2nd temporal difference, Laplacian sharpness, Lab statistics) kept misfiring, e.g. the jitter gate failed an owner-approved shot because it measured motion, not flicker.

**How to apply:**
- Before writing any new check or metric, search for a current open model for it ([[latest-fastest-tools]]): commercial licence, no login, fits the RTX 4060 8 GB.
- Run each model under an allocator cap (vf/gen/guard.py process_cap), one heavy model at a time, so nothing spills into shared RAM.
- Keep a formula only for exact facts (frame counts, LUFS, pixel identity), and say so.
- Calibrate each model against the owner verdicts in videoforge/qa/cases.yaml ([[review-workflow-v2]]).
- First instance: the RAFT flicker gate (vf/gen/flowqc.py, harness qa-9).

**State 2026-11-06:**
- **Wired in:** RAFT flicker (qa-9), Qwen3.5-9B director vote (qa-10), VisualQuality-R1 face-quality gate (qa-11).
- **Built, not wired:**
  - smear yes/no (unstable);
  - delivery MOS/ECAPA (advisory; no owner labels to calibrate against).
- **Lessons:**
  - VLMs are position-biased in side-by-side prompts: use one take at a time.
  - Read scores from token probabilities, not from the printed answer.
  - Big-vocabulary models (Qwen3.5, Qwen2.5-VL) fit 8 GB only with NF4 quantised on the CPU plus embedding and lm_head kept in RAM (vf/gen/director.py _cpu_ends).
  - pip into the tts venv must pin torch: an unpinned install broke Chatterbox.
