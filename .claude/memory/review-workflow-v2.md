---
name: review-workflow-v2
description: "How to review video fixes: vf.qa harness gates + blind Opus/Sonnet critics + owner-anchored derived score; old loop scored ~2/10"
metadata:
  node_type: memory
  type: feedback
  originSessionId: bcdc64f6-e189-45d4-91c7-0dd42090e64c
  modified: 2026-11-05T08:33:48.666Z
---

On 2026-11-03 the user called my reviewing "flawed" and asked for an adversarial multi-model audit, with a target of at least 7.5/10.

**Old loop:** Opus, Sonnet and Haiku scored it 2.6, 2.6 and 1.9. Its flaws:
- stills only;
- one critic, fed earlier scores, "NEW" labels and my account of the fix, so it stayed pinned at 7 even for an owner-rejected build;
- the Laplacian "detail" metric rewards over-sharpening;
- no gates and no rubric.

**New loop:** `videoforge/docs/review_workflow.md`. The parts:
- the `python -m vf.qa` harness (gates, motion shown as 8-frame strips, cut pairs, a blind critic packet);
- two blind critics (Opus and Sonnet) on a verbatim PROMPT.txt;
- the score is derived from their defects (10 − 2 per severity-2 class − 1 per severity-1 class), anchored to the owner's verdicts in `videoforge/qa/verdicts.csv`.

**Why:** the owner kept catching defects the review had passed: the mouth flicker, smears, a regression, the Q8 rejection.

**How to apply:**
- Never put history, earlier scores, the fix description or metrics in a critic prompt.
- A harness FAIL never goes to the owner as "fixed".
- Log every owner verdict.
- An owner-caught defect means a harness bug: add a gate for it before the next round.
- Relative (vs-base) scoring cannot block a defect the approved base also has. Genre/tone ("looks more horror than neo noir") is therefore judged on the new cut alone (qa-6 `ABSOLUTE_CLASSES`, tone blocks at severity 1). An approval of a build covers only the shots the owner was asked about.

**Audit on 2026-11-05 (held-out):**
- Results: 3 of 5 right, worse than "always BLOCK" (4 of 5). Only 1 of 3 SHIP calls was approved.
- Causes:
  - The intent line tells critics which cut is new.
  - The derived score reduces to "any sev-2 class".
  - Pasted/realism defects get only sev 1.
- Treat SHIP as weak evidence until this is fixed (videoforge/qa/audit/2026-11-05/AUDIT.md).

**Fixed the same day as qa-8:**
- The brief goes only to a separate read of the new cut alone ("delivered?", "looks real?"), never into the A/B prompt.
- "partly" gives CHECK (the owner's call, with the critics' reasons). Critics say "partly" about approved and rejected builds alike, so never turn it into SHIP or BLOCK.
- A review = 4 critic reads (Opus and Sonnet × A/B and brief). Launch with "Read <packet>/PROMPT_r.txt" or "BRIEF_r.txt", plus the model name.

**qa-9 (same day):** the jitter number grows with motion and face sharpness; a jitter rise counts as flicker only if the flicker index (share of back-and-forth change) also rose. Check motion before trusting any jitter-rise FAIL on a regenerated shot.

**qa-10 (same day, owner "let it vote"):** a director model (Qwen3.5-9B NF4, vf/gen/director.py) rates each focus shot one take at a time (6 yes/no logits + notes); "worse" (mean change < -0.02) turns CHECK into BLOCK. Side-by-side prompts were position-biased on both models tried: never ask a VLM to pick between two takes in one prompt. Past cases: 7 right, 2 wrong (q6c, endB), 2 your call.

Related: [[one-fix-at-a-time]], [[user-creative-workflow]], [[procedural-film-lessons]].

**Viewer-transcript test (2026-11-08, owner: "dialogue makes no sense, broken dialogue"):** script critics that read the
logline + puzzle notes rated a draft's clarity 6-8; the same critics given ONLY a viewer transcript (per shot: what is on
screen + what is heard) rated it clarity 3-4 and listed 9-10 lost moments (unnamed "her", an unexplained number, "from in here"
over an outside shot). Always run critics on the viewer transcript, ask them to RETELL the story + list confusion points.
