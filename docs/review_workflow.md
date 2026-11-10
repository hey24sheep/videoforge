# Review workflow v5 (harness qa-11, 2026-11-06)

## qa-11 change: face image quality gate
- `vf/gen/iqa.py`: VisualQuality-R1-7B (MIT, Qwen2.5-VL-7B base; NF4 on CPU -> GPU, embedding/lm_head in RAM; 4.0 GB peak, no spill). Answer-only mode; score = expected value over the digit tokens (continuous 1-5). Face crop (YuNet 1.6x box) on 8 frames, median per take; cached in `cache/qa/quality`.
- **Gate:** face score new − base < −0.3 = FAIL (`face_quality_drop_base`); model failure = FAIL (`face_quality_unmeasured`). Shots without a face: no check.
- Evidence (`qa/calibration.md` 2026-11-06): rejected q8 −0.39, p43 −0.47, whereAB −0.67, whereAB3 −0.70, whereA −0.84; approved ≥ −0.18. Whole-frame score not used (false drop on an intended fade to black).
- `vf/gen/smear.py` (Qwen3.5-9B yes/no on face/body crops) exists but is NOT wired: unstable across frame samples.
- `vf/gen/delivery.py` (tts venv, CPU): Distill-MOS naturalness + ECAPA same-voice per dialogue line. Advisory (no owner labels yet).


## qa-10 change: a director model votes (owner: "let it vote")
- **What:** `vf/gen/director.py` runs Qwen3.5-9B (Apache-2.0; 4-bit NF4 quantised on the CPU, layers and vision on the GPU, embedding and lm_head in system RAM; peak 4.2 GB under the 7.2 GB cap). It rates every focus shot's new take and base take **one take at a time** and writes director notes on the new take. Takes are cached in `cache/qa/director`.
- **Questions** (Yes/No probabilities, neutral, no history): looks pasted/composited/melted/AI-generated; framed and lit like a professional neo-noir film; faces/bodies/clothes/objects keep their look; shows what the shot should show; looks more like horror than neo-noir; a director would accept it as it is.
- **Verdict per shot:** mean change (new − base) of the 6 scores < −0.02 = worse, > +0.02 = better, else same.
- **Vote:** only on a would-be CHECK (your call). If any focus shot is "worse" the prediction becomes BLOCK, naming the director. It never changes SHIP or BLOCK. If the director did not run on a would-be CHECK, the prediction is INCOMPLETE.
- **Report line 5b** lists each shot's verdict, the per-question changes and the notes.
- **Rejected:** side-by-side (two takes in one prompt) on both UnifiedReward-4B and Qwen3.5-9B: position bias. Evidence: `qa/calibration.md` (2026-11-05, director tries 1-2).
- **Cost:** ~40 s model load + ~15 s per take per review.


## qa-9 change: face flicker judged by an optical-flow model
- **Problem:** the old lip/face jitter number (2nd temporal difference) grows with motion speed and face detail. The owner-approved pier3 failed it: its regenerated pier_wide is a sharp profile moving about 2x more than the soft base (lip jitter x2.23, face x1.91).
- **Owner, 2026-11-05:** a hand-made "flicker index" fix was rejected ("wrong fix, use some other models").
- **Now:** `vf/gen/flowqc.py` runs torchvision RAFT-large (BSD-3, 0.7 GB VRAM, ~10 s per shot) on 1.6x face crops: flow t->t-1, warp t-1 onto t, the L difference left over is what motion can't explain. Per frame the 99th percentile over the face and over the mouth zone; per shot the 99th percentile over frames (`flow_face`, `flow_mouth`).
- **Gate:** `flow_mouth` and `flow_face` vs base and anchor <= x1.3 (FAIL), the same 1.3 as before, not re-tuned. A failed RAFT run FAILs the shot (`flow_unmeasured`). The old lip/face jitter numbers are INFO; the whole-frame jitter rise and the absolute lip-sync gate (lip jitter <= 22) are unchanged.
- **Evidence:** in `qa/calibration.md` (2026-11-05, qa-9).

## qa-8 changes (audit 2026-11-05, `qa/audit/2026-11-05/AUDIT.md`)
**Why:** On held-out cases (prediction logged before the owner's verdict), the qa-7 system got 3 of 5 right; "always BLOCK" would have got 4 of 5. Two causes:
- The A/B prompt carried the owner's brief, which told critics which cut was new and to "judge execution, not existence".
- The score never decided anything.

The sections further down still describe v4. **Where they differ, this section wins.**

**1. Two reads per critic.**
- **A/B read** (`PROMPT_r.txt`): both cuts, blind, with NO brief and no `requested` field. Critics are told that differences between the cuts can be on purpose.
- **Brief read** (`BRIEF_r.txt`, a separate packet that holds only the new cut, under neutral names):
  - for each numbered request: delivered? yes / partly / no;
  - for each shot: would a phone viewer believe it is real footage? yes / mostly / no.
- The brief read exists only when `--intent` is given.

**2. One prompt and one empty answer folder per read.**
- Read r uses `PROMPT_r.txt` and `BRIEF_r.txt`.
- Opus = read 1, Sonnet = read 2, the third read (disputes only) = read 3.
- The launch instruction stays "Read <packet>/PROMPT_r.txt and follow it exactly", plus the model name for the answer file. Nothing else.
- Score with: `python -m vf.qa score <out> <ab answers...> --brief <brief answers...>`.

**3. Score.**
- Score = 10 − 2.5 per blocking class − 1.0 per realism class (smear, identity, compositing, tone) at sev 1 − 0.5 per other class at sev 1.
- **BLOCK** if any of these:
  - harness FAIL;
  - a blocking class;
  - score < 8.0;
  - a request delivered "no";
  - a shot judged "no" for realism.
- **CHECK (your call)** when a request is only "partly" delivered. In a blind brief read of 8 past cases, critics said "partly" about approved builds (q6c, q11) as often as about rejected ones (whereA, whereAB), so the system says it cannot tell and quotes their reasons.
- **SHIP** only when nothing above applies and there is no open dispute.
- A shot judged realism "mostly" is listed as an owner check.
- A packet with a brief but no brief answers gives INCOMPLETE, never SHIP.
- The 1.0 / 8.0 weights are a judgement, not fitted, and are checked only in-sample.

**4. Merging.**
- Max across critics.
- A class whose readers fall on both sides of its blocking line is DISPUTED: with 2 reads it needs a third read; with 3 reads it takes the median of **that class only**.
- The same rule applies to brief requests ("yes" against anything else) and realism ("no" against anything else).

**5. Harness changes.**
- `identity_min` = the minimum of a rolling median of 3 consecutive samples, so one noisy SFace sample no longer fails a build. The raw minimum is kept as `identity_min_raw`.
- The report lists every KNOWN gate with its shot and value. Identity below the owner's 0.60 rule is called out, never just counted.
- The cache key now covers cast refs, `cast.py`/`identity.py` and the lip-synced set.

**6. Integrity.**
- Held-out scoring refuses a `qa.json` from another QA_VERSION.
- Each video gets one held-out prediction; only a NEEDS THIRD READ or INCOMPLETE line may be followed by another.
- Every prediction logs the version, the scoring-code hash and both prompt hashes.
- `verify-log` prints the head hash, which goes into `qa/verdicts.log` with each owner verdict.

**7. Replay.**
- Compares the system prediction with the owner's verdict (`verdict:` in `qa/cases.yaml`), not with `expect_system`.
- Cases stored without their A/B key show NO KEY.

**Not changed yet:** the VLM judge's fail-open path (Q10), clipqc's sustained flicker (Q11), and multi-character identity (Q12).

---

# Review workflow v4 (harness qa-6, 2026-11-03)

## Audit history

All audits used Opus 5.5, Sonnet 5.5 and Haiku 5.5. The Fable model is never used.

| Version | Opus | Sonnet | Haiku | Main problems found |
|---|---|---|---|---|
| v1, the old loop | 2.6 | 2.6 | 1.9 | stills only; one critic anchored on my own story; no gates |
| v2 | 4.7 | 4.2 | 4.5 | metrics pointing the wrong way; KNOWN collision; I settled my own disputes |
| v3 | 5.7 | 4.9 | 4.3 | doc claimed code that was manual; mean-only diffs; compounding tolerances; leaky file names |

This document describes v4 (harness frozen as qa-4). Each step is marked **[code]** (done by `vf/qa.py`) or **[manual]** (done by me, Claude). Manual steps: 0 (confirm the item), 3 (launching the critic subagents, and exactly one third read on a dispute) and 7 (adding cases after a verdict, the repeat read every 5 rounds). Everything else is code.

## Principle

- Machines gate.
- Blind critics look.
- The owner is ground truth.

I (Claude) build the fixes. I never lower a gate or a critic finding, and I never hand-merge critic answers.

## The round, step by step

### 0. Confirm the item [manual]
- Restate the owner's request in their words.
- If the item was approved earlier, ask whether the change is still wanted.

### 1. Harness [code]
Command: `python -m vf.qa <episode> <video> --base <last approved> --anchor <first approved of these shots> --focus <shots meant to change> --intent "<shot: aspect; ...>"`

**Preflight (FAIL):**
- frame size, fps and duration;
- an audio stream is present;
- every mask's aspect ratio matches the frame.

**Decode:** one frame-exact pass at native resolution, with the face found on every frame. SFace identity is sampled on every 2nd frame. Tracked face and lip crops are taken for the focus shots and their cut neighbours.

**Absolute gates (FAIL):**

| Gate | Threshold | Calibration evidence |
|---|---|---|
| identity median | ≥ 0.60 | owner rule |
| identity min | ≥ 0.50 | — |
| identity not measurable on a cast shot | FAIL | fails closed |
| lip-synced shots: lip jitter L | ≤ 22 | owner-caught p43 31.2 vs approved q6c 16.3 |
| lip-synced shots: lip jitter a* | ≤ 10 | 14.0 vs 7.1 |
| same-character cut: face ΔE | ≤ 15 (WARN above 10) | — |
| jump cut | FAIL | — |
| loudness | −14 ± 1.5 LUFS | — |
| true peak | ≤ −1 dBTP | — |
| clipped samples | 0 | — |
| silent dropouts | none over 0.3 s | — |

**Relative gates (FAIL)**, checked twice: against the **base** (the last approved build) and against the **anchor** (the first approved build of these shots). Because of the anchor, tolerances can't compound across rounds; "worse every iteration" was the v8 failure.

| Gate | Threshold | Calibration evidence |
|---|---|---|
| identity drop | ≤ 0.05 | rejected q8 −0.068 |
| face flicker rise (RAFT motion-compensated residual, mouth and face; qa-9) and whole-frame jitter rise | ≤ ×1.3 | rejected p43 mouth ×1.43; approved builds ≤ ×1.05 |
| fail closed | a metric the reference could measure but this build cannot = FAIL | — |
| untouched shots (base) | worst 10×10 block diff over **every** frame ≤ 1.0, and equal frame count | renders are deterministic: 0.0 on all pinned cases |
| focus shot unchanged | FAIL ("the intended fix is not in this build") | — |

**Other severities:**
- **NEAR (shown to the owner, not blocking):** any passing gate within 10 % of its threshold.
- **KNOWN (listed, not blocking):** an absolute failure the base already had, on the same shot and full metric name, and not worse than in the base. Relative gates are never KNOWN.
- **WARN:** native face sharpness drop of more than 25 % vs base; per-shot audio level change vs base; cut luma jump.
  - Sharpness stays a WARN because, on owner-judged builds, the sharpness, over-sharpening and smear metrics did *not* rank the rejected Q8 below the approved builds. Blur, smear and "pasted" therefore belong to the critics.

**Outputs:** `qa.json` (the verdict, every metric, every gate, code hash, prompt hash, wall time) and the critic packet.

### 2. Blind critic packet [code]
- **Location:** `cache/qa_packets/<random id>/`, a different tree from the metrics. The A/B key stays with the metrics.
- **Both cuts** of the focus shots, named `A_`/`B_` at random. The prompt says only "two cuts of the same shots".
- **Aligned windows:** up to 6 timecoded 8-frame windows per face shot, the union of both builds' worst-jitter, median and random windows. Both cuts are shown at the same timecodes.
- **Face-less shots:** 6-frame windows at the frames that changed most against the base.
- **Also:** native-resolution face crops and start/middle/end frames.
- **Cut images** are named by position (`cut_into_<focus>`, `cut_out_of_<focus>`) and stamped "before cut" and "after cut", never with neighbour ids, which leaked reorders.
- **Prompt:** generated from the single `CRITIC_PROMPT` constant, whose hash is stored in `qa.json` and in the key. It contains the style note and the aspect-only intent.
- **Never in the packet:** history, scores, my account of the fix, owner quotes, or diff maps.
- **Critic output:** JSON written to `answer_<model>.json`, with each defect's cut, file, position, timecode, class, severity, `worse_here` and `requested`.

### 3. Critics [manual launch, code everything else]
- **[manual]** I launch Opus 5.5 and Sonnet 5.5 as subagents. The whole instruction is: "Read <packet>/PROMPT.txt and follow it exactly."
- **[code]** `python -m vf.qa score <qa out dir> <answer_opus.json> <answer_sonnet.json>`:
  - **Merge** per class over the new cut's `worse_here = yes` rows, taking each critic's worst severity.
  - **Disputes:** with 2 critics, if one says 2 and the other less than 2, the class is undecided.
    - The verdict is tested with the disputed class at the lower value. If even that blocks, the result is BLOCK; otherwise "NEEDS THIRD READ".
    - Pre-registered rule: exactly ONE fresh Opus read per dispute, launched immediately and always, and its result is used whatever it says (median of 3). No further reads.
    - A disputed class that is already routed to the owner (TASTE or UNCERTAIN) needs no third read.
    - Worked example: Q11, cut [2, 0], then a third read gave 1, so the median is 1 → SHIP, matching the owner.
- I never edit, transcribe or adjudicate the answers.

### 4. Score and routing [code: `score_critics`]
- **Relative score:** 10 − 2.5 × (blocking classes) − 0.5 × (other classes at severity 1), using only the new cut's `worse_here = yes` defects.
- **Absolute classes (qa-6): `tone`** (the shot reads as another genre, or something happens with no in-world cause) is judged on the new cut alone: `worse_here` is ignored and it blocks at severity **1**.
  - Why: relative scoring cannot block a defect the approved base also has. On q12b both critics rated the base's tone 2 and the new cut's tone 1, so nothing counted and the system predicted SHIP; the owner rejected it.
  - A dispute on tone means critics on opposite sides of severity 1, and gets the same single third read.
  - Evidence is in-sample: q12b (rejected → now BLOCK) and endB (approved, tone 1/0/0 → median 0 → SHIP).
- **Routed to the owner, not blocking:**
  - **TASTE:** severity 2, requested, class `grade`. Grade only, because a q8 test showed that taste routing on identity or cut lets regressions through.
  - **UNCERTAIN:** a severity-2 flicker on a lip-synced shot whose calibrated jitter gate passed. Strips can't see sub-visible jitter, and the critics flagged the approved q6c's accepted snap.
- **Prediction:**
  - **SHIP:** harness PASS, no counted severity-2 class, relative score ≥ 7, no open dispute. TASTE, UNCERTAIN and NEAR items are listed as owner checks.
  - **NEEDS THIRD READ:** the same, but with an open dispute.
  - **BLOCK:** anything else.

### 5. Report and prediction log [code]
- `score` writes `report.md`, always 8 lines in a fixed order:
  1. the item;
  2. the harness result, with FAIL / KNOWN / NEAR;
  3. critic defects, with TASTE / UNCERTAIN / DISPUTE;
  4. the scores per critic;
  5. not covered;
  6. attachments;
  7. the prediction;
  8. waiting for the OK.
- `score` also appends the prediction to `qa/predictions.log` before the owner sees the build. That log is append-only and hash-chained: each line carries the SHA-256 of the previous line, plus the video's hash. `python -m vf.qa verify-log` checks the chain.

### 6. Not covered (stated in every report)
- Lip-sync timing: the proxy is INFO only (correlation about 0.1). A slow-motion preview is made by hand; no code makes it.
- Sound by ear.
- Story and acting.
- Non-focus shots beyond the untouched check (worst 10×10 block on 90×160 thumbnails, every frame; it measures 0.0 because renders are deterministic, but it is not a full-resolution bit comparison). The full-episode run (`--focus` omitted) happens before every final render.

### 7. Calibration loop [code + manual]
- **Pinned cases:** `qa/cases.yaml` pins every owner-judged build in `qa/cases/`. These are not touched by `vf gc`.
- **Replay:** `python -m vf.qa replay` re-runs every case and checks:
  1. the harness verdict, **and** that the expected gates are among the failures (right verdict, right reason);
  2. stored critic answers, re-scored in code against `expect_system`.

  Unjudged cases are reported separately and never counted as agreement.
- **After every owner verdict [manual]:**
  - Add the case with its verdict.
  - Compare the verdict with the logged prediction. Held-out accuracy = logged predictions that match later verdicts, reported as k/n.
- **When they disagree:** an owner reject that the system shipped means adding or tuning a gate or prompt rule for that defect class, then re-running replay.
- **Critic consistency:** a repeat read of one packet every 5 rounds. If the relative score moves by more than 1, revise the prompt.

## Known limits (honest)
- The thresholds and the prompt were tuned **in-sample** on 4 owner verdicts. Held-out evidence starts with the q12b booth fix, whose prediction is logged.
- Both critics are Claude models. The only local non-Claude reader is a 4B VLM, which was not used for scoring.
- Absolute class severities vary between repeat reads (3 of 9 classes moved by 1 in one test). Only the relative score was stable (9.5 both times), so only the relative score gates, except `tone`, which gates absolutely at severity 1 (one read of 1 against one of 0 goes to a third read).
- An owner approval of a build covers the shots the owner was asked about, not every shot in it: q11 was approved for other shots while its booth_dies ending was never judged.

## v4 audit fixes (qa-4)
- Both builds now produce the same cut file set: `cut_into_<focus>` and `cut_out_of_<focus>` for every focus shot.
- Packet files are copied without timestamps (one fixed mtime), so copy order no longer reveals which cut is new.
- Answers are written to `cache/qa_answers/<id>/`, outside the packet, so a later reader cannot see earlier answers.
- A shot counts as "flicker gate passed" only if its lip-jitter gate actually ran and passed.
- The identity NEAR condition is fixed: it now means within 10 % of the gate.
- The video hash is SHA-256 of the whole file.
- The prediction is written back into `qa.json`.
- The report says "not logged" on re-scores.
