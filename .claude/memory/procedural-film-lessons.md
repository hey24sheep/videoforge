---
name: procedural-film-lessons
description: "What moved review scores when building code-rendered (numpy/PIL) cinematic animation, and the realistic quality ceiling"
metadata:
  node_type: memory
  type: project
  originSessionId: bcdc64f6-e189-45d4-91c7-0dd42090e64c
  modified: 2026-11-01T21:45:42.160Z
---

Learned on the ECHOFALL short (2026-10-31), see [[superhero-film-project]].

Score trajectory with an independent scorer subagent (10 = Marvel theatrical, 7 = strong indie): cut1 5.2 -> cut2 5.8 -> cut3 6.1 -> final 6.3 (delivered as hero_film/out/echofall_film.mp4; user's target was 8, not reached). Scorer's estimate: ~7.0-7.5 is the ceiling for pure procedural 2D; 8+ needs hand-authored key art for close-ups and many more animation keys.

Biggest gains: making a power *displace the image* (shear) rather than glow; close/low framing; hit-stop + impact frames; DOF background blur + ground mist; backlit hero at the payoff. Lightning-like branching cracks read as "electric power" — use straight angular cuts.

Reviewer pitfalls: subagents judging 480px filmstrips misjudge small detail (subtitle size, title crack) and misread a log-loudness plot as "no dynamics" — verify claims by measuring (per-section RMS) before acting.

v5 (2026-11-01, AI sprites via Wan 1.3B fast + SDXL key art, composited): my reviewers said 6.5, the USER rated it 5/10 — "many issues, bugs and artifacts, the story isn't there and it's broken", and was frustrated by how long the review-fix rounds took. Reviewer scores ran ~1.5 points above the user's: weight artifacts and story clarity far more, and stop iterating once comp fixes plateau. The 1.3B model could not hold character identity across clips even with reference images.

Lessons from a first episode, v3 (2026-11-02):
- Video models invent things in "nothing moves" shots: walk-ins, hands, gaits. 6 of 14 shots needed 2.5D stills:
  the keyframe moved by a compositor camera with depth parallax, plus procedural rain. They looked better than the AI
  clips.
- The VLM judge's "did anything appear?" question was right far more often than the pixel QC.
- The worst "shake" was caused by our own 24→60 fps nearest-frame holds, not by the AI. Fixed with RIFE retiming.
- Show the twist with compositor text on props (ledger, stamp, badge). The model can't draw legible text.

**Why:** user's target was 8/10; avoids repeating low-yield post-processing passes and long loops that don't move their score.
**How to apply:** for future generated-film work, set expectations about the ceiling early and ask whether to bring in real key art / image-gen; give reviewers >=640px frames. Ship a watchable full video early and often; keep rounds short; fix story clarity before polish. Related: [[user-creative-workflow]], [[film-pipeline-performance]].

**Lip-sync, MuseTalk 1.5 (2026-11-02):**
- **Good when:** a frontal, evenly lit, clean-shaven face (a close-up).
- **Fails when:** a 3/4, half-shadowed, bearded face that is small in the frame (the detective). The user saw a smeared blob and doubled lips.
- **Requirements:**
  - Use MuseTalk's crop: nose at the vertical centre, chin at the bottom.
  - Centre the mask on the landmark mouth.
  - The source clip must have the mouth closed.
- **Fallback:** a silent close-up plus inner-voice VO. Check a 1:1 crop before shipping.

## Rain on solid surfaces (owner reject, 2026-11-03, a badge shot)
- Expanding ripple rings read as drops hitting **liquid**, not wood or metal. The owner rejected them outright.
- **How to apply:** never put ripple rings on a non-puddle surface. Rain on wood or metal shows as tiny splashes, beading and a wet sheen, or not at all. Show a sample before testing any rain-interaction effect.

## Background swap by compositing (owner reject, 2026-11-05, a phone-call shot)
- Goal: move the detective from a street set to "home". Regenerating the clip with a new background made the K6 model put the cigarette out on every seed. Compositing the approved clip over a Klein room scored 9.0 with the critics, who rated "fixes the location". The owner said "now one looks better" and rejected it.
- "Polishing" the composite (removing light spill to kill a rim) made him look more pasted. That version scored 7.0 / BLOCK.
- **How to apply:** critics reward a fixed story point; the owner judges whether the shot still looks real. Before a long composite loop, show one rough composite frame next to the original and ask. Don't spend 4+ builds first.
