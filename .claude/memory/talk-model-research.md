---
name: talk-model-research
description: "talk-stage speed/heat research log lives in docs/model_research.md; add a dated row (status + why) before testing any new model idea"
metadata:
  type: project
---

Owner (2026-11-14): keep model research in videoforge with the model, the date and why it was rejected.

**Why:** the same ideas (GGUF Q4, FP8, GPUDirect, FP4) came up again; a log stops re-researching and re-testing them.

**How to apply:** docs/model_research.md. State 2026-11-14: heat is mostly GPU load on the shared laptop cooler
(2 cores still 81-83 C); GGUF Q4 low value (no fast kernel wheel for py312/torch 2.11); TalkVerse rejected (non-commercial);
1.3B talk models (StableAvatar MIT, HuMo 1.7B Apache, OmniAvatar) are the candidates, licence files not yet read.
Related: [[thermal-limit-80c]], [[models-not-formulas]].
