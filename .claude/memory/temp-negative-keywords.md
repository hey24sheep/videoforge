---
name: temp-negative-keywords
description: Owner's temporary 315-term negative keyword list for videoforge, with an on/off switch the owner may flip
metadata:
  type: project
---

On 2026-11-04 the owner asked for a temporary negative keyword list ("temp keyword file"). It lives at `videoforge/prompts/temp_negative_keywords.txt`, with an `active: yes|no` line, and is loaded by `vf.styles.temp_negative()`.

**Why:** the owner wants it applied to generation, but may ask to deactivate it if it "backfires on creation capabilities". Deactivated means do not use those negative keywords at all.

**How to apply:**
- When the owner says deactivate, set `active: no`. Never delete the file.
- Every current model (K6 Lite distill, Z-Image Turbo, Wan Turbo, FLUX.2 Klein) runs at guidance 1.0 and ignores negatives. The list only takes effect with a real-CFG model. Do not claim it improved a generation.

Related: [[videoforge-pipeline]], [[procedural-film-lessons]].
