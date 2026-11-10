# videoforge workflow: idea to delivered Short

One path, in order. Every step names its command, what it checks, and when it blocks. Written 2026-11-06 from the Dead
Air ep01 work; each gate below exists because a defect got through without it. Hard rules first.

## Rules that apply to every step
- Fully local, open commercial licences only, no accounts, no paid APIs, no real-person voice cloning, no InsightFace.
- Never delete: move to `_trash/<date>` (videoforge or hero_film). Work only inside videoforge, the scratchpad and hero_film.
- One heavy GPU process at a time, always under `process_cap` (7.2 GB). Before any heavy run: `nvidia-smi` shows 0 MiB, no
  stray python. Wrap long runs with `tools/memwatch.ps1 <csv>`; afterwards `tools/passcheck.ps1 <csv>` (shared memory must stay flat).
- Never add flashes, dips, flicker, crash zooms or transitions the owner did not ask for. Disclose every choice of mine.
- Ask on taste decisions. Ask which formats before a final render. Models for every check, formulas only for exact facts.
- Python: `%LOCALAPPDATA%\Programs\Python\Python312\python.exe` (render, vf CLI); gen venv and tts venv are used through
  `vf.gen.bridge` only. pip into the tts venv needs `--no-deps` or a torch pin.

## 0. One command
`vf run <ep>_gen.py --until keyframes --detach`, review the sheet, then `vf run <ep>_gen.py --detach` (gen, lint,
voices, keyframes, release cut). `vf doctor` first on a new machine. Steps below are what it runs.

## 1. Script (gate: model rubric + blind critics)
1. Write 2-3 genuinely different drafts as `series/<s>/<ep>/script_options/<id>_<name>.md` (logline, tone, hooks advanced,
   dialogue as `- <time> WHO: text` bullets, shot list). Keep the approved previous episode as `REF_*.md`.
2. `python -m vf script-qa series/<s>/<ep>/script_options/*.md --bible series/<s>/bible.md --ref <REF.md> --json out.json`
   Qwen3.5-9B answers 12 yes/no questions per script (hook, stakes, reveal, rising tension, dialogue, standalone, cliffhanger,
   no gore, shown not told, freshness, bible fit, surprise) from the Yes/No logits, plus editor notes. A draft must reach the
   reference mean minus 0.25 or it BLOCKs. Spoken seconds are measured (2.6 words/s) and must fit 57 s.
3. Blind critics: Opus and Sonnet as fresh agents with `prompts/script_critic.md` (scripts relabelled, different order per
   critic, no opinions passed in). Score = mean of their criteria, computed by the operator. Gate: winner >= 7.0 on both,
   no criterion < 5, same winner (or within 0.3).
4. Revise at most twice, re-score, then take the winner forward. Show the owner the winner and both score tables.

## 2. Episode file
Write a generator script `<ep>/<ep>_gen.py` on `vf/autolayout.py` (beats with `insert()` / `talk_shot()` + lines with
acting directions) and let it write `episode.yaml`: shot lengths and `at:` times then come from the real voice lengths
(`voice_durs(out/voices.json)`, re-run after step 3). Scaffold one with `vf new <series> <ep> --no N --title T`.
Set `episode: {no, title}` (the intro's title card shows it). `python -m vf lint <episode.yaml>` (0 errors).
Authoring rules that matter: the AI camera never moves; props and states live in the keyframe; no mouth movement words
in motion prompts; one shot per beat, each <= 5 s of clip (dur x speed <= 5).
- Text on a prop (a log entry, a note) is written INTO the prop: `gen: {ink: {text, center, angle, size}}`
  (vf/inktext.py, carried onto the clip by the ink stamp). Never a `text` layer with `in_scene` (owner: "pasted on top").
- Talk shots keep the hands out of frame: never name a hand-held prop in their keyframe unless a hand holds it in the
  picture ("lantern glow below his face" made a lantern float). Lint warns; `vf check` reviews it.
In-person voices that must not sound like a telephone: `fx: null` on that dialogue line (the series voice default stays).

## 3. Voices
`python -m vf voices <episode.yaml>` acts every line (directed voices, vf/voicedir.py) and writes out/voices.json;
re-run the generator so the timings follow the real takes. A line that
scores low in the delivery check (vf/gen/delivery.py: MOS + same-voice) can be re-taken, but short lines score low by nature.

## 4. Keyframes and cast
`python -m vf keyframes <episode.yaml> --formats short` and review `out/keyframes_sheet.jpg`. Cast lock is on: a shot below
identity 0.60 is regenerated and flagged; profile shots are unreliable for the face model, check by eye.

## 5. Clips and build
`python -m vf build <episode.yaml> --formats short --preview` first, then the final once the owner picks formats.
`VF_GC=0` while iterating. Clip QC, repairs and re-rolls are automatic (`out/clip_qc.json`).
Lip-sync: talking shots use `gen.talk` (vf/talk.py, InfiniteTalk acting + cut points from the silent-mouth analysis, see CLAUDE.md); `gen.lipsync` (LatentSync, 8 steps, guidance 1.0, ~2 min per 4 s shot, 3.8 GB) is the old paste-on path, rejected by the owner for faces that should act.
A shot with `speed != 1` gets its voice time-compressed by the speed automatically (`rate`); before 2026-11-06 it did
not and the mouth drifted 440 ms. Never go back to the 20-step default: it spills past 8 GB and takes hours.

## 6. Checks on the rendered video (all must pass or be explained to the owner)
- `python -m vf check <episode.yaml>`: brightness dips (> 18 % and back within 14 frames), frozen frames, SyncNet
  lip-sync offset (|offset| <= 4 frames = 160 ms, confidence >= 4) on every lip-synced shot, open mouth without voice.
  Exit code 1 = BLOCK. WARN gates to look at yourself: floating props on talk shots (out/prop_review.jpg; series
  `check_props`) and loudness (true peak > -1 dBTP).
  A shot that really wants a dip says `allow_dip: true` and the owner must have asked for it.
- Review harness for a revised cut against the approved base (`vf/qa.py`, qa-11): RAFT flicker/motion, director vote
  (Qwen3.5-9B), face-quality drop (VisualQuality-R1), blind Opus + Sonnet critics, see `docs/review_workflow.md`.
- Identity < 0.60 on a shot: regenerate (new seeds), keep the best, flag it to the owner. Do not hide it.
- The owner's own eyes are the final gate. Weight story clarity and visible artifacts above polish.

## 7. Deliver
`python tools/jobwatch.py run --name <x> --env VF_GC=0 -- python -m vf cut <episode.yaml>` (vf/cut.py): series intro
(its `{episode}` text = the episode label) + episode + series outro, `vf check`, joined at 60 fps ->
`out/<Name>_epNN_<Title>_FULL.mp4`, plus `_FULL_phone.mp4` (HEVC sized to < 28 MB). `--skip-build` re-joins the newest
renders. A music-only part (intro) sets `audio: {lufs: -18}`: at the dialogue target of -14 it is too loud.
Send both with the file tool; say what changed, what was not verified, what is still flawed.

## 8. Close out
Update `STATE.md` (decisions, files, blockers), save a memory for anything non-obvious, move superseded renders to `_trash`.

## Known limits (not bugs in the finishing chain)
- Kandinsky 6 output is 24 fps with a 4-frame cadence from its temporal decoder; the 60 fps conversion keeps that cadence
  (final alternation 0.16-0.31 vs raw 0.14-0.22). Faces drift off the cast anchor in some shots (ep01: detective_phone 0.54,
  two wide shots at 0.48 and 0.57, all flagged).
- LatentSync redraws the lips, so lipstick colour is not preserved (muted natural red).
- The director vote can wrongly block an owner-approved change (ep01 cases q6c, endB): it is advisory when the owner approved.
