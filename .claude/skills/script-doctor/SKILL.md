---
name: script-doctor
description: Write or fix a videoforge episode script until it passes the owner's bar (blind viewer-transcript critics >= 7, story matches earlier episodes). Use before ANY episode production, and whenever the owner says the dialogue/story is bad, broken, not enticing, or makes no sense.
---

# script-doctor

Owner bar (2026-11-08): a script ships to production only when the judges score it >= 7 mean with no criterion < 5,
the story matches the earlier episodes, and the owner approves it. Never start GPU work on an unapproved script.

## 0. Series first (owner 2026-11-08: a series runs 10-15 episodes)
- Read `series/<s>/season_arc.md` before writing. Each episode spends a small reveal budget; never resolve the saga question.
- Use the installed creative-writing skills: `story-planning` (arc-design, structural-problems), `story-workflow`
  (2 critique rounds per piece, then present to the owner), `reader-sim`, `story-review/resources/developmental-edit.md`.
- Critic prompt adds: "It is episode N of a M-episode serial. Viewers have seen: <2-line recap>." (real viewers have).

## 1. Load canon first (never write before this)
- `series/<s>/bible.md`, every earlier episode's `episode.yaml` dialogue (`grep -n "text:"`) and its on-screen text layers.
- Write a FACT SHEET (in the episode's HANDOFF.md): who called whom, when, what was found, what each character can
  physically know, when each clue appeared, what the earlier episodes said word for word. Every line must obey it.
  Patching lines without a fact sheet stalled at 5.5-6.6 for six rounds on a test episode.

## 2. Craft rules (researched 2026-11-08)
- Micro-drama shape: hook in the first 3 s (a face, a conflict, or an impossible image + one guilt/stakes line),
  one escalation, a turn, cliffhanger one beat before resolution. Ask: what is different at the end?
- Interrogation = power shift: interrogator starts in control, suspect pushes back with a plausible excuse, the
  suspect turns it on the interrogator (subtext: the thing he can't say), then breaks on PROOF he can't dodge
  (an object, a sightline), never on one soft question. Make clear what the suspect stands to lose.
- Fair-play clues: plant early, pay off later; every clue visible on screen; each clue physically plausible
  (fog writing fades in minutes; a searched tray can't hold ten-day-old evidence; a witness can't know who was called).
- Characters only say what they could know. Keep ONE uncanny element per episode, framed as intended mystery.
- Narrated series: the lead's voice-over stays (owner: "not a silent series"), short punchy lines.
- Lines carry conflict or subtext; cut lines that only explain.

## 3. Viewer transcript
Write `<ep>/viewer_Pn.md`: one line per shot = [what is on screen] + who speaks + the line. No logline, no notes:
critics get ONLY what a viewer sees and hears (critics who read the logline rated a confusing script clear).

## 3b. METHOD V2 (2026-11-10, replaces the gap-list loop below for drama decisions)
The section-4 loop (critics list every gap, patch, re-score) plateaued at 6.1-7.1 on a test episode for 10 drafts and
produced "a radio play of facts" (owner: "you don't know cinema"). Read the series' CRAFT.md first, if it has one. Write 2-3
structurally different spines as timed beat sheets (sound, silence); rank them pairwise against the owner-approved
previous episode with persona judges (noir showrunner, development exec, vertical phone viewer); drama 70 %, logic
capped at the 3 worst "lost" points; max 2 patch rounds; the owner picks. Section 4 stays as a logic check only.

## 4. Judges (never Fable)
- Opus x2 and Sonnet x2, each a fresh agent, neutral prompt (template below), runs in parallel, run in background.
- Plus your own judgement: read the transcript cold as a viewer and list gaps before the critics do.
- Means computed by you (8 criteria). Sonnet varies +-0.5 run to run: average runs, never trust one.
- Fix the gaps that 2+ judges name; re-score. Record every round's scores in the episode HANDOFF.md.

Critic prompt (do not add hints, do not say which draft is new):
"You are a first-time viewer test for a <length> vertical crime short. Read the shot-by-shot transcript at <path>.
Each line is what is on screen [in brackets] and what is heard. You know nothing else about the series. Treat it as
watched once at normal speed. A. Retell the story in at most 5 plain sentences. B. List every moment where a viewer
would be confused, lost, bored, or notice a logic gap (shot number + why). C. Integer 1-10 with one sentence of
evidence each: hook, clarity, suspense, dialogue, interrogation (if any), believability, ending, visual proof.
Do not rewrite it. End with the one change that would most improve it. Plain text, under 450 words."

## 5. Hand to the owner
Send the full script as readable lines + the score table + what changed. Wait for approval.
