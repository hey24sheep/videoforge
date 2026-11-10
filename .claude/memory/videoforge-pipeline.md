---
name: videoforge-pipeline
description: "The user's generic video-series pipeline (D:\\claude_work\\videoforge) — rules, layout, speed facts; use for ANY new video/episode request"
metadata:
  node_type: memory
  type: project
  originSessionId: bcdc64f6-e189-45d4-91c7-0dd42090e64c
  modified: 2026-11-01T20:05:10.934Z
---

Since 2026-11-01 all video work goes through **videoforge** (D:\claude_work\videoforge; read its README.md and
CLAUDE.md). It's generic: each series/episode picks its own story and style (10 profiles in styles/styles.yaml,
master design prompt in prompts/director.md). Outputs YouTube 16:9 1080p and/or Shorts/Reels 9:16.

Rules from the user:
- Before a final render, ask which formats (not always both).
- A 1-min episode re-renders in 1–2 min after edits. Measured: ~32 s per format cold at 1080p60.
- Quality bar: no artifacts. Auto-QC and regenerate.
- Dialogue must be dramatic and mature.
- Memory stays generic, not tied to one film.
- The ECHOFALL project is scrapped (see [[superhero-film-project]]).

- Never start an episode unless the user asks for one (they objected once).
- AI clip size defaults to 480x832 / 832x480 (~47 s/clip). The user chose speed over the sharper 576x1024 (2026-11-01).

Stages, in order (built 2026-11-01): `vf voices` (Chatterbox + Whisper check) → `vf keyframes` (Z-Image-Turbo) →
review → `vf build` (Wan i2v clips, then render). First real episode: a private neo-noir series (9:16).

- A series runs many episodes. Supporting characters change, but the protagonist and antagonist stay. Keep them
  identical across episodes with series-level cast anchors (vf/identity.py).
- Writing bar (2026-11-02): "Oscar-worthy" story, screenplay and direction. Show, don't tell. Every twist must
  be proven on screen.
- When the user is away and says "do everything", decide autonomously and deliver the final render. Don't block
  on questions.

**Why:** the user wants one reusable instruction set for many series.
**How to apply:** start new videos as `series/<name>/<ep>/episode.yaml`. Keep vf/ free of story-specific code. Put
lessons in [[film-pipeline-performance]] and [[procedural-film-lessons]] (generic).

**2026-11-06 workflow:** read `videoforge/docs/workflow.md` first for any episode. Gates: `vf script-qa` + blind Opus/Sonnet critics (script), `vf lint`, `vf check` (brightness dips, frozen frames, SyncNet lip-sync) after every render.

**2026-11-09 integrated (owner: "fix the pipeline so it's all integrated"):** episodes are written as `<ep>_gen.py` on `vf/autolayout.py` (timings from real voice takes); text on props = `gen.ink`; release = `vf cut <episode.yaml>` (intro with the episode label + episode + outro + check + phone copy < 28 MB); music-only parts set `audio: {lufs: -18}`; `vf check` warns on floating props (talk shots) and loudness. Never hand-write concat/ffmpeg scripts per episode again.

**2026-11-09 package:** run everything with `vf run <gen.py|episode.yaml> [--until keyframes] --detach` (vf/runner.py; `vf doctor`, `vf new`, `vf.cmd`). Trailers: `release: {wrap: false}`. `vf check` is cached per shot (re-release ~4x faster). Owner OK'd permanent delete of _trash folders once (2026-11-09); still ask each time.
**Trailers (2026-11-09):** a private series' trailer_gen.py = template: reuse approved shots via pinned seed/seconds/image/motion_style (zero new AI clips), approved voice takes as `wav:` lines, black title cards, ACE-Step music ranked by CLAP + bass share with its biggest hit on the title card; `release: {wrap: false}`; check dialogue clarity with faster-whisper (CPU int8; the tts venv lacks cuBLAS for GPU).
**Showreel (2026-11-11):** series/showreel/reel = generic 15 s 16:9 README hero reel (sci-fi, music only, built-in hey24sheep.com/videoforge mark). README hero = animated WebP + linked mp4 (GitHub does not play repo mp4s inline).
**Judges + MCP (2026-11-11, owner asked):** `vf judge script|dialogue|music|clip|image|video` (vf/judge.py) is the one front door to every open-model judge; `vf judge critics ... --out DIR` writes blind Opus/Sonnet critic prompts from generic rubrics (prompts/judges/), `vf judge table DIR` derives scores. MCP server vf/mcp_server.py (mcp 2.3 `MCPServer`, 21 tools, docs resources, `make_video` prompt): stdio via `.mcp.json` / `vf-mcp`, `--http PORT` for remote; GPU tools start jobwatch jobs and return ids; paths stay in the repo, writes only under series/ and examples/. New style `realistic_scenic` (realistic without face/skin words: the `realistic` suffix pulled a man into empty space shots).
