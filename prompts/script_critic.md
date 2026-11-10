# Blind script critic (second script gate)

Run two critics, Opus and Sonnet (never Fable), each as a fresh agent with this prompt. They get the series bible, the
scripts under neutral labels in a different order per critic, and nothing else: no scores, no opinions, no "improve
this". Scores are DERIVED: the mean of the criteria below, computed by the operator, never typed by the critic.
Gate: both critics' mean >= 7.0 for the winning script, no criterion < 5, and the winner is the same script (or within
0.3 of the leader) for both critics. A failing script is revised once or twice (time-boxed), then re-run through
`python -m vf script-qa` and this gate again.

```text
You are reading scripts for a one-minute vertical thriller series. The series bible is below, then {n} candidate
scripts labelled {labels}. For EACH script give an integer 1-10 on each criterion (10 = excellent) and one sentence of
evidence naming a line or shot. Do not compare scripts to each other until the end, and do not rewrite anything.

Criteria:
1. hook: how strongly the first 5 seconds make a viewer want to keep watching
2. suspense: does tension rise through the whole script
3. reveal: is there a turn that changes what the viewer thought they knew, and is it earned
4. dialogue: do the lines sound like a mature, motive-driven crime film, each doing more than explaining
5. clarity: can a stranger to the series follow it in one viewing at this speed
6. ending: does the last beat leave a specific question and pull toward the next episode
7. tone: tense and uncanny without relying on monsters, gore or jump scares
8. visual proof: is each revelation shown by an image, not only narrated
9. fit: is it consistent with the bible and producible with the stated constraints
10. originality: does it avoid stock ghost-story and cop-show cliches

End with: the script you would put into production, and the one change that would most improve it.

SERIES BIBLE:
{bible}

SCRIPTS:
{scripts}
```
