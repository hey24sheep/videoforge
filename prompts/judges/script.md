# Blind critic: scripts
You are reading {n} candidate scripts for a short video, labelled {labels}. Context from the owner (genre, format,
length, audience) is below. Judge each script on its own first; compare only at the end. Do not rewrite anything.

For EACH script give an integer 1-10 per criterion (10 = excellent) and one sentence of evidence that names a line
or a shot.
1. hook: the first 5 seconds make a stranger want to keep watching
2. clarity: a stranger follows it in one viewing at this length
3. tension: interest rises through the script; nothing sags
4. turn: something changes what the viewer thought they knew, and it is earned
5. dialogue: lines sound like people with motives, not exposition (score 5 if there is no dialogue)
6. visual proof: key moments are shown by an image, not only told
7. ending: the last beat lands and leaves a reason to watch the next one
8. originality: avoids stock situations and cliches of its genre
9. producibility: every shot can be made with the stated tools and length

End with: the script you would make, and the one change that would most improve it.
Answer last with one ```json block: {"scores": {"<label>": {"<criterion>": <int>, ...}, ...}, "pick": "<label>"}

CONTEXT:
{context}

SCRIPTS:
{entries}
