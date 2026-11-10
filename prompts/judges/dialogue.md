# Blind critic: dialogue lines
You are reading {n} versions of the same dialogue, labelled {labels}. Context (characters, scene, intended tone) is
below. Read each version aloud in your head at speaking speed.

For EACH version give an integer 1-10 per criterion (10 = excellent) and one sentence of evidence quoting a line.
1. voice: each character sounds like one specific person (age, job, mood), not like the narrator
2. subtext: people want and hide things; nobody tells the other what both already know
3. economy: every line does work; nothing can be cut without loss
4. speakability: easy to say aloud in one breath, no tongue-twisters or written-only grammar
5. tone: matches the intended tone without melodrama
6. clarity: a listener with no screen still follows who wants what

Answer last with one ```json block: {"scores": {"<label>": {"<criterion>": <int>, ...}, ...}, "pick": "<label>"}

CONTEXT:
{context}

VERSIONS:
{entries}
