"""Script scoring with a local model: Qwen3.5-9B (Apache-2.0, NF4 on the CPU, embedding + lm_head in RAM, 4.2 GB peak;
vf/gen/director.py's loader) reads the whole script and answers a rubric of yes/no questions. Every score is the
probability of the good answer read from the Yes/No logits, never the printed word. Runs ONLY inside the gen venv; from
the main interpreter use vf.gen.bridge.run_gen('scriptqa', jobs, persistent=False).

    job = dict(script='<text>', bible='<text>', label='A')
    run_jobs(jobs) -> [dict(i, ok, label, scores={key: p_good}, p_yes={key: p}, notes='...', words, spoken_s)]

Exact facts are measured, not asked: words spoken and the speaking time they need (2.6 words/s, the pipeline's TTS).
The rubric asks one question at a time (a multi-part question made the models answer the easiest part).
"""
import copy, json, os, re, sys, time
_AI = os.environ.get('VF_AI_ROOT', r'D:\claude_work\hero_film')  # root of the AI venvs, tools and models (README: Configuration)

os.environ.setdefault('HF_HOME', os.path.join(_AI, 'models', 'hf'))

SCRIPTQA_VERSION = 'scriptqa-1'
WPS = 2.6   # spoken words per second of the pipeline's TTS (ep01 measured 2.4-2.8)
QS = [
    dict(key='hook', good='yes', q='Does the first spoken line make the viewer ask a question they want answered?'),
    dict(key='stakes', good='yes', q='By the end, is it clear what the main character stands to lose?'),
    dict(key='reveal', good='yes', q='Does the script contain a reveal that changes what the viewer thought they knew?'),
    dict(key='rising', good='yes', q='Does the tension rise from the first line to the last, without a flat stretch?'),
    dict(key='dialogue', good='yes', q='Do the spoken lines sound like a mature, motive-driven crime film, with each '
                                       'line doing more than explaining the plot?'),
    dict(key='standalone', good='yes', q='Could a viewer who has never seen the series follow what is happening?'),
    dict(key='cliffhanger', good='yes', q='Does the ending leave one specific unanswered question that makes the viewer '
                                          'want the next episode?'),
    dict(key='gore', good='no', q='Does this script depend on monsters, gore or jump scares to frighten the viewer?'),
    dict(key='shown', good='yes', q='Is every important revelation shown by an image, rather than only told by a '
                                    'narrator?'),
    dict(key='fresh', good='yes', q='Does the script avoid stock horror and ghost-story cliches?'),
    dict(key='bible', good='yes', q='Is the script consistent with the series bible?'),
    dict(key='surprise', good='yes', q='Would most viewers be surprised by the final line?'),
]
ASK = ('You are a script editor for a one-minute neo-noir thriller series.\n\nSERIES BIBLE:\n{bible}\n\nSCRIPT:\n{script}\n\n'
       'Question: {q}\nAnswer with only Yes or No.')
CRIT = ('You are a script editor for a one-minute neo-noir thriller series.\n\nSERIES BIBLE:\n{bible}\n\nSCRIPT:\n{script}\n\n'
        'Give your notes in at most four short sentences: the single biggest weakness of this script, the line that '
        'works least well (quote it), and what one change would improve it most.')
_SPLIT = '\x00Q\x00'


def spoken(script):
    """Words in the dialogue lines ('- <time> WHO: text' bullets), parentheticals and stage directions removed."""
    n = 0
    for ln in script.splitlines():
        m = re.match(r'\s*-\s*[0-9.]+\s+[A-Z ]+(?:\([^)]*\))?:\s*(.*)', ln)
        if m:
            t = re.sub(r'\([^)]*\)', '', m.group(1))
            n += len(re.findall(r"[A-Za-z0-9']+", t))
    return n


def shots_total(script):
    """Sum of the shot durations in the shot list ('N. name (3.4): ...') and the last dialogue time: exact facts."""
    tot = sum(float(m.group(1)) for m in re.finditer(r'^\s*\d+\.\s+\w+\s+\(([0-9.]+)\)', script, re.M))
    last = max([float(m.group(1)) for m in re.finditer(r'^\s*-\s*([0-9.]+)\s', script, re.M)] or [0.0])
    return round(tot, 1), last


def score_one(script, bible, critique=True):
    import torch
    from .director import _load, _fused_attention
    proc, mdl = _load('q9b')
    tok = proc.tokenizer
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', 'yes', ' Yes')})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ('No', 'no', ' No')})

    def templ(text):
        return proc.apply_chat_template([dict(role='user', content=[dict(type='text', text=text)])], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    pre, suf = templ(ASK.format(bible=bible, script=script, q=_SPLIT)).split(_SPLIT)
    ids = tok(pre, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
    p_yes = {}
    with torch.inference_mode(), _fused_attention():
        r = mdl(input_ids=ids, use_cache=True, logits_to_keep=1)
        cache = r.past_key_values
        del r
        for qd in QS:
            q = tok(qd['q'] + suf, add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
            c = copy.deepcopy(cache)      # the hybrid linear-attention cache cannot be cropped: copy it
            pr = torch.softmax(mdl(input_ids=q, past_key_values=c, use_cache=True, logits_to_keep=1).logits[0, -1].float(), -1)
            y, n = pr[yes].sum().item(), pr[no].sum().item()
            p_yes[qd['key']] = round(y / max(y + n, 1e-9), 3)
            del c, pr
    del cache
    notes = None
    if critique:
        x = tok(templ(CRIT.format(bible=bible, script=script)), add_special_tokens=False, return_tensors='pt')['input_ids'].to('cuda')
        with torch.inference_mode(), _fused_attention():
            out = mdl.generate(input_ids=x, max_new_tokens=160, do_sample=False)
        notes = tok.decode(out[0, x.shape[1]:], skip_special_tokens=True).strip()
        del x, out
    torch.cuda.empty_cache()
    scores = {qd['key']: (p_yes[qd['key']] if qd['good'] == 'yes' else round(1 - p_yes[qd['key']], 3)) for qd in QS}
    return scores, p_yes, notes


def run_jobs(jobs, emit=None):
    emit = emit or (lambda r: print(json.dumps(r), flush=True))
    from ..gpulock import gpu_lock
    from .guard import process_cap
    from .director import release
    import torch
    out = []
    with gpu_lock():
        process_cap()
        for i, j in enumerate(jobs):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            try:
                sc, py, notes = score_one(j['script'], j.get('bible', ''), bool(j.get('critique', True)))
                w = spoken(j['script'])
                shots_s, last_line = shots_total(j['script'])
                r = dict(i=i, ok=True, label=j.get('label'), scores=sc, p_yes=py, notes=notes, words=w,
                         spoken_s=round(w / WPS, 1), shots_s=shots_s, last_line=last_line, version=SCRIPTQA_VERSION, seconds=round(time.time() - t0, 1),
                         peak_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2))
            except Exception as e:
                import traceback
                traceback.print_exc()
                r = dict(i=i, ok=False, label=j.get('label'), error=f'{type(e).__name__}: {e}',
                         seconds=round(time.time() - t0, 1))
            out.append(r)
            emit(r)
        release()
    return out


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m vf.gen.scriptqa jobs.json')
    with open(sys.argv[1]) as f:
        run_jobs(json.load(f))
