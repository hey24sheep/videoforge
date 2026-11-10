"""Write the generation jobs an episode WOULD send to K6 or Klein, without generating anything (for same-input A/B runs).
    python tools/precision/dump_jobs.py k6    <episode.yaml> <out.json>    (runs `vf build` up to the K6 call)
    python tools/precision/dump_jobs.py klein <episode.yaml> <out.json>    (runs `vf keyframes` up to the Klein call)
Then: K6_PREC=<p> <gen python> -m vf.gen.k6_batch jobs.json | VF_KLEIN_PREC=<p> <gen python> -m vf.gen.klein jobs.json,
after pointing each job's `out` somewhere else (tools/precision/ab_compare.py)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
KIND, EP, OUT = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])


def dump(jobs, *a, **kw):
    json.dump(jobs, open(OUT, 'w'), indent=1)
    print('DUMPED', len(jobs), '->', OUT)
    raise SystemExit(0)


if KIND == 'k6':
    import vf.k6 as m
    m.clips = dump
    cmd = 'build'
else:
    import vf.cast as m
    m.run_klein = dump
    cmd = 'keyframes'
from vf import build  # noqa: E402
sys.argv = ['vf', cmd, EP, '--formats', 'short']
build.main()
