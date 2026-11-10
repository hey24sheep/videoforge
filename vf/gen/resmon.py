"""Run a command and sample system resources while it runs: system RAM used, process-tree RSS, CPU %, GPU
temperature / power / utilisation / VRAM. Prints the command's interesting lines and a summary.
Usage: python -m vf.gen.resmon -- <command...>"""
import subprocess, sys, threading, time
import psutil


def gpu():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        t, p, u, m = [x.strip() for x in r.stdout.strip().splitlines()[0].split(',')]
        return int(t), float(p), int(u), int(m)
    except Exception:
        return 0, 0.0, 0, 0


def main(cmd):
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    KEYS = ('DONE', 'vae decode', 'sampled in', 'DiT back', 'retrying smaller', 'loaded', 'thermal', 'resumed', 'encoded', 'Error', 'Traceback', 'RTX', 'FRAME', 'LIPSYNC',
            'finished', 'short:', 'ai stage', 're-roll', 'QC fail', 'early finish', 'k6:', 'wan', 'total')

    def pump():  # print key lines as they arrive: a log held until exit hid an hour-long stall (2026-11-02)
        for ln in proc.stdout:
            if any(k in ln for k in KEYS):
                print(ln.rstrip(), flush=True)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    samples = []
    pp = psutil.Process(proc.pid)
    psutil.cpu_percent(None)
    while proc.poll() is None:
        time.sleep(2)
        try:
            kids = [pp] + pp.children(recursive=True)
            rss = sum(k.memory_info().rss for k in kids if k.is_running()) / 2**30
        except Exception:
            rss = 0
        vm = psutil.virtual_memory()
        samples.append((time.time() - t0, vm.used / 2**30, rss, psutil.cpu_percent(None)) + gpu())
    th.join(timeout=5)
    if samples:
        col = list(zip(*samples))
        print(f'SUMMARY wall {time.time() - t0:.0f}s | sys RAM used max {max(col[1]):.1f} GB | proc RSS max '
              f'{max(col[2]):.1f} GB | CPU avg {sum(col[3]) / len(col[3]):.0f}% max {max(col[3]):.0f}% | GPU temp max '
              f'{max(col[4])} C avg {sum(col[4]) / len(col[4]):.0f} C | power max {max(col[5]):.0f} W | util avg '
              f'{sum(col[6]) / len(col[6]):.0f}% | VRAM max {max(col[7])} MiB | exit {proc.returncode}')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[a.index('--') + 1:] if '--' in a else a)
