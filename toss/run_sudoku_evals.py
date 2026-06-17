import csv
import re
import subprocess
import sys
from pathlib import Path

# Matches: "Sudoku accuracy: 45/100 (45.00%)"
ACCURACY_RE = re.compile(r'Sudoku accuracy:\s*(\d+)\s*/\s*(\d+)\s*\(([\d.]+)%\)')

def run_sudoku_evals(args, pattern='*.ckpt', extra_overrides=None,
                     cwd=None, csv_path=None):
    """Run sudoku_eval over every checkpoint in a folder and tabulate accuracy.

    Returns a list of dicts: {checkpoint, path, accuracy, num_correct,
    num_total, returncode}. accuracy is a 0-1 float, or None if parsing failed.
    """
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(p.resolve() for p in ckpt_dir.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints matching {pattern!r} in {ckpt_dir}')

    results = []
    for ckpt in ckpts:
        print(f'\n=== Evaluating {ckpt.name} ===', flush=True)
        cmd = [
            sys.executable, 'main.py',
            'mode=sudoku_eval',
            'data=sudoku-gen',
            'model=small',
            f'loader.global_batch_size={args.batch_size}',
            f'algo={args.algo}',
            f'eval.checkpoint_path={ckpt}',   # absolute, so hydra's chdir is irrelevant
        ]
        if extra_overrides:
            cmd.extend(extra_overrides)

        acc = num_correct = num_total = None
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in proc.stdout:          # stream + tee
            print(line, end='')
            m = ACCURACY_RE.search(line)
            if m:
                num_correct, num_total = int(m.group(1)), int(m.group(2))
                # Compute from the counts, not the printed %, which is rounded to 2dp.
                acc = num_correct / num_total if num_total else None
        ret = proc.wait()

        if ret != 0:
            print(f'  [warn] {ckpt.name} exited with code {ret}', flush=True)
        if acc is None:
            print(f'  [warn] no accuracy parsed for {ckpt.name}', flush=True)

        results.append({
            'checkpoint': ckpt.name, 'path': str(ckpt), 'accuracy': acc,
            'num_correct': num_correct, 'num_total': num_total, 'returncode': ret,
        })

    # --- summary table ---
    print('\n\n=== Summary ===')
    name_w = max((len(r['checkpoint']) for r in results), default=9)
    print(f'{"CKPT_Name".ljust(name_w)} | accuracy')
    print(f'{"-" * name_w}-+----------')
    for r in results:
        acc_str = f'{r["accuracy"] * 100:.2f}%' if r['accuracy'] is not None else 'FAILED'
        print(f'{r["checkpoint"].ljust(name_w)} | {acc_str}')

    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['checkpoint', 'accuracy', 'num_correct',
                                              'num_total', 'returncode', 'path'])
            w.writeheader()
            w.writerows(results)
        print(f'\nSaved table to {csv_path}')

    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', type=str)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--algo', type=str, default='discrete_loop_flm')
    args = ap.parse_args()
    run_sudoku_evals(args)