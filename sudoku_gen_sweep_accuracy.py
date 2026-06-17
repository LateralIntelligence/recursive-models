#!/usr/bin/env python3
"""Collect sudoku_eval accuracies across a sweep into a single JSON.

Walks a sweep directory (default: outputs/sweep), finds run folders that have a
non-empty sudoku_eval result, parses the (gamma, T, B) hyperparameters out of
the run folder name, extracts the accuracy, and writes a JSON summary.

Run folder names are expected to encode the hyperparameters, e.g.
  Disc_FLM_Looped_Sudoku_gen_g=0.0_T=6_bp=2
Both '=' and ':' separators are accepted (g=0.0 / g:0.0).

Each run may have several checkpoints under
  <run>/sudoku_eval/<ckpt_stem>/results.json
so --select chooses which one represents the run:
  best (default) -> the checkpoint with the highest accuracy
  last           -> the highest version-sorted checkpoint stem
  all            -> emit one record per checkpoint (adds a "checkpoint" field)

Usage:
  python sudoku_gen_sweep_accuracy.py
  python sudoku_gen_sweep_accuracy.py --sweep-dir outputs/sweep --select best
  python sudoku_gen_sweep_accuracy.py --output outputs/sweep/summary.json --select all
"""

import argparse
import glob
import json
import os
import re


# gamma=float, T=int, bp/B=int; tolerate '=' or ':' as the separator.
_PATTERNS = {
    'gamma': re.compile(r'(?:^|[_-])g[=:]([0-9]*\.?[0-9]+)'),
    'T': re.compile(r'(?:^|[_-])T[=:]([0-9]+)'),
    'B': re.compile(r'(?:^|[_-])(?:bp|B)[=:]([0-9]+)'),
}


def _natural_key(s):
    """Version-ish sort key so step-10000 < step-100000, last < last-v1, etc."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s)]


def parse_hparams(run_name):
    """Pull {gamma, T, B} out of a run folder name. Returns None if incomplete."""
    out = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(run_name)
        if m is None:
            return None
        val = m.group(1)
        out[key] = float(val) if key == 'gamma' else int(val)
    return out


def load_result(results_path):
    """Return a result dict if the file parses and is non-empty, else None."""
    try:
        with open(results_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    total = data.get('num_total')
    if not total or total <= 0:
        return None  # empty / no puzzles scored
    accuracy = data.get('accuracy')
    if accuracy is None:
        accuracy = data['num_correct'] / total
    return {
        'accuracy': accuracy,
        'num_correct': data.get('num_correct'),
        'num_total': total,
    }


def collect(sweep_dir, select):
    records = []
    skipped = []
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(sweep_dir, '*')) if os.path.isdir(d))

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        hparams = parse_hparams(run_name)
        if hparams is None:
            skipped.append((run_name, 'could not parse g/T/bp from name'))
            continue

        # Each checkpoint gets its own sudoku_eval/<stem>/results.json.
        result_paths = sorted(
            glob.glob(os.path.join(run_dir, 'sudoku_eval', '*', 'results.json')),
            key=lambda p: _natural_key(os.path.basename(os.path.dirname(p))))

        ckpt_results = []
        for rp in result_paths:
            res = load_result(rp)
            if res is None:
                continue
            ckpt_results.append((os.path.basename(os.path.dirname(rp)), res))

        if not ckpt_results:
            skipped.append((run_name, 'no non-empty sudoku_eval results'))
            continue

        if select == 'all':
            chosen = ckpt_results
        elif select == 'last':
            chosen = [ckpt_results[-1]]
        else:  # best
            chosen = [max(ckpt_results, key=lambda kr: kr[1]['accuracy'])]

        for ckpt_stem, res in chosen:
            rec = {
                'gamma': hparams['gamma'],
                'T': hparams['T'],
                'B': hparams['B'],
                'accuracy': res['accuracy'],
                'num_correct': res['num_correct'],
                'num_total': res['num_total'],
                'run': run_name,
            }
            if select == 'all':
                rec['checkpoint'] = ckpt_stem
            records.append(rec)

    # Stable, readable ordering.
    records.sort(key=lambda r: (r['gamma'], r['T'], r['B'],
                                r.get('checkpoint', '')))
    return records, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep-dir', default='outputs/sweep',
                    help='Directory containing per-run folders (default: outputs/sweep)')
    ap.add_argument('--output', default=None,
                    help='Output JSON path (default: <sweep-dir>/sweep_accuracy.json)')
    ap.add_argument('--select', default='best', choices=['best', 'last', 'all'],
                    help='Which checkpoint represents each run (default: best)')
    args = ap.parse_args()

    if not os.path.isdir(args.sweep_dir):
        raise SystemExit(f'sweep dir not found: {args.sweep_dir}')
    output = args.output or os.path.join(args.sweep_dir, 'sweep_accuracy.json')

    records, skipped = collect(args.sweep_dir, args.select)

    with open(output, 'w') as f:
        json.dump(records, f, indent=2)

    print(f'Collected {len(records)} record(s) from {args.sweep_dir} '
          f'(select={args.select}) -> {output}')
    for r in records:
        ck = f"  [{r['checkpoint']}]" if 'checkpoint' in r else ''
        print(f"  gamma={r['gamma']}  T={r['T']}  B={r['B']}  "
              f"acc={r['accuracy']:.4f}  ({r['num_correct']}/{r['num_total']}){ck}")
    if skipped:
        print(f'\nSkipped {len(skipped)} run(s):')
        for name, why in skipped:
            print(f'  {name}: {why}')


if __name__ == '__main__':
    main()
