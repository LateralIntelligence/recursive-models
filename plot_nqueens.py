#!/usr/bin/env python
"""GRAM-style plots for N-Queens eval results.

Reads one or more ``results.json`` files produced by ``main._nqueens_eval`` and
renders, with the # of possible solutions per puzzle on the x-axis:
  - accuracy_vs_solutions.png  (y = accuracy)
  - coverage_vs_solutions.png  (y = coverage)

Multiple results files are overlaid as separate lines (one per checkpoint/run),
so you can compare models on the same axes.

Usage:
  python plot_nqueens.py RESULTS.json [MORE.json ...] [--out-dir DIR]
                         [--labels "a,b,..."] [--bins "1,2,4,8,16,32"]

If --out-dir is omitted the PNGs are written next to the first results file.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _series_from_results(results, bin_edges=None):
    """Return sorted (x, accuracy, coverage) arrays from a results dict.

    x is the # of possible solutions. When ``bin_edges`` is given, puzzles are
    grouped into [edge_i, edge_{i+1}) buckets (labelled by the left edge) and
    accuracy/coverage are averaged per bucket; otherwise each exact solution
    count is its own point.
    """
    records = results.get("records", [])
    if bin_edges:
        buckets = defaultdict(lambda: {"acc": [], "cov": []})
        for r in records:
            c = r["solution_count"]
            label = bin_edges[0]
            for e in bin_edges:
                if c >= e:
                    label = e
                else:
                    break
            buckets[label]["acc"].append(r["accuracy"])
            buckets[label]["cov"].append(r["coverage"])
        xs = sorted(buckets)
        acc = [sum(buckets[x]["acc"]) / len(buckets[x]["acc"]) for x in xs]
        cov = [sum(buckets[x]["cov"]) / len(buckets[x]["cov"]) for x in xs]
        return xs, acc, cov

    per = results.get("per_solution_count")
    if per:
        xs = sorted(int(k) for k in per)
        acc = [per[str(x)]["accuracy"] for x in xs]
        cov = [per[str(x)]["coverage"] for x in xs]
        return xs, acc, cov

    buckets = defaultdict(lambda: {"acc": [], "cov": []})
    for r in records:
        buckets[r["solution_count"]]["acc"].append(r["accuracy"])
        buckets[r["solution_count"]]["cov"].append(r["coverage"])
    xs = sorted(buckets)
    acc = [sum(buckets[x]["acc"]) / len(buckets[x]["acc"]) for x in xs]
    cov = [sum(buckets[x]["cov"]) / len(buckets[x]["cov"]) for x in xs]
    return xs, acc, cov


def _plot(metric_name, series, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("# of possible solutions in the puzzle")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"N-Queens {metric_name} vs # possible solutions")
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="one or more results.json paths")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: next to the first results file)")
    p.add_argument("--labels", default=None,
                   help="comma-separated labels, one per results file")
    p.add_argument("--bins", default=None,
                   help="comma-separated left bin edges, e.g. '1,2,4,8,16,32'")
    args = p.parse_args()

    bin_edges = [int(x) for x in args.bins.split(",")] if args.bins else None
    if args.labels:
        labels = args.labels.split(",")
        assert len(labels) == len(args.results), "need one label per results file"
    else:
        labels = [os.path.splitext(os.path.basename(os.path.dirname(r) or r))[0]
                  or os.path.basename(r) for r in args.results]

    acc_series, cov_series = {}, {}
    for path, label in zip(args.results, labels):
        with open(path) as f:
            results = json.load(f)
        xs, acc, cov = _series_from_results(results, bin_edges)
        acc_series[label] = (xs, acc)
        cov_series[label] = (xs, cov)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.results[0]))
    os.makedirs(out_dir, exist_ok=True)
    _plot("accuracy", acc_series, os.path.join(out_dir, "accuracy_vs_solutions.png"))
    _plot("coverage", cov_series, os.path.join(out_dir, "coverage_vs_solutions.png"))


if __name__ == "__main__":
    main()
