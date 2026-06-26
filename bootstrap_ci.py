"""Bootstrap confidence intervals for Sudoku / N-Queens eval results.

Each eval ``results.json`` has a ``records`` list. Two layouts are supported and
auto-detected:

- **Sudoku** — each record is one generation with a boolean ``correct``. We
  bootstrap over generations; the statistic is accuracy = mean(correct).
- **N-Queens** — each record is one *input board* (with ``num_samples``
  generations summarized into per-board ``accuracy`` and ``coverage``). We
  bootstrap over input boards; the statistics are the mean per-board accuracy
  and the mean per-board coverage, matching ``nqueens_eval`` aggregation.

Usage (programmatic):
    from bootstrap_ci import bootstrap_file, summarize
    res = bootstrap_file("check/best_nll/results.json", name="sudoku-baseline")
    print(summarize([res]))

Usage (CLI) -- pass NAME=PATH pairs (NAME optional, defaults to the path):
    python bootstrap_ci.py baseline=check/best_nll/results.json \
        nqueens-0.2=outputs/.../nqueens_eval/last/results.json
    python bootstrap_ci.py --ci 0.95 --n-boot 10000 --seed 0 run1=foo.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


def detect_kind(records: List[dict]) -> str:
    """Return 'sudoku' or 'nqueens' based on record schema."""
    if not records:
        raise ValueError("empty records list -- cannot detect result type")
    r0 = records[0]
    if "correct" in r0:
        return "sudoku"
    if "accuracy" in r0 and "coverage" in r0:
        return "nqueens"
    raise ValueError(f"unrecognized record schema with keys {sorted(r0)}")


def _bootstrap_mean(values: np.ndarray, n_boot: int, ci: float,
                    rng: np.random.Generator) -> Dict[str, float]:
    """Bootstrap the mean of a 1-D array of per-unit statistics."""
    n = len(values)
    # idx shape (n_boot, n); each row is one resample of unit indices.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {
        "point": float(values.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se": float(boot_means.std(ddof=1)),
        "n": int(n),
    }


@dataclass
class BootResult:
    name: str
    path: str
    kind: str
    n_units: int
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


def bootstrap_records(records: List[dict], *, name: str, path: str = "",
                      n_boot: int = 10000, ci: float = 0.95,
                      seed: int = 0) -> BootResult:
    """Compute bootstrap CIs for one records list."""
    kind = detect_kind(records)
    rng = np.random.default_rng(seed)
    metrics: Dict[str, Dict[str, float]] = {}

    if kind == "sudoku":
        vals = np.array([bool(r["correct"]) for r in records], dtype=float)
        metrics["accuracy"] = _bootstrap_mean(vals, n_boot, ci, rng)
    else:  # nqueens -- bootstrap over input boards
        acc = np.array([float(r["accuracy"]) for r in records], dtype=float)
        cov = np.array([float(r["coverage"]) for r in records], dtype=float)
        metrics["accuracy"] = _bootstrap_mean(acc, n_boot, ci, rng)
        metrics["coverage"] = _bootstrap_mean(cov, n_boot, ci, rng)

    return BootResult(name=name, path=path, kind=kind,
                      n_units=len(records), metrics=metrics)


def bootstrap_file(path: str, *, name: str = "", n_boot: int = 10000,
                   ci: float = 0.95, seed: int = 0) -> BootResult:
    with open(path) as f:
        data = json.load(f)
    records = data["records"]
    return bootstrap_records(records, name=name or path, path=path,
                             n_boot=n_boot, ci=ci, seed=seed)


def summarize(results: List[BootResult], ci: float = 0.95) -> str:
    pct = int(round(ci * 100))
    lines = []
    width = max((len(r.name) for r in results), default=4)
    header = f"{'name':<{width}}  {'kind':<8}  {'metric':<9}  {'point':>8}  {f'{pct}% CI':>20}  {'units':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        for metric, m in r.metrics.items():
            ci_str = f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]"
            lines.append(
                f"{r.name:<{width}}  {r.kind:<8}  {metric:<9}  "
                f"{m['point']:>8.4f}  {ci_str:>20}  {m['n']:>6}"
            )
    return "\n".join(lines)


def _parse_spec(spec: str) -> tuple[str, str]:
    """Parse a NAME=PATH (or PATH) CLI argument."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, path
    return spec, spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+", metavar="NAME=PATH",
                    help="results.json files, optionally prefixed with NAME=")
    ap.add_argument("--ci", type=float, default=0.95, help="confidence level")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="number of bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0, help="rng seed")
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of a table")
    args = ap.parse_args()

    results = []
    for spec in args.specs:
        name, path = _parse_spec(spec)
        results.append(bootstrap_file(path, name=name, n_boot=args.n_boot,
                                      ci=args.ci, seed=args.seed))

    if args.json:
        out = [{"name": r.name, "path": r.path, "kind": r.kind,
                "n_units": r.n_units, "metrics": r.metrics} for r in results]
        print(json.dumps(out, indent=2))
    else:
        print(summarize(results, ci=args.ci))


if __name__ == "__main__":
    main()
