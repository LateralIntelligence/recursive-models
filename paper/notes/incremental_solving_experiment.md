# Incremental-solving mechanism experiment (board-state reliance)

**Date:** 2026-06-25
**Figure:** `figures/sudoku_incremental_solving.png` (also copied to `paper/figures/`)
**Script:** `figures/sudoku_incremental_solving.py` (self-contained; touches no training code or model files)
**Cached results:** `figures/sudoku_incremental_solving_data.json`
**Paper location:** Section 5 (Analysis & Discussion), new figure `\label{fig:incremental}` + 4th point in "Why does noising the conditioning help?"

---

## Claim this supports

Noising the conditioning during training changes *how* the model solves the task: the
low-`p` (heavily noised) model learns to **solve incrementally** — it increasingly uses its
own evolving partial board over sampling steps — whereas the clean paste-in model (`p=1.0`)
stays clue-bound and only reads the board near the end. This is measured at **clean-clue
inference** (the standard regime, *no* conditioning noise at test time), so it is **not** a
noise-robustness result.

## Metric: causal board-swap reliance

During the standard Euler sampling rollout (clean clues clamped every step), at each step `i`:

1. Run the model on the normal state `z = (own clues, own current partial board)` → prediction `pred`.
2. Build a counterfactual `z' = (own clues, ANOTHER puzzle's partial board)` by rolling the
   batch by 1 along the puzzle axis and pasting each row's *own* clues back
   (`torch.where(cond_mask, z, roll(z,1))`). Both partial boards sit at the **same noise
   level**; only the board-state *content* differs. → prediction `pred_s`.
3. **Reliance(i)** = fraction of to-fill (non-clue) cells whose argmax prediction flips
   between `pred` and `pred_s`, averaged over puzzles.
   High ⇒ the model relies on its evolving solution (incremental); low ⇒ clue-bound.
4. Also log per-step `x1` fill accuracy (`pred == ground-truth` on to-fill cells).

Curves are means over 512 held-out puzzles with 95% bootstrap bands (resampled over puzzles).

## Models / checkpoints (the key detail)

Three Sudoku-hard runs, board infill layout, `N=10000`, **all at global_step 40000 (epoch 999;
`max_steps=40001`, the paper's hard recipe)**, configs **identical except
`algo.conditioning_prob_clean`** (verified by diffing `.hydra/config.yaml`: the only line that
differs is `conditioning_prob_clean`). These are the checkpoints behind Table 1 — they
reproduce its Sudoku-hard accuracies (see sanity check below).

```
sudoku-hard/sweep-flm-infill-sudoku-hard/FLM_Infill_Sudoku_hard_board_N:10000_cond_time_random:true_clean_prob:0.2/checkpoints/999-40000.ckpt
sudoku-hard/sweep-flm-infill-sudoku-hard/FLM_Infill_Sudoku_hard_board_N:10000_cond_time_random:true_clean_prob:0.5/checkpoints/999-40000.ckpt
sudoku-hard/sweep-flm-infill-sudoku-hard/FLM_Infill_Sudoku_hard_board_N:10000_cond_time_random:true_clean_prob:1.0/checkpoints/999-40000.ckpt
```

> **Checkpoint provenance (important).** Use the `sudoku-hard/sweep-...` dir at **40k steps**,
> not `outputs/sweep-flm-infill-sudoku-hard/.../479-150000.ckpt`. The 150k run is a different,
> longer run whose accuracy has drifted *down* (its own saved eval gives p=0.2 ≈ 15.8%, which
> does **not** match Table 1); the 40k checkpoints reproduce Table 1 (p=0.2 ≈ 25%). An earlier
> version of this experiment used the 150k checkpoints and got the lower (non-paper) numbers;
> switched to the 40k checkpoints 2026-06-25. Also avoid `last.ckpt` in any of these dirs — it
> is mislabeled to inconsistent global_steps per run.

## Data / sampling config

- **Task:** Sudoku-hard, in-place infill layout, `length=81`, `infill_loss_region=board`,
  `difficulty=hard` → 30/81 cells given as clues (so ~30 clue / ~51 to-fill cells; the model
  *could* in principle solve from clues alone, which is what makes "ignore the board" a real
  alternative strategy and the test meaningful).
- **Puzzles:** first 512 real puzzles from the eval dataloader under `L.seed_everything(0)`;
  the same 512 puzzles are used for all three models (identical configs ⇒ identical eval stream).
  Eval set is the run's `num_valid=2000` split.
- **Inference:** clean clues clamped each step (standard `conditional_generate_samples`
  rollout, reimplemented inside the script so we can capture the per-step counterfactual);
  `steps=128`, batch 256.
- **EMA weights (CRITICAL):** the script calls `m._eval_mode()` after loading, which copies
  the EMA shadow parameters onto the model — this is what `mode=sudoku_eval` and the other
  figures use. Plain `.eval()` leaves the *raw* (non-EMA) weights, which underperform
  (p=0.2 full-board solve drops ~15% → ~10%). An earlier version of this script used `.eval()`
  and produced low solve rates; fixed 2026-06-25. Always use `_eval_mode()`.
- **Hardware:** 1× NVIDIA RTX 5090; full run ≈ 2–3 min.

## Sanity check: does the model solve at the expected rate?

Full-board exact-match solve = `(generated == ground_truth).all()` — the same metric as
`mode=sudoku_eval` (`main.py`). The generated board is the model's final `x1` prediction
(`conditional_generate_samples` sets `z = x1_pred` at the last step), so the right "solve rate"
to compare is **`x1solve`**, not the interpolated `zsolve`.

| metric (EMA weights, 40k ckpts) | p=0.2 | p=0.5 | p=1.0 |
|---|---|---|---|
| this run, x1solve, full 2000 | **25.00%** | 7.95% | 1.10% |
| paper Table 1 (Sudoku-hard) | 25.45% | 7.70% | 1.25% |
| this run, x1solve, 512-subset | 22.07% | 7.03% | 0.98% |
| this run, zsolve, 512-subset | 13.09% | 1.95% | 0.20% |

- **The 40k checkpoints reproduce Table 1** (25.0 vs 25.45, 7.95 vs 7.70, 1.10 vs 1.25). The
  figure uses a 512-puzzle subset (first 512 eval puzzles, slightly harder) giving x1solve
  22.07 / 7.03 / 0.98% — same ordering, faithful sampler (our rollout reproduces
  `conditional_generate_samples` exactly).
- `x1solve` (not `zsolve`) is the comparable metric: `conditional_generate_samples` returns the
  final `x1` prediction (it sets `z = x1_pred` at the last step), so the generated board is the
  argmax of `x1`, whereas `zsolve` measures the interpolated `z` just before that projection
  and runs lower.
- **Why per-cell looks close but solve-rate is far apart** (the original surprise): final
  per-cell `x1` accuracy is 76.9 / 69.2 / 65.5% (p=0.2/0.5/1.0) — only ~11 pts apart — but
  full-board solve needs all ~51 to-fill cells correct simultaneously, so the modest per-cell
  edge compounds into a large solve-rate gap (25% vs 1.1%).

## Headline numbers (512 puzzles, 128 steps, EMA weights; from the cached json)

Board-state reliance (% of to-fill tokens that flip under board swap):

| sampling progress | step | p=0.2 (heavy noise) | p=0.5 | p=1.0 (paste-in) |
|---|---|---|---|---|
| ~0.12 | 16  | **19.4** | 12.3 | 9.8 |
| 0.25  | 32  | **29.0** | 19.5 | 15.7 |
| 1.00  | 127 | 83.8 | 84.2 | 81.2 |

- `p=1.0` (clean paste-in) relies on the evolving board the **least** across the entire
  trajectory — that is the robust claim. `p=0.2` is highest through the early/mid region; the
  ordering 0.2 > 0.5 > 1.0 holds until ~65% of sampling, after which p=0.2 and p=0.5 cross
  (both still well above p=1.0) and all three converge to ~82–84% (every model reads a
  near-final board, so swapping it flips most cells).
- Robust to checkpoint choice: the earlier 150k-checkpoint / non-EMA runs gave nearly identical
  early/mid separation (step 16 ≈ 25.6 / 13.7 / 11.4), so the mechanism is not an artifact of
  the weight set.

### Accuracy-over-time (does NOT support a "p=1.0 plateaus" story — recorded honestly)

Also tracked in the json (`acc` = x1 per-cell, `zacc` = proposed-board per-cell):

- **x1 per-cell accuracy is flat from step 0** (p=0.2 ~75–77%, p=1.0 ~64–65% throughout): each
  model commits to its guess immediately; p=0.2 is simply a better predictor, not a
  slower-but-better refiner.
- **z-state (proposed board) per-cell accuracy** rises ~linearly 9% → final for *all* models
  (final 76.3 / 68.3 / 64.5%); the p=0.2−p=1.0 gap widens over the back half but **no model
  plateaus early**. So an "accuracy steadily increases / p=1.0 plateaus" figure is *not*
  supported by the data; the board-swap reliance plot is the clean mechanism result.

## Reproduction

```bash
# compute (GPU) -> writes figures/sudoku_incremental_solving_data.json
python figures/sudoku_incremental_solving.py --compute --num-puzzles 512 --batch 256 --steps 128
# render only (cheap, reads the cached json) -> writes the png
python figures/sudoku_incremental_solving.py
```

`data json` schema: `{"meta": {"num_puzzles", "steps"}, "per_p": {"<p>": {"flip", "acc", "zacc", "zsolve", "x1solve": [steps][puzzles]}}}`.

## Caveats / why this design

- **Raw attention is NOT used** because it is confounded by token counts: post-softmax
  attention mass on the ~51 board cells is ≈ 51/81 ≈ 0.62 for *all* p (near-uniform), so it
  cannot distinguish the strategies. The causal board-swap avoids this confound — it changes
  only the board-state *information*, holding token counts and noise level fixed.
- Reliance is a behavioral proxy for "uses the partial board," measured via a single
  within-batch roll (shift 1) as the swap partner.
- Convergence to ~80%+ at the final step is expected and applies to all models (a near-final
  board trivially determines the readout); the discriminating signal is the early/mid ramp.
- Single inference seed (`seed=0`); evaluation-sampling variance is captured by the bootstrap
  bands, but training-seed variance is not (same limitation noted in the paper).

## Rejected alternatives (recorded so we don't re-try them)

- **Attention-to-clue reliance (N-Queens):** flat / non-monotonic across p (~0.07 for all);
  N-Queens clues are too sparse (~2 queens) for an attention-mass story.
- **Inference-time clue-corruption robustness curve:** rejected as ~trivial — "model trained
  with noised clues is robust to noised clues" is close to circular; the only non-trivial part
  (the clean-clue endpoint advantage) is already in the main accuracy/coverage tables.
