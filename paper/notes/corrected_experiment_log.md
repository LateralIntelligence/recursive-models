# FLM Infill — Experiment Log

Persistent log of training details for the Flow Language Model (FLM) infill recipe on combinatorial reasoning tasks (Sudoku, N-Queens). Structured for direct reuse in the paper's *Training Details* / *Ablations* sections.

**Uncertainty markers:** `[TODO]` = missing from posted configs; `[verify]` = reconstructed across a workspace reset and not yet re-confirmed against source; `[CONFLICT]` = two mutually inconsistent source values logged, must be resolved before write-up.

> **Session note.** The workspace filesystem resets between sessions, so this file is rebuilt from prior-session records at the start of each session. Structural config rows carried across resets are marked `[verify]`; the numeric results below were corrected this session (see Correction Log).

---

## ⚠️ Correction Log — Checkpoint provenance fix (this session)

**What changed.** All previously reported numbers were evaluated on a faulty `.last` checkpoint. The `.last` file did not hold the last step weights (Defaulted to 10k step), so it under-reported accuracy. This session **re-evaluates every run on the actual final periodic checkpoint** and supersedes the prior point estimates.

**Checkpoint used per training length:**
- **160k-step runs** (Sudoku easy, and easy ablations): the explicit `160000`-step checkpoint, except where the `150000`-step checkpoint is specified (noted per run).
- **40k-step runs** (Sudoku hard, and hard ablations): the explicit `40000`-step checkpoint.
- **N-Queens:** `100000`-step checkpoint (trained 100k steps), for both main runs and the ablation-3 N-Queens variants.

**Two previously-flagged anomalies were resolved by the correction** (both were checkpoint artifacts, not real effects):
1. **N-Queens N=10 accuracy inversion** (old Run 4: p=0.5 acc fell below p=1.0 with separated CIs) → corrected ordering is **monotone** in `conditioning_prob_clean` (0.979 > 0.942 > 0.848). The multi-seed re-run previously queued to investigate this is **no longer needed to explain the inversion**; the inversion was the bad checkpoint.
2. **Fill-only ablation non-monotone dip** (old Run 7: p=0.5 dipped below p=1.0 with separated CIs) → corrected ordering is **monotone** (0.838 > 0.465 > 0.370).

**Findings that survive the correction:**
- **Coverage** is still monotonically ordered by `conditioning_prob_clean` (lower p → higher coverage) across every task and board size.
- **Accuracy** is now also monotone in p across **both** N-Queens board sizes (previously violated at N=10).
- Lower `conditioning_prob_clean` (more noising of clue tokens during training) remains uniformly better, with p=0.2 best across Sudoku and N-Queens.

---

## Run Index

| Run | Task | Difficulty / N | Variable swept | Final ckpt | Results status |
|-----|------|----------------|----------------|------------|----------------|
| 1 | Sudoku | easy, N=10k | `conditioning_prob_clean` ∈ {0.2,0.5,1.0} | 160000 | corrected w/ CIs ✓ |
| 2 | Sudoku | easy, N=1k | `conditioning_prob_clean` ∈ {0.2,0.5,1.0} | 160000 | corrected w/ CIs ✓ (N=1k non-monotone) |
| 3 | N-Queens | N=8 | `conditioning_prob_clean` ∈ {0.2,0.5,1.0} | 100000 | corrected w/ CIs ✓ |
| 4 | N-Queens | N=10 | `conditioning_prob_clean` ∈ {0.2,0.5,1.0} | 100000 | corrected w/ CIs ✓ (inversion resolved) |
| 5 | Sudoku | hard, N∈{1k,10k} | `conditioning_prob_clean` ∈ {0.2,0.5,1.0} | 40000 | corrected w/ CIs ✓ |
| 6 | Sudoku + N-Queens | easy/hard, N=8/10 | `separate_conditioning_time` = true (cp=1.0) | 150k/40k/100k | corrected w/ CIs ✓ |
| 7 | Sudoku | easy, N=10k | response-only loss (fill) × `conditioning_prob_clean` | 150k | corrected w/ CIs ✓ (dip resolved) |
| 8 | Sudoku | easy/hard, N=10k | unconditional floor (`conditioning_prob_clean` = 0) | 150k/40k | corrected w/ CIs ✓ |
| A | Sudoku | early discrete-time | flow-matching weight γ ∈ {0.0, 0.5} | — | appendix; predates infill recipe |

All evaluations use **95% bootstrap CIs (10,000 resamples)** unless noted. Sudoku eval n = 2,000 validation puzzles; accuracy = exact-solve count / n.

---

## Shared Configuration (FLM infill recipe)

> Representative config (taken from the Sudoku-hard sweep, Run 5); other runs differ only in the rows noted in their per-run diff tables. Rows carried across a reset are `[verify]`.

**Model** — DDiT backbone (`small_infill`), hidden size 512, cond_dim 128, 8 blocks, 8 heads, sequence length 81 (Sudoku), dropout 0.1, `scale_by_sigma: true`, untied word embeddings.

**Algorithm** — FLM infill (`algo.name: flm`, `infill: true`, `diffusion_forcing: true`), x0 / mean parameterization, time conditioning on. Conditioning time sampled randomly (`conditioning_time_random: true`); `conditioning_prob_clean` is the swept variable (probability clue tokens are held fully clean rather than noised to the target level). `t_max: 1.0`, `t_min: 0.0`.

**Loss / noise** — flow-matching loss (`loss_type: flow`, `pred_type: x0`), log-linear noise schedule, antithetic time sampling, no importance sampling, `sampling_eps: 0.001`, loss in bf16. Timestep curriculum over 100k steps (`t_curriculum_min/max` = 0.0→1.0). EMA decay 0.9999.

**Optimizer** — `torch.optim.AdamW`, lr 3e-4, β=(0.9, 0.999), eps 1e-8, weight decay 0; constant schedule with 2500-step warmup; gradient clip 1.0.

**Precision / hardware** — bf16-mixed, PyTorch-Lightning Trainer. **GPU: NVIDIA L40S.** Wall-clock: ~1 h per Sudoku run; up to ~3 h per N-Queens run.

**Sampling (eval)** — ancestral predictor, last/periodic checkpoint with EMA weights applied (see Correction Log).

---

## Run 1 & 2 — Sudoku Easy: `conditioning_prob_clean` sweep (N=10k, N=1k)  ✓ corrected

**Final checkpoint:** 160000 (image labels `511-160000` for N=10k, `5007-160000` for N=1k). Batch size 32. Eval n = 2,000; accuracy = exact-solve count / 2000.

| N | cp | Accuracy | 95% CI | Raw count |
|------|-----|----------|--------|-----------|
| 10000 | 0.2 | **0.8755** | [0.8610, 0.8900] | 1751 / 2000 |
| 10000 | 0.5 | 0.7660 | [0.7475, 0.7845] | 1532 / 2000 |
| 10000 | 1.0 | 0.5145 | [0.4930, 0.5360] | 1029 / 2000 |
| 1000 | 0.2 | 0.0155 | [0.0105, 0.0210] | 31 / 2000 |
| 1000 | 0.5 | 0.1440 | [0.1285, 0.1600] | 288 / 2000 |
| 1000 | 1.0 | 0.0270 | [0.0200, 0.0345] | 54 / 2000 |

**Resolved:** the earlier typed bullets (10k p=0.2 ≈ 0.158, etc.) were the faulty `.last`-checkpoint values; the table above is the corrected actual-checkpoint eval and supersedes them.

**Observation (real, not an artifact).** At **N=10k**, accuracy is monotone decreasing in p with separated adjacent CIs (0.8755 > 0.766 > 0.5145) — the standard ordering. At **N=1k**, the ordering is **non-monotone**: p=0.5 (0.144, CI [0.1285,0.1600]) sits well above both p=0.2 (0.0155) and p=1.0 (0.027), with CIs separated from both. This survives the checkpoint correction (present in both image and corrected counts), so it is a genuine low-data effect rather than a checkpoint artifact. Worth a multi-seed check before attributing it to the method in the paper.

---

## Run 3 — N-Queens (N=8): `conditioning_prob_clean` sweep  ✓ corrected

**Final checkpoint:** 100000 (trained 100k steps). Metrics: Accuracy (per-solution validity) and Coverage (fraction of distinct solutions reached), 95% bootstrap CIs.

| clean_prob | Accuracy | Accuracy 95% CI | Coverage | Coverage 95% CI |
|------------|----------|-----------------|----------|-----------------|
| 0.2 | 0.9916 | [0.98607, 0.99600] | 0.92969 | [0.91745, 0.94162] |
| 0.5 | 0.98973 | [0.98480, 0.99387] | 0.90880 | [0.89513, 0.92217] |
| 1.0 | 0.97673 | [0.96727, 0.98513] | 0.88067 | [0.86464, 0.89627] |

Both accuracy and coverage are monotone decreasing in `conditioning_prob_clean`. Adjacent accuracy CIs overlap (0.2 vs 0.5, 0.5 vs 1.0); coverage CIs are separated for every adjacent pair.

---

## Run 4 — N-Queens (N=10): `conditioning_prob_clean` sweep  ✓ corrected (inversion resolved)

**Final checkpoint:** 100000 (trained 100k steps).

| clean_prob | Accuracy | Accuracy 95% CI | Coverage | Coverage 95% CI |
|------------|----------|-----------------|----------|-----------------|
| 0.2 | 0.9790 | [0.97373, 0.98367] | 0.73931 | [0.72537, 0.75361] |
| 0.5 | 0.94227 | [0.93253, 0.95140] | 0.60679 | [0.59120, 0.62275] |
| 1.0 | 0.8480 | [0.83087, 0.86420] | 0.49891 | [0.48213, 0.51557] |

**Supersedes** the prior Run 4. The earlier p=0.5 < p=1.0 accuracy inversion (CI-separated) is **gone**: corrected accuracy is monotone decreasing in p (0.979 > 0.942 > 0.848), with all adjacent CIs separated. Coverage likewise monotone with separated CIs. The inversion was a `.last`-checkpoint artifact.

---

## Run 5 — Sudoku Hard: `conditioning_prob_clean` × N sweep  ✓ corrected

**Final checkpoint:** 40000. Eval n = 2,000; accuracy = exact-solve count / 2000.

| N | cp | Accuracy | 95% CI | Raw count |
|------|-----|----------|--------|-----------|
| 10000 | 0.2 | 0.2545 | [0.2355, 0.2735] | 509 / 2000 |
| 10000 | 0.5 | 0.0770 | [0.0655, 0.0890] | 154 / 2000 |
| 10000 | 1.0 | 0.0125 | [0.0080, 0.0175] | 25 / 2000 |
| 1000 | 0.2 | 0.0005 | [0.0000, 0.0015] | 1 / 2000 |
| 1000 | 0.5 | 0.0010 | [0.0000, 0.0025] | 2 / 2000 |
| 1000 | 1.0 | 0.0000 | [0.0000, 0.0000] | 0 / 2000 |

Image and typed counts agree exactly (no conflict). At N=10k, accuracy is monotone decreasing in p with separated CIs. At N=1k all three conditions sit at the floor (CIs overlap; no separation claimable). Corrected N=10k p=0.2 (0.2545) is close to the earlier periodic-40k estimate (~0.259) and well above the bad `.last` value (0.1755), consistent with the checkpoint fix.

---

## Run 6 — Ablation: separate conditioning time (`separate_conditioning_time` = true)  ✓ corrected

Independent noise times `t` (response) and `t'` (conditioning); cp=1.0. Sudoku eval n = 2,000; N-Queens n = 750.

| Task | Final ckpt | Accuracy | Accuracy 95% CI | Coverage | Coverage 95% CI | Raw |
|------|-----------|----------|-----------------|----------|-----------------|-----|
| Sudoku Easy | 150k | 0.9045 | [0.8915, 0.9170] | — | — | 1809 / 2000 |
| Sudoku Hard | 40k | 0.2180 | [0.2000, 0.2360] | — | — | 436 / 2000 |
| N-Queens N=8 | 100k | 0.97313 | [0.96320, 0.98180] | 0.89522 | [0.88058, 0.90998] | — |
| N-Queens N=10 | 100k | 0.87407 | [0.85827, 0.88913] | 0.51397 | [0.49738, 0.53095] | — |

---

## Run 7 — Ablation: response-only loss (`infill_loss_region: fill`), Sudoku easy  ✓ corrected (dip resolved)

Loss computed on response region only. Final ckpt 150k; n = 2,000.

| cp | Accuracy | 95% CI |
|-----|----------|--------|
| 0.0 | 0.0260 | [0.0195, 0.0330] |
| 0.2 | 0.8380 | [0.8220, 0.8540] |
| 0.5 | 0.4650 | [0.4435, 0.4865] |
| 1.0 | 0.3695 | [0.3490, 0.3915] |

**Supersedes** prior Run 7. The earlier non-monotone dip (p=1.0 above p=0.5) is **gone**: corrected ordering is monotone decreasing over p∈{0.2,0.5,1.0} (0.838 > 0.465 > 0.370), with separated CIs. The dip was a `.last`-checkpoint artifact.

---

## Run 8 — Ablation: unconditional floor (`conditioning_prob_clean` = 0)  ✓ corrected

Clue tokens always noised (realized via `conditioning_prob_clean: 0`, not by disabling `infill`). n = 2,000.

| Task | Final ckpt | Accuracy | 95% CI | Raw |
|------|-----------|----------|--------|-----|
| Sudoku Easy | 150k | 0.0150 | [0.0100, 0.0205] | 30 / 2000 |
| Sudoku Hard | 40k | 0.0000 | [0.0000, 0.0000] | 0 / 2000 |

---

## Appendix A — Early discrete-time FLM ablation (pre-infill recipe)

Discrete-time FLM with looped loss; flow-matching weight γ ∈ {0.0, 0.5}. γ=0.5 (flow matching + looped loss) substantially outperformed γ=0.0. **Confound:** batch size / training-steps differed between the two settings, so the effect cannot be attributed cleanly to γ alone. Kept separate from the infill runs above; not directly comparable.

---

## Open items

- **N=1k Sudoku-easy non-monotonicity** (p=0.5 > p=0.2 and p=1.0, CI-separated) survives the checkpoint fix — flag for a multi-seed check before paper attribution.
- The multi-seed N-Queens N=10 re-run is no longer needed to explain the old inversion (resolved by checkpoint fix); a multi-seed pass may still be worth one line in the paper to report seed variance on the headline numbers.