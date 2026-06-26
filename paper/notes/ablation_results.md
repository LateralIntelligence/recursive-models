# FLM Infill — Experiment Log

Persistent log of training details for the Flow Language Model (FLM) infill recipe on combinatorial reasoning tasks (Sudoku, N-Queens). Structured for reuse in the paper's *Training Details* / *Ablations* sections. Uncertainty is flagged explicitly with `[TODO]` (missing) and `[verify]` (reconstructed / unconfirmed).

> **Session note:** The workspace filesystem resets between sessions. Runs 1–5 below are referenced from prior-session summaries and their per-run numeric tables (means, 95% bootstrap CIs, raw counts) must be re-pasted to repopulate. Runs 6–7 are newly logged this session from posted configs/results.

---

## Run Index

| Run | Task | Difficulty / N | Variable swept | Config provided | Results status |
|-----|------|----------------|----------------|-----------------|----------------|
| 1 | Sudoku | easy, N∈{10k, 1k} | `conditioning_prob_clean` ∈ {0.0, 0.2, 0.5, 1.0} | yes (prior) | logged w/ CIs `[verify]` |
| 2 | Sudoku | easy, N=1k | `conditioning_prob_clean` sweep | yes (prior) | logged w/ CIs `[verify]` |
| 3 | N-Queens | N=8 | `conditioning_prob_clean` ∈ {0.2, 0.5, 1.0} | yes (prior) | logged w/ CIs `[verify]` |
| 4 | N-Queens | N=10 | `conditioning_prob_clean` ∈ {0.2, 0.5, 1.0} | yes (prior) | logged w/ CIs `[verify]`; p=0.5 acc inversion flagged |
| 5 | Sudoku | hard, N∈{1k,10k} | `conditioning_prob_clean` ∈ {0.2, 0.5, 1.0} | yes (prior) | logged w/ CIs `[verify]` |
| **6** | **Sudoku + N-Queens** | **hard & easy N=10k; NQ 8 & 10** | **`separate_conditioning_time` = true** | hard only | logged w/ CIs |
| **7** | **Sudoku** | **easy, N=10k** | **response-only loss** (`region=fill`) × `conditioning_prob_clean` ∈ {0.0, 0.2, 0.5, 1.0} | none | logged w/ CIs |
| **8** | **Sudoku** | **easy & hard, N=10k** | **unconditional** (`conditioning_prob_clean=0`) | hard only | logged w/ CIs |

---

## Shared Configuration

Unless overridden in a run's hyperparameter diff, all runs use the following. (Recovered from posted run configs; consistent across Runs 1, 6.)

**Backbone / model** — DDiT (`type: ddit`, `name: small_infill`): hidden_size 512, cond_dim 128, 8 blocks, 8 heads, sequence length 81, dropout 0.1, `scale_by_sigma: true`, `vocab_lookup: true`, `tie_word_embeddings: false`.

**Algorithm** — FLM (`name: flm`, `backbone: dit`), `parameterization: mean`, `infill: true`, `diffusion_forcing: true`, `time_conditioning: true`, `conditioning_time_random: true`. Noise schedule log-linear (`eps: 0`). `learnable_loss_weighting: false`.

**Training objective** — `loss_type: flow`, `pred_type: x0`, EMA 0.9999, antithetic time sampling, `sampling_eps: 0.001`, no importance sampling, no change-of-variables. Loss precision bf16. Timestep curriculum: 100,000 steps, t ∈ [0, 1] (`t_curriculum_min: 0.0`, `t_curriculum_max: 1.0`).

**Optimizer** — `[TODO: confirm class — AdamW vs Adam]`; lr 3e-4, β1 0.9, β2 0.999, eps 1e-8, weight_decay 0. LR schedule: constant with linear warmup, 2500 warmup steps (`transformers.get_constant_schedule_with_warmup`).

**Sampling (eval)** — ancestral predictor + noise removal, 128 steps, Euler solver, γ 0.0, temperature 1.0, p_nucleus 1.0, float64.

**Trainer** — Lightning, CUDA, 1 device / 1 node, DDP (`find_unused_parameters: false`), precision bf16-mixed, gradient_clip_val 1.0, `val_check_interval: 10000`, 2 sanity val steps. Seed 1. `accumulate_grad_batches: 1`.

**Data** — Sudoku tokenizer (`IdentityTokenizer`), `num_valid: 2000`, no EOS insertion, not iterable, not streaming, `subset_seed: 0`.

**GPU** — `[TODO: GPU model]`.

---

## Run 6 — Separate conditioning time ablation

**Hypothesis / change.** Decouple the noise level applied to the conditioning (prompt/clue) tokens from that applied to the response tokens. Instead of a single sampled time `t`, sample two times independently: `t, t' ~ iid`; noise the **response** with `t` and the **conditioning** with `t'`. Implemented via `separate_conditioning_time: true` (with `conditioning_time_random: true`).

**Results (validation accuracy / coverage; 95% bootstrap CI, 10k resamples).** Sudoku n=2000, N-Queens n=750.

| Task | Setting | Metric | Score | 95% CI |
|------|---------|--------|-------|--------|
| Sudoku easy | region=board, cprob=1.0 | accuracy | 0.9120 | [0.8995, 0.9245] |
| Sudoku hard | region=board, cprob=1.0 | accuracy | 0.1690 | [0.1525, 0.1855] |
| N-Queens 8 | — | accuracy | 0.9381 | [0.9256, 0.9497] |
| N-Queens 8 | — | coverage | 0.9349 | [0.9226, 0.9467] |
| N-Queens 10 | — | accuracy | 0.5945 | [0.5781, 0.6103] |
| N-Queens 10 | — | coverage | 0.6682 | [0.6517, 0.6855] |

> **CI-vs-prior-estimate discrepancies (flagged):**
> - **Sudoku hard:** earlier logged *last*-checkpoint point estimate was **0.219**; the proper 10k-bootstrap eval gives **0.1690** (CI excludes 0.219). The 0.219 was a last-checkpoint single eval — superseded by the CI value here.
> - **Sudoku easy:** earlier 0.912 ≈ CI 0.9120 — consistent.

**Hyperparameter diff vs Shared Config.**

- `algo.separate_conditioning_time: true` (vs `false` baseline)
- `algo.conditioning_prob_clean: 1.0` (hard run)
- `data.difficulty: hard`, `data.infill_loss_region: board` (hard run)
- `loader.global_batch_size: 256` (hard run)
- `trainer.max_steps: 40001` (hard run); checkpoint every 10k steps
- wandb id `8eb2460b`; run `FLM_Infill_Sudoku_hard_region:board_septime:true_N:10000_cond_time_random:true_clean_prob:1.0`

**`[verify]` — easy / N-Queens run configs.** The config JSON was provided only for the **hard** Sudoku run. The Sudoku-easy and the two N-Queens runs' `global_batch_size`, `max_steps`, region, and `conditioning_prob_clean` are not confirmed; assume same recipe pending those config dumps.

**Observations.**
- Under separate conditioning time, Sudoku easy stays strong (0.9120) while hard drops to 0.1690 — large easy/hard gap, as expected for this recipe.
- N-Queens: coverage > accuracy at 10×10 (0.6682 vs 0.5945, CI-separated), while at 8×8 the two are statistically indistinguishable (overlapping CIs, 0.9381 vs 0.9349) — consistent with the broader pattern that coverage and accuracy dissociate more at larger board sizes.

---

## Run 7 — Response-only loss ablation

**Hypothesis / change.** Compute the training loss **only on the response tokens** (the fill region), excluding the prompt/clue tokens — realized as `infill_loss_region: fill` (confirmed; vs `board`). Sweep `conditioning_prob_clean` to see how loss masking interacts with how often the conditioning is kept clean.

**`[verify]` — config.** Full config JSON not posted; `global_batch_size`, `max_steps`, and checkpoint selection unconfirmed. Region confirmed as `fill` (artifact `ablation2-fillonly`). Task Sudoku easy, N=10k.

**Results (validation accuracy, Sudoku easy, region=fill; 95% bootstrap CI, 10k resamples, n=2000).**

| `conditioning_prob_clean` | Accuracy | 95% CI |
|---------------------------|----------|--------|
| 0.0 | 0.0265 | [0.0195, 0.0335] |
| 0.2 | 0.8375 | [0.8215, 0.8540] |
| 0.5 | 0.3290 | [0.3090, 0.3500] |
| 1.0 | 0.3900 | [0.3685, 0.4115] |

> **CI-vs-prior-estimate discrepancies (flagged) — substantial:**
> - p=0.2: prior **0.854** → CI **0.8375** (CI excludes 0.854).
> - p=0.5: prior **0.4565** → CI **0.3290** (large gap, CI excludes 0.4565).
> - p=1.0: prior **0.3545** → CI **0.3900** (CI excludes 0.3545).
> - p=0.0 (0.0265) is newly added.
> The earlier point estimates were preliminary single evals; the 10k-bootstrap values here supersede them.

**Observations.**
- **Ordering changed under the proper eval.** With CIs, the ranking is p=0.2 (0.8375) ≫ p=1.0 (0.3900) > p=0.5 (0.3290) > p=0.0 (0.0265). The earlier point estimates implied a *monotone* decline (0.2 > 0.5 > 1.0); the corrected numbers instead show a **non-monotone dip at p=0.5**, with p=1.0 recovering above p=0.5 (CIs for 0.5 and 1.0 are separated: [0.3090,0.3500] vs [0.3685,0.4115]).
- The non-monotonicity (a U-ish shape across p=0.2 → 0.5 → 1.0) is notable and unlike the monotone coverage ordering of the standard recipe. Treat as a candidate interaction between `region=fill` loss masking and `conditioning_prob_clean`; warrants a multi-seed re-run before any causal claim.
- p=0.2 remains the clear best by a wide margin (~0.45–0.5 absolute above the other positive-p settings).

---

## Run 8 — Unconditional training ablation (`conditioning_prob_clean = 0`)

**Hypothesis / change.** Train **unconditionally** in the sense that the conditioning (clue) tokens are *never* kept clean — `conditioning_prob_clean: 0`, so the clues are always noised at the same sampled time `t` as the response. The model thus never sees clean clues during training. Note this is realized **not** by `infill: false`: `infill: true`, `diffusion_forcing: true`, `separate_conditioning_time: false`, `region: board` are all retained. Serves as the zero-clean-conditioning floor — it isolates how much accuracy depends on ever exposing the model to clean clues.

**Results (validation accuracy, N=10k; 95% bootstrap CI, 10k resamples, n=2000).**

| Difficulty | Accuracy | 95% CI |
|------------|----------|--------|
| easy | 0.0165 | [0.0110, 0.0225] |
| hard | 0.0000 | [0.0000, 0.0000] |

> **CI-vs-prior-estimate (flagged):** easy prior **0.0175** → CI **0.0165** (within CI band, minor). Hard 0.0 confirmed exactly.

**Hyperparameter diff vs Shared Config (hard run, confirmed from config).**

- `algo.conditioning_prob_clean: 0` (vs swept values in baseline)
- `algo.separate_conditioning_time: false`
- `data.difficulty: hard`, `data.infill_loss_region: board`
- `loader.global_batch_size: 256`
- `trainer.max_steps: 40001`; checkpoint every 10k steps
- wandb id `43d3c27a`; artifact `ablation1-zerocond`; run `FLM_Infill_Sudoku_hard_region:board_septime:false_N:10000_cond_time_random:true_clean_prob:0`

**`[verify]` — easy run config.** Config JSON provided for the **hard** run only; easy run's batch/steps/region assumed identical with `difficulty: easy`.

**Observations.**
- Near-floor accuracy (easy 0.0165, hard 0.0), as expected: with clues always noised the model effectively learns the unconditional board distribution and can't exploit the given cells, so exact-match accuracy collapses. Hard at exactly 0.0 means no sampled board matched.
- This is the `conditioning_prob_clean = 0` endpoint of the sweep, so it also anchors the low end of the coverage/accuracy monotonicity story logged for Runs 1–5 — useful to cite as the floor against which p > 0 runs show their lift.
- `[verify]` checkpoint/protocol; treat as a sanity floor rather than a tuned comparison.



- `[TODO]` Optimizer class (AdamW vs Adam) and GPU model — global.
- `[verify]` Runs 1–5 numeric tables (means, 95% bootstrap CIs over 10k resamples, raw counts) — re-paste to repopulate after session reset.
- `[verify]` Run 6 easy-Sudoku and N-Queens 8/10 run configs (batch / steps / region / clean_prob).
- `[verify]` Run 7 full config (batch/steps); region confirmed `fill`.
- `[verify]` Run 8 easy-run config (batch/steps/region) — hard confirmed (`clean_prob:0`, batch 256, 40k steps, region board).
- Runs 6–8 now carry 10k-bootstrap CIs (n=2000 Sudoku / 750 N-Queens). Prior point estimates superseded; discrepancies flagged inline.
- Multi-seed re-run for **Run 7** to confirm the non-monotone p=0.5 dip (CI-separated from p=1.0) under `region=fill`.
- N-Queens entries for these two ablations — pending (user to add next).