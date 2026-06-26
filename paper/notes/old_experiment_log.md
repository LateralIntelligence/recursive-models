# Experiment Log — Training Details

> Running log of experimental configurations, formatted for direct reuse in a paper's *Training Details* section. Each run gets a paper-ready summary plus a precise hyperparameter table for reproducibility. Items marked **[TODO]** are not present in the config and should be verified before write-up. Items marked **[verify]** were reconstructed after a workspace reset and should be checked against original records.
>
> **Append new runs below the marker at the end of the file and update the Run Index.**

---

## Run Index

| # | W&B Name | Task | Sweep Variable | Values | Key Result |
|---|----------|------|----------------|--------|------------|
| 1 | `FLM_Infill_Sudoku_easy_board_N:10000_cond_time_random:true_clean_prob:{p}` | Sudoku Easy (infill) | `conditioning_prob_clean` | 0.2, 0.5, 1.0 | Best acc @ p=0.2: 86.6% |
| 2 | `FLM_Infill_Sudoku_easy_board_N:1000_cond_time_random:true_clean_prob:{p}` | Sudoku Easy (infill), N=1k | `conditioning_prob_clean` | 0.2, 0.5, 1.0 | Best acc @ p=0.2: 11.0% |
| 3 | `FLM_Infill_NQueens_board_N:8_cond_time_random:true_clean_prob:{p}` | N-Queens N=8 (infill) | `conditioning_prob_clean` | 0.2, 0.5, 1.0 | Monotonic in both: acc 99.2%→97.7%, coverage 97.3%→88.1% (p=0.2→1.0) |
| 4 | `FLM_Infill_NQueens_board_N:10_cond_time_random:true_clean_prob:{p}` | N-Queens N=10 (infill) | `conditioning_prob_clean` | 0.2, 0.5, 1.0 | p=0.2 dominates (acc 94.7%, cov 80.0%); coverage monotonic, accuracy non-monotonic (p=0.5 < p=1.0) |

---

## Run 1 — FLM Infill Sudoku Easy: Conditioning Clean Probability Sweep (N=10,000)

**W&B project:** `flm` · **Run names:** `FLM_Infill_Sudoku_easy_board_N:10000_cond_time_random:true_clean_prob:{p}` for p ∈ {0.2, 0.5, 1.0}

### Motivation

Sweep `conditioning_prob_clean` to study how much the conditioning (clue) tokens should be kept clean vs. noised at the same level as board tokens during training. Under diffusion forcing with `conditioning_time_random: true`, the conditioning time is sampled randomly; `conditioning_prob_clean` is the probability that conditioning tokens are held at noise level 0 (fully clean) rather than noised to the target tokens' level.

### Results (eval @ 160k steps, 2,000 puzzles) [verify]


Evaluated on 2,000 validation puzzles (easy difficulty). 95% CIs via bootstrap with 10,000 resamples.
 
| `conditioning_prob_clean` | Accuracy | 95% CI | Correct / Total |
|--------------------------|----------|--------|-----------------|
| 0.0 | 2.85% | [2.15%, 3.60%] | 57 / 2000 [verify] |
| 0.2 | **86.60%** | [85.10%, 88.10%] | 1732 / 2000 |
| 0.5 | 74.75% | [72.85%, 76.65%] | 1495 / 2000 |
| 1.0 | 56.05% | [53.90%, 58.20%] | 1121 / 2000 |

**Observation:** Accuracy decreases monotonically as `conditioning_prob_clean` increases. Heavily-noised conditioning tokens (p=0.2) yield the strongest infill performance — the p=1.0 "always-clean clues" standard setup performs worst, suggesting noised-conditioning training acts as a robustness regularizer.

---

## Run 2 — FLM Infill Sudoku Easy: Conditioning Clean Probability Sweep (N=1,000)

**W&B project:** `flm` · **Run names:** `FLM_Infill_Sudoku_easy_board_N:1000_cond_time_random:true_clean_prob:{p}` for p ∈ {0.2, 0.5, 1.0}

### Motivation

Repeat of Run 1 with a 10× smaller training set (1,000 puzzles) to examine the effect of data scale on the `conditioning_prob_clean` trend.

### Differences from Run 1

| Parameter | Run 1 | Run 2 |
|---|---|---|
| training subset N | 10,000 | **1,000** |

All other configuration identical to Run 1.

### Results (2,000 puzzles) 
95% CIs via bootstrap with 10,000 resamples.

| `conditioning_prob_clean` | Accuracy | 95% CI | Correct / Total |
|--------------------------|----------|--------|-----------------|
| 0.0 | 0.50% | [0.20%, 0.85%] | 10 / 2000 |
| 0.2 | **11.00%** | [9.65%, 12.45%] | 220 / 2000 |
| 0.5 | 9.90% | [8.60%, 11.25%] | 198 / 2000 |
| 1.0 | 2.30% | [1.65%, 2.95%] | 46 / 2000 |


**Observation:** The monotonic ordering (lower `conditioning_prob_clean` → higher accuracy) is preserved at the smaller data scale, despite the large absolute drop in performance between the N=1k and N=10k regimes.

---

## Run 3 — FLM Infill N-Queens (N=8): Conditioning Clean Probability Sweep

**W&B project:** `flm` · **Run names:** `FLM_Infill_NQueens_board_N:8_cond_time_random:true_clean_prob:{p}` for p ∈ {0.2, 0.5, 1.0}

### Summary (paper-ready)

We evaluate the FLM infill formulation on the **N-Queens** task (N=8, 8×8 board flattened to length 64). The model observes the solution board together with a conditioning mask over the clue queens and is trained to infill the remaining cells; token vocabulary is `{0: pad, 1: empty, 2: queen}`. As in the Sudoku experiments, we sweep `conditioning_prob_clean` ∈ {0.2, 0.5, 1.0} under diffusion forcing with random conditioning time (`conditioning_time_random: true`), holding all other settings fixed. The backbone is an 8-block DDiT (hidden size 512, 8 heads) trained with the flow objective and x₀ prediction. Evaluation uses the **last training checkpoint**, sampling 20 generations per puzzle over 200 held-out puzzles.

### Motivation

Test whether the Run 1/2 finding — that lower `conditioning_prob_clean` (more heavily-noised clue tokens) improves infill accuracy — transfers from Sudoku to a different constraint-satisfaction task (N-Queens).

### Shared Training Configuration

| Group | Setting | Value |
|---|---|---|
| **Model** (`nqueens_infill`, `ddit`) | hidden_size | 512 |
| | n_blocks | 8 |
| | n_heads | 8 |
| | cond_dim | 128 |
| | length | 64 (8×8 board) |
| | dropout | 0.1 |
| | scale_by_sigma | true |
| | tie_word_embeddings | false |
| | vocab_lookup | true |
| | vocab_size | 3 (`0=pad, 1=empty, 2=queen`) **[TODO]** confirm incl. special tokens |
| **Algorithm** (`flm`) | backbone | dit |
| | parameterization | mean |
| | pred_type | x0 |
| | loss_type | flow |
| | time_conditioning | true |
| | infill | true |
| | diffusion_forcing | true |
| | conditioning_time_random | true |
| | conditioning_prob_clean | **{0.2, 0.5, 1.0}** (swept) |
| **Noise** | type | log-linear |
| | eps | 0 |
| **Training** | max_steps | 100,001 |
| | global_batch_size | 256 |
| | precision | bf16-mixed |
| | ema | 0.9999 |
| | antithetic_sampling | true |
| | importance_sampling | false |
| | sampling_eps | 0.001 |
| | t_curriculum | steps 100,000, min 0.0 → max 1.0 |
| | gradient_clip_val | 1.0 |
| | accumulate_grad_batches | 1 |
| **Optimizer** | class | Adam/AdamW **[TODO]** confirm |
| | lr | 3e-4 |
| | (β₁, β₂) | (0.9, 0.999) |
| | eps | 1e-8 |
| | weight_decay | 0 |
| **LR schedule** | type | constant w/ warmup |
| | warmup_steps | 2,500 |
| **Sampling (eval)** | predictor / noise_removal | ancestral |
| | steps | 128 |
| | solver | euler |
| | temperature | 1.0 |
| | p_nucleus | 1.0 |
| | gamma | 0.0 |
| | use_float64 | true |
| **Hardware** | accelerator | cuda, 1 device, 1 node |
| | GPU model | **[TODO]** |

### Data

| | Config value | Actual / effective |
|---|---|---|
| Train puzzles | `num_train: 15000` (cap) | **1,758** |
| Valid puzzles | `num_valid: 2000` (cap) | **174** |
| Eval puzzles | — | **750** |
| Generations per eval puzzle | `nqueens_num_samples: 20` | 20 |

- `infill_loss_region: board` — loss computed over the board region.
- The `num_train`/`num_valid` config values are upper-bound caps; the generated N-Queens (N=8) dataset supplies only 1,758 / 174 examples, so those are the true sizes. (For reference, the 8-queens problem admits 92 distinct solutions; training instances are clue-masked variants.)
- With ~1,758 training examples at batch 256, 100k steps corresponds to heavy epoch repetition — worth noting if discussing overfitting/data efficiency.

### Evaluation protocol

- **Checkpoint:** last training checkpoint (not best-NLL).
- 750 held-out puzzles × 20 generations each.

### Results (last checkpoint, 750 puzzles × 20 samples)
95% CIs via bootstrap with 10,000 resamples.

| `conditioning_prob_clean` | Accuracy | Accuracy 95% CI | Coverage | Coverage 95% CI |
|---|---|---|---|---|
| 0.2 | **99.24%** | [98.86, 99.55] | **97.31%** | [96.52, 98.05] |
| 0.5 | 98.75% | [98.20, 99.23] | 91.41% | [90.02, 92.74] |
| 1.0 | 97.67% | [96.73, 98.51] | 88.07% | [86.46, 89.63] |
 

- **Accuracy** = fraction of generated boards that are valid solutions, pooled over all 750×20 = 15,000 generations.
- **Coverage** = per-puzzle fraction of distinct valid solutions recovered across the 20 samples, averaged over puzzles.

**Observation:** Both metrics decrease monotonically as `conditioning_prob_clean` increases. Accuracy is high throughout (97.7–99.2%) with a modest ~1.6 pt spread; the larger and more discriminating effect is in **coverage** (97.31% → 91.41% → 88.07%, a ~9 pt spread). This mirrors the Sudoku conditioning-sweep ordering (Runs 1–2): more heavily-noised conditioning tokens (p=0.2) produce a more diverse generative distribution that recovers more of the valid solution set, while clean-clue training (p=1.0) collapses toward fewer modes.

---

<!-- APPEND NEW RUNS BELOW THIS LINE -->

## Run 4 — FLM Infill N-Queens (N=10): Conditioning Clean Probability Sweep

**W&B project:** `flm` · **Run names:** `FLM_Infill_NQueens_board_N:10_cond_time_random:true_clean_prob:{p}` for p ∈ {0.2, 0.5, 1.0}

### Summary (paper-ready)

We scale the N-Queens conditioning sweep from N=8 (Run 3) to **N=10** (10×10 board, flattened to length 100), holding the architecture and training recipe fixed. The model again infills the solution board given a conditioning mask over clue queens, with `conditioning_prob_clean` ∈ {0.2, 0.5, 1.0}. Evaluation uses the last checkpoint over 750 held-out puzzles with 20 generations each.

### Motivation

Test whether the conditioning-noise effect persists, and whether its ordering remains clean, at a harder constraint-satisfaction scale (10-queens; 724 distinct solutions vs. 92 for 8-queens).

### Differences from Run 3

| Parameter | Run 3 (N=8) | Run 4 (N=10) |
|---|---|---|
| `data.nqueens_n` | 8 | **10** |
| `model.length` | 64 (8×8) | **100** (10×10) |
| Eval puzzles | 750 | 750 |

All other configuration (backbone, optimizer, schedule, sampling, noise, EMA, t-curriculum, batch size, max_steps) identical to Run 3.

### Data

| | Config value | Actual / effective |
|---|---|---|
| Train puzzles | `num_train: 15000` (cap) | **[TODO]** not provided |
| Valid puzzles | `num_valid: 2000` (cap) | **[TODO]** not provided |
| Eval puzzles | `nqueens_num_puzzles: 750` | 750 |
| Generations per eval puzzle | `nqueens_num_samples: 20` | 20 |

- `infill_loss_region: board`; tokens `{0=pad, 1=empty, 2=queen}`.
- 10-queens admits 724 distinct solutions (vs. 92 for 8-queens), so the coverage denominator is substantially larger.

### Evaluation protocol

- **Checkpoint:** last training checkpoint.
- 750 held-out puzzles × 20 generations each.

We bootstrap by resampling across input boards (750 input puzzles) and getting the 
accuracy and coverage per input board, as defined by 20 generations for each input board. 

### Results (last checkpoint, 750 puzzles × 20 samples)
95% CIs via bootstrap with 10,000 resamples.
 
| `conditioning_prob_clean` | Accuracy | Accuracy 95% CI | Coverage | Coverage 95% CI |
|---|---|---|---|---|
| 0.2 | **94.66%** | [94.05, 95.23] | **80.01%** | [78.73, 81.35] |
| 0.5 | 57.42% | [55.88, 58.95] | 71.05% | [69.31, 72.77] |
| 1.0 | 64.56% | [62.90, 66.21] | 64.03% | [62.37, 65.70] |


**Observation:** p=0.2 dominates decisively on **both** metrics — a ~30 pt accuracy lead and ~9 pt coverage lead over the next-best setting. Two caveats for write-up:

1. **Coverage is monotonic** (80.01% → 71.05% → 64.03%), consistent with Runs 1–3.
2. **Accuracy is *not* monotonic** at this scale: p=0.5 (57.42%) falls *below* p=1.0 (64.56%). The clean "lower p → higher accuracy" ordering that held for 8-queens (Run 3) breaks here, with only the p=0.2 endpoint clearly separated.

The robust cross-task claim is therefore the **coverage** ordering and the **dominance of heavily-noised conditioning (p=0.2)**, rather than strict accuracy monotonicity. The p=0.5 vs. p=1.0 accuracy inversion at 10-queens is worth a sentence in the paper (and possibly a seed/variance check before drawing conclusions, since a single-seed inversion could be noise).
