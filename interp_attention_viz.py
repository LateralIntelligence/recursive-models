#!/usr/bin/env python
"""
interp_attention_viz.py

Interpretability harness for the FLM sudoku models.

What it does
------------
1. Loads a trained FLM checkpoint (and the config it was trained with).
2. Loads the generated sudoku dataset (``--split train`` or ``--split val``)
   and pulls a single deterministic batch.
3. Runs the FLM ``conditional_generate`` Euler-ODE rollout for a configurable
   number of sampling steps. At each (selected) timestep it captures every
   block's self-attention matrix AND the model's x0 (clean-data) argmax
   prediction.
4. Renders a figure: for each selected timestep (one column) it stacks the
   per-layer attention matrices, then -- in a separate row underneath -- draws
   the model's predicted sudoku board.

Attention permutation
----------------------
Within each attention heatmap the token indices are permuted so that all the
*conditioning* tokens (the given clue cells, which are clamped clean at time
level 1) come first (the leading ``k`` rows/cols) and the remaining ``N - k``
non-conditioning tokens come after. The same permutation is applied to both
rows and columns, so the result is still a valid attention matrix -- just
re-ordered. A divider is drawn at index ``k`` so the conditioning vs.
non-conditioning blocks are visually obvious. The point is to read off how much
attention mass the model routes to the conditioning tokens.

Board readout
-------------
Beneath the attention grid each column shows the model's x0 argmax prediction
as a 9x9 board:
  * given clue cells (conditioning)            -> blue
  * correctly predicted cells                  -> green
  * incorrectly predicted cells                -> red

The figure is written to ``<repo-root>/figures/<--out>``.

Example
-------
    python interp_attention_viz.py \
        --checkpoint outputs/sudoku/.../checkpoints/last.ckpt \
        --split val --num-steps 16 --example-idx 0 \
        --out attn_example0.png

This file relies on flash_attn / CUDA, exactly like the training stack, so it
must be run on a GPU box where the FLM dependencies are installed.
"""
import argparse
import functools
import os

import numpy as np
import torch
import torch.nn.functional as F

torch.load = functools.partial(torch.load, weights_only=False)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import omegaconf
from omegaconf import OmegaConf
import lightning as L
import logging 

import algo
import dataloader
import utils
import models.dit as dit


# ----------------------------------------------------------------------------- #
# OmegaConf resolvers / safe globals (mirror main.py so cfg + ckpt load cleanly)
# ----------------------------------------------------------------------------- #
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.base.Metadata,
])
for _name, _fn in [
    ("cwd", os.getcwd),
    ("device_count", torch.cuda.device_count),
    ("eval", eval),
    ("div_up", lambda x, y: (x + y - 1) // y),
]:
    if not OmegaConf.has_resolver(_name):
        OmegaConf.register_new_resolver(_name, _fn)


ALGO_BY_NAME = {
    "flm": algo.FLM,
    "discrete_loop_flm": algo.DiscreteLoopFLM,
    "discrete_recurrent_flm": algo.DiscreteRecurrentFLM,
    "cond_uncond_loop_flm": algo.CondUncondLoopFLM,
}

# A cell value convention shared across the sudoku pipeline (value_offset=1):
#   pad -> 0, blank cell -> 1, digits 1..9 -> 2..10  (display digit = token - 1)
BLANK_ID = 1


# ----------------------------------------------------------------------------- #
# config loading
# ----------------------------------------------------------------------------- #
def load_run_config(ckpt_path):
    """Return the OmegaConf a run was trained with.

    Prefers a sibling ``.hydra/config.yaml`` (cheap), then the config pickled
    into the checkpoint's ``hyper_parameters``.
    """
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for cand in (
        os.path.join(d, ".hydra", "config.yaml"),
        os.path.join(d, "..", ".hydra", "config.yaml"),
        os.path.join(d, "..", "..", ".hydra", "config.yaml"),
    ):
        if os.path.exists(cand):
            return OmegaConf.load(cand)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {}) or {}
    cfg = hp.get("config", hp)
    if isinstance(cfg, (dict, omegaconf.DictConfig)) and len(cfg):
        return OmegaConf.create(cfg)
    raise SystemExit(f"could not recover a config for {ckpt_path}")


def model_class(cfg):
    name = OmegaConf.select(cfg, "algo.name")
    if name not in ALGO_BY_NAME:
        raise ValueError(f"unknown algo {name!r}; expected one of {list(ALGO_BY_NAME)}")
    return ALGO_BY_NAME[name]

# ----------------------------------------------------------------------------- #
# attention capture
# ----------------------------------------------------------------------------- #
class AttentionCapture:
    """Monkeypatch ``DDiTBlock.custom_sdpa`` so each block records its softmax
    attention probabilities into an ordered list.

    The blocks run sequentially inside one backbone forward, so appends arrive
    in layer order. Set ``enabled`` to gate storage (we only want to keep the
    timesteps we plan to visualise). ``forward(..., use_jvp_attn=True)`` routes
    the model through ``custom_sdpa`` instead of flash-attn.
    """

    def __init__(self):
        self.enabled = False
        self.store = []
        self._orig = None

    def __enter__(self):
        cap = self
        self._orig = dit.DDiTBlock.custom_sdpa

        def patched(self, q, k, v, softcap=-1.0):
            # q,k,v: (B, H, S, D). Mirror the original custom_sdpa math exactly.
            B, H, S, D = q.shape
            qs = q / (D ** 0.5)
            attn_weights = torch.einsum("bhid,bhjd->bhij", qs, k)
            if softcap > 0.0:
                attn_weights = softcap * torch.tanh(attn_weights / softcap)
            attn_probs = torch.softmax(attn_weights, dim=-1)
            if cap.enabled:
                cap.store.append(attn_probs.detach().float().cpu())
            return torch.einsum("bhij,bhjd->bhid", attn_probs, v)

        dit.DDiTBlock.custom_sdpa = patched
        return self

    def __exit__(self, *exc):
        dit.DDiTBlock.custom_sdpa = self._orig

    def reset(self):
        self.store = []

    def stacked(self):
        """(n_layers, B, H, S, S) tensor of everything captured this step."""
        return torch.stack(self.store, dim=0)


# ----------------------------------------------------------------------------- #
# generation with capture (re-implements FLM.conditional_generate_samples so we
# can request use_jvp_attn=True and snapshot per-step attention + x0 prediction)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def conditional_generate_with_capture(model, x_input, conditioning_mask,
                                      capture, capture_steps,
                                      example_idx, num_steps, eps=1e-5):
    """Run the FLM Euler-ODE rollout, capturing attention + x0 predictions.

    Returns:
        final_pred:  (B, L) argmax over the final z (the generated solution).
        snapshots:   dict keyed by step index -> {
                         "attn":   (n_layers, S, S) head-averaged attention for
                                   ``example_idx`` at that step,
                         "x0":     (L,) argmax of the model's clean prediction
                                   for ``example_idx`` at that step,
                     }
    """
    device = model.device
    B, L = x_input.shape
    V = model.vocab_size
    x_input = x_input.to(device)
    conditioning_mask = conditioning_mask.to(device).bool()

    x_input_onehot = F.one_hot(x_input, num_classes=V)
    tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    # z at t=0: pure noise with conditioning tokens clamped to their clean value.
    z, _ = model.corrupt_continuous(
        x_input, t=torch.tensor([0.0], device=device),
        conditioning_mask=conditioning_mask)
    z = z.to(device).to(model.dtype)

    snapshots = {}
    capture_steps = set(capture_steps)
    n_blocks = len(model.backbone.blocks)
    attn_sum = None 

    for i in range(num_steps):
        tau_t_in = tau_vals[i].expand(B)
        t_in = model._tau_to_t(tau_t_in)
        dt = model._tau_to_t(tau_vals[i + 1].expand(B)) - t_in
        df_mask = conditioning_mask if model.diffusion_forcing else None

        want = i in capture_steps
        capture.enabled = True
        capture.reset()
        x_1_pred = model.forward(z, tau_t_in, conditioning_mask=df_mask,
                                 use_jvp_attn=True)
        capture.enabled = False

        assert len(capture.store) == n_blocks, (
                f"captured {len(capture.store)} attention matrices but the model "
                f"has {n_blocks} blocks -- attention capture is not wired up")
        attn = capture.stacked()            # (n_layers, B, H, S, S)
        attn = attn[:, example_idx]         # (n_layers, H, S, S)
        attn = attn.mean(dim=1).numpy()             # head-average -> (n_layers, S, S)
        attn_sum = attn if attn_sum is None else attn_sum + attn 
        if i in capture_steps:
            snapshots[i] = {
                'attn' : attn,
                'x0' : x_1_pred[example_idx].argmax(dim=-1).cpu().numpy(),
                't': float(t_in[0].item())
            }


        x_1_pred_probs = x_1_pred.exp()
        if i == num_steps - 1:
            z = x_1_pred_probs
            z = torch.where(conditioning_mask.unsqueeze(-1), x_input_onehot, z)
            break
        v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + eps)
        z = z + dt.view(-1, 1, 1) * v
        z = torch.where(conditioning_mask.unsqueeze(-1), x_input_onehot, z)

    mean_attn = attn_sum / num_steps
    return z.argmax(dim=-1), snapshots, mean_attn

# ----------------------------------------------------------------------------- #
# board decoding + rendering
# ----------------------------------------------------------------------------- #
HINT_RGB = (0.80, 0.86, 1.00)     # blue   -- given clue cells (conditioning)
RIGHT_RGB = (0.74, 0.93, 0.74)    # green  -- correctly predicted
WRONG_RGB = (0.97, 0.72, 0.72)    # red    -- incorrectly predicted

def report_conditioning_attention(mean_attn, cond_mask, logger):
    """Print, per layer, the mean post-softmax attention weight placed on
    conditioning columns vs. non-conditioning columns.

    ``mean_attn`` is (n_layers, S, S), head-averaged and averaged over all
    sampling steps; entry [l, i, j] is how much query token i attends to key
    token j in layer l. For every query row we select the conditioning columns
    (via ``cond_mask``) and average those weights -- so this is the mean
    attention received *per conditioning token*, directly comparable to the mean
    received *per non-conditioning token* (a ratio > 1 means the model routes
    proportionally more attention onto each conditioning/clue token).

    Returns a list of (layer, cond_mean, noncond_mean, ratio) for reuse.
    """
    cond = np.asarray(cond_mask).astype(bool)
    noncond = ~cond
    n_layers = mean_attn.shape[0]

    rows = []
    logger.info("Mean post-softmax attention weight per key token "
                "(head-averaged, averaged over all sampling steps):")
    logger.info("  %-6s %14s %14s %8s", "layer", "conditioning",
                "non-cond", "ratio")
    for l in range(n_layers):
        a = mean_attn[l]                      # (S, S)
        cond_mean = float(a[:, cond].mean())
        noncond_mean = float(a[:, noncond].mean())
        ratio = cond_mean / max(noncond_mean, 1e-12)
        rows.append((l, cond_mean, noncond_mean, ratio))
        logger.info("  %-6d %14.6f %14.6f %8.2f", l, cond_mean, noncond_mean, ratio)

    overall_cond = float(mean_attn[:, :, cond].mean())
    overall_noncond = float(mean_attn[:, :, noncond].mean())
    logger.info("  %-6s %14.6f %14.6f %8.2f", "all",
                overall_cond, overall_noncond,
                overall_cond / max(overall_noncond, 1e-12))
    return rows


def decode_board(tokens, cond_mask, pred, gt):
    """Build the 9x9 board arrays for one example.

    Args:
        tokens: (L,) the example token ids (solution grid in infill mode, or
                ``[puzzle | solution]`` otherwise).
        cond_mask: (L,) bool conditioning mask aligned with ``tokens``.
        pred: (L,) the model's x0 argmax prediction.
        gt:  (L,) the ground-truth token ids (== ``tokens``).

    Returns:
        digits: (9, 9) int digit to display in each cell.
        kind:   (9, 9) int in {0: hint, 1: correct, 2: wrong}.
    """
    tokens = np.asarray(tokens)
    cond_mask = np.asarray(cond_mask).astype(bool)
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    L = tokens.shape[0]

    if L == 81:
        sol_gt, sol_pred, sol_cond = gt, pred, cond_mask
    elif L == 162:
        half = 81
        sol_gt, sol_pred = gt[half:], pred[half:]
        # A solution cell is a "hint" iff its aligned puzzle clue is a real digit.
        puzzle = tokens[:half]
        sol_cond = puzzle > BLANK_ID
    else:
        raise ValueError(f"unexpected sequence length {L}; expected 81 or 162")

    digits = np.zeros((9, 9), dtype=int)
    kind = np.zeros((9, 9), dtype=int)
    for c in range(81):
        r, col = divmod(c, 9)
        if sol_cond[c]:
            digits[r, col] = int(sol_gt[c]) - 1
            kind[r, col] = 0
        else:
            digits[r, col] = int(sol_pred[c]) - 1
            kind[r, col] = 1 if sol_pred[c] == sol_gt[c] else 2
    return digits, kind


def draw_board(ax, digits, kind):
    rgb = np.zeros((9, 9, 3))
    palette = {0: HINT_RGB, 1: RIGHT_RGB, 2: WRONG_RGB}
    for r in range(9):
        for c in range(9):
            rgb[r, c] = palette[kind[r, c]]
    ax.imshow(rgb, interpolation="nearest")
    for r in range(9):
        for c in range(9):
            d = digits[r, c]
            ax.text(c, r, "" if d <= 0 else str(int(d)),
                    ha="center", va="center", fontsize=8, color="black")
    # 3x3 box grid lines.
    for x in range(10):
        lw = 2.0 if x % 3 == 0 else 0.4
        ax.axhline(x - 0.5, color="black", lw=lw)
        ax.axvline(x - 0.5, color="black", lw=lw)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def render(snapshots, steps, perm, k, board_args, num_steps, out_path, title):
    """One column per visualised timestep; per-layer attention stacked above the
    predicted board."""
    n_layers = snapshots[steps[0]]["attn"].shape[0]
    n_cols = len(steps)
    n_rows = n_layers + 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols + 1.0, 2.6 * n_rows + 0.8),
        squeeze=False,
    )

    last_im = None
    for ci, step in enumerate(steps):
        snap = snapshots[step]
        attn = snap["attn"]                      # (n_layers, S, S)
        attn = attn[:, perm][:, :, perm]         # permute rows + cols
        for l in range(n_layers):
            ax = axes[l][ci]
            last_im = ax.imshow(attn[l], cmap="viridis", interpolation="nearest")
            # divider between conditioning (first k) and non-conditioning block.
            ax.axhline(k - 0.5, color="red", lw=0.8)
            ax.axvline(k - 0.5, color="red", lw=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(f"layer {l}", fontsize=8)
            if l == 0:
                ax.set_title(f"step {step}/{num_steps}\nt={snap['t']:.3f}",
                             fontsize=9)

        # board readout for this timestep
        ax = axes[n_layers][ci]
        digits, kind = decode_board(pred=snap["x0"], **board_args)
        draw_board(ax, digits, kind)
        if ci == 0:
            ax.set_ylabel("x0 board", fontsize=8)

    # one shared colourbar for the attention heatmaps
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:n_layers, :].ravel().tolist(),
                     shrink=0.6, label="attention weight")

    legend = [
        Patch(facecolor=HINT_RGB, edgecolor="k", label="given clue (conditioning)"),
        Patch(facecolor=RIGHT_RGB, edgecolor="k", label="correct prediction"),
        Patch(facecolor=WRONG_RGB, edgecolor="k", label="wrong prediction"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="path to the FLM checkpoint to analyse")
    ap.add_argument("--split", choices=["train", "val"], default="val",
                    help="which sudoku-gen split to pull the batch from")
    ap.add_argument("--num-steps", type=int, default=16,
                    help="number of Euler sampling steps for conditional_generate")
    ap.add_argument("--timesteps", type=int, nargs="*", default=None,
                    help="explicit step indices to visualise; default = a few "
                         "evenly-spaced steps including the last")
    ap.add_argument("--num-timesteps-show", type=int, default=4,
                    help="how many evenly-spaced steps to show (if --timesteps "
                         "not given)")
    ap.add_argument("--example-idx", type=int, default=0,
                    help="which (real) example in the batch to visualise")
    ap.add_argument("--num-examples", type=int, default=8,
                    help="cap the batch to this many real examples before "
                         "generating (keeps attention capture cheap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None,
                    help="output png filename (saved under <repo-root>/figures/)")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = utils.get_logger(__name__)

    cfg = load_run_config(args.checkpoint)
    logger.info("loaded config for algo=%s, data=%s",
                OmegaConf.select(cfg, "algo.name"),
                OmegaConf.select(cfg, "data.train"))

    tokenizer = dataloader.get_tokenizer(cfg)

    # --- one deterministic batch from the requested split ---
    L.seed_everything(args.seed)
    train_dl, eval_dl = dataloader.get_dataloaders(cfg, tokenizer, rank=0, world_size=1)
    dl = train_dl if args.split == "train" else eval_dl
    batch = next(iter(dl))

    input_ids = batch["input_ids"].to(args.device)
    valid_tokens = batch["valid_tokens"].to(args.device).bool()
    if "conditioning_mask" in batch:
        conditioning_mask = batch["conditioning_mask"].to(args.device).bool()
    else:
        conditioning_mask = torch.logical_not(valid_tokens)

    # drop padded rows, then cap to a small set of examples
    real_rows = valid_tokens.any(dim=-1)
    input_ids = input_ids[real_rows]
    conditioning_mask = conditioning_mask[real_rows]
    n = min(args.num_examples, input_ids.shape[0])
    input_ids = input_ids[:n]
    conditioning_mask = conditioning_mask[:n]
    if args.example_idx >= n:
        raise SystemExit(f"--example-idx {args.example_idx} out of range (have {n} examples)")

    # --- load the model with its own config ---
    Model = model_class(cfg)
    model = Model.load_from_checkpoint(
        args.checkpoint, tokenizer=tokenizer, config=cfg, weights_only=False
    ).to(args.device).eval()

    # which steps to snapshot
    if args.timesteps:
        steps = sorted({s for s in args.timesteps if 0 <= s < args.num_steps})
    else:
        m = max(1, min(args.num_timesteps_show, args.num_steps))
        steps = sorted(set(
            np.linspace(0, args.num_steps - 1, num=m).round().astype(int).tolist()))
    logger.info("visualising steps %s of %d", steps, args.num_steps)

    capture = AttentionCapture()
    with capture, torch.inference_mode():
        final_pred, snapshots, mean_attn = conditional_generate_with_capture(
            model, input_ids, conditioning_mask, capture,
            capture_steps=steps, example_idx=args.example_idx,
            num_steps=args.num_steps)

    # --- conditioning-first permutation for the chosen example ---
    ex_cond = conditioning_mask[args.example_idx].cpu().numpy().astype(bool)
    cond_idx = np.nonzero(ex_cond)[0]
    noncond_idx = np.nonzero(~ex_cond)[0]
    perm = np.concatenate([cond_idx, noncond_idx])
    k = len(cond_idx)
    logger.info("example %d: %d conditioning tokens, %d non-conditioning",
                args.example_idx, k, len(noncond_idx))
    
    report_conditioning_attention(mean_attn, ex_cond, logger)

    ex_tokens = input_ids[args.example_idx].cpu().numpy()
    board_args = {
        "tokens": ex_tokens,
        "cond_mask": ex_cond,
        "gt": ex_tokens,   # in infill mode input_ids IS the ground-truth solution
    }

    # report final accuracy on the cells the model had to fill
    fill = ~ex_cond
    final = final_pred[args.example_idx].cpu().numpy()
    n_fill = int(fill.sum())
    n_ok = int(((final == ex_tokens) & fill).sum())
    logger.info("example %d final fill accuracy: %d/%d = %.3f",
                args.example_idx, n_ok, n_fill, n_ok / max(n_fill, 1))

    figures_dir = os.path.join(dataloader.original_cwd(), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    out_name = args.out or (
        f"attn_{OmegaConf.select(cfg, 'algo.name')}_{args.split}"
        f"_ex{args.example_idx}_steps{args.num_steps}.png")
    out_path = os.path.join(figures_dir, out_name)

    title = (f"FLM conditional_generate attention "
             f"(algo={OmegaConf.select(cfg, 'algo.name')}, split={args.split}, "
             f"example {args.example_idx}, fill acc {n_ok}/{n_fill}) -- "
             f"conditioning tokens are the first {k} indices")

    render(snapshots, steps, perm, k, board_args, args.num_steps, out_path, title)
    logger.info("saved visualization to %s", out_path)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()