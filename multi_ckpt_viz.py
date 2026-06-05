#!/usr/bin/env python
"""
multi_ckpt_viz.py

Crawl an output/ tree, find every run whose *saved* config matches a criterion,
load each run's `last.ckpt`, conditionally generate from a SINGLE deterministic
val batch shared across all checkpoints, and render one figure:

    row 0      : original (ground-truth label-half images)
    row 1 .. N : conditional generations, one row per checkpoint,
                 left-labelled with that run's hyperparameters.

Because the conditioning batch is built ONCE and reused, and (optionally) the RNG
is reseeded before each model, differences between rows are attributable to the
model / hyperparameters rather than to data or noise.

Crucially: each model is instantiated with ITS OWN config, so sampling-time
hyperparameters (num_timesteps, backprop_steps, gamma) are the ones the run was
trained/sampled with -- not whatever is on the CLI.

Run from the repo root (so `algo`, `dataloader`, `utils` import):

    python multi_ckpt_viz.py \
        --output-dir output \
        --out sweep_compare.png \
        --num-display 16 --seed 0
"""
import argparse
import functools
import glob
import os
import hydra

import torch
torch.load = functools.partial(torch.load, weights_only=False)

import omegaconf
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import lightning as L

import algo
import dataloader
import utils
import torch
torch.load = functools.partial(torch.load, weights_only=False)

# --- resolvers / safe globals (mirror main.py so ckpt + cfg load cleanly) ----
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

IMG_SIZE = 14


# ----------------------------------------------------------------------------- #
# config discovery + filtering
# ----------------------------------------------------------------------------- #
def load_run_config(ckpt_path):
    """Return the OmegaConf the run was trained with, or None.

    Tries a sibling .hydra/config.yaml first (cheap), then the config pickled
    into the checkpoint's hyper_parameters.  >>> ADJUST the candidate paths if
    your output layout differs. <<<
    """
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for cand in (
        os.path.join(d, ".hydra", "config.yaml"),
        os.path.join(d, "..", ".hydra", "config.yaml"),
        os.path.join(d, "..", "..", ".hydra", "config.yaml"),
    ):
        if os.path.exists(cand):
            return OmegaConf.load(cand)
    # Fall back to the config save_hyperparameters() pickled into the ckpt.
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = ck.get("hyper_parameters", {}) or {}
        cfg = hp.get("config", hp)  # either {"config": cfg} or cfg saved directly
        if isinstance(cfg, (dict, omegaconf.DictConfig)) and len(cfg):
            return OmegaConf.create(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read config from {ckpt_path}: {e}")
    return None


def hp_tuple(cfg):
    return (
        OmegaConf.select(cfg, "algo.gamma"),
        OmegaConf.select(cfg, "algo.num_timesteps"),
        OmegaConf.select(cfg, "algo.backprop_steps"),
    )


def hp_label(cfg):
    g, T, bp = hp_tuple(cfg)
    return f"\u03b3={g}\nT={T} bp={bp}"


def criterion(cfg):
    """EDIT ME. Return True for runs you want in the figure."""
    return OmegaConf.select(cfg, "algo.name") == "discrete_loop_flm"


def model_class(cfg):
    name = OmegaConf.select(cfg, "algo.name")
    if name == "flm":
        return algo.FLM
    if name == "discrete_loop_flm":
        return algo.DiscreteLoopFLM
    raise ValueError(f"unknown algo {name}")


def find_runs(output_dir, ckpt_name="last.ckpt"):
    runs = []
    pattern = os.path.join(output_dir, "**", ckpt_name)
    for ckpt in sorted(glob.glob(pattern, recursive=True)):
        
        cfg = load_run_config(ckpt)
        if cfg is None:
            print(f"[skip] no config found for {ckpt}")
            continue
        if not criterion(cfg):
            continue
        runs.append((ckpt, cfg))
    # stable, readable ordering by (gamma, T, bp)
    runs.sort(key=lambda r: tuple(-1 if x is None else x for x in hp_tuple(r[1])))
    return runs


# ----------------------------------------------------------------------------- #
# image decoding + rendering
# ----------------------------------------------------------------------------- #
def tokens_to_image(tokens):
    """(B, 196) label-half tokens -> (B, 14, 14) {0,1} float images.

    pixel = token - 1 for tokens in {1, 2}; the rare PAD token (0) -> -1, so
    clamp back into [0, 1].
    """
    imgs = tokens.float() - 1.0
    imgs = imgs.clamp(0.0, 1.0)
    return imgs.reshape(-1, IMG_SIZE, IMG_SIZE)


def render(originals, gen_rows, labels, B, out_path, steps_note=""):
    n_rows = 1 + len(gen_rows)
    fig, axes = plt.subplots(
        n_rows, B,
        figsize=(0.8 * B + 2.4, 0.8 * n_rows + 0.5),
        squeeze=False,
    )

    def show(ax, img):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    for c in range(B):
        show(axes[0][c], originals[c])
    axes[0][0].set_ylabel("orig", rotation=0, ha="right", va="center", fontsize=8)

    for r, (row_imgs, lab) in enumerate(zip(gen_rows, labels), start=1):
        for c in range(B):
            show(axes[r][c], row_imgs[c])
        axes[r][0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)

    title = "Conditional MNIST \u2014 original (top) vs per-checkpoint generations"
    if steps_note:
        title += f"  ({steps_note})"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/mnist/2026.06.05")
    ap.add_argument("--ckpt-name", default="last.ckpt",
                    help="checkpoint filename to pick from each run "
                         "(e.g. last.ckpt or best_nll.ckpt)")
    ap.add_argument("--out", default="sweep_compare.png")
    ap.add_argument("--num-display", type=int, default=16,
                    help="number of conditioning images (columns)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug-with-training-set", action="store_true")
    ap.add_argument("--same-noise", action="store_true",
                    help="reseed before each model so every checkpoint sees "
                         "identical init noise (isolates the model effect)")
    args = ap.parse_args()

    logger = utils.get_logger(__name__)

    runs = find_runs(args.output_dir, args.ckpt_name)
    if not runs:
        raise SystemExit(f"no matching runs under {args.output_dir}")
    logger.info("matched %d runs", len(runs))
    for ckpt, cfg in runs:
        logger.info("  %-22s  ->  %s", hp_label(cfg).replace("\n", " "), ckpt)

    # Reference config drives ONLY the data pipeline (shared across all runs).
    ref_cfg = runs[0][1]
    tokenizer = dataloader.get_tokenizer(ref_cfg)

    # --- build ONE deterministic batch, shared across every checkpoint ---
    L.seed_everything(args.seed)
    train_dl, eval_dl = dataloader.get_dataloaders(
        ref_cfg, tokenizer, rank=0, world_size=1
    )
    dl = train_dl if args.debug_with_training_set else eval_dl
    batch = next(iter(dl))

    input_ids = batch["input_ids"].to(args.device)            # (B, 2*L)
    gen_mask = batch["valid_tokens"].to(args.device).bool()   # 1 over solution half
    conditioning_mask = torch.logical_not(gen_mask)           # 1 over puzzle half
    real_rows = gen_mask.any(dim=-1)                          # drop padded rows

    input_ids = input_ids[real_rows]
    conditioning_mask = conditioning_mask[real_rows]
    B = min(args.num_display, input_ids.shape[0])
    input_ids = input_ids[:B]
    conditioning_mask = conditioning_mask[:B]

    half = input_ids.shape[1] // 2
    originals = tokens_to_image(input_ids[:, half:]).cpu()

    # --- one generation row per checkpoint, each model loaded with its OWN cfg ---
    gen_rows, labels = [], []
    for ckpt, cfg in runs:
        logger.info("generating: %s", hp_label(cfg).replace("\n", " "))
        if args.same_noise:
            L.seed_everything(args.seed)

        Model = model_class(cfg)
        model = Model.load_from_checkpoint(
            ckpt, tokenizer=tokenizer, config=cfg, weights_only=False   # cfg is what makes T/bp/gamma correct
        ).to(args.device).eval()

        with torch.inference_mode():
            pred = model.conditional_generate_samples(
                input_ids, conditioning_mask, num_steps=cfg.algo.num_timesteps, #NOTE: sampling steps is same as number trained on
            )

        gen_rows.append(tokens_to_image(pred[:, half:]).cpu())
        labels.append(hp_label(cfg))

        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    steps_note = "sampling.steps=per-run"
    render(originals, gen_rows, labels, B, args.out, steps_note=steps_note)
    logger.info("saved visualization to %s", args.out)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()