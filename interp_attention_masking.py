import torch 
import argparse 
import functools 
import torch.nn.functional as F 

torch.load = functools.partial(torch.load, weights_only=False)

import lightning as L 
from omegaconf import OmegaConf 
from tqdm import tqdm 
import logging

import dataloader 
import utils 
import models.dit as dit 
from interp_attention_viz import load_run_config, model_class 

BLANK_ID = 1


class ConditioningAttentionMask:
    """Monkeypatch ``DDiTBlock.custom_sdpa`` so that, when ``enabled``, the
    post-softmax attention is restricted to conditioning key tokens.

    ``mask`` is a ``(B, S)`` boolean tensor that is True on conditioning
    (clue) tokens. When the intervention is on we zero the attention columns of
    every non-conditioning key token and (optionally) renormalize each query
    distribution so it sums to 1 over the surviving conditioning tokens. The
    model is run with ``use_jvp_attn=True`` so this code path is the one used.
    """

    def __init__(self, renormalize=True):
        self.enabled = False
        self.mask = None          # (B, S) bool, True = conditioning key to keep
        self.renormalize = renormalize
        self._orig = None

    def __enter__(self):
        intervention = self
        self._orig = dit.DDiTBlock.custom_sdpa

        def patched(self, q, k, v, softcap=-1.0):
            # q,k,v: (B, H, S, D). Mirror the original custom_sdpa math.
            B, H, S, D = q.shape
            qs = q / (D ** 0.5)
            attn_weights = torch.einsum("bhid,bhjd->bhij", qs, k)
            if softcap > 0.0:
                attn_weights = softcap * torch.tanh(attn_weights / softcap)
            attn_probs = torch.softmax(attn_weights, dim=-1)

            if intervention.enabled and intervention.mask is not None:
                # keep[b, j] == 1 for conditioning key tokens. Broadcast over
                # heads (H) and query rows (S) -> zero out non-conditioning keys.
                keep = intervention.mask.to(device=attn_probs.device,
                                            dtype=attn_probs.dtype)
                keep = keep[:, None, None, :]            # (B,1,1,S)
                attn_probs = attn_probs * keep
                if intervention.renormalize:
                    denom = attn_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    attn_probs = attn_probs / denom

            return torch.einsum("bhij,bhjd->bhid", attn_probs, v)

        dit.DDiTBlock.custom_sdpa = patched
        return self

    def __exit__(self, *exc):
        dit.DDiTBlock.custom_sdpa = self._orig

def _to_predict_mask(input_ids, valid_tokens, cond, has_cond_field, pad_id):
    """Boolean mask of the cells the model actually had to infer (the blanks).

    Infill layout (a ``conditioning_mask`` is present): the sequence IS the
    solution grid; the cells to predict are the real, non-conditioning cells.
    Concat layout ``[puzzle | solution]``: the cells to predict are the
    solution-half positions whose aligned puzzle cell was blank.
    """
    if has_cond_field:
        real_cell = input_ids != pad_id
        return (~cond) & real_cell
    S = input_ids.shape[1]
    half = S // 2
    blank = input_ids[:, :half] == BLANK_ID
    to_predict = torch.zeros_like(valid_tokens)
    to_predict[:, half:] = blank
    to_predict &= valid_tokens
    return to_predict

# ----------------------------------------------------------------------------- #
# generation (FLM conditional_generate rollout, forced through custom_sdpa)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def conditional_generate(model, x_input, conditioning_mask, num_steps, eps=1e-5):
    """Euler-ODE conditional generation, mirroring
    ``FLM.conditional_generate_samples`` but with ``use_jvp_attn=True`` so the
    (patchable) ``custom_sdpa`` attention path is used."""
    device = model.device
    B, L = x_input.shape
    V = model.vocab_size
    x_input = x_input.to(device)
    conditioning_mask = conditioning_mask.to(device).bool()

    x_input_onehot = F.one_hot(x_input, num_classes=V)
    tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

    z, _ = model.corrupt_continuous(
        x_input, t=torch.tensor([0.0], device=device),
        conditioning_mask=conditioning_mask)
    z = z.to(device).to(model.dtype)

    for i in range(num_steps):
        tau_t_in = tau_vals[i].expand(B)
        t_in = model._tau_to_t(tau_t_in)
        dt = model._tau_to_t(tau_vals[i + 1].expand(B)) - t_in
        df_mask = conditioning_mask if model.diffusion_forcing else None

        x_1_pred = model.forward(z, tau_t_in, conditioning_mask=df_mask,
                                 use_jvp_attn=True)
        x_1_pred_probs = x_1_pred.exp()

        if i == num_steps - 1:
            z = x_1_pred_probs
            z = torch.where(conditioning_mask.unsqueeze(-1), x_input_onehot, z)
            break
        v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + eps)
        z = z + dt.view(-1, 1, 1) * v
        z = torch.where(conditioning_mask.unsqueeze(-1), x_input_onehot, z)

    return z.argmax(dim=-1)

@torch.no_grad()
def evaluate(model, batches, intervention, mask_on, num_steps, pad_id, seed,
             device, logger):
    '''
    Generate over a fixed list of batches.
    Reseed before the loop so diffusion init noise is identical across on/off runs
    '''
    intervention.enabled = mask_on
    L.seed_everything(seed, verbose=False)

    n_solved = n_total = 0 
    cells_ok = cells_total = 0
    for batch in tqdm(batches, desc=f"mask={'on' if mask_on else 'off'}"):
        input_ids = batch['input_ids'].to(device)
        valid_tokens = batch['valid_tokens'].to(device).bool()
        has_cond = 'conditioning_mask' in batch 
        cond = (batch['conditioning_mask'].to(device).bool() if has_cond else torch.logical_not(valid_tokens))
        real_rows = valid_tokens.any(dim=-1)

        intervention.mask = cond 
        pred = conditional_generate(model, input_ids, cond, num_steps)
        gt = input_ids 

        to_predict = _to_predict_mask(input_ids, valid_tokens, cond, has_cond,pad_id)
        correct = (pred == gt) & to_predict 
        row_hits = correct.sum(dim=-1)
        need = to_predict.sum(dim=-1)
        exact = (row_hits == need) & real_rows & (need > 0)

        n_solved += int(exact[real_rows].sum().item())
        n_total += int((real_rows & (need > 0)).sum().item())
        cells_ok += int(correct[real_rows].sum().item())
        cells_total += int(to_predict[real_rows].sum().item())
        
    return {
        "exact_match": n_solved / max(n_total, 1),
        "cell_acc": cells_ok / max(cells_total, 1),
        "n_puzzles": n_total,
        "n_solved": n_solved,
        "cells_ok": cells_ok,
        "cells_total": cells_total,
    }



# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="path to the FLM checkpoint to analyse")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--num-steps", type=int, default=16,
                    help="number of Euler sampling steps for conditional_generate")
    ap.add_argument("--max-batches", type=int, default=-1,
                    help="cap the number of batches evaluated (-1 = whole split)")
    ap.add_argument("--no-renormalize", action="store_true",
                    help="zero non-conditioning attention WITHOUT renormalizing "
                         "(rows no longer sum to 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = utils.get_logger(__name__)
    cfg = load_run_config(args.checkpoint)
    logger.info(f'loaded config for algo={OmegaConf.select(cfg, "algo.name")} data={OmegaConf.select(cfg, "data.train")}')
    tokenizer = dataloader.get_tokenizer(cfg)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0 

    # deterministic batches 
    L.seed_everything(args.seed)
    train_dl, eval_dl = dataloader.get_dataloaders(cfg, tokenizer, rank=0, world_size=1)
    dl = train_dl if args.split == "train" else eval_dl 


    # get same batches for both ON and OFF 
    batches = []
    for batch in dl:
        if args.max_batches > 0 and len(batches) >= args.max_batches:
            break 
        batches.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in batch.items()})

    logger.info(f"collected {len(batches)} batches from the {args.split} split")

    Model = model_class(cfg)
    model = Model.load_from_checkpoint(
        args.checkpoint, tokenizer=tokenizer, config=cfg, weights_only=False
    ).to(args.device).eval()

    intervention = ConditioningAttentionMask(renormalize=not args.no_renormalize)
    with intervention, torch.inference_mode():
        res_off = evaluate(model, batches, intervention, mask_on=False,
                           num_steps=args.num_steps, pad_id=pad_id,
                           seed=args.seed, device=args.device, logger=logger)
        res_on = evaluate(model, batches, intervention, mask_on=True,
                          num_steps=args.num_steps, pad_id=pad_id,
                          seed=args.seed, device=args.device, logger=logger)
    renorm = "renormalized" if not args.no_renormalize else "NOT renormalized"
    logger.info("=" * 64)
    logger.info("Attention-masking intervention (%s split, %d puzzles, "
                "%d steps, mask %s)", args.split, res_off["n_puzzles"],
                args.num_steps, renorm)
    logger.info("  %-18s %14s %14s", "metric", "mask OFF", "mask ON")
    logger.info("  %-18s %14.4f %14.4f", "exact-match rate",
                res_off["exact_match"], res_on["exact_match"])
    logger.info("  %-18s %14.4f %14.4f", "per-cell accuracy",
                res_off["cell_acc"], res_on["cell_acc"])
    logger.info("  %-18s %14d %14d", "puzzles solved",
                res_off["n_solved"], res_on["n_solved"])
    logger.info("  %-18s %14d %14d", "cells correct",
                res_off["cells_ok"], res_on["cells_ok"])
    logger.info("=" * 64)
    logger.info("delta (ON - OFF): exact-match %+.4f, cell-acc %+.4f",
                res_on["exact_match"] - res_off["exact_match"],
                res_on["cell_acc"] - res_off["cell_acc"])




if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True 
    main()