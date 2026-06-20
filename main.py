import json
import os
import itertools
import functools
import argparse
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
torch.load = functools.partial(torch.load, weights_only=False)
from torch.distributed import init_process_group, destroy_process_group
import wandb
import algo
import dataloader
import utils
from tqdm import tqdm 
from collections import defaultdict
from dataset_code import nqueens_common as nq
from dataset_code.generate_nqueens_dataset import build_nqueens_eval_puzzles

import numpy as np
from datetime import datetime

import uuid

torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig, omegaconf.base.ContainerMetadata, omegaconf.base.Metadata])

omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)

#### Utilities ####
@L.pytorch.utilities.rank_zero_only
def _print_config(
        config: omegaconf.DictConfig,
        resolve: bool = True,
        save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)

@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [
            ('train', train_ds), ('valid', valid_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        if tokenizer is None:
            print(f'First {k} tokens:', first)
            print(f'Last {k} tokens:', last)
        else:
            if hasattr(tokenizer, 'decode'):
                print(f'First {k} tokens:', tokenizer.decode(first))
                print(f'Last {k} tokens:', tokenizer.decode(last))
            else:
                print(f'First {k} tokens:', first)
                print(f'Last {k} tokens:', last)

def _train(
    diffusion_model,
    config,
    logger,
    tokenizer,
    rank: int, 
    world_size: int
    ):
    logger.info("start training")
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wid = config.wandb.get('id')
        if not wid or len(str(wid)) > 16:
            wid = str(uuid.uuid4().hex[:8])
        config.wandb.id = wid 
        if config.wandb.get('name'):
            config.wandb.name = f'{config.wandb.name}_{wid}'
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb
        )

    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(
            config.checkpointing.resume_ckpt_path
        )):
        ckpt_path = config.checkpointing.resume_ckpt_path 
    else:
        ckpt_path = None  

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for  _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(
        config,tokenizer, rank, world_size
    )
    _print_batch(train_ds, valid_ds, tokenizer)

    if config.training.finetune_path != '':
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path,
            tokenizer=tokenizer,
            config=config,
            weights_only=False
        )
    else:
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)
    
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

@L.pytorch.utilities.rank_zero_only
def _generate_samples(diffusion_model, config, tokenizer, logger):
    # TODO: refactor so we have unconditional generation too 
    logger.info("Start sample_eval")
    train_dl, eval_dl = dataloader.get_dataloaders(config, tokenizer, rank=0, world_size=1)
    if config.eval.debug_with_training_set:
        test_dl = train_dl 
    else:
        test_dl = eval_dl 

    model = diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        weights_only=False
    ).to("cuda").eval()

    n_solved = n_total = to_predict_cells_ok = to_predict_cells_total = total_batches = 0
    solution_hints_cells_ok = solution_hints_cells_total = 0 # also track how well model copies over input hints
    with torch.inference_mode():
        for batch in tqdm(test_dl):
            if config.eval.max_batches > 0 and total_batches >= config.eval.max_batches:
                break  
            input_ids = batch['input_ids'].cuda()
            gen_mask = batch['valid_tokens'].cuda().bool() # 1 for solution region
            gt = input_ids.clone()
            real_rows = gen_mask.any(dim=-1) # padded rows have no valid tokens

            conditioning_mask = torch.logical_not(gen_mask).bool()
            full_seq_pred = model.conditional_generate_samples(input_ids, conditioning_mask)

            # Score ONLY cells the model had to infer: blanks (token 1) in the puzzle half,
            # projected onto their aligned positions in the solution half.
            B, S = input_ids.shape
            assert S % 2 == 0, "expected [puzzle | solution] layout with aligned halves"
            half = S // 2
            if config.data.valid == "mnist":
                unprovided_id = 0 
            else:
                unprovided_id = 1

            blank = (input_ids[:, :half] == unprovided_id)                  # (B, half): unprovided cells
            given_hints = (input_ids[:, :half] > unprovided_id)    
            solution_hints = torch.zeros_like(gen_mask)
            solution_hints[:, half:] = given_hints
            solution_hints &= gen_mask 
            to_predict = torch.zeros_like(gen_mask)
            to_predict[:, half:] = blank
            to_predict &= gen_mask                              # stay inside the solution region

            to_predict_correct = (full_seq_pred == gt) & to_predict
            to_predict_row_hits = to_predict_correct.sum(-1)
            row_need = to_predict.sum(-1)
            to_predict_exact = (to_predict_row_hits == row_need) & real_rows & (row_need > 0)
            solution_hints_correct = (full_seq_pred == gt) & solution_hints

            n_solved    += to_predict_exact[real_rows].sum().item() #all predicted to fill cells correct
            n_total     += (real_rows & (row_need > 0)).sum().item()
            to_predict_cells_ok    += to_predict_correct[real_rows].sum().item()
            to_predict_cells_total += to_predict[real_rows].sum().item()
            solution_hints_cells_ok += solution_hints_correct[real_rows].sum().item()
            solution_hints_cells_total += solution_hints[real_rows].sum().item()

            total_batches += 1

    prefix = config.data.valid
    metrics = {
        f'{prefix}/to_predict_exact_match': n_solved / max(n_total, 1),
        f'{prefix}/to_predict_cell_acc':    to_predict_cells_ok / max(to_predict_cells_total, 1),
        f'{prefix}/given_cells_solution_acc': solution_hints_cells_ok / max(solution_hints_cells_total, 1),
        f'{prefix}/n':           n_total,
    }
    logger.info(metrics)
    return metrics

def _resolve_against_original_cwd(path):
    """Resolve a relative path against the launch cwd, not hydra's run dir.

    hydra runs with chdir=True, so the process cwd becomes a fresh timestamped
    run dir. Relative CLI paths (e.g. eval.checkpoint_path) are meant to be
    relative to where the user invoked `python main.py`, which is what
    dataloader.original_cwd() returns under @hydra.main.
    """
    if not path or os.path.isabs(path):
        return path
    return os.path.join(dataloader.original_cwd(), path)


def _resolve_sudoku_output_dir(config, ckpt):
    """Where to drop per-run sudoku eval records.

    `ckpt` is the already-resolved absolute checkpoint path. Always namespaced
    by the checkpoint's filename stem so evaluating several checkpoints from the
    same run doesn't overwrite a shared results.json. Prefer an explicit
    override; otherwise sit next to the evaluated checkpoint
    (``<run>/checkpoints/x.ckpt`` -> ``<run>/sudoku_eval/x``); finally fall back
    to the current run's save_dir.
    """
    stem = os.path.splitext(os.path.basename(ckpt))[0] if ckpt else 'eval'
    override = config.eval.get('sudoku_output_dir', None)
    if override:
        return os.path.join(_resolve_against_original_cwd(override), stem)
    if ckpt:
        run_dir = os.path.dirname(os.path.dirname(ckpt))
        return os.path.join(run_dir, 'sudoku_eval', stem)
    return os.path.join(config.checkpointing.save_dir, 'sudoku_eval', stem)


@L.pytorch.utilities.rank_zero_only
@torch.no_grad()
def _sudoku_eval(diffusion_model, config, tokenizer, logger):
    """Generate solutions for the sudoku validation set and save per-run records.

    Records (generated vs. ground-truth solution, per-puzzle correctness) and
    the aggregate accuracy are written to ``results.json`` under the run's
    output dir, alongside the checkpoints.
    NOTE: Uses the model ckpt saved config to determine samping steps! 
    """
    logger.info('Starting Sudoku eval.')
    _, eval_dl = dataloader.get_dataloaders(
        config, tokenizer, rank=0, world_size=1, skip_train=True)

    assert config.eval.checkpoint_path, \
        'config.eval.checkpoint_path must be set for sudoku_eval'
    # Resolve relative to the launch cwd, since hydra has chdir'd us into a new
    # run dir and a relative ckpt path would otherwise be looked up there.
    ckpt_path = _resolve_against_original_cwd(config.eval.checkpoint_path)

    model = diffusion_model.load_from_checkpoint(
        ckpt_path,
        tokenizer=tokenizer,
        weights_only=False).to('cuda')
    model._eval_mode()  # applies EMA weights unless config.eval.disable_ema

    output_dir = _resolve_sudoku_output_dir(config, ckpt_path)
    os.makedirs(output_dir, exist_ok=True)

    records = []
    num_correct = total = total_batches = 0
    with torch.inference_mode():
        for batch in tqdm(eval_dl, desc='Sudoku eval'):
            if (config.eval.sudoku_max_batches > 0
                    and total_batches >= config.eval.sudoku_max_batches):
                break
            input_ids = batch['input_ids'].cuda()
            
            if not config.sampling.override_algo_steps:
                num_sampling_steps = model.config.algo.get('num_timesteps', None)
            else:
                num_sampling_steps = config.sampling.steps
            
            if config.algo.get('infill', False):
                conditioning_mask = batch['conditioning_mask'].cuda().bool()
                real_rows = conditioning_mask.any(dim=-1)       # padded rows have no clues
                full_seq_pred = model.conditional_generate_samples(
                    input_ids, conditioning_mask, num_steps=num_sampling_steps)
                gt = input_ids
                generated = full_seq_pred
            else:
                valid_tokens = batch['valid_tokens'].cuda().bool()  # 1 over solution
                real_rows = valid_tokens.any(dim=-1)                # drop padded rows
                conditioning_mask = torch.logical_not(valid_tokens)
                full_seq_pred = model.conditional_generate_samples(
                    input_ids, conditioning_mask, num_steps=num_sampling_steps)
                S = input_ids.shape[1]
                assert S % 2 == 0, 'expected [puzzle | solution] layout'
                half = S // 2 
                gt = input_ids[:, half:]
                generated = full_seq_pred[:, half:]

            B = input_ids.shape[0]
            correct = (generated == gt).all(dim=1)

            for i in range(B):
                if not real_rows[i]:
                    continue
                is_correct = bool(correct[i].item())
                records.append({
                    'generated': tokenizer.decode(generated[i].cpu()),
                    'ground_truth': tokenizer.decode(gt[i].cpu()),
                    'correct': is_correct,
                })
                num_correct += int(is_correct)
                total += 1
            total_batches += 1

    accuracy = num_correct / max(total, 1)
    logger.info(
        f'Sudoku accuracy: {num_correct}/{total} ({accuracy * 100:.2f}%)')
    results = {
        'accuracy': accuracy,
        'num_correct': num_correct,
        'num_total': total,
        'checkpoint_path': config.eval.checkpoint_path,
        'records': records,
    }
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Sudoku eval results saved to {results_path}')

    if wandb.run is not None:
        wandb.log({
            'sudoku/accuracy': accuracy,
            'sudoku/num_correct': num_correct,
            'sudoku/num_total': total,
        })
    return results


def _resolve_nqueens_output_dir(config, ckpt):
    """Where to drop per-run N-Queens eval records (mirrors the sudoku resolver)."""
    stem = os.path.splitext(os.path.basename(ckpt))[0] if ckpt else 'eval'
    override = config.eval.get('nqueens_output_dir', None)
    if override:
        return os.path.join(_resolve_against_original_cwd(override), stem)
    if ckpt:
        run_dir = os.path.dirname(os.path.dirname(ckpt))
        return os.path.join(run_dir, 'nqueens_eval', stem)
    return os.path.join(config.checkpointing.save_dir, 'nqueens_eval', stem)


@L.pytorch.utilities.rank_zero_only
@torch.no_grad()
def _nqueens_eval(diffusion_model, config, tokenizer, logger):
    """Evaluate N-Queens completion: accuracy and coverage, binned by #solutions.

    For each eval puzzle (a board with `k` clue queens) we draw
    ``eval.nqueens_num_samples`` stochastic completions and measure:
      - accuracy = fraction of samples that satisfy ALL N-Queens constraints and
        respect the clues (equivalently, land in the puzzle's completion set);
      - coverage = distinct valid solutions found / total valid completions.
    Puzzles are binned by their completion count (the GRAM-style x-axis). Results
    are written to results.json; aggregates are logged to wandb.
    NOTE: uses the model ckpt's saved config for sampling steps, like sudoku eval.
    """

    logger.info('Starting N-Queens eval.')
    assert config.eval.checkpoint_path, \
        'config.eval.checkpoint_path must be set for nqueens_eval'
    ckpt_path = _resolve_against_original_cwd(config.eval.checkpoint_path)

    model = diffusion_model.load_from_checkpoint(
        ckpt_path, tokenizer=tokenizer, weights_only=False).to('cuda')
    model._eval_mode()

    n = int(config.data.get('nqueens_n', 8))
    num_samples = int(config.eval.get('nqueens_num_samples', 20))
    num_puzzles = int(config.eval.get('nqueens_num_puzzles', 200)) #TODO: should change default num of eval puzzles
    assert model.num_tokens == n * n, (
        f'model.length ({model.num_tokens}) must equal n*n ({n * n}); '
        f'set model=nqueens_infill with model.length={n * n}.')

    if not config.sampling.override_algo_steps:
        num_sampling_steps = model.config.algo.get('num_timesteps', None)
    else:
        num_sampling_steps = config.sampling.steps

    puzzles = build_nqueens_eval_puzzles(n, num_puzzles, seed=config.seed)
    logger.info(f'Evaluating {len(puzzles)} N-Queens puzzles, '
                f'{num_samples} samples each (n={n}).')

    output_dir = _resolve_nqueens_output_dir(config, ckpt_path)
    os.makedirs(output_dir, exist_ok=True)

    records = []
    by_count = defaultdict(lambda: {'accuracy': [], 'coverage': []})
    with torch.inference_mode():
        for puzzle in tqdm(puzzles, desc='N-Queens eval'):
            clue_board = np.asarray(puzzle['puzzle_board'])
            completions = set(puzzle['completions'])
            solution_count = puzzle['solution_count']

            # Batch `num_samples` identical clue boards; only the clue queens are
            # held clean, the rest is re-noised, so the samples differ.
            board_t = torch.tensor(clue_board, dtype=torch.long, device='cuda')
            input_ids = board_t.unsqueeze(0).repeat(num_samples, 1)
            conditioning_mask = (input_ids == nq.QUEEN_ID)

            generated = model.conditional_generate_samples(
                input_ids, conditioning_mask, num_steps=num_sampling_steps)

            num_correct = 0
            found = set()
            for row in generated.cpu().numpy():
                cols = nq.board_to_cols(row, n)
                if cols is not None and cols in completions:
                    num_correct += 1
                    found.add(cols)

            accuracy = num_correct / num_samples
            coverage = len(found) / max(solution_count, 1)
            by_count[solution_count]['accuracy'].append(accuracy)
            by_count[solution_count]['coverage'].append(coverage)
            records.append({
                'solution_count': solution_count,
                'num_clues': len(puzzle['clue_cols']),
                'num_samples': num_samples,
                'num_correct': num_correct,
                'num_distinct_found': len(found),
                'accuracy': accuracy,
                'coverage': coverage,
            })

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    per_count = {
        str(c): {
            'num_puzzles': len(v['accuracy']),
            'accuracy': _mean(v['accuracy']),
            'coverage': _mean(v['coverage']),
        }
        for c, v in sorted(by_count.items())
    }
    overall_accuracy = _mean([r['accuracy'] for r in records])
    overall_coverage = _mean([r['coverage'] for r in records])
    logger.info(
        f'N-Queens overall accuracy={overall_accuracy:.4f} '
        f'coverage={overall_coverage:.4f} over {len(records)} puzzles.')

    results = {
        'n': n,
        'num_samples': num_samples,
        'num_puzzles': len(records),
        'overall_accuracy': overall_accuracy,
        'overall_coverage': overall_coverage,
        'checkpoint_path': config.eval.checkpoint_path,
        'per_solution_count': per_count,   # x-axis aggregates for plotting
        'records': records,
    }
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'N-Queens eval results saved to {results_path}')

    if wandb.run is not None:
        wandb.log({
            'nqueens/accuracy': overall_accuracy,
            'nqueens/coverage': overall_coverage,
            'nqueens/num_puzzles': len(records),
        })
    return results


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    '''
    Training
    '''
    # TODO: FIX!! don't hardcode anymore. 
    RANK = 0
    WORLD_SIZE = 1

    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == "flm":
        diffusion_model = algo.FLM 
    elif config.algo.name == "discrete_loop_flm":
        diffusion_model = algo.DiscreteLoopFLM
    elif config.algo.name == "discrete_recurrent_flm":
        diffusion_model = algo.DiscreteRecurrentFLM 
    elif config.algo.name == "cond_uncond_loop_flm":
        diffusion_model = algo.CondUncondLoopFLM
    else:
        raise ValueError(f"Given incorrect algo {config.algo.name}")

    kwargs = {'diffusion_model': diffusion_model,
              'config': config,
              'tokenizer': tokenizer,
              'logger': logger}
    
    if config.mode == 'sample_eval':
        _generate_samples(**kwargs)
    elif config.mode == "sudoku_eval":
        _sudoku_eval(**kwargs)
    elif config.mode == "nqueens_eval":
        _nqueens_eval(**kwargs)
    else:
        _train(**kwargs, rank=RANK, world_size=WORLD_SIZE)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()