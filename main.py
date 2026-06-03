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
            blank = (input_ids[:, :half] == 1)                  # (B, half): unprovided cells
            given_hints = (input_ids[:, :half] > 1)    
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

    metrics = {
        'sudoku/to_predict_exact_match': n_solved / max(n_total, 1),
        'sudoku/to_predict_cell_acc':    to_predict_cells_ok / max(to_predict_cells_total, 1),
        'sudoku/given_cells_solution_acc': solution_hints_cells_ok / max(solution_hints_cells_total, 1),
        'sudoku/n':           n_total,
    }
    logger.info(metrics)
    return metrics


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
    else:
        raise ValueError(f"Given incorrect algo {config.algo.name}")

    kwargs = {'diffusion_model': diffusion_model,
              'config': config,
              'tokenizer': tokenizer,
              'logger': logger}
    
    if config.mode == 'sample_eval':
        _generate_samples(**kwargs)
    else:
        _train(**kwargs, rank=RANK, world_size=WORLD_SIZE)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()