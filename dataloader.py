'''
Code acknowledgements: 
- https://github.com/david3684/flm/blob/main/dataloader.py
- https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/pretrain.py

TRM 
config.data_paths_test
config.data_paths
config.seed
conifg.eval_interval 

config.data.valid needs to have these things above
'''

import functools
import itertools
import json
import math
import os
import numpy as np
import pydantic 
from typing import Tuple, List, Dict, Optional

import re
import shutil
import typing
import urllib
import zipfile
from typing import Optional

import datasets
from dataset_code.dataset_common import PuzzleDatasetMetadata, IGNORE_LABEL_ID
import fsspec
import numpy as np
import requests
import tokenizers
import torch
import transformers
from torch.utils.data import IterableDataset, get_worker_info  
from einops import repeat
import utils
from hydra.utils import get_original_cwd
LOGGER = utils.get_logger(__name__)

class SudokuTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self, pad_token='[PAD]', unk_token='[UNK]', **kwargs):
        self.characters = list('0123456789')
        self._vocab_str_to_int = {
            '[PAD]': 0,
            '[UNK]': len(self.characters)+1,
            **{ch: i+1 for i,ch in enumerate(self.characters)}
        }
        self._vocab_int_to_str = {v:k for k,v in self._vocab_str_to_int.items()}
        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)
    
    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())
    
    def _convert_token_to_id(self, token:str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int['[UNK]']
        )
    
    def _convert_id_to_token(self, index:int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)
    
    def get_vocab(self) -> typing.Dict[str,int]:
        return self._vocab_str_to_int

class IdentityTokenizer:
    """Tokenizer shim for discrete data with no string vocabulary.
    Implements only what TrainerBase actually touches."""
    def __init__(self, vocab_size, pad_token_id=None):
        self._vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def __len__(self):
        return self._vocab_size

    def batch_decode(self, samples, **kwargs):
        if torch.is_tensor(samples):
            samples = samples.tolist()
        return [' '.join(map(str, row)) for row in samples]


def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == "sudoku-extreme":
        return IdentityTokenizer(vocab_size=11, pad_token_id=0)
    elif config.data.tokenizer_name_or_path == "mnist":
        # Pixel space: token 0 = PAD/EMPTY, tokens 1/2 binary
        return IdentityTokenizer(vocab_size=3, pad_token_id=0)
    else:
        raise ValueError("Only data tokenizer names are 'sudoku-extreme' and 'mnist'")


#### Dataset code ####
def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)

class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        with open(os.path.join(get_original_cwd(), dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                self._data[set_name_] = {
                    field_name: np.load(os.path.join(get_original_cwd(), dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }
                
    def _collate_batch(self, batch):
        """
        Returns {'input_ids', 'valid_tokens'}.
        input_ids: puzzle (inputs) concatenated with solution (labels), all valid token ids.
        valid_tokens: boolean mask where 1 means tokens that are included in loss calculation.
            0 on empty all padded rows and also on conditioning context. Loss only on solution area. 
        """
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # The ONLY thing that branches: pad rows up to local_batch_size.
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }
            batch = {
                k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1),
                        constant_values=pad_values[k])
                for k, v in batch.items()
            }

        inputs = batch["inputs"].reshape(batch["inputs"].shape[0], -1)
        labels = batch["labels"].reshape(batch["labels"].shape[0], -1)

        # Mask first — signal are tokens which aren't pad_id / -100
        input_mask = inputs != self.metadata.pad_id
        label_mask = labels != IGNORE_LABEL_ID

        # Then sanitize so input_ids holds only valid token ids (replace -100 label to pad_id)
        labels = np.where(label_mask, labels, self.metadata.pad_id)

        input_ids      = np.concatenate([inputs, labels], axis=-1)
        # valid tokens (tokens that go into loss) exclude conditioning input tokens and padded tokens   
        valid_tokens = np.concatenate([np.zeros_like(input_mask), label_mask], axis=-1).astype(np.int32)
        
        return {
            "input_ids": torch.from_numpy(input_ids).long(),
            "valid_tokens": torch.from_numpy(valid_tokens),
        }
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield batch #NOTE: only yield batch, different than TRM 
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield batch #NOTE: only yield batch, different than TRM
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


def get_puzzle_dataset(config, rank: int, world_size:int):
    train_epochs_per_iter = config.data.eval_interval if config.data.eval_interval is not None else config.trainer.max_steps
    train_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data.data_paths,
        rank=rank,
        num_replicas=world_size,
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.loader.batch_size * world_size 
    ), split='train')
    test_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data.data_paths_test if len(config.data.data_paths_test)>0  else config.data.data_paths,
        rank=rank,
        num_replicas=world_size,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.loader.eval_batch_size * world_size 
    ), split='test')

    return {
        'train': train_dataset, 
        'test': test_dataset
    }

def get_dataset(dataset_name, 
                mode,
                rank,
                world_size,
                config=None):
    if dataset_name in ("sudoku-extreme", "mnist"):
        dataset = get_puzzle_dataset(
            config, rank, world_size 
        )   
        data = dataset[mode]
        return data 
    else:
        raise ValueError(f"Only valid dataset name is sudoku-extreme and mnist. Received {dataset_name}")


def get_dataloaders(config, tokenizer, rank:int, world_size:int, skip_train=False, skip_valid=False, valid_seed=None):
    num_gpus = torch.cuda.device_count()
    is_iterable = config.data.is_iterable

    assert (config.loader.global_batch_size
            == (config.loader.batch_size
                * config.trainer.num_nodes
                * num_gpus
                * config.trainer.accumulate_grad_batches))
    
    if config.loader.global_batch_size % (
            num_gpus * config.trainer.accumulate_grad_batches) != 0:
        raise ValueError(
            f'Train Batch Size {config.training.batch_size}'
            f'not divisible by {num_gpus} gpus with accumulation '
            f'{config.trainer.accumulate_grad_batches}.')
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f'Eval Batch Size for {config.loader.eval_batch_size} '
            f'not divisible by {num_gpus}.')
    
    if skip_train:
        train_set = None 
    else:
        train_set = get_dataset(
            config.data.train,
            mode='train',
            config=config,
            rank=rank,
            world_size=world_size)
    
    validation_split = 'test'
    if skip_valid:
        valid_set = None 
    else:
        valid_set = get_dataset(
            config.data.valid,
            mode=validation_split,
            config=config,
            rank=rank,
            world_size=world_size
        )
    
    if skip_train:
        train_loader = None 
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=None if is_iterable else config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False if is_iterable else (not config.data.streaming),
            persistent_workers=config.loader.num_workers>0
        )
        train_loader.tokenizer = tokenizer 

    if skip_valid:
        valid_loader = None 
    else:
        if valid_seed is None:
            shuffle_valid = False 
            generator = None 
        else:
            shuffle_valid = True  
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=None if is_iterable else config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False if is_iterable else (shuffle_valid and not config.data.streaming),
            generator=generator)
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader 
