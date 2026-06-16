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

LOGGER = utils.get_logger(__name__)

def original_cwd():
    """get_original_cwd() when under @hydra.main, else just the real cwd."""
    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.utils import get_original_cwd
        if HydraConfig.initialized():
            return get_original_cwd()
    except (ImportError, ValueError):
        pass
    return os.getcwd()

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
    
    def decode(self, sample): #convert sequence to string
        if torch.is_tensor(sample):
            sample = sample.tolist()
        return ''.join(str(c) for c in sample)


def get_tokenizer(config):
    if config.data.tokenizer_name_or_path in ("sudoku-extreme", "sudoku"):
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

def _infill_view(puzzle, solution, pad_id, loss_region='fill'):
    """Build the in-place *infilling* view of a sudoku example.

    Instead of the default ``[puzzle | solution]`` layout (where the model is
    conditioned on a prepended puzzle and predicts the appended solution), the
    model sees the *solution* grid directly together with a ``conditioning_mask``
    that marks which cells were given as clues (held clean during diffusion) and
    which it must fill in.

    This relies on the sudoku token convention shared by
    ``generate_sudoku_dataset`` and ``build_sudoku_dataset`` (both use
    ``value_offset=1``): ``pad -> pad_id``, blank cell -> ``pad_id + 1``, given
    digits 1..9 -> ``pad_id + 2 .. pad_id + 10``. A cell is therefore a clue iff
    its puzzle value is neither pad nor blank.

    The ``conditioning_mask`` (cells held clean) and ``valid_tokens`` (cells in
    the loss) are independent: the conditioning is always exactly the given
    clues, while ``loss_region`` selects what the loss covers.

    Args:
        puzzle: (..., L) int array of puzzle cells (clues + blanks).
        solution: (..., L) int array of the fully solved grid.
        pad_id: padding token id (blank cell id is ``pad_id + 1``).
        loss_region: which cells contribute to the loss:
          - "fill" (default): only the blank cells the model must predict;
          - "board": every non-pad cell (clues + blanks), so the loss also
            scores the clamped clue cells.
    Returns:
        (input_ids, conditioning_mask, valid_tokens), each (..., L):
          - input_ids: the solution grid (what the model operates on),
          - conditioning_mask: 1 on given clue cells (kept clean),
          - valid_tokens: 1 on the cells in the loss (per ``loss_region``).
    """
    empty_id = pad_id + 1
    is_blank = (puzzle == empty_id)                                   # cells to fill
    conditioning_mask = (puzzle != pad_id) & (puzzle != empty_id)     # given clues
    if loss_region == "board":
        valid_tokens = (puzzle != pad_id)                            # all real cells
    elif loss_region == "fill":
        valid_tokens = is_blank
    else:
        raise ValueError(
            f"Unknown infill loss_region {loss_region!r}; expected 'fill' or 'board'.")
    return (solution,
            conditioning_mask.astype(np.int32),
            valid_tokens.astype(np.int32))

class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int
    # False -> [puzzle | solution] layout; True -> solution grid +
    # conditioning_mask (see _infill_view). Driven by config.algo.infill.
    infill: bool = False
    # For the infill format, which cells contribute to the loss: "fill" (blanks
    # only) or "board" (all non-pad cells). Unused for the prepend format.
    infill_loss_region: str = "fill"

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
        with open(os.path.join(original_cwd(), dataset_path, self.split, "dataset.json"), "r") as f:
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
                    field_name: np.load(os.path.join(original_cwd(), dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }
                
    def _collate_batch(self, batch):
        """
        Returns {'input_ids', 'valid_tokens'} or {'input_ids', 'valid_tokens', 'conditioning_mask'}.
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

        if self.config.infill:
            input_ids, conditioning_mask, valid_tokens = _infill_view(
                inputs, labels, self.metadata.pad_id,
                loss_region=self.config.infill_loss_region
            )
            return {
                "input_ids": torch.from_numpy(input_ids).long(),
                "valid_tokens": torch.from_numpy(valid_tokens),
                "conditioning_mask" : torch.from_numpy(conditioning_mask)
            }
        
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
    infill = getattr(config.algo, "infill", False)
    infill_loss_region = getattr(config.data, "infill_loss_region", "fill")
    
    train_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data.data_paths,
        rank=rank,
        num_replicas=world_size,
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.loader.batch_size * world_size,
        infill=infill,
        infill_loss_region=infill_loss_region 
    ), split='train')
    test_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data.data_paths_test if len(config.data.data_paths_test)>0  else config.data.data_paths,
        rank=rank,
        num_replicas=world_size,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.loader.eval_batch_size * world_size,
        infill=infill,
        infill_loss_region=infill_loss_region 
    ), split='test')

    return {
        'train': train_dataset, 
        'test': test_dataset
    }

class SudokuGeneratedDataset(torch.utils.data.Dataset):
    """Map-style dataset over generated sudoku examples.

    Each item is either a flat [puzzle(81) | solution(81)] sequence with a
    matching valid_tokens mask (0 over the puzzle/conditioning half,
    1 over the solution half) OR
        [solution(81)] sequence with a conditioning mask on hints, and 
        valid tokens on either the entire board or only empty cells to fill 
    """
    def __init__(self, data, infill=False, infill_loss_region='fill'):
        # Generated sudoku uses value_offset=1, so pad=0 and the blank cell is 1.
        self.PAD_ID = 0
        self.input_ids = data["input_ids"]
        self.valid_tokens = data["valid_tokens"]
        self.infill=infill
        self.infill_loss_region = infill_loss_region

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.infill:
            # Stored layout is [puzzle | solution]; derive the in-place infilling
            # view (solution grid + conditioning_mask over the given clues).
            ids = np.asarray(self.input_ids[idx])
            half = len(ids) // 2
            input_ids, conditioning_mask, valid_tokens = _infill_view(
                ids[:half], ids[half:], self.PAD_ID,
                loss_region=self.infill_loss_region)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "valid_tokens": torch.tensor(valid_tokens, dtype=torch.long),
                "conditioning_mask": torch.tensor(conditioning_mask, dtype=torch.long),
            }

        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "valid_tokens": torch.tensor(self.valid_tokens[idx], dtype=torch.long),
        }
    
# Cache generated splits so the train and (separate) valid calls reuse the
# same deterministic, deduplicated generation instead of regenerating.
_SUDOKU_SPLIT_CACHE: Dict[tuple, dict] = {}
_SUDOKU_SPLITS = ("train", "validation")

def _sudoku_cache_dir(config):
    """Deterministic on-disk location for a given generation config.

    Anchored to original_cwd() (not the hydra run dir) so the cache is shared
    across runs, and keyed by the params that affect generation so changing
    any of them produces a fresh dataset.
    """
    data_cfg = config.data
    base = getattr(data_cfg, "gen_output_dir", "data/sudoku-gen")
    name = (f"{data_cfg.difficulty}"
            f"_train{data_cfg.num_train}"
            f"_valid{data_cfg.num_valid}"
            f"_seed{config.seed}")
    return os.path.join(original_cwd(), base, name)


def _load_sudoku_from_disk(cache_dir):
    splits = {}
    for split in _SUDOKU_SPLITS:
        splits[split] = {
            "input_ids": np.load(os.path.join(cache_dir, f"{split}__input_ids.npy")),
            "valid_tokens": np.load(os.path.join(cache_dir, f"{split}__valid_tokens.npy")),
        }
    return splits


def _disk_cache_complete(cache_dir):
    return all(
        os.path.exists(os.path.join(cache_dir, f"{split}__{field}.npy"))
        for split in _SUDOKU_SPLITS
        for field in ("input_ids", "valid_tokens"))


def _save_sudoku_to_disk(cache_dir, splits, config):
    # Write to a temp dir then atomically rename, so concurrent ranks never
    # observe a half-written cache (generation is deterministic, so a losing
    # writer's identical result is simply discarded).
    parent = os.path.dirname(cache_dir) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_dir = f"{cache_dir}.tmp.{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        for split in _SUDOKU_SPLITS:
            # Values are 0..10 and the mask is 0/1, so uint8 is plenty.
            np.save(os.path.join(tmp_dir, f"{split}__input_ids.npy"),
                    np.asarray(splits[split]["input_ids"], dtype=np.uint8))
            np.save(os.path.join(tmp_dir, f"{split}__valid_tokens.npy"),
                    np.asarray(splits[split]["valid_tokens"], dtype=np.uint8))
        with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
            json.dump({
                "num_train": config.data.num_train,
                "num_valid": config.data.num_valid,
                "difficulty": config.data.difficulty,
                "seed": config.seed,
            }, f)
        os.replace(tmp_dir, cache_dir)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

def _build_sudoku_splits(config):
    from dataset_code.generate_sudoku_dataset import generate_sudoku_dataset

    data_cfg = config.data
    num_train = data_cfg.num_train
    num_valid = data_cfg.num_valid
    difficulty = data_cfg.difficulty
    num_workers = getattr(data_cfg, "gen_num_workers", 1)
    key = (num_train, num_valid, difficulty, config.seed, num_workers)
    if key in _SUDOKU_SPLIT_CACHE:
        return _SUDOKU_SPLIT_CACHE[key]

    cache_dir = _sudoku_cache_dir(config)
    if _disk_cache_complete(cache_dir):
        LOGGER.info("Loading cached sudoku dataset from %s", cache_dir)
        splits = _load_sudoku_from_disk(cache_dir)
    else:
        LOGGER.info(
            "Generating sudoku dataset (train=%d, valid=%d, difficulty=%s, "
            "seed=%d) -> caching to %s",
            num_train, num_valid, difficulty, config.seed, cache_dir)
        splits = generate_sudoku_dataset(
            num_train=num_train,
            num_valid=num_valid,
            difficulty=difficulty,
            seed=config.seed,
            num_workers=num_workers,
        )
        _save_sudoku_to_disk(cache_dir, splits, config)

    _SUDOKU_SPLIT_CACHE[key] = splits
    return splits

def _deterministic_subset_indices(num_total, subset_n, subset_seed):
    """Pick `subset_n` example indices deterministically.

    The selection depends only on (num_total, subset_n, subset_seed), NOT on
    the training `seed`, so the same examples are reused across training runs
    that vary the training seed. Returns sorted indices to preserve the
    underlying (already deterministic) example order.
    """
    rng = np.random.Generator(np.random.Philox(seed=subset_seed))
    chosen = rng.choice(num_total, size=subset_n, replace=False)
    return np.sort(chosen).tolist()


def get_sudoku_dataset(config, mode, rank=0, world_size=1):
    """Return a map-style generated-sudoku split.

    Deliberately kept separate from get_puzzle_dataset: that path streams
    pre-built .npy shards via index machinery, whereas this one generates
    examples in-memory. Both expose the same {input_ids, valid_tokens}
    batch fields downstream.
    """
    splits = _build_sudoku_splits(config)
    split = "train" if mode == "train" else "validation"
    infill = getattr(config.algo, "infill", False)
    infill_loss_region = getattr(config.data, "infill_loss_region", "fill")
    dataset = SudokuGeneratedDataset(splits[split], infill=infill,
                infill_loss_region=infill_loss_region)
    
    if mode == 'train':
        subset_n = getattr(config.data, "train_subset_n", None)
        if subset_n is not None and 0 < subset_n < len(dataset):
            subset_seed = getattr(config.data, "subset_seed", 0)
            indices = _deterministic_subset_indices(
                len(dataset),subset_n, subset_seed 
            )
            LOGGER.info(
                f"subsetting sudoku train split to {subset_n}/{len(dataset)} examples"
            )
            dataset = torch.utils.data.Subset(dataset, indices)
    
    if world_size > 1:
        # Shard across ranks (no DistributedSampler is configured upstream).
        dataset = torch.utils.data.Subset(
            dataset, list(range(rank, len(dataset), world_size)))
    return dataset

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
    elif dataset_name == "sudoku":
        return get_sudoku_dataset(config, mode, rank, world_size)
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

# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py
class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop('shuffle', None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'random_state': self.generator.get_state(),
                'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get('random_state'))
        self.counter = state_dict['counter']
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(
                    padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0
