"""
Build a (toy) conditional binarized-MNIST dataset in the same on-disk format
used by `PuzzleDataset` in `dataloader.py` (i.e. the sudoku / maze layout).

Image preprocessing:
  * MNIST images start as 28x28, pixels in [0, 255].
  * Downsampled 2x2 -> 14x14 by taking the top-left pixel of each block.
  * Binarized to {0, 1} via a fixed threshold (pixel > threshold -> 1).
    This is *static* binarization (baked into the file, fully reproducible),
    not the dynamic/stochastic kind used by the LL benchmark.

Task setup (conditional generation, analogous to sudoku):
  * The model is *conditioned* on the top half (first 7 of the 14 rows) of the
    image and must generate the full image.
        - `inputs` : top half = pixel value + 1 (tokens 1..2),
                     bottom half = 0  (PAD / EMPTY, the part to be generated)
        - `labels` : the full image, pixel value + 1 (tokens 1..2)
  * `dataloader._collate_batch` concatenates `[inputs, labels]` into `input_ids`
    and sets `valid_tokens = [zeros_like(inputs), label_mask]`, so the loss is
    only taken on the full image (the label half), exactly like sudoku. Note
    that the top half of the label is a verbatim copy of the input, so ~half the
    label tokens are a trivial copy task -- grade the generated bottom half
    separately if you want a clean generation metric.

Pixel-space tokenizer convention:
  * token id 0    -> PAD / EMPTY
  * token id 1..2 -> pixel values 0..1   (recover pixel via `token_id - 1`)
  * vocab_size = 3

Determinism:
  * Pass `--seed` to make the (optional) subsampling reproducible. Note that
    the *training order* is governed by `config.seed` in `PuzzleDataset`
    (deterministic given the same seed), so the same seed always yields the
    same iteration order.

Usage:
    cd dataset_code
    python build_mnist_dataset.py --output-dir data/mnist --subsample-size 1000
"""

from typing import Optional, Tuple
import json
import os

import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from dataset_common import PuzzleDatasetMetadata

cli = ArgParser()

IMG_SIZE = 14
SEQ_LEN = IMG_SIZE * IMG_SIZE          # 196
NUM_PIXEL_VALUES = 2                    # binarized: pixels are 0 or 1
PAD_ID = 0
VOCAB_SIZE = NUM_PIXEL_VALUES + 1      # PAD + {0, 1} = 3


class DataProcessConfig(BaseModel):
    # HuggingFace dataset id holding MNIST. "ylecun/mnist" is the canonical id;
    # "mnist" is kept as a fallback for older `datasets` caches.
    source_repo: str = "ylecun/mnist"
    output_dir: str = "data/mnist"

    # Keep it a *toy* dataset by default: subsample the (large) train split.
    # Set to None to use every example.
    subsample_size: Optional[int] = None 

    # Controls reproducibility of the subsampling.
    seed: int = 0
    
    # Pixels with value > threshold become 1, else 0. MNIST is near-bimodal
    # so this is not sensitive; 127 == ">0.5 of full scale".
    binarize_threshold: int = 127

def downsample_2x2(x):
    """Subsample the top-left pixel of each 2x2 block: 28x28 -> 14x14 (uint8)."""
    n, h, w = x.shape
    x = x.reshape(n, h // 2, 2, w // 2, 2)
    return x[:, :, 0, :, 0]

def binarize(x, threshold):
    """Threshold to {0, 1}: pixel > threshold -> 1 else 0 (uint8)."""
    return (x > threshold).astype(np.uint8)

def _load_mnist_split(source_repo: str, set_name: str, binarize_threshold: int) -> np.ndarray:
    """Return MNIST images for a split as a uint8 {0,1} array of shape (N, 14, 14)."""
    import datasets

    split = "train" if set_name == "train" else "test"
    try:
        ds = datasets.load_dataset(source_repo, split=split)
    except Exception:
        ds = datasets.load_dataset("mnist", split=split)

    images = np.stack([np.asarray(img, dtype=np.uint8) for img in ds["image"]])
    images = downsample_2x2(images)              # 28x28 -> 14x14
    images = binarize(images, binarize_threshold)  # {0, 1}
    assert images.shape[1:] == (IMG_SIZE, IMG_SIZE), f"unexpected image shape {images.shape}"
    return images


def _image_to_input_label(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Turn one (14, 14) uint8 image into flat (inputs, labels) token rows.

    inputs : top half tokenized (pixel + 1), bottom half = PAD (0).
    labels : full image tokenized (pixel + 1).
    """
    tokens = image.astype(np.int32) + 1            # pixels 0..255 -> tokens 1..256

    inputs = tokens.copy()
    inputs[IMG_SIZE // 2:, :] = PAD_ID             # zero out the bottom half rows

    labels = tokens                                # full original image
    return inputs.reshape(-1), labels.reshape(-1)


def convert_subset(set_name: str, config: DataProcessConfig):
    images = _load_mnist_split(config.source_repo, set_name, config.binarize_threshold)


    # Optional (deterministic) subsampling of the train split to keep it toy-sized.
    if set_name == "train" and config.subsample_size is not None and config.subsample_size < len(images):
        rng = np.random.default_rng(config.seed)
        indices = rng.choice(len(images), size=config.subsample_size, replace=False)
        images = images[indices]

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for image in tqdm(images):
        inp, out = _image_to_input_label(image)
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1

        # One example per puzzle, one puzzle per group (mirrors sudoku/maze).
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)

    results = {
        "inputs": np.vstack(results["inputs"]).astype(np.int32),
        "labels": np.vstack(results["labels"]).astype(np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Sanity: token ids stay within [0, VOCAB_SIZE).
    assert results["inputs"].min() >= 0 and results["inputs"].max() < VOCAB_SIZE
    assert results["labels"].min() >= 1 and results["labels"].max() < VOCAB_SIZE

    metadata = PuzzleDatasetMetadata(
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,            # PAD + 256 pixel values
        pad_id=PAD_ID,
        ignore_label_id=0,               # labels are 1..256, so 0 never collides
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # IDs mapping (for visualization only).
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()