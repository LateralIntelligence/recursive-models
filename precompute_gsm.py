# scripts/precompute_tinygsm.py
"""Build the TinyGSM .npy cache once so every training rank starts warm.
Run from the same dir you launch training from (original_cwd() must match),
or set data.gen_output_dir to an absolute path."""
import hydra
from omegaconf import DictConfig
import os 
from pathlib import Path

import dataloader   # module with _build_tiny_gsm_splits, get_tokenizer, _tiny_gsm_cache_dir
import utils

LOGGER = utils.get_logger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    assert "tinygsm" in (config.data.train, config.data.valid), \
        "This script only precomputes the TinyGSM cache."

    tokenizer = dataloader.get_tokenizer(config)
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        assert getattr(tokenizer, attr) is not None, \
            f"tokenizer.{attr} is None — fix get_tokenizer before precomputing."

    tag = str(config.data.tokenizer_name_or_path).replace("/", "__")
    cache_dir = dataloader._tiny_gsm_cache_dir(config, tag)

    if dataloader._tiny_gsm_disk_complete(cache_dir):
        LOGGER.info("Cache already complete at %s — nothing to do.", cache_dir)
        return

    LOGGER.info("Building TinyGSM cache -> %s", cache_dir)
    dataloader._build_tiny_gsm_splits(config, tokenizer)   # tokenize, pad, split, write
    LOGGER.info("Done: %s", cache_dir)


if __name__ == "__main__":
    main()