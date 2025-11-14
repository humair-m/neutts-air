#!/usr/bin/env python3
"""
train_with_misaki_shards.py

- Loads only a contiguous range of parquet shards from the emilia dataset (start..end inclusive)
  - If local_dir is provided in config, reads files from there.
  - Otherwise downloads shards from HF Hub (base URL is set by default).
- Uses Misaki G2P with optional espeak fallback (enable via config).
- Safe token/special-token handling and hf.datasets-friendly preprocess (returns lists).
- Training using HuggingFace Trainer.

Usage:
    python train_with_misaki_shards.py config.yaml
"""
import os
import re
import shutil
import warnings
from functools import partial

from omegaconf import OmegaConf
from loguru import logger as LOGGER

warnings.filterwarnings("ignore")

# ML libs
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset

# Misaki (English). If misaki not installed this will raise early.
try:
    from misaki import en as misaki_en
except Exception as e:
    LOGGER.error("Failed to import misaki. Install with: pip install \"misaki[en]\"")
    raise

# Regexes for filtering
ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){3,}")


def make_shard_filenames(start_idx: int, end_idx: int, prefix="train", total_shards=241):
    """Return list of shard file names (zero-padded). start/end inclusive. Example: train-00001-of-00241.parquet."""
    # validate
    if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
        raise ValueError("Invalid start/end indices")
    return [f"{prefix}-{i:05d}-of-{total_shards:05d}.parquet" for i in range(start_idx, end_idx + 1)]


def _detect_espeak_binary():
    """Look for either 'espeak-ng' or 'espeak' on PATH. Return binary name or None."""
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            return name
    return None


def data_filter(sample):
    text = sample.get("text", "")
    if not text:
        return False
    if re.search(r"\d", text):
        return False
    if re.search(ACRONYM, text) or re.search(ACRONYM_NO_PERIOD, text):
        return False
    if text[-1] not in ".,?!":
        return False
    if "£" in text or "$" in text:
        return False
    return True


def preprocess_sample(sample, tokenizer, max_len, g2p):
    """
    Return Python lists (not torch tensors) for datasets.map compatibility.
    g2p callable -> (phonemes, tokens) = g2p(text)
    """
    speech_gen_start_tok = "<|SPEECH_GENERATION_START|>"
    ignore_index = -100
    pad_id = tokenizer.pad_token_id

    vq_codes = sample.get("codes", [])
    text = sample.get("text", "")

    try:
        phonemes, _tokens = g2p(text)
    except Exception as e:
        LOGGER.warning(f"g2p failed for sample {sample.get('__key__')}: {e}")
        return None

    if not phonemes:
        LOGGER.warning(f"Empty phonemization for sample {sample.get('__key__')}: {text}")
        return None

    phones = " ".join(str(phonemes).split())
    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])

    chat = (
        "user: Convert the text to speech:"
        "<|TEXT_PROMPT_START|>"
        f"{phones}"
        "<|TEXT_PROMPT_END|>\n"
        "assistant:"
        f"{speech_gen_start_tok}"
        f"{codes_str}<|SPEECH_GENERATION_END|>"
    )

    enc = tokenizer(chat, add_special_tokens=False)
    ids = enc["input_ids"]

    # pad/truncate
    if len(ids) < max_len:
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id is None; set / add a pad token")
        ids = ids + [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    labels = [ignore_index] * len(ids)
    speech_gen_start_id = tokenizer.convert_tokens_to_ids(speech_gen_start_tok)
    try:
        start_idx = ids.index(speech_gen_start_id)
        labels[start_idx:] = ids[start_idx:]
    except ValueError:
        LOGGER.debug("speech start token not found in tokenized ids")

    attention_mask = [0 if x == pad_id else 1 for x in ids]
    return {"input_ids": ids, "labels": labels, "attention_mask": attention_mask}


def load_selected_shards(config, start_idx: int, end_idx: int):
    """
    Load dataset shards either from local_dir or HF hub.
    Returns a datasets.Dataset object (split 'train').
    """
    total_shards = config.get("data", {}).get("total_shards", 241)
    prefix = config.get("data", {}).get("prefix", "train")
    names = make_shard_filenames(start_idx, end_idx, prefix=prefix, total_shards=total_shards)

    use_hf_hub = config.get("data", {}).get("use_hf_hub", False)
    if use_hf_hub:
        base = config.get("data", {}).get(
            "hf_base",
            "https://huggingface.co/datasets/neuphonic/emilia-yodas-english-neucodec/resolve/main",
        )
        urls = [f"{base}/{n}" for n in names]
        LOGGER.info(f"Loading {len(urls)} shards from HF Hub (HTTP).")
        ds = load_dataset("parquet", data_files=urls, split="train")
    else:
        local_dir = config.get("data", {}).get("local_dir", "./data")
        paths = [os.path.join(local_dir, n) for n in names]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            LOGGER.error("Missing local shard files. Missing examples:\n" + "\n".join(missing))
            raise FileNotFoundError(f"Missing shard files. Check data.local_dir in config. Missing {len(missing)} files.")
        LOGGER.info(f"Loading {len(paths)} local parquet shards from {local_dir}")
        ds = load_dataset("parquet", data_files=paths, split="train")

    return ds


def setup_tokenizer_and_model(restore_from):
    tokenizer = AutoTokenizer.from_pretrained(restore_from, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(restore_from, torch_dtype="auto")

    # ensure pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            LOGGER.info("No pad token found; set pad_token = eos_token")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            LOGGER.info("Added [PAD] as pad token")

    special_tokens = {
        "additional_special_tokens": [
            "<|TEXT_PROMPT_START|>",
            "<|TEXT_PROMPT_END|>",
            "<|SPEECH_GENERATION_START|>",
            "<|SPEECH_GENERATION_END|>",
            "<|SPEECH_REPLACE|>",
        ]
    }
    added = tokenizer.add_special_tokens(special_tokens)
    if added:
        model.resize_token_embeddings(len(tokenizer))
        LOGGER.info(f"Added {added} special tokens and resized model embeddings.")

    return tokenizer, model


def build_g2p_from_config(misaki_cfg):
    """
    Create misaki_en.G2P(...) instance using config dict:
       misaki_cfg: {use_transformer: bool, british: bool, fallback: "espeak" or None}
    If fallback is "espeak", try to instantiate misaki.espeak.EspeakFallback (lazily imported).
    """
    use_trf = bool(misaki_cfg.get("use_transformer", False))
    british = bool(misaki_cfg.get("british", False))
    fallback_opt = misaki_cfg.get("fallback", None)

    fallback_obj = None
    if fallback_opt and str(fallback_opt).lower() == "espeak":
        # lazy import misaki.espeak (some installs may not have this submodule)
        try:
            from misaki import espeak as misaki_espeak
            bin_name = _detect_espeak_binary()
            try:
                if bin_name:
                    fallback_obj = misaki_espeak.EspeakFallback(british=british, program_name=bin_name)
                else:
                    fallback_obj = misaki_espeak.EspeakFallback(british=british)
                LOGGER.info("Created Misaki EspeakFallback for G2P fallback.")
            except Exception as e:
                LOGGER.warning(f"Failed to instantiate EspeakFallback: {e}. Continuing without fallback.")
                fallback_obj = None
        except Exception as e:
            LOGGER.warning(f"misaki.espeak submodule not available: {e}. Install optional deps or skip fallback.")
            fallback_obj = None

    # instantiate main G2P
    try:
        g2p = misaki_en.G2P(trf=use_trf, british=british, fallback=fallback_obj)
    except Exception as e:
        LOGGER.error(f"Failed to initialize misaki.en.G2P: {e}")
        raise
    return g2p


def main(config_path):
    # Load config
    config = OmegaConf.load(config_path)

    # required config fields & defaults
    start = int(config.get("data", {}).get("start", 1))
    end = int(config.get("data", {}).get("end", 19))
    if end < start:
        raise ValueError("data.end must be >= data.start")

    LOGGER.info(f"Requested shard range: {start} .. {end}")

    # load only selected shards
    ds = load_selected_shards(config, start, end)
    LOGGER.info(f"Loaded dataset with {len(ds)} examples from selected shards.")

    # filter if user wants
    ds = ds.filter(data_filter)

    # tokenizer + model
    restore_from = config.get("restore_from")
    tokenizer, model = setup_tokenizer_and_model(restore_from)

    # misaki G2P
    misaki_cfg = config.get("misaki", {}) or {}
    g2p = build_g2p_from_config(misaki_cfg)

    # map preprocess
    partial_pre = partial(preprocess_sample, tokenizer=tokenizer, max_len=config.max_seq_len, g2p=g2p)
    ds = ds.map(partial_pre, remove_columns=["text", "codes"], batched=False)

    # training args
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info(f"Checkpoints will be saved to: {checkpoints_dir}")

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        fp16=True if getattr(config, "use_fp16", True) else False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        ignore_data_skip=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=getattr(config, "dataloader_num_workers", 8),
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(checkpoints_dir)
    LOGGER.info("Training complete and model saved.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_with_misaki_shards.py config.yaml")
        sys.exit(1)
    cfg_path = sys.argv[1]
    main(cfg_path)
