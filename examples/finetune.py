#!/usr/bin/env python3
"""
train_with_misaki_shards.py

Optimized training script with:
- Fast batched preprocessing with multiprocessing
- Efficient shard loading
- Better error handling
- Progress tracking

Usage:
    python train_with_misaki_shards.py config.yaml
"""
import os
import re
import sys
import shutil
import warnings
from functools import partial
from typing import Dict, List, Any

# Suppress warnings early
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from omegaconf import OmegaConf
from loguru import logger as LOGGER

# ML libs
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm

# Misaki G2P
try:
    from misaki import en as misaki_en
except Exception as e:
    LOGGER.error("Failed to import misaki. Install with: pip install \"misaki[en]\"")
    raise

# Regexes for filtering
ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){3,}")


def make_shard_filenames(start_idx: int, end_idx: int, prefix="train", total_shards=241):
    """Generate shard filenames (zero-padded). start/end inclusive."""
    if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
        raise ValueError(f"Invalid shard range: {start_idx}-{end_idx}")
    return [f"{prefix}-{i:05d}-of-{total_shards:05d}.parquet" for i in range(start_idx, end_idx + 1)]


def _detect_espeak_binary():
    """Look for espeak-ng or espeak on PATH."""
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            return name
    return None


def data_filter(sample: Dict) -> bool:
    """Fast filtering of invalid samples."""
    text = sample.get("text", "")
    if not text or len(text) < 5:
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


def preprocess_batch(batch: Dict, tokenizer, max_len: int, g2p, ignore_index: int = -100):
    """
    Batched preprocessing for speed.
    Returns dict with lists of input_ids, labels, attention_mask.
    """
    batch_size = len(batch["text"])
    
    all_input_ids = []
    all_labels = []
    all_attention_mask = []
    
    speech_gen_start_tok = "<|SPEECH_GENERATION_START|>"
    speech_gen_start_id = tokenizer.convert_tokens_to_ids(speech_gen_start_tok)
    pad_id = tokenizer.pad_token_id
    
    for i in range(batch_size):
        text = batch["text"][i]
        vq_codes = batch["codes"][i]
        
        # G2P conversion
        try:
            phonemes, _ = g2p(text)
            if not phonemes:
                continue
            phones = " ".join(str(phonemes).split())
        except Exception:
            continue
        
        # Build prompt
        codes_str = "".join([f"<|speech_{code}|>" for code in vq_codes])
        chat = (
            "user: Convert the text to speech:"
            "<|TEXT_PROMPT_START|>"
            f"{phones}"
            "<|TEXT_PROMPT_END|>\n"
            "assistant:"
            f"{speech_gen_start_tok}"
            f"{codes_str}<|SPEECH_GENERATION_END|>"
        )
        
        # Tokenize
        enc = tokenizer(chat, add_special_tokens=False)
        ids = enc["input_ids"]
        
        # Pad or truncate
        if len(ids) < max_len:
            ids = ids + [pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        
        # Create labels (only supervise after speech generation start)
        labels = [ignore_index] * len(ids)
        try:
            start_idx = ids.index(speech_gen_start_id)
            labels[start_idx:] = ids[start_idx:]
        except ValueError:
            pass
        
        attention_mask = [0 if x == pad_id else 1 for x in ids]
        
        all_input_ids.append(ids)
        all_labels.append(labels)
        all_attention_mask.append(attention_mask)
    
    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask,
    }


def load_selected_shards(config, start_idx: int, end_idx: int) -> Dataset:
    """Load dataset shards from local or HF Hub."""
    total_shards = config.get("data", {}).get("total_shards", 241)
    prefix = config.get("data", {}).get("prefix", "train")
    names = make_shard_filenames(start_idx, end_idx, prefix=prefix, total_shards=total_shards)
    
    use_hf_hub = config.get("data", {}).get("use_hf_hub", False)
    
    if use_hf_hub:
        dataset_name = config.get("data", {}).get("hf_dataset_name", "neuphonic/emilia-yodas-english-neucodec")
        LOGGER.info(f"Loading from HF Hub: {dataset_name}")
        
        # Load full dataset (shards are already combined in HF datasets)
        ds = load_dataset(dataset_name, split="train", streaming=False)
        LOGGER.info(f"Loaded {len(ds)} examples from HF Hub")
        
    else:
        local_dir = config.get("data", {}).get("local_dir", "./data")
        paths = [os.path.join(local_dir, n) for n in names]
        
        # Check for missing files
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            LOGGER.error(f"Missing {len(missing)} shard files:")
            for p in missing[:3]:
                LOGGER.error(f"  - {p}")
            raise FileNotFoundError(f"Missing shard files in {local_dir}")
        
        LOGGER.info(f"Loading {len(paths)} local shards from {local_dir}")
        ds = load_dataset("parquet", data_files=paths, split="train")
    
    return ds


def setup_tokenizer_and_model(restore_from: str):
    """Initialize tokenizer and model with special tokens."""
    LOGGER.info(f"Loading tokenizer and model from: {restore_from}")
    
    tokenizer = AutoTokenizer.from_pretrained(restore_from, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        restore_from, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    # Setup pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            LOGGER.info("Set pad_token = eos_token")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            LOGGER.info("Added [PAD] as pad token")
    
    # Add special tokens
    special_tokens = {
        "additional_special_tokens": [
            "<|TEXT_PROMPT_START|>",
            "<|TEXT_PROMPT_END|>",
            "<|SPEECH_GENERATION_START|>",
            "<|SPEECH_GENERATION_END|>",
            "<|SPEECH_REPLACE|>",
        ]
    }
    
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        LOGGER.info(f"Added {num_added} special tokens and resized embeddings")
    
    return tokenizer, model


def build_g2p_from_config(misaki_cfg: Dict) -> Any:
    """Create Misaki G2P instance with optional espeak fallback."""
    use_trf = bool(misaki_cfg.get("use_transformer", False))
    british = bool(misaki_cfg.get("british", False))
    fallback_opt = misaki_cfg.get("fallback", None)
    
    fallback_obj = None
    if fallback_opt and str(fallback_opt).lower() == "espeak":
        try:
            from misaki import espeak as misaki_espeak
            bin_name = _detect_espeak_binary()
            if bin_name:
                fallback_obj = misaki_espeak.EspeakFallback(british=british, program_name=bin_name)
                LOGGER.info(f"Created EspeakFallback with binary: {bin_name}")
            else:
                LOGGER.warning("espeak binary not found, continuing without fallback")
        except Exception as e:
            LOGGER.warning(f"Failed to setup EspeakFallback: {e}")
    
    g2p = misaki_en.G2P(trf=use_trf, british=british, fallback=fallback_obj)
    LOGGER.info(f"Initialized Misaki G2P (transformer={use_trf}, british={british})")
    return g2p


def main(config_path: str):
    """Main training loop."""
    # Load config
    config = OmegaConf.load(config_path)
    LOGGER.info(f"Loaded config from: {config_path}")
    
    # Parse shard range
    start = int(config.get("data", {}).get("start", 1))
    end = int(config.get("data", {}).get("end", 19))
    if end < start:
        raise ValueError(f"Invalid shard range: start={start}, end={end}")
    
    LOGGER.info(f"Training on shards {start} to {end} (inclusive)")
    
    # Load dataset
    LOGGER.info("Loading dataset shards...")
    ds = load_selected_shards(config, start, end)
    LOGGER.info(f"Loaded {len(ds):,} examples")
    
    # Filter data
    LOGGER.info("Filtering dataset...")
    original_size = len(ds)
    ds = ds.filter(data_filter, num_proc=config.get("num_proc", 8))
    filtered_size = len(ds)
    LOGGER.info(f"Filtered: {original_size:,} -> {filtered_size:,} ({filtered_size/original_size*100:.1f}% kept)")
    
    # Setup tokenizer and model
    restore_from = config.get("restore_from")
    tokenizer, model = setup_tokenizer_and_model(restore_from)
    
    # Setup G2P
    misaki_cfg = config.get("misaki", {}) or {}
    g2p = build_g2p_from_config(misaki_cfg)
    
    # Preprocess dataset with batching
    LOGGER.info("Preprocessing dataset (this may take a few minutes)...")
    batch_size = config.get("map_batch_size", 1000)
    num_proc = config.get("num_proc", 8)
    
    preprocess_fn = partial(
        preprocess_batch,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        g2p=g2p
    )
    
    ds = ds.map(
        preprocess_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Preprocessing"
    )
    
    LOGGER.info(f"Preprocessed dataset size: {len(ds):,} examples")
    
    # Setup training
    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info(f"Checkpoints will be saved to: {checkpoints_dir}")
    
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=torch.cuda.is_available(),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        optim=config.get("optimizer", "adamw_torch"),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        report_to=["none"],  # Disable wandb/tensorboard unless configured
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds,
        data_collator=default_data_collator,
    )
    
    LOGGER.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_path = os.path.join(checkpoints_dir, "final")
    trainer.save_model(final_path)
    LOGGER.info(f"Training complete! Model saved to: {final_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_with_misaki_shards.py config.yaml")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    if not os.path.exists(cfg_path):
        print(f"Error: Config file not found: {cfg_path}")
        sys.exit(1)
    
    main(cfg_path)
