#!/usr/bin/env python3
"""
train_optimized.py

High-performance training script with:
- Ultra-fast parallel tokenization
- Efficient memory management
- Optimized batch processing
- Better error handling and logging

Usage:
    python train_optimized.py config.yaml
"""
import os
import re
import sys
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable for fast tokenizers

from omegaconf import OmegaConf
from loguru import logger

import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm.auto import tqdm
import numpy as np

# Misaki G2P
try:
    from misaki import en as misaki_en
except ImportError as e:
    logger.error("Failed to import misaki. Install with: pip install 'misaki[en]'")
    raise

# Compiled regex patterns for speed
PATTERNS = {
    "digit": re.compile(r"\d"),
    "acronym": re.compile(r"(?:[a-zA-Z]\.){2,}"),
    "acronym_no_period": re.compile(r"(?:[A-Z]){3,}"),
    "currency": re.compile(r"[£$€¥]"),
}


@dataclass
class TrainingConfig:
    """Structured config holder."""
    # Data
    start_shard: int
    end_shard: int
    total_shards: int = 241
    shard_prefix: str = "train"
    local_dir: Optional[str] = None
    hf_dataset_name: Optional[str] = None
    use_hf_hub: bool = False
    
    # Model
    restore_from: str = None
    max_seq_len: int = 2048
    
    # Processing
    num_proc: int = 8
    map_batch_size: int = 2000
    filter_batch_size: int = 10000
    
    # Training
    run_name: str = "experiment"
    save_root: str = "./checkpoints"
    lr: float = 5e-5
    max_steps: int = 10000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    save_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    bf16: bool = True
    dataloader_num_workers: int = 4
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    
    # G2P
    use_transformer_g2p: bool = False
    british_english: bool = False
    use_espeak_fallback: bool = False
    
    @classmethod
    def from_omega(cls, cfg: OmegaConf):
        """Create from OmegaConf."""
        data_cfg = cfg.get("data", {})
        misaki_cfg = cfg.get("misaki", {})
        
        return cls(
            start_shard=int(data_cfg.get("start", 1)),
            end_shard=int(data_cfg.get("end", 19)),
            total_shards=int(data_cfg.get("total_shards", 241)),
            shard_prefix=data_cfg.get("prefix", "train"),
            local_dir=data_cfg.get("local_dir"),
            hf_dataset_name=data_cfg.get("hf_dataset_name"),
            use_hf_hub=bool(data_cfg.get("use_hf_hub", False)),
            restore_from=cfg.get("restore_from"),
            max_seq_len=int(cfg.get("max_seq_len", 2048)),
            num_proc=int(cfg.get("num_proc", 8)),
            map_batch_size=int(cfg.get("map_batch_size", 2000)),
            filter_batch_size=int(cfg.get("filter_batch_size", 10000)),
            run_name=cfg.get("run_name", "experiment"),
            save_root=cfg.get("save_root", "./checkpoints"),
            lr=float(cfg.get("lr", 5e-5)),
            max_steps=int(cfg.get("max_steps", 10000)),
            per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 4)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
            save_steps=int(cfg.get("save_steps", 500)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_total_limit=int(cfg.get("save_total_limit", 3)),
            gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
            bf16=bool(cfg.get("bf16", True)),
            dataloader_num_workers=int(cfg.get("dataloader_num_workers", 4)),
            optimizer=cfg.get("optimizer", "adamw_torch"),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            use_transformer_g2p=bool(misaki_cfg.get("use_transformer", False)),
            british_english=bool(misaki_cfg.get("british", False)),
            use_espeak_fallback=bool(misaki_cfg.get("fallback") == "espeak"),
        )


class FastG2PProcessor:
    """Fast G2P processor with caching."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.g2p = self._setup_g2p()
        self._cache = {}
    
    def _setup_g2p(self):
        """Initialize G2P with optional fallback."""
        fallback = None
        if self.config.use_espeak_fallback:
            try:
                from misaki import espeak as misaki_espeak
                bin_name = shutil.which("espeak-ng") or shutil.which("espeak")
                if bin_name:
                    fallback = misaki_espeak.EspeakFallback(
                        british=self.config.british_english,
                        program_name=bin_name
                    )
                    logger.info(f"Enabled espeak fallback: {bin_name}")
            except Exception as e:
                logger.warning(f"Espeak fallback failed: {e}")
        
        return misaki_en.G2P(
            trf=self.config.use_transformer_g2p,
            british=self.config.british_english,
            fallback=fallback
        )
    
    def process(self, text: str) -> Optional[str]:
        """Convert text to phonemes with caching."""
        if text in self._cache:
            return self._cache[text]
        
        try:
            phonemes, _ = self.g2p(text)
            if not phonemes:
                return None
            result = " ".join(str(phonemes).split())
            self._cache[text] = result
            return result
        except Exception:
            return None


class FastTokenizer:
    """Optimized tokenizer with batch processing."""
    
    SPECIAL_TOKENS = {
        "additional_special_tokens": [
            "<|TEXT_PROMPT_START|>",
            "<|TEXT_PROMPT_END|>",
            "<|SPEECH_GENERATION_START|>",
            "<|SPEECH_GENERATION_END|>",
            "<|SPEECH_REPLACE|>",
        ]
    }
    
    def __init__(self, tokenizer_path: str):
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,  # Use fast Rust-based tokenizer
            padding_side="right",
        )
        self._setup_special_tokens()
        
        # Cache special token IDs
        self.pad_id = self.tokenizer.pad_token_id
        self.speech_start_id = self.tokenizer.convert_tokens_to_ids(
            "<|SPEECH_GENERATION_START|>"
        )
    
    def _setup_special_tokens(self):
        """Add special tokens and setup padding."""
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        num_added = self.tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
        logger.info(f"Added {num_added} special tokens")
    
    def build_prompt(self, phonemes: str, vq_codes: List[int]) -> str:
        """Build training prompt."""
        codes_str = "".join([f"<|speech_{c}|>" for c in vq_codes])
        return (
            "user: Convert the text to speech:"
            "<|TEXT_PROMPT_START|>"
            f"{phonemes}"
            "<|TEXT_PROMPT_END|>\n"
            "assistant:"
            "<|SPEECH_GENERATION_START|>"
            f"{codes_str}<|SPEECH_GENERATION_END|>"
        )
    
    def tokenize_batch(
        self,
        prompts: List[str],
        max_length: int,
        return_labels: bool = True
    ) -> Dict[str, List[List[int]]]:
        """Fast batch tokenization using HF's batch_encode_plus."""
        # Use fast batch encoding
        encoded = self.tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors=None,  # Return lists for flexibility
        )
        
        result = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        
        if return_labels:
            # Create labels (supervise only after speech generation start)
            labels = []
            for ids in encoded["input_ids"]:
                label = [-100] * len(ids)
                try:
                    start_idx = ids.index(self.speech_start_id)
                    label[start_idx:] = ids[start_idx:]
                except ValueError:
                    pass
                labels.append(label)
            result["labels"] = labels
        
        return result


def fast_filter(batch: Dict, batch_size: int = 10000) -> List[bool]:
    """Vectorized filtering for speed."""
    texts = batch["text"]
    n = len(texts)
    
    # Pre-allocate boolean array
    keep = np.ones(n, dtype=bool)
    
    # Vectorized checks
    for i, text in enumerate(texts):
        if not text or len(text) < 5:
            keep[i] = False
            continue
        
        # Fast pattern checks
        if (PATTERNS["digit"].search(text) or
            PATTERNS["acronym"].search(text) or
            PATTERNS["acronym_no_period"].search(text) or
            PATTERNS["currency"].search(text)):
            keep[i] = False
            continue
        
        # Ending check
        if text[-1] not in ".,?!":
            keep[i] = False
    
    return keep.tolist()


def preprocess_batch_optimized(
    batch: Dict,
    g2p_processor: FastG2PProcessor,
    fast_tokenizer: FastTokenizer,
    max_len: int
) -> Dict[str, List]:
    """Optimized batch preprocessing with parallel G2P and fast tokenization."""
    batch_size = len(batch["text"])
    
    # Step 1: Parallel G2P processing
    prompts = []
    valid_indices = []
    
    for i in range(batch_size):
        phonemes = g2p_processor.process(batch["text"][i])
        if phonemes is None:
            continue
        
        prompt = fast_tokenizer.build_prompt(phonemes, batch["codes"][i])
        prompts.append(prompt)
        valid_indices.append(i)
    
    if not prompts:
        return {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
        }
    
    # Step 2: Fast batch tokenization
    tokenized = fast_tokenizer.tokenize_batch(
        prompts,
        max_length=max_len,
        return_labels=True
    )
    
    return tokenized


def load_shards_efficient(config: TrainingConfig) -> Dataset:
    """Load dataset shards efficiently."""
    if config.use_hf_hub:
        logger.info(f"Loading from HF Hub: {config.hf_dataset_name}")
        ds = load_dataset(
            config.hf_dataset_name,
            split="train",
            streaming=False,
            num_proc=config.num_proc
        )
        logger.info(f"Loaded {len(ds):,} examples")
        return ds
    
    # Load local shards
    shard_files = [
        f"{config.shard_prefix}-{i:05d}-of-{config.total_shards:05d}.parquet"
        for i in range(config.start_shard, config.end_shard + 1)
    ]
    
    paths = [Path(config.local_dir) / f for f in shard_files]
    missing = [p for p in paths if not p.exists()]
    
    if missing:
        logger.error(f"Missing {len(missing)} shard files")
        raise FileNotFoundError(f"Missing shards in {config.local_dir}")
    
    logger.info(f"Loading {len(paths)} local shards")
    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in paths],
        split="train",
        num_proc=config.num_proc
    )
    
    logger.info(f"Loaded {len(ds):,} examples")
    return ds


def setup_model(tokenizer: FastTokenizer, config: TrainingConfig):
    """Load and setup model."""
    logger.info(f"Loading model from: {config.restore_from}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.restore_from,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        use_cache=not config.gradient_checkpointing,
    )
    
    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer.tokenizer))
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    return model


def main(config_path: str):
    """Main training pipeline."""
    # Load config
    omega_cfg = OmegaConf.load(config_path)
    config = TrainingConfig.from_omega(omega_cfg)
    logger.info(f"Loaded config from: {config_path}")
    logger.info(f"Training shards {config.start_shard} to {config.end_shard}")
    
    # Load dataset
    logger.info("Loading dataset...")
    ds = load_shards_efficient(config)
    
    # Fast filtering
    logger.info("Filtering dataset...")
    original_size = len(ds)
    ds = ds.filter(
        fast_filter,
        batched=True,
        batch_size=config.filter_batch_size,
        num_proc=config.num_proc,
        desc="Filtering"
    )
    logger.info(
        f"Filtered: {original_size:,} -> {len(ds):,} "
        f"({len(ds)/original_size*100:.1f}% kept)"
    )
    
    # Setup processors
    logger.info("Initializing G2P and tokenizer...")
    g2p_processor = FastG2PProcessor(config)
    fast_tokenizer = FastTokenizer(config.restore_from)
    
    # Fast preprocessing
    logger.info("Preprocessing dataset...")
    ds = ds.map(
        lambda batch: preprocess_batch_optimized(
            batch, g2p_processor, fast_tokenizer, config.max_seq_len
        ),
        batched=True,
        batch_size=config.map_batch_size,
        num_proc=config.num_proc,
        remove_columns=ds.column_names,
        desc="Preprocessing",
        load_from_cache_file=False,  # Disable cache for consistency
    )
    
    # Remove empty examples
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0, desc="Removing empty")
    logger.info(f"Final dataset size: {len(ds):,} examples")
    
    if len(ds) == 0:
        logger.error("No valid examples after preprocessing!")
        return
    
    # Setup model
    model = setup_model(fast_tokenizer, config)
    
    # Training arguments
    output_dir = Path(config.save_root) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        learning_rate=config.lr,
        max_steps=config.max_steps,
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=not config.bf16 and torch.cuda.is_available(),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        optim=config.optimizer,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to=["none"],
        logging_first_step=True,
        ddp_find_unused_parameters=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=fast_tokenizer.tokenizer,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(
            tokenizer=fast_tokenizer.tokenizer,
            padding="longest",
        ),
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Total steps: {config.max_steps}")
    logger.info(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    
    trainer.train()
    
    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    logger.info(f"Training complete! Model saved to: {final_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_optimized.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    main(config_path)
