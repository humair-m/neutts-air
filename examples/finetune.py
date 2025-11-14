#!/usr/bin/env python3
"""
train_with_misaki_espeak.py

Same training pipeline as before, but Misaki G2P is instantiated with an optional
espeak/espeak-ng fallback when enabled in the config (misaki.fallback = "espeak").

Usage:
    python train_with_misaki_espeak.py config.yaml
"""
import warnings
import re
import os
import shutil
from functools import partial
from omegaconf import OmegaConf
from loguru import logger as LOGGER

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset

warnings.filterwarnings("ignore")

# === Misaki imports ===
try:
    # primary misaki modules
    from misaki import en as misaki_en
    from misaki import espeak as misaki_espeak  # this module provides EspeakFallback
except Exception as e:
    LOGGER.error("Failed to import misaki. Install with: pip install \"misaki[en]\"")
    raise

# --- convenience ---
ACRONYM = re.compile(r"(?:[a-zA-Z]\.){2,}")
ACRONYM_NO_PERIOD = re.compile(r"(?:[A-Z]){3,}")


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
    Return Python lists (not tensors) for hf.datasets.map compatibility.
    g2p callable -> (phonemes, tokens) = g2p(text)
    """
    speech_gen_start_tok = "<|SPEECH_GENERATION_START|>"
    ignore_index = -100
    pad_id = tokenizer.pad_token_id

    vq_codes = sample.get("codes", [])
    text = sample.get("text", "")

    try:
        phonemes, toks = g2p(text)
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
        # not found: leave labels as -100
        LOGGER.debug("speech start token not found in tokenized ids")

    attention_mask = [0 if x == pad_id else 1 for x in ids]

    return {"input_ids": ids, "labels": labels, "attention_mask": attention_mask}


def _detect_espeak_binary():
    """
    Look for either 'espeak-ng' or 'espeak' on PATH. Return binary name or None.
    """
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            return name
    return None


def main(config_fpath: str):
    print(f"Loading config from {config_fpath}")
    config = OmegaConf.load(config_fpath)

    checkpoints_dir = os.path.join(config.save_root, config.run_name)
    LOGGER.info(f"Logging to: {checkpoints_dir}")

    restore_from = config.restore_from
    print(f"Loading checkpoint from {restore_from}")

    # tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(restore_from, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(restore_from, torch_dtype="auto")

    # ensure pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            LOGGER.info("No pad token found; set pad_token = eos_token")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            LOGGER.info("Added [PAD] as pad token")

    # special tokens used by the prompt
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

    # --- Misaki G2P + optional espeak fallback ---
    misaki_cfg = getattr(config, "misaki", {}) or {}
    use_trf = bool(misaki_cfg.get("use_transformer", False))
    british = bool(misaki_cfg.get("british", False))
    fallback_opt = misaki_cfg.get("fallback", None)  # expect "espeak" or None

    fallback_obj = None
    if fallback_opt:
        if str(fallback_opt).lower() == "espeak":
            bin_name = _detect_espeak_binary()
            if bin_name is None:
                LOGGER.warning(
                    "espeak/espeak-ng binary not found on PATH. Install espeak-ng (system package) "
                    "or set misaki.fallback=null in config if you don't want fallback."
                )
            # misaki exposes an espeak helper module; create the fallback object if available
            try:
                # EspeakFallback signature: EspeakFallback(british=bool, program_name=None, ...)
                # If binary name detected, pass it as program_name to be explicit.
                if bin_name:
                    fallback_obj = misaki_espeak.EspeakFallback(british=british, program_name=bin_name)
                else:
                    # still try to create fallback (misaki will attempt default program)
                    fallback_obj = misaki_espeak.EspeakFallback(british=british)
                LOGGER.info("Created Misaki EspeakFallback (will be used for OOD tokens).")
            except Exception as e:
                LOGGER.warning(f"Failed to create Misaki EspeakFallback: {e}. Continuing without fallback.")
                fallback_obj = None
        else:
            LOGGER.warning(f"Unknown misaki.fallback='{fallback_opt}' — supported: 'espeak' or null")

    # instantiate G2P
    try:
        g2p = misaki_en.G2P(trf=use_trf, british=british, fallback=fallback_obj)
    except Exception as e:
        LOGGER.error(f"Failed to initialize misaki.en.G2P: {e}")
        raise

    partial_preprocess = partial(preprocess_sample, tokenizer=tokenizer, max_len=config.max_seq_len, g2p=g2p)

    # dataset
    emilia_dataset = load_dataset("neuphonic/emilia-yodas-english-neucodec", split="train[:2000]")
    emilia_dataset = emilia_dataset.filter(data_filter)
    emilia_dataset = emilia_dataset.map(partial_preprocess, remove_columns=["text", "codes"], batched=False)

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
        train_dataset=emilia_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(checkpoints_dir)
    LOGGER.info("Training done and model saved.")


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
