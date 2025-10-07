#!/usr/bin/env python3
\"\"\"train.py
Fine-tune a Hugging Face Transformer (DistilBERT by default) on the `emotion` dataset
using the Trainer API. Saves the trained model to the output directory.
\"\"\"
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from utils import set_seed, compute_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer for emotion classification")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="models/distilbert-emotion", help="Directory to save model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true", help="Push the final model to the Hugging Face Hub (requires HUGGINGFACE_TOKEN)")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub model id to push to (e.g., username/repo-name)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("Loading dataset `emotion` from Hugging Face datasets")
    dataset = load_dataset("emotion")

    logger.info("Loading tokenizer and model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=6)

    # Tokenize function
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    logger.info("Tokenizing dataset (this may take a while)")
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=max(1, len(tokenized["train"]) // args.batch_size // 10),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        log_level="info",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        logger.info("Pushing model to the Hub (model id=%s)", args.hub_model_id)
        trainer.push_to_hub()

    logger.info("Done.")


if __name__ == "__main__":
    main()
