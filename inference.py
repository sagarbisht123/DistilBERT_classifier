#!/usr/bin/env python3
\"\"\"inference.py
Run inference using either:
- finetune mode: use a saved Hugging Face SequenceClassification model (or Hub id)
- feature mode: use a frozen encoder to extract CLS embedding and apply a saved sklearn classifier
\"\"\"
import os
import argparse
import logging
from pathlib import Path
import pickle

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, pipeline
from utils import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for emotion classifier (finetune or feature)")
    parser.add_argument("--mode", type=str, choices=["finetune", "feature"], default="finetune", help="Which model type to use for inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory (for finetune) or encoder checkpoint (for feature)")
    parser.add_argument("--clf_path", type=str, default=None, help="Path to sklearn classifier pickle (required for feature mode)")
    parser.add_argument("--texts", type=str, nargs="+", required=True, help="One or more texts to classify")
    parser.add_argument("--device", type=str, default=None, help="torch device (cuda/cpu)")
    return parser.parse_args()


def infer_finetune(model_path, texts, device=None):
    logger.info("Loading finetuned model from %s", model_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = pipeline("text-classification", model=model_path, device=0 if device.startswith("cuda") else -1)
    results = classifier(texts, truncation=True)
    return results


def infer_feature(encoder_path, clf_pickle, texts, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    logger.info("Loading encoder: %s", encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    encoder = AutoModel.from_pretrained(encoder_path).to(device)
    logger.info("Loading classifier pickle: %s", clf_pickle)
    with open(clf_pickle, "rb") as f:
        clf = pickle.load(f)

    # Tokenize and extract CLS embeddings
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = encoder(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()

    preds = clf.predict(cls)
    # Try to get label names if available from tokenizer/encoder repo metadata
    label_names = None
    try:
        ds = None
    except Exception:
        label_names = None

    results = []
    for t, p in zip(texts, preds):
        label = p if label_names is None else label_names[p]
        results.append({"text": t, "label_id": int(p), "label": str(label)})
    return results


def main():
    args = parse_args()
    set_seed(42)
    if args.mode == "finetune":
        results = infer_finetune(args.model_path, args.texts, device=args.device)
        for r in results:
            print(r)
    else:
        if args.clf_path is None:
            raise ValueError("feature mode requires --clf_path pointing to the sklearn pickle file")
        results = infer_feature(args.model_path, args.clf_path, args.texts, device=args.device)
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
