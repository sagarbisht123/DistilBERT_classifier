#!/usr/bin/env python3
\"\"\"feature_extract.py
Extract CLS token embeddings from a pretrained encoder (AutoModel) and
train a LogisticRegression classifier on top of frozen embeddings.
Saves embeddings and the trained classifier (pickle) to the output directory.
\"\"\"
import os
import argparse
import logging
from pathlib import Path
import pickle

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils import set_seed, plot_confusion_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Feature-extraction pipeline (CLS embeddings)")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained encoder checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/feature_extraction", help="Directory to save embeddings and classifier")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_cls_embeddings(dataset_split, model, tokenizer, device, batch_size=32):
    \"\"\"Return numpy array of CLS embeddings for the given dataset split (datasets.Dataset).\"\"\"
    model.eval()
    cls_embeddings = []

    # We'll iterate in batches using the dataset's built-in iterator to avoid memory blowup
    for i in range(0, len(dataset_split), batch_size):
        batch = dataset_split[i:i+batch_size]
        texts = batch["text"]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            last_hidden = out.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            cls = last_hidden[:, 0, :].cpu().numpy()
            cls_embeddings.append(cls)
    cls_embeddings = np.vstack(cls_embeddings)
    return cls_embeddings


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info("Loading dataset `emotion`")
    dataset = load_dataset("emotion")

    logger.info("Loading encoder and tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name).to(device)

    logger.info("Extracting CLS embeddings (train/validation/test)")
    X_train = extract_cls_embeddings(dataset["train"], encoder, tokenizer, device, batch_size=args.batch_size)
    X_valid = extract_cls_embeddings(dataset["validation"], encoder, tokenizer, device, batch_size=args.batch_size)
    X_test  = extract_cls_embeddings(dataset["test"], encoder, tokenizer, device, batch_size=args.batch_size)

    y_train = np.array(dataset["train"]["label"])
    y_valid = np.array(dataset["validation"]["label"])
    y_test  = np.array(dataset["test"]["label"])

    logger.info("Shapes: X_train=%s, X_valid=%s, X_test=%s", X_train.shape, X_valid.shape, X_test.shape)

    # Train logistic regression
    logger.info("Training LogisticRegression on extracted embeddings")
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    preds_valid = clf.predict(X_valid)
    acc = accuracy_score(y_valid, preds_valid)
    f1 = f1_score(y_valid, preds_valid, average="weighted")
    logger.info("Validation accuracy: %.4f  weighted F1: %.4f", acc, f1)

    # Save classifier and embeddings
    clf_path = outdir / "logistic_clf.pkl"
    emb_path = outdir / "embeddings.npz"
    logger.info("Saving classifier to %s and embeddings to %s", clf_path, emb_path)
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
    np.savez_compressed(emb_path, X_train=X_train, X_valid=X_valid, X_test=X_test,
                        y_train=y_train, y_valid=y_valid, y_test=y_test)

    # Confusion matrix on validation set
    try:
        labels = dataset["train"].features["label"].names
    except Exception:
        labels = [str(i) for i in range(len(np.unique(y_train)))]
    plot_confusion_matrix(preds_valid, y_valid, labels, save_path=outdir / "confusion_valid.png")

    logger.info("Feature-extraction finished.")


if __name__ == "__main__":
    main()
