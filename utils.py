\"\"\"utils.py
Utility functions for dataset loading, metrics and small helpers used by the scripts.
\"\"\"
import os
import random
import logging
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(pred):
    \"\"\"Compute metrics for Hugging Face Trainer `compute_metrics` argument.
    Expects `pred` to have `predictions` and `label_ids` attributes (or keys).
    Returns a dict with `accuracy` and `f1` (weighted).
    \"\"\"
    labels = pred.label_ids if hasattr(pred, "label_ids") else pred["label_ids"]
    preds = pred.predictions.argmax(-1) if hasattr(pred, "predictions") else np.argmax(pred["predictions"], axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def plot_confusion_matrix(y_pred, y_true, labels, save_path=None):
    \"\"\"Plot and optionally save a normalized confusion matrix. Returns the matplotlib Figure.
    - y_pred, y_true: iterable of ints
    - labels: list of label names in the correct order
    \"\"\"
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=None, values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
