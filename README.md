# Text Emotion Classification with DistilBERT

## Short Description
This repository demonstrates the training and evaluation of a text emotion classifier using the Hugging Face `emotion` dataset. It explores two approaches:

1. **Feature Extraction + Classical ML Baseline**: Extract [CLS] token embeddings from a frozen DistilBERT model and train a simple classifier (Logistic Regression) on them, compared against a dummy baseline.
2. **End-to-End Fine-Tuning**: Fine-tune the `distilbert-base-uncased` model for multi-class emotion classification (6 labels: sadness, joy, love, anger, fear, surprise) using Hugging Face's `Trainer` API.

The core workflow is implemented in the Jupyter notebook `notebooks/Text_classifier_transformer_training.ipynb`, which includes data loading, preprocessing, training, evaluation, and inference. This README provides detailed instructions for reproduction, explanations of the implementation, and suggestions for extension.

## Table of Contents
- [Repository Structure](#repository-structure)
- [Project Overview & Goals](#project-overview--goals)
- [Dataset Used](#dataset-used)
- [Environment & Requirements](#environment--requirements)
- [Quickstart: Reproduce Everything](#quickstart-reproduce-everything)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Scripts in src/](#scripts-in-src)
- [Hyperparameters & Key Values](#hyperparameters--key-values)
- [Results & Expected Outputs](#results--expected-outputs)
- [Reproducibility Checklist & Tips](#reproducibility-checklist--tips)
- [Security / Housekeeping Notes](#security--housekeeping-notes)
- [Troubleshooting](#troubleshooting)
- [How to Extend](#how-to-extend)
- [License & Citation](#license--citation)

## Repository Structure
```
.
├── README.md                  # This file: project documentation and reproduction guide
├── requirements.txt           # Pinned Python dependencies for reproducibility
├── notebooks/
│   └── Text_classifier_transformer_training.ipynb  # Main notebook with full pipeline and experiments
├── src/
│   ├── train.py               # Script for fine-tuning the model via CLI
│   ├── feature_extract.py     # Script to extract and save CLS embeddings
│   ├── inference.py           # Script for model inference on new text
│   └── utils.py               # Helper functions (e.g., metrics, plotting, seeding)
├── models/                    # Directory for saved model checkpoints (gitignored to avoid large files)
├── LICENSE                    # Project license (e.g., MIT)
└── .gitignore                 # Ignores large files like models/ and .venv/
```

- **notebooks/**: Contains the interactive Jupyter notebook for end-to-end experimentation.
- **src/**: Modular scripts for production-like runs (e.g., CLI-based training without Jupyter).
- **models/**: Stores trained model artifacts (not committed to Git; use locally or push to Hugging Face Hub).

## Project Overview & Goals
The goal is to build an accurate multi-class text emotion classifier that predicts one of six emotions (sadness, joy, love, anger, fear, surprise) from input text.

**Why two approaches?**
- **Feature Extraction Baseline**: Uses frozen transformer embeddings + classical ML (e.g., Logistic Regression). This is compute-efficient, requires no GPU for training the classifier, and serves as a benchmark to show the value of fine-tuning.
- **Fine-Tuning**: Optimizes the entire DistilBERT model end-to-end, typically yielding superior performance on downstream tasks like emotion classification.

This project highlights practical NLP workflows with Hugging Face libraries, including tokenization, model training, evaluation metrics (accuracy, weighted F1), and confusion matrix visualization. It's suitable for beginners learning transformers while being extensible for advanced users.

High-level pipeline:
1. Load and explore the dataset.
2. Tokenize text.
3. (Baseline) Extract features and train/evaluate classical models.
4. (Main) Fine-tune DistilBERT and evaluate.
5. Perform inference on new examples.

## Dataset Used
- **ID**: `emotion` from Hugging Face Datasets (loaded via `datasets.load_dataset("emotion")`).
- **Description**: A dataset of English Twitter messages labeled with one of six emotions.
- **Splits & Sizes**:
  - Train: 16,000 examples
  - Validation: 2,000 examples
  - Test: 2,000 examples
- **Features**: `text` (string), `label` (int, mapped to ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']).
- **Class Distribution**: Moderate imbalance (e.g., 'joy' and 'sadness' are more common); use weighted metrics like F1.
- **License/Citation**: Public domain (see [dataset card](https://huggingface.co/datasets/dair-ai/emotion)). Cite as per the card if publishing results:  
  ```
  @inproceedings{saravia-etal-2018-carer,
      title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
      author = "Saravia, Elvis  and
        Liu, Hsien-Chi Toby  and
        Huang, Yen-Hao  and
        Wu, Junlin  and
        Chen, Yi-Shin",
      booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
      month = oct # "-" # nov,
      year = "2018",
      address = "Brussels, Belgium",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/D18-1404",
      doi = "10.18653/v1/D18-1404",
      pages = "3687--3697",
  }
  ```

## Environment & Requirements
- **Python Version**: 3.8+ (tested on 3.9–3.10).
- **Hardware**: GPU recommended for fine-tuning (e.g., NVIDIA with ≥4GB VRAM). CPU works for baseline but slows fine-tuning.
- **Dependencies**: Listed in `requirements.txt`. Key packages:
  ```
  transformers>=4.30.0
  datasets>=2.8.0
  huggingface-hub>=0.11.0
  torch>=1.13.0
  scikit-learn>=1.1.0
  pandas
  numpy
  matplotlib
  jupyter
  ```
  Install with: `pip install -r requirements.txt`.

To generate `requirements.txt` from your environment: `pip freeze > requirements.txt`.

## Quickstart: Reproduce Everything
1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set Up Virtual Environment**:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Authenticate with Hugging Face** (optional, for pushing models):
   ```
   huggingface-cli login
   ```
   Or set `export HF_TOKEN="your_token"`.

4. **Run the Notebook**:
   ```
   jupyter lab
   ```
   Open `notebooks/Text_classifier_transformer_training.ipynb` and execute cells sequentially.

5. **Headless Run (Alternative)**:
   Use Papermill for parameterized execution:
   ```
   pip install papermill
   papermill notebooks/Text_classifier_transformer_training.ipynb output.ipynb -p num_train_epochs 2
   ```

## Notebook Walkthrough
The notebook `Text_classifier_transformer_training.ipynb` is self-contained. Below is a step-by-step mapping with explanations (add these as Markdown cells in your notebook for clarity):

1. **Intro / Motivation**:
   - What: Sets up imports and introduces the project.
   - Explanation: "This notebook trains an emotion classifier on text data. We compare a feature-extraction baseline with full fine-tuning of DistilBERT for better accuracy in NLP tasks like sentiment/emotion analysis."

2. **Load Dataset**:
   - What: `datasets.load_dataset("emotion")`.
   - Explanation: "Loads the 'emotion' dataset from Hugging Face. Check splits and sample data to understand structure and class balance."

3. **Exploratory Data Analysis (EDA)**:
   - What: Samples, label distributions, text lengths.
   - Explanation: "EDA reveals class imbalance (e.g., more 'joy' samples), guiding metric choice (weighted F1). Texts are short (tweets), suitable for DistilBERT."

4. **Tokenization**:
   - What: `AutoTokenizer.from_pretrained("distilbert-base-uncased")` and `dataset.map(tokenize, batched=True)`.
   - Explanation: "Tokenizes text into input_ids and attention_mask. Padding/truncation ensures fixed-length inputs for batching."

5. **Feature Extraction (Baseline)**:
   - What: Use `AutoModel` to extract CLS embeddings.
   - Explanation: "Freezes DistilBERT, extracts [CLS] token (sentence embedding) for classical ML. Alternatives: mean pooling over tokens."

6. **Classical ML Baseline**:
   - What: Train `LogisticRegression` and `DummyClassifier`.
   - Explanation: "Baseline checks if embeddings capture emotions. Dummy uses most frequent class for comparison."

7. **Evaluation & Visualization**:
   - What: Accuracy, F1, confusion matrix.
   - Explanation: "Uses weighted F1 for imbalance. Confusion matrix shows misclassifications (e.g., love vs. joy)."

8. **Fine-Tuning with Trainer**:
   - What: `AutoModelForSequenceClassification`, `TrainingArguments`, `Trainer.train()`.
   - Explanation: "Fine-tunes all layers. Trainer handles optimization, logging, and evaluation."

9. **Inference & Hub Push**:
   - What: `pipeline("text-classification")` and `trainer.push_to_hub()`.
   - Explanation: "Demonstrates prediction on new text. Push to Hub for sharing/reuse."

10. **Wrap-Up**:
    - Explanation: "Fine-tuning outperforms baseline (~91% vs ~63% accuracy). Limitations: Dataset size, English-only."

## Scripts in src/
These scripts modularize the notebook for CLI runs:

- **feature_extract.py**: Extracts CLS embeddings.
  ```
  python src/feature_extract.py --model_ckpt distilbert-base-uncased --dataset emotion --split train --out_dir models/embeddings --batch_size 64
  ```

- **train.py**: Fine-tunes the model.
  ```
  python src/train.py --model_ckpt distilbert-base-uncased --dataset emotion --num_train_epochs 2 --batch_size 64 --learning_rate 2e-5 --output_dir models/distilbert-finetuned-emotion
  ```

- **inference.py**: Predicts on new text.
  ```
  python src/inference.py --model_dir models/distilbert-finetuned-emotion --text "I am so excited today!"
  ```

- **utils.py**: Shared functions (e.g., `compute_metrics`, plotting).

Port notebook code to these if not already implemented.

## Hyperparameters & Key Values
- Model: `distilbert-base-uncased`
- Num Labels: 6
- Batch Size: 64
- Epochs: 2
- Learning Rate: 2e-5
- Weight Decay: 0.01
- Logistic Regression: `max_iter=3000`

Tune via notebook or script args.

## Results & Expected Outputs
- **Baseline (Logistic Regression)**: ~63% accuracy (validation).
- **Dummy Baseline**: ~35% accuracy.
- **Fine-Tuned DistilBERT**: ~91% accuracy, ~91% weighted F1 (test).
- Outputs: Printed metrics, confusion matrices (saved as images), model checkpoints in `models/`.

## Reproducibility Checklist & Tips
- Pin dependencies: Use `requirements.txt`.
- Set seeds: Add `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` at the start.
- Dataset Version: Use latest from Hugging Face (record commit if needed).
- Save Artifacts: Model, metrics, logs.
- Run End-to-End: Execute notebook sequentially to avoid state issues.
- Resource Tips: Reduce batch size or use `fp16=True` for OOM errors.

