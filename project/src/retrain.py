import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess_dataframe


# ================================
# Paths
# ================================
BASE = Path("project")
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "model"
EVAL_DIR = BASE / "evaluation"

TRAIN_PATH = DATA_DIR / "train.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
NORM_PATH = DATA_DIR / "normalization.json"
CONFIG_PATH = BASE / "config.json"

MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"


# ================================
# Load Config
# ================================
if CONFIG_PATH.exists():
    config = json.load(open(CONFIG_PATH))
else:
    config = {"feedback_sample_weight": 3}

FEEDBACK_WEIGHT = config.get("feedback_sample_weight", 3)


# ================================
# Ensure dirs exist
# ================================
MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)


# ================================
# Train Function
# ================================
def retrain():

    print("\n===============================")
    print("        RETRAINING MODEL       ")
    print("===============================\n")

    # ----------------------------------
    # Load datasets
    # ----------------------------------
    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)

    if FEEDBACK_PATH.exists() and os.path.getsize(FEEDBACK_PATH) > 0:
        fb_df = pd.read_csv(FEEDBACK_PATH)
    else:
        fb_df = pd.DataFrame(columns=["transaction", "predicted", "corrected"])

    print(f"Base training rows  : {len(train_df)}")
    print(f"Feedback rows       : {len(fb_df)}")

    # ----------------------------------
    # Convert feedback to training rows
    # ----------------------------------
    if len(fb_df) > 0:
        fb_train = pd.DataFrame({
            "transaction": fb_df["transaction"],
            "category": fb_df["corrected"]
        })
    else:
        fb_train = pd.DataFrame(columns=["transaction", "category"])

    # ----------------------------------
    # Combine base + feedback
    # ----------------------------------
    combined_df = pd.concat([train_df, fb_train], ignore_index=True)
    print(f"Total training rows : {len(combined_df)}")

    # ----------------------------------
    # Preprocess
    # ----------------------------------
    print("Preprocessing text...")
    combined_df = preprocess_dataframe(
        combined_df,
        text_col="transaction",
        norm_table_path=str(NORM_PATH)
    )

    full_text = (
        combined_df["cleaned_text"].fillna("") + " " +
        combined_df["merchant_normalized"].fillna("")
    )
    labels = combined_df["category"]

    # ----------------------------------
    # Apply sample weights
    # ----------------------------------
    print(f"Applying feedback weight = {FEEDBACK_WEIGHT}")

    base_weights = np.ones(len(train_df))
    fb_weights = np.ones(len(fb_train)) * FEEDBACK_WEIGHT

    sample_weights = np.concatenate([base_weights, fb_weights])

    # ----------------------------------
    # Vectorizer
    # ----------------------------------
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(full_text)

    # ----------------------------------
    # Train model
    # ----------------------------------
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X, labels, sample_weight=sample_weights)

    # ----------------------------------
    # Save model
    # ----------------------------------
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to:      {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")

    # ----------------------------------
    # Evaluate on *TRAIN* (or optionally use validation split)
    # ----------------------------------
    print("\nEvaluating model...")

    preds = model.predict(X)
    report = classification_report(labels, preds, output_dict=True)
    conf_matrix = confusion_matrix(labels, preds)

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }

    json.dump(metrics, open(EVAL_DIR / "metrics_report.json", "w"), indent=4)
    print("Saved metrics → evaluation/metrics_report.json")

    print("\nMacro F1:", report["macro avg"]["f1-score"])
    print("Retraining complete!\n")


# ================================
# CLI
# ================================
if __name__ == "__main__":
    retrain()