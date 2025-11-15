# Evaluation script
# project/src/evaluate.py



# project/src/evaluate.py

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from preprocess import preprocess_dataframe

# -----------------------------
# PATHS
# -----------------------------
BASE = Path("project")
MODEL_PATH = BASE / "model/model.pkl"
VEC_PATH = BASE / "model/vectorizer.pkl"
TEST_PATH = BASE / "data/test.csv"
TAX_PATH = BASE / "data/taxonomy.json"
NORM_PATH = BASE / "data/normalization.json"

OUT_JSON = BASE / "evaluation/metrics_report.json"
OUT_CM = BASE / "evaluation/confusion_matrix.png"

os.makedirs(BASE / "evaluation", exist_ok=True)


# -----------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# -----------------------------
# Load taxonomy
# -----------------------------
taxonomy = json.load(open(TAX_PATH))


# -----------------------------
# Softmax
# -----------------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_batch(cleaned_texts):
    X = vectorizer.transform(cleaned_texts)
    logits = model.decision_function(X)
    preds = np.argmax(logits, axis=1)
    labels = model.classes_[preds]
    return labels


# -----------------------------
# MAIN EVALUATION PIPELINE
# -----------------------------
def run_evaluation():
    print("Loading test dataset...")
    df = pd.read_csv(TEST_PATH)

    if "transaction" not in df.columns or "category" not in df.columns:
        raise ValueError("test.csv must contain 'transaction' and 'category' columns")

    print("Preprocessing...")
    df = preprocess_dataframe(df, text_col="transaction", norm_table_path=NORM_PATH)

    combined = (df["cleaned_text"].fillna("") + " " +
                df["merchant_normalized"].fillna(""))

    print("Running predictions...")
    preds = predict_batch(combined)
    true = df["category"]

    print("Generating metrics...")

    report = classification_report(true, preds, output_dict=True)
    cm = confusion_matrix(true, preds)
    labels = sorted(list(set(true)))

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump({
            "classification_report": report,
            "classes": labels
        }, f, indent=4)

    print(f"Saved metrics JSON → {OUT_JSON}")

    # Save Confusion Matrix Image
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUT_CM)
    plt.close()

    print(f"Saved confusion matrix → {OUT_CM}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    run_evaluation()