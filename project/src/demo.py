# project/src/demo.py

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import joblib
from preprocess import preprocess_row, load_normalization_table
from predict import softmax
from taxonomy import load_taxonomy

# -----------------------------
# Paths
# -----------------------------
BASE = Path("project")
MODEL_PATH = BASE / "model/model.pkl"
VEC_PATH = BASE / "model/vectorizer.pkl"
MEMORY_PATH = BASE / "data/memory.json"
CONFIG_PATH = BASE / "config.json"
TEST_PATH = BASE / "data/test.csv"
NORM_PATH = BASE / "data/normalization.json"
METRICS_PATH = BASE / "evaluation/metrics_report.json"
ROBUSTNESS_PATH = BASE / "evaluation/robustness_report.json"
PERF_PATH = BASE / "evaluation/performance_report.json"

# -----------------------------
# Load everything
# -----------------------------
print("\n==============================")
print("         WIKIT DEMO")
print("==============================")

print("\nLoading model and dependencies...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
memory = json.load(open(MEMORY_PATH)) if MEMORY_PATH.exists() else {}
config = json.load(open(CONFIG_PATH))
taxonomy = load_taxonomy()
norm_table = load_normalization_table(NORM_PATH)

print("Loaded ✓\n")


# -----------------------------
# Predict function (same as UI)
# -----------------------------
def predict_demo(text):
    cleaned, merchant = preprocess_row(text, norm_table)
    combined = f"{cleaned} {merchant}"

    # memory override
    if merchant and merchant.lower() in memory:
        return {
            "prediction": memory[merchant.lower()],
            "confidence": 1.0,
            "top_tokens": [],
            "cleaned": cleaned,
            "merchant": merchant,
            "needs_feedback": False
        }

    X = vectorizer.transform([combined])
    logits = model.decision_function(X)[0]
    conf = softmax(logits)
    idx = np.argmax(conf)

    pred = model.classes_[idx]
    confidence = float(conf[idx])

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[idx]
    top = [feature_names[i] for i in np.argsort(coefs)[-5:][::-1]]

    needs_fb = confidence < config["confidence_threshold"]

    return {
        "prediction": pred,
        "confidence": confidence,
        "top_tokens": top,
        "cleaned": cleaned,
        "merchant": merchant,
        "needs_feedback": needs_fb
    }


# -----------------------------
# SECTION 1 — Single Prediction
# -----------------------------
print("▶ SINGLE PREDICTION DEMO")
sample_text = "Zomato order 299"
out = predict_demo(sample_text)

print(f"Input: {sample_text}")
print(f"Prediction: {out['prediction']}")
print(f"Confidence: {out['confidence']:.4f}")
print(f"Merchant: {out['merchant']}")
print(f"Cleaned: {out['cleaned']}")
print(f"Top tokens: {out['top_tokens']}")
print(f"Needs feedback: {out['needs_feedback']}\n")


# -----------------------------
# SECTION 2 — Batch Prediction
# -----------------------------
print("▶ BATCH PREDICTION DEMO")

if TEST_PATH.exists():
    df_test = pd.read_csv(TEST_PATH)
    print(f"Loaded test.csv ({len(df_test)} rows)")

    start = time.time()
    preds = df_test["transaction"].apply(predict_demo)
    batch_time = (time.time() - start)

    print(f"Batch prediction time: {batch_time:.3f}s")
    print("Sample output:")
    print(preds.iloc[0])
else:
    print("No test.csv found (skipping)…")


# -----------------------------
# SECTION 3 — Taxonomy
# -----------------------------
print("\n▶ TAXONOMY")
print("Available categories:", taxonomy["categories"], "\n")


# -----------------------------
# SECTION 4 — Config
# -----------------------------
print("▶ CONFIG")
print(json.dumps(config, indent=4), "\n")


# -----------------------------
# SECTION 5 — Metrics Report
# -----------------------------
if METRICS_PATH.exists():
    print("▶ METRICS REPORT")
    metrics = json.load(open(METRICS_PATH))

    # Support both formats: {macro avg: ...} and {metrics: {macro avg: ...}}
    if "macro avg" in metrics:
        macro_f1 = metrics["macro avg"]["f1-score"]
    elif "metrics" in metrics and "macro avg" in metrics["metrics"]:
        macro_f1 = metrics["metrics"]["macro avg"]["f1-score"]
    else:
        macro_f1 = "N/A"

    print("Macro F1:", macro_f1)

else:
    print("▶ METRICS REPORT MISSING")

print()


# -----------------------------
# SECTION 6 — Performance Stats
# -----------------------------
if PERF_PATH.exists():
    print("▶ PERFORMANCE REPORT")
    perf = json.load(open(PERF_PATH))
    print(json.dumps(perf, indent=4))
else:
    print("▶ PERFORMANCE REPORT MISSING")

print()


# -----------------------------
# SECTION 7 — Robustness Results
# -----------------------------
if ROBUSTNESS_PATH.exists():
    print("▶ ROBUSTNESS REPORT")
    robust = json.load(open(ROBUSTNESS_PATH))
    print(json.dumps(robust, indent=4))
else:
    print("▶ ROBUSTNESS REPORT MISSING")

print()

print("🎉 DEMO COMPLETE")