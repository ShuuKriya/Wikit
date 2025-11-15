# project/src/performance.py

import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess import preprocess_dataframe

# -----------------------------
# PATHS
# -----------------------------
BASE = Path("project")
MODEL_PATH = BASE / "model/model.pkl"
VEC_PATH = BASE / "model/vectorizer.pkl"
TEST_PATH = BASE / "data/test.csv"
NORM_PATH = BASE / "data/normalization.json"

OUT_JSON = BASE / "evaluation/performance_report.json"
BASE.mkdir(exist_ok=True)
(BASE / "evaluation").mkdir(exist_ok=True)

# -----------------------------
# LOAD MODEL + VECTOR
# -----------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

def benchmark():
    print("Loading test dataset...")
    df = pd.read_csv(TEST_PATH)

    if "transaction" not in df.columns:
        raise ValueError("test.csv must contain 'transaction' column")

    print("Preprocessing...")
    df = preprocess_dataframe(df, text_col="transaction", norm_table_path=NORM_PATH)
    combined = (df["cleaned_text"].fillna("") + " " +
                df["merchant_normalized"].fillna("")).tolist()

    # -------------------------
    # SINGLE-PREDICTION LATENCY
    # -------------------------
    print("Measuring latency...")
    sample = combined[0]

    # Warm-up (avoid cold-start skew)
    for _ in range(5):
        model.decision_function(vectorizer.transform([sample]))

    t0 = time.time()
    runs = 200
    for _ in range(runs):
        X = vectorizer.transform([sample])
        model.decision_function(X)
    t1 = time.time()

    avg_latency_ms = ((t1 - t0) / runs) * 1000

    # -------------------------
    # THROUGHPUT (pred/s)
    # -------------------------
    print("Measuring throughput...")
    t0 = time.time()
    for text in combined:
        X = vectorizer.transform([text])
        model.decision_function(X)
    t1 = time.time()

    total = len(combined)
    throughput = total / (t1 - t0)

    # -------------------------
    # BATCH INFERENCE SPEED
    # -------------------------
    print("Measuring batch inference...")
    t0 = time.time()
    X = vectorizer.transform(combined)
    model.decision_function(X)
    t1 = time.time()

    batch_time_ms = (t1 - t0) * 1000

    # -------------------------
    # SAVE RESULTS
    # -------------------------
    results = {
        "avg_latency_ms": avg_latency_ms,
        "throughput_predictions_per_sec": throughput,
        "batch_inference_time_ms": batch_time_ms,
        "num_samples": total
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved →", OUT_JSON)
    print("\n=== PERFORMANCE REPORT ===")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    benchmark()