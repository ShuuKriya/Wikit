# project/src/robustness.py

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from predict import predict_single
from preprocess import preprocess_row
from rapidfuzz import fuzz

BASE = Path("project")
TEST_PATH = BASE / "data/test.csv"
OUT_JSON = BASE / "evaluation/robustness_report.json"
(BASE / "evaluation").mkdir(exist_ok=True)

# ----------------------------
# Noise injection functions
# ----------------------------

def add_typo(text):
    if len(text) < 4:
        return text
    idx = random.randint(0, len(text) - 2)
    return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]

def random_case(text):
    return ''.join(
        c.upper() if random.random() < 0.5 else c.lower()
        for c in text
    )

def add_noise(text):
    noises = [
        lambda t: t + "!!!",
        lambda t: "$$ " + t,
        lambda t: t.replace(" ", "   "),
        lambda t: add_typo(t),
        lambda t: random_case(t)
    ]
    fn = random.choice(noises)
    return fn(text)

# ----------------------------
# Merchant-bias test
# ----------------------------

def test_merchant_bias():
    merchants = ["zomato", "amazon", "flipkart", "hpcl", "bigbasket"]

    results = []
    for m in merchants:
        base = m + " order 123"
        out = predict_single(base)
        results.append({
            "merchant": m,
            "prediction": out["prediction"],
            "confidence": out["confidence"]
        })
    return results

# ----------------------------
# Region / city bias
# ----------------------------

def test_region_bias(text):
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyd"]
    results = []

    for city in cities:
        noisy = f"{text} {city} branch"
        out = predict_single(noisy)
        results.append({
            "city": city,
            "prediction": out["prediction"],
            "confidence": out["confidence"]
        })
    return results

# ----------------------------
# Full robustness evaluation
# ----------------------------

def run_robustness():
    df = pd.read_csv(TEST_PATH)
    if "transaction" not in df.columns:
        raise ValueError("test.csv must contain `transaction` column")

    samples = df["transaction"].sample(min(40, len(df)), random_state=42)

    full_report = []

    mismatches = 0

    for text in samples:
        clean_pred = predict_single(text)

        noisy_text = add_noise(text)
        noisy_pred = predict_single(noisy_text)

        same_class = clean_pred["prediction"] == noisy_pred["prediction"]
        conf_drop = clean_pred["confidence"] - noisy_pred["confidence"]

        if not same_class:
            mismatches += 1

        full_report.append({
            "original": text,
            "noisy": noisy_text,
            "clean_prediction": clean_pred["prediction"],
            "noisy_prediction": noisy_pred["prediction"],
            "clean_conf": clean_pred["confidence"],
            "noisy_conf": noisy_pred["confidence"],
            "confidence_drop": conf_drop,
            "same_class": same_class
        })

    robustness_score = 1 - (mismatches / len(samples))

    # merchant + region bias
    merchant_bias = test_merchant_bias()
    region_bias = test_region_bias("Starbucks Coffee")

    final = {
        "num_samples": len(samples),
        "robustness_score": robustness_score,
        "mismatch_rate": mismatches / len(samples),
        "samples": full_report,
        "merchant_bias": merchant_bias,
        "region_bias": region_bias
    }

    with open(OUT_JSON, "w") as f:
        json.dump(final, f, indent=4)

    print("\nSaved robustness report →", OUT_JSON)
    print("\nRobustness Score:", robustness_score)


if __name__ == "__main__":
    run_robustness()