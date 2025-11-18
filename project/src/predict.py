# project/src/predict.py
import argparse
import pandas as pd
import joblib
import numpy as np
import os
import json
from preprocess import preprocess_row, load_normalization_table

# -------------------------------
# PATHS
# -------------------------------
MODEL_PATH = "project/model/model.pkl"
VECTORIZER_PATH = "project/model/vectorizer.pkl"
CONFIG_PATH = "project/config.json"
NORM_PATH = "project/data/normalization.json"

# -------------------------------
# LOAD CONFIG
# -------------------------------
def load_config():
    if not os.path.exists(CONFIG_PATH):
        # sensible defaults if config missing
        return {
            "confidence_threshold": 0.60,
            "batch_low_confidence_behavior": "other",
            "interactive_low_confidence": True
        }
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

config = load_config()
THRESHOLD = config.get("confidence_threshold", 0.60)
BATCH_BEHAVIOR = config.get("batch_low_confidence_behavior", "other")     # "other" or "keep"

# -------------------------------
# LOAD NORMALIZATION TABLE
# -------------------------------
norm_table = load_normalization_table(NORM_PATH)

# -------------------------------
# LOAD MODEL + VECTORIZER
# -------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer not found. Run train.py first.")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -------------------------------
# Softmax for LR decision_function
# -------------------------------
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

# -------------------------------
# Predict a single transaction
# -------------------------------
def predict_single(text: str, is_batch=False):
    # preprocess with normalization table
    cleaned, merchant = preprocess_row(text, norm_table)

    combined_text = (cleaned or "") + " " + (merchant or "")
    X = vectorizer.transform([combined_text])


    try:
        logits = model.decision_function(X)[0]
        confs = softmax(logits)
    except Exception:

        probs = model.predict_proba(X)[0]
        confs = probs

    pred_idx = int(np.argmax(confs))
    pred_class = model.classes_[pred_idx]
    confidence = float(confs[pred_idx])

    # --------------------------
    # Explanation tokens (top positive coefs for predicted class)
    # --------------------------
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        feature_names = vectorizer.get_feature_names()
    coefs = model.coef_[pred_idx]
    top_indices = np.argsort(coefs)[-5:][::-1]
    top_tokens = [feature_names[i] for i in top_indices]

    # --------------------------
    # Low-confidence logic
    # --------------------------
    needs_feedback = confidence < THRESHOLD


    if is_batch and needs_feedback and BATCH_BEHAVIOR == "other":
        pred_class = "Other"

    return {
        "input": text,
        "cleaned_text": cleaned,
        "merchant": merchant,
        "prediction": pred_class,
        "confidence": confidence,
        "needs_feedback": needs_feedback,
        "top_tokens": top_tokens
    }

def explain_prediction(model, vectorizer, cleaned_text, merchant, combined_text):
    """
    Returns dict: token → {coef_weight, perturb_impact, combined_score}
    """
    feature_names = vectorizer.get_feature_names_out()


    X = vectorizer.transform([combined_text])
    logits = model.decision_function(X)[0]
    base_conf = softmax(logits)[np.argmax(logits)]

    explanation = {}

    tokens = cleaned_text.split()
    tokens = [t for t in tokens if t.strip()]

    for tok in tokens:
        if tok in feature_names:
            idx = np.where(feature_names == tok)[0][0]
            coef_weight = float(model.coef_[np.argmax(logits)][idx])
        else:
            coef_weight = 0.0


        perturbed = " ".join([t for t in tokens if t != tok])
        X_pert = vectorizer.transform([perturbed])
        logits_pert = model.decision_function(X_pert)[0]
        pert_conf = softmax(logits_pert)[np.argmax(logits_pert)]

        impact = float(pert_conf - base_conf)

        explanation[tok] = {
            "coef_weight": round(coef_weight, 4),
            "perturbation_impact": round(impact, 4),
            "combined_score": round(coef_weight + impact, 4)
        }

    return explanation
# -------------------------------
# Batch CSV prediction
# -------------------------------
def predict_batch(csv_path):
    df = pd.read_csv(csv_path)

    if "transaction" not in df.columns:
        raise ValueError("CSV must contain a 'transaction' column.")

    results = []
    for text in df["transaction"].astype(str):
        out = predict_single(text, is_batch=True)
        results.append(out)

    return pd.DataFrame(results)

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Single transaction")
    parser.add_argument("--batch", type=str, help="Path to CSV for batch mode")
    args = parser.parse_args()

    if args.text:
        out = predict_single(args.text)
        print("\n===== Prediction =====")
        for k, v in out.items():
            print(f"{k}: {v}")

    elif args.batch:
        print("\nRunning batch predictions...\n")
        df = predict_batch(args.batch)
        save_path = "project/data/batch_output.csv"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(df)
        print(f"\nSaved output to: {save_path}\n")

    else:
        print("Provide either --text or --batch")