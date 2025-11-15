# project/src/explain.py

import json
import numpy as np
from predict import predict_single
from preprocess import preprocess_row
from pathlib import Path
import joblib

BASE = Path("project")
MODEL_PATH = BASE / "model/model.pkl"
VEC_PATH = BASE / "model/vectorizer.pkl"
NORM_PATH = BASE / "data/normalization.json"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# -----------------------------
# Softmax
# -----------------------------
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

# -----------------------------
# Perturbation-based scoring
# -----------------------------
def perturb(token_list, merchant):
    """
    Remove each token one by one and measure effect on confidence.
    """
    scores = {}
    base_text = " ".join(token_list) + " " + (merchant or "")
    base_pred = predict_single(base_text)

    base_conf = base_pred["confidence"]
    base_label = base_pred["prediction"]

    for tok in token_list:
        modified = " ".join([t for t in token_list if t != tok])
        modified_text = modified + " " + (merchant or "")

        new_pred = predict_single(modified_text)

        if new_pred["prediction"] != base_label:
            drop = base_conf  # big effect
        else:
            drop = base_conf - new_pred["confidence"]

        scores[tok] = round(drop, 4)

    return scores


# -----------------------------
# Coefficient-based scoring
# -----------------------------
def coef_importance(cleaned_text):
    """
    Match tokens against TF-IDF vocabulary, return LR weight contribution.
    """
    tokens = cleaned_text.split()
    vocab = vectorizer.vocabulary_
    classes = model.classes_

    # get model probabilities
    X = vectorizer.transform([cleaned_text])
    logits = model.decision_function(X)[0]
    confs = softmax(logits)
    pred_idx = np.argmax(confs)

    weights = model.coef_[pred_idx]

    imp = {}
    for t in tokens:
        if t in vocab:
            imp[t] = round(float(weights[vocab[t]]), 4)
        else:
            imp[t] = 0.0

    return imp


# -----------------------------
# Main explanation function
# -----------------------------
def explain(text):
    from preprocess import load_normalization_table
    norm_table = load_normalization_table(str(NORM_PATH))

    cleaned, merchant = preprocess_row(text, norm_table)
    token_list = cleaned.split()

    coef_scores = coef_importance(cleaned)
    pert_scores = perturb(token_list, merchant)

    # merge importance
    final_importance = {}
    for t in token_list:
        final_importance[t] = {
            "coef_weight": coef_scores.get(t, 0.0),
            "perturbation_impact": pert_scores.get(t, 0.0),
            "combined_score": round(
                coef_scores.get(t, 0.0) + pert_scores.get(t, 0.0), 4
            )
        }

    pred = predict_single(text)

    return {
        "input": text,
        "cleaned": cleaned,
        "merchant": merchant,
        "prediction": pred["prediction"],
        "confidence": pred["confidence"],
        "token_importance": final_importance
    }


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    out = explain(args.text)
    print(json.dumps(out, indent=4))