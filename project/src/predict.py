# Prediction script


# project/src/predict.py

import argparse
import pandas as pd
import joblib
import numpy as np
import os
from preprocess import preprocess_row
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "/Users/shu/test/Wikit/Wikit/project/model/model.pkl"
VECTORIZER_PATH = "/Users/shu/test/Wikit/Wikit/project/model/vectorizer.pkl"
NORM_PATH = "/Users/shu/test/Wikit/Wikit/project/data/normalization.json"

# -------------------------------
# LOAD MODEL + VECTORIZER
# -------------------------------
model: LogisticRegression = joblib.load(MODEL_PATH)
vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_PATH)

# -------------------------------
# Softmax for confidence scores
# -------------------------------
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

# -------------------------------
# Predict a single transaction text
# -------------------------------
def predict_single(text: str):
    from preprocess import load_normalization_table
    norm_table = load_normalization_table(NORM_PATH)

    cleaned, norm_merchant = preprocess_row(text, norm_table)

    combined_text = (cleaned or "") + " " + (norm_merchant or "")
    X = vectorizer.transform([combined_text])

    logits = model.decision_function(X)[0]
    confs = softmax(logits)
    pred_idx = np.argmax(confs)
    pred_class = model.classes_[pred_idx]
    confidence = confs[pred_idx]

    # --------------------------
    # Explanation (top important words)
    # --------------------------
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[pred_idx]
    top_indices = np.argsort(coefs)[-5:][::-1]
    top_tokens = [feature_names[i] for i in top_indices]

    LOW_CONF_THRESHOLD = 0.60
    needs_feedback = confidence < LOW_CONF_THRESHOLD


    return {
        "input": text,
        "cleaned_text": cleaned,
        "merchant": norm_merchant,
        "prediction": pred_class,
        "confidence": float(confidence),
        "needs_feedback": needs_feedback,
        "top_tokens": top_tokens
    }


# -------------------------------
# Batch CSV prediction
# -------------------------------
def predict_batch(csv_path):
    df = pd.read_csv(csv_path)
    results = []

    for text in df["transaction"]:
        out = predict_single(text)
        results.append(out)

    return pd.DataFrame(results)


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Single transaction text")
    parser.add_argument("--batch", type=str, help="Path to CSV file for batch mode")
    args = parser.parse_args()

    if args.text:
        prediction = predict_single(args.text)
        print("\n===== Prediction =====")
        for k, v in prediction.items():
            print(f"{k}: {v}")

    elif args.batch:
        print("Running batch predictions...")
        df = predict_batch(args.batch)
        print(df)
        df.to_csv("/Users/shu/test/Wikit/Wikit/project/data/batch_output.csv", index=False)
        print("\nSaved output to: project/data/batch_output.csv")

    else:
        print("Please provide --text or --batch")
