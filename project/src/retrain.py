# Retraining script


# project/src/retrain.py

import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import preprocess_dataframe
from feedback import FEEDBACK_PATH

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = "project/data"
MODEL_DIR = "project/model"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
NORM_PATH = os.path.join(DATA_DIR, "normalization.json")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

TEXT_COL = "transaction"
LABEL_COL = "category"

# -------------------------------
# RETRAIN PIPELINE
# -------------------------------

def load_feedback():
    """Load user corrections from feedback.csv."""
    if not os.path.exists(FEEDBACK_PATH):
        print("No feedback file found. Using only original training data.")
        return pd.DataFrame(columns=["transaction", "corrected"])

    fb = pd.read_csv(FEEDBACK_PATH)
    fb = fb.rename(columns={"corrected": "category"})
    fb = fb[["transaction", "category"]]

    print(f"Loaded {len(fb)} feedback rows.")
    return fb


def retrain():

    print("\n=== LOADING ORIGINAL TRAIN DATA ===")
    train_df = pd.read_csv(TRAIN_PATH)

    print("=== LOADING FEEDBACK DATA ===")
    fb_df = load_feedback()

    print("=== MERGING DATASETS ===")
    full_train = pd.concat([train_df, fb_df], ignore_index=True)
    full_train = full_train.sample(frac=1).reset_index(drop=True)
    print(f"Total training rows after merging: {len(full_train)}")

    print("\n=== PREPROCESSING TRAINING DATA ===")
    full_train = preprocess_dataframe(full_train, text_col=TEXT_COL, norm_table_path=NORM_PATH)

    print("=== PREPROCESSING TEST DATA ===")
    test_df = pd.read_csv(TEST_PATH)
    test_df = preprocess_dataframe(test_df, text_col=TEXT_COL, norm_table_path=NORM_PATH)

    # Combine cleaned + normalized merchant for best features
    X_train_text = (full_train["cleaned_text"].fillna("") + " " +
                    full_train["merchant_normalized"].fillna(""))

    X_test_text = (test_df["cleaned_text"].fillna("") + " " +
                   test_df["merchant_normalized"].fillna(""))

    y_train = full_train[LABEL_COL]
    y_test = test_df[LABEL_COL]

    print("\n=== TF-IDF VECTORIZATION ===")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("=== RETRAINING MODEL ===")
    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n=== EVALUATING NEW MODEL ===")
    preds = model.predict(X_test)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, preds))

    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, preds))

    print("\n=== SAVING UPDATED MODEL ===")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Saved updated model → {MODEL_PATH}")
    print(f"Saved updated vectorizer → {VECTORIZER_PATH}")
    print("\nRetraining complete!")


if __name__ == "__main__":
    retrain()
