# Evaluation script
# project/src/evaluate.py
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_dataframe
import joblib

DATA_DIR = "project/data"
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
NORM_PATH = os.path.join(DATA_DIR, "normalization.json")
MODEL_DIR = "project/model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

def evaluate():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model/vectorizer missing. Train first.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Loading test set...")
    test_df = pd.read_csv(TEST_PATH)

    print("Preprocessing test set...")
    test_df = preprocess_dataframe(test_df, text_col="transaction", norm_table_path=NORM_PATH)

    X_test_text = (test_df["cleaned_text"].fillna("") + " " + test_df["merchant_normalized"].fillna(""))
    y_true = test_df["category"]

    X_test = vectorizer.transform(X_test_text)
    preds = model.predict(X_test)

    print("\n=== CLASSIFICATION REPORT ===\n")
    print(classification_report(y_true, preds))

    print("\n=== CONFUSION MATRIX ===\n")
    print(confusion_matrix(y_true, preds))

    # Basic bias/robustness hooks (counts by category)
    counts = test_df["category"].value_counts()
    print("\nCategory distribution in test set:")
    print(counts)

if __name__ == "__main__":
    evaluate()