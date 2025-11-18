# project/src/train.py
import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import preprocess_dataframe

# ----------------------------------
# CONFIG (relative to project root)
# ----------------------------------
DATA_DIR = "project/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
NORM_PATH = os.path.join(DATA_DIR, "normalization.json")

MODEL_DIR = "project/model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

TEXT_COL = "transaction"
LABEL_COL = "category"

# ----------------------------------
# TRAINING SCRIPT
# ----------------------------------
def train_model():
    print("Loading dataset...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Preprocessing training data...")
    train_df = preprocess_dataframe(train_df, text_col=TEXT_COL, norm_table_path=NORM_PATH)

    print("Preprocessing test data...")
    test_df = preprocess_dataframe(test_df, text_col=TEXT_COL, norm_table_path=NORM_PATH)

    # cleaned_text + merchant_normalized as features
    train_text = (train_df["cleaned_text"].fillna("") + " " +
                  train_df["merchant_normalized"].fillna(""))
    test_text = (test_df["cleaned_text"].fillna("") + " " +
                 test_df["merchant_normalized"].fillna(""))

    train_labels = train_df[LABEL_COL]
    test_labels = test_df[LABEL_COL]

    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X_train, train_labels)

    print("Evaluating on test set...")
    preds = model.predict(X_test)
    report = classification_report(test_labels, preds)
    print("\n=== CLASSIFICATION REPORT ===\n")
    print(report)

    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, preds))

    print("Saving model + vectorizer...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")

    # some example predictions
    print("\nSample predictions:")
    for i in range(min(5, len(test_df))):
        print(f"Text: {test_df[TEXT_COL].iloc[i]}")
        print(f"Pred : {preds[i]}")
        print("-----")

def load_model_vectorizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model or vectorizer not found. Please train first.")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

if __name__ == "__main__":
    train_model()