

⸻


# Training Pipeline Documentation

This document describes the complete end-to-end training workflow used to develop the WIKIT transaction categorisation model. It covers dataset preparation, preprocessing, feature engineering, model training, evaluation, and artefact generation.

---

## 1. Overview

The training pipeline is implemented inside:

project/src/train.py

It follows these stages:

1. Load training and testing datasets  
2. Preprocess transaction text  
3. Generate combined feature text  
4. Compute TF-IDF representations  
5. Train Logistic Regression classifier  
6. Evaluate on test set  
7. Save trained artefacts  
8. Output metrics and sample predictions  

This pipeline is entirely local and reproducible.

---

## 2. Dataset Setup

### Files

project/data/train.csv
project/data/test.csv
project/data/normalization.json

### Format  
Both `train.csv` and `test.csv` contain:

| Column       | Description                                  |
|--------------|----------------------------------------------|
| transaction  | raw input string                             |
| category     | true class label (7-category taxonomy)       |

### Distribution (Provided Results)

#### Train Distribution (650 samples)
Bills: 88  
Dining: 96  
Entertainment: 104  
Fuel: 81  
Groceries: 98  
Shopping: 91  
Travel: 93  

#### Test Distribution (352 samples)
Bills: 46  
Dining: 52  
Entertainment: 60  
Fuel: 41  
Groceries: 52  
Shopping: 55  
Travel: 46  

The balanced nature across 7 classes ensures stable macro-F1 performance.

---

## 3. Preprocessing Pipeline

Preprocessing is performed using:

project/src/preprocess.py

### Steps Applied

1. Lowercasing  
2. Removal of punctuation, redundant symbols  
3. Normalisation of merchant substrings (using lookup table)  
4. Extraction of likely merchant names  
5. Construction of:
   - cleaned_text  
   - merchant_normalized  
6. Final combined input feature:

cleaned_text + “ “ + merchant_normalized

Example:

Input:      “*AMAZON PAY ORDER 8392”
Cleaned:    “amazon pay order”
Merchant:   “amazon”
Combined:   “amazon pay order amazon”

---

## 4. TF-IDF Feature Extraction

Vectorization using:

TfidfVectorizer(
max_features=5000,
ngram_range=(1, 2)
)

### Why TF-IDF  
- Performs extremely well for short structured text  
- Allows interpretability via token coefficients  
- Efficient storage as sparse vectors  
- Works seamlessly with linear models  

---

## 5. Model Training

The classifier used:

LogisticRegression(
max_iter=2000,
n_jobs=-1
)

### Reasons for Selection
- Strong performance on sparse TF-IDF  
- Fast convergence  
- Provides decision_function scores for softmax confidence  
- Fully explainable token contributions  
- Suitable for real-time applications  

Training command (automatically run inside the pipeline):

model.fit(X_train, train_labels)

---

## 6. Evaluation

After training, the pipeline evaluates on the test set using:

- Accuracy  
- Precision, Recall, F1 (per class)  
- Macro-averaged F1  
- Confusion Matrix  

### Final Metrics (From Latest Run)

Accuracy: **0.93**  
Macro F1: **0.93**

#### Per-Class F1 Scores
Bills: 0.97  
Dining: 0.91  
Entertainment: 0.92  
Fuel: 0.91  
Groceries: 0.92  
Shopping: 0.93  
Travel: 0.95  

### Confusion Matrix

Saved automatically to:

project/evaluation/confusion_matrix.png

### Performance Benchmarks

The training script measures:

- Average latency: **0.12 ms**
- Throughput: **8,376 predictions/sec**
- Batch inference time (352 samples): **1.09 ms**

These metrics meet production-grade performance expectations.

---

## 7. Saving Artefacts

After successful training, the following files are generated:

project/model/model.pkl
project/model/vectorizer.pkl

These artefacts are used by:
- the Streamlit UI
- batch pipelines
- evaluation scripts
- retraining pipeline

---

## 8. Sample Predictions (Auto-Generated)

After training, the script prints example predictions:

Text: *ORDER # FRESHTOHOME *ONLINE
Pred : Groceries

Text: #88324 NYKAA PAYMENT PAYMENT
Pred : Shopping

Text: Ticketmaster Concert Ticket Purchase
Pred : Entertainment

This confirms the model’s ability to generalize to new unseen data.

---

## 9. Retraining and Continuous Learning

Retraining is handled separately by:

project/src/retrain.py

Features:
- consumes new feedback from `feedback.csv`
- applies weighted sampling to give more importance to corrections
- rebuilds the model + vectorizer
- fully automated

---

## 10. Reproducibility

To rerun training from scratch:

python3 project/src/train.py

Requirements:
- Python 3.10+
- scikit-learn
- pandas
- joblib

---

## 11. Summary

The training pipeline is:

- stable  
- reproducible  
- explainable  
- performance-optimized  
- aligned with the problem statement requirements  
- capable of continuous improvement through retraining  




⸻

