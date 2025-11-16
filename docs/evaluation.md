
⸻


# Evaluation Report

This document describes the evaluation methodology, dataset splits, performance metrics, and analysis of the WIKIT transaction categorisation model.

---

## 1. Evaluation Overview

The model is evaluated on a held-out test dataset using standard classification metrics.  
All evaluation is performed locally using the script:

python3 project/src/evaluate.py

Outputs generated:
- `evaluation/metrics_report.json`
- `evaluation/confusion_matrix.png`

---

## 2. Dataset Split

Total dataset size: 1000 samples  
Train/Test split: 650 / 350 (stratified manually)

### Training Distribution (650 samples)

Entertainment    104
Groceries         98
Dining            96
Travel            93
Shopping          91
Bills             88
Fuel              81

### Test Distribution (352 samples)

Entertainment    60
Shopping         55
Groceries        52
Dining           52
Travel           46
Bills            46
Fuel             41

Both splits are well-balanced across categories.

---

## 3. Model

- Logistic Regression (scikit-learn)
- TF-IDF vectorizer (unigram + bigram, max_features = 5000)
- Input features: `cleaned_text + merchant_normalized`

The model was chosen for:
- interpretability
- stable performance on sparse text features
- robustness to noisy merchant strings

---

## 4. Evaluation Metrics

Metrics used:
- Accuracy
- Precision
- Recall
- F1-score (macro, weighted, per-class)
- Confusion matrix

All metrics are computed using scikit-learn.

---

## 5. Results

### Overall Metrics

Accuracy: 0.93
Macro F1-score: 0.93
Weighted F1-score: 0.93

### Per-Class Performance

| Category       | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| Bills          | 0.94      | 1.00   | 0.97     | 46      |
| Dining         | 0.92      | 0.90   | 0.91     | 52      |
| Entertainment  | 1.00      | 0.85   | 0.92     | 60      |
| Fuel           | 1.00      | 0.83   | 0.91     | 41      |
| Groceries      | 0.86      | 0.98   | 0.92     | 52      |
| Shopping       | 0.89      | 0.98   | 0.93     | 55      |
| Travel         | 0.94      | 0.96   | 0.95     | 46      |

Macro Average:
- Precision: 0.94  
- Recall: 0.93  
- F1-score: 0.93  

These scores exceed the required threshold (macro F1 >= 0.90).

---

## 6. Confusion Matrix

A confusion matrix image is saved at:

project/evaluation/confusion_matrix.png

Main observations:
- Slight confusion between Entertainment ↔ Shopping  
- Minor confusion between Dining ↔ Groceries  
These are expected overlaps due to semantically similar transaction descriptions.

---

## 7. Latency and Throughput

Measured using 352 test samples.

Average inference latency: 0.12 ms per sample
Throughput: ~8300 predictions per second
Batch inference (352 samples): ~1.09 ms

The system supports high-performance, low-latency local inference without external API calls.

---

## 8. Conclusion

The model meets and exceeds the evaluation requirement of achieving a macro F1-score of at least 0.90.

It is:
- accurate
- stable across all categories
- robust to real-world noisy transaction text
- fast enough for real-time applications

This evaluation is fully reproducible using the included dataset, training script, and evaluation script.


⸻
