



# Model Documentation

This document provides a detailed overview of the machine learning model, feature pipeline, training workflow, and evaluation methodology used in the WIKIT transaction categorisation system.

---

## 1. Objective

The model performs **single-label multiclass classification**, mapping raw transaction text into financial categories such as:
Bills, Dining, Entertainment, Fuel, Groceries, Shopping, and Travel.

The system is designed to be:
- fully local  
- explainable  
- customizable  
- lightweight and fast  

---

## 2. Architecture Overview

The model follows a classical but strong NLP pipeline optimized for short-text classification.

### Components

1. **Preprocessing Layer**  
   - text cleaning (lowercasing, punctuation removal)  
   - merchant extraction  
   - merchant normalisation (via lookup table)  
   - combined text feature (cleaned_text + merchant_normalized)

2. **Vectorization Layer**  
   - TF-IDF vectorizer  
   - character/word n-grams: (1, 2)  
   - max_features = 5000  
   - produces sparse vectors suitable for linear models  

3. **Classification Model**  
   - Logistic Regression (multinomial)  
   - max_iter = 2000  
   - n_jobs = -1 (parallel training)  
   - regularization provided implicitly via classical LR  
   - softmax applied to model.decision_function scores for confidence  

This pipeline provides:
- high interpretability  
- strong performance on short noisy texts  
- fast inference (sub-ms latency)  
- easy retraining with new samples  

---

## 3. Why Logistic Regression?

Logistic Regression was selected over SVMs, Naive Bayes, or deep learning because:

- performs very well on TF-IDF sparse text  
- fully explainable token contributions  
- robust on small/medium datasets  
- extremely fast to train and deploy  
- supports class probabilities via softmax on decision scores  

For transaction categorisation (short, structured text), LR is a proven industry-standard baseline.

---

## 4. Feature Engineering

### Text Features
- unigrams and bigrams extracted from fully cleaned text
- strong for picking up multi-token patterns (e.g., “petrol pump”, “flight ticket”)

### Merchant Features
From the preprocessing pipeline, many transaction strings contain merchant identifiers.  
These are normalised and appended to the text, increasing accuracy significantly.

Example:

Raw:   “AMAZON PAY *ORDER ID 8237”
Clean: “amazon pay order id”
Merchant_normalized: “amazon”
Combined final input: “amazon pay order id amazon”

This amplifies merchant identity signal for the model.

---

## 5. Training Workflow

Training is conducted using:

project/src/train.py

Steps:

1. Load `train.csv`  
2. Preprocess all rows  
3. Create combined text features  
4. Vectorize using TF-IDF  
5. Train Logistic Regression  
6. Evaluate on separate `test.csv`  
7. Save:
   - model.pkl  
   - vectorizer.pkl  

All artefacts stored under:

project/model/

---

## 6. Model Retraining with Feedback

The system supports human-in-the-loop learning.

### Feedback Flow:

1. User corrects a wrong prediction in the UI  
2. Entry appended to:

project/data/feedback.csv

3. Retraining weights feedback samples higher:

feedback_sample_weight = 3

4. `retrain.py` rebuilds the model using:
- base training data  
- feedback-derived training samples  

This enables continuous improvement without manual labeling.

---

## 7. Evaluation Summary

Evaluation performed on an **entirely separate test set (352 samples)**.

### Macro-Averaged Metrics

| Metric     | Score |
|------------|--------|
| Accuracy   | 0.93   |
| Macro F1   | 0.93   |

### Per-Class F1

Bills: 0.97  
Dining: 0.91  
Entertainment: 0.92  
Fuel: 0.91  
Groceries: 0.92  
Shopping: 0.93  
Travel: 0.95  

### Confusion Matrix (7×7)

Stored at:

project/evaluation/confusion_matrix.png

### Performance Benchmarks

From training script measurements:

- Average inference latency: **0.12 ms**  
- Throughput: **8,376 predictions/sec**  
- Batch inference (352 samples): **1.09 ms**

This satisfies the performance requirement for scalable financial categorisation systems.

---

## 8. Explainability

The system includes an explainability module:

predict.py → explain_prediction()

It provides:
- top positive coefficient tokens per class  
- perturbation-based sensitivity  
- merged feature contribution score  
- UI display with token-level bars  

Explainability is integrated into both:
- Single Prediction mode  
- Batch “Explain Row” mode  

This aligns with the requirement of transparent and responsible AI.

---

## 9. Summary

The final model is:

- fast  
- accurate  
- explainable  
- customisable  
- easy to retrain  
- fully local (no external APIs)  

The combination of TF-IDF + Logistic Regression provides near-production performance without requiring heavy compute.


⸻
