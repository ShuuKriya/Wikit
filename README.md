# Wikit

Here’s a **simple, clean, beginner-friendly README** you can drop directly into your repo right now.
No fluff — just the essentials for Round 2.

---

# **README**

## **AI-Based Financial Transaction Categorization System**

This project is an in-house AI system that automatically classifies financial transaction text (e.g., `"ZOMATO *ORDER #392847"`) into meaningful categories such as **Dining**, **Shopping**, **Fuel**, etc.
The system does **not rely on any third-party APIs** and is fully customizable, explainable, and self-learning.

---

## **Features**

* **End-to-end autonomous categorization**
* **Customizable taxonomy** (via `taxonomy.json`)
* **Batch mode** for large datasets
* **Interactive mode** with human-in-the-loop feedback
* **Explainable outputs** (confidence scores + top keyword indicators)
* **Self-learning pipeline** using feedback data (`feedback.csv`)
* **Retrainable model** using simple retraining scripts

---

## **Current Tech Stack**

* **Python**
* **scikit-learn** (TF-IDF + Logistic Regression)
* **pandas / numpy**
* **Streamlit** (UI)
* **JSON / CSV** for configuration & data

---

## **Model Overview**

1. Preprocessing

   * Regex cleaning
   * Tokenization
   * Merchant name normalization

2. Feature Extraction

   * TF-IDF Vectorizer
     *(Future upgrade: MiniLM / SBERT embeddings)*

3. Model

   * Logistic Regression
     *(Future upgrade: MLP for richer patterns)*

4. Confidence Evaluation

   * Softmax probability threshold
   * Low-confidence → ask user / mark as “Other”

5. Feedback Loop

   * User corrections saved to `feedback.csv`
   * Combined into training during retraining

---

## **Project Structure**

```
project/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── feedback.csv
│   ├── taxonomy.json
│   └── memory.json
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── retrain.py
│   ├── feedback.py
│   └── utils.py
│
└── ui/
    └── app.py
```

---

## **How to Run**

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Train the Model

```
python src/train.py
```

### 3. Run Batch Mode

```
python src/predict.py --batch data/test.csv
```

### 4. Run Interactive Mode

```
python src/predict.py --interactive
```

### 5. Retrain with Feedback

```
python src/retrain.py
```

### 6. Launch UI

```
streamlit run ui/app.py
```

---

## **Next Steps**

* Add explainability UI with SHAP/LIME
* Add bias and fairness testing
* Upgrade to embeddings + MLP model
* Improve robustness to noisy inputs
* Add automated retraining triggers

---


