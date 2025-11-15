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

## **CLI Setup Commands**




 Command Reference

This section lists all CLI commands you can run in the current project setup.
Each command is ready to paste directly into your terminal.

⸻

1. Preprocessing (Clean & Extract Merchant)

Single transaction

python3 project/src/preprocess.py --text "AMAZON PAY ORDER 1299"

Multiple samples

for t in \
"ZOMATO ORDER 450" \
"IRCTC TICKET 1299" \
"HPCL PETROL PUMP" \
"SPOTIFY PREMIUM PLAN" \
"FLIPKART ORDER ID 2344"
do
    python3 project/src/preprocess.py --text "$t"
done


⸻

2. Train the Model (TF-IDF + Logistic Regression)

python3 project/src/train.py

This:
	•	Loads train.csv + test.csv
	•	Preprocesses data
	•	Trains & evaluates model
	•	Saves model.pkl + vectorizer.pkl to project/model/

⸻

3. Predict for a Single Input

python3 project/src/predict.py --text "Starbucks Coffee Payment 210"

Outputs:
	•	cleaned text
	•	merchant
	•	predicted category
	•	confidence
	•	top tokens
	•	whether feedback is needed

⸻

4. Batch Prediction (CSV Input)

CSV format:

transaction
ZOMATO ORDER 450
SPOTIFY PREM 129
HPCL PUMP

Run:

python3 project/src/predict.py --batch project/data/test.csv

Output file:

project/data/batch_output.csv


⸻

5. Save Manual Feedback (Human-Corrected Labels)

python3 project/src/feedback.py \
  --text "sbi bank loan premium" \
  --pred Bills \
  --correct Bills \
  --merchant "sbi bank"

Writes to:
	•	project/data/feedback.csv
	•	project/data/memory.json

⸻

6. Retrain Using Feedback (Active Learning Loop)

python3 project/src/retrain.py

This updates:
	•	model
	•	vectorizer
	•	evaluation metrics

⸻

7. Generate Synthetic Dataset (Optional)

python3 project/data/gen_data.py

Creates fresh:
	•	train.csv
	•	test.csv

⸻

8. View Merchant Normalization Map

cat project/data/normalization.json


⸻

9. View Feedback History

cat project/data/feedback.csv
cat project/data/memory.json


⸻

10. Full Pipeline (One-Command)

python3 project/src/train.py && \
python3 project/src/predict.py --text "ZOMATO ORDER 299" && \
python3 project/src/predict.py --batch project/data/test.csv


⸻

11. Print File Structure (Optional)

Install tree (if missing):

brew install tree

Then:

tree project


⸻



