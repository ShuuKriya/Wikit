

# WIKIT — AI-Based Financial Transaction Categorisation
Local, explainable, customisable ML pipeline for classifying raw financial transaction strings.

WIKIT is a fully in-house machine learning system that converts messy transaction text  
(e.g., `"AMAZON PAY *ORDER"`, `"SWIGGY * FOOD"`, `"HP PETROL PUMP"`)  
into clean, meaningful financial categories like **Groceries**, **Dining**, **Fuel**, etc.

No external APIs.  
No recurring billing.  
No vendor lock-in.  
Just your own classifier — fast, transparent, fully local.

---

# Features

### AI-Powered Transaction Categorisation
- Logistic Regression + TF-IDF  
- Cleaned + merchant-normalised features  
- Confidence scoring for every prediction  

### Evaluation & Metrics
- Macro F1: **0.93**  
- Accuracy: **0.93**  
- Per-class F1: **0.91–0.97**  
- Confusion matrix + classification report included  

### Data Preprocessing Pipeline
- Text cleaning  
- Merchant normalisation  
- Noise reduction  
- All rules configurable via JSON  

### Admin Tools
- Modify taxonomy (add/remove categories)  
- Change confidence thresholds  
- Inspect or clear merchant memory  
- Review and clear feedback CSV  

### Human-in-the-loop Feedback
- Low-confidence predictions highlight feedback
- User corrections populate:
  - `feedback.csv`
  - `memory.json` (merchant→category overrides)

### Model Retraining (One Click)
- Merges base training data + feedback samples  
- Reweights feedback samples  
- Retrains LR model  
- Updates model + vectorizer  
- Triggered via **Refresh** tab in UI  

### Explainability
Token-level insights including:
- coefficient influence  
- perturbation sensitivity  
- combined impact score with UI bars  

### Batch Mode
- Upload CSV  
- Vectorised inference  
- Optional low-confidence → “Other” routing  
- Downloadable results CSV  

### UI
- Streamlit  
- Custom dark-blue theme  
- Clean production-style layout  

---

# Technology Stack

| Component | Tech |
|----------|------|
| Preprocessing | Python, regex, merchant normaliser |
| Model | Logistic Regression (scikit-learn) |
| Vectoriser | TF-IDF (1–2 grams, 5000 features) |
| UI | Streamlit |
| Storage | JSON + CSV |
| Evaluation | scikit-learn + matplotlib |

---

# Documentation

- `docs/architecture.md`  
- `docs/dataset.md`  
- `docs/training_pipeline.md`  
- `docs/model.md`  
- `docs/explainability.md`  
- `docs/feedback_loop.md`  
- `docs/evaluation.md`  
- `docs/ui.md`  

---

# Project Structure


Wikit/
│
├── project/
│   ├── src/
│   │   ├── train.py
│   │   ├── retrain.py
│   │   ├── evaluate.py
│   │   ├── preprocess.py
│   │   ├── predict.py
│   │   ├── feedback.py
│   │   ├── explain.py
│   │   ├── performance.py
│   │   ├── robustness.py
│   │   ├── taxonomy.py
│   │   └── utils.py
│   │
│   ├── ui/
│   │   ├── .streamlit/
│   │   │   └── config.toml
│   │   └── app.py
│   │
│   ├── model/
│   │   ├── model.pkl
│   │   └── vectorizer.pkl
│   │
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── normalization.json
│   │   ├── taxonomy.json
│   │   ├── memory.json
│   │   └── feedback.csv
│   │
│   ├── evaluation/
│   │   ├── metrics_report.json
│   │   ├── confusion_matrix.png
│   │   ├── performance_report.json
│   │   └── robustness_report.json
│   │
│   ├── config.json
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   ├── dataset.md
│   ├── training_pipeline.md
│   ├── model.md
│   ├── explainability.md
│   ├── feedback_loop.md
│   ├── evaluation.md
│   └── ui.md
│
├── requirements.txt
└── README.md




⸻




# Running the App

Install dependencies:

```bash
pip install -r requirements.txt

Launch UI:

streamlit run project/ui/app.py


⸻

Re-training

Base training:

python3 project/src/train.py

Human-feedback retraining:

python3 project/src/retrain.py


⸻

Evaluation

python3 project/src/evaluate.py

Outputs:
	•	evaluation/metrics_report.json
	•	evaluation/confusion_matrix.png

⸻

Dataset Summary

Total samples: 1000
Train: 650
Test: 352

Train Distribution
	•	Entertainment: 104
	•	Groceries: 98
	•	Dining: 96
	•	Travel: 93
	•	Shopping: 91
	•	Bills: 88
	•	Fuel: 81

Test Distribution
	•	Entertainment: 60
	•	Shopping: 55
	•	Groceries: 52
	•	Dining: 52
	•	Travel: 46
	•	Bills: 46
	•	Fuel: 41

⸻

Performance Summary
	•	Macro F1: 0.93
	•	Accuracy: 0.93
	•	Latency: 0.12 ms / prediction
	•	Throughput: 8300+ predictions/sec
	•	Explainability: token-level contributions

⸻

Demo Checklist (PS Requirements)

This solution includes:
	•	End-to-end ML pipeline
	•	Evaluation with reproducible metrics
	•	Customisable taxonomy
	•	Explainability features
	•	Human feedback mechanism
	•	Batch inference
	•	One-click model retraining
	•	Real + synthetic data usage

⸻

Acknowledgements

Developed by Nishant Bidhu and Swati Nim
Created for AnitaB.org India GHCI 25 Hackathon

