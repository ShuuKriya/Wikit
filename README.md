
# WIKIT — AI-Based Financial Transaction Categorisation  
**Local, explainable, customisable ML pipeline for classifying raw financial transaction strings.**

WIKIT is a fully in-house machine learning system that converts messy transaction text  
(e.g., `"AMAZON PAY *ORDER"`, `"SWIGGY * FOOD"`, `"HP PETROL PUMP"`)  
into clean, meaningful financial categories like **Groceries**, **Dining**, **Fuel**, etc.

No external APIs.  
No recurring billing.  
No vendor lock-in.  
Just **your own classifier**, fast and transparent.

---

#  Features

### ** AI-Powered Transaction Categorisation**
- Logistic Regression + TF-IDF  
- Cleaned + merchant-normalised features  
- Confidence scoring for every prediction  

### ** Evaluation & Metrics**
- Macro F1: **0.93**  
- Accuracy: **0.93**  
- Per-class F1: **0.91–0.97**  
- Confusion matrix + classification report included  

### ** Data Preprocessing Pipeline**
- Text cleaning  
- Merchant normalisation  
- Noise/stopword reduction  
- Normalisation rules configurable via JSON  

### ** Admin Tools**
- Modify taxonomy (add/remove categories)  
- Change confidence thresholds  
- Inspect or clear merchant memory  
- Review and clear feedback.csv  

### ** Human-in-the-loop Feedback**
- Low-confidence predictions automatically highlight feedback section  
- User corrections go into:
  - `feedback.csv`
  - `memory.json` (merchant→category mapping)

### ** Model Retraining (One Click)**
- Merges base training data + feedback  
- Reweights feedback samples  
- Retrains LR model  
- Updates model + vectorizer live  
- Triggered via **Refresh** tab in UI  

### ** Explainability**
Token-level explanations showing:
- coefficient influence  
- perturbation sensitivity  
- combined impact score with progress bars  

### ** Batch Mode**
- Upload CSV  
- Vectorised inference  
- Configurable low-confidence handling  
- Download results CSV  

### ** UI**
- Built with Streamlit  
- Custom dark-blue theme  
- Clean, minimal, production-style layout  

---

# 🔧 Technology Stack

| Component | Tech |
|----------|------|
| Preprocessing | Python, regex, custom merchant normaliser |
| Model | Logistic Regression (scikit-learn) |
| Vectoriser | TF-IDF (unigram + bigram, 5000 max features) |
| UI | Streamlit |
| Storage | JSON + CSV |
| Evaluation | scikit-learn + matplotlib |

---

#  Project Structure

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
│   │
│   ├── ui/
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
│   │   └── confusion_matrix.png
│   │
│   ├── config.json
│   └── README.md
│
└── requirements.txt

---

#  Running the App

Install dependencies:

```bash
pip install -r requirements.txt

Launch Streamlit UI:

streamlit run project/ui/app.py


⸻

 Re-training

python3 project/src/train.py

Human-feedback based retraining:

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

Train Set (650 samples)

Entertainment    104
Groceries         98
Dining            96
Travel            93
Shopping          91
Bills             88
Fuel              81

Test Set (352 samples)

Entertainment    60
Shopping         55
Groceries        52
Dining           52
Travel           46
Bills            46
Fuel             41


⸻

 Performance Summary
	•	Macro F1: 0.93
	•	Accuracy: 0.93
	•	Latency: 0.12 ms / prediction
	•	Throughput: 8300+ predictions/second
	•	Explainability: token-level contributions

⸻

 Demo Requirements (PS Guidelines)

This solution covers:

✔ End-to-end pipeline
✔ Evaluation with reproducible metrics
✔ Customisable taxonomy
✔ Explainability
✔ Human feedback loop
✔ Batch inference
✔ Model retraining
✔ Real + synthetic data usage

⸻

 Acknowledgements

Developed by Shuu with ML engineering support from Smile 🫶
