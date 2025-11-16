Here you go — clean, professional, minimal-but-complete architecture.md with no emojis and no fluff.
Fits perfectly with the rest of your documentation set.

⸻

architecture.md

System Architecture Overview

This document provides the end-to-end technical architecture for the WIKIT transaction categorisation system.
It includes the data flow, major components, responsibilities, and interactions between modules.

⸻

1. High-Level Architecture

WIKIT is structured as a modular, offline-first machine learning pipeline with a thin Streamlit-based UI.
All computation is performed locally with no external API calls.

                ┌──────────────────┐
                │   User / Admin    │
                └─────────┬────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │   Streamlit UI │
                 └───────┬────────┘
                         │
                 ┌───────┴───────────────────────────┐
                 │        Application Core            │
                 │   (Batch, Single, Admin, Refresh)  │
                 └───────┬──────────────┬────────────┘
                         │              │
           ┌─────────────▼───┐     ┌────▼────────────────┐
           │ Prediction Engine│     │ Admin Configuration  │
           └───────────┬─────┘     └───────────┬─────────┘
                       │                       │
             ┌─────────▼──────────┐     ┌──────▼─────────┐
             │ Preprocessing Layer │     │Config/Taxonomy │
             └─────────┬──────────┘     └──────┬─────────┘
                       │                       │
             ┌─────────▼──────────────┐        │
             │ TF-IDF Vectorizer      │        │
             └──────────┬─────────────┘        │
                        │                      │
             ┌──────────▼──────────────┐       │
             │ Logistic Regression Model│       │
             └───────────┬─────────────┘       │
                        │                      │
 ┌───────────────┐ ┌────▼─────────┐ ┌─────────▼──────────┐
 │ normalization  │ │ memory.json  │ │  feedback.csv       │
 │ rules (JSON)   │ │ merchant map │ │  user corrections   │
 └───────────────┘ └───────────────┘ └────────────────────┘


⸻

2. Component Breakdown

2.1 Preprocessing Layer

File: preprocess.py

Functions:
	•	Normalises raw transaction strings
	•	Extracts merchant name
	•	Converts text into cleaned_text + merchant_normalized
	•	Applies rule-based normalization (JSON-driven)

Inputs:
	•	Raw text
	•	normalization.json

Outputs:
	•	cleaned_text
	•	merchant_normalized

This ensures consistency across training, inference, and retraining.

⸻

2.2 Feature Engineering

Vectorizer: TF-IDF (unigram + bigram, max_features=5000)

Responsibilities:
	•	Transform cleaned text into numerical feature vectors
	•	Preserve vocabulary at training time for consistent inference
	•	Support bigram patterns helpful for merchant/phrase detection
(e.g., “petrol pump”, “movie ticket”)

Artifacts:
	•	vectorizer.pkl

⸻

2.3 Classification Model

Model: Logistic Regression (scikit-learn)

Reasons for selection:
	•	Fast to train
	•	High interpretability (coefficients used for explainability)
	•	Robust performance for text classification
	•	Works well with sparse TF-IDF matrices

Outputs:
	•	Category label
	•	Confidence score (softmax-applied)

Artifacts:
	•	model.pkl

⸻

3. Prediction Engine

Files:
	•	predict.py
	•	UI call in app.py

Pipeline:
	1.	Preprocess input
	2.	Apply memory override (if merchant exists in memory.json)
	3.	TF-IDF transform
	4.	Logistic Regression inference
	5.	Softmax-based confidence
	6.	Determine if feedback is needed

Explainability is generated in this stage using:
	•	coefficient inspection
	•	token perturbation scoring
	•	combined contribution metrics

⸻

4. Memory System (Merchant → Category)

File: memory.json

Purpose:
	•	Store deterministic overrides for known merchants
	•	Avoid repeating mistakes
	•	Improve system stability

Usage:
	•	On prediction: memory checked first
	•	On feedback: memory is updated
	•	On retrain: model does not override memory; both coexist

⸻

5. Feedback Loop

File: feedback.py

If user indicates prediction is wrong:
	•	Append row to feedback.csv
	•	Add merchant correction to memory.json
	•	Corrections are incorporated during retraining with higher sample weight

Artifacts:
	•	feedback.csv
	•	memory.json

⸻

6. Retraining Pipeline

File: retrain.py

Steps:
	1.	Load original training data
	2.	Load feedback data
	3.	Convert feedback rows into training samples
	4.	Apply sample weighting (feedback_sample_weight)
	5.	Preprocess all rows
	6.	Refit TF-IDF
	7.	Train LR model
	8.	Save artifacts
	9.	Output metrics to evaluation/

This enables a human-in-the-loop reinforcement mechanism.

⸻

7. Evaluation Subsystem

File: evaluate.py

Outputs:
	•	Macro F1
	•	Per-class metrics
	•	Confusion matrix
	•	metrics_report.json
	•	confusion_matrix.png

Evaluation ensures reproducibility and transparency.

⸻

8. Streamlit UI Architecture

File: project/ui/app.py

Tabs:
	•	Single Prediction
	•	Batch Mode
	•	Admin Panel
	•	Refresh

Admin Panel enables:
	•	Editing taxonomy
	•	Changing config values
	•	Inspecting memory
	•	Viewing feedback
	•	System info

Batch Mode:
	•	Vectorised inference
	•	Low-confidence handling
	•	Row-level explainability

Refresh:
	•	Executes retraining script
	•	Reloads updated model live

⸻

9. Configuration System

Files:
	•	config.json
	•	taxonomy.json

Configurable parameters:
	•	Confidence threshold
	•	Batch low confidence handling
	•	Feedback sample weighting
	•	Taxonomy categories

All stored as JSON for easy version control.

⸻

10. Persistent Files

project/data/
    train.csv
    test.csv
    normalization.json
    taxonomy.json
    memory.json
    feedback.csv

project/model/
    model.pkl
    vectorizer.pkl

project/evaluation/
    metrics_report.json
    confusion_matrix.png


⸻

11. End-to-End Data Flow Summary

Prediction Flow
Raw text → preprocessing → memory check → TF-IDF → model → confidence → UI → optional feedback

Training Flow
train.csv → preprocessing → vectorizer.fit → LR.fit → save model/vectorizer

Retraining Flow
(train.csv + feedback.csv) → weighted preprocessing → TF-IDF fit → LR fit → update model

Evaluation Flow
test.csv → preprocessing → vectorizer.transform → model.predict → metrics + plots

⸻

If you want, I can also generate:
	•	architecture_diagram.png (ASCII → PNG)
	•	pipeline diagram
	•	component summary table

Just say the word.