

⸻

User Interface (UI)

This document explains the design and functionality of the WIKIT UI, implemented using Streamlit.

The UI is structured around four main tabs:
	1.	Single Prediction
	2.	Batch Mode
	3.	Admin Panel
	4.	Refresh (Model Retraining)

The interface follows a clean dark-blue theme configured via config.toml.

⸻

1. Technology
	•	Framework: Streamlit
	•	Layout: Wide mode
	•	Theme: Custom dark-blue (set globally via .streamlit/config.toml)
	•	Backend Integration: All predictions and config operations call local Python modules:
	•	preprocess.py
	•	predict.py
	•	feedback.py
	•	retrain.py

⸻

2. Tabs Overview

2.1 Single Prediction

This is the interactive inference mode for individual transactions.

Features:
	•	Text input box for a single transaction string
	•	Prediction output:
	•	category
	•	confidence score
	•	cleaned text
	•	detected merchant
	•	top contributing tokens
	•	Explainability section:
	•	token-level score
	•	coefficient influence
	•	perturbation sensitivity
	•	Feedback section:
	•	“Was this prediction correct?” → Yes / No
	•	If “No”: user chooses the correct category
	•	Incorrect predictions appended to feedback.csv
	•	Merchant override added to memory.json

Low Confidence Behaviour

If confidence < threshold in config.json, UI displays:

Low confidence — consider giving feedback.

This triggers the human-in-the-loop flow.

⸻

2.2 Batch Mode

Used for processing large CSV files.

Required input: CSV with a column named:

transaction

Features:
	•	Upload CSV
	•	Run batch classification
	•	Automatic handling of low-confidence predictions:
	•	“other” (default)
	•	“keep” (configurable)
	•	Downloadable results CSV
	•	Row-level explainability option:
	•	Users can select a row index
	•	UI shows the same explainability block used in Single Prediction

⸻

2.3 Admin Panel

Admin-level control for taxonomy, configuration, memory, and feedback.

2.3.1 Taxonomy Management
	•	Add category
	•	Remove category

Stored in:

project/data/taxonomy.json

2.3.2 Config Management

Editable settings:
	•	confidence_threshold
	•	batch_low_confidence_behavior
	•	interactive_low_confidence
	•	feedback_sample_weight

Stored in:

project/config.json

2.3.3 Memory Management

Merchant → Category mapping stored in:

project/data/memory.json

Admin tools include:
	•	Display memory
	•	Delete specific merchant mapping
	•	Clear all memory

2.3.4 Feedback Review

Displays contents of feedback.csv.

Admin tools:
	•	Clear feedback.csv
	•	Helpful for resetting the system for new demos

2.3.5 System Info

Shows paths to:
	•	Model
	•	Vectorizer
	•	Memory
	•	Config
	•	Taxonomy
	•	Normalization file
	•	Feedback file

Useful for debugging or evaluation.

⸻

2.4 Refresh (Retrain Model)

This tab triggers the full retraining pipeline via:

python3 project/src/retrain.py

It:
	•	Combines training dataset + feedback samples
	•	Applies sample weights
	•	Fits new TF-IDF vectorizer
	•	Retrains Logistic Regression
	•	Reloads the updated model live in the UI
	•	Streams console logs inside the UI
	•	Shows a progress bar

This supports continuous improvement with zero command-line interaction.

⸻

3. UI Theme

Configured using .streamlit/config.toml:

[theme]
primaryColor="#4E8DFF"
backgroundColor="#14213D"
secondaryBackgroundColor="#1E2A47"
textColor="#F8FAFC"
font="sans serif"

This gives a clean dark-blue theme while keeping all elements readable and consistent.

⸻

4. UX Principles Followed
	•	Minimalist layout
	•	Clear separation of functionality via tabs
	•	Right-side “Quick Actions” to avoid clutter
	•	Progressive disclosure:
	•	Explainability hidden in expanders
	•	Admin tools grouped logically
	•	Error-friendly:
	•	Validation for CSV format
	•	Graceful failure for missing fields
	•	Works offline (no external API dependencies)

⸻

5. Summary

The Streamlit UI provides a complete operational front-end for WIKIT:
	•	Real-time predictions
	•	Transparent explanations
	•	Batch processing
	•	Admin configurability
	•	Human feedback
	•	One-click retraining

All running locally with no external dependencies.

⸻

