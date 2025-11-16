

⸻

Feedback Loop

This document explains how WIKIT implements a human-in-the-loop feedback system for improving and correcting transaction categorisation over time.

The feedback loop supports two goals:
	1.	Allow users to correct low-confidence predictions.
	2.	Enable the system to learn from corrections without retraining from scratch manually.

⸻

1. Purpose of Feedback

Financial transaction text can be ambiguous or domain-specific.
Examples:
	•	“FOODHUB DIGITAL” → Dining or Groceries
	•	“NYKAA ONLINE” → Shopping or Cosmetics
	•	“METRO GAS STN” → Fuel or Transport

A static classifier cannot cover every scenario.
The feedback loop ensures:
	•	Continuous improvement
	•	Adaptation to new merchants
	•	Cleaner predictions in future sessions
	•	Customisation for domain-specific rules

⸻

2. When Feedback is Triggered

During Single Prediction, each output includes a confidence score.

If the confidence is below the threshold defined in config.json, the UI marks the result as:

Low confidence — consider giving feedback.

User sees two options:
	•	“Yes” → Prediction is correct
	•	“No” → Prediction is incorrect (triggers correction UI)

⸻

3. Feedback Data Capture

When a user corrects a prediction, the following fields are appended to project/data/feedback.csv:
	•	transaction → original input text
	•	predicted → model’s predicted category
	•	corrected → user-selected true category
	•	merchant → extracted merchant string
	•	timestamp → UTC timestamp

This CSV acts as an incremental dataset for future retraining.

⸻

4. Merchant Memory Update

In addition to saving feedback, the system updates memory.json.

If a merchant string is detected (e.g., “swiggy”, “croma”, “nykaa”), it stores:

"swiggy": "Dining"

This acts as a rule-based override:
	•	Merchant appears again
	•	Model prediction is bypassed
	•	Category is returned with confidence = 1.0

This improves accuracy on repeated merchants without retraining.

⸻

5. Retraining With Feedback

The Refresh tab runs:

python3 project/src/retrain.py

Retraining merges:
	1.	Original training dataset
	2.	Feedback samples from feedback.csv

Feedback samples are assigned increased importance:

sample_weight = feedback_weight (default = 3)

This ensures the model learns faster from human corrections.

Outputs saved:
	•	Updated model.pkl
	•	Updated vectorizer.pkl
	•	Updated evaluation metrics

⸻

6. Behaviour Summary

Scenario	What Happens
User clicks Yes	No changes; prediction was correct
User clicks No	Correction saved to feedback.csv
Merchant found	memory.json updated for instant recall
Retrain triggered	Model re-learns from base + feedback

This satisfies the PS requirement for continuous learning, adaptability, and responsible AI.

⸻

7. Limitations
	•	Merchant memory is literal-string based (no fuzzy matching).
	•	Feedback impacts the model only after retraining.
	•	Feedback is assumed correct (no conflict resolution).
	•	Heavy feedback bias may distort category balance (mitigated via weights).

⸻

8. Summary

The feedback loop gives WIKIT:
	•	Personalisation
	•	Continuous improvement
	•	Reduced recurring mistakes
	•	Stronger robustness over time


⸻

