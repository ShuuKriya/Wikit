
⸻

Explainability

This document describes how interpretability is implemented in the WIKIT transaction categorisation system.
The goal is to provide transparent insights into why the model assigns a given category to a transaction.

⸻

1. Rationale

Financial transactions often contain noisy text, abbreviations, or merchant-specific tokens.
Explainability is required to ensure:
	•	Transparency for auditors and reviewers
	•	Debuggability for developers
	•	Trust for end users
	•	Compliance with responsible AI principles

The system exposes token-level contribution scores that reveal which parts of the text influenced the prediction.

⸻

2. Explainability Method

The WIKIT explainability module uses a hybrid approach combining:
	1.	Linear Model Coefficients (from Logistic Regression)
	2.	Perturbation Sensitivity Analysis
	3.	Combined Normalised Score

This approach is computationally lightweight and works fully offline.

2.1 Coefficient-Based Attribution

Since Logistic Regression is linear, each token has a learned weight for each category.

For a predicted category C, if:
	•	weight(token, C) is positive → token pushes prediction toward C
	•	weight(token, C) is negative → token pushes prediction away from C

The top contributing tokens are extracted via:

top_indices = np.argsort(model.coef_[class_index])[-5:][::-1]

These represent the strongest positive features for the chosen class.

⸻

2.2 Perturbation Sensitivity

For each token in the processed input string:
	1.	Remove or mask the token
	2.	Re-run the model
	3.	Measure how much the predicted confidence changes

If removing a token reduces confidence significantly, the token was important.

This provides an additional signal independent of model coefficients.

⸻

3. Combined Explainability Score

For each token, WIKIT computes:

combined_score = (normalized_coefficient + normalized_perturbation) / 2

The values are scaled between [-1, 1].
They are then visualised with a progress bar in the UI.

Interpretation:
	•	Close to +1 → strong positive influence
	•	Close to 0 → neutral
	•	Close to –1 → negative influence

⸻

4. Output Structure

Each token is returned as a dictionary:

{
  "coefficient_score": float,
  "perturbation_score": float,
  "combined_score": float
}

Example:

"swiggy": {
  "coefficient_score": 0.92,
  "perturbation_score": 0.81,
  "combined_score": 0.86
}


⸻

5. Usage in the UI

The Streamlit UI includes an expandable section “Why this prediction?” which displays:
	•	Tokens extracted from processed input
	•	Contribution bars (based on combined score)
	•	Underlying explanation values as JSON for debugging

This ensures full transparency of the system’s decision-making process.

⸻

6. Limitations

Although effective for this problem domain, the following considerations apply:
	•	Coefficients assume linear independence between features
	•	Perturbation analysis is approximate and may vary for ambiguous text
	•	Explainability is token-level, not phrase-level

These limitations are documented to encourage informed use and future extensions.

⸻

7. Summary

The explainability subsystem provides a clear, reproducible, offline-friendly mechanism to understand classification outputs.
It satisfies the Problem Statement requirements for:
	•	Transparency
	•	Responsible AI
	•	Debuggability
	•	Auditability

⸻
