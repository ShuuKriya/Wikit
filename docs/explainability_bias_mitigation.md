
## Explainability & Bias Mitigation

This document briefly describes how **Explainability** and **Bias Mitigation** are implemented in the **WIKIT / Transify** financial transaction categorisation system.

-----

## 1\. Explainability

### Why Explainability?

Financial transaction strings are often noisy and inconsistent. Explainability ensures:

  * **Transparency** for auditors
  * **Debuggability** for developers
  * **Trust and clarity** for users
  * **Compliance** with Responsible AI practices

The system provides **token-level insights** into why a prediction was made.

### Explainability Method

Transify uses a **hybrid offline explainability pipeline**:

  * Linear Model Coefficients (Logistic Regression)
  * Perturbation Sensitivity Analysis
  * Combined Normalised Score

This approach is lightweight, fast, and fully offline.

#### 1\. Coefficient-Based Attribution

Logistic Regression assigns weights to each token for each class.

  * **Positive weight** → token supports the predicted category
  * **Negative weight** → token pushes away

Top contributing features for a predicted class are extracted via:

> `top_indices = np.argsort(model.coef_[class_index])[-5:][::-1]`

#### 2\. Perturbation Sensitivity

For each token:

  * Remove or mask the token
  * Re-run the model
  * Compute confidence drop

A large drop implies a strong influence.

This acts as a **model-agnostic validation** for coefficient-based explanations.

#### 3\. Combined Score

Each token receives a final combined score:

> $combined\_score = (\text{normalized\_coefficient} + \text{normalized\_perturbation}) / 2$

Score Interpretation:

  * **+1** → strong positive influence
  * **0** → neutral
  * **–1** → negative influence

These are displayed in the UI using visual contribution bars.

#### Output Format (Example)

```json
"swiggy": {
  "coefficient_score": 0.92,
  "perturbation_score": 0.81,
  "combined_score": 0.86
}
```

#### UI Integration

Under the “Why this prediction?” section, the UI shows:

  * Tokens extracted from the processed input
  * Contribution bars based on combined score
  * Raw explainability values for debugging

This ensures **transparent and auditable predictions**.

-----

## 2\. Bias Mitigation

Even though the dataset contains **no sensitive demographic attributes**, Transify includes multiple mechanisms to protect against structural and noise-based biases.

### 1\. Merchant Normalization

Different representations like:

  * swiggy online
  * SWIGGY \* ORDER
  * swigy

are normalized to:

> "swiggy"

This prevents unequal treatment caused by formatting differences.

### 2\. Deterministic Merchant Memory

If a merchant repeatedly maps to a category based on user corrections, e.g.,

> nykaa → Shopping

it is stored in **memory.json**.

This ensures:

  * **Consistent** predictions
  * **Reduced error** on rare merchants
  * **User-governed** correctness

### 3\. Feedback-Weighted Learning

All corrections are logged in **feedback.csv**.
During retraining, these samples are **weighted more heavily**, ensuring:

  * the model adapts to real user data
  * the system becomes **fairer** over time
  * misclassifications are **not repeated**

### 4\. Offline-Only Inference

Transify runs fully offline:

  * **No external API biases**
  * **No personal data leakage**
  * No external influence on model behaviour

### 5\. Explainability Reinforces Fairness

Because every prediction is explainable, users can:

  * **Identify anomalies**
  * **Detect potential model drift**
  * **Correct incorrect associations**
  * Improve the model continuously

-----
