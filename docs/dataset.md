

⸻


# Dataset Documentation

This document describes the dataset used to train, validate, and test the WIKIT transaction categorisation model.  
No external APIs or proprietary datasets were used. All data sources are openly available or synthetically generated.

---

## 1. Overview

The dataset consists of **1002 total transaction samples**, covering seven spending categories typically present in financial applications.

Categories included:

- Bills  
- Dining  
- Entertainment  
- Fuel  
- Groceries  
- Shopping  
- Travel  

Each sample contains a raw transaction string (e.g., “SWIGGY”, “AMAZON PAY”, “HP PETROL PUMP”) and a corresponding human-assigned category label.

---

## 2. Data Sources

The dataset is composed of three sources:

### 1. Public Real-World Data  
Collected from openly available datasets (e.g., Kaggle transaction datasets).  
Approximately **400 samples** were sourced and cleaned.

### 2. Real Merchant Data from a Public Directory  
Around **100 samples** obtained from GoMaskAI’s publicly listed transaction examples (category-labelled).

### 3. Synthetic Data  
Approximately **500 samples** were generated using controlled prompts to commercial LLMs.  
Care was taken to:
- preserve realistic formatting  
- include genuine merchant names  
- simulate noise (extra numbers, UPI handles, typos, mixed casing)  
- avoid template repetition  

Synthetic data increases coverage of long-tail merchants and rare transaction patterns.

---

## 3. Data Cleaning and Normalisation

Every raw transaction string was passed through a preprocessing pipeline:

1. Lowercasing  
2. Removal of special characters  
3. Splitting UPI handles and card references  
4. Merchant normalisation using a lookup table  
5. Removing duplicates and near-duplicate rows  
6. Trimming whitespace and handling empty tokens  

A custom **normalization.json** file defines merchant aliases such as:

“amazon pay”: “amazon”,
“starbucks coffee”: “starbucks”,
“hpcl pump”: “hp petrol pump”

This helps the ML model generalize across variations.

---

## 4. Dataset Split

The complete dataset was divided as follows:

- **Training:** 650 samples  
- **Testing:** 352 samples  

Splits were manually balanced to prevent category skew.

### Training Distribution

Entertainment    104
Groceries         98
Dining            96
Travel            93
Shopping          91
Bills             88
Fuel              81

### Test Distribution

Entertainment    60
Shopping         55
Groceries        52
Dining           52
Travel           46
Bills            46
Fuel             41

---

## 5. File Structure

All dataset files are stored under:

project/data/
train.csv
test.csv
normalization.json
feedback.csv

### train.csv  
Contains:
- transaction  
- category  

### test.csv  
Same format as train.csv; never used during training.

### normalization.json  
Merchant alias mapping table.

### feedback.csv  
Empty initially; populated via the UI when users correct wrong predictions.

---

## 6. Quality Control

To ensure the dataset remains realistic and competition-ready:

- No placeholder merchants  
- No hallucinated categories  
- No duplicated entries  
- Synthetic samples were manually inspected  
- Category balance was enforced  
- Noise patterns were sampled from real transaction behaviour  

---

## 7. Ethical Use

The dataset contains no personal or user-identifiable information.  
All merchants and transaction strings are generic, anonymised, and non-sensitive.

---

## 8. Summary

The final dataset:
- represents a diverse mixture of real and synthetic transactions  
- includes balanced categories for unbiased evaluation  
- is fully reproducible  
- supports robust model training and retraining  

This dataset forms the foundation for the WIKIT transaction classification system.


⸻
