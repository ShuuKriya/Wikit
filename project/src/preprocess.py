# Preprocessing functions


# src/preprocess.py
"""
Preprocessing utilities for transaction text.
Functions:
- load_normalization_table(path)
- clean_text(text)
- extract_merchant(text)
- normalize_merchant(merchant, norm_table)
- preprocess_row(text, norm_table)
- preprocess_dataframe(df, text_col, norm_table)
"""

import re
import json
from typing import Tuple, Optional
import pandas as pd

# ---- Configurable patterns ----
# Patterns that commonly appear in bank SMS / descriptors and can be removed or normalized.
ORDER_ID_PATTERN = re.compile(r'\b(order|ord|txn|tx|ref|id|utr)[\s:\-#]*[A-Za-z0-9]+\b', flags=re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r'\b(?:rs\.?|inr|₹)\s*[0-9,]+(?:\.[0-9]{1,2})?\b', flags=re.IGNORECASE)
UPI_HANDLE_PATTERN = re.compile(r'\b[\w\.\-]{2,}@[a-zA-Z0-9]+\b')  # e.g., ramesh@okicici
NON_ALPHANUMERIC = re.compile(r'[^a-zA-Z0-9\s]')
MULTI_SPACES = re.compile(r'\s+')

# common short words to remove (optional)
STOP_TOKENS = set([
    'online', 'payment', 'paid', 'debit', 'credit', 'txn', 'ref',
    'transaction', 'transfer', 'to', 'via', 'on', 'through',
    'order', 'id', 'inr', 'rs', 'amt', 'amount'
])



def load_normalization_table(path: str = "data/normalization.json") -> dict:
    """
    Load merchant normalization table from JSON.
    JSON format example:
    {
       "amazon": ["amazon", "amazon.in", "amazon pay", "amzn"],
       "zomato": ["zomato", "zomato order", "zomatod"]
    }
    The function returns a map from alias -> canonical (reverse mapping).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    # build reverse map: alias -> canonical
    rev = {}
    for canon, aliases in raw.items():
        # ensure canonical itself maps
        rev[canon.lower()] = canon
        if isinstance(aliases, list):
            for a in aliases:
                rev[a.lower()] = canon
        elif isinstance(aliases, str):
            rev[aliases.lower()] = canon
    return rev

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    t = text.strip()

    # Remove alphanumeric IDs (e.g., 8E3T29)
    t = re.sub(r'\b[A-Za-z]*\d+[A-Za-z\d]*\b', " ", t)

    # Remove amounts like 1299, 1299.00, ₹1299, Rs 1299
    t = re.sub(r'\b(?:rs|inr|usd|eur|gbp|aed|cad|₹|\$)\s*[0-9,]+(?:\.[0-9]{1,2})?\b', " ", t, flags=re.IGNORECASE)
    t = re.sub(r'\b[0-9,]+(?:\.[0-9]{1,2})?\b', " ", t)

    # Remove currency words alone
    t = re.sub(r'\b(?:inr|rs|usd|eur|gbp|aed|cad|jpy|cny)\b', " ", t, flags=re.IGNORECASE)

    # Remove UPI handles
    t = UPI_HANDLE_PATTERN.sub(" ", t)

    # Remove 'order', 'id'
    t = re.sub(r'\border\b', " ", t, flags=re.IGNORECASE)
    t = re.sub(r'\bid\b', " ", t, flags=re.IGNORECASE)

    # Remove special chars
    t = NON_ALPHANUMERIC.sub(" ", t)

    # Lowercase
    t = t.lower()

    # Remove stop tokens
    tokens = [tok for tok in t.split() if tok not in STOP_TOKENS]

    # Collapse multiple spaces
    return " ".join(tokens).strip()


def extract_merchant(text: str) -> Optional[str]:
    """
    Heuristic to extract a merchant candidate from the text.
    Strategy:
    - Often the first token(s) are the merchant name.
    - We attempt:
      1) If text contains common separators, take the first chunk.
      2) Else, take first 1-2 tokens as merchant.
    Returns merchant candidate string (lowercased) or None.
    """
    if not text or not isinstance(text, str):
        return None
    t = text.strip()
    # split on common separators used in bank descriptors
    for sep in [" - ", " | ", " / ", " * ", " # "]:
        if sep in t:
            cand = t.split(sep)[0]
            return cand.lower().strip()

    # otherwise use first 1-2 tokens if they look like a name
    toks = t.split()
    if len(toks) == 0:
        return None
    # prefer first token, but if first is like 'upi' or 'pay' skip
    cand = toks[0]
    if cand in ("upi", "pay", "payment", "card"):
        if len(toks) > 1:
            cand = toks[1]
    # also try first two tokens combined
    if len(toks) > 1:
        two = f"{toks[0]} {toks[1]}"
        # if the two-token looks like a brand (contains letters only), prefer it
        if re.match(r'^[a-zA-Z\s]+$', two):
            return two.lower()
    return cand.lower()


def normalize_merchant(merchant: Optional[str], norm_table: dict) -> Optional[str]:
    """
    Map merchant candidate to canonical merchant using normalization table (alias->canonical).
    If not found, return the merchant string as-is (title-cased).
    """
    if not merchant:
        return None
    m = merchant.lower().strip()
    if m in norm_table:
        return norm_table[m]
    # try token-wise matching
    parts = m.split()
    for i in range(len(parts)):
        for j in range(i+1, min(len(parts), i+3) + 1):
            sub = " ".join(parts[i:j])
            if sub in norm_table:
                return norm_table[sub]
    # fallback: title case the merchant candidate
    return m.title()


def preprocess_row(text: str, norm_table: dict) -> Tuple[str, Optional[str]]:
    """
    Clean and extract merchant and normalized merchant.
    Returns: (cleaned_text, normalized_merchant)
    """
    cleaned = clean_text(text)
    merchant = extract_merchant(text)
    norm_merchant = normalize_merchant(merchant, norm_table) if merchant else None
    return cleaned, norm_merchant


def preprocess_dataframe(df: pd.DataFrame,
                         text_col: str = "transaction",
                         norm_table_path: str = "data/normalization.json") -> pd.DataFrame:
    """
    Apply preprocessing to a DataFrame with a column containing raw transaction text.
    Adds columns:
      - cleaned_text
      - merchant
      - merchant_normalized
    """
    norm_table = load_normalization_table(norm_table_path)
    # We rely on pandas apply for clarity
    def _apply(row):
        raw = row.get(text_col, "")
        cleaned, norm_merchant = preprocess_row(raw, norm_table)
        merchant = extract_merchant(raw)
        return pd.Series({
            "cleaned_text": cleaned,
            "merchant": merchant,
            "merchant_normalized": norm_merchant
        })
    processed = df.apply(_apply, axis=1)
    return pd.concat([df.reset_index(drop=True), processed.reset_index(drop=True)], axis=1)


# --- Simple CLI test utility ---
if __name__ == "__main__":
    # small interactive test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Sample transaction string to preprocess")
    parser.add_argument("--norm", type=str, default="data/normalization.json", help="Normalization JSON path")
    args = parser.parse_args()

    norm = load_normalization_table(args.norm)
    sample = args.text or "ZOMATO *ORDER #392847 UPI 420.00"
    cleaned, norm_merchant = preprocess_row(sample, norm)
    print("Raw:      ", sample)
    print("Cleaned:  ", cleaned)
    print("Merchant: ", extract_merchant(sample))
    print("Normalized Merchant: ", norm_merchant)
