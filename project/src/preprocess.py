# project/src/preprocess.py
import re
import json
from typing import Tuple, Optional
import pandas as pd

# try optional rapidfuzz for fuzzy matching
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# ---- Configurable patterns ----
ORDER_ID_PATTERN = re.compile(r'\b(order|ord|txn|tx|ref|id|utr)[\s:\-#]*[A-Za-z0-9]+\b', flags=re.IGNORECASE)
UPI_HANDLE_PATTERN = re.compile(r'\b[\w\.\-]{2,}@[a-zA-Z0-9]+\b')  # e.g., ramesh@okicici
NON_ALPHANUMERIC = re.compile(r'[^a-zA-Z0-9\s]')
MULTI_SPACES = re.compile(r'\s+')

STOP_TOKENS = set([
    'online', 'payment', 'paid', 'debit', 'credit', 'txn', 'ref',
    'transaction', 'transfer', 'to', 'via', 'on', 'through',
    'order', 'id', 'inr', 'rs', 'amt', 'amount'
])

def load_normalization_table(path: str = "project/data/normalization.json") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    rev = {}
    for canon, aliases in raw.items():
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

    # remove typical IDs (keep words, remove long alphanum IDs)
    t = re.sub(r'\b[A-Za-z]*\d+[A-Za-z\d]*\b', " ", t)

    # remove currency-like tokens and amounts
    t = re.sub(r'\b(?:rs|inr|usd|eur|gbp|aed|cad|₹|\$)\s*[0-9,]+(?:\.[0-9]{1,2})?\b', " ", t, flags=re.IGNORECASE)
    t = re.sub(r'\b[0-9,]+(?:\.[0-9]{1,2})?\b', " ", t)

    # remove upi handles
    t = UPI_HANDLE_PATTERN.sub(" ", t)

    # remove certain words
    t = re.sub(r'\border\b', " ", t, flags=re.IGNORECASE)
    t = re.sub(r'\bid\b', " ", t, flags=re.IGNORECASE)

    # remove special chars
    t = NON_ALPHANUMERIC.sub(" ", t)

    # lowercase
    t = t.lower()

    # remove stop tokens
    tokens = [tok for tok in t.split() if tok not in STOP_TOKENS]

    # collapse spaces
    return " ".join(tokens).strip()

def extract_merchant(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    t = text.strip()

    # split on common separators
    for sep in [" - ", " | ", " / ", " * ", " # ", ",", ":" ]:
        if sep in t:
            cand = t.split(sep)[0]
            return cand.lower().strip()

    toks = t.split()
    if len(toks) == 0:
        return None

    # heuristics: skip leading UPI/prefix tokens
    first = toks[0].lower()
    if first in ("upi", "pay", "payment", "card", "debit", "credit"):
        if len(toks) > 1:
            cand = toks[1]
        else:
            cand = first
    else:
        # take first two tokens if they look like text
        if len(toks) > 1:
            two = f"{toks[0]} {toks[1]}"
            if re.match(r'^[A-Za-z\s]+$', two):
                return two.lower()
        cand = toks[0]
    return cand.lower()

def _fuzzy_match(term: str, candidates: list) -> Optional[str]:
    if not term or not candidates:
        return None
    if _HAS_RAPIDFUZZ:
        match = rf_process.extractOne(term, candidates, scorer=rf_fuzz.token_sort_ratio)
        if match and match[1] >= 70:
            return match[0]
        return None
    else:
        # simple exact substring fallback
        term = term.lower()
        for c in candidates:
            if c in term or term in c:
                return c
        return None

def normalize_merchant(merchant: Optional[str], norm_table: dict) -> Optional[str]:
    if not merchant:
        return None
    m = merchant.lower().strip()
    if m in norm_table:
        return norm_table[m]
    # try token-wise match
    parts = m.split()
    for i in range(len(parts)):
        for j in range(i+1, min(len(parts), i+3) + 1):
            sub = " ".join(parts[i:j])
            if sub in norm_table:
                return norm_table[sub]
    # fuzzy match against known aliases
    candidates = list(norm_table.keys())
    fm = _fuzzy_match(m, candidates)
    if fm:
        return norm_table.get(fm, fm.title())
    return m.title()

def preprocess_row(text: str, norm_table: dict) -> Tuple[str, Optional[str]]:
    cleaned = clean_text(text)
    merchant = extract_merchant(text)
    norm_merchant = normalize_merchant(merchant, norm_table) if merchant else None
    return cleaned, norm_merchant

def preprocess_dataframe(df: pd.DataFrame,
                         text_col: str = "transaction",
                         norm_table_path: str = "project/data/normalization.json") -> pd.DataFrame:
    norm_table = load_normalization_table(norm_table_path)
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

# CLI quick test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Sample transaction string to preprocess")
    parser.add_argument("--norm", type=str, default="project/data/normalization.json", help="Normalization JSON path")
    args = parser.parse_args()

    norm = load_normalization_table(args.norm)
    sample = args.text or "ZOMATO *ORDER #392847 UPI 420.00"
    cleaned, norm_merchant = preprocess_row(sample, norm)
    print("Raw:      ", sample)
    print("Cleaned:  ", cleaned)
    print("Merchant: ", extract_merchant(sample))
    print("Normalized Merchant: ", norm_merchant)