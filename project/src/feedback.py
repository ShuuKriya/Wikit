# project/src/feedback.py

import csv
import json
import os
import pandas as pd
from datetime import datetime

FEEDBACK_PATH = "project/data/feedback.csv"
MEMORY_PATH = "project/data/memory.json"

def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return {}
    with open(MEMORY_PATH, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=4)

def add_feedback(original_text, predicted, corrected, merchant):

    # Ensure feedback file exists with headers
    header = ["transaction", "predicted", "corrected", "merchant", "timestamp"]

    # Case 1: File doesn't exist -> create it
    if not os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Case 2: File exists but is empty -> write header
    if os.path.exists(FEEDBACK_PATH) and os.path.getsize(FEEDBACK_PATH) == 0:
        with open(FEEDBACK_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the row safely
    with open(FEEDBACK_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            original_text,
            predicted,
            corrected,
            merchant,
            datetime.utcnow().isoformat()
        ])

    # Update memory.json
    memory = load_memory()
    if merchant and corrected:
        memory[merchant.lower()] = corrected
        save_memory(memory)

    print("\nFeedback saved! Memory updated.")
    print("→ CSV appended")
    print("→ Merchant learned\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--correct", type=str, required=True)
    parser.add_argument("--merchant", type=str, required=True)
    args = parser.parse_args()

    add_feedback(args.text, args.pred, args.correct, args.merchant)