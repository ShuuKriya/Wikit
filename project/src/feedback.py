# Feedback saving logic


import csv
import json
import os

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
    # 1. Write to feedback.csv
    header = ["transaction", "predicted", "corrected"]

    file_exists = os.path.exists(FEEDBACK_PATH)
    with open(FEEDBACK_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([original_text, predicted, corrected])

    # 2. Update memory.json (merchant → corrected category)
    memory = load_memory()
    if merchant and corrected:
        memory[merchant.lower()] = corrected
        save_memory(memory)

    print("\nFeedback saved! Memory updated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--correct", type=str, required=True)
    parser.add_argument("--merchant", type=str, required=True)
    args = parser.parse_args()

    add_feedback(args.text, args.pred, args.correct, args.merchant)
