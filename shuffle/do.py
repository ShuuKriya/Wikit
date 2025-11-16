import pandas as pd

# ---- CHANGE THIS ----
CSV_PATH = "/Users/shu/test/Wikit/Wikit/shuffle/total_sets.csv"   # e.g. "project/data/train.csv"
# ----------------------

df = pd.read_csv(CSV_PATH)

if "category" not in df.columns:
    raise ValueError("CSV must contain a 'category' column")

# Count categories
dist = df["category"].value_counts()

print("\n=== Category Distribution ===\n")
print(dist)

print("\n=== Percentage Breakdown ===\n")
print((dist / len(df) * 100).round(2))