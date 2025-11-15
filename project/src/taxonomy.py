# project/src/taxonomy.py
import json
import os
import argparse

TAXONOMY_PATH = "project/data/taxonomy.json"

DEFAULT_TAXONOMY = {
    "categories": [
        "Dining",
        "Shopping",
        "Fuel",
        "Groceries",
        "Bills",
        "Entertainment",
        "Travel",
        "Other"
    ]
}

# ----------------------------------------------------
# Load taxonomy
# ----------------------------------------------------
def load_taxonomy():
    if not os.path.exists(TAXONOMY_PATH):
        save_taxonomy(DEFAULT_TAXONOMY)
        return DEFAULT_TAXONOMY
    
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if "categories" not in data:
                raise ValueError
            return data
        except Exception:
            # Reset corrupted file
            save_taxonomy(DEFAULT_TAXONOMY)
            return DEFAULT_TAXONOMY


# ----------------------------------------------------
# Save taxonomy
# ----------------------------------------------------
def save_taxonomy(taxonomy):
    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=4)


# ----------------------------------------------------
# Add category
# ----------------------------------------------------
def add_category(cat: str) -> bool:
    taxonomy = load_taxonomy()
    cats = taxonomy["categories"]

    cat = cat.strip()
    if cat in cats:
        return False

    cats.append(cat)
    save_taxonomy(taxonomy)
    return True


# ----------------------------------------------------
# Remove category
# ----------------------------------------------------
def remove_category(cat: str) -> bool:
    taxonomy = load_taxonomy()
    cats = taxonomy["categories"]

    cat = cat.strip()
    if cat not in cats:
        return False

    cats.remove(cat)
    save_taxonomy(taxonomy)
    return True


# ----------------------------------------------------
# Pretty print
# ----------------------------------------------------
def print_taxonomy():
    taxonomy = load_taxonomy()
    print("\n=== Current Categories ===")
    for c in taxonomy["categories"]:
        print(f"• {c}")
    print()


# ----------------------------------------------------
# CLI
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage category taxonomy.")
    parser.add_argument("--list", action="store_true", help="List all categories")
    parser.add_argument("--add", type=str, help="Add a category")
    parser.add_argument("--remove", type=str, help="Remove a category")

    args = parser.parse_args()

    if args.list:
        print_taxonomy()
        exit(0)

    if args.add:
        if add_category(args.add):
            print(f"Added category: {args.add}")
        else:
            print(f"Category already exists: {args.add}")

    if args.remove:
        if remove_category(args.remove):
            print(f"Removed category: {args.remove}")
        else:
            print(f"Category not found: {args.remove}")

    # Always print final state
    print_taxonomy()