import os
import json

# Define folder structure
folders = [
    "project/data",
    "project/model",
    "project/src",
    "project/ui"
]

# Define empty files to create
files = {
    "project/data/train.csv": "",
    "project/data/test.csv": "",
    "project/data/feedback.csv": "",
    "project/data/taxonomy.json": json.dumps({
        "Dining": [],
        "Shopping": [],
        "Fuel": [],
        "Bills": [],
        "Groceries": [],
        "Travel": [],
        "Subscriptions": []
    }, indent=4),
    "project/data/memory.json": json.dumps({}, indent=4),

    "project/model/model.pkl": "",
    "project/model/vectorizer.pkl": "",

    "project/src/preprocess.py": "# Preprocessing functions\n",
    "project/src/train.py": "# Training script\n",
    "project/src/predict.py": "# Prediction script\n",
    "project/src/evaluate.py": "# Evaluation script\n",
    "project/src/retrain.py": "# Retraining script\n",
    "project/src/feedback.py": "# Feedback saving logic\n",
    "project/src/utils.py": "# Utility functions\n",

    "project/ui/app.py": "# Streamlit UI\n"
}


def create_structure():
    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    # Create files
    for filepath, content in files.items():
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created file: {filepath}")

    print("\nProject structure generated successfully!")


if __name__ == "__main__":
    create_structure()
