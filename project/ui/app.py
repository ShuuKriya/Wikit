# project/ui/app.py
"""
Streamlit UI for WIKIT Transaction Classifier
Tabs:
 - Single Prediction
 - Batch Mode
 - Admin Panel (taxonomy, config, memory, feedback)
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------
# Ensure src package is importable
# -----------------------
BASE = Path("project")
SRC = (BASE / "src").resolve()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# -----------------------
# Paths
# -----------------------
MODEL_PATH = BASE / "model/model.pkl"
VEC_PATH = BASE / "model/vectorizer.pkl"
MEMORY_PATH = BASE / "data/memory.json"
CONFIG_PATH = BASE / "config.json"
TAX_PATH = BASE / "data/taxonomy.json"
NORM_PATH = BASE / "data/normalization.json"
FEEDBACK_PATH = BASE / "data/feedback.csv"

# -----------------------
# Helpers
# -----------------------
def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return default if default is not None else {}

def save_json(path, obj):
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=4)

# -----------------------
# Load model + vectorizer (fail early if missing)
# -----------------------
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
except Exception as e:
    st.error(f"Failed to load model/vectorizer: {e}")
    raise

memory = load_json(MEMORY_PATH, default={})
config = load_json(CONFIG_PATH, default={
    "confidence_threshold": 0.6,
    "batch_low_confidence_behavior": "other",
    "interactive_low_confidence": True,
    "feedback_sample_weight": 3
})
taxonomy_json = load_json(TAX_PATH, default={"categories": ["Dining","Shopping","Fuel","Groceries","Bills","Entertainment","Travel","Other"]})
taxonomy = taxonomy_json.get("categories", ["Dining","Shopping","Fuel","Groceries","Bills","Entertainment","Travel","Other"])

# -----------------------
# Import preprocess utilities
# -----------------------
from preprocess import preprocess_row, load_normalization_table
from predict import explain_prediction
norm_table = load_normalization_table(str(NORM_PATH))

# -----------------------
# Softmax
# -----------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# -----------------------
# Prediction logic
# -----------------------
def predict_single(raw_text: str):
    cleaned, merchant = preprocess_row(raw_text, norm_table)
    combined = f"{cleaned} {merchant or ''}".strip()

    # memory override
    if merchant and merchant.lower() in memory:
        return {
            "prediction": memory[merchant.lower()],
            "confidence": 1.0,
            "cleaned": cleaned,
            "merchant": merchant,
            "top_tokens": [],
            "needs_feedback": False
        }

    X = vectorizer.transform([combined])
    logits = model.decision_function(X)[0]
    confs = softmax(logits)
    idx = int(np.argmax(confs))

    pred = model.classes_[idx]
    conf = float(confs[idx])

    # explanation tokens (top positive coefficients for predicted class)
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[idx]
    top_idx = np.argsort(coefs)[-5:][::-1]
    top_tokens = [feature_names[i] for i in top_idx]

    needs_feedback = conf < config.get("confidence_threshold", 0.6)

    return {
        "prediction": pred,
        "confidence": conf,
        "cleaned": cleaned,
        "merchant": merchant,
        "top_tokens": top_tokens,
        "needs_feedback": needs_feedback
    }

def predict_batch(df: pd.DataFrame):
    out_rows = []
    for t in df["transaction"].astype(str):
        out = predict_single(t)
        # apply batch rule
        if out["needs_feedback"] and config.get("batch_low_confidence_behavior", "other") == "other":
            out["prediction"] = "Other"
        out_rows.append({
            "transaction": t,
            "prediction": out["prediction"],
            "confidence": out["confidence"],
            "merchant": out["merchant"],
            "cleaned": out["cleaned"],
            "top_tokens": ", ".join(out["top_tokens"])
        })
    return pd.DataFrame(out_rows)

# -----------------------
# Streamlit UI Layout
# -----------------------
st.set_page_config(page_title="WIKIT Classifier", page_icon="💸", layout="wide")
st.title(" Wikit Transaction Classifier")
st.caption("In-house transaction categorisation — single, batch, and admin tools")

tabs = st.tabs([
    "Single Prediction",
    "Batch Mode",
    "Admin",
    "Refresh"
])
# -----------------------
# Tab 1: Single Prediction
# -----------------------
with tabs[0]:
    st.header(" Single Prediction")

    # -----------------------
    # INPUT + LAYOUT
    # -----------------------
    text = st.text_area("Enter transaction text:", height=90)

    col1, col2 = st.columns([3, 1])

    # ----------------------------------------
    # RIGHT COLUMN (Quick Actions — ALWAYS visible)
    # ----------------------------------------
    with col2:
        st.info("Quick actions")

        if st.button("Show config", key="show_cfg"):
            st.json(config)

        if st.button("Show taxonomy", key="show_tax"):
            st.write(taxonomy)

    # ----------------------------------------
    # LEFT COLUMN (Prediction Engine)
    # ----------------------------------------
    with col1:

        # Handle prediction click
        if st.button("Predict", key="single_predict"):
            if not text.strip():
                st.error("Type a transaction string first.")
            else:
                # Save in session state so feedback works
                st.session_state.single_input = text
                st.session_state.single_output = predict_single(text)

        # If prediction exists → show it
        if "single_output" in st.session_state:
            out = st.session_state.single_output

            st.subheader("Prediction")
            st.write(f"**Category:** `{out['prediction']}`")
            st.write(f"**Confidence:** `{out['confidence']:.2f}`")
            st.write(f"**Merchant:** `{out['merchant']}`")
            st.write(f"**Cleaned Text:** `{out['cleaned']}`")
            st.write("**Top tokens:**", out["top_tokens"])

            if out["needs_feedback"]:
                st.warning("Low confidence — consider giving feedback.")
            else:
                st.success("Confident prediction ✔")

            # -------------------------------------------------
            # EXPLAINABILITY BLOCK
            # -------------------------------------------------
            with st.expander(" Why this prediction? (Explainability)"):
                st.caption("Token-level feature contribution based on model coefficients + perturbation sensitivity.")

                exp = explain_prediction(
                    model,
                    vectorizer,
                    out["cleaned"],
                    out["merchant"],
                    f"{out['cleaned']} {out['merchant']}"
                )

                for tok, data in exp.items():
                    st.write(f"**{tok}**")

                    bar_val = max(min(data["combined_score"] / 3, 1), -1)
                    st.progress((bar_val + 1) / 2)

                    st.json(data)

            # -------------------------------------------------
            # FEEDBACK SECTION
            # -------------------------------------------------
            st.markdown("---")
            st.subheader("Feedback")

            correct = st.radio(
                "Was this prediction correct?",
                ["Yes", "No"],
                horizontal=True,
                key="fb_radio_single"
            )

            if correct == "No":
                corrected = st.selectbox(
                    "Choose correct category:",
                    taxonomy,
                    key="fb_select_single"
                )

                if st.button("Save Feedback", key="save_fb_single"):
                    fb_row = pd.DataFrame([{
                        "transaction": st.session_state.single_input,
                        "predicted": out["prediction"],
                        "corrected": corrected,
                        "timestamp": datetime.utcnow().isoformat()
                    }])

                    fb_path = FEEDBACK_PATH
                    if fb_path.exists():
                        fb_row.to_csv(fb_path, mode="a", index=False, header=False)
                    else:
                        fb_row.to_csv(fb_path, index=False)

                    # Update memory small-scale override
                    if out["merchant"]:
                        mem = load_json(MEMORY_PATH, default={})
                        mem[out["merchant"].lower()] = corrected
                        json.dump(mem, open(MEMORY_PATH, "w"), indent=4)

                    st.success("Feedback saved and memory updated.")

            elif correct == "Yes":
                if out["needs_feedback"]:
                    st.info("Thanks! Since the model was low-confidence but correct, no correction is needed. "
                            "The model will improve naturally during normal retraining.")
                else:
                    st.success("Nice — model seems right.")

# -----------------------
# Tab 2: Batch Mode
# -----------------------
with tabs[1]:
    st.header(" Batch Mode")
    st.write("Upload a CSV with a `transaction` column. Low-confidence rows may be labeled 'Other' based on config.")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="batch_upload")

    # Session state for batch predictions
    if "batch_output" not in st.session_state:
        st.session_state.batch_output = None

    # -----------------------------
    # RUN BATCH PREDICTION
    # -----------------------------
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None:
            if "transaction" not in df.columns:
                st.error("CSV must contain a `transaction` column.")
            else:
                st.success(f"Loaded {len(df)} rows.")
                run_batch = st.button("Run Batch Prediction", key="run_batch_btn")

                if run_batch:
                    with st.spinner("Running batch predictions..."):
                        out_df = predict_batch(df)

                    # Save for explainability
                    st.session_state.batch_output = out_df

                    st.subheader("Results")
                    st.dataframe(out_df, use_container_width=True)

                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download results CSV",
                        data=csv_bytes,
                        file_name="batch_output.csv",
                        mime="text/csv",
                        key="download_batch"
                    )
                    st.success("Batch complete.")

    # -----------------------------
    # EXPLAIN SELECTED ROW
    # -----------------------------
    if st.session_state.batch_output is not None:
        out_df = st.session_state.batch_output

        st.markdown("---")
        st.subheader(" Explain a specific row")

        row_idx = st.number_input(
            f"Enter row index (0 – {len(out_df) - 1}):",
            min_value=0,
            max_value=len(out_df) - 1,
            step=1,
            key="exp_row_idx"
        )

        if st.button("Explain Row", key="exp_row_btn"):
            row = out_df.iloc[row_idx]

            # Display details
            st.write(f"**Text:** `{row['transaction']}`")
            st.write(f"**Prediction:** `{row['prediction']}`")
            st.write(f"**Confidence:** `{row['confidence']:.2f}`")
            st.write(f"**Merchant:** `{row['merchant']}`")
            st.write(f"**Cleaned:** `{row['cleaned']}`")

            # Explainability section
            with st.expander(" Why this prediction? (Explainability)"):
                exp = explain_prediction(
                    model,
                    vectorizer,
                    row["cleaned"],
                    row["merchant"],
                    f"{row['cleaned']} {row['merchant']}"
                )

                for tok, data in exp.items():
                    st.write(f"**{tok}**")

                    bar_val = max(min(data["combined_score"] / 3, 1), -1)
                    st.progress((bar_val + 1) / 2)

                    st.json(data)
# -----------------------
# Tab 3: Admin Panel
# -----------------------
with tabs[2]:
    st.header("🛠 Admin Panel - Taxonomy / Config / Memory / Feedback")

    # TAXONOMY
    st.subheader(" Taxonomy")
    tcols = st.columns([3,1])
    with tcols[0]:
        st.write("Current categories:")
        st.write(taxonomy)
        new_cat = st.text_input("Add category (exact string)", key="tax_add")
        if st.button("Add Category", key="tax_add_btn"):
            if new_cat.strip() == "":
                st.error("Cannot add empty category.")
            elif new_cat in taxonomy:
                st.warning("Category exists.")
            else:
                taxonomy.append(new_cat)
                save_json(TAX_PATH, {"categories": taxonomy})
                st.success(f"Added '{new_cat}' to taxonomy.")
    with tcols[1]:
        remove_cat = st.selectbox("Remove category", options=[""] + taxonomy, key="tax_rem")
        if st.button("Remove Category", key="tax_rem_btn"):
            if remove_cat == "" or remove_cat not in taxonomy:
                st.warning("Select a valid category to remove.")
            else:
                taxonomy = [c for c in taxonomy if c != remove_cat]
                save_json(TAX_PATH, {"categories": taxonomy})
                st.success(f"Removed '{remove_cat}' from taxonomy.")

    st.markdown("---")

    # CONFIG
    st.subheader(" Config")
    conf = load_json(CONFIG_PATH, default=config)
    new_thresh = st.slider("Confidence threshold", 0.0, 1.0, conf.get("confidence_threshold", 0.6), 0.01, key="cfg_thresh")
    batch_beh = st.selectbox("Batch low-confidence behavior", ["other", "keep"], index=0 if conf.get("batch_low_confidence_behavior","other")=="other" else 1, key="cfg_batch")
    interactive_flag = st.checkbox("Interactive low-confidence (ask user)", value=conf.get("interactive_low_confidence", True), key="cfg_inter")

    if st.button("Save Config", key="cfg_save"):
        conf["confidence_threshold"] = float(new_thresh)
        conf["batch_low_confidence_behavior"] = batch_beh
        conf["interactive_low_confidence"] = bool(interactive_flag)
        save_json(CONFIG_PATH, conf)
        st.success("Config saved.")

    st.markdown("---")

    # MEMORY
    st.subheader(" Memory (merchant -> category)")
    mem = load_json(MEMORY_PATH, default={})
    st.write(mem)
    del_key = st.selectbox("Select merchant to delete", options=[""] + list(mem.keys()), key="mem_del")
    if st.button("Delete mapping", key="mem_del_btn"):
        if del_key and del_key in mem:
            mem.pop(del_key)
            save_json(MEMORY_PATH, mem)
            st.success(f"Deleted mapping: {del_key}")
        else:
            st.warning("Pick a mapping to delete.")
    if st.button("Clear all memory", key="mem_clear"):
        save_json(MEMORY_PATH, {})
        st.success("Memory cleared.")

    st.markdown("---")

    # FEEDBACK
    st.subheader(" Feedback (feedback.csv)")
    if FEEDBACK_PATH.exists() and FEEDBACK_PATH.stat().st_size > 0:
        try:
            fb_df = pd.read_csv(FEEDBACK_PATH)
            st.dataframe(fb_df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed loading feedback.csv: {e}")
    else:
        st.info("No feedback.csv present or file is empty.")

    if st.button("Clear feedback.csv", key="fb_clear"):
        try:
            if FEEDBACK_PATH.exists():
                FEEDBACK_PATH.unlink()
            st.success("feedback.csv removed.")
        except Exception as e:
            st.error(f"Failed to delete feedback.csv: {e}")

    st.markdown("---")

    # SYSTEM INFO
    st.subheader(" System Info")
    st.json({
        "model": str(MODEL_PATH),
        "vectorizer": str(VEC_PATH),
        "memory": str(MEMORY_PATH),
        "config": str(CONFIG_PATH),
        "taxonomy": str(TAX_PATH),
        "normalization": str(NORM_PATH),
        "feedback": str(FEEDBACK_PATH)
    })

# -----------------------
# Tab 4: Refresh
# -----------------------

with tabs[3]:
    st.header(" Refresh Model")
    st.caption("Retrains the model using training data + feedback data + memory weights")

    st.write("Click the button below to rebuild your model from scratch using the latest feedback.")

    run_refresh = st.button(" Refresh Now")

    if run_refresh:
        st.warning("Refreshing model… sit tight ")
        
        import subprocess
        import joblib

        log_box = st.empty()
        progress = st.progress(0)

        process = subprocess.Popen(
            ["python3", "project/src/retrain.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        logs = ""
        i = 0

        for line in process.stdout:
            logs += line
            log_box.code(logs)
            i += 1
            progress.progress(min(1.0, i / 80))

        process.wait()

        if process.returncode == 0:
            st.success(" Model refreshed successfully!")

            # Reload updated model
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VEC_PATH)

            st.info("Model reloaded and live!")
        else:
            st.error(" Refresh failed. Check logs above.")
        
        progress.empty()

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Wikit — local, explainable transaction categorisation. Built for Round 2.")