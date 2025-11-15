# project/src/generate_report.py
"""
Generate a final evaluation bundle:
- Runs evaluation / performance / robustness scripts if outputs missing
- Runs explainability on a few sample rows
- Produces final_report.json and final_report.md in project/evaluation/
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# make sure project/src is importable for explain.py / utils
sys.path.append(str(Path("project/src").resolve()))

BASE = Path("project")
EVAL_DIR = BASE / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# expected outputs
METRICS_JSON = EVAL_DIR / "metrics_report.json"
PERF_JSON = EVAL_DIR / "performance_report.json"
ROBUST_JSON = EVAL_DIR / "robustness_report.json"
CM_PNG = EVAL_DIR / "confusion_matrix.png"
FINAL_JSON = EVAL_DIR / "final_report.json"
FINAL_MD = EVAL_DIR / "final_report.md"

SCRIPTS = {
    "evaluate": Path("project/src/evaluate.py"),
    "performance": Path("project/src/performance.py"),
    "robustness": Path("project/src/robustness.py"),
}

def run_script_if_missing(key, script_path, expected_files):
    """Run a script (via subprocess) if any expected file is missing."""
    missing = [f for f in expected_files if not Path(f).exists()]
    if missing:
        print(f"[report] Running {script_path} because {missing} missing...")
        subprocess.run(["python3", str(script_path)], check=True)
    else:
        print(f"[report] {script_path.name} outputs present, skipping run.")

def safe_load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def collect_explanations(sample_n=5):
    """
    Import explain.explain and run on sample rows from test.csv.
    If import fails, try to run explain.py via subprocess for each sample.
    """
    test_csv = BASE / "data/test.csv"
    samples = []
    if not test_csv.exists():
        return []

    import pandas as pd
    df = pd.read_csv(test_csv)
    if "transaction" not in df.columns:
        return []

    rows = df["transaction"].dropna().astype(str).tolist()
    # pick first N distinct examples
    chosen = rows[:sample_n]

    explanations = []
    # try programmatic import first
    try:
        from explain import explain as explain_fn  # type: ignore
        for t in chosen:
            try:
                out = explain_fn(t)
                explanations.append(out)
            except Exception as e:
                explanations.append({"input": t, "error": str(e)})
    except Exception:
        # fallback: call explain.py via subprocess and parse stdout
        for t in chosen:
            try:
                proc = subprocess.run(
                    ["python3", "project/src/explain.py", "--text", t],
                    capture_output=True, text=True, check=True
                )
                explanations.append(json.loads(proc.stdout))
            except Exception as e:
                explanations.append({"input": t, "error": str(e)})
    return explanations

def generate_markdown(bundle: dict):
    lines = []
    lines.append("# Final Evaluation Report\n")
    lines.append("## Summary\n")
    lines.append(f"- Metric file: `{METRICS_JSON.relative_to(BASE)}`")
    lines.append(f"- Performance file: `{PERF_JSON.relative_to(BASE)}`")
    lines.append(f"- Robustness file: `{ROBUST_JSON.relative_to(BASE)}`")
    lines.append("")
    # classification summary (if present)
    if bundle.get("metrics"):
        cr = bundle["metrics"].get("classification_report")
        if cr:
            macro_f1 = cr.get("macro avg", {}).get("f1-score")
            lines.append(f"**Macro F1 (from metrics):** `{macro_f1}`\n")
    lines.append("## Metrics (summary)\n")
    if bundle.get("metrics"):
        lines.append("### Classification Report (macro & per-class)\n")
        lines.append("```json")
        lines.append(json.dumps(bundle["metrics"]["classification_report"], indent=2))
        lines.append("```\n")
    if bundle.get("performance"):
        lines.append("## Performance\n")
        lines.append("```json")
        lines.append(json.dumps(bundle["performance"], indent=2))
        lines.append("```\n")
    if bundle.get("robustness"):
        lines.append("## Robustness\n")
        lines.append("```json")
        # keep it short
        rb_short = {
            "num_samples": bundle["robustness"].get("num_samples"),
            "robustness_score": bundle["robustness"].get("robustness_score"),
            "mismatch_rate": bundle["robustness"].get("mismatch_rate")
        }
        lines.append(json.dumps(rb_short, indent=2))
        lines.append("```\n")
    if bundle.get("explanations"):
        lines.append("## Example Explainability Outputs\n")
        for ex in bundle["explanations"]:
            lines.append(f"### Input: `{ex.get('input')}`")
            if ex.get("error"):
                lines.append(f"- Error generating explanation: `{ex['error']}`")
                continue
            lines.append(f"- Prediction: **{ex.get('prediction')}**")
            lines.append(f"- Confidence: `{ex.get('confidence')}`")
            lines.append("```json")
            # include token_importance but keep it readable
            ti = ex.get("token_importance", {})
            lines.append(json.dumps(ti, indent=2))
            lines.append("```")
    lines.append("\n---\n")
    lines.append("Generated by `project/src/generate_report.py`\n")
    return "\n".join(lines)

def main():
    # 1) ensure evaluation etc. outputs exist (run scripts if needed)
    run_script_if_missing("evaluate", SCRIPTS["evaluate"], [METRICS_JSON, CM_PNG])
    run_script_if_missing("performance", SCRIPTS["performance"], [PERF_JSON])
    run_script_if_missing("robustness", SCRIPTS["robustness"], [ROBUST_JSON])

    # 2) load outputs
    metrics = safe_load_json(METRICS_JSON)
    performance = safe_load_json(PERF_JSON)
    robustness = safe_load_json(ROBUST_JSON)

    # 3) collect explanations
    explanations = collect_explanations(sample_n=5)

    # 4) bundle & save JSON
    bundle = {
        "metrics": metrics,
        "performance": performance,
        "robustness": robustness,
        "explanations": explanations
    }
    FINAL_JSON.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"[report] Wrote final JSON → {FINAL_JSON}")

    # 5) produce markdown
    md = generate_markdown(bundle)
    FINAL_MD.write_text(md, encoding="utf-8")
    print(f"[report] Wrote final markdown → {FINAL_MD}")

    print("[report] DONE. check project/evaluation/ for results.")

if __name__ == "__main__":
    main()