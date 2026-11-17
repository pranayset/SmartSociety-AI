# Final Pre-Training Dataset Validation

This validation is a CPU-only check. It does not load Gemma, use a GPU, train a model, modify source datasets, or rewrite any split files.

## Inputs

Default Google Drive paths:

- Source datasets: `/content/drive/MyDrive/SocietySmart_AI/Datasets/Extracted`
- Operational splits: `/content/drive/MyDrive/SocietySmart_AI/Datasets/splits_v2`
- Output report: `/content/drive/MyDrive/SocietySmart_AI/outputs/pretraining_validation_report.json`

Expected CSV split naming is `{dataset}_{split}.csv` (for example, `visitor_train.csv`, `visitor_validation.csv`, `visitor_test.csv`). The validator also accepts `{split}_{dataset}.csv`. Chatbot JSONL uses `chatbot_train.jsonl`, `chatbot_validation.jsonl`, and `chatbot_unseen_test.jsonl` (also accepts `chatbot_test.jsonl` and split-first names). The unseen test file is the grouped-intent holdout and must remain separate and untouched.

## Run in Google Colab

1. Mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

2. Get the repository script. If the repository is already present in `/content/SmartSociety-AI`, pull the latest commit:

```bash
%cd /content/SmartSociety-AI
!git pull origin main
```

If the folder does not exist, clone the repository instead:

```bash
%cd /content
!git clone https://github.com/pranayset/SmartSociety-AI.git
%cd /content/SmartSociety-AI
```

3. Install/check the CPU dependencies:

```bash
!python -m pip install -q pandas
```

4. Run the validation:

```bash
!python scripts/final_pretraining_validation.py
```

5. Inspect the complete report:

```python
import json
from pathlib import Path

report_path = Path(
    "/content/drive/MyDrive/SocietySmart_AI/outputs/pretraining_validation_report.json"
)
report = json.loads(report_path.read_text(encoding="utf-8"))

print("Status:", report["quality_status"])
print("Checks:", json.dumps(report["overall_checks"], indent=2))
print("Errors:", json.dumps(report["errors"], indent=2))
print("Warnings:", json.dumps(report["warnings"], indent=2))
```

## Publish the generated report

Only publish after reviewing the status and errors. Copy the report into the repository:

```python
from pathlib import Path
import shutil

source = Path(
    "/content/drive/MyDrive/SocietySmart_AI/outputs/pretraining_validation_report.json"
)
destination = Path("/content/SmartSociety-AI/reports/pretraining_validation_report.json")
destination.parent.mkdir(parents=True, exist_ok=True)

assert source.is_file(), f"Report not found: {source}"
shutil.copy2(source, destination)
print("Copied:", destination, "Bytes:", destination.stat().st_size)
```

Then commit and push:

```bash
%cd /content/SmartSociety-AI
!git status --short
!git add scripts/final_pretraining_validation.py docs/pretraining_validation.md reports/pretraining_validation_report.json
!git commit -m "Add final pre-training dataset validation results"
!git push origin main
```

If the script was already committed through GitHub, Git will report that there is nothing new to commit for that script; that is fine.

## Status interpretation

- `PASS_WITH_WARNINGS`: all programmed structural, split, leakage, schema, and domain checks passed; warnings still need review.
- `REVIEW_REQUIRED`: one or more checks failed or could not be completed. Do not start GPU training until the errors are resolved and the validator is rerun.

## Scope and limitations

The validator checks expected row counts and schemas; missing values; exact duplicate rows; exact row overlap across splits; selected numerical/domain rules; strict timestamp ordering for parking, water, and electricity; chatbot JSONL turn structure; category validity; expected chatbot split sizes; and canonicalized user-intent overlap.

A zero exact-row overlap does not prove there is no semantic leakage. The chatbot canonicalizer is intentionally simple and cannot identify every paraphrase. The validation does not test model performance, label correctness, fairness, calibration, or deployment safety. The synthetic timestamp ordering is a split protocol only; it does not establish real-world forecasting validity. The report should be read together with the existing `reports/dataset_quality_report.json`.

All datasets are synthetic and for educational experimentation only. Do not use these records or resulting models to make real resident security, access, billing, water, electricity, or maintenance decisions.
