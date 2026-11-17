#!/usr/bin/env python3
"""Final pre-training validation for SmartSociety-AI synthetic datasets.

Run in Colab after mounting Google Drive:
python /content/SmartSociety-AI/scripts/final_pretraining_validation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPECTED = {
    "visitor": {"total": 5000, "train": 3500, "validation": 750, "test": 750,
                "columns": ["visitor_id", "entry_time", "exit_time", "visitor_type", "entry_gate", "visit_purpose", "duration_hours", "unusual_activity"]},
    "parking": {"total": 5000, "train": 3500, "validation": 750, "test": 750,
                "columns": ["timestamp", "parking_zone", "total_spaces", "occupied_spaces", "available_spaces", "occupancy_rate", "hour", "weekend", "high_occupancy"]},
    "water": {"total": 3000, "train": 2100, "validation": 450, "test": 450,
              "columns": ["timestamp", "tank_id", "tank_capacity_litres", "tank_level_percent", "water_consumption_litres", "inflow_litres", "leak_anomaly", "low_tank_level"]},
    "electricity": {"total": 3000, "train": 2100, "validation": 450, "test": 450,
                    "columns": ["timestamp", "building_id", "hour", "temperature_c", "occupancy_estimate", "electricity_consumption_kwh", "peak_hour", "energy_anomaly"]},
    "maintenance": {"total": 2000, "train": 1400, "validation": 300, "test": 300,
                    "columns": ["equipment_id", "equipment_type", "operating_hours", "temperature_c", "vibration_mm_s", "pressure_bar", "fault_within_7_days"]},
    "complaints": {"total": 1000, "train": 700, "validation": 150, "test": 150,
                   "columns": ["complaint_id", "created_at", "category", "description", "priority", "resolution_hours", "status"]},
}
CHAT_EXPECTED = {"train": 210, "validation": 30, "test": 60}
CHAT_CATEGORIES = {"water", "lift", "electricity", "parking", "security", "complaints"}


def find_file(folder: Path, stem: str, split: str, suffix: str) -> Path:
    candidates = [
        folder / f"{stem}_{split}.{suffix}",
        folder / f"{split}_{stem}.{suffix}",
        folder / f"{stem}_{'val' if split == 'validation' else split}.{suffix}",
    ]
    matches = [p for p in candidates if p.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {stem}/{split} {suffix} file in {folder}; "
            f"checked {[p.name for p in candidates]}, found {[p.name for p in matches]}"
        )
    return matches[0]


def row_hashes(df: pd.DataFrame) -> set[str]:
    normalized = df.fillna("<NA>").astype(str)
    return set(pd.util.hash_pandas_object(normalized, index=False).astype(str))


def canonical_intent(question: str) -> str:
    q = str(question).strip().lower()
    q = re.sub(r"^(please help me|i need guidance|can you explain|for the society helpdesk)\s*:\s*", "", q)
    q = re.sub(r"[^a-z0-9 ]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def jsonl_records(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def class_balance(df: pd.DataFrame, label: str | None) -> dict:
    if not label or label not in df.columns:
        return {}
    counts = df[label].value_counts(dropna=False)
    total = len(df)
    return {str(k): {"count": int(v), "percentage": round(100 * int(v) / total, 3) if total else 0}
            for k, v in counts.items()}


def validate(data_dir: Path, split_dir: Path, output: Path) -> dict:
    errors, warnings, checks = [], [], {}
    report = {
        "project": "SmartSociety-AI",
        "report_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_classification": "Synthetic",
        "source_directory": str(data_dir),
        "split_directory": str(split_dir),
        "datasets": {},
        "chatbot": {},
    }

    label_columns = {
        "visitor": "unusual_activity", "parking": "high_occupancy",
        "water": "leak_anomaly", "electricity": "energy_anomaly",
        "maintenance": "fault_within_7_days", "complaints": "category",
    }
    original_frames = {}
    split_frames = {}

    for name, spec in EXPECTED.items():
        try:
            source_path = data_dir / f"{name}_dataset.csv"
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing source file: {source_path}")
            original = pd.read_csv(source_path)
            original_frames[name] = original
            missing_cols = sorted(set(spec["columns"]) - set(original.columns))
            extra_cols = sorted(set(original.columns) - set(spec["columns"]))
            source_ok = len(original) == spec["total"] and not missing_cols and not extra_cols
            if not source_ok:
                errors.append(f"{name}: source count/schema mismatch (rows={len(original)}, missing_columns={missing_cols}, extra_columns={extra_cols})")

            split_detail, frames = {}, {}
            for split in ("train", "validation", "test"):
                path = find_file(split_dir, name, split, "csv")
                df = pd.read_csv(path)
                frames[split] = df
                missing = int(df.isna().sum().sum())
                duplicates = int(df.duplicated().sum())
                schema_ok = list(df.columns) == list(original.columns)
                count_ok = len(df) == spec[split]
                if not count_ok:
                    errors.append(f"{name}/{split}: expected {spec[split]} rows, found {len(df)}")
                if not schema_ok:
                    errors.append(f"{name}/{split}: columns/order differ from source")
                if missing:
                    errors.append(f"{name}/{split}: {missing} missing values")
                if duplicates:
                    errors.append(f"{name}/{split}: {duplicates} exact duplicate rows")
                split_detail[split] = {
                    "file": path.name, "records": len(df), "columns": len(df.columns),
                    "missing_values": missing, "exact_duplicate_rows": duplicates,
                    "schema_matches_source": schema_ok, "record_count_matches_expected": count_ok,
                    "class_balance": class_balance(df, label_columns[name]),
                }

            pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
            overlap = {}
            for a, b in pairs:
                n = len(row_hashes(frames[a]) & row_hashes(frames[b]))
                overlap[f"{a}_{b}"] = n
                if n:
                    errors.append(f"{name}: {n} exact rows overlap between {a} and {b}")
            total_split = sum(len(x) for x in frames.values())
            if total_split != spec["total"]:
                errors.append(f"{name}: split total {total_split} does not equal source total {spec['total']}")

            time_report = {"applicable": "timestamp" in original.columns}
            if "timestamp" in original.columns:
                bounds = {}
                for split, df in frames.items():
                    parsed = pd.to_datetime(df["timestamp"], errors="coerce")
                    if parsed.isna().any():
                        errors.append(f"{name}/{split}: invalid timestamps")
                    bounds[split] = {
                        "minimum": parsed.min().isoformat() if parsed.notna().any() else None,
                        "maximum": parsed.max().isoformat() if parsed.notna().any() else None,
                        "valid_timestamps": int(parsed.notna().sum()),
                    }
                ordered = (
                    pd.to_datetime(frames["train"]["timestamp"], errors="coerce").max()
                    < pd.to_datetime(frames["validation"]["timestamp"], errors="coerce").min()
                    and pd.to_datetime(frames["validation"]["timestamp"], errors="coerce").max()
                    < pd.to_datetime(frames["test"]["timestamp"], errors="coerce").min()
                )
                time_report.update(bounds)
                time_report["chronological_order_valid"] = bool(ordered)
                if not ordered:
                    errors.append(f"{name}: timestamp splits are not strictly chronological")

            # Dataset-specific domain checks
            numeric = {}
            def record_rule(rule, condition):
                passed = bool(condition)
                numeric[rule] = {"passed": passed}
                if not passed:
                    errors.append(f"{name}: domain rule failed: {rule}")

            if name == "visitor":
                record_rule("duration_hours_non_negative", (original["duration_hours"] >= 0).all())
                entry = pd.to_datetime(original["entry_time"], errors="coerce")
                exit_ = pd.to_datetime(original["exit_time"], errors="coerce")
                record_rule("exit_time_not_before_entry_time", (exit_ >= entry).all() and entry.notna().all() and exit_.notna().all())
            elif name == "parking":
                record_rule("space_counts_non_negative", (original[["total_spaces", "occupied_spaces", "available_spaces"]] >= 0).all().all())
                record_rule("occupied_plus_available_equals_total", (original["occupied_spaces"] + original["available_spaces"] == original["total_spaces"]).all())
                record_rule("occupancy_rate_between_0_and_1", original["occupancy_rate"].between(0, 1).all())
            elif name == "water":
                record_rule("tank_level_between_0_and_100", original["tank_level_percent"].between(0, 100).all())
                record_rule("water_values_non_negative", (original[["tank_capacity_litres", "water_consumption_litres", "inflow_litres"]] >= 0).all().all())
            elif name == "electricity":
                record_rule("consumption_non_negative", (original["electricity_consumption_kwh"] >= 0).all())
                record_rule("occupancy_non_negative", (original["occupancy_estimate"] >= 0).all())
            elif name == "maintenance":
                record_rule("operating_hours_non_negative", (original["operating_hours"] >= 0).all())
                record_rule("vibration_and_pressure_non_negative", (original[["vibration_mm_s", "pressure_bar"]] >= 0).all().all())
            elif name == "complaints":
                record_rule("resolution_hours_non_negative", (original["resolution_hours"] >= 0).all())
                record_rule("category_non_empty", original["category"].notna().all() and original["category"].astype(str).str.strip().ne("").all())

            report["datasets"][name] = {
                "source_file": source_path.name,
                "expected_records": spec["total"], "actual_records": len(original),
                "columns": list(original.columns), "source_missing_values": int(original.isna().sum().sum()),
                "source_exact_duplicate_rows": int(original.duplicated().sum()),
                "splits": split_detail, "split_total_records": total_split,
                "split_count_matches_original": total_split == len(original) == spec["total"],
                "leakage_check": {"overlap_counts": overlap, "no_exact_row_overlap": all(v == 0 for v in overlap.values())},
                "time_order": time_report, "numerical_validation": numeric,
            }
            split_frames[name] = frames
            if original.isna().sum().sum() or original.duplicated().any():
                errors.append(f"{name}: source dataset contains missing or duplicate rows")
            if name in ("visitor", "water", "electricity") and label_columns[name]:
                positive = int((original[label_columns[name]].astype(str).isin(["1", "True", "true"])).sum())
                prevalence = positive / len(original) if len(original) else 0
                if prevalence < 0.10:
                    warnings.append(f"{name}: positive anomaly class is {prevalence:.2%}; report precision, recall, F1 and PR-AUC, not accuracy alone.")
        except Exception as exc:
            errors.append(f"{name}: validation could not complete: {type(exc).__name__}: {exc}")

    # Chatbot JSONL: grouped canonical intents, fixed split sizes
    try:
        chat = {}
        for split in ("train", "validation", "test"):
            path = find_file(split_dir, "chatbot", split, "jsonl")
            rows = jsonl_records(path)
            chat[split] = rows
            if len(rows) != CHAT_EXPECTED[split]:
                errors.append(f"chatbot/{split}: expected {CHAT_EXPECTED[split]} records, found {len(rows)}")
        intents, categories, structure_errors = {}, {}, []
        for split, rows in chat.items():
            intents[split] = set()
            categories[split] = {}
            for i, row in enumerate(rows):
                messages = row.get("messages")
                category = row.get("category")
                if not isinstance(messages, list) or len(messages) < 2:
                    structure_errors.append(f"{split}[{i}]: messages must contain user and assistant turns")
                    continue
                user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
                assistant_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
                if not user_msgs or not assistant_msgs or not str(user_msgs[0].get("content", "")).strip() or not str(assistant_msgs[-1].get("content", "")).strip():
                    structure_errors.append(f"{split}[{i}]: missing non-empty user or assistant content")
                    continue
                if category not in CHAT_CATEGORIES:
                    structure_errors.append(f"{split}[{i}]: invalid category {category!r}")
                categories[split][category] = categories[split].get(category, 0) + 1
                intents[split].add(canonical_intent(user_msgs[0]["content"]))
        overlaps = {
            "train_validation": len(intents["train"] & intents["validation"]),
            "train_test": len(intents["train"] & intents["test"]),
            "validation_test": len(intents["validation"] & intents["test"]),
        }
        if structure_errors:
            errors.extend(["chatbot structure: " + x for x in structure_errors[:20]])
        if any(overlaps.values()):
            errors.append(f"chatbot: canonical intent leakage detected: {overlaps}")
        if sum(len(v) for v in chat.values()) != 300:
            errors.append("chatbot: total record count is not 300")
        report["chatbot"] = {
            "split_files": {s: find_file(split_dir, "chatbot", s, "jsonl").name for s in chat},
            "total_records": sum(map(len, chat.values())),
            "split_sizes": {s: {"records": len(rows), "unique_intents": len(intents[s])} for s, rows in chat.items()},
            "category_counts_by_split": categories,
            "unique_canonical_intents": len(set.union(*intents.values())),
            "intent_overlap": overlaps,
            "record_structure_valid": not structure_errors,
            "no_intent_leakage": not any(overlaps.values()),
            "expected_split_sizes": CHAT_EXPECTED,
            "split_sizes_match_expected": all(len(chat[s]) == CHAT_EXPECTED[s] for s in CHAT_EXPECTED),
        }
    except Exception as exc:
        errors.append(f"chatbot: validation could not complete: {type(exc).__name__}: {exc}")

    # Known limitations are explicitly retained in the report.
    warnings.extend([
        "All records are synthetic and are suitable for educational experimentation only, not real society security, billing, water, electricity or maintenance decisions.",
        "Parking, water and electricity splits are chronological, but source timestamps were synthetically generated; this is not evidence of real-world forecasting performance.",
        "The chatbot corpus has 60 canonical intents with repeated paraphrases and category-level answers; evaluate generalization with a separate, untouched challenge set.",
    ])
    checks = {
        "all_six_source_datasets_validated": len(report["datasets"]) == 6,
        "all_expected_split_sizes_and_schemas_valid": not any("expected" in e or "schema" in e for e in errors),
        "no_missing_values_or_duplicate_rows": not any("missing" in e or "duplicate" in e for e in errors),
        "no_exact_row_overlap": not any("exact rows overlap" in e for e in errors),
        "domain_rules_passed": not any("domain rule failed" in e for e in errors),
        "chatbot_structure_and_intent_isolation_valid": report.get("chatbot", {}).get("record_structure_valid", False) and report.get("chatbot", {}).get("no_intent_leakage", False),
        "chronological_splits_valid": not any("not strictly chronological" in e for e in errors),
    }
    report["overall_checks"] = checks
    report["warnings"] = warnings
    report["errors"] = errors
    report["quality_status"] = "PASS_WITH_WARNINGS" if all(checks.values()) and not errors else "REVIEW_REQUIRED"
    report["publication_note"] = "Generated from synthetic datasets. PASS_WITH_WARNINGS means programmed structural checks passed; it does not establish model quality or real-world validity."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "quality_status": report["quality_status"],
        "overall_checks": checks,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "output": str(output),
    }, indent=2))
    if errors:
        print("\nERRORS:")
        for item in errors:
            print("-", item)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/content/drive/MyDrive/SocietySmart_AI/Datasets/Extracted")
    parser.add_argument("--split-dir", default="/content/drive/MyDrive/SocietySmart_AI/Datasets/splits_v2")
    parser.add_argument("--output", default="/content/drive/MyDrive/SocietySmart_AI/outputs/pretraining_validation_report.json")
    args = parser.parse_args()
    validate(Path(args.data_dir), Path(args.split_dir), Path(args.output))


if __name__ == "__main__":
    main()
