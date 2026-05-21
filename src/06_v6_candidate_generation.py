"""V6 - Controlled candidate composition generation.

This script uses the best model saved by V4 and creates a small controlled
candidate space. The predictions are model-based prioritization outputs,
not experimentally validated efficiencies.

Revision V6.2:
- Keeps the V6.1 behavior: uses --model path directly; no retraining in V6.
- Adds candidate_prediction_summary.csv for thesis/reporting.
- Adds prediction_value_counts.csv to document repeated identical predictions.
- Adds top_candidates_diverse.csv to reduce repetitive top-candidate tables.
- Adds diversity controls for repeated rounded prediction values and duplicate
  candidate compositions.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from common import build_dataset, ensure_out, save_json


def parse_v6_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6 - Candidate generation using a saved V4 model")
    parser.add_argument("--data", default="Perovskite_database_content_all_data.csv", help="CSV file path")
    parser.add_argument("--model", required=True, help="Path to saved best_model.joblib from V4")
    parser.add_argument("--out", default="outputs/candidates", help="Output directory")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=None, help="Use same value as V4 quick run, if applicable")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--diverse-top-n", type=int, default=30)
    parser.add_argument(
        "--max-per-rounded-prediction",
        type=int,
        default=5,
        help="Maximum number of rows with the same rounded prediction in the diverse table",
    )
    return parser.parse_args()


def get_expected_raw_features(model_bundle, fallback_columns: List[str]) -> List[str]:
    """Return raw input feature names expected by the saved preprocessing pipeline."""
    preprocess = getattr(model_bundle, "preprocess", None)
    if preprocess is None and hasattr(model_bundle, "named_steps"):
        preprocess = model_bundle.named_steps.get("preprocess")

    if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
        return [str(c) for c in preprocess.feature_names_in_]

    return list(fallback_columns)


def make_base_defaults(X: pd.DataFrame) -> Dict[str, object]:
    """Create one candidate row template using medians/modes from the modelling data."""
    defaults: Dict[str, object] = {}
    for col in X.columns:
        if col.startswith(("A_", "B_", "X_")):
            defaults[col] = 0.0
        elif pd.api.types.is_numeric_dtype(X[col]):
            med = X[col].median()
            defaults[col] = float(med) if pd.notna(med) else 0.0
        else:
            mode = X[col].dropna().mode()
            defaults[col] = mode.iloc[0] if len(mode) else "Unknown"
    return defaults


def generate_candidates(raw_feature_columns: List[str], base_defaults: Dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a controlled and chemically simple candidate design space."""
    a_options = [
        {"A_FA": 0.80, "A_MA": 0.10, "A_Cs": 0.10, "A_Rb": 0.0, "A_K": 0.0, "A_other": 0.0},
        {"A_FA": 0.85, "A_MA": 0.00, "A_Cs": 0.15, "A_Rb": 0.0, "A_K": 0.0, "A_other": 0.0},
        {"A_FA": 0.75, "A_MA": 0.15, "A_Cs": 0.10, "A_Rb": 0.0, "A_K": 0.0, "A_other": 0.0},
    ]
    b_options = [
        {"B_Pb": 1.00, "B_Sn": 0.00, "B_Ge": 0.0, "B_other": 0.0},
        {"B_Pb": 0.95, "B_Sn": 0.05, "B_Ge": 0.0, "B_other": 0.0},
        {"B_Pb": 0.90, "B_Sn": 0.10, "B_Ge": 0.0, "B_other": 0.0},
    ]
    x_options = [
        {"X_I": 0.90, "X_Br": 0.10, "X_Cl": 0.0, "X_other": 0.0},
        {"X_I": 0.85, "X_Br": 0.15, "X_Cl": 0.0, "X_other": 0.0},
        {"X_I": 0.80, "X_Br": 0.20, "X_Cl": 0.0, "X_other": 0.0},
    ]
    architectures = ["nip", "pin"]
    etls = ["SnO2-np", "TiO2-c", "PCBM-60"]
    htls = ["Spiro-MeOTAD", "PTAA", "P3HT"]

    rows: List[Dict[str, object]] = []
    meta: List[Dict[str, object]] = []

    for a, b, x, arch, etl, htl in itertools.product(a_options, b_options, x_options, architectures, etls, htls):
        row = dict(base_defaults)
        row.update({k: v for k, v in a.items() if k in raw_feature_columns})
        row.update({k: v for k, v in b.items() if k in raw_feature_columns})
        row.update({k: v for k, v in x.items() if k in raw_feature_columns})

        if "Cell_architecture" in raw_feature_columns:
            row["Cell_architecture"] = arch
        if "ETL_stack_sequence" in raw_feature_columns:
            row["ETL_stack_sequence"] = etl
        if "HTL_stack_sequence" in raw_feature_columns:
            row["HTL_stack_sequence"] = htl
        if "Perovskite_band_gap" in raw_feature_columns:
            row["Perovskite_band_gap"] = 1.60

        rows.append({col: row.get(col, None) for col in raw_feature_columns})
        formula = (
            f"FA{a['A_FA']:.2f}MA{a['A_MA']:.2f}Cs{a['A_Cs']:.2f}-"
            f"Pb{b['B_Pb']:.2f}Sn{b['B_Sn']:.2f}-"
            f"I{x['X_I'] * 3:.2f}Br{x['X_Br'] * 3:.2f}"
        )
        meta.append({
            "Candidate_composition": formula,
            "Architecture": arch,
            "ETL": etl,
            "HTL": htl,
            "A_FA": a["A_FA"],
            "A_MA": a["A_MA"],
            "A_Cs": a["A_Cs"],
            "B_Pb": b["B_Pb"],
            "B_Sn": b["B_Sn"],
            "X_I": x["X_I"],
            "X_Br": x["X_Br"],
        })

    return pd.DataFrame(rows, columns=raw_feature_columns), pd.DataFrame(meta)


def describe_model(model_bundle) -> str:
    model = getattr(model_bundle, "model", model_bundle)
    return model.__class__.__name__


def select_diverse_candidates(
    result: pd.DataFrame,
    top_n: int,
    max_per_rounded_prediction: int = 5,
) -> pd.DataFrame:
    """Select a less repetitive candidate table for the thesis.

    XGBoost/LightGBM/CatBoost are tree-based models and can assign identical
    predicted values to many similar candidate feature vectors. This helper keeps
    the ranking but limits over-representation of identical rounded predictions
    and avoids duplicate composition-only rows when possible.
    """
    if result.empty:
        return result.copy()

    df = result.copy().sort_values("Predicted_PCE", ascending=False).reset_index(drop=True)
    df["Predicted_PCE_rounded6"] = df["Predicted_PCE"].round(6)

    selected_rows = []
    seen_compositions: set[str] = set()
    prediction_counts: Dict[float, int] = {}

    # First pass: prefer unique candidate compositions and limit identical rounded predictions.
    for _, row in df.iterrows():
        rounded_pred = float(row["Predicted_PCE_rounded6"])
        composition = str(row["Candidate_composition"])

        if composition in seen_compositions:
            continue
        if prediction_counts.get(rounded_pred, 0) >= max_per_rounded_prediction:
            continue

        selected_rows.append(row)
        seen_compositions.add(composition)
        prediction_counts[rounded_pred] = prediction_counts.get(rounded_pred, 0) + 1

        if len(selected_rows) >= top_n:
            break

    # Second pass: if the first pass is too strict, fill the remaining rows by rank.
    if len(selected_rows) < top_n:
        used_indices = {int(r.name) for r in selected_rows}
        for idx, row in df.iterrows():
            if int(idx) in used_indices:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= top_n:
                break

    diverse = pd.DataFrame(selected_rows).reset_index(drop=True)
    return diverse.drop(columns=["Predicted_PCE_rounded6"], errors="ignore")


def make_summary(result: pd.DataFrame, diverse: pd.DataFrame, model_name: str, model_path: Path) -> pd.DataFrame:
    """Create key-value summary rows for reporting and thesis notes."""
    rounded = result["Predicted_PCE"].round(6)
    max_pred = float(result["Predicted_PCE"].max()) if len(result) else float("nan")
    count_at_max = int((rounded == round(max_pred, 6)).sum()) if len(result) else 0

    summary = [
        ("loaded_model_class", model_name),
        ("loaded_model_path", str(model_path)),
        ("candidate_count", int(len(result))),
        ("diverse_candidate_count", int(len(diverse))),
        ("unique_prediction_count_rounded_6", int(rounded.nunique()) if len(result) else 0),
        ("max_predicted_pce", max_pred),
        ("candidate_count_at_max_predicted_pce", count_at_max),
        ("mean_predicted_pce", float(result["Predicted_PCE"].mean()) if len(result) else float("nan")),
        ("median_predicted_pce", float(result["Predicted_PCE"].median()) if len(result) else float("nan")),
        ("min_predicted_pce", float(result["Predicted_PCE"].min()) if len(result) else float("nan")),
        ("top_architecture", result["Architecture"].mode().iloc[0] if len(result) and len(result["Architecture"].mode()) else ""),
        ("top_etl", result["ETL"].mode().iloc[0] if len(result) and len(result["ETL"].mode()) else ""),
        ("top_htl", result["HTL"].mode().iloc[0] if len(result) and len(result["HTL"].mode()) else ""),
        ("note", "Predicted PCE values are model-based prioritization outputs, not experimental validation."),
    ]
    return pd.DataFrame(summary, columns=["metric", "value"])


def main() -> None:
    args = parse_v6_args()
    out = ensure_out(args.out)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model file not found: {model_path}")

    model_bundle = joblib.load(model_path)

    # Build the same strict feature frame only to obtain defaults and feature columns.
    # The saved V4 model itself is not retrained here.
    X, y, groups, feature_cols = build_dataset(
        args.data,
        mode="strict",
        max_groups=args.max_groups,
        random_state=args.random_state,
    )
    raw_feature_columns = get_expected_raw_features(model_bundle, list(X.columns))
    missing_from_current_data = [c for c in raw_feature_columns if c not in X.columns]
    if missing_from_current_data:
        raise ValueError(
            "Saved model expects features that are not present in the rebuilt dataset: "
            + ", ".join(missing_from_current_data[:20])
        )

    X_for_defaults = X[raw_feature_columns].copy()
    base_defaults = make_base_defaults(X_for_defaults)
    candidate_X, candidate_meta = generate_candidates(raw_feature_columns, base_defaults)

    preds = model_bundle.predict(candidate_X)
    result = candidate_meta.copy()
    result["Predicted_PCE"] = preds
    result = result.sort_values("Predicted_PCE", ascending=False).reset_index(drop=True)

    diverse = select_diverse_candidates(
        result,
        top_n=args.diverse_top_n,
        max_per_rounded_prediction=args.max_per_rounded_prediction,
    )

    model_name = describe_model(model_bundle)
    summary = make_summary(result, diverse, model_name, model_path)
    prediction_counts = (
        result.assign(Predicted_PCE_rounded6=result["Predicted_PCE"].round(6))
        .groupby("Predicted_PCE_rounded6", as_index=False)
        .size()
        .rename(columns={"size": "candidate_count"})
        .sort_values(["Predicted_PCE_rounded6", "candidate_count"], ascending=[False, False])
    )

    result.to_csv(out / "candidate_predictions.csv", index=False)
    result.head(args.top_n).to_csv(out / f"top{args.top_n}_candidate_predictions.csv", index=False)
    diverse.to_csv(out / "top_candidates_diverse.csv", index=False)
    candidate_X.to_csv(out / "candidate_feature_matrix.csv", index=False)
    summary.to_csv(out / "candidate_prediction_summary.csv", index=False)
    prediction_counts.to_csv(out / "prediction_value_counts.csv", index=False)

    max_pred = float(result["Predicted_PCE"].max()) if len(result) else None
    count_at_max = int((result["Predicted_PCE"].round(6) == round(max_pred, 6)).sum()) if max_pred is not None else 0
    save_json(out / "metadata.json", {
        "version": "V6.2",
        "purpose": "Controlled candidate generation using the saved best V4 model; no retraining in V6.",
        "loaded_model_path": str(model_path),
        "loaded_model_class": model_name,
        "candidate_count": int(len(result)),
        "top_n": int(args.top_n),
        "diverse_top_n": int(args.diverse_top_n),
        "unique_prediction_count_rounded_6": int(result["Predicted_PCE"].round(6).nunique()) if len(result) else 0,
        "max_predicted_pce": max_pred,
        "candidate_count_at_max_predicted_pce": count_at_max,
        "max_groups": args.max_groups,
        "warning": "Predicted PCE values are model-based prioritization outputs, not experimental validation.",
    })

    print(f"Loaded model for candidate prediction: {model_name}")
    print(f"Candidate outputs written to: {out}")
    print("\nCandidate summary:")
    print(summary.to_string(index=False))
    print("\nTop diverse candidates:")
    print(diverse.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
