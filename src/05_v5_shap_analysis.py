"""V5 - SHAP analysis for the selected leakage-safe model.

This script can either:
1) load the best model produced by V4 via --model outputs/v4/best_model.joblib, or
2) retrain the V4 protocol and select the best holdout model if --model is not given.

It rebuilds the same strict feature matrix, applies the saved preprocessing pipeline,
and generates SHAP feature-importance outputs for a manageable sample of test rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from common import (
    build_dataset,
    group_split,
    fit_evaluate_models,
    ensure_out,
    save_json,
)


def parse_v5_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5 - SHAP analysis")
    parser.add_argument("--data", default="Perovskite_database_content_all_data.csv", help="CSV file path")
    parser.add_argument("--model", default=None, help="Optional path to a saved V4 best_model.joblib")
    parser.add_argument("--out", default="outputs/shap", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=None, help="Use the same value as V4 quick run if applicable")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--shap-sample", type=int, default=1000, help="Maximum number of test rows used for SHAP")
    return parser.parse_args()


def to_dense(matrix):
    """Convert scipy sparse matrices to dense arrays for SHAP compatibility."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def normalize_shap_values(shap_values, n_features: int) -> np.ndarray:
    """Return a 2D SHAP array with shape (n_samples, n_features).

    Different tree libraries may return SHAP values as:
    - ndarray: (n_samples, n_features)
    - ndarray: (n_samples, n_features, 1)
    - list[ndarray]
    - shap.Explanation
    This function normalizes those cases for downstream CSV/plot creation.
    """
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    arr = np.asarray(shap_values)

    if arr.ndim == 3:
        # Regression models sometimes return (n_samples, n_features, 1).
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        # Some explainers may include an output dimension first.
        elif arr.shape[0] == 1:
            arr = arr[0, :, :]
        else:
            # For multi-output cases, use the first output; this should not normally happen here.
            arr = arr[:, :, 0]

    if arr.ndim != 2:
        raise ValueError(f"SHAP values must be 2D after normalization, got shape {arr.shape}")

    # If the array is transposed, fix it.
    if arr.shape[0] == n_features and arr.shape[1] != n_features:
        arr = arr.T

    if arr.shape[1] != n_features:
        raise ValueError(
            f"Feature count mismatch: SHAP has {arr.shape[1]} columns, "
            f"but preprocessing produced {n_features} feature names."
        )

    return arr


args = parse_v5_args()
out = ensure_out(args.out)

X, y, groups, feature_cols = build_dataset(
    args.data,
    mode="strict",
    max_groups=args.max_groups,
    random_state=args.random_state,
)
X_train, X_test, y_train, y_test, g_train, g_test = group_split(
    X,
    y,
    groups,
    test_size=args.test_size,
    random_state=args.random_state,
)

if args.model:
    bundle = joblib.load(args.model)
    best_name = type(bundle.named_steps["model"]).__name__
    best_pipe = bundle
    results = pd.DataFrame([{"Model": best_name, "Status": "loaded_from_joblib"}])
else:
    results, fitted, predictions = fit_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    best_name = results.iloc[0]["Model"]
    best_pipe = fitted[best_name]

pre = best_pipe.named_steps["preprocess"]
model = best_pipe.named_steps["model"]

# Limit SHAP sample to keep runtime manageable.
X_sample = X_test.sample(n=min(args.shap_sample, len(X_test)), random_state=args.random_state)
X_trans = pre.transform(X_sample)
feature_names = np.asarray(pre.get_feature_names_out(), dtype=str)
X_for_shap = to_dense(X_trans)

# TreeExplainer works for RandomForest, XGBoost, LightGBM and CatBoost tree models.
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_for_shap)
shap_values = normalize_shap_values(shap_values_raw, n_features=len(feature_names))

abs_mean = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs_mean.reshape(-1),
})
shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_df.to_csv(out / "shap_top_features.csv", index=False)

# Bar plot of top 20 features.
top = shap_df.head(20).iloc[::-1]
plt.figure(figsize=(8, 8))
plt.barh(top["feature"], top["mean_abs_shap"])
plt.xlabel("Mean |SHAP value|")
plt.title(f"Top SHAP Features - {best_name}")
plt.tight_layout()
plt.savefig(out / "shap_top20_bar.png", dpi=300)
plt.close()

# Summary plot can be heavy; use sample only.
shap.summary_plot(
    shap_values,
    X_for_shap,
    feature_names=feature_names,
    show=False,
    max_display=20,
)
plt.tight_layout()
plt.savefig(out / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.close()

results.to_csv(out / "v5_model_selection_holdout.csv", index=False)
joblib.dump(best_pipe, out / "best_model_for_shap.joblib")
save_json(out / "metadata.json", {
    "version": "V5",
    "purpose": "SHAP analysis for the selected DOI-group-safe leakage-safe model.",
    "loaded_model_path": str(args.model) if args.model else None,
    "best_model": str(best_name),
    "rows": int(len(X)),
    "features_before_encoding": int(len(feature_cols)),
    "features_after_encoding": int(len(feature_names)),
    "shap_sample_rows": int(len(X_sample)),
    "random_state": args.random_state,
})

print(f"Best/loaded model for SHAP: {best_name}")
print(shap_df.head(20).to_string(index=False))
print(f"\nSHAP outputs written to: {out}")
