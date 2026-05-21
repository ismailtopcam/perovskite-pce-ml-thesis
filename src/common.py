"""Common utilities for the perovskite PCE machine-learning workflow.

This module intentionally keeps the preprocessing transparent, because the
thesis must explain why each modelling revision was made.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "JV_default_PCE"
GROUP_COL = "Ref_DOI_number"

# Measurement/result columns that may carry post-measurement information.
# In the strict model these are not allowed as features.
LEAKAGE_PREFIXES = (
    "JV_",
    "EQE_",
    "Stability_",
    "Stabilised_performance_",
    "Outdoor_",
    "Module_JV_",
)
LEAKAGE_EXACT = {
    TARGET,
    "JV_reverse_scan_PCE",
    "JV_forward_scan_PCE",
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
}

SAFE_BASE_COLUMNS = [
    "Ref_ID",
    GROUP_COL,
    TARGET,
    "Cell_architecture",
    "Cell_flexible",
    "Cell_semitransparent",
    "ETL_stack_sequence",
    "HTL_stack_sequence",
    "Perovskite_band_gap",
    "Perovskite_thickness",
    "Perovskite_additives_compounds",
    "Perovskite_deposition_procedure",
    "Perovskite_deposition_synthesis_atmosphere",
    "Perovskite_deposition_solvents",
    "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_thermal_annealing_time",
    "Perovskite_composition_a_ions",
    "Perovskite_composition_a_ions_coefficients",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_b_ions_coefficients",
    "Perovskite_composition_c_ions",
    "Perovskite_composition_c_ions_coefficients",
]

# Intentionally risky features used only in V1/V2 to demonstrate why the code
# was revised. Do not use these in final strict experiments.
RISKY_RESULT_COLUMNS = [
    "JV_default_Voc",
    "JV_default_Jsc",
    "JV_default_FF",
    "JV_reverse_scan_PCE",
    "EQE_integrated_Jsc",
    "Stabilised_performance_PCE",
    "Stability_PCE_initial_value",
]

A_IONS = ["FA", "MA", "Cs", "Rb", "K"]
B_IONS = ["Pb", "Sn", "Ge"]
X_IONS = ["I", "Br", "Cl"]


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", default="Perovskite_database_content_all_data.csv", help="CSV file path")
    parser.add_argument("--out", default="outputs/run", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=None, help="Optional fast-test limit by DOI groups")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--model-output", default=None, help="Optional previous model-output directory; accepted for workflow compatibility")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    # Also allow running scripts from repository root when the data is one level up.
    candidate = Path.cwd() / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Data file not found: {path}")


def read_raw_csv(path: str | Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    path = resolve_path(path)
    if columns is None:
        return pd.read_csv(path, low_memory=False)
    header = pd.read_csv(path, nrows=0, low_memory=False)
    available = [c for c in columns if c in header.columns]
    return pd.read_csv(path, usecols=available, low_memory=False)


def split_semicolon(value) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "unknown"}:
        return []
    # Perovskite Database commonly uses semicolon-separated lists.
    return [x.strip() for x in re.split(r"\s*;\s*", text) if x.strip()]


def parse_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).replace(",", ".")
    # Extract first numeric token, e.g. "100 nm" -> 100.
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        val = float(match.group(0))
        return val if math.isfinite(val) else None
    except ValueError:
        return None


def parse_coefficients(value) -> List[float]:
    parts = split_semicolon(value)
    nums = []
    for part in parts:
        val = parse_float(part)
        if val is None:
            return []
        nums.append(val)
    return nums


def normalize(vals: List[float]) -> List[float]:
    total = sum(vals)
    if total == 0:
        return vals
    return [v / total for v in vals]


def vectorize_site(ions_value, coeff_value, known_ions: List[str], prefix: str) -> Dict[str, float]:
    result = {f"{prefix}_{ion}": 0.0 for ion in known_ions}
    result[f"{prefix}_other"] = 0.0

    ions = split_semicolon(ions_value)
    coeffs = parse_coefficients(coeff_value)
    if not ions or not coeffs or len(ions) != len(coeffs):
        # Return zeros; cleaning statistics can be handled separately.
        return result

    coeffs = normalize(coeffs)
    for ion, coeff in zip(ions, coeffs):
        ion = ion.strip()
        key = f"{prefix}_{ion}"
        if ion in known_ions:
            result[key] += coeff
        else:
            result[f"{prefix}_other"] += coeff
    return result


def add_composition_vectors(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        features = {}
        features.update(vectorize_site(
            row.get("Perovskite_composition_a_ions"),
            row.get("Perovskite_composition_a_ions_coefficients"),
            A_IONS,
            "A",
        ))
        features.update(vectorize_site(
            row.get("Perovskite_composition_b_ions"),
            row.get("Perovskite_composition_b_ions_coefficients"),
            B_IONS,
            "B",
        ))
        features.update(vectorize_site(
            row.get("Perovskite_composition_c_ions"),
            row.get("Perovskite_composition_c_ions_coefficients"),
            X_IONS,
            "X",
        ))
        rows.append(features)
    comp = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), comp.reset_index(drop=True)], axis=1)


def clean_target_and_groups(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].between(0, 35, inclusive="both")]
    if GROUP_COL in df.columns:
        df[GROUP_COL] = df[GROUP_COL].astype(object)
        missing_group = df[GROUP_COL].isna() | (df[GROUP_COL].astype(str).str.strip() == "")
        df.loc[missing_group, GROUP_COL] = [f"NO_DOI_{i}" for i in df.index[missing_group]]
    return df.reset_index(drop=True)


def sample_by_groups(df: pd.DataFrame, max_groups: Optional[int], random_state: int = 42) -> pd.DataFrame:
    if max_groups is None or GROUP_COL not in df.columns:
        return df
    rng = np.random.default_rng(random_state)
    groups = pd.Series(df[GROUP_COL].astype(str).unique())
    if len(groups) <= max_groups:
        return df
    chosen = set(rng.choice(groups, size=max_groups, replace=False))
    return df[df[GROUP_COL].astype(str).isin(chosen)].reset_index(drop=True)


def build_dataset(
    data_path: str | Path,
    mode: str = "strict",
    max_groups: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Build feature matrix.

    mode='risky': includes selected post-measurement columns for demonstration.
    mode='strict': excludes leakage-risk columns and uses design/procedure features.
    """
    requested = list(SAFE_BASE_COLUMNS)
    if mode == "risky":
        requested += RISKY_RESULT_COLUMNS
    df = read_raw_csv(data_path, requested)
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET} not found in data")

    df = clean_target_and_groups(df)
    df = sample_by_groups(df, max_groups, random_state)
    df = add_composition_vectors(df)

    numeric_cols = [
        "Perovskite_band_gap",
        "Perovskite_thickness",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
        "Cell_flexible",
        "Cell_semitransparent",
    ]
    if mode == "risky":
        numeric_cols += [c for c in RISKY_RESULT_COLUMNS if c in df.columns]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].map(parse_float)

    comp_cols = [c for c in df.columns if re.match(r"^[ABX]_", c)]
    numeric_cols = [c for c in numeric_cols if c in df.columns] + comp_cols

    cat_cols = [
        "Cell_architecture",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Perovskite_additives_compounds",
        "Perovskite_deposition_procedure",
        "Perovskite_deposition_synthesis_atmosphere",
        "Perovskite_deposition_solvents",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    feature_cols = numeric_cols + cat_cols
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    groups = df[GROUP_COL].astype(str).copy() if GROUP_COL in df.columns else pd.Series(df.index.astype(str))
    return X, y, groups, feature_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_models(n_estimators: int = 300, random_state: int = 42) -> Dict[str, object]:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=n_estimators,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def random_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def group_split(X, y, groups, test_size=0.2, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], groups.iloc[train_idx], groups.iloc[test_idx]



class FittedModelBundle:
    """Small wrapper to avoid sklearn Pipeline compatibility issues with CatBoost."""
    def __init__(self, preprocess, model):
        self.preprocess = preprocess
        self.model = model

    def predict(self, X):
        return self.model.predict(self.preprocess.transform(X))

    @property
    def named_steps(self):
        return {"preprocess": self.preprocess, "model": self.model}


def fit_evaluate_models(X_train, X_test, y_train, y_test, n_estimators=300, random_state=42):
    models = build_models(n_estimators=n_estimators, random_state=random_state)
    rows = []
    fitted = {}
    predictions = {}
    for name, model in models.items():
        try:
            preprocessor = make_preprocessor(X_train)
            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)
            model.fit(X_train_trans, y_train)
            y_pred = model.predict(X_test_trans)
            row = {"Model": name, **evaluate_regression(y_test, y_pred), "Status": "ok"}
            fitted[name] = FittedModelBundle(preprocessor, model)
            predictions[name] = y_pred
        except Exception as exc:
            row = {"Model": name, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Status": f"failed: {exc}"}
        rows.append(row)
    results = pd.DataFrame(rows).sort_values(["RMSE", "MAE"], na_position="last").reset_index(drop=True)
    return results, fitted, predictions

def cross_validate_grouped(X, y, groups, cv=5, n_estimators=300, random_state=42):
    model_names = list(build_models(n_estimators=n_estimators, random_state=random_state).keys())
    rows = []
    n_splits = min(cv, groups.nunique())
    if n_splits < 2:
        return pd.DataFrame(columns=["Model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std", "Status"])
    splitter = GroupKFold(n_splits=n_splits)
    for name in model_names:
        fold_rows = []
        failed = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            try:
                model = build_models(n_estimators=n_estimators, random_state=random_state)[name]
                preprocessor = make_preprocessor(X_train)
                X_train_trans = preprocessor.fit_transform(X_train)
                X_test_trans = preprocessor.transform(X_test)
                model.fit(X_train_trans, y_train)
                y_pred = model.predict(X_test_trans)
                metrics = evaluate_regression(y_test, y_pred)
                fold_rows.append({"Model": name, "Fold": fold, **metrics})
            except Exception as exc:
                failed.append(f"fold {fold}: {exc}")
        if fold_rows:
            fold_df = pd.DataFrame(fold_rows)
            rows.append({
                "Model": name,
                "MAE_mean": fold_df["MAE"].mean(),
                "MAE_std": fold_df["MAE"].std(),
                "RMSE_mean": fold_df["RMSE"].mean(),
                "RMSE_std": fold_df["RMSE"].std(),
                "R2_mean": fold_df["R2"].mean(),
                "R2_std": fold_df["R2"].std(),
                "Status": "ok" if not failed else "; ".join(failed),
            })
        else:
            rows.append({
                "Model": name,
                "MAE_mean": np.nan,
                "MAE_std": np.nan,
                "RMSE_mean": np.nan,
                "RMSE_std": np.nan,
                "R2_mean": np.nan,
                "R2_std": np.nan,
                "Status": "; ".join(failed) if failed else "failed",
            })
    return pd.DataFrame(rows).sort_values(["RMSE_mean", "MAE_mean"], na_position="last").reset_index(drop=True)

def save_json(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_out(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
