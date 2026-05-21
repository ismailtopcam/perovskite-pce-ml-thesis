"""V4 - Nihai model karşılaştırma raporu.

Bu sürüm tezde kullanılacak ana karşılaştırma hattıdır:
- DOI-group-safe holdout
- Leakage-safe feature set
- RandomForest, XGBoost, LightGBM, CatBoost
- Holdout MAE/RMSE/R2
- GroupKFold CV MAE/RMSE/R2
- En iyi model için gerçek-tahmin grafiği
"""
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from common import (
    parse_args, build_dataset, group_split, fit_evaluate_models,
    cross_validate_grouped, ensure_out, save_json, LEAKAGE_PREFIXES
)

args = parse_args("V4 - Final DOI-group-safe leakage-safe model comparison")
out = ensure_out(args.out)

X, y, groups, feature_cols = build_dataset(
    args.data, mode="strict", max_groups=args.max_groups, random_state=args.random_state
)
X_train, X_test, y_train, y_test, g_train, g_test = group_split(
    X, y, groups, test_size=args.test_size, random_state=args.random_state
)

holdout_results, fitted, predictions = fit_evaluate_models(
    X_train, X_test, y_train, y_test,
    n_estimators=args.n_estimators,
    random_state=args.random_state,
)
cv_results = cross_validate_grouped(
    X, y, groups,
    cv=args.cv,
    n_estimators=args.n_estimators,
    random_state=args.random_state,
)

holdout_results.to_csv(out / "model_comparison_holdout.csv", index=False)
cv_results.to_csv(out / "model_comparison_groupkfold.csv", index=False)
pd.DataFrame({"feature": feature_cols}).to_csv(out / "feature_list.csv", index=False)

best_name = holdout_results.iloc[0]["Model"]
best_model = fitted[best_name]
best_pred = predictions[best_name]
joblib.dump(best_model, out / "best_model.joblib")

pred_df = pd.DataFrame({
    "Actual_PCE": y_test.values,
    "Predicted_PCE": best_pred,
    "Residual": y_test.values - best_pred,
})
pred_df.to_csv(out / "best_model_predictions.csv", index=False)

plt.figure(figsize=(7, 6))
plt.scatter(pred_df["Actual_PCE"], pred_df["Predicted_PCE"], alpha=0.35)
lo = min(pred_df["Actual_PCE"].min(), pred_df["Predicted_PCE"].min())
hi = max(pred_df["Actual_PCE"].max(), pred_df["Predicted_PCE"].max())
plt.plot([lo, hi], [lo, hi], linestyle="--")
plt.xlabel("Actual PCE")
plt.ylabel("Predicted PCE")
plt.title(f"Actual vs Predicted PCE - {best_name}")
plt.tight_layout()
plt.savefig(out / "actual_vs_predicted_best_model.png", dpi=300)
plt.close()

save_json(out / "metadata.json", {
    "version": "V4",
    "purpose": "Final model comparison with DOI-group-safe and leakage-safe protocol.",
    "split": "GroupShuffleSplit holdout + GroupKFold CV",
    "group_col": "Ref_DOI_number",
    "mode": "strict",
    "excluded_leakage_prefixes": list(LEAKAGE_PREFIXES),
    "rows": int(len(X)),
    "groups": int(groups.nunique()),
    "features": int(len(feature_cols)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_groups": int(g_train.nunique()),
    "test_groups": int(g_test.nunique()),
    "best_model": str(best_name),
    "n_estimators": args.n_estimators,
    "cv": args.cv,
    "random_state": args.random_state,
})

print("HOLDOUT RESULTS")
print(holdout_results.to_string(index=False))
print("\nGROUPKFOLD CV RESULTS")
print(cv_results.to_string(index=False))
print(f"\nBest model: {best_name}")
