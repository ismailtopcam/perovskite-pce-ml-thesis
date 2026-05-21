"""V3 - DOI-group-safe + leakage-safe model.

V2'den farkı:
- JV_*, EQE_*, Stability_*, Stabilised_performance_*, Outdoor_* ve Module_JV_*
  gibi ölçüm sonrası bilgi taşıyabilecek kolonlar model girdisi yapılmaz.
- Bu nedenle skor düşebilir; amaç daha güvenilir genelleme ölçmektir.
"""
import pandas as pd
from common import parse_args, build_dataset, group_split, fit_evaluate_models, ensure_out, save_json, LEAKAGE_PREFIXES

args = parse_args("V3 - DOI-group-safe and leakage-safe ML models")
out = ensure_out(args.out)

X, y, groups, feature_cols = build_dataset(
    args.data, mode="strict", max_groups=args.max_groups, random_state=args.random_state
)
X_train, X_test, y_train, y_test, g_train, g_test = group_split(
    X, y, groups, test_size=args.test_size, random_state=args.random_state
)

results, fitted, predictions = fit_evaluate_models(
    X_train, X_test, y_train, y_test,
    n_estimators=args.n_estimators,
    random_state=args.random_state,
)

results.to_csv(out / "v3_leakage_safe_model_comparison.csv", index=False)
pd.DataFrame({"feature": feature_cols}).to_csv(out / "v3_feature_list.csv", index=False)
save_json(out / "v3_metadata.json", {
    "version": "V3",
    "purpose": "DOI-group-safe and leakage-safe model.",
    "split": "GroupShuffleSplit",
    "group_col": "Ref_DOI_number",
    "mode": "strict",
    "excluded_leakage_prefixes": list(LEAKAGE_PREFIXES),
    "rows": int(len(X)),
    "groups": int(groups.nunique()),
    "features": int(len(feature_cols)),
    "train_groups": int(g_train.nunique()),
    "test_groups": int(g_test.nunique()),
    "test_size": args.test_size,
    "random_state": args.random_state,
})
print(results.to_string(index=False))
