"""V2 - DOI-group-safe split.

V1'den farkı:
- Rastgele train_test_split yerine GroupShuffleSplit kullanılır.
- Grup kolonu Ref_DOI_number'dır.
- Aynı DOI'ye ait cihazların eğitim ve testte karışması engellenir.
- Leakage riski henüz tamamen temizlenmemiştir; amaç split revizyonunun etkisini görmek.
"""
import pandas as pd
from common import parse_args, build_dataset, group_split, fit_evaluate_models, ensure_out, save_json

args = parse_args("V2 - DOI-group-safe ML models")
out = ensure_out(args.out)

X, y, groups, feature_cols = build_dataset(
    args.data, mode="risky", max_groups=args.max_groups, random_state=args.random_state
)
X_train, X_test, y_train, y_test, g_train, g_test = group_split(
    X, y, groups, test_size=args.test_size, random_state=args.random_state
)

results, fitted, predictions = fit_evaluate_models(
    X_train, X_test, y_train, y_test,
    n_estimators=args.n_estimators,
    random_state=args.random_state,
)

results.to_csv(out / "v2_group_safe_model_comparison.csv", index=False)
pd.DataFrame({"feature": feature_cols}).to_csv(out / "v2_feature_list.csv", index=False)
save_json(out / "v2_metadata.json", {
    "version": "V2",
    "purpose": "DOI-group-safe split; leakage risk may still exist.",
    "split": "GroupShuffleSplit",
    "group_col": "Ref_DOI_number",
    "mode": "risky",
    "rows": int(len(X)),
    "groups": int(groups.nunique()),
    "features": int(len(feature_cols)),
    "train_groups": int(g_train.nunique()),
    "test_groups": int(g_test.nunique()),
    "test_size": args.test_size,
    "random_state": args.random_state,
})
print(results.to_string(index=False))
