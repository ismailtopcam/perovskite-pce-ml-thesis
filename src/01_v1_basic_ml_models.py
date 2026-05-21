"""V1 - İlk çalışan model.

Amaç: Veri setinin ML ile öğrenilebilir olup olmadığını hızlıca görmek.
Bu sürüm kasıtlı olarak basittir:
- Rastgele train/test split kullanır.
- Bazı ölçüm sonrası kolonları içerebilir.
- Nihai tez sonucu olarak kullanılmamalıdır.
"""
from pathlib import Path
import pandas as pd
from common import parse_args, build_dataset, random_split, fit_evaluate_models, ensure_out, save_json

args = parse_args("V1 - Basic random-split ML models")
out = ensure_out(args.out)

X, y, groups, feature_cols = build_dataset(
    args.data, mode="risky", max_groups=args.max_groups, random_state=args.random_state
)
X_train, X_test, y_train, y_test = random_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)

results, fitted, predictions = fit_evaluate_models(
    X_train, X_test, y_train, y_test,
    n_estimators=args.n_estimators,
    random_state=args.random_state,
)

results.to_csv(out / "v1_basic_model_comparison.csv", index=False)
pd.DataFrame({"feature": feature_cols}).to_csv(out / "v1_feature_list.csv", index=False)
save_json(out / "v1_metadata.json", {
    "version": "V1",
    "purpose": "Initial runnable model; not DOI-group-safe and not leakage-safe.",
    "split": "train_test_split",
    "mode": "risky",
    "rows": int(len(X)),
    "features": int(len(feature_cols)),
    "test_size": args.test_size,
    "random_state": args.random_state,
})
print(results.to_string(index=False))
