"""Model kayit defteri. sklearn modelleri her zaman; XGBoost/LightGBM/CatBoost
kuruluysa eklenir, kurulu degilse zarafetle atlanir (ortamdan bagimsiz calisir)."""
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build_models(seed: int = 42) -> dict:
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge()),                  # lineer referans
        # KNN cikarildi: 41k satirda brute-force mesafe cok yavas. Ridge lineer referans yeterli.
        "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed),
        "HistGBM": HistGradientBoostingRegressor(random_state=seed),        # sklearn boosting (her zaman var)
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed)
    except Exception as e:
        print(f"[models] XGBoost atlandi: {e.__class__.__name__}")
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed, verbose=-1)
    except Exception as e:
        print(f"[models] LightGBM atlandi: {e.__class__.__name__}")
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
            random_seed=seed, verbose=0, allow_writing_files=False)
    except Exception as e:
        print(f"[models] CatBoost atlandi: {e.__class__.__name__}")
    return models
