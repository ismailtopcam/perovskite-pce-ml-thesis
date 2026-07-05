"""enumerate_candidates cikti sozlesmesi: satir sayisi, sutun sirasi, kodlama dogrulugu."""
import numpy as np

from perovskite_ml.candidates.candidate_space import (
    A_LEVELS, ARCH_LEVELS, ETL_LEVELS, HTL_LEVELS, SOLV_LEVEL, axis_sizes,
    enumerate_candidates,
)

FEATURES = [
    "A_FA", "A_MA", "A_Cs", "B_Pb", "B_Sn", "X_I", "X_Br",
    "band_gap", "anneal_temp", "anneal_time",
    *ARCH_LEVELS, *ETL_LEVELS, *HTL_LEVELS, SOLV_LEVEL,
]


def test_output_shape_and_column_order():
    recipes, enc = enumerate_candidates(FEATURES, 1.6, 100.0, 20.0)
    n = int(np.prod(list(axis_sizes().values())))
    assert len(recipes) == len(enc) == n == 756
    # sutun sirasi model semasiyla birebir ayni olmali (tahmin dogrulugu buna bagli)
    assert list(enc.columns) == FEATURES


def test_composition_sums_and_onehot_exclusivity():
    _, enc = enumerate_candidates(FEATURES, 1.6, 100.0, 20.0)
    a_sum = enc[["A_FA", "A_MA", "A_Cs"]].sum(axis=1)
    x_sum = enc[["X_I", "X_Br"]].sum(axis=1)
    assert np.allclose(a_sum, 1.0) and np.allclose(x_sum, 1.0)
    # her adayda tam bir mimari, bir ETL, bir HTL secili; cozucu sabit
    assert (enc[ARCH_LEVELS].sum(axis=1) == 1.0).all()
    assert (enc[ETL_LEVELS].sum(axis=1) == 1.0).all()
    assert (enc[HTL_LEVELS].sum(axis=1) == 1.0).all()
    assert (enc[SOLV_LEVEL] == 1.0).all()


def test_fixed_numeric_context():
    _, enc = enumerate_candidates(FEATURES, 1.6, 100.0, 20.0)
    assert (enc["band_gap"] == 1.6).all()
    assert (enc["anneal_temp"] == 100.0).all()
    assert (enc["anneal_time"] == 20.0).all()


def test_missing_feature_columns_are_ignored_gracefully():
    # Modelde olmayan sutunlar (orn. arch_pin cikarildi) sessizce atlanir;
    # kalan sema yine tam 756 satir uretir.
    feats = [f for f in FEATURES if f != "arch_pin"]
    _, enc = enumerate_candidates(feats, 1.6, 100.0, 20.0)
    assert len(enc) == 756 and "arch_pin" not in enc.columns
