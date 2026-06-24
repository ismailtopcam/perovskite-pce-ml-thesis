import pandas as pd, pytest
from perovskite_ml.validation.schema import (validate_model_ready, validate_clean, SchemaError)

CFG = {"target":"JV_default_PCE","group_col":"Ref_DOI_number",
       "cleaning":{"pce_min":0.0,"pce_max":35.0},
       "known_ions":{"A":["FA","MA"],"B":["Pb"],"X":["I"]},
       "features":{"onehot_top_n":15}}

def _ok_mr():
    return pd.DataFrame({"A_FA":[1.0,0.0],"A_MA":[0.0,1.0],"B_Pb":[1.0,1.0],
                         "X_I":[1.0,1.0],"band_gap":[1.6,1.5],
                         "JV_default_PCE":[15.0,12.0],"Ref_DOI_number":["d1","d2"]})

def test_model_ready_ok():
    rep = validate_model_ready(_ok_mr(), CFG)
    assert rep["n_rows"] == 2

def test_leakage_detected():
    df = _ok_mr(); df["JV_default_Voc"] = [1.1, 1.0]   # sizinti kolonu
    with pytest.raises(SchemaError):
        validate_model_ready(df, CFG)

def test_clean_rejects_out_of_range():
    df = pd.DataFrame({"JV_default_PCE":[15.0, 99.0],"Ref_DOI_number":["d1","d2"]})
    with pytest.raises(SchemaError):
        validate_clean(df, CFG)

def test_clean_rejects_nonnumeric_target():
    df = pd.DataFrame({"JV_default_PCE":[15.0,"abc"],"Ref_DOI_number":["d1","d2"]})
    with pytest.raises(SchemaError):
        validate_clean(df, CFG)
