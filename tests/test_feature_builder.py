"""build_features ana akisinin birim testleri (sentetik veri; ham veri gerektirmez)."""
import numpy as np
import pandas as pd

from perovskite_ml.features.feature_builder import build_features


def make_cfg(top_n=2):
    return {
        "target": "JV_default_PCE",
        "group_col": "Ref_DOI_number",
        "features": {
            "onehot_top_n": top_n,
            "add_descriptors": False,
            "band_gap_col": "Perovskite_band_gap",
            "solvent_col": "Perovskite_deposition_solvents",
            "anneal_temp_col": "Perovskite_deposition_thermal_annealing_temperature",
            "anneal_time_col": "Perovskite_deposition_thermal_annealing_time",
            "arch_col": "Cell_architecture",
            "etl_col": "ETL_stack_sequence",
            "htl_col": "HTL_stack_sequence",
            "flexible_col": "Cell_flexible",
            "semitransparent_col": "Cell_semitransparent",
        },
    }


def make_df():
    n = 6
    return pd.DataFrame({
        "A_FA": [1.0, 0.8, 0.0, 1.0, 0.5, 1.0],
        "A_MA": [0.0, 0.2, 1.0, 0.0, 0.5, 0.0],
        "Perovskite_band_gap": [1.5, np.nan, 1.7, 1.6, np.nan, 1.6],
        "Cell_flexible": ["TRUE", "false", np.nan, "True", "FALSE", "true"],
        "Cell_semitransparent": ["false"] * n,
        "Perovskite_deposition_thermal_annealing_temperature":
            ["100", "100-150 C", "RT", np.nan, "80 | 120", "100"],
        "Perovskite_deposition_thermal_annealing_time":
            ["10", "20", np.nan, "30", "5; 15", "10"],
        "Cell_architecture": ["nip", "nip", "pin", "pin", "mesa", "nip"],
        "ETL_stack_sequence": ["TiO2-c", "SnO2-np", "TiO2-c", np.nan, "TiO2-c", "SnO2-np"],
        "HTL_stack_sequence": ["Spiro", "Spiro", "PTAA", "PTAA", "none", "Spiro"],
        "Perovskite_deposition_solvents": ["DMF", "DMF", np.nan, "DMSO", "DMF", "GBL"],
        # sizinti adayi kolon: beyaz-liste geregi CIKTIDA OLMAMALI
        "JV_default_Voc": [1.0] * n,
        "JV_default_PCE": [10.0, 12.0, 8.0, 15.0, 5.0, 11.0],
        "Ref_DOI_number": ["d1", "d1", "d2", "d2", "d3", "d3"],
    })


def test_band_gap_median_fill_and_flag():
    out = build_features(make_df(), make_cfg())
    assert list(out["band_gap_missing"]) == [0, 1, 0, 0, 1, 0]
    # gozlemli medyan: median(1.5, 1.7, 1.6, 1.6) = 1.6
    assert np.allclose(out.loc[out["band_gap_missing"] == 1, "band_gap"], 1.6)


def test_anneal_parsing_takes_max_and_flags_nonnumeric():
    out = build_features(make_df(), make_cfg())
    # "100-150 C" -> 150 (maks), "80 | 120" -> 120, "RT" -> sayisiz -> eksik bayragi
    assert out.loc[1, "anneal_temp"] == 150.0
    assert out.loc[4, "anneal_temp"] == 120.0
    assert list(out["anneal_temp_missing"]) == [0, 0, 1, 1, 0, 0]


def test_onehot_topn_and_other_bucket():
    out = build_features(make_df(), make_cfg(top_n=2))
    # arch: nip(3) ve pin(2) top-2; "mesa" -> arch_other kovasi
    assert "arch_nip" in out.columns and "arch_pin" in out.columns
    other = [c for c in out.columns if c.startswith("arch_") and c.endswith("_other")]
    assert len(other) == 1
    assert out[other[0]].sum() == 1
    # NaN cozucu "unknown" kategorisine gider (top-2 disi ise other'a duser) — toplam korunur
    solv_cols = [c for c in out.columns if c.startswith("solv_")]
    assert int(out[solv_cols].to_numpy().sum()) == len(out)


def test_whitelist_blocks_leakage_columns():
    out = build_features(make_df(), make_cfg())
    jv_cols = [c for c in out.columns if c.startswith("JV_")]
    # hedef kolonu haric hicbir JV_ kolonu ozellik matrisine gecmez
    assert jv_cols == ["JV_default_PCE"]
    assert "Ref_DOI_number" in out.columns


def test_flexible_binary_and_safe_column_names():
    out = build_features(make_df(), make_cfg())
    assert list(out["flexible"]) == [1, 0, 0, 1, 0, 1]
    import re
    assert all(re.fullmatch(r"[0-9A-Za-z_]+", c) for c in out.columns)
