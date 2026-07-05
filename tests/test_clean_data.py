"""clean_rows birim testleri: her cikarma nedeni + sinir-degeri (0 ve 35 DAHIL) davranisi."""
import pandas as pd

from perovskite_ml.data.clean_data import clean_rows

CFG = {
    "target": "JV_default_PCE",
    "cleaning": {"pce_min": 0.0, "pce_max": 35.0},
    "columns": {
        "module_flag": "Module",
        "composition": {
            "a_ions": "A_ions", "a_coef": "A_coef",
            "b_ions": "B_ions", "b_coef": "B_coef",
            "x_ions": "X_ions", "x_coef": "X_coef",
        },
    },
}

OK_COMP = {"A_ions": "FA", "A_coef": "1", "B_ions": "Pb", "B_coef": "1",
           "X_ions": "I", "X_coef": "1"}


def row(pce, module="false", **comp):
    r = {"JV_default_PCE": pce, "Module": module, **OK_COMP}
    r.update(comp)
    return r


def test_each_removal_reason_logged():
    df = pd.DataFrame([
        row(10.0),                                        # gecerli
        row("abc"),                                       # pce_not_numeric
        row(36.0),                                        # pce_out_of_range (ust)
        row(-0.5),                                        # pce_out_of_range (alt)
        row(12.0, module="TRUE"),                         # module_record
        row(12.0, A_ions=None),                           # missing_ion_or_coef
        row(12.0, X_ions="I; Br", X_coef="0.8"),          # ion_coef_count_mismatch
        row(12.0, B_coef="bir"),                          # coef_not_numeric
    ])
    clean, log = clean_rows(df, CFG)
    assert len(clean) == 1
    counts = log["reason"].value_counts().to_dict()
    assert counts == {"pce_not_numeric": 1, "pce_out_of_range": 2, "module_record": 1,
                      "missing_ion_or_coef": 1, "ion_coef_count_mismatch": 1,
                      "coef_not_numeric": 1}


def test_pce_bounds_are_inclusive():
    # PCE = 0 ve PCE = 35 GECERLI kayittir (between sinirlar dahil).
    # Bu davranis Stage 11 denetim betiginin tanimiyla da hizalidir.
    df = pd.DataFrame([row(0.0), row(35.0), row(35.0001)])
    clean, log = clean_rows(df, CFG)
    assert len(clean) == 2
    assert list(log["reason"]) == ["pce_out_of_range"]
