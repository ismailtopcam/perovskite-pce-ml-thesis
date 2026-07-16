"""Commit'li aday-uyumluluk ciktilarinin ic tutarliligi (regresyon testi).

546 -> 504 duzeltmesi sirasinda candidates_compat_N5.csv'nin bayat kalmasi
gibi bir tutarsizligin bir daha sessizce yasanmamasi icin: filtreli CSV,
triple_cooccurrence + candidate_predictions'tan YENIDEN turetilip karsilastirilir.
Commit'li ciktilar mevcut degilse (hafif kurulum/CI disi senaryo) atlanir.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "outputs" / "candidates_full"

pytestmark = pytest.mark.skipif(
    not (CAND / "candidates_compat_N5.csv").exists(),
    reason="commit'li aday ciktilari bulunamadi")

KEY = ["arch", "ETL", "HTL"]


def test_compat_filter_consistent_with_sources():
    cand = pd.read_csv(CAND / "candidate_predictions.csv")
    tri = pd.read_csv(CAND / "triple_cooccurrence.csv")
    compat = pd.read_csv(CAND / "candidates_compat_N5.csv")

    assert len(cand) == 756 and len(tri) == 18
    valid = tri.loc[tri["egitim_kayit_sayisi"] >= 5, KEY]
    beklenen = cand.merge(valid, on=KEY)

    # 756/504 sozlesmesi: filtreli liste, kaynaklarindan birebir turetilebilmeli
    assert len(compat) == len(beklenen) == 504
    lhs = compat[KEY + ["A", "B", "X"]].sort_values(KEY + ["A", "B", "X"]).reset_index(drop=True)
    rhs = beklenen[KEY + ["A", "B", "X"]].sort_values(KEY + ["A", "B", "X"]).reset_index(drop=True)
    assert lhs.equals(rhs)

    # filtre tahmin tavanini degistirmez (tez Bolum 4.5)
    assert abs(compat["Predicted_PCE"].max() - cand["Predicted_PCE"].max()) < 1e-9


def test_top30_compat_subset():
    top30 = pd.read_csv(CAND / "top30_compat_N5.csv")
    compat = pd.read_csv(CAND / "candidates_compat_N5.csv")
    assert len(top30) == 30
    merged = top30.merge(compat, on=KEY + ["A", "B", "X"], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
