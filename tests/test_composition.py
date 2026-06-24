"""Kompozisyon vektorizasyonu birim testleri (pytest).
SE acisindan: cekirdek donusumun dogrulugu otomatik test ile garanti altinda.
Calistirma: pytest -q
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from perovskite_ml.features.composition import parse_site, vectorize_site

KNOWN_A = ["FA", "MA", "Cs", "Rb", "K"]

def test_parse_normalizes_to_one():
    parsed = parse_site("Cs; FA; MA", "0.05; 0.788; 0.162")
    assert abs(sum(c for _, c in parsed) - 1.0) < 1e-9

def test_known_ions_routed_correctly():
    parsed = parse_site("FA; MA", "0.8; 0.2")
    v = vectorize_site(parsed, KNOWN_A, "A")
    assert abs(v["A_FA"] - 0.8) < 1e-9
    assert abs(v["A_MA"] - 0.2) < 1e-9
    assert v["A_other"] == 0.0

def test_unknown_ion_goes_to_other():
    parsed = parse_site("FA; GU", "0.5; 0.5")   # GU bilinen listede yok
    v = vectorize_site(parsed, KNOWN_A, "A")
    assert abs(v["A_FA"] - 0.5) < 1e-9
    assert abs(v["A_other"] - 0.5) < 1e-9

def test_mismatch_raises():
    with pytest.raises(ValueError):
        parse_site("Cs; FA", "1.0")             # 2 iyon, 1 katsayi
