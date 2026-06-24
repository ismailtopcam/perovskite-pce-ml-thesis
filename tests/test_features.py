import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perovskite_ml.features.feature_builder import _first_number
import math

def test_first_number_basic():
    assert _first_number("100") == 100.0
def test_first_number_pipe_takes_max():
    assert _first_number("100 | 150") == 150.0
def test_first_number_textual():
    assert math.isnan(_first_number("RT"))
def test_first_number_range():
    assert _first_number("100-150 C") == 150.0
