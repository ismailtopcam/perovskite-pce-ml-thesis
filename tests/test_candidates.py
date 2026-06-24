from perovskite_ml.candidates.candidate_space import axis_sizes

def test_candidate_count_is_documented_756():
    ax = axis_sizes()
    prod = 1
    for v in ax.values():
        prod *= v
    assert prod == 756, f"Aday uzayi 756 olmali, {prod} cikti — config/eksen degismis."
