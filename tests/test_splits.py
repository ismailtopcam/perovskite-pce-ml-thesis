import numpy as np, pandas as pd
from perovskite_ml.validation.splits import holdout_split, group_kfold

def _data(n=200, n_groups=20):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f": rng.randn(n)})
    y = rng.randn(n)
    groups = np.array([f"doi{i%n_groups}" for i in range(n)])
    return X, y, groups

def test_holdout_groups_disjoint():
    X, y, g = _data()
    tr, te = holdout_split(X, y, g, 0.2, 42)
    assert set(g[tr]).isdisjoint(set(g[te]))   # ayni DOI iki tarafta olamaz

def test_kfold_groups_disjoint():
    X, y, g = _data()
    for tr, te in group_kfold(X, y, g, 5):
        assert set(g[tr]).isdisjoint(set(g[te]))


def test_nan_doi_sentinel_single_group():
    """Betiklerdeki sozlesme: DOI'siz (NaN) kayitlar fillna('unknown_doi') ile TEK grup olur
    ve bu grup hicbir bolmede egitim/test arasinda bolunmez (bkz. tez, Bolum 3.8 / P313)."""
    X, y, g = _data()
    raw = pd.Series(g).astype(object)
    raw.iloc[::10] = np.nan                      # her 10. kayit DOI'siz
    groups = raw.fillna("unknown_doi").astype(str).values
    assert (groups == "unknown_doi").sum() == 20  # tek sentinel grup, tum NaN'lar toplandi

    tr, te = holdout_split(X, y, groups, 0.2, 42)
    sides = {"tr" if i in set(tr) else "te" for i in np.where(groups == "unknown_doi")[0]}
    assert len(sides) == 1                        # sentinel grup tek tarafta

    for tr_i, te_i in group_kfold(X, y, groups, 5):
        assert set(groups[tr_i]).isdisjoint(set(groups[te_i]))


# ---- Kat atamasi: kararli siralama ve kayit mekanizmasi (tez v121) ----
import hashlib, json
from pathlib import Path
import pytest
from perovskite_ml.validation import splits as SP


def _bagli_gruplar(n_groups=60, seed=1):
    """Cok sayida esit boyutlu grup: kararsiz siralamada kat uyeliginin ortama bagli oldugu durum."""
    rng = np.random.RandomState(seed)
    boyut = rng.choice([1, 2, 3, 5], size=n_groups)
    g = np.repeat([f"doi{i:03d}" for i in range(n_groups)], boyut)
    return g[rng.permutation(len(g))]


def test_kararli_atama_ayrik_dengeli_ve_tekrarlanir():
    g = _bagli_gruplar()
    a = SP.kararli_kat_atamasi(g, 5)
    assert np.array_equal(a, SP.kat_atamasi(g, 5, atama="kararli"))
    assert np.array_equal(a, SP.kararli_kat_atamasi(g.astype(object), 5))   # dizi turunden bagimsiz
    SP._ayriklik_denetle(g, a)                                              # her grup tek katta
    boy = np.bincount(a, minlength=5)
    assert boy.min() > 0 and boy.max() - boy.min() <= 5                     # acgozlu dagitim dengeli
    katlar = SP.group_kfold(None, None, g, 5, atama="kararli")
    assert [len(te) for _, te in katlar] == list(boy)
    assert sorted(np.concatenate([te for _, te in katlar])) == list(range(len(g)))


def test_kararli_atama_sklearn_1_9_ile_ayni():
    import sklearn
    if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) < (1, 9):
        pytest.skip("sklearn < 1.9 GroupKFold kararsiz siralar; esitlik yalnizca >= 1.9 icin beklenir")
    for seed in (1, 2, 3):
        g = _bagli_gruplar(seed=seed)
        assert np.array_equal(SP.kararli_kat_atamasi(g, 5), SP._sklearn_atama(g, 5))
        assert np.array_equal(SP.kararli_kat_atamasi(g, 3), SP._sklearn_atama(g, 3))


def test_parmak_izi_etikete_degil_yapiya_bagli():
    g = _bagli_gruplar().astype(object)
    g1, g2 = g.copy(), g.copy()
    g1[:7], g2[:7] = "unknown_doi", "u"            # ikisi de 'doi...' etiketlerinden sonra siralanir
    assert SP.kat_parmak_izi(g1, 5) == SP.kat_parmak_izi(g2, 5)
    assert SP.kat_parmak_izi(g1, 5) != SP.kat_parmak_izi(g1, 3)
    assert SP.kat_parmak_izi(g1, 5) != SP.kat_parmak_izi(g, 5)


def test_kayit_gidis_donus_ve_butunluk(tmp_path, monkeypatch):
    monkeypatch.setenv("PEROVSKITE_KAT_DIZINI", str(tmp_path))
    monkeypatch.delenv("PEROVSKITE_KAT_KAYIT", raising=False)
    monkeypatch.delenv("PEROVSKITE_KAT_DOGRULA", raising=False)
    g = _bagli_gruplar()
    assert SP.kayitli_kat_atamasi(g, 5) is None
    with pytest.warns(UserWarning, match="kayitli kat atamasi yok"):
        SP.kat_atamasi(g, 5)                                    # kayit yok -> sklearn + uyari
    ozel = (SP.kararli_kat_atamasi(g, 5) + 2) % 5               # sklearn'inkinden farkli, gecerli bir atama
    kimlik = SP.kat_kaydet(g, 5, ozel, aciklama="test", kaynak="test")
    assert kimlik == SP.kat_parmak_izi(g, 5)
    assert np.array_equal(SP.kat_atamasi(g, 5), ozel)           # artik kayit kullanilir
    for f, (tr, te) in enumerate(SP.group_kfold(None, None, g, 5)):
        assert np.array_equal(te, np.where(ozel == f)[0]) and len(tr) + len(te) == len(g)
    assert SP.kayitli_kat_atamasi(g, 3) is None                 # baska kat sayisi baska kayittir

    monkeypatch.setenv("PEROVSKITE_KAT_DOGRULA", "1")
    with pytest.raises(AssertionError):                         # kayit bu ortamin atamasindan farkli
        SP.kat_atamasi(g, 5)
    monkeypatch.delenv("PEROVSKITE_KAT_DOGRULA")

    dosya = tmp_path / f"{kimlik}.txt"
    s = dosya.read_text(encoding="ascii")
    dosya.write_text(str((int(s[0]) + 1) % 5) + s[1:], encoding="ascii")
    with pytest.raises(ValueError, match="bozuk"):              # dosya degisti -> sha256 tutmaz
        SP.kayitli_kat_atamasi(g, 5)


def test_kayit_modu_yazar_dogrula_modu_gecer(tmp_path, monkeypatch):
    monkeypatch.setenv("PEROVSKITE_KAT_DIZINI", str(tmp_path))
    monkeypatch.setenv("PEROVSKITE_KAT_KAYIT", "1")
    g = _bagli_gruplar(seed=4)
    a = SP.kat_atamasi(g, 5)                                    # kayit yok -> yazilir
    monkeypatch.delenv("PEROVSKITE_KAT_KAYIT")
    monkeypatch.setenv("PEROVSKITE_KAT_DOGRULA", "1")
    assert np.array_equal(SP.kat_atamasi(g, 5), a)              # ayni ortamda dogrulama gecer
    assert np.array_equal(a, SP._sklearn_atama(g, 5))


def test_kat_kaydet_gecersiz_atamayi_reddeder(tmp_path, monkeypatch):
    monkeypatch.setenv("PEROVSKITE_KAT_DIZINI", str(tmp_path))
    g = _bagli_gruplar()
    bozuk = np.arange(len(g)) % 5                               # gruplari katlara boler -> sizinti
    with pytest.raises(ValueError):
        SP.kat_kaydet(g, 5, bozuk, aciklama="test", kaynak="test")
    with pytest.raises(ValueError):
        SP.kat_atamasi(g, 5, atama="bilinmeyen")


def test_depodaki_kat_kayitlari_butun():
    """Depoya islenmis atamalar (kat_atamalari/) kendi kayitlariyla tutarli olmali; veri gerektirmez."""
    d = Path(__file__).resolve().parents[1] / "kat_atamalari"
    if not (d / "kayit.json").exists():
        pytest.skip("kat_atamalari/ yok")
    kayitlar = json.loads((d / "kayit.json").read_text(encoding="utf-8"))["kayitlar"]
    assert kayitlar
    for kimlik, k in kayitlar.items():
        s = (d / k["dosya"]).read_text(encoding="ascii").strip()
        assert k["dosya"] == f"{kimlik}.txt" and len(s) == k["n_kayit"]
        assert hashlib.sha256(s.encode("ascii")).hexdigest() == k["sha256"]
        a = np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(int) - 48
        assert a.min() == 0 and a.max() == k["n_splits"] - 1
        assert [int(v) for v in np.bincount(a, minlength=k["n_splits"])] == k["kat_boyutlari"]
