"""DOI-grup-guvenli veri ayrimlari. Ayni yayina ait kayitlar egitim ve teste
birlikte dusmez -> ezberden kaynakli yapay yuksek skor onlenir.

Kat atamasi (tez v121)
----------------------
sklearn GroupKFold, esit buyuklukteki gruplari katlara dagitirken gruplari boyuta gore
siralar. Bu siralama sklearn < 1.9'da KARARSIZDIR: bag durumundaki gruplarin sirasi
kutuphane surumune ve islemcinin komut setine gore degisir. Bu veride 7.353 DOI grubunun
7.338'i boyutunu baska bir grupla paylastigindan kat uyeligi ortama baglidir.

Bu modul atamayi ortamdan bagimsiz kilar:

  atama="kayit"   (varsayilan) Bu (gruplar, kat sayisi) icin depoda kayitli bir atama
                  varsa o kullanilir (kat_atamalari/). Tezde raporlanan arsiv kosulari
                  boyle yeniden uretilir. Kayit yoksa sklearn GroupKFold'a dusulur ve
                  uyari verilir (eski davranis; ortama baglidir).
  atama="kararli" Gruplar KARARLI siralanir; her ortamda ayni sonucu verir
                  (sklearn >= 1.9 GroupKFold ile aynidir). Tez v118 denetimleri
                  (scripts/19, 22, 23, tarama_yonu_denetimi.py) bunu kullanir.

Ortam degiskenleri:
  PEROVSKITE_KAT_KAYIT=1    kayit yoksa sklearn atamasini depoya yazar (arsiv ortaminda kosulur).
  PEROVSKITE_KAT_DOGRULA=1  kayitli atamayi bu ortamin sklearn atamasiyla karsilastirir; farkliysa durur.
  PEROVSKITE_KAT_DIZINI     kayit dizini (varsayilan: <repo koku>/kat_atamalari).
"""
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold


def holdout_split(X, y, groups, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(X, y, groups))


def _kat_dizini():
    return Path(os.environ.get("PEROVSKITE_KAT_DIZINI",
                               Path(__file__).resolve().parents[3] / "kat_atamalari"))


def _grup_indeksi(groups):
    """sklearn GroupKFold ile ayni: benzersiz gruplar (sirali) ve her kaydin grup indeksi."""
    ug, gi = np.unique(np.asarray(groups), return_inverse=True)
    return ug, np.asarray(gi).ravel()


def _dagit(gi, sira, n_splits):
    """GroupKFold'un acgozlu dagitimi: gruplar `sira` ile gezilir, her biri en hafif kata eklenir."""
    npg = np.bincount(gi)
    yuk = np.zeros(n_splits)
    g2f = np.zeros(len(npg), dtype=int)
    for g in sira:
        f = int(np.argmin(yuk))
        yuk[f] += npg[g]
        g2f[g] = f
    return g2f[gi]


def kararli_kat_atamasi(groups, n_splits=5):
    """Ortamdan bagimsiz kat atamasi (kayit basina kat numarasi)."""
    _, gi = _grup_indeksi(groups)
    sira = np.argsort(np.bincount(gi), kind="stable")[::-1]
    return _dagit(gi, sira, n_splits)


def kat_parmak_izi(groups, n_splits):
    """(grup yapisi, kat sayisi) icin kimlik. Grup ETIKETLERINE degil yapisina baglidir:
    DOI'siz kayitlarin 'unknown_doi' / 'nan' / 'u' diye adlandirilmasi kimligi degistirmez
    (siralamadaki yeri ayni kaldigi surece)."""
    ug, gi = _grup_indeksi(groups)
    h = hashlib.sha256()
    h.update(f"{int(n_splits)}|{len(gi)}|{len(ug)}|".encode("ascii"))
    h.update(np.ascontiguousarray(gi, dtype="<i8").tobytes())
    return h.hexdigest()[:20]


def _kayit_oku():
    p = _kat_dizini() / "kayit.json"
    if not p.exists():
        return {"kayitlar": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def kayitli_kat_atamasi(groups, n_splits=5):
    """Depoda kayitli atamayi dondurur; kayit yoksa None."""
    k = _kayit_oku()["kayitlar"].get(kat_parmak_izi(groups, n_splits))
    if k is None:
        return None
    s = (_kat_dizini() / k["dosya"]).read_text(encoding="ascii").strip()
    if hashlib.sha256(s.encode("ascii")).hexdigest() != k["sha256"]:
        raise ValueError(f"kat atamasi dosyasi bozuk: {k['dosya']}")
    a = np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(int) - 48
    if len(a) != len(np.asarray(groups)):
        raise ValueError(f"kat atamasi uzunlugu tutmuyor: {k['dosya']}")
    return a


def kat_kaydet(groups, n_splits, atama, aciklama, kaynak):
    """Bir atamayi depoya yazar (kat_atamalari/<kimlik>.txt + kayit.json)."""
    a = np.asarray(atama, dtype=int)
    if n_splits > 10 or a.min() < 0 or a.max() >= n_splits or len(a) != len(np.asarray(groups)):
        raise ValueError("kat atamasi kaydedilemez (kat sayisi <= 10, 0..n-1, kayit sayisi kadar olmali)")
    _ayriklik_denetle(groups, a)
    d = _kat_dizini()
    d.mkdir(parents=True, exist_ok=True)
    kimlik = kat_parmak_izi(groups, n_splits)
    s = "".join(str(int(v)) for v in a)
    (d / f"{kimlik}.txt").write_text(s + "\n", encoding="ascii", newline="\n")
    kayit = _kayit_oku()
    kayit.setdefault("aciklama", "Tezde raporlanan kat atamalari. Uretim ve dogrulama: README, 'Kat atamalari'.")
    ug, _ = _grup_indeksi(groups)
    kayit["kayitlar"][kimlik] = {
        "dosya": f"{kimlik}.txt", "n_kayit": int(len(a)), "n_grup": int(len(ug)), "n_splits": int(n_splits),
        "kat_boyutlari": [int(v) for v in np.bincount(a, minlength=n_splits)],
        "sha256": hashlib.sha256(s.encode("ascii")).hexdigest(), "aciklama": aciklama, "kaynak": kaynak}
    kayit["kayitlar"] = dict(sorted(kayit["kayitlar"].items()))
    (d / "kayit.json").write_text(json.dumps(kayit, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8", newline="\n")
    return kimlik


def _sklearn_atama(groups, n_splits):
    g = np.asarray(groups)
    a = np.empty(len(g), dtype=int)
    for f, (_, te) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(g)), None, g)):
        a[te] = f
    return a


def _ayriklik_denetle(groups, a):
    """Her grup tek bir katta olmali (sizinti korumasi)."""
    _, gi = _grup_indeksi(groups)
    g2f = np.full(gi.max() + 1, -1)
    g2f[gi] = a
    if not np.array_equal(g2f[gi], a):
        raise ValueError("kat atamasi grup-ayrik degil: bir grup birden cok kata dusuyor")


def kat_atamasi(groups, n_splits=5, atama="kayit"):
    """Kayit basina kat numarasi (0..n_splits-1). Bkz. modul aciklamasi."""
    if atama == "kararli":
        return kararli_kat_atamasi(groups, n_splits)
    if atama != "kayit":
        raise ValueError(f"bilinmeyen atama: {atama!r} ('kayit' ya da 'kararli')")
    a = kayitli_kat_atamasi(groups, n_splits)
    if a is not None:
        _ayriklik_denetle(groups, a)
        if os.environ.get("PEROVSKITE_KAT_DOGRULA") == "1":
            if not np.array_equal(a, _sklearn_atama(groups, n_splits)):
                raise AssertionError("kayitli kat atamasi bu ortamdaki GroupKFold atamasindan farkli "
                                     f"(kimlik {kat_parmak_izi(groups, n_splits)})")
        return a
    a = _sklearn_atama(groups, n_splits)
    if os.environ.get("PEROVSKITE_KAT_KAYIT") == "1":
        import sklearn
        kat_kaydet(groups, n_splits, a, aciklama=os.environ.get("PEROVSKITE_KAT_ACIKLAMA", "(aciklama verilmedi)"),
                   kaynak=f"PEROVSKITE_KAT_KAYIT: sklearn {sklearn.__version__}, numpy {np.__version__}")
    else:
        warnings.warn("Bu (gruplar, kat sayisi) icin kayitli kat atamasi yok; sklearn GroupKFold kullanildi. "
                      "Atama ortama baglidir (bkz. perovskite_ml.validation.splits).", stacklevel=3)
    return a


def group_kfold(X, y, groups, n_splits=5, atama="kayit"):
    """[(egitim indeksleri, test indeksleri), ...] — GroupKFold.split ile ayni bicim ve kat sirasi."""
    a = kat_atamasi(groups, n_splits, atama)
    return [(np.where(a != f)[0], np.where(a == f)[0]) for f in range(n_splits)]
