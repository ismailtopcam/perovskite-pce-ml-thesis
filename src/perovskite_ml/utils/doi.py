"""Kanonik DOI anahtari (v0.4.0, 1-C yeniden-dogrulama paketi).

Ham Ref_DOI_number dizgileri ayni yayini farkli yazimlarla (buyuk/kucuk harf,
https://doi.org/ on-eki, sondaki / veya .) temsil edebiliyor; bu varyantlar
DOI-grup ayriminda ayni yayini kat/holdout sinirlarinin iki yanina dusurebilir
(tez Bolum 5.7'deki denetim: 44 yayin, 420 kayit). normalize_doi() bu
varyantlari tek kanonik anahtara indirger. Kural kumesi, v0.3.x'teki
doi_normalizasyon_duyarlilik.py ile birebir aynidir; v0.4.0'da ana hatta
alinmistir.
"""
import re

def normalize_doi(s: str) -> str:
    """Kanonik DOI: BOM/bosluk temizligi, kucuk harf, protokol/on-ek ve
    sondaki /. temizligi. Bos veya sayisal-olmayan girdiler oldugu gibi
    kucuk-harfli dondurulur (grup anahtari olarak yine tutarli calisir)."""
    s = s.strip().lstrip("﻿").strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.rstrip("/.")
