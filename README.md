# Perovskite PCE ML — Tekrar Uretilebilir, Sizinti-Guvenli ML Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21196013.svg)](https://doi.org/10.5281/zenodo.21196013)

Perovskite Database kayitlarindan PCE tahmini icin **dogrulama-metodolojisi odakli**
bir makine ogrenmesi / yazilim muhendisligi calismasi. Vurgu, en yuksek skor degil;
**leakage-safe, DOI-grup-guvenli, tekrar uretilebilir** bir hat ve muhendislik
kararlarinin sonuc gecerliligini nasil degistirdiginin olculmesidir.

## Kurulum
```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux/Mac:      source .venv/bin/activate
pip install -r requirements.txt
# Kesin surumlerle (tekrar uretilebilirlik):  pip install -r requirements-lock.txt
```

## Veri
```
data/raw/Perovskite_database_content_all_data.csv
```
Veri kaynagi: **Perovskite Database** (Jacobsson vd., 2022, *Nature Energy*, DOI: 10.1038/s41560-021-00941-3).
Ham CSV kendi ozgun kaynagindan indirilir; bu depoda **yeniden dagitilmaz** (`.gitignore`).

## Calistirma sirasi

Tek komut (tum cekirdek hat):
```bash
python run_all.py            # cekirdek hat (01-10)
python run_all.py --all      # + ek dogrulama/saglamlik betikleri (11-14)
```

Ya da adim adim:
```bash
python scripts/01_prepare_data.py          # veri temizleme + kompozisyon vektorizasyonu
python scripts/02_build_features.py        # ozellik matrisi (cihaz + surec)
python scripts/03_compare_models.py        # 4+ model, DOI-grup holdout/CV
python scripts/04_validation_experiments.py# rastgele vs grup, ozellik kapsami
python scripts/05_shap_analysis.py         # SHAP aciklanabilirlik
python scripts/06_generate_candidates.py   # aday uzayi + ekstrapolasyon analizi
python scripts/07_screening_utility.py     # model tabanli onceliklendirme faydasi (enrichment)
python scripts/08_hyperparameter_tuning.py # grup-guvenli hiperparametre aramasi (default vs tuned)
python scripts/09_descriptor_ablation.py   # fiziksel tanimlayici (Goldschmidt/tau) ablasyonu
python scripts/10_conformal_uncertainty.py # tahmin belirsizligi: DOI-grup-guvenli conformal + uygulanabilirlik-alani
```

## Ek dogrulama / saglamlik betikleri
Cekirdek hattin (01-10) sonuclarini stres-testen ek dogrulama betikleri (manifest + outputs uretir):
```bash
python scripts/11_pce_outlier_audit.py     # PCE>30 denetimi + PCE<=30/<=28 duyarlilik
python scripts/12_catboost_tuning.py        # CatBoost dogrudan grup-guvenli hiperparametre aramasi (default vs tuned)
python scripts/13_preprocessing_and_bandgap.py # on-isleme kat-invaryansi + band gap ablasyonu
python scripts/14_pipeline_cv.py            # on-isleme sizintisi: fold-local sklearn Pipeline vs global (ayni GroupKFold)
python scripts/thesis_data_analysis.py data/processed/model_ready_dataset.csv
                                            # betimsel istatistik: DOI grup dagilimi, gurultu tabani (Tablo 5.1), band gap (ML yok)
python scripts/thesis_figures.py            # tez veri sekillerinin (4.1-4.7, 4.12, 4.13) repo icinden yeniden uretimi -> outputs/figures/
```

Tezde atif yapilan iki ek denetim betigi (repo kokunde; ciktilari surum kontrolunde):
```bash
python pce30_ustu_stack_incelemesi.py    # PCE>30 kayitlarin hucre yigini denetimi (tandem izi var mi?) -> outputs/robustness/pce30_ustu_stack.csv
python aday_uyumluluk_filtresi.py        # (mimari,ETL,HTL) uclulerinin egitimde birlikte-gorulme denetimi -> outputs/candidates_full/*_compat_N5.csv, triple_cooccurrence.csv
```

## DOI-grup denetim betikleri
Gruplama anahtarinin (ham DOI dizgileri) kalitesini ve olasi kalinti sizintiyi olcen bagimsiz denetim betikleri (repo kokunde):
```bash
python doi_grup_dogrulama_v2.py          # DOI'siz kayit sayimi, normalizasyon carpismalari, holdout/CV kat-siniri asimlari
python doi_normalizasyon_duyarlilik.py   # normalize_doi() uygulanmis gruplarla ayni CatBoost'un duyarlilik kosumu (~40 sn)
python kontamine_test_maesi.py           # sinir asan yayinlarin holdout-test kayitlarinda hata analizi (~15 sn)
```
Bulgu ozeti: 44 yayin buyuk/kucuk harf varyantiyla cift kayitli; kat-siniri asan 362 kayit (%0,87).
Normalizasyonla fark |dR2| <= 0,003 (kat gurultusu icinde); skor sisirme izi yok. Baslik sonuclar bu nedenle
ozgun kosumdan raporlanir; kusur olculmus kalinti risk olarak belgelenmistir. Duyarlilik kosumunun sayisal
ciktisi `outputs/robustness/doi_normalizasyon_duyarlilik.json` olarak kaydedilir (tez Bolum 5.7'deki degerler).

## Testler
```bash
pytest -q
```

## Klasor yapisi
- `config.yaml` — tum parametreler (seed, kolonlar, temizleme esikleri)
- `src/perovskite_ml/` — paket (data, features, models, validation, explain, candidates)
- `scripts/` — sirayla calisan asama runner'lari
- `tests/` — birim testler
- `data/`, `outputs/` — uretilen icerik (varsayilan git-disi; tezde atif yapilan kucuk sonuc dosyalari surum kontrolune dahildir)

## Tam calisma ortami (tekrar uretilebilirlik)
Calismanin kaydedildigi ortam (run-manifest'ten):
Python 3.14.4, numpy 2.4.6, pandas 3.0.3, scikit-learn 1.8.0, scipy 1.17.1, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, shap 0.51.0.
Kesin surumler `requirements-lock.txt` dosyasinda; ayrica her kosu `outputs/manifests/` altina timestamp + seed + paket surumlerini JSON olarak yazar.

**Not:** Tezde raporlanan tum sayisal ciktilar bu lock ortaminda uretilmistir. `requirements.txt`
genis uyumluluk icin alt-sinir bildirir; birebir tekrar uretim icin `requirements-lock.txt` kullanin.
Seed sabitleme ayni ortam icinde determinizmi garanti eder; farkli isletim sistemi / derleyici /
kutuphane surumu kombinasyonlarinda (ozellikle boosting kutuphanelerinde) binde-bir duzeyinde
kucuk sayisal farklar olusabilir.

## Yazilim muhendisligi katmanlari
- **Veri-dogrulama / sema** (`validation/schema.py`): hat calismadan once veri sozlesmesini denetler; **model-ready icinde olcum-sonrasi (sizinti) kolonu kalmissa hatti durdurur** (Breck vd., 2019 operasyonel hali).
- **Run-manifest** (`utils/manifest.py`): her asama icin timestamp, seed, paket surumleri, satir/ozellik sayilari ve cikti listesini JSON olarak yazar (Sculley vd., 2015 — provenans/tekrar uretilebilirlik).
- **Testler** (`tests/`): kompozisyon vektorizasyonu, ozellik insasi, **sema/sizinti kontrolu**, **DOI-grup ayrim ayrikligi**, **aday uzayi sayisi (756)**.
- **Paketleme**: `pyproject.toml` (`pip install -e .`, `pytest`).
