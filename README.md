# Perovskite PCE ML — Tekrar Uretilebilir, Sizinti-Guvenli ML Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20829027.svg)](https://doi.org/10.5281/zenodo.20829027)

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

## Calistirma sirasi
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

## Testler
```bash
pytest -q
```

## Klasor yapisi
- `config.yaml` — tum parametreler (seed, kolonlar, temizleme esikleri)
- `src/perovskite_ml/` — paket (data, features, models, validation, explain, candidates)
- `scripts/` — sirayla calisan asama runner'lari
- `tests/` — birim testler
- `data/`, `outputs/` — uretilen icerik (git'e girmez)

## Tam calisma ortami (tekrar uretilebilirlik)
Calismanin kaydedildigi ortam (run-manifest'ten):
Python 3.14.4, numpy 2.4.6, pandas 3.0.3, scikit-learn 1.8.0, scipy 1.17.1, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, shap 0.51.0.
Kesin surumler `requirements-lock.txt` dosyasinda; ayrica her kosu `outputs/manifests/` altina timestamp + seed + paket surumlerini JSON olarak yazar.

## Yazilim muhendisligi katmanlari
- **Veri-dogrulama / sema** (`validation/schema.py`): hat calismadan once veri sozlesmesini denetler; **model-ready icinde olcum-sonrasi (sizinti) kolonu kalmissa hatti durdurur** (Breck vd., 2019 operasyonel hali).
- **Run-manifest** (`utils/manifest.py`): her asama icin timestamp, seed, paket surumleri, satir/ozellik sayilari ve cikti listesini JSON olarak yazar (Sculley vd., 2015 — provenans/tekrar uretilebilirlik).
- **Testler** (`tests/`): kompozisyon vektorizasyonu, ozellik insasi, **sema/sizinti kontrolu**, **DOI-grup ayrim ayrikligi**, **aday uzayi sayisi (756)**.
- **Paketleme**: `pyproject.toml` (`pip install -e .`, `pytest`).
