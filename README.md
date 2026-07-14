# Perovskite PCE ML — Tekrar Üretilebilir, Sızıntı-Güvenli ML Hattı

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21196013.svg)](https://doi.org/10.5281/zenodo.21196013)
[![CI](https://github.com/ismailtopcam/perovskite-pce-ml-thesis/actions/workflows/ci.yml/badge.svg)](https://github.com/ismailtopcam/perovskite-pce-ml-thesis/actions/workflows/ci.yml)

Perovskite Database kayıtlarından PCE tahmini için **doğrulama-metodolojisi odaklı**
bir makine öğrenmesi / yazılım mühendisliği çalışması. Vurgu, en yüksek skor değil;
**sızıntı-güvenli (leakage-safe), DOI-grup-güvenli, tekrar üretilebilir** bir hat ve
mühendislik kararlarının sonuç geçerliliğini nasıl değiştirdiğinin ölçülmesidir.

## Sonuçlar (bir bakışta)

| Başlık | Sonuç |
|---|---|
| Temiz veri | 41.485 kayıt (ham: 43.398 x 410 kolon) |
| Özellik sayısı | 77 |
| Nihai model | CatBoost (ilk 4 boosting ~eşit; salt-sklearn HistGBM dahil) |
| Ana doğrulama | DOI-grup 5-kat GroupKFold CV |
| **R2 (grup-güvenli)** | **0,413 ± 0,006** |
| R2 (rastgele bölme — sızıntı şişmesi) | 0,481 (+0,068) |
| R2 (+Voc/Jsc/FF — hedef sızıntısı) | 0,938 (bu yüzden dışlandı) |
| MAE / RMSE | 3,12 / 4,02 PCE puanı |
| Aday uzayı / eğitimde desteklenen | 756 / 504 |

Düşük görünen 0,413 bilinçli bir tercihtir: görülmemiş yayınlara genellemeyi ölçer.
Tüm değerler `outputs/` altındaki commit'li dosyalardan izlenebilir.
Model ve veri özeti: [MODEL_CARD.md](MODEL_CARD.md) · [DATA_CARD.md](DATA_CARD.md)

## Etkileşimli web uygulaması

Tezin veri hattı, doğrulama deneyleri ve bulguları etkileşimli olarak yayında:
**<https://tez.ismailtopcam.dev>**

- Kaynak kodu ayrı depodadır: <https://github.com/ismailtopcam/perovskite-pce-ml-app>
  (Streamlit; 9 sayfa — veri hattı gezgini, doğrulama deneyleri, canlı tahmin +
  conformal aralık + yerel SHAP, DOI gezgini, provenans).
- Uygulamanın tüm verisi ve modeli **bu depodan deterministik üretilir**
  (`tools/prepare_app_data.py`, pipeline commit'i uygulama içinde görünür); model
  artefaktı SHA-256'sı [MODEL_CARD.md](MODEL_CARD.md)'de kayıtlıdır. Yani web
  sitesindeki her sayı, buradaki commit'li çıktılarla aynı kaynaktan gelir.

## Kurulum
```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# Linux/Mac:      source .venv/bin/activate
pip install -r requirements.txt
# Kesin sürümlerle (tekrar üretilebilirlik):  pip install -r requirements-lock.txt
```

## Veri
Veri kaynağı: **Perovskite Database** (Jacobsson vd., 2022, *Nature Energy*, DOI: 10.1038/s41560-021-00941-3).
Ham CSV bu depoda **yeniden dağıtılmaz** (`.gitignore`); özgün kaynağından indirilir:

1. https://www.perovskitedatabase.com adresinden (Download bölümü) tüm-veri CSV'sini indirin.
2. Dosyayı tam olarak şu yola yerleştirin (ad dahil):
   ```
   data/raw/Perovskite_database_content_all_data.csv
   ```
3. Bu çalışmada kullanılan anlık görüntünün kimliği (indirdiğiniz dosyayı doğrulayın):
   - **43.398 kayıt x 410 kolon**, ~87 MB
   - SHA-256: `da66a634e9106e58ce4d012558d468bc2f19b95987f149fb9c5d208d363ea67a`

Perovskite Database yaşayan bir veri tabanıdır; daha güncel bir anlık görüntüyle kayıt sayıları ve
türetilen sonuçlar küçük farklar gösterebilir. Tezde raporlanan tüm sayılar yukarıdaki anlık
görüntüye aittir; farklı bir kopya kullanıyorsanız önce satır sayısı ve sağlama toplamını (checksum) karşılaştırın.

## Çalıştırma sırası

Tek komut (tüm çekirdek hat):
```bash
python run_all.py            # çekirdek hat (01-10)
python run_all.py --all      # + ek doğrulama/sağlamlık betikleri (11-14 + thesis_data_analysis + thesis_figures)
```

Ya da adım adım:
```bash
python scripts/01_prepare_data.py           # veri temizleme + kompozisyon vektörizasyonu
python scripts/02_build_features.py         # özellik matrisi (cihaz + süreç)
python scripts/03_compare_models.py         # 4+ model, DOI-grup holdout/CV
python scripts/04_validation_experiments.py # rastgele vs grup, özellik kapsamı
python scripts/05_shap_analysis.py          # SHAP açıklanabilirlik
python scripts/06_generate_candidates.py    # aday uzayı + ekstrapolasyon analizi
python scripts/07_screening_utility.py      # model tabanlı önceliklendirme faydası (enrichment)
python scripts/08_hyperparameter_tuning.py  # grup-güvenli hiperparametre araması (default vs tuned)
python scripts/09_descriptor_ablation.py    # fiziksel tanımlayıcı (Goldschmidt/tau) ablasyonu
python scripts/10_conformal_uncertainty.py  # tahmin belirsizliği: DOI-grup-güvenli conformal + uygulanabilirlik-alanı
```

## Ek doğrulama / sağlamlık betikleri
Çekirdek hattın (01-10) sonuçlarını stres testine tabi tutan ek doğrulama betikleri (manifest + outputs üretir):
```bash
python scripts/11_pce_outlier_audit.py      # PCE>30 denetimi + PCE<=30/<=28 duyarlılık
python scripts/12_catboost_tuning.py        # CatBoost doğrudan grup-güvenli hiperparametre araması (default vs tuned)
python scripts/13_preprocessing_and_bandgap.py # ön-işleme kat-değişmezliği + band gap ablasyonu
python scripts/14_pipeline_cv.py            # ön-işleme sızıntısı: kat-içi (fold-local) sklearn Pipeline vs global (aynı GroupKFold)
python scripts/thesis_data_analysis.py data/processed/model_ready_dataset.csv
                                            # betimsel istatistik: DOI grup dağılımı, gürültü tabanı (Tablo 5.1), band gap (ML yok)
python scripts/thesis_figures.py            # tez veri şekillerinin (4.1-4.7, 4.12, 4.13) repo içinden yeniden üretimi -> outputs/figures/
```

Tezde atıf yapılan üç ek denetim betiği (repo kökünde; çıktıları sürüm kontrolünde):
```bash
python pce30_ustu_stack_incelemesi.py    # PCE>30 kayıtların hücre yığını denetimi (tandem izi var mı?) -> outputs/robustness/pce30_ustu_stack.csv
python aday_uyumluluk_filtresi.py        # (mimari,ETL,HTL) üçlülerinin eğitimde birlikte-görülme denetimi -> outputs/candidates_full/*_compat_N5.csv, triple_cooccurrence.csv
python htl_free_yayin_ici.py             # HTL-free vs HTL'li hücrelerin yayın-içi (paired) PCE farkı (tez Bölüm 5.3) -> outputs/robustness/htl_free_yayin_ici.json
```

## DOI-grup denetim betikleri
Gruplama anahtarının (ham DOI dizgileri) kalitesini ve olası kalıntı sızıntıyı ölçen bağımsız denetim betikleri (repo kökünde):
```bash
python doi_grup_dogrulama_v2.py          # DOI'siz kayıt sayımı, normalizasyon çarpışmaları, holdout/CV kat-sınırı aşımları -> outputs/robustness/doi_grup_dogrulama.json
python doi_normalizasyon_duyarlilik.py   # normalize_doi() uygulanmış gruplarla aynı CatBoost'un duyarlılık koşumu (~40 sn) -> outputs/robustness/doi_normalizasyon_duyarlilik.json
python kontamine_test_maesi.py           # sınır aşan yayınların holdout-test kayıtlarında hata analizi (~15 sn) -> outputs/robustness/kontamine_test_maesi.json
```
Bulgu özeti: 44 yayın büyük/küçük harf varyantıyla çift kayıtlı; kat-sınırı aşan 362 kayıt (%0,87).
Normalizasyonla fark |dR2| <= 0,003 (kat gürültüsü içinde); skor şişirme izi yok. Başlık sonuçlar bu nedenle
özgün koşumdan raporlanır; kusur ölçülmüş kalıntı risk olarak belgelenmiştir. Üç koşumun sayısal çıktıları
`outputs/robustness/` altında commit'lidir: `doi_grup_dogrulama.json` (44/362 denetimi),
`doi_normalizasyon_duyarlilik.json` ve `kontamine_test_maesi.json` (tez Bölüm 5.7'deki değerler).

## Testler
```bash
pytest -q     # 31 test; ham veri gerektirmez (sentetik veriyle çalışır)
```
CI (`.github/workflows/ci.yml`) her push/PR'da testleri **Python 3.10 ve 3.13** üzerinde koşar;
güncel durum yukarıdaki CI rozetinde görünür. Kapsam: kompozisyon dönüşümü, özellik inşası
(medyan+bayrak, top-N one-hot), şema/sızıntı kontrolü, DOI-grup ayrım ayrıklığı, veri temizleme
nedenleri ve sınır dahilliği, aday uzayı kodlama sözleşmesi (756) ve **uçtan uca duman testi**
(temizleme → vektörleştirme → özellik → grup-güvenli eğitim/tahmin → koşu-manifesti;
`tests/test_smoke_pipeline.py`).

**Kapsam ayrımı:** CI, paket modüllerinin birlikte çalıştığını hafif bağımlılık kümesiyle ve
eski/yeni Python sürümleriyle doğrular; tezde raporlanan sayılar ise "Tam çalışma ortamı"
bölümündeki lock ortamında, 41.485 kayıtlık gerçek koşumla üretilmiştir. CI rozeti bu tam
koşumu değil, uyumluluğu kanıtlar.

## Klasör yapısı
- `config.yaml` — tüm parametreler (seed, kolonlar, temizleme eşikleri)
- `src/perovskite_ml/` — paket (data, features, models, validation, explain, candidates, simulation, utils)
- `scripts/` — sırayla çalışan aşama çalıştırıcıları
- `tests/` — birim testler
- `data/`, `outputs/` — üretilen içerik (varsayılan git-dışı; tezde atıf yapılan küçük sonuç dosyaları sürüm kontrolüne dahildir)

## Tam çalışma ortamı (tekrar üretilebilirlik)
Çalışmanın kaydedildiği ortam (run-manifest'ten):
Python 3.14.4, numpy 2.4.6, pandas 3.0.3, scikit-learn 1.8.0, scipy 1.17.1, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, shap 0.51.0.
Kesin sürümler `requirements-lock.txt` dosyasında; ayrıca her koşu `outputs/manifests/` altına timestamp + seed + paket sürümlerini JSON olarak yazar.

**Not:** Tezde raporlanan tüm sayısal çıktılar bu lock ortamında üretilmiştir. `requirements.txt`
geniş uyumluluk için alt-sınır bildirir; birebir tekrar üretim için `requirements-lock.txt` kullanın.
Seed sabitleme aynı ortam içinde determinizmi garanti eder; farklı işletim sistemi / derleyici /
kütüphane sürümü kombinasyonlarında (özellikle boosting kütüphanelerinde) binde-bir düzeyinde
küçük sayısal farklar oluşabilir.

## Lisans ve atıf

Bu deponun **MIT lisansı yalnızca kodu kapsar**. Perovskite Database verisi kendi
lisansına tabidir (**CC BY 4.0** — atıf zorunlu: Jacobsson vd., 2022, *Nature Energy*;
ayrıntı: [DATA_CARD.md](DATA_CARD.md)). Ham veri bu depoda yeniden dağıtılmaz.
Bu çalışmaya atıf için `CITATION.cff` dosyasına bakın.

## Yazılım mühendisliği katmanları
- **Veri-doğrulama / şema** (`validation/schema.py`): hat çalışmadan önce veri sözleşmesini denetler; **model-ready içinde ölçüm-sonrası (sızıntı) kolonu kalmışsa hattı durdurur** (Breck vd., 2019 operasyonel hâli).
- **Run-manifest** (`utils/manifest.py`): her aşama için timestamp, seed, paket sürümleri, satır/özellik sayıları ve çıktı listesini JSON olarak yazar (Sculley vd., 2015 — provenans/tekrar üretilebilirlik).
- **Testler** (`tests/`): kompozisyon vektörizasyonu, özellik inşası, **şema/sızıntı kontrolü**, **DOI-grup ayrım ayrıklığı**, **aday uzayı sayısı (756)**.
- **Paketleme**: `pyproject.toml` (`pip install -e .`, `pytest`).
