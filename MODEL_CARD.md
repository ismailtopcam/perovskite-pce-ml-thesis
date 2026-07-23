# Model Kartı — Perovskit PCE Tahmin Modeli (CatBoost)

*Biçim: Mitchell vd. (2019) "Model Cards for Model Reporting" uyarlaması.*
*Son güncelleme: Temmuz 2026 · İsmailTopçam_Tez_2026 ile uyumlu.*

## Model kimliği

| Alan | Değer |
|---|---|
| Model türü | CatBoostRegressor (gradyan-artırmalı karar ağaçları) |
| Hiperparametreler | iterations=500, depth=6, learning_rate=0.05, seed=42 (test/holdout'a bakılmadan baştan sabitlendi; grup-güvenli ızgara araması aynı yapılandırmayı seçti — tez Bölüm 4.8) |
| Eğitim ortamı | Python 3.14.4, catboost 1.2.10, scikit-learn 1.8.0 (tam liste: `requirements-lock.txt` ve `outputs/manifests/`) |
| Kod sürümü | Bu depo (v0.3.3); kalıcı arşiv — tüm sürümleri kapsayan Zenodo DOI: 10.5281/zenodo.20829027 |
| Dağıtılan artefakt | Web uygulamasındaki `data_app/model.cbm` — SHA-256: `a9a0f431427c8023e2d10b9bb1cb1cd122a71ecbe2e50f46bbf7a8fad6ea0ae1` |

## Eğitim verisi

- **Kaynak:** Perovskite Database (Jacobsson vd., 2022) — anlık görüntü: 43.398 kayıt × 410 kolon, SHA-256 `da66a634…67a` (README "Veri").
- **Temizleme sonrası:** 41.485 kayıt; eğitim bölmesi 33.338 kayıt (DOI-grup-güvenli %80).
- **Hedef değişken:** `JV_default_PCE` (güç dönüşüm verimi, %).

## Özellikler

- **77 özellik:** A/B/X kompozisyon oranları (14), band gap + eksiklik bayrağı, esnek/yarı-saydam, tavlama sıcaklığı/süresi + bayraklar (8), mimari (7), ETL (16), HTL (16), çözücü (16) one-hot kodları.
- **Bilinçli olarak DIŞLANANLAR:** ölçüm-sonrası tüm kolonlar (`JV_*` — Voc, Jsc, FF dahil —, `EQE_*`, `Stabilised_*`, `Outdoor_*` vb.). Bu kolonlar hedefin dolaylı kopyasıdır; dahil edilmeleri R²'yi 0,938'e şişirir (tez Deney B). Şema katmanı (`validation/schema.py`) model-hazır veride bu kolonları görürse hattı durdurur.

## Başarım (hedef-sızıntısı denetimli, DOI-grup-ayrık protokol)

| Ölçüt | Değer |
|---|---|
| DOI-grup 5-kat CV R² | **0,413 ± 0,006** (birincil ölçüt) |
| Holdout R² (8.147 kayıt) | 0,392 |
| MAE / RMSE | 3,12 / 4,02 PCE puanı |
| Karşılaştırma | Rastgele bölme aynı modeli 0,481 gösterir (+0,068 yapay şişme) |

## Belirsizlik (conformal, DOI-grup-güvenli kalibrasyon)

- %90 hedef: yarı-genişlik ±6,42; **ampirik kapsama %88,2**.
- %80 hedef: yarı-genişlik ±4,68; **ampirik kapsama %76,1** (3,9 puanlık belirgin eksik kapsama — kalibrasyon/test DOI grupları arası dağılım kayması).
- Mondrian (tahmin-çeyreği koşullu): kapsama %88,3; çeyrek yarı-genişlikleri ±6,27 / ±6,99 / ±6,65 / ±5,50.
- Bu aralıklar **görülmemiş DOI gruplarındaki ampirik kapsama aralıkları** olarak okunmalıdır; kesin kapsama garantisi sunmaz (tez Bölüm 4.10).

## Bilinen sınırlılıklar

1. **Ekstrapolasyon yapamaz:** ağaç tabanlı model; aday tavanı 17,78 < eğitim maksimumu 34,8. Rekor kompozisyon öneremez (tez Bölüm 5.4).
2. **Etiket gürültüsü tabanı:** özdeş reçeteler yayınlar arasında ~3,5 puanlık giderilemeyen RMSE saçılımı taşır; bu da pratik R² tavanını ~0,52'ye sınırlar (tez Tablo 5.1). Model bu tavana yakın çalışır.
3. **Uygulanabilirlik alanı ayrıştırmıyor:** özellik-uzayı kNN mesafesi holdout hatasını ayırt etmedi (MAE 3,28 vs 3,19) — mesafe bir güven skoru değildir (tez Bölüm 4.10).
4. **Tek-eklemli rejim:** eğitim verisi tek-eklemli hücre ağırlıklıdır; gözlemli band gap değerlerinin yalnızca %6,2'si ≥1,70 eV. Tandem üst-hücre bölgesi kapsam dışıdır.
5. DOI anahtarında 44 büyük/küçük-harf çakışması ölçülmüş kalıntı risktir; normalizasyon duyarlılık koşumunda fark kat gürültüsü içindedir (tez Bölüm 5.7).
6. **Karşılaştırma modellerinde LightGBM parametre davranışı:** `subsample=0.8` yapılandırılmış olsa da `bagging_freq` verilmediğinden satır alt-örneklemesi LightGBM'de fiilen devre dışıdır; yalnızca sütun alt-örneklemesi (`colsample_bytree=0.8`) etkindir. Arşivlenmiş koşumların tekrar üretilebilirliği için parametre bilerek değiştirilmemiştir (`src/perovskite_ml/models/registry.py` içinde belgelidir; tez Bölüm 3.9). Nihai model (CatBoost) bu durumdan etkilenmez.

## Kullanım amacı ve kapsam dışı kullanım

- **Amaç:** deneysel taramayı daraltmak için **aday önceliklendirme**; doğrulama-metodolojisi araştırması; eğitim/gösterim (web uygulaması).
- **Kapsam DIŞI:** tek bir hücrenin PCE'sinin kesin tahmini; tandem/çok-eklemli cihazlar; stabilite/toksisite kararları; deneysel doğrulamanın yerine geçen her tür kullanım; nedensel çıkarım (SHAP değerleri gözlemsel ilişkidir).

## Bağlantılar

- Kod: <https://github.com/ismailtopcam/perovskite-pce-ml-thesis> · Arşiv (tüm sürümler): DOI 10.5281/zenodo.20829027
- Etkileşimli uygulama: <https://tez.ismailtopcam.dev>
- Veri kartı: [DATA_CARD.md](DATA_CARD.md)
