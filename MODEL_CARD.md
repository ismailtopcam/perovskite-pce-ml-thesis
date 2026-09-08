# Model Kartı — Perovskit PCE Tahmin Modeli (CatBoost)

*Biçim: Mitchell vd. (2019) "Model Cards for Model Reporting" uyarlaması.*
*Son güncelleme: Ağustos 2026 (v0.4.0 / 1-C yeniden-doğrulama) · İsmailTopçam_Tez_2026 ile uyumlu.*

## Model kimliği

| Alan | Değer |
|---|---|
| Model türü | CatBoostRegressor (gradyan-artırmalı karar ağaçları) |
| Hiperparametreler | iterations=500, depth=6, learning_rate=0.05, seed=42 (test/holdout'a bakılmadan baştan sabitlendi; grup-güvenli ızgara araması aynı yapılandırmayı seçti — tez Bölüm 4.8) |
| Eğitim ortamı | Python 3.14.4, catboost 1.2.10, scikit-learn 1.8.0 (tam liste: `requirements-lock.txt` ve `outputs/manifests/`) |
| Kod sürümü | Bu depo (v0.4.0); kalıcı arşiv — tüm sürümleri kapsayan Zenodo DOI: 10.5281/zenodo.20829027 |
| Dağıtılan artefakt | Web uygulamasındaki `data_app/model.cbm` — SHA-256: `a260d88e6c018e1263ba5f742600c0e70a1b15041a590cc8e02222df0f8f0af3` (v0.4.0 kanonik-DOI eğitim bölmesiyle üretildi; 10 Ağustos 2026'da dağıtıldı) |

## Eğitim verisi

- **Kaynak:** Perovskite Database (Jacobsson vd., 2022) — anlık görüntü: 43.398 kayıt × 410 kolon, SHA-256 `da66a634…67a` (README "Veri").
- **Temizleme sonrası:** 41.485 kayıt; eğitim bölmesi 33.130 kayıt (kanonik-DOI-grup-güvenli %80).
- **Grup anahtarı (v0.4.0):** kanonik DOI — yazım varyantları ana hatta normalize edilir (44 varyant grubu birleşti; 7.352 benzersiz DOI; kat/holdout sınırı aşan kayıt: 0).
- **Hedef değişken:** `JV_default_PCE` (güç dönüşüm verimi, %).

## Özellikler

- **77 özellik:** A/B/X kompozisyon oranları (14), band gap + eksiklik bayrağı, esnek/yarı-saydam, tavlama sıcaklığı/süresi + bayraklar (8), mimari (7), ETL (16), HTL (16), çözücü (16) one-hot kodları.
- **Bilinçli olarak DIŞLANANLAR:** ölçüm-sonrası tüm kolonlar (`JV_*` — Voc, Jsc, FF dâhil —, `EQE_*`, `Stabilised_*`, `Outdoor_*` vb.). Bu kolonlar hedefin dolaylı kopyasıdır; dâhil edilmeleri R²'yi 0,938'e şişirir (tez Deney B). Şema katmanı (`validation/schema.py`) model-hazır veride bu kolonları görürse hattı durdurur.

## Başarım (hedef-sızıntısı denetimli, DOI-grup-ayrık protokol)

| Ölçüt | Değer |
|---|---|
| DOI-grup 5-kat CV R² | **0,410 ± 0,016** (birincil ölçüt) |
| Holdout R² (8.355 kayıt) | 0,394 (mühürlü protokol: model seçimi yalnızca eğitim-içi CV ile yapıldığında da aynı model seçilir ve değer değişmez; nested CV: 0,408 ± 0,017) |
| MAE / RMSE | 3,12 / 4,02 PCE puanı |
| Karşılaştırma | Rastgele bölme aynı modeli 0,481 gösterir (+0,071 iyimserlik; yayın grupları göz ardı edildiğinde) |

## Belirsizlik (conformal, DOI-grup-güvenli kalibrasyon)

- %90 hedef: yarı-genişlik ±6,66; **ampirik kapsama %89,9 (kayıt) / %91,5 (yayın-ağırlıklı)**.
- %80 hedef: yarı-genişlik ±4,86; **ampirik kapsama %79,6 (kayıt) / %81,1 (yayın-ağırlıklı)** — v0.3.x'teki eksik kapsama kanonik gruplamayla büyük ölçüde kapanmıştır.
- Mondrian (tahmin-çeyreği koşullu): kapsama %89,8; çeyrek yarı-genişlikleri ±6,54 / ±6,66 / ±7,41 / ±6,06.
- Group-conformal duyarlılık (DOI değişebilirlik birimi; betik 18): tek-temsilci ±6,48 ve DOI-ağırlıklı ±6,51 varyantları marjinal aralıklarla uyumludur (kapsama %89,1–%89,2 kayıt / %90,6–%90,7 yayın).
- Bu aralıklar **görülmemiş DOI gruplarındaki ampirik kapsama aralıkları** olarak okunmalıdır; kesin kapsama garantisi sunmaz (tez Bölüm 4.10).

## Bilinen sınırlılıklar

1. **Ekstrapolasyon yapamaz:** ağaç tabanlı model; aday tavanı 17,95 < eğitim maksimumu 34,8. Rekor kompozisyon öneremez (tez Bölüm 5.4). Birincil aday listesi v0.4.0'dan itibaren eğitim desteği N≥5 olan (mimari, ETL, HTL) üçlüleriyle sınırlı filtreli listedir (`outputs/candidates_full/primary_designation.json`).
2. **Etiket gürültüsü tabanı (yaklaşık):** tekrar eden reçetelerdeki yayınlar-arası saçılım, hesap varsayımına göre 2,3–3,6 puan aralığında bir taban RMSE ve 0,52–0,81 aralığında bir tavan R² ima eder (homoskedastik kestirim: 3,6 / 0,52; yayın-ağırlıklı: 3,3 / 0,58; kapsam: kayıtların ~%42'si — betik 17, tez Tablo 5.1). Taban kesin bir sınır değil yaklaşık bir kıyas noktasıdır.
3. **Uygulanabilirlik alanı ayrıştırmıyor:** özellik-uzayı kNN mesafesi holdout hatasını ayırt etmedi (MAE iç 3,23, dış 2,89) — mesafe bir güven skoru değildir (tez Bölüm 4.10).
4. **Tek-eklemli rejim:** eğitim verisi tek-eklemli hücre ağırlıklıdır; gözlemli band gap değerlerinin yalnızca %6,2'si ≥1,70 eV. Tandem üst-hücre bölgesi kapsam dışıdır.
5. **DOI anahtarı (v0.4.0'da çözüldü):** v0.3.x'teki 44 büyük/küçük-harf varyantı ana hatta kanonik anahtara birleştirilmiştir; kat/holdout sınırı aşan kayıt kalmamıştır. Ham-anahtar (v0.3.x) davranışı duyarlılık katmanında korunur: GKF R² ham 0,413 ± 0,006; kanonik 0,410 ± 0,016 (tez Bölüm 5.7).
6. **Karşılaştırma modellerinde LightGBM parametre davranışı:** `subsample=0.8` yapılandırılmış olsa da `bagging_freq` verilmediğinden satır alt-örneklemesi LightGBM'de fiilen devre dışıdır; yalnızca kolon alt-örneklemesi (`colsample_bytree=0.8`) etkindir. Arşivlenmiş koşuların tekrar üretilebilirliği için parametre bilerek değiştirilmemiştir (`src/perovskite_ml/models/registry.py` içinde belgelidir; tez Bölüm 3.9). Nihai model (CatBoost) bu durumdan etkilenmez.
7. **CatBoost kurulum notu:** adil karşılaştırma için tüm modeller ortak one-hot sayısal matrisle eğitilir; CatBoost'a `cat_features` verilmez, dolayısıyla yerleşik kategorik hedef istatistikleri devrede değildir. `boosting_type` belirtilmediğinden bu veri boyutunda CPU varsayılanı `Plain` olarak çözümlenir (ampirik olarak doğrulandı); koşuda Ordered boosting kullanılmaz. Bu kurulumda CatBoost'un ayırt edici yönleri simetrik ağaç yapısı ve varsayılan düzenlileştirmedir (tez Bölüm 3.9).

## Kullanım amacı ve kapsam dışı kullanım

- **Amaç:** deneysel taramayı daraltmak için **aday önceliklendirme**; doğrulama-metodolojisi araştırması; eğitim/gösterim (web uygulaması).
- **Kapsam DIŞI:** tek bir hücrenin PCE'sinin kesin tahmini; tandem/çok-eklemli cihazlar; stabilite/toksisite kararları; deneysel doğrulamanın yerine geçen her tür kullanım; nedensel çıkarım (SHAP değerleri gözlemsel ilişkidir).

## Bağlantılar

- Kod: <https://github.com/ismailtopcam/perovskite-pce-ml-thesis> · Arşiv (tüm sürümler): DOI 10.5281/zenodo.20829027
- Etkileşimli uygulama: <https://tez.ismailtopcam.dev>
- Veri kartı: [DATA_CARD.md](DATA_CARD.md)
