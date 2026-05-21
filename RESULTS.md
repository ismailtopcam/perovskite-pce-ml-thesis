# Sonuç Özeti

Bu dosya, tezde kullanılan nihai tekrar üretilebilir makine öğrenmesi hattının ana çıktılarını özetler. Sonuçlar `src/04_v4_model_comparison_report.py`, `src/05_v5_shap_analysis.py` ve `src/06_v6_candidate_generation.py` betikleri ile üretilmiştir.

---

## 1. Nihai Modelleme Protokolü

| Alan | Değer |
|---|---|
| Hedef değişken | `JV_default_PCE` |
| Grup değişkeni | `Ref_DOI_number` |
| Veri ayrımı | `GroupShuffleSplit` holdout |
| Çapraz doğrulama | `GroupKFold` |
| Özellik modu | Strict / leakage-safe |
| Karşılaştırılan modeller | RandomForest, XGBoost, LightGBM, CatBoost |
| Temel metrikler | MAE, RMSE, R² |

Leakage riski taşıyabilecek `JV_*`, `EQE_*`, `Stability_*`, `Stabilised_performance_*`, `Outdoor_*` ve `Module_JV_*` kolonları final model girdilerinden çıkarılmıştır.

---

## 2. Holdout Model Performansı

| Model | MAE | RMSE | R² | Durum |
|---|---:|---:|---:|---|
| LightGBM | 3.124745 | 4.015784 | 0.401576 | ok |
| XGBoost | 3.151727 | 4.026685 | 0.398322 | ok |
| CatBoost | 3.183068 | 4.049266 | 0.391555 | ok |
| RandomForest | 3.212309 | 4.165193 | 0.356218 | ok |

Holdout testine göre en iyi model **LightGBM** olmuştur.

---

## 3. GroupKFold Çapraz Doğrulama Sonuçları

| Model | MAE ort. | MAE std | RMSE ort. | RMSE std | R² ort. | R² std | Durum |
|---|---:|---:|---:|---:|---:|---:|---|
| LightGBM | 3.078223 | 0.055052 | 3.978972 | 0.086871 | 0.425092 | 0.027243 | ok |
| XGBoost | 3.108257 | 0.051059 | 3.990629 | 0.078538 | 0.421726 | 0.025987 | ok |
| CatBoost | 3.149350 | 0.057824 | 4.014503 | 0.085222 | 0.414675 | 0.029836 | ok |
| RandomForest | 3.164403 | 0.033868 | 4.148748 | 0.060994 | 0.375106 | 0.021900 | ok |

GroupKFold sonucunda da en iyi ortalama R² değeri **LightGBM** modelinde elde edilmiştir.

---

## 4. SHAP Özellik Önemi

SHAP analizi nihai LightGBM modeli üzerinde çalıştırılmıştır. İlk 20 özellik aşağıdaki gibidir.

| Sıra | Özellik | Ortalama mutlak SHAP |
|---:|---|---:|
| 1 | `num__A_FA` | 1.020681 |
| 2 | `cat__Perovskite_deposition_solvents_DMF; DMSO` | 0.497697 |
| 3 | `cat__HTL_stack_sequence_Spiro-MeOTAD` | 0.349682 |
| 4 | `cat__HTL_stack_sequence_none` | 0.282889 |
| 5 | `num__Perovskite_deposition_thermal_annealing_time` | 0.272339 |
| 6 | `cat__HTL_stack_sequence_PEDOT:PSS` | 0.270927 |
| 7 | `num__B_Pb` | 0.265210 |
| 8 | `cat__Perovskite_deposition_solvents_DMF` | 0.246592 |
| 9 | `num__A_MA` | 0.244076 |
| 10 | `cat__Perovskite_additives_compounds_Unknown` | 0.214571 |
| 11 | `num__X_Br` | 0.190135 |
| 12 | `num__Perovskite_band_gap` | 0.177735 |
| 13 | `cat__ETL_stack_sequence_TiO2-c | TiO2-mp` | 0.159431 |
| 14 | `cat__HTL_stack_sequence_PTAA` | 0.155013 |
| 15 | `num__B_Sn` | 0.145652 |
| 16 | `cat__Perovskite_deposition_solvents_DMF >> IPA` | 0.135754 |
| 17 | `num__Perovskite_thickness` | 0.131358 |
| 18 | `cat__ETL_stack_sequence_PCBM-60` | 0.117268 |
| 19 | `cat__Perovskite_additives_compounds_infrequent_sklearn` | 0.116106 |
| 20 | `cat__Perovskite_deposition_solvents_GBL` | 0.106398 |

Bu sonuçlar, modelin yalnızca A/B/X kompozisyon oranlarına değil; çözücü sistemi, HTL seçimi, tavlama süresi, band gap, kalınlık ve ETL bilgilerine de duyarlı olduğunu göstermektedir.

---

## 5. Aday Kompozisyon Üretimi

Nihai LightGBM modeli ile kontrollü aday uzayı oluşturulmuş ve 486 aday için tahmini PCE değeri hesaplanmıştır.

| Metrik | Değer |
|---|---:|
| Kullanılan model | LGBMRegressor |
| Toplam aday sayısı | 486 |
| Diverse aday sayısı | 30 |
| Benzersiz tahmin sayısı | 252 |
| En yüksek tahmini PCE | 18.127339 |
| En yüksek PCE değerini alan aday sayısı | 2 |
| Ortalama tahmini PCE | 15.515157 |
| Medyan tahmini PCE | 15.693648 |
| Minimum tahmini PCE | 12.483861 |

---

## 6. İlk 10 Diverse Aday

| Sıra | Aday kompozisyon | Mimari | ETL | HTL | Tahmini PCE |
|---:|---|---|---|---|---:|
| 1 | FA0.80MA0.10Cs0.10-Pb1.00Sn0.00-I2.55Br0.45 | pin | SnO2-np | PTAA | 18.127339 |
| 2 | FA0.80MA0.10Cs0.10-Pb1.00Sn0.00-I2.70Br0.30 | pin | SnO2-np | PTAA | 18.127339 |
| 3 | FA0.75MA0.15Cs0.10-Pb1.00Sn0.00-I2.55Br0.45 | pin | SnO2-np | PTAA | 18.036075 |
| 4 | FA0.75MA0.15Cs0.10-Pb1.00Sn0.00-I2.70Br0.30 | pin | SnO2-np | PTAA | 18.036075 |
| 5 | FA0.80MA0.10Cs0.10-Pb1.00Sn0.00-I2.40Br0.60 | pin | SnO2-np | PTAA | 17.991930 |
| 6 | FA0.75MA0.15Cs0.10-Pb1.00Sn0.00-I2.40Br0.60 | pin | SnO2-np | PTAA | 17.892189 |
| 7 | FA0.85MA0.00Cs0.15-Pb1.00Sn0.00-I2.70Br0.30 | pin | SnO2-np | PTAA | 17.825895 |
| 8 | FA0.85MA0.00Cs0.15-Pb1.00Sn0.00-I2.55Br0.45 | pin | SnO2-np | PTAA | 17.825895 |
| 9 | FA0.85MA0.00Cs0.15-Pb1.00Sn0.00-I2.40Br0.60 | pin | SnO2-np | PTAA | 17.330734 |
| 10 | FA0.80MA0.10Cs0.10-Pb0.95Sn0.05-I2.70Br0.30 | nip | SnO2-np | Spiro-MeOTAD | 17.243869 |

---

## 7. Yorum Notu

Aday kompozisyonlar deneysel olarak doğrulanmış performans değerleri değildir. Ağaç tabanlı modellerde benzer aday vektörleri aynı yaprak düğümlere düşebildiği için bazı adaylar aynı tahmini PCE değerini alabilir. Bu nedenle ham sıralı aday listesine ek olarak `top_candidates_diverse.csv` dosyası da üretilmiştir.
