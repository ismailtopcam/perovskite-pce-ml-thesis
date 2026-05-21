# Çıktı Klasörü

Bu klasör, tez kapsamında çalıştırılan Python betiklerinin ürettiği sonuçları içerir.

Final tez sonuçları için esas klasörler şunlardır:

```text
outputs/v4/
outputs/shap_full/
outputs/candidates_full/
```

`outputs/quick`, `outputs/shap`, `outputs/candidates` ve benzeri klasörler ara deneme veya hata ayıklama sürecinden kalmış olabilir. Nihai tez tabloları için full run çıktıları kullanılmalıdır.

---

## 1. Nihai Model Karşılaştırması Çıktıları

Komut:

```powershell
python src/04_v4_model_comparison_report.py --data Perovskite_database_content_all_data.csv --out outputs/v4 --n-estimators 300 --cv 5
```

Beklenen temel dosyalar:

```text
outputs/v4/model_comparison_holdout.csv
outputs/v4/model_comparison_groupkfold.csv
outputs/v4/best_model.joblib
outputs/v4/best_model_predictions.csv
outputs/v4/actual_vs_predicted_best_model.png
outputs/v4/feature_list.csv
outputs/v4/metadata.json
```

Tezde kullanılan ana dosyalar:

```text
model_comparison_holdout.csv
model_comparison_groupkfold.csv
actual_vs_predicted_best_model.png
metadata.json
```

---

## 2. SHAP Çıktıları

Komut:

```powershell
python src/05_v5_shap_analysis.py --data Perovskite_database_content_all_data.csv --model outputs/v4/best_model.joblib --out outputs/shap_full
```

Beklenen temel dosyalar:

```text
outputs/shap_full/shap_top_features.csv
outputs/shap_full/shap_summary_plot.png
outputs/shap_full/shap_top20_bar.png
outputs/shap_full/metadata.json
```

`shap_top_features.csv` dosyası, özelliklerin ortalama mutlak SHAP değerlerini içerir. `shap_summary_plot.png` ve `shap_top20_bar.png` dosyaları tezde açıklanabilirlik grafikleri olarak kullanılabilir.

---

## 3. Aday Kompozisyon Çıktıları

Komut:

```powershell
python src/06_v6_candidate_generation.py --data Perovskite_database_content_all_data.csv --model outputs/v4/best_model.joblib --out outputs/candidates_full
```

Beklenen temel dosyalar:

```text
outputs/candidates_full/candidate_predictions.csv
outputs/candidates_full/top30_candidate_predictions.csv
outputs/candidates_full/top_candidates_diverse.csv
outputs/candidates_full/candidate_prediction_summary.csv
outputs/candidates_full/prediction_value_counts.csv
outputs/candidates_full/candidate_feature_matrix.csv
outputs/candidates_full/metadata.json
```

Tezde özellikle şu iki dosya kullanılmıştır:

```text
candidate_prediction_summary.csv
top_candidates_diverse.csv
```

---

## 4. GitHub Notu

Ham veri dosyaları, eğitilmiş model dosyaları ve büyük ara çıktılar GitHub'a eklenmeyebilir. Özellikle şu dosyalar dışarıda bırakılabilir:

```text
Perovskite_database_content_all_data.csv
*.joblib
__pycache__/
outputs/quick/
```

Nihai sonuçları belgeleyen küçük CSV ve görsel dosyalar ise repoda tutulabilir.
