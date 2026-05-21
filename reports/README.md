# Rapor Dosyaları

Bu klasör, tez veya sunumda kullanılacak seçilmiş tablo ve görselleri toplamak için ayrılmıştır.

Önerilen yapı:

```text
reports/
├─ figures/
└─ tables/
```

Bu klasör zorunlu değildir. Çıktılar doğrudan `outputs/` klasörlerinden de kullanılabilir.

---

## Kullanılabilecek Nihai Görseller

```text
outputs/v4/actual_vs_predicted_best_model.png
outputs/shap_full/shap_summary_plot.png
outputs/shap_full/shap_top20_bar.png
```

## Kullanılabilecek Nihai Tablolar

```text
outputs/v4/model_comparison_holdout.csv
outputs/v4/model_comparison_groupkfold.csv
outputs/shap_full/shap_top_features.csv
outputs/candidates_full/candidate_prediction_summary.csv
outputs/candidates_full/top_candidates_diverse.csv
```

Bu dosyalar rapor hazırlığı sırasında `reports/figures/` veya `reports/tables/` altına kopyalanabilir. Ancak dosyaların asıl üretildiği yer `outputs/` klasörüdür.
