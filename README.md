# Perovskit PCE Makine Öğrenmesi Tez Kodları

Bu depo, **Perovskit Güneş Hücrelerinde Makine Öğrenmesi Tabanlı PCE Tahmini ve Yüksek Verim Potansiyelli Aday Kompozisyonların Belirlenmesi** başlıklı tez çalışması için geliştirilen Python kodlarını içerir.

Kod seti; Perovskite Database verisinin okunması, DOI bazlı güvenli veri ayrımı, hedef sızıntısı kontrolü, model karşılaştırması, SHAP analizi ve model tabanlı aday kompozisyon üretimi adımlarını kapsar.

Bu depodaki sonuçlar deneysel doğrulama sonucu değildir. Aday kompozisyonlar, eğitilen modelin ürettiği **önceliklendirme çıktıları** olarak değerlendirilmelidir.

---

## 1. Kapsam

Bu kod deposunun amacı şunlardır:

1. Perovskite Database verisini makine öğrenmesi için kullanılabilir hale getirmek.
2. Aynı DOI'ye ait kayıtların eğitim ve test kümelerinde karışmasını önlemek.
3. PCE ile doğrudan veya dolaylı ilişkili ölçüm sonrası kolonların modele girmesini engellemek.
4. Random Forest, XGBoost, LightGBM ve CatBoost modellerini aynı protokolle karşılaştırmak.
5. En iyi model için SHAP açıklanabilirlik analizi yapmak.
6. Model tabanlı aday perovskit kompozisyonları üretmek.
7. Kod geliştirme sürecini tezde izlenebilir ve tekrar üretilebilir hale getirmek.

---

## 2. Klasör Yapısı

```text
perovskite-pce-ml-thesis/
├─ README.md
├─ RESULTS.md
├─ requirements.txt
├─ src/
│  ├─ common.py
│  ├─ 01_v1_basic_ml_models.py
│  ├─ 02_v2_group_safe_models.py
│  ├─ 03_v3_leakage_safe_models.py
│  ├─ 04_v4_model_comparison_report.py
│  ├─ 05_v5_shap_analysis.py
│  └─ 06_v6_candidate_generation.py
├─ outputs/
│  ├─ v4/
│  ├─ shap_full/
│  └─ candidates_full/
├─ reports/
└─ thesis/
```

Tezde kullanılan nihai akışın ana betikleri `src/04_v4_model_comparison_report.py`, `src/05_v5_shap_analysis.py` ve `src/06_v6_candidate_generation.py` dosyalarıdır. V1–V3 betikleri, kod geliştirme sürecindeki ara sürümleri belgelemek için korunmuştur.

---

## 3. Kurulum

Önce terminal veya PowerShell ile proje klasörüne girilir:

```powershell
cd <repo-klasörü>
```

İsteğe bağlı olarak sanal ortam oluşturulabilir:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Gerekli Python paketleri yüklenir:

```powershell
python -m pip install -r requirements.txt
```

Linux/macOS ortamında sanal ortam aktivasyonu farklıdır:

```bash
source .venv/bin/activate
```

---

## 4. Veri Dosyası

Ana veri dosyası büyük olduğu için GitHub deposuna eklenmemelidir. Çalıştırma öncesinde dosya proje kök dizinine konabilir:

```text
Perovskite_database_content_all_data.csv
```

Alternatif olarak `--data` parametresiyle farklı bir dosya yolu verilebilir.

---

## 5. Hızlı Test Çalıştırması

Kodun çalıştığını hızlıca kontrol etmek için sınırlı DOI grubu ve daha düşük ağaç sayısı kullanılabilir:

```powershell
python src/04_v4_model_comparison_report.py --data Perovskite_database_content_all_data.csv --out outputs/quick --max-groups 300 --n-estimators 50 --cv 3
```

Hızlı test modelinden SHAP çıktısı üretmek için:

```powershell
python src/05_v5_shap_analysis.py --data Perovskite_database_content_all_data.csv --model outputs/quick/best_model.joblib --out outputs/shap_quick --max-groups 300
```

Hızlı test modelinden aday kompozisyon üretmek için:

```powershell
python src/06_v6_candidate_generation.py --data Perovskite_database_content_all_data.csv --model outputs/quick/best_model.joblib --out outputs/candidates_quick --max-groups 300
```

Hızlı test sonuçları tezde nihai sonuç olarak değil, kod geliştirme ve hata ayıklama sürecinin parçası olarak değerlendirilmiştir.

---

## 6. Nihai Model Karşılaştırması

Tezde kullanılan nihai model karşılaştırması aşağıdaki komutla yapılmıştır:

```powershell
python src/04_v4_model_comparison_report.py --data Perovskite_database_content_all_data.csv --out outputs/v4 --n-estimators 300 --cv 5
```

Bu aşamada kullanılan protokol:

- hedef değişken: `JV_default_PCE`
- grup değişkeni: `Ref_DOI_number`
- holdout ayrımı: `GroupShuffleSplit`
- çapraz doğrulama: `GroupKFold`
- modeller: RandomForest, XGBoost, LightGBM, CatBoost
- metrikler: MAE, RMSE, R²

Beklenen temel çıktılar:

```text
outputs/v4/model_comparison_holdout.csv
outputs/v4/model_comparison_groupkfold.csv
outputs/v4/best_model.joblib
outputs/v4/best_model_predictions.csv
outputs/v4/actual_vs_predicted_best_model.png
outputs/v4/metadata.json
```

`best_model.joblib` model dosyası büyük olabileceği için GitHub'a eklenmeyebilir. Ancak bu dosya SHAP ve aday üretimi adımlarında kullanılır.

---

## 7. SHAP Analizi

Nihai model karşılaştırması tamamlandıktan sonra en iyi model için SHAP analizi çalıştırılır:

```powershell
python src/05_v5_shap_analysis.py --data Perovskite_database_content_all_data.csv --model outputs/v4/best_model.joblib --out outputs/shap_full
```

Beklenen temel çıktılar:

```text
outputs/shap_full/shap_top_features.csv
outputs/shap_full/shap_summary_plot.png
outputs/shap_full/shap_top20_bar.png
outputs/shap_full/metadata.json
```

Bu çıktılar, modelin PCE tahmini yaparken hangi değişkenleri daha fazla dikkate aldığını yorumlamak için kullanılır.

---

## 8. Aday Kompozisyon Üretimi

Nihai model kullanılarak aday perovskit kompozisyonları aşağıdaki komutla üretilir:

```powershell
python src/06_v6_candidate_generation.py --data Perovskite_database_content_all_data.csv --model outputs/v4/best_model.joblib --out outputs/candidates_full
```

Beklenen temel çıktılar:

```text
outputs/candidates_full/candidate_predictions.csv
outputs/candidates_full/top30_candidate_predictions.csv
outputs/candidates_full/top_candidates_diverse.csv
outputs/candidates_full/candidate_prediction_summary.csv
outputs/candidates_full/prediction_value_counts.csv
outputs/candidates_full/candidate_feature_matrix.csv
outputs/candidates_full/metadata.json
```

Aday kompozisyonlar deneysel olarak doğrulanmış değildir. Bu nedenle sonuçlar kesin verim iddiası olarak değil, model tabanlı önceliklendirme olarak yorumlanmalıdır.

---

## 9. Kod Sürümleri

| Sürüm | Dosya | Amaç |
|---|---|---|
| V1 | `01_v1_basic_ml_models.py` | İlk çalışan model hattı |
| V2 | `02_v2_group_safe_models.py` | DOI bazlı grup güvenli veri ayrımı |
| V3 | `03_v3_leakage_safe_models.py` | Hedef sızıntısı riski taşıyan kolonların dışlanması |
| V4 | `04_v4_model_comparison_report.py` | Nihai model karşılaştırması |
| V5 | `05_v5_shap_analysis.py` | SHAP açıklanabilirlik analizi |
| V6 | `06_v6_candidate_generation.py` | Aday kompozisyon üretimi |

---

## 10. Nihai Sonuç Özeti

Tam veri seti üzerinde yapılan nihai çalıştırmada en iyi model **LightGBM** olmuştur.

Holdout sonucu:

```text
Model : LightGBM
MAE   : 3.124745
RMSE  : 4.015784
R²    : 0.401576
```

GroupKFold çapraz doğrulama sonucu:

```text
Model     : LightGBM
MAE_mean  : 3.078223
RMSE_mean : 3.978972
R²_mean   : 0.425092
R²_std    : 0.027243
```

Nihai aday üretimi sonucu:

```text
Toplam aday sayısı        : 486
Diverse aday sayısı       : 30
En yüksek tahmini PCE     : 18.127339
En yüksek PCE aday sayısı : 2
```

---

## 11. Tezde Kullanım Notu

Bu depo, tezdeki hesaplamalı iş akışının tekrar üretilebilir olduğunu göstermek için hazırlanmıştır.

Quick run çıktıları kod geliştirme ve hata ayıklama süreci için; full run çıktıları ise nihai modelleme sonuçları için kullanılmıştır.

Kod geliştirme sürecinde özellikle şu revizyonlar yapılmıştır:

1. İlk çalışan model kurulmuştur.
2. DOI bazlı grup güvenli veri ayrımı eklenmiştir.
3. Hedef sızıntısı riski taşıyan kolonlar çıkarılmıştır.
4. Model karşılaştırması standart metriklerle yapılmıştır.
5. SHAP analizinde oluşan boyut uyumsuzluğu giderilmiştir.
6. Aday üretiminde kaydedilmiş en iyi modelin doğrudan yüklenmesi sağlanmıştır.
7. Benzer tahmin değerleri için ayrıca diverse aday listesi üretilmiştir.

---

## 12. Uyarı

LightGBM çalıştırmalarında bazı durumlarda şu uyarı görülebilir:

```text
X does not have valid feature names, but LGBMRegressor was fitted with feature names
```

Bu uyarı, pipeline dönüşümünden sonra modele verilen matrisin feature isimleri taşımamasından kaynaklanır. Model eğitimi ve çıktı üretimi tamamlandığı sürece kritik hata olarak değerlendirilmemiştir.
