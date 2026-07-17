# Veri Kartı — Perovskite Database Anlık Görüntüsü

*Biçim: Gebru vd. (2021) "Datasheets for Datasets" uyarlaması.*
*Son güncelleme: Temmuz 2026 · İsmailTopçam_Tez_2026 ile uyumlu.*

## Kaynak ve lisans

- **Kaynak:** The Perovskite Database Project — <https://www.perovskitedatabase.com>
  (Jacobsson, T. J., vd. (2022). An open-access database and analysis tool for
  perovskite solar cells based on the FAIR data principles. *Nature Energy*, 7, 107–115.
  DOI: 10.1038/s41560-021-00941-3)
- **Veri lisansı:** CC BY 4.0 — kullanım ve türetme serbesttir, **atıf zorunludur**.
- **Yeniden dağıtım kararı:** ham CSV bu depoda yeniden dağıtılmaz (`.gitignore`);
  kullanıcı özgün kaynaktan indirir, README'deki SHA-256 ile doğrular. Web
  uygulamasında gösterilen kayıt-düzeyi alanlar CC BY 4.0 atfıyla sunulur.
- **Önemli ayrım:** bu deponun MIT lisansı **yalnızca kodu** kapsar; veri setinin
  lisans ve kullanım koşullarını kapsamaz.

## Anlık görüntü kimliği

| Alan | Değer |
|---|---|
| Boyut | 43.398 kayıt × 410 kolon (~87 MB) |
| SHA-256 | `da66a634e9106e58ce4d012558d468bc2f19b95987f149fb9c5d208d363ea67a` |
| Not | Veritabanı yaşayan bir kaynaktır; farklı bir anlık görüntü kayıt sayılarını ve türetilen tüm sonuçları değiştirebilir. Tezdeki her sayı yukarıdaki görüntüye aittir. |

## Temizleme kararları (tez Bölüm 3.4; kayıt-düzeyi log: `outputs/logs/removed_records_log.csv`)

| Neden | Kayıt |
|---|---|
| PCE sayısal değil / eksik | 1.002 |
| Modül kaydı (tek hücre değil) | 344 |
| Katsayı sayısal değil | 291 |
| İyon–katsayı sayısı uyuşmuyor | 141 |
| İyon/katsayı bilgisi eksik | 130 |
| PCE aralık dışı (0–35; sınırlar dahil) | 5 |
| **Toplam elenen** | **1.913** → temiz küme **41.485** |

## Eksiklik ve doldurma (medyan + eksiklik bayrağı)

- Band gap: %23,9 · Tavlama süresi: %11,5 · Tavlama sıcaklığı: %8,9.
- Doldurma istatistikleri kat-değişmezidir (band gap medyanı 5 katın 5'inde 1,6 eV — tez Bölüm 3.7/4.9 çevresi, `outputs/robustness/fold_invariance.csv`).

## Bilinen aykırılıklar ve tuhaflıklar

- **PCE > 30 olan 8 kayıt** (4 yayından; sertifikasız; medyan alan 0,1 cm²): alan
  rekorlarıyla ve bildirilen cihaz yapılarıyla uyumsuz görünen, güvenilirliği şüpheli
  uç kayıtlar (tandem izi yok — `pce30_ustu_stack_incelemesi.py`). Dışlanmaları ana
  sonucu değiştirmez (tez Bölüm 4.2).
- **DOI anahtarı:** 44 yayın büyük/küçük-harf varyantıyla çift kayıtlı (362 kayıt,
  %0,87 kat-sınırı aşımı). Ölçülmüş kalıntı risk; duyarlılık koşumu farkı kat
  gürültüsü içinde (`outputs/robustness/doi_grup_dogrulama.json`).
- **Çözücü 'unknown' / 'Unknown':** boş bırakılan 279 kayıt ile 'Unknown' yazılmış
  1.044 kayıt iki ayrı kategoridir — belgelenmiş bilinçli karar (tez Bölüm 5.7).

## Temsil yanlılıkları (modelin öğrendiği dağılım)

- Yayın başına kayıt sağa çarpık: medyan 4, p95 14, maks 127; DOI'siz 201 kayıt tek
  yapay grupta (7.397 grup) — DOI-grup-güvenli bölmenin gerekçesi.
- Cihaz yığını yoğunlaşması: n-i-p mimari, TiO₂ tabanlı ETL ve Spiro-MeOTAD baskın;
  HTL-free 2.494 kayıt (647 yayın; medyan PCE 8,0 — genel 12,77).
- Tek-eklemli rejim baskın; gözlemli band gap p95 = 1,75 eV, ≥1,70 eV payı %6,2.
- Özdeş reçeteler yayınlar arasında ~2,6–3,0 puanlık PCE saçılımı taşır (etiket
  gürültüsü; tez Tablo 5.1) — her modelin pratik başarım tavanını sınırlar.

## Bağlantılar

- Model kartı: [MODEL_CARD.md](MODEL_CARD.md)
- Etkileşimli keşif: <https://tez.ismailtopcam.dev> (Veri Hattı ve DOI Gezgini sayfaları)
