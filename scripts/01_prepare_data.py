"""Stage 1 — Veri hazirlama runner'i.
Akis: ham CSV -> satir temizligi -> A/B/X kompozisyon vektorizasyonu -> kaydet.
Cikti:
  data/processed/cleaned_dataset.csv
  data/processed/composition_vectorized.csv
  outputs/logs/removed_records_log.csv
  outputs/logs/removed_reason_counts.csv
  outputs/logs/clean_summary.json
Calistirma (repo kokunden):
  python scripts/01_prepare_data.py
"""
import sys, json
from pathlib import Path

# src/ paketini yola ekle
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perovskite_ml.config import load_config
from perovskite_ml.data.load_data import load_raw
from perovskite_ml.data.clean_data import clean_rows
from perovskite_ml.features.composition import vectorize_dataframe
from perovskite_ml.validation.schema import validate_clean
from perovskite_ml.utils.manifest import write_manifest

def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    proc = ROOT / cfg["paths"]["processed_dir"]
    logs = ROOT / cfg["paths"]["outputs_dir"] / "logs"
    proc.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    df = load_raw(cfg)
    n_raw = len(df)

    cleaned, removed_log = clean_rows(df, cfg)
    n_clean = len(cleaned)
    validate_clean(cleaned, cfg)   # veri sozlesmesi: hedef sayisal & aralikta
    cleaned.to_csv(proc / "cleaned_dataset.csv", index=False)

    vec = vectorize_dataframe(cleaned, cfg)
    vec.to_csv(proc / "composition_vectorized.csv", index=False)

    write_manifest(ROOT / cfg["paths"]["outputs_dir"] / "manifests", "stage1_clean", cfg,
                   metrics={"n_clean": int(len(cleaned))}, outputs=["data/processed/cleaned_dataset.csv"])
    removed_log.to_csv(logs / "removed_records_log.csv", index=False)
    reason_counts = removed_log["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    reason_counts.to_csv(logs / "removed_reason_counts.csv", index=False)

    pce = cleaned[cfg["target"]]
    summary = {
        "raw_rows": int(n_raw),
        "removed_total": int(len(removed_log)),
        "clean_rows": int(n_clean),
        "removed_by_reason": reason_counts.set_index("reason")["count"].to_dict(),
        "pce_mean": round(float(pce.mean()), 4),
        "pce_median": round(float(pce.median()), 4),
        "pce_std": round(float(pce.std()), 4),
        "n_unique_doi": int(cleaned[cfg["group_col"]].nunique()),
        "vectorized_columns": [c for c in vec.columns if c.split("_")[0] in ("A","B","X")],
    }
    with open(logs / "clean_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================= STAGE 1 RESULTS =================")
    print(f"Ham satir            : {n_raw}")
    print(f"Cikarilan (toplam)   : {len(removed_log)}")
    print(f"Temiz satir          : {n_clean}")
    print(f"Cikarma nedenleri    : {summary['removed_by_reason']}")
    print(f"PCE  mean/median/std : {summary['pce_mean']} / {summary['pce_median']} / {summary['pce_std']}")
    print(f"Benzersiz DOI        : {summary['n_unique_doi']}")
    print(f"Kompozisyon sutunlari: {summary['vectorized_columns']}")
    print("===================================================")

if __name__ == "__main__":
    main()
