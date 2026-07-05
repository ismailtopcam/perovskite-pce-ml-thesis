#!/usr/bin/env python3
"""Tek komutla tum pipeline'i sirayla calistirir.

Kullanim:
  python run_all.py          # cekirdek hat (01-10)
  python run_all.py --all    # + ek dogrulama/saglamlik betikleri (11-14 + thesis_data_analysis)

Her adim bir onceki adimin ciktilarina dayanir; bir adim basarisiz olursa
(cikis kodu != 0) calisma durur. Betikler config.yaml ve data/ yollarini
kendileri okudugundan ek arguman gerekmez. Calismayi depo kokunden baslatir.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

CORE = [
    "01_prepare_data.py",
    "02_build_features.py",
    "03_compare_models.py",
    "04_validation_experiments.py",
    "05_shap_analysis.py",
    "06_generate_candidates.py",
    "07_screening_utility.py",
    "08_hyperparameter_tuning.py",
    "09_descriptor_ablation.py",
    "10_conformal_uncertainty.py",
]

SUPP = [
    "11_pce_outlier_audit.py",
    "12_catboost_tuning.py",
    "13_preprocessing_and_bandgap.py",
    "14_pipeline_cv.py",
]


def run(script: str, args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script)] + (args or [])
    print(f"\n{'=' * 72}\n>>> {script}\n{'=' * 72}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(
            f"\nHATA: {script} cikis kodu {result.returncode} ile durdu. "
            f"Calisma durduruldu."
        )
    print(f"--- {script} tamamlandi ({time.time() - start:.1f} s) ---", flush=True)


def main() -> None:
    run_supp = "--all" in sys.argv
    steps = CORE + (SUPP if run_supp else [])
    t0 = time.time()
    for script in steps:
        run(script)
    if run_supp:
        model_ready = ROOT / "data" / "processed" / "model_ready_dataset.csv"
        run("thesis_data_analysis.py", [str(model_ready)])
        run("thesis_figures.py")
    n = len(steps) + (2 if run_supp else 0)
    print(
        f"\nTUM ADIMLAR BASARIYLA TAMAMLANDI "
        f"({time.time() - t0:.1f} s, {n} betik)."
    )


if __name__ == "__main__":
    main()
