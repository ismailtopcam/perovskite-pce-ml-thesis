#!/usr/bin/env python3
"""Kok-dizin saglamlik ve provenans denetimlerini sirayla calistirir.

Kullanim:
  python run_thesis_audits.py

On kosul: cekirdek hattin (python run_all.py) ciktilari mevcut olmalidir;
denetimler outputs/ altindaki holdout tahminleri, aday listeleri ve
temizlenmis veri kumesine dayanir.

Kapsam (tez bolum referanslariyla):
  - pce30_ustu_stack_incelemesi.py   : PCE > %30 uc kayitlarin yigin dizilimi (Bolum 3.4)
  - doi_grup_dogrulama_v2.py         : ham DOI anahtarinda yazim-varyanti denetimi (Bolum 5.7)
  - doi_normalizasyon_duyarlilik.py  : DOI normalizasyonu duyarlilik kosumu (Bolum 5.7)
  - kontamine_test_maesi.py          : sinir asan kayitlarda ek skor-sisme denetimi (Bolum 5.7)
  - doi_makro_mae.py                 : DOI basina makro MAE ozeti (Bolum 4.7)
  - htl_free_yayin_ici.py            : HTL-free etkisinin yayin-ici sinamasi (Bolum 5.3)
  - aday_uyumluluk_filtresi.py       : (mimari, ETL, HTL) birlikte-gorulme denetimi (Bolum 4.5)
  - aday_destek_doi_sayilari.py      : aday desteginin yayin cesitliligi (Bolum 4.5)
  - aday_esik_duyarliligi.py         : destek esigi duyarliligi N>=1/3/10 (Bolum 4.5)
  - final_refit_aday_siralamasi.py   : tum-veri refit ile aday siralamasi denetimi (Bolum 4.5)
  - bandgapsiz_aday_siralamasi.py    : band gap'siz modelle aday siralamasi (Bolum 5.4)

Bir denetim basarisiz olursa (cikis kodu != 0) calisma durur.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

AUDITS = [
    "pce30_ustu_stack_incelemesi.py",
    "doi_grup_dogrulama_v2.py",
    "doi_normalizasyon_duyarlilik.py",
    "kontamine_test_maesi.py",
    "doi_makro_mae.py",
    "htl_free_yayin_ici.py",
    "aday_uyumluluk_filtresi.py",
    "aday_destek_doi_sayilari.py",
    "aday_esik_duyarliligi.py",
    "final_refit_aday_siralamasi.py",
    "bandgapsiz_aday_siralamasi.py",
]


def run(script: str) -> None:
    cmd = [sys.executable, str(ROOT / script)]
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
    t0 = time.time()
    for script in AUDITS:
        run(script)
    print(
        f"\nTUM DENETIMLER BASARIYLA TAMAMLANDI "
        f"({time.time() - t0:.1f} s, {len(AUDITS)} betik)."
    )


if __name__ == "__main__":
    main()
