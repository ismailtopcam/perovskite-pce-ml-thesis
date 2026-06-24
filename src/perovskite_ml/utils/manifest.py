"""Run-manifest: her calistirmanin izlenebilir kaydi (Sculley vd., 2015 — gizli
teknik borca karsi provenans). Hangi config, hangi seed, hangi paket surumleri,
kac satir/ozellik, hangi ciktilar -> tek JSON."""
from __future__ import annotations
import json, sys, platform
from datetime import datetime, timezone
from pathlib import Path

def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm", "catboost", "shap"):
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "yok"
    return out

def write_manifest(outdir, stage, cfg, metrics=None, outputs=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    man = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg.get("seed"),
        "target": cfg.get("target"),
        "group_col": cfg.get("group_col"),
        "onehot_top_n": cfg.get("features", {}).get("onehot_top_n"),
        "versions": _versions(),
        "metrics": metrics or {},
        "outputs": outputs or [],
    }
    p = outdir / f"manifest_{stage}.json"
    json.dump(man, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return str(p)
