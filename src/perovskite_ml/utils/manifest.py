"""Run-manifest: her calistirmanin izlenebilir kaydi (Sculley vd., 2015 — gizli
teknik borca karsi provenans). Hangi config, hangi seed, hangi paket surumleri,
hangi git commit'i (temiz mi?), hangi ciktilar (SHA-256 ile) -> tek JSON."""
from __future__ import annotations
import hashlib, json, subprocess, sys, platform
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_HASH_LIMIT = 100 * 1024 * 1024   # >100 MB dosyalar hash'lenmez (ham veri vb.)

def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm", "catboost", "shap"):
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "yok"
    return out

def _git_info():
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(_ROOT), *a],
                                        capture_output=True, text=True, timeout=10)
        commit = run("rev-parse", "HEAD").stdout.strip()
        dirty = bool(run("status", "--porcelain").stdout.strip())
        return {"commit": commit or "bilinmiyor", "dirty": dirty}
    except Exception:
        return {"commit": "bilinmiyor", "dirty": None}

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def _output_hashes(outputs):
    hashes = {}
    for rel in outputs or []:
        p = _ROOT / rel
        try:
            if p.is_file() and p.stat().st_size <= _HASH_LIMIT:
                hashes[rel] = _sha256(p)
        except OSError:
            pass
    return hashes

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
        "git": _git_info(),
        "metrics": metrics or {},
        "outputs": outputs or [],
        "output_hashes": _output_hashes(outputs),
    }
    p = outdir / f"manifest_{stage}.json"
    json.dump(man, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return str(p)
