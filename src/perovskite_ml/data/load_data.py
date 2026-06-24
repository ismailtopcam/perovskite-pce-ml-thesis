"""Ham CSV'yi okur ve sadece calismada kullanilacak kolonlari secer.
Yol cozumu saglamdir: config'teki ad bulunamazsa data/raw/ icindeki CSV
otomatik bulunur; calistirma dizininden bagimsiz olarak repo kokune gore arar."""
from pathlib import Path
import pandas as pd
from perovskite_ml.config import project_root

def _resolve_raw_path(cfg: dict) -> Path:
    raw = Path(cfg["paths"]["raw_csv"])
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)        # calistirma dizinine gore
        candidates.append(project_root() / raw)     # repo kokune gore
    for c in candidates:
        if c.exists():
            return c
    # config'teki ad tutmadi -> data/raw/ icindeki .csv'leri tara
    raw_dir = project_root() / "data" / "raw"
    csvs = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if len(csvs) == 1:
        print(f"[load] config'teki ad bulunamadi; data/raw/ icindeki tek CSV kullanildi: {csvs[0].name}")
        return csvs[0]
    if len(csvs) > 1:
        raise FileNotFoundError(
            "data/raw/ icinde birden cok CSV var. config.yaml > paths.raw_csv ile "
            f"hangisi oldugunu belirt: {[c.name for c in csvs]}")
    raise FileNotFoundError(
        "Ham CSV bulunamadi. Asagidakilerden birine koy:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nveya herhangi bir adla data/raw/ klasorune koy (otomatik bulunur).")

def build_usecols(cfg: dict) -> list:
    c = cfg["columns"]
    cols = [cfg["target"], cfg["group_col"], c["module_flag"]]
    cols += list(c["composition"].values())
    cols += list(c["device"])
    cols += list(c["process"])
    seen, out = set(), []
    for x in cols:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def load_raw(cfg: dict) -> pd.DataFrame:
    raw_path = _resolve_raw_path(cfg)
    header = pd.read_csv(raw_path, encoding="utf-8-sig", nrows=0)
    wanted = build_usecols(cfg)
    present = [c for c in wanted if c in header.columns]
    missing = [c for c in wanted if c not in header.columns]
    if missing:
        print(f"[load] UYARI: CSV'de bulunamayan kolonlar atlandi: {missing}")
    df = pd.read_csv(raw_path, encoding="utf-8-sig", usecols=present, low_memory=False)
    print(f"[load] Ham veri: {df.shape[0]} satir, {df.shape[1]} kolon ({raw_path.name})")
    return df
