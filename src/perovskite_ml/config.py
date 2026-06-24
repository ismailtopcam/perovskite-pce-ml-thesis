"""config.yaml yukleyici. Tum parametreler buradan okunur."""
from pathlib import Path
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def project_root() -> Path:
    # config.py -> perovskite_ml -> src -> ROOT
    return Path(__file__).resolve().parents[2]
