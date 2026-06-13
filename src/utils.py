from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
for _dir in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
def dump_artifact(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
def load_artifact(path: Path):
    return joblib.load(path)
