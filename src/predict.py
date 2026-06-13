from __future__ import annotations
import json
import pandas as pd
import numpy as np
from .utils import MODELS_DIR, load_artifact
def _load_metadata() -> dict:
    return json.loads((MODELS_DIR / "metadata.json").read_text())
def predict_segment(features: dict) -> dict:
    metadata = _load_metadata()
    scaler = load_artifact(MODELS_DIR / "scaler.joblib")
    model = load_artifact(MODELS_DIR / "kmeans.joblib")
    row = pd.DataFrame([features])[metadata["feature_columns"]].copy()
    for col in ["frequency","monetary","total_quantity","avg_order_value","monetary_per_order"]:
        if col in row.columns:
            row[col] = np.log1p(row[col])
    scaled = scaler.transform(row)
    cluster = int(model.predict(scaled)[0])
    return {"cluster": cluster, "segment_name": metadata["segment_name_map"].get(str(cluster), metadata["segment_name_map"].get(cluster, f"Segment {cluster}"))}
