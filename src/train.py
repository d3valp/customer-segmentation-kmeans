from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .utils import PROCESSED_DIR, MODELS_DIR, dump_artifact
FEATURES_PATH = PROCESSED_DIR / "customer_features.csv"
FEATURE_COLUMNS = ["recency_days","frequency","monetary","total_quantity","avg_order_value","active_span_days","country_count","avg_items_per_order","monetary_per_order"]
def assign_segment_names(summary: pd.DataFrame) -> dict[int, str]:
    ranked = summary.copy()
    ranked["score"] = ranked["frequency_rank"] + ranked["monetary_rank"] + ranked["recency_rank"]
    ordered = ranked.sort_values("score", ascending=False)["cluster"].tolist()
    candidate_names = ["Champions","Loyal High-Value","Promising Regulars","Low Engagement","At-Risk / Cooling","Occasional Buyers"]
    mapping = {}
    for cluster_id, label in zip(ordered, candidate_names):
        mapping[int(cluster_id)] = label
    for missing in set(summary["cluster"].tolist()) - set(mapping):
        mapping[int(missing)] = f"Segment {missing}"
    return mapping
def main(n_clusters: int = 5) -> None:
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_COLUMNS].copy()
    for col in ["frequency","monetary","total_quantity","avg_order_value","monetary_per_order"]:
        X[col] = np.log1p(X[col])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    result = df.copy()
    result["cluster"] = labels
    summary = result.groupby("cluster")[["recency_days","frequency","monetary","avg_order_value"]].mean().reset_index()
    summary["recency_rank"] = summary["recency_days"].rank(ascending=False, method="dense")
    summary["frequency_rank"] = summary["frequency"].rank(ascending=True, method="dense")
    summary["monetary_rank"] = summary["monetary"].rank(ascending=True, method="dense")
    name_map = assign_segment_names(summary)
    result["segment_name"] = result["cluster"].map(name_map)
    summary["segment_name"] = summary["cluster"].map(name_map)
    silhouette = silhouette_score(X_scaled, labels)
    result.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "cluster_summary.csv", index=False)
    dump_artifact(scaler, MODELS_DIR / "scaler.joblib")
    dump_artifact(model, MODELS_DIR / "kmeans.joblib")
    metadata = {"feature_columns": FEATURE_COLUMNS, "n_clusters": n_clusters, "silhouette_score": round(float(silhouette), 4), "segment_name_map": name_map}
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
if __name__ == "__main__":
    main()
