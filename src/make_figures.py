from __future__ import annotations
import json
import pandas as pd
import matplotlib.pyplot as plt
from .utils import PROCESSED_DIR, MODELS_DIR

ROOT = PROCESSED_DIR.parents[1]
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    segments = pd.read_csv(PROCESSED_DIR / "customer_segments.csv")
    summary = pd.read_csv(PROCESSED_DIR / "cluster_summary.csv")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())

    plt.figure(figsize=(9, 5.5))
    for name, g in segments.groupby("segment_name"):
        plt.scatter(g["recency_days"], g["monetary"], s=18, alpha=.65, label=name)
    plt.xlabel("Recency in days: lower means more recent")
    plt.ylabel("Customer lifetime value in sample")
    plt.title("Customer segments by recency and value")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "segment_value_recency.png", dpi=160)
    plt.close()

    dash = summary[["segment_name", "frequency", "monetary", "avg_order_value"]].copy().sort_values("monetary", ascending=False)
    plt.figure(figsize=(10, 5.8))
    plt.bar(dash["segment_name"], dash["monetary"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Average monetary value")
    plt.title(f"Dashboard preview: segment value overview | silhouette={meta['silhouette_score']}")
    for i, row in enumerate(dash.itertuples(index=False)):
        label = f"freq {row.frequency:.1f}\nAOV {row.avg_order_value:.0f}"
        plt.text(i, row.monetary, label, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dashboard_preview.png", dpi=160)
    plt.close()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
