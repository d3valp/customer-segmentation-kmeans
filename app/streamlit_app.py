from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("Customer Segmentation Dashboard")
summary_path = PROCESSED_DIR / "cluster_summary.csv"
segments_path = PROCESSED_DIR / "customer_segments.csv"
metadata_path = MODELS_DIR / "metadata.json"
if not summary_path.exists() or not segments_path.exists():
    st.error("Run feature engineering and training first.")
    st.stop()
summary = pd.read_csv(summary_path)
segments = pd.read_csv(segments_path)
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
col1, col2 = st.columns(2)
with col1:
    st.metric("Customers scored", len(segments))
with col2:
    st.metric("Silhouette score", metadata.get("silhouette_score", "n/a"))
st.subheader("Segment profile table")
st.dataframe(summary, use_container_width=True)
st.subheader("Segment size")
segment_counts = segments["segment_name"].value_counts().reset_index()
segment_counts.columns = ["segment_name", "count"]
fig = px.bar(segment_counts, x="segment_name", y="count", title="Customers per Segment")
st.plotly_chart(fig, use_container_width=True)
st.subheader("Monetary vs Frequency")
fig2 = px.scatter(segments.sample(min(len(segments), 3000), random_state=42), x="frequency", y="monetary", color="segment_name", hover_data=["CustomerID", "recency_days"], title="Customer distribution by segment")
st.plotly_chart(fig2, use_container_width=True)
