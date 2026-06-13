
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import RAW_DIR, PROCESSED_DIR, save_dataframe

RAW_PATH = RAW_DIR / "online_retail_sample.csv"
FEATURES_PATH = PROCESSED_DIR / "customer_features.csv"
SQLITE_PATH = PROCESSED_DIR / "segmentation.db"


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid sales rows and create line-level revenue.

    The sample intentionally contains a few negative quantities and malformed values
    to make the cleaning step visible during review.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID", "Quantity", "UnitPrice"])
    df = df[df["CustomerID"] > 0]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    invoice_totals = df.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(
        order_revenue=("Revenue", "sum"),
        order_quantity=("Quantity", "sum"),
        order_date=("InvoiceDate", "max"),
    )
    customer_orders = invoice_totals.groupby("CustomerID").agg(
        recency_days=("order_date", lambda s: (snapshot_date - s.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("order_revenue", "sum"),
        total_quantity=("order_quantity", "sum"),
        avg_order_value=("order_revenue", "mean"),
        active_span_days=("order_date", lambda s: max((s.max() - s.min()).days, 0)),
    ).reset_index()
    country_count = df.groupby("CustomerID")["Country"].nunique().rename("country_count").reset_index()
    grouped = customer_orders.merge(country_count, on="CustomerID", how="left")
    grouped["avg_items_per_order"] = grouped["total_quantity"] / grouped["frequency"].clip(lower=1)
    grouped["monetary_per_order"] = grouped["monetary"] / grouped["frequency"].clip(lower=1)
    return grouped.replace([np.inf, -np.inf], np.nan).fillna(0)


def persist_sqlite(df: pd.DataFrame, path: Path) -> None:
    with sqlite3.connect(path) as conn:
        df.to_sql("customer_features", conn, if_exists="replace", index=False)


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    cleaned = clean_transactions(df)
    features = build_customer_features(cleaned)
    save_dataframe(cleaned, PROCESSED_DIR / "clean_transactions.csv")
    save_dataframe(features, FEATURES_PATH)
    persist_sqlite(features, SQLITE_PATH)
    print(f"Saved {len(features):,} customer feature rows to {FEATURES_PATH}")

if __name__ == "__main__":
    main()
