from src.features import clean_transactions, build_customer_features
import pandas as pd
def test_build_customer_features_basic():
    df = pd.DataFrame(
        {
            "InvoiceNo": ["10001", "10002"],
            "StockCode": ["A", "B"],
            "Description": ["X", "Y"],
            "Quantity": [2, 3],
            "InvoiceDate": ["2011-01-01", "2011-01-10"],
            "UnitPrice": [10.0, 20.0],
            "CustomerID": [1, 1],
            "Country": ["UK", "UK"],
        }
    )
    cleaned = clean_transactions(df)
    features = build_customer_features(cleaned)
    assert len(features) == 1
    assert "monetary" in features.columns
    assert features.loc[0, "frequency"] == 2
