from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.predict import predict_segment
app = FastAPI(title="Customer Segmentation API", version="1.0.0")
class CustomerFeatures(BaseModel):
    recency_days: float = Field(..., ge=0)
    frequency: float = Field(..., ge=0)
    monetary: float = Field(..., ge=0)
    total_quantity: float = Field(..., ge=0)
    avg_order_value: float = Field(..., ge=0)
    active_span_days: float = Field(..., ge=0)
    country_count: float = Field(..., ge=0)
    avg_items_per_order: float = Field(..., ge=0)
    monetary_per_order: float = Field(..., ge=0)
@app.get("/")
def healthcheck() -> dict:
    return {"status": "ok", "message": "Customer segmentation API is running."}
@app.post("/predict")
def predict(payload: CustomerFeatures) -> dict:
    return predict_segment(payload.model_dump())
