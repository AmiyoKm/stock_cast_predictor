from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
from datetime import datetime


class Stock(BaseModel):
    id: int
    date: datetime
    tradingCode: str
    ltp: float
    high: float
    low: float
    openp: float
    closep: float
    ycp: float
    trade: int
    value: float
    volume: int


class StockDataRequest(BaseModel):
    tradingCode: str = Field(..., description="Trading code/symbol of the stock")
    nhead: int = Field(
        ..., description="Number of days to predict (1, 3, 7, 15, or 30)"
    )
    history: List[Stock] = Field(
        ..., description="Historical stock data (at least 60 days)"
    )

    @validator("nhead")
    def validate_nhead(cls, v):
        if v not in [1, 3, 7, 15, 30]:
            raise ValueError("nhead must be 1, 3, 7, 15, or 30")
        return v

    @validator("history")
    def validate_history_length(cls, v):
        if len(v) < 60:
            raise ValueError(
                f"Not enough history data. Need at least 60 days, got {len(v)}"
            )
        return v


class PredictionResponse(BaseModel):
    success: bool
    tradingCode: str
    predictions: Dict[str, Any]
    data_points_used: int
    prediction_dates: List[str]
