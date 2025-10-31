from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from config.prediction_config import SUPPORTED_MODELS


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
    history: list[Stock] = Field(
        ..., description="Historical stock data (at least 60 days)"
    )
    model: str = Field(..., description="Model to use for prediction")

    @field_validator("nhead")
    def validate_nhead(cls, v: int):
        if v not in [1, 3, 7, 15, 30]:
            raise ValueError("nhead must be 1, 3, 7, 15, or 30")
        return v

    @field_validator("history")
    def validate_history_length(cls, v: list[Stock]):
        if len(v) < 60:
            raise ValueError(
                f"Not enough history data. Need at least 60 days, got {len(v)}"
            )
        return v

    @field_validator("model")
    def validate_model(cls, v: str):
        if v not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {', '.join(SUPPORTED_MODELS)}")
        return v


class PredictionResult(BaseModel):
    predicted_prices: list[float]
    dates: list[str]
    final_price: float


class PredictionResponse(BaseModel):
    success: bool
    tradingCode: str
    predictions: dict[str, PredictionResult]
    data_points_used: int
    prediction_dates: list[str]
