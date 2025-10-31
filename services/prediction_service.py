from typing import Any

from models.stock import Stock
from predict.seperate import predict_seperate
from predict.unified import predict_unified
from utils.format import format_prediction_output

from config.prediction_config import SUPPORTED_HORIZONS


def get_prediction(
    history: list[Stock], trading_code: str, nhead: int, model_type: str
) -> tuple[dict[str, Any], list[str]]:
    if nhead not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported prediction horizon: {nhead}. Must be one of {SUPPORTED_HORIZONS}."
        )

    if model_type == "StockCast/seperate":
        prices = predict_seperate(history, trading_code, nhead)
    elif model_type == "StockCast/unified":
        prices = predict_unified(history, trading_code, nhead)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    last_date = history[-1].date
    prediction_key = f"{nhead}_day"
    prediction_output, prediction_dates = format_prediction_output(
        last_date, prices, nhead
    )

    return {prediction_key: prediction_output}, prediction_dates