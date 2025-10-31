from fastapi import HTTPException
from models.stock import Stock
from config.prediction_config import (
    MIN_HISTORY_LENGTH,
    SUPPORTED_HORIZONS,
    SUPPORTED_MODELS,
)
from utils.seperate_artifacts import get_available_trading_codes as get_codes


def validate_history_length(
    history: list[Stock], min_length: int = MIN_HISTORY_LENGTH
) -> None:
    if len(history) < min_length:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough historical data. Need at least {min_length} days, got {len(history)}",
        )


def validate_trading_code(trading_code: str) -> None:
    if not is_valid_trading_code(trading_code):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown trading code: {trading_code}. Available codes: {get_codes()}...",
        )


def validate_prediction_horizon(nhead: int) -> None:
    if nhead not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported prediction horizon: {nhead}. Supported values: {SUPPORTED_HORIZONS}",
        )


def validate_model(model: str) -> None:
    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}. Supported models: {', '.join(SUPPORTED_MODELS)}",
        )


def validate_prediction_request(history: list[Stock]) -> list[Stock]:
    validate_history_length(history)

    sorted_history = sorted(history, key=lambda x: x.date)

    return sorted_history

def is_valid_trading_code(trading_code: str) -> bool:
    available_codes = get_codes()
    return trading_code in available_codes
