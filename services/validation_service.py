from fastapi import HTTPException
from models.stock import Stock
from config.prediction_config import MIN_HISTORY_LENGTH, SUPPORTED_HORIZONS
from services.prediction_service import (
    is_valid_trading_code,
    get_available_trading_codes,
)


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
            detail=f"Unknown trading code: {trading_code}. Available codes: {get_available_trading_codes()}...",
        )


def validate_prediction_horizon(nhead: int) -> None:
    if nhead not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported prediction horizon: {nhead}. Supported values: {SUPPORTED_HORIZONS}",
        )


def validate_prediction_request(history: list[Stock], trading_code: str) -> list[Stock]:
    validate_history_length(history)
    #validate_trading_code(trading_code)`

    sorted_history = sorted(history, key=lambda x: x.date)

    return sorted_history
