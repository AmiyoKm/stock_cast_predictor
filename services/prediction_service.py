from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple, Any

from models.stock import Stock
from utils.artifacts import (
    load_stock_artifacts,
    get_available_trading_codes as get_codes,
)
from utils.preprocessing import prepare_data
from config.prediction_config import SUPPORTED_HORIZONS


def inverse_transform_target(arr, scaler):
    """Inverse transforms the target column for a univariate model."""
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


def format_prediction_output(
    last_date: datetime, prices: List[float], num_days: int
) -> Tuple[Dict[str, Any], List[str]]:
    """Formats the prediction output for the API response."""
    prediction_dates = [
        (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(num_days)
    ]

    prediction = {
        "predicted_prices": [round(p, 2) for p in prices],
        "dates": prediction_dates,
        "final_price": round(prices[-1], 2),
    }

    return prediction, prediction_dates


def get_prediction(
    history: List[Stock], trading_code: str, nhead: int
) -> Tuple[Dict, List[str]]:
    """
    Main prediction function that loads artifacts for a specific stock
    and predicts for the requested horizon.
    For 15 and 30-day predictions, it uses the 7-day model iteratively.
    """
    if nhead not in SUPPORTED_HORIZONS:
        raise ValueError(
            f"Unsupported prediction horizon: {nhead}. Must be one of {SUPPORTED_HORIZONS}."
        )

    # Load the specific scaler and models for the given trading code
    scaler, models = load_stock_artifacts(trading_code)

    # For 15 and 30-day predictions, we will use the 7-day model.
    model_horizon = nhead
    if nhead in [15, 30]:
        model_horizon = 7

    # Check if a model for the determined horizon is available
    if model_horizon not in models:
        raise ValueError(
            f"No model available for {trading_code} with a {model_horizon}-day horizon to perform the prediction."
        )

    model = models[model_horizon]

    # Prepare the data using the stock-specific scaler
    input_data = prepare_data(history, scaler)

    all_predicted_prices = []

    if nhead in [15, 30]:
        # --- Iterative Prediction for longer horizons ---
        current_sequence = input_data.copy()
        remaining_days = nhead

        while remaining_days > 0:
            # Predict the next `model_horizon` days
            scaled_prediction = model.predict(
                current_sequence, verbose=0
            )  # Shape: (1, sequence_length, 1)

            # How many days to take from this prediction batch
            days_to_take = min(scaled_prediction.shape[1], remaining_days)

            # Inverse transform the entire prediction to get actual prices
            full_predicted_prices = inverse_transform_target(
                scaled_prediction[0], scaler
            )
            # Then take the slice of prices that are needed for this iteration
            all_predicted_prices.extend(full_predicted_prices[:days_to_take])

            # Update the input sequence for the next iteration
            new_part = scaled_prediction[0][:days_to_take]
            # Roll the sequence to the left
            current_sequence = np.roll(current_sequence, -days_to_take, axis=1)
            # Place the new predictions at the end of the sequence
            current_sequence[0, -days_to_take:, 0] = new_part.flatten()

            remaining_days -= days_to_take

        prices = [max(0.0, float(p)) for p in all_predicted_prices]

    else:
        # Original logic for 1, 3, 7 days
        scaled_prediction = model.predict(input_data, verbose=0).flatten()
        predicted_prices = inverse_transform_target(scaled_prediction, scaler)
        prices = [max(0.0, float(p)) for p in predicted_prices]

    # Format the output
    last_date = history[-1].date
    prediction_key = f"{nhead}_day"
    prediction_output, prediction_dates = format_prediction_output(
        last_date, prices, nhead
    )

    return {prediction_key: prediction_output}, prediction_dates


def is_valid_trading_code(trading_code: str) -> bool:
    available_codes = get_codes()
    return trading_code in available_codes


def get_available_trading_codes(limit: int = 5) -> List[str]:
    return get_codes()[:limit]
