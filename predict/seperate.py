from models.stock import Stock
from utils.preprocessing import prepare_seperate_data
from utils.seperate_artifacts import (
    load_stock_artifacts,
    get_available_trading_codes as get_codes,
)
from utils.transform import inverse_transform_seperate_target
import numpy as np


def predict_seperate(
    history: list[Stock], trading_code: str, nhead: int
) -> list[float]:
    model_horizon = nhead
    if nhead in [15, 30]:
        model_horizon = 7
    scaler, models = load_stock_artifacts(trading_code)
    model = models[model_horizon]
    input_data = prepare_seperate_data(history, scaler)
    prices = []
    if nhead in [15, 30]:
        current_sequence = input_data.copy()
        remaining_days = nhead
        while remaining_days > 0:
            scaled_prediction = model.predict(current_sequence, verbose=0)
            days_to_take = min(scaled_prediction.shape[1], remaining_days)
            full_predicted_prices = inverse_transform_seperate_target(
                scaled_prediction[0], scaler
            )
            prices.extend(full_predicted_prices[:days_to_take])
            new_part = scaled_prediction[0][:days_to_take]
            current_sequence = np.roll(current_sequence, -days_to_take, axis=1)
            current_sequence[0, -days_to_take:, 0] = new_part.flatten()
            remaining_days -= days_to_take
    else:
        scaled_prediction = model.predict(input_data, verbose=0).flatten()
        predicted_prices = inverse_transform_seperate_target(scaled_prediction, scaler)
        prices = [max(0.0, float(p)) for p in predicted_prices]
    return prices


def get_available_trading_codes(limit: int = 5) -> list[str]:
    return get_codes()[:limit]
