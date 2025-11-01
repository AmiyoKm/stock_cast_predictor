from models.stock import Stock
from utils.preprocessing import prepare_seperate_data
from utils.seperate_artifacts import (
    load_stock_artifacts,
    get_available_trading_codes as get_codes,
)
from utils.transform import inverse_transform_seperate_target


def predict_seperate(
    history: list[Stock], trading_code: str, nhead: int
) -> list[float]:
    model_horizon = nhead
    scaler, models = load_stock_artifacts(trading_code)
    model = models[model_horizon]
    input_data = prepare_seperate_data(history, scaler)
    prices = []
    scaled_prediction = model.predict(input_data, verbose=0).flatten()
    predicted_prices = inverse_transform_seperate_target(scaled_prediction, scaler)
    prices = [max(0.0, float(p)) for p in predicted_prices]
    return prices


def get_available_trading_codes(limit: int = 5) -> list[str]:
    return get_codes()[:limit]
