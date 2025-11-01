from models.stock import Stock
from utils.preprocessing import prepare_unified_data
from utils.transform import inverse_transform_unified_target
import numpy as np

from utils.unified_artifacts import load_unified_artifacts


def predict_unified(history: list[Stock], trading_code: str, nhead: int) -> list[float]:
    if nhead in [15, 30]:
        raise ValueError("Unified model not available for nhead=15 or nhead=30.")
    scaler, models, scrip_map = load_unified_artifacts(nhead)
    scrip_id = scrip_map.get(trading_code)
    if scrip_id is None:
        raise ValueError(
            f"Trading code {trading_code} not found in the unified model scrip map."
        )
    model = models[nhead]
    prices = []

    input_data = prepare_unified_data(history, scaler)
    scaled_prediction = model.predict(
        [input_data, np.array([scrip_id]).reshape(1, 1)], verbose=0
    ).ravel()
    predicted_prices = inverse_transform_unified_target(scaled_prediction, scaler, 5)
    prices = [max(0.0, float(p)) for p in predicted_prices]
    return prices
