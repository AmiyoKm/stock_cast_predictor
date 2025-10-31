import datetime
from models.stock import Stock
from utils.preprocessing import prepare_unified_data
from utils.transform import inverse_transform_unified_target
import numpy as np

from utils.unified_artifacts import load_unified_artifacts


def predict_unified(history: list[Stock], trading_code: str, nhead: int) -> list[float]:
    model_horizon = nhead
    if nhead in [15, 30]:
        model_horizon = 3
    scaler, models, scrip_map = load_unified_artifacts(model_horizon)
    scrip_id = scrip_map.get(trading_code)
    if scrip_id is None:
        raise ValueError(
            f"Trading code {trading_code} not found in the unified model scrip map."
        )
    model = models[model_horizon]
    prices = []
    if nhead in [15, 30]:
        temp_history = history.copy()
        remaining_days = nhead
        while remaining_days > 0:
            input_data = prepare_unified_data(temp_history, scaler)
            scaled_prediction = model.predict(
                [input_data, np.array([scrip_id]).reshape(1, 1)], verbose=0
            ).ravel()
            days_to_take = min(len(scaled_prediction), remaining_days)
            full_predicted_prices = inverse_transform_unified_target(
                scaled_prediction, scaler, 5
            )
            prices.extend(full_predicted_prices[:days_to_take])
            for p in full_predicted_prices[:days_to_take]:
                last_date = temp_history[-1].date
                new_date = last_date + datetime.timedelta(days=1)
                temp_history.append(
                    Stock(
                        date=new_date,
                        openp=p,
                        high=p,
                        low=p,
                        closep=p,
                        volume=0,
                        id=0,
                        tradingCode=trading_code,
                        ltp=p,
                        ycp=p,
                        trade=0,
                        value=0,
                    )
                )
            remaining_days -= days_to_take
    else:
        input_data = prepare_unified_data(history, scaler)
        scaled_prediction = model.predict(
            [input_data, np.array([scrip_id]).reshape(1, 1)], verbose=0
        ).ravel()
        predicted_prices = inverse_transform_unified_target(
            scaled_prediction, scaler, 5
        )
        prices = [max(0.0, float(p)) for p in predicted_prices]
    return prices
