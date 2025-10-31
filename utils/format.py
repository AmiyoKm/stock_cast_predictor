import datetime
from typing import Any


def format_prediction_output(
    last_date: datetime.datetime, prices: list[float], num_days: int
) -> tuple[dict[str, Any], list[str]]:
    prediction_dates = [
        (last_date + datetime.timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(num_days)
    ]

    prediction = {
        "predicted_prices": [round(p, 2) for p in prices],
        "dates": prediction_dates,
        "final_price": round(prices[-1], 2),
    }

    return prediction, prediction_dates
