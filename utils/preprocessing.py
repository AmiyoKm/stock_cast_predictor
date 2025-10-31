from typing import Any
import numpy as np

from models.stock import Stock


def prepare_seperate_data(stock_history: list[Stock], scaler: Any, sequence_length: int = 60):
    """Prepare data for prediction with an individual stock model."""
    # Extract the single feature ('closep') used by the univariate models
    features = np.array([[s.closep] for s in stock_history])

    # Scale the data
    scaled_features = scaler.transform(features)

    if len(scaled_features) >= sequence_length:
        # Get the last sequence_length points
        sequence = scaled_features[-sequence_length:]
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        return sequence
    else:
        raise ValueError(
            f"Not enough data points. Need at least {sequence_length}, got {len(scaled_features)}"
        )


def prepare_unified_data(
    stock_history: list[Stock], scaler: Any, sequence_length: int = 60
):
    """Prepare data for prediction with a unified model."""
    # Extract features used by the unified model
    features = np.array(
        [[s.openp, s.high, s.low, s.closep, s.volume] for s in stock_history]
    )

    # Scale the data
    scaled_features = scaler.transform(features)

    if len(scaled_features) >= sequence_length:
        # Get the last sequence_length points
        sequence = scaled_features[-sequence_length:]
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        return sequence
    else:
        raise ValueError(
            f"Not enough data points. Need at least {sequence_length}, got {len(scaled_features)}"
        )
