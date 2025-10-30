import numpy as np


def prepare_data(stock_history, scaler, sequence_length=60):
    """Prepare data for prediction with an individual stock model."""
    # Extract the single feature ('closep') used by the univariate models
    features = np.array([[s.closep] for s in stock_history])

    # Scale the data
    scaled_features = scaler.transform(features)

    # Create sequence data
    if len(scaled_features) >= sequence_length:
        # Get the last sequence_length points
        sequence = scaled_features[-sequence_length:]
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        return sequence
    else:
        raise ValueError(
            f"Not enough data points. Need at least {sequence_length}, got {len(scaled_features)}"
        )
