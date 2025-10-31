from typing import Any
import numpy as np


def inverse_transform_seperate_target(arr: Any, scaler: Any):
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


def inverse_transform_unified_target(arr: Any, scaler: Any, n_features: int):
    dummy_features = np.zeros((len(arr), n_features))
    dummy_features[:, 3] = arr.flatten()
    return scaler.inverse_transform(dummy_features)[:, 3]
