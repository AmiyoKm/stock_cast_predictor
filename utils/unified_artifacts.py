import joblib
import json
from keras.models import load_model
from huggingface_hub import hf_hub_download
from typing import Any

UNIFIED_REPO_ID = "StockCast/unified"

_unified_artifact_cache: dict[int, tuple[Any, dict[int, Any], dict[str, int]]] = {}


def load_unified_artifacts(
    nhead: int,
) -> tuple[Any, dict[int, Any], dict[str, int]]:
    if nhead in _unified_artifact_cache:
        return _unified_artifact_cache[nhead]

    # Load the unified scaler
    try:
        scaler_path = hf_hub_download(
            repo_id=UNIFIED_REPO_ID,
            filename="global_scaler.bin",
            repo_type="model",
        )
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Unified scaler not found in {UNIFIED_REPO_ID}. Error: {e}"
        )

    # Load the scrip to id mapping
    try:
        scrip_map_path = hf_hub_download(
            repo_id=UNIFIED_REPO_ID,
            filename="scrip_to_id.json",
            repo_type="model",
        )
        with open(scrip_map_path, "r") as f:
            scrip_map = json.load(f)
    except Exception as e:
        raise FileNotFoundError(
            f"Scrip map not found in {UNIFIED_REPO_ID}. Error: {e}"
        )

    models: dict[int, Any] = {}
    model_filename = f"unified_lstm_nahead{nhead}.keras"
    try:
        model_path = hf_hub_download(
            repo_id=UNIFIED_REPO_ID,
            filename=model_filename,
            repo_type="model",
        )
        models[nhead] = load_model(model_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Model for horizon {nhead} not found in {UNIFIED_REPO_ID}. Error: {e}"
        )

    _unified_artifact_cache[nhead] = (scaler, models, scrip_map)

    return scaler, models, scrip_map
