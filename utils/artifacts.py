import joblib
from keras.models import load_model
from huggingface_hub import hf_hub_download, list_repo_files
from typing import Dict, List, Tuple

# The Hugging Face repository ID
REPO_ID = "StockCast/seperate"
MODELS_SUBDIR = "models"

# Cache for loaded artifacts to avoid re-downloading from Hugging Face on every request
_artifact_cache = {}


def get_available_trading_codes() -> List[str]:
    """Returns a list of all trading codes for which models are available from the Hugging Face repository."""
    try:
        # List all files in the models subdirectory of the repository
        repo_files = list_repo_files(REPO_ID, repo_type="model")

        # Extract trading codes from the file paths
        # e.g., from "models/1JANATAMF/scaler_1JANATAMF.bin" we get "1JANATAMF"
        trading_codes = set()
        for filepath in repo_files:
            if filepath.startswith(f"{MODELS_SUBDIR}/"):
                parts = filepath.split("/")
                if len(parts) > 2:
                    trading_codes.add(parts[1])
        return sorted(list(trading_codes))
    except Exception as e:
        # Handle cases where the repo is not found or other API errors
        print(f"Could not fetch trading codes from Hugging Face: {e}")
        return []


def load_stock_artifacts(
    trading_code: str,
) -> Tuple[joblib.load, Dict[int, load_model]]:
    """
    Loads the scaler and LSTM models for a specific trading code from the Hugging Face repository.
    Implements caching to avoid redundant downloads.
    """
    if trading_code in _artifact_cache:
        return _artifact_cache[trading_code]

    # Download the scaler
    try:
        scaler_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{MODELS_SUBDIR}/{trading_code}/scaler_{trading_code}.bin",
            repo_type="model",
        )
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Scaler not found for {trading_code} in {REPO_ID}. Error: {e}"
        )

    # Download models for all supported horizons
    models = {}
    horizons_to_check = [1, 3, 7]

    for horizon in horizons_to_check:
        model_filename = f"{MODELS_SUBDIR}/{trading_code}/lstm_{trading_code}_seq60_nahead{horizon}.keras"
        try:
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=model_filename,
                repo_type="model",
            )
            models[horizon] = load_model(model_path)
        except Exception:
            # It's okay if a model for a specific horizon doesn't exist.
            # We simply won't be able to make predictions for that horizon.
            print(
                f"Model for horizon {horizon} not found for {trading_code}. Skipping."
            )
            pass

    if not models:
        raise FileNotFoundError(f"No models found for {trading_code} in {REPO_ID}")

    # Cache the loaded artifacts
    _artifact_cache[trading_code] = (scaler, models)

    return scaler, models
