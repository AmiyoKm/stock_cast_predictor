from typing import Any
import joblib
from keras.models import load_model
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "StockCast/seperate"
MODELS_SUBDIR = "models"

_artifact_cache: dict[str, tuple[Any, dict[int, Any]]] = {}


def get_available_trading_codes() -> list[str]:
    """Returns a list of all trading codes for which models are available from the Hugging Face repository."""
    try:
        repo_files = list_repo_files(REPO_ID, repo_type="model")

        trading_codes: set[str] = set()
        for filepath in repo_files:
            if filepath.startswith(f"{MODELS_SUBDIR}/"):
                parts = filepath.split("/")
                if len(parts) > 2:
                    trading_codes.add(parts[1])
        return sorted(list(trading_codes))
    except Exception as e:
        print(f"Could not fetch trading codes from Hugging Face: {e}")
        return []


def load_stock_artifacts(
    trading_code: str,
) -> tuple[Any, dict[int, Any]]:
    if trading_code in _artifact_cache:
        return _artifact_cache[trading_code]

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

    models: dict[int, Any] = {}
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

    _artifact_cache[trading_code] = (scaler, models)

    return scaler, models
