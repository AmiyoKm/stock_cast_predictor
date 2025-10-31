from fastapi import APIRouter, HTTPException
from models.stock import StockDataRequest, PredictionResponse
from services.prediction_service import get_prediction
from services.validation_service import (
    validate_model,
    validate_prediction_request,
    validate_trading_code,
    validate_prediction_horizon,
)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_stock_prices(request: StockDataRequest) -> PredictionResponse:
    try:
        validate_prediction_horizon(request.nhead)
        validate_trading_code(request.tradingCode)
        validate_model(request.model)

        history = validate_prediction_request(request.history)

        predictions, prediction_dates = get_prediction(
            history, request.tradingCode, request.nhead, model_type=request.model
        )

        return PredictionResponse(
            success=True,
            tradingCode=request.tradingCode,
            predictions=predictions,
            data_points_used=len(history),
            prediction_dates=prediction_dates,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during prediction."
        )
