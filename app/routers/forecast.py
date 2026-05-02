"""Router para entrenamiento y pronóstico con ARIMA/SARIMA."""
from fastapi import APIRouter, HTTPException

from app.models.schemas import ForecastRequest, ForecastResponse
from app.services.forecast_service import fit_and_forecast

router = APIRouter(prefix="/forecast", tags=["Pronóstico"])


@router.post(
    "",
    response_model=ForecastResponse,
    summary="Entrenar modelo y generar pronóstico",
    description=(
        "Recibe una serie temporal y la configuración del modelo, "
        "entrena ARIMA o SARIMA (manual o automático), evalúa sobre conjunto de prueba, "
        "y devuelve pronóstico futuro con intervalos de confianza."
    ),
)
def create_forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        return fit_and_forecast(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al entrenar el modelo: {str(e)}",
        )
