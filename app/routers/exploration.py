"""Router para análisis exploratorio de series temporales."""
from fastapi import APIRouter, HTTPException

from app.models.schemas import TimeSeriesData, ExplorationResponse
from app.services.exploration_service import explore_series

router = APIRouter(prefix="/exploration", tags=["Exploración"])


@router.post(
    "/analyze",
    response_model=ExplorationResponse,
    summary="Análisis exploratorio completo",
    description=(
        "Recibe una serie temporal y devuelve: test de estacionariedad (ADF), "
        "descomposición STL (tendencia/estacionalidad/residuo), funciones ACF y PACF, "
        "y estadísticas descriptivas."
    ),
)
def analyze_series(data: TimeSeriesData) -> ExplorationResponse:
    try:
        return explore_series(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en análisis exploratorio: {str(e)}",
        )
