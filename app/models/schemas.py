"""
Esquemas Pydantic.

Definen el "contrato" de datos entre el frontend (React/TypeScript) y el backend.
Cualquier cambio aquí debe reflejarse en los tipos del frontend.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ============================================================
# Entrada: serie temporal que envía el frontend
# ============================================================

class TimeSeriesPoint(BaseModel):
    """Un punto de la serie: fecha (ISO string) + valor."""
    date: str = Field(..., description="Fecha en formato ISO 8601, ej: '2017-01-15'")
    value: float = Field(..., description="Valor numérico de la observación")


class TimeSeriesData(BaseModel):
    """Serie temporal completa enviada por el frontend."""
    points: list[TimeSeriesPoint] = Field(..., min_length=10)
    frequency: Literal["D", "W", "M", "Q"] = Field(
        ...,
        description="Frecuencia: D=diaria, W=semanal, M=mensual, Q=trimestral"
    )


# ============================================================
# Análisis exploratorio
# ============================================================

class StationarityResult(BaseModel):
    """Resultado del test de Augmented Dickey-Fuller."""
    statistic: float
    p_value: float
    is_stationary: bool = Field(..., description="True si p_value < 0.05")
    critical_values: dict[str, float]
    interpretation: str


class DecompositionResult(BaseModel):
    """Descomposición de la serie en tendencia, estacionalidad y residuo."""
    trend: list[Optional[float]]
    seasonal: list[Optional[float]]
    residual: list[Optional[float]]
    dates: list[str]


class AcfPacfResult(BaseModel):
    """Función de autocorrelación y autocorrelación parcial."""
    acf: list[float]
    pacf: list[float]
    confidence_interval: float = Field(..., description="Banda de significancia (±)")
    lags: list[int]


class ExplorationResponse(BaseModel):
    """Respuesta completa del análisis exploratorio."""
    stationarity: StationarityResult
    decomposition: DecompositionResult
    acf_pacf: AcfPacfResult
    summary_stats: dict[str, float]


# ============================================================
# Configuración del modelo
# ============================================================

class ArimaOrder(BaseModel):
    """Parámetros (p, d, q) de ARIMA."""
    p: int = Field(..., ge=0, le=10)
    d: int = Field(..., ge=0, le=3)
    q: int = Field(..., ge=0, le=10)


class SeasonalOrder(BaseModel):
    """Parámetros estacionales (P, D, Q, s) de SARIMA."""
    P: int = Field(..., ge=0, le=5)
    D: int = Field(..., ge=0, le=2)
    Q: int = Field(..., ge=0, le=5)
    s: int = Field(..., ge=2, description="Período estacional: 7=semanal, 12=mensual, 4=trimestral")


class ForecastRequest(BaseModel):
    """Solicitud de pronóstico."""
    series: TimeSeriesData
    model_type: Literal["ARIMA", "SARIMA", "AUTO"] = "AUTO"
    arima_order: Optional[ArimaOrder] = None
    seasonal_order: Optional[SeasonalOrder] = None
    train_size: float = Field(0.8, gt=0.5, lt=1.0, description="Proporción para entrenamiento")
    forecast_horizon: int = Field(12, ge=1, le=365, description="Períodos a pronosticar")
    confidence_level: float = Field(0.95, gt=0.5, lt=1.0)


# ============================================================
# Resultados del pronóstico
# ============================================================

class ForecastMetrics(BaseModel):
    """Métricas de evaluación del modelo sobre el conjunto de prueba."""
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error (%)")
    aic: float = Field(..., description="Akaike Information Criterion")
    bic: float = Field(..., description="Bayesian Information Criterion")


class ForecastPoint(BaseModel):
    """Un punto pronosticado con su intervalo de confianza."""
    date: str
    forecast: float
    lower_bound: float
    upper_bound: float


class ResidualDiagnostics(BaseModel):
    """Diagnóstico de residuos."""
    ljung_box_p_value: float
    is_white_noise: bool = Field(..., description="True si los residuos parecen ruido blanco")
    residuals: list[float]
    interpretation: str


class ForecastResponse(BaseModel):
    """Respuesta completa del pronóstico."""
    model_type_used: str
    arima_order: list[int] = Field(..., description="[p, d, q] usados")
    seasonal_order: Optional[list[int]] = Field(None, description="[P, D, Q, s] si aplica")
    metrics: ForecastMetrics
    train_data: list[TimeSeriesPoint]
    test_data: list[TimeSeriesPoint]
    test_predictions: list[float] = Field(..., description="Predicciones sobre el set de prueba")
    forecast: list[ForecastPoint] = Field(..., description="Pronóstico futuro con bandas")
    residual_diagnostics: ResidualDiagnostics


# ============================================================
# Errores
# ============================================================

class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None
