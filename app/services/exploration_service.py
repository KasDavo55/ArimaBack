"""
Servicio de análisis exploratorio de series temporales.

Implementa:
- Test de estacionariedad (Augmented Dickey-Fuller).
- Descomposición STL (tendencia + estacionalidad + residuo).
- ACF y PACF para identificación de parámetros.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from app.models.schemas import (
    StationarityResult,
    DecompositionResult,
    AcfPacfResult,
    ExplorationResponse,
    TimeSeriesData,
)


# Mapeo de frecuencias a períodos estacionales por defecto
SEASONAL_PERIODS = {
    "D": 7,    # Diaria → estacionalidad semanal
    "W": 52,   # Semanal → estacionalidad anual
    "M": 12,   # Mensual → estacionalidad anual
    "Q": 4,    # Trimestral → estacionalidad anual
}


def _series_to_pandas(data: TimeSeriesData) -> pd.Series:
    """Convierte el formato de entrada a una Serie de pandas con índice de fechas."""
    dates = pd.to_datetime([p.date for p in data.points])
    values = [p.value for p in data.points]
    series = pd.Series(values, index=dates)
    series = series.sort_index()
    return series


def run_stationarity_test(series: pd.Series) -> StationarityResult:
    """
    Ejecuta el test de Augmented Dickey-Fuller.

    H0 (hipótesis nula): la serie tiene raíz unitaria → NO estacionaria.
    Si p_value < 0.05 → rechazamos H0 → la serie ES estacionaria.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    statistic = float(result[0])
    p_value = float(result[1])
    critical_values = {k: float(v) for k, v in result[4].items()}

    is_stationary = p_value < 0.05

    if is_stationary:
        interpretation = (
            f"La serie ES estacionaria (p-value={p_value:.4f} < 0.05). "
            "No requiere diferenciación adicional para ARIMA."
        )
    else:
        interpretation = (
            f"La serie NO es estacionaria (p-value={p_value:.4f} >= 0.05). "
            "Se recomienda aplicar diferenciación (parámetro d>=1 en ARIMA)."
        )

    return StationarityResult(
        statistic=statistic,
        p_value=p_value,
        is_stationary=is_stationary,
        critical_values=critical_values,
        interpretation=interpretation,
    )


def run_decomposition(
    series: pd.Series, frequency: str
) -> DecompositionResult:
    """
    Descompone la serie en tendencia + estacionalidad + residuo.

    Usa modelo aditivo: Y(t) = Tendencia(t) + Estacional(t) + Residuo(t).
    """
    period = SEASONAL_PERIODS.get(frequency, 12)

    # Necesitamos al menos 2 períodos completos
    if len(series) < 2 * period:
        period = max(2, len(series) // 2)

    decomposition = seasonal_decompose(
        series,
        model="additive",
        period=period,
        extrapolate_trend="freq",
    )

    def _to_list(arr) -> list:
        return [None if pd.isna(v) else float(v) for v in arr]

    return DecompositionResult(
        trend=_to_list(decomposition.trend),
        seasonal=_to_list(decomposition.seasonal),
        residual=_to_list(decomposition.resid),
        dates=[d.strftime("%Y-%m-%d") for d in series.index],
    )


def run_acf_pacf(series: pd.Series, n_lags: int = 30) -> AcfPacfResult:
    """
    Calcula ACF y PACF para ayudar a identificar parámetros (p, q) de ARIMA.

    Reglas de interpretación:
    - PACF se corta en lag p → sugiere AR(p)
    - ACF se corta en lag q → sugiere MA(q)
    """
    clean_series = series.dropna()

    # Ajustar n_lags si la serie es corta
    max_lags = min(n_lags, len(clean_series) // 2 - 1)
    max_lags = max(max_lags, 5)

    acf_values = acf(clean_series, nlags=max_lags, fft=True)
    pacf_values = pacf(clean_series, nlags=max_lags, method="ywm")

    # Banda de confianza al 95%: ±1.96/sqrt(N)
    confidence = 1.96 / np.sqrt(len(clean_series))

    return AcfPacfResult(
        acf=[float(v) for v in acf_values],
        pacf=[float(v) for v in pacf_values],
        confidence_interval=float(confidence),
        lags=list(range(len(acf_values))),
    )


def get_summary_stats(series: pd.Series) -> dict[str, float]:
    """Estadísticas descriptivas básicas de la serie."""
    return {
        "count": float(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }


def explore_series(data: TimeSeriesData) -> ExplorationResponse:
    """Punto de entrada: ejecuta todo el análisis exploratorio."""
    series = _series_to_pandas(data)

    return ExplorationResponse(
        stationarity=run_stationarity_test(series),
        decomposition=run_decomposition(series, data.frequency),
        acf_pacf=run_acf_pacf(series),
        summary_stats=get_summary_stats(series),
    )
