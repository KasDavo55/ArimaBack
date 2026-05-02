"""
Servicio de pronóstico con ARIMA y SARIMA.

Implementa:
- Entrenamiento manual ARIMA(p,d,q) y SARIMA(p,d,q)(P,D,Q,s).
- Selección automática con auto_arima (pmdarima).
- Cálculo de métricas: RMSE, MAE, MAPE, AIC, BIC.
- Diagnóstico de residuos (Ljung-Box).
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from pmdarima import auto_arima  # type: ignore
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False

from app.models.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastMetrics,
    ForecastPoint,
    ResidualDiagnostics,
    TimeSeriesPoint,
)
from app.services.exploration_service import _series_to_pandas, SEASONAL_PERIODS


# ============================================================
# Métricas
# ============================================================

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    aic: float,
    bic: float,
) -> ForecastMetrics:
    """Calcula métricas de evaluación del modelo."""
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))

    # MAPE: evitar división por cero
    mask = actual != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    else:
        mape = 0.0

    return ForecastMetrics(
        rmse=rmse,
        mae=mae,
        mape=mape,
        aic=float(aic),
        bic=float(bic),
    )


# ============================================================
# Diagnóstico de residuos
# ============================================================

def diagnose_residuals(residuals: np.ndarray) -> ResidualDiagnostics:
    """
    Verifica si los residuos parecen ruido blanco mediante el test de Ljung-Box.

    H0: los residuos son independientes (ruido blanco).
    Si p > 0.05 → no rechazamos H0 → el modelo capturó bien la estructura.
    """
    clean_residuals = residuals[~np.isnan(residuals)]

    # Ljung-Box con 10 lags (estándar)
    n_lags = min(10, len(clean_residuals) // 5)
    lb_test = acorr_ljungbox(clean_residuals, lags=[n_lags], return_df=True)
    p_value = float(lb_test["lb_pvalue"].iloc[0])

    is_white_noise = p_value > 0.05

    if is_white_noise:
        interpretation = (
            f"Los residuos parecen ruido blanco (Ljung-Box p={p_value:.4f} > 0.05). "
            "El modelo capturó adecuadamente la estructura de la serie."
        )
    else:
        interpretation = (
            f"Los residuos NO son ruido blanco (Ljung-Box p={p_value:.4f} <= 0.05). "
            "Existe estructura no capturada; considere ajustar los parámetros."
        )

    return ResidualDiagnostics(
        ljung_box_p_value=p_value,
        is_white_noise=is_white_noise,
        residuals=[float(v) for v in clean_residuals],
        interpretation=interpretation,
    )


# ============================================================
# Selección automática de parámetros
# ============================================================

def find_best_arima_order(
    train_series: pd.Series,
    seasonal: bool,
    seasonal_period: int,
) -> tuple[tuple[int, int, int], Optional[tuple[int, int, int, int]]]:
    """
    Usa auto_arima para encontrar los mejores parámetros minimizando AIC.
    Si pmdarima no está disponible, usa una búsqueda manual simple.
    """
    if AUTO_ARIMA_AVAILABLE:
        model = auto_arima(
            train_series,
            seasonal=seasonal,
            m=seasonal_period if seasonal else 1,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            max_d=2,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        order = model.order
        s_order = model.seasonal_order if seasonal else None
        return order, s_order

    # Fallback: búsqueda manual sencilla
    best_aic = np.inf
    best_order = (1, 1, 1)
    best_seasonal: Optional[tuple[int, int, int, int]] = None

    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    if seasonal:
                        m = SARIMAX(
                            train_series,
                            order=(p, d, q),
                            seasonal_order=(1, 1, 1, seasonal_period),
                        ).fit(disp=False)
                        s_order = (1, 1, 1, seasonal_period)
                    else:
                        m = ARIMA(train_series, order=(p, d, q)).fit()
                        s_order = None

                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = (p, d, q)
                        best_seasonal = s_order
                except Exception:
                    continue

    return best_order, best_seasonal


# ============================================================
# Entrenamiento y pronóstico
# ============================================================

def fit_and_forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Punto de entrada principal: entrena el modelo y genera el pronóstico.

    Pasos:
    1. Convertir datos a Serie de pandas.
    2. Dividir en train/test.
    3. Determinar parámetros (manual o auto).
    4. Ajustar modelo SARIMAX (que generaliza ARIMA y SARIMA).
    5. Predecir sobre test, calcular métricas.
    6. Predecir hacia el futuro con bandas de confianza.
    7. Diagnosticar residuos.
    """
    # 1. Preparación
    series = _series_to_pandas(request.series)
    n = len(series)
    train_end = int(n * request.train_size)
    train = series.iloc[:train_end]
    test = series.iloc[train_end:]

    if len(test) < 1:
        raise ValueError("El conjunto de prueba está vacío. Reduce train_size.")

    seasonal_period = SEASONAL_PERIODS.get(request.series.frequency, 12)

    # 2. Determinar parámetros
    use_seasonal = request.model_type in ("SARIMA", "AUTO")

    if request.model_type == "AUTO":
        order, s_order = find_best_arima_order(
            train, seasonal=True, seasonal_period=seasonal_period
        )
        # Si auto no eligió componente estacional significativo, queda como ARIMA
        model_used = "SARIMA" if (s_order and any(s_order[:3])) else "ARIMA"
    elif request.model_type == "ARIMA":
        if not request.arima_order:
            raise ValueError("ARIMA requiere arima_order (p, d, q).")
        order = (request.arima_order.p, request.arima_order.d, request.arima_order.q)
        s_order = None
        model_used = "ARIMA"
    else:  # SARIMA
        if not request.arima_order or not request.seasonal_order:
            raise ValueError("SARIMA requiere arima_order y seasonal_order.")
        order = (request.arima_order.p, request.arima_order.d, request.arima_order.q)
        s_order = (
            request.seasonal_order.P,
            request.seasonal_order.D,
            request.seasonal_order.Q,
            request.seasonal_order.s,
        )
        model_used = "SARIMA"

    # 3. Ajustar modelo (usamos SARIMAX que cubre ambos casos)
    if s_order is not None:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=s_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    else:
        model = SARIMAX(train, order=order).fit(disp=False)

    # 4. Predicción sobre test
    test_predictions = model.forecast(steps=len(test))
    test_pred_values = np.array(test_predictions)

    # 5. Métricas sobre test
    metrics = calculate_metrics(
        actual=test.values,
        predicted=test_pred_values,
        aic=model.aic,
        bic=model.bic,
    )

    # 6. Pronóstico futuro: re-entrenamos con TODOS los datos para mejor pronóstico
    if s_order is not None:
        full_model = SARIMAX(
            series,
            order=order,
            seasonal_order=s_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    else:
        full_model = SARIMAX(series, order=order).fit(disp=False)

    forecast_result = full_model.get_forecast(steps=request.forecast_horizon)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=1 - request.confidence_level)

    # Generar fechas futuras
    last_date = series.index[-1]
    freq_map = {"D": "D", "W": "W", "M": "ME", "Q": "QE"}
    pandas_freq = freq_map.get(request.series.frequency, "ME")
    future_dates = pd.date_range(
        start=last_date,
        periods=request.forecast_horizon + 1,
        freq=pandas_freq,
    )[1:]

    forecast_points = [
        ForecastPoint(
            date=date.strftime("%Y-%m-%d"),
            forecast=float(mean_val),
            lower_bound=float(conf_int.iloc[i, 0]),
            upper_bound=float(conf_int.iloc[i, 1]),
        )
        for i, (date, mean_val) in enumerate(zip(future_dates, forecast_mean))
    ]

    # 7. Diagnóstico de residuos
    residuals = np.array(model.resid)
    residual_diag = diagnose_residuals(residuals)

    # 8. Construir respuesta
    train_data = [
        TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=float(v))
        for d, v in zip(train.index, train.values)
    ]
    test_data = [
        TimeSeriesPoint(date=d.strftime("%Y-%m-%d"), value=float(v))
        for d, v in zip(test.index, test.values)
    ]

    return ForecastResponse(
        model_type_used=model_used,
        arima_order=list(order),
        seasonal_order=list(s_order) if s_order else None,
        metrics=metrics,
        train_data=train_data,
        test_data=test_data,
        test_predictions=[float(v) for v in test_pred_values],
        forecast=forecast_points,
        residual_diagnostics=residual_diag,
    )
