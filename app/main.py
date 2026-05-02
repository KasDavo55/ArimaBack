"""
Aplicación principal FastAPI.

Backend para pronóstico de series temporales con ARIMA/SARIMA.
Diseñado para consumirse desde el frontend React en localhost:5173.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import exploration, forecast

app = FastAPI(
    title="ARIMA/SARIMA Forecast API",
    description=(
        "Backend para análisis y pronóstico de series temporales. "
        "Proyecto de Investigación de Operaciones."
    ),
    version="1.0.0",
)

# CORS: permitir que el frontend de Vite (localhost:5173) consuma esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # alternativa común
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(exploration.router)
app.include_router(forecast.router)


@app.get("/", tags=["Health"])
def root():
    """Endpoint de verificación."""
    return {
        "service": "ARIMA/SARIMA Forecast API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
