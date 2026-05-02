# Backend ARIMA/SARIMA — Guía de Instalación y Uso

Backend en Python + FastAPI para el proyecto de pronóstico de series temporales.

## 📦 Requisitos previos

- **Python 3.10 o superior** (verifica con `python --version`)
- **pip** actualizado (`python -m pip install --upgrade pip`)
- El frontend de React ya configurado y funcionando

## 🚀 Instalación paso a paso

### 1. Crear la estructura del backend

Ubícate en una carpeta **fuera** del proyecto React (NO dentro de `arima-forecast-app`). Por ejemplo, en la misma carpeta padre:

```
mi-proyecto/
├── arima-forecast-app/    ← tu frontend (ya existe)
└── arima-backend/         ← lo crearás ahora
```

### 2. Crear entorno virtual

```bash
# Windows (PowerShell)
cd ruta/a/arima-backend
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv venv
venv\Scripts\activate.bat

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

Cuando el entorno esté activo verás `(venv)` al inicio de tu terminal.

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ **Nota sobre `pmdarima`:** En Windows puede dar problemas de compilación. Si falla la instalación, el sistema seguirá funcionando con búsqueda manual de parámetros (un poco más lenta pero igual de efectiva). Solución alternativa:
> ```bash
> pip install -r requirements.txt --no-deps pmdarima
> # o simplemente comentar la línea de pmdarima en requirements.txt
> ```

### 4. Estructura final del backend

```
arima-backend/
├── venv/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── exploration.py
│   │   └── forecast.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── exploration_service.py
│   │   └── forecast_service.py
│   └── utils/
│       └── __init__.py
└── requirements.txt
```

### 5. Ejecutar el backend

Con el entorno virtual activado:

```bash
uvicorn app.main:app --reload --port 8000
```

Deberías ver:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 6. Verificar que funciona

Abre en el navegador:

- **http://localhost:8000** → mensaje de health check
- **http://localhost:8000/docs** → documentación interactiva (Swagger UI)

En `/docs` puedes probar los endpoints directamente sin código.

## 🔌 Conexión con el frontend

### Archivos a copiar al frontend

Copia estos archivos del directorio `frontend-files/` a tu proyecto React:

| Archivo origen | Destino en frontend |
|---|---|
| `frontend-files/src/types/api.types.ts` | `arima-forecast-app/src/types/api.types.ts` |
| `frontend-files/src/services/forecastApi.ts` | `arima-forecast-app/src/services/forecastApi.ts` |
| `frontend-files/src/utils/aggregator.ts` | `arima-forecast-app/src/utils/aggregator.ts` |

> Nota: Vas a tener que crear la carpeta `src/services/` en el frontend.

### Variable de entorno (opcional pero recomendado)

En la raíz de `arima-forecast-app/` crea un archivo `.env`:

```
VITE_API_URL=http://localhost:8000
```

## 📡 Endpoints disponibles

### `POST /exploration/analyze`

Análisis exploratorio de la serie.

**Request:**
```json
{
  "points": [
    {"date": "2017-01-31", "value": 14236.9},
    {"date": "2017-02-28", "value": 4519.9}
  ],
  "frequency": "M"
}
```

**Response:** Test ADF, descomposición, ACF, PACF, estadísticas.

### `POST /forecast`

Entrena el modelo y genera pronóstico.

**Request:**
```json
{
  "series": {
    "points": [...],
    "frequency": "M"
  },
  "model_type": "AUTO",
  "train_size": 0.8,
  "forecast_horizon": 12,
  "confidence_level": 0.95
}
```

**Response:** Modelo usado, parámetros, métricas, datos de train/test, pronóstico futuro con bandas de confianza, diagnóstico de residuos.

## 🔄 Flujo completo de uso

```
1. Usuario carga CSV en frontend
   ↓
2. Frontend agrega los datos (aggregator.ts)
   → 10,000 filas → ~48 puntos mensuales
   ↓
3. Frontend llama POST /exploration/analyze
   → Muestra ADF, descomposición, ACF/PACF
   ↓
4. Usuario configura modelo (manual o AUTO)
   ↓
5. Frontend llama POST /forecast
   → Backend entrena, evalúa y pronostica
   ↓
6. Frontend muestra: gráfico, métricas, residuos
```

## 🐛 Resolución de problemas comunes

**Error: `ModuleNotFoundError: No module named 'app'`**
→ Asegúrate de ejecutar `uvicorn` desde la raíz de `arima-backend/`, no desde dentro de `app/`.

**Error CORS en el frontend**
→ Verifica que `localhost:5173` esté en la lista `allow_origins` de `app/main.py`.

**El backend tarda mucho**
→ Normal en la primera solicitud (carga de modelos). Solicitudes siguientes son rápidas. `auto_arima` con búsqueda exhaustiva puede tomar 10-30 segundos en series largas.

**Error al instalar pmdarima**
→ Es opcional. Comenta esa línea en `requirements.txt` y el sistema usará búsqueda manual de parámetros.

## 🧪 Probar con datos de ejemplo

Puedes probar directamente desde Swagger UI (`/docs`) con este payload mínimo:

```json
{
  "series": {
    "points": [
      {"date": "2020-01-01", "value": 100},
      {"date": "2020-02-01", "value": 110},
      {"date": "2020-03-01", "value": 105},
      {"date": "2020-04-01", "value": 120},
      {"date": "2020-05-01", "value": 125},
      {"date": "2020-06-01", "value": 130},
      {"date": "2020-07-01", "value": 135},
      {"date": "2020-08-01", "value": 140},
      {"date": "2020-09-01", "value": 145},
      {"date": "2020-10-01", "value": 150},
      {"date": "2020-11-01", "value": 160},
      {"date": "2020-12-01", "value": 170}
    ],
    "frequency": "M"
  },
  "model_type": "AUTO",
  "train_size": 0.8,
  "forecast_horizon": 6,
  "confidence_level": 0.95
}
```
