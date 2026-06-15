import os
import json
from anthropic import Anthropic

client = Anthropic(api_key="sk-ant-api03")

SYSTEM_PROMPT = """Eres un experto en análisis de series temporales y modelos econométricos.
Recibirás un resumen estadístico de una serie temporal y debes recomendar el modelo de pronóstico más adecuado entre: ARIMA, SARIMA, o señalar si los datos son inadecuados para estos modelos.

Criterios de decisión:
- Si seasonal_strength es alto (>0.4) y hay lags significativos en múltiplos del período (ej. 12, 24): recomienda SARIMA.
- Si hay tendencia clara pero poca estacionalidad: recomienda ARIMA con diferenciación.
- Si adf_pvalue > 0.05: la serie no es estacionaria, indica el grado de diferenciación (d) sugerido.
- Considera el coeficiente de variación (cv) para la varianza; si es muy alta sugiere transformación logarítmica en el reasoning.
- Si n_observations es muy bajo (<24) advierte sobre baja confiabilidad.

Responde ÚNICAMENTE con un objeto JSON válido, sin markdown ni texto adicional, con esta estructura exacta:
{
  "recommended_model": "SARIMA" | "ARIMA" | "NONE",
  "suggested_params": {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int, "s": int},
  "confidence": "alta" | "media" | "baja",
  "reasoning": "explicación breve en 2-3 frases citando los valores específicos del resumen"
}

Si recomiendas ARIMA (sin estacionalidad), usa 0 para los parámetros estacionales P, D, Q, s."""


def recommend_model(summary: dict) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Resumen estadístico de la serie:\n{json.dumps(summary, indent=2)}",
            }
        ],
    )

    raw = message.content[0].text.strip()

    # defensa por si el modelo envuelve la respuesta en fences pese a la instrucción
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"La IA no devolvió un JSON válido. Respuesta cruda: {raw[:200]}"
        ) from e