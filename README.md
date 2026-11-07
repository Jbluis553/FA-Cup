# FA Cup BTTS Yes

Proyecto para estimar la probabilidad de que ambos equipos marquen al menos un gol (BTTS Yes) en próximos partidos de la FA Cup.

## Requisitos

Instala las dependencias con `pip install -r requirements.txt` y configura la variable `THESTATSAPI_API_KEY`.

## Entrenar

```bash
python train_fa_cup.py
```

El entrenamiento descarga partidos finalizados y define `btts_yes = 1` cuando `home_goals > 0` y `away_goals > 0`. El dataset queda en `fa_cup_dataset.csv` y los artefactos del modelo en `models/btts/`.

## Predecir próximos partidos

```bash
python predict_fa_cup.py
```

El predictor consulta fixtures futuros y escribe `fa_cup_predictions_btts.csv`, con las probabilidades de BTTS Yes y BTTS No. Los fixtures se predicen usando únicamente el historial anterior a su fecha.

Se puede cambiar el estado consultado por la API con `UPCOMING_MATCH_STATUS` (por defecto `scheduled`).

El modelo estima probabilidades, no garantiza resultados ni constituye consejo de apuestas.
