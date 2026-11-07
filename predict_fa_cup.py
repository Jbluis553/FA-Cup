"""Genera probabilidades de victoria local para próximos partidos de la FA Cup.

Requiere los artefactos descargados de una ejecución de train_fa_cup.py:
fa_cup_dataset.csv y models/. Usa THESTATSAPI_API_KEY para consultar los partidos.
"""

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train_fa_cup import (
    API_KEY,
    PER_PAGE,
    SLEEP_BETWEEN_CALLS,
    _is_neutral,
    _match_odds,
    _match_xg,
    _team_level,
    api_get,
    engineer_features,
    find_fa_cup_competition,
    get_seasons,
)

DATASET_PATH = Path(os.getenv("FA_CUP_DATASET", "fa_cup_dataset.csv"))
MODELS_DIR = Path(os.getenv("FA_CUP_MODELS_DIR", "models"))
OUTPUT_PATH = Path(os.getenv("FA_CUP_PREDICTIONS", "fa_cup_predictions.csv"))
UPCOMING_STATUS = os.getenv("UPCOMING_MATCH_STATUS", "scheduled")


def fetch_upcoming_matches(competition_id, season_id):
    """Descarga fixtures futuros de una temporada, recorriendo todas las páginas."""
    matches = []
    page = 1
    while True:
        data = api_get("/football/matches", {
            "competition_id": competition_id,
            "season_id": season_id,
            "status": UPCOMING_STATUS,
            "per_page": PER_PAGE,
            "page": page,
            "sort": "utc_date",
        })
        if not data:
            break
        page_matches = data.get("data", [])
        if not page_matches:
            break
        matches.extend(page_matches)
        total_pages = data.get("meta", {}).get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(SLEEP_BETWEEN_CALLS)
    return matches


def upcoming_row(match, season_id):
    """Convierte un fixture al esquema bruto que usa el entrenador."""
    home = match.get("home_team") or {}
    away = match.get("away_team") or {}
    odds_prob, odds_margin, home_odds = _match_odds(match)
    return {
        "match_id": match.get("id"),
        "season_id": season_id,
        "utc_date": match.get("utc_date"),
        "home_team_id": home.get("id"),
        "home_team": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team": away.get("name"),
        # Valores de relleno: engineer_features calcula las features antes de
        # incorporar el resultado del partido al historial.
        "home_goals": 0,
        "away_goals": 0,
        "result": "D",
        "home_xg": _match_xg(match, "home"),
        "away_xg": _match_xg(match, "away"),
        "home_level": _team_level(home),
        "away_level": _team_level(away),
        "neutral_venue": _is_neutral(match, home),
        "market_home_prob": odds_prob,
        "market_overround": odds_margin,
        "market_home_odds": home_odds,
    }


def probability_for_class_one(model, X):
    """Devuelve P(home win), localizando explícitamente la clase positiva 1."""
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"El modelo no contiene la clase positiva 1: {classes}")
    return float(probabilities[classes.index(1)])


def predict_fixture(history, fixture, feature_cols, model_or_ensemble):
    """Calcula features usando solo resultados anteriores al kickoff del fixture."""
    fixture_date = pd.to_datetime(fixture["utc_date"], utc=True, errors="coerce")
    if pd.isna(fixture_date):
        raise ValueError(f"Fecha inválida para el partido {fixture.get('match_id')}")

    history = history.copy()
    history["utc_date"] = pd.to_datetime(history["utc_date"], utc=True, errors="coerce")
    history = history.dropna(subset=["utc_date"])
    history = history[history["utc_date"] < fixture_date]

    # Cada fixture se procesa por separado, así ningún partido futuro puede
    # contaminar las features de otro fixture.
    raw = pd.concat([history, pd.DataFrame([fixture])], ignore_index=True, sort=False)
    engineered = engineer_features(raw)
    fixture_features = engineered.tail(1)[feature_cols]
    X = fixture_features.replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")

    if isinstance(model_or_ensemble, dict):
        model_probs = [probability_for_class_one(model, X) for model in model_or_ensemble.values()]
        return float(np.mean(model_probs))
    return probability_for_class_one(model_or_ensemble, X)


def main():
    if not API_KEY:
        raise SystemExit("Falta la variable de entorno THESTATSAPI_API_KEY.")
    if not DATASET_PATH.is_file():
        raise SystemExit(
            f"No existe {DATASET_PATH}. Descarga el dataset del artefacto de entrenamiento."
        )

    ensemble_path = MODELS_DIR / "ensemble_components.joblib"
    model_path = MODELS_DIR / "best_model.joblib"
    feature_path = MODELS_DIR / "feature_cols.joblib"
    if not feature_path.is_file() or not model_path.is_file():
        raise SystemExit(
            f"Faltan modelos en {MODELS_DIR}. Descarga los artefactos del entrenamiento."
        )

    history = pd.read_csv(DATASET_PATH)
    feature_cols = joblib.load(feature_path)
    if ensemble_path.is_file():
        model = joblib.load(ensemble_path)
    else:
        model = joblib.load(model_path)

    competition_id = find_fa_cup_competition()
    seasons = get_seasons(competition_id)
    fixtures = []
    for season in seasons:
        season_id = season["id"]
        for match in fetch_upcoming_matches(competition_id, season_id):
            fixture = upcoming_row(match, season_id)
            if fixture["utc_date"] and fixture["home_team"] and fixture["away_team"]:
                fixtures.append(fixture)

    if not fixtures:
        print(
            f"No se encontraron partidos futuros con status={UPCOMING_STATUS!r}. "
            "Si la API usa otro estado, configúralo en UPCOMING_MATCH_STATUS."
        )
        return

    predictions = []
    for fixture in fixtures:
        home_win_prob = predict_fixture(history, fixture, feature_cols, model)
        predictions.append({
            "match_id": fixture["match_id"],
            "utc_date": fixture["utc_date"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "home_win_probability": home_win_prob,
            "no_home_win_probability": 1.0 - home_win_prob,
        })

    output = pd.DataFrame(predictions).sort_values("utc_date")
    output.to_csv(OUTPUT_PATH, index=False)
    print(output.to_string(index=False))
    print(f"\nPredicciones guardadas en {OUTPUT_PATH}")
    print("Nota: el modelo estima victoria local frente a no victoria local; no predice empate por separado.")


if __name__ == "__main__":
    main()
