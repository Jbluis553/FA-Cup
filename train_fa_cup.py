"""
Entrenamiento de modelos para predecir victorias locales en la FA Cup.

Las features de cada partido se construyen exclusivamente con datos anteriores
al encuentro. xG histórico y descanso se derivan de partidos anteriores; odds,
neutralidad y nivel se utilizan solo si vienen en el registro del encuentro.
"""

import os
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:  # Logistic Regression sigue siendo utilizable sin LightGBM
    LGBMClassifier = None

# ============================================================
# CONFIGURACIÓN
# ============================================================
API_KEY = os.getenv("THESTATSAPI_API_KEY")
BASE_URL = "https://api.thestatsapi.com/api"
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
MAX_SEASONS = int(os.getenv("MAX_SEASONS", "10"))
PER_PAGE = 100
SLEEP_BETWEEN_CALLS = 0.35
RANDOM_STATE = 42

# ============================================================
# HELPERS DE LA API
# ============================================================
def api_get(endpoint: str, params: dict = None):
    """Hace GET a TheStatsAPI con reintento limitado ante rate limits."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
        except requests.RequestException as exc:
            print(f"Error de red en {endpoint}: {exc}")
            return None
        if response.status_code == 429:
            retry = int(response.headers.get("Retry-After", 60))
            print(f"Rate limit. Esperando {retry}s...")
            time.sleep(retry)
            continue
        if response.status_code != 200:
            print(f"Error {response.status_code} en {endpoint}: {response.text[:300]}")
            return None
        try:
            return response.json()
        except ValueError:
            print(f"Respuesta JSON inválida en {endpoint}")
            return None
    print(f"Se agotaron los reintentos para {endpoint}")
    return None


def find_fa_cup_competition():
    """Busca el competition_id de la FA Cup."""
    data = api_get("/football/competitions", {
        "search": "FA Cup", "country_code": "GB-ENG", "type": "cup", "per_page": 10
    })
    if not data or not data.get("data"):
        data = api_get("/football/competitions", {"search": "FA Cup", "per_page": 10})
    if not data or not data.get("data"):
        raise ValueError("No se encontró la FA Cup. Revisa tu API key o cobertura.")
    for comp in data["data"]:
        name = comp.get("name", "").lower()
        if "fa cup" in name and "women" not in name:
            print(f"Encontrada: {comp['name']} → {comp['id']}")
            return comp["id"]
    comp = data["data"][0]
    print(f"Usando: {comp['name']} → {comp['id']}")
    return comp["id"]


def get_seasons(competition_id: str, max_seasons: int = MAX_SEASONS):
    """Devuelve las últimas N temporadas (más recientes primero)."""
    data = api_get(f"/football/competitions/{competition_id}/seasons")
    if not data or not data.get("data"):
        return []
    seasons = data["data"][:max_seasons]
    print(f"Temporadas a descargar: {[s.get('name') or s.get('year') for s in seasons]}")
    return seasons


def fetch_all_matches(competition_id: str, season_id: str, status: str = "finished"):
    """Descarga todos los partidos de una temporada (paginado)."""
    all_matches, page = [], 1
    while True:
        data = api_get("/football/matches", {
            "competition_id": competition_id, "season_id": season_id,
            "status": status, "per_page": PER_PAGE, "page": page, "sort": "utc_date"
        })
        if not data:
            break
        matches = data.get("data", [])
        if not matches:
            break
        all_matches.extend(matches)
        total_pages = data.get("meta", {}).get("total_pages", 1)
        print(f"  Página {page}/{total_pages} → {len(matches)} partidos")
        if page >= total_pages:
            break
        page += 1
        time.sleep(SLEEP_BETWEEN_CALLS)
    return all_matches


def _first_number(obj, keys):
    """Busca un valor numérico por nombre en un diccionario, incluso anidado."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in keys and isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        for value in obj.values():
            found = _first_number(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _first_number(value, keys)
            if found is not None:
                return found
    return None


def _team_level(team):
    """Lee nivel/división/rango si la API lo incluye en el objeto del equipo."""
    keys = {"league_level", "division_level", "tier", "level", "division_rank", "league_rank"}
    return _first_number(team, keys)


def _match_odds(match):
    """Obtiene cuotas 1X2 prepartido si existen directamente en el payload."""
    odds = match.get("odds") or match.get("pre_match_odds") or match.get("prematch_odds")
    if not odds:
        return None, None, None
    home = _first_number(odds, {"home", "home_win", "home_odds", "1"})
    draw = _first_number(odds, {"draw", "draw_odds", "x"})
    away = _first_number(odds, {"away", "away_win", "away_odds", "2"})
    # Descarta precios que no parecen cuotas decimales válidas.
    values = [v if v is not None and v > 1.0 else None for v in (home, draw, away)]
    if all(v is not None for v in values):
        inv = [1.0 / v for v in values]
        total = sum(inv)
        return inv[0] / total, total, home
    return None, None, home


def _is_neutral(match, home_team):
    """Devuelve 1/0 si hay señal explícita de campo neutral, si no NaN."""
    for key in ("neutral", "is_neutral", "neutral_venue"):
        value = match.get(key)
        if isinstance(value, bool):
            return float(value)
    venue = match.get("venue") or {}
    venue_id = venue.get("id") if isinstance(venue, dict) else None
    home_venue_id = home_team.get("venue_id") or home_team.get("stadium_id")
    if venue_id is not None and home_venue_id is not None:
        return float(str(venue_id) != str(home_venue_id))
    return np.nan


def _match_xg(match, side):
    """Extrae xG del partido cuando está embebido en el registro."""
    side_keys = ({"home", "home_team", "home_xg", "xg_home"} if side == "home"
                 else {"away", "away_team", "away_xg", "xg_away"})
    for container_name in ("xg", "expected_goals", "stats", "statistics", "match_stats"):
        container = match.get(container_name)
        if isinstance(container, dict):
            # Claves explícitas son preferibles a búsquedas recursivas ambiguas.
            for key in (f"{side}_xg", f"xg_{side}", side):
                value = container.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, dict):
                    for xg_key in ("xg", "expected_goals"):
                        if isinstance(value.get(xg_key), (int, float)):
                            return float(value[xg_key])
    return _first_number(match.get("xg") or match.get("expected_goals"), side_keys)


# ============================================================
# CONSTRUCCIÓN DEL DATASET
# ============================================================
def build_dataset():
    if not API_KEY:
        raise ValueError("Falta la variable de entorno THESTATSAPI_API_KEY")
    competition_id = find_fa_cup_competition()
    seasons = get_seasons(competition_id)
    rows = []
    for season in seasons:
        season_id = season["id"]
        print(f"\nDescargando temporada {season.get('name') or season.get('year')} ({season_id})...")
        matches = fetch_all_matches(competition_id, season_id)
        print(f"  Total partidos: {len(matches)}")
        for match in matches:
            score = match.get("score") or {}
            home_goals, away_goals = score.get("home"), score.get("away")
            if home_goals is None or away_goals is None:
                continue
            winner = score.get("winner")
            if winner == "home" or (winner is None and home_goals > away_goals):
                result = "H"
            elif winner == "away" or (winner is None and away_goals > home_goals):
                result = "A"
            else:
                result = "D"
            home, away = match.get("home_team") or {}, match.get("away_team") or {}
            odds_prob, odds_margin, home_odds = _match_odds(match)
            row = {
                "match_id": match.get("id"), "season_id": season_id,
                "utc_date": match.get("utc_date"),
                "home_team_id": home.get("id"), "home_team": home.get("name"),
                "away_team_id": away.get("id"), "away_team": away.get("name"),
                "home_goals": home_goals, "away_goals": away_goals, "result": result,
                "home_xg": _match_xg(match, "home"), "away_xg": _match_xg(match, "away"),
                "home_level": _team_level(home), "away_level": _team_level(away),
                "neutral_venue": _is_neutral(match, home),
                "market_home_prob": odds_prob, "market_overround": odds_margin,
                "market_home_odds": home_odds,
            }
            rows.append(row)
        time.sleep(SLEEP_BETWEEN_CALLS)
    df = pd.DataFrame(rows)
    print(f"\nDataset bruto: {len(df)} partidos")
    return df


def _weighted_average(records, field, window):
    valid = [r[field] for r in records[-window:] if r.get(field) is not None]
    if not valid:
        return np.nan
    weights = np.power(0.75, np.arange(len(valid) - 1, -1, -1))
    return float(np.average(valid, weights=weights))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera features prepartido, forma reciente ponderada y ventanas 3/5/10."""
    df = df.copy()
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["utc_date", "home_team", "away_team"]).sort_values("utc_date").reset_index(drop=True)
    history = {}
    feature_rows = []

    for _, row in df.iterrows():
        date = row["utc_date"]
        teams = (row["home_team"], row["away_team"])
        features = {}
        for side, team, opponent in (("home", teams[0], teams[1]), ("away", teams[1], teams[0])):
            records = history.get(team, [])
            for window in (3, 5, 10):
                for key in ("points", "gf", "ga", "xg_for", "xg_against"):
                    features[f"{side}_{key}_{window}"] = _weighted_average(records, key, window)
            all_records = records[-10:]
            last_date = records[-1]["date"] if records else None
            features[f"{side}_rest_days"] = (
                min(max((date - last_date).total_seconds() / 86400.0, 0), 60)
                if last_date is not None else np.nan
            )
            # La forma local/visitante distingue el rendimiento según la sede previa.
            for venue_role in ("home", "away"):
                role_records = [r for r in records if r["venue_role"] == venue_role][-5:]
                features[f"{side}_{venue_role}_form5"] = _weighted_average(role_records, "points", 5)
            features[f"{side}_matches_last_10"] = float(len(all_records))

        for window in (3, 5, 10):
            features[f"form_diff_{window}"] = features[f"home_points_{window}"] - features[f"away_points_{window}"]
            features[f"xg_diff_{window}"] = features[f"home_xg_for_{window}"] - features[f"away_xg_for_{window}"]
        features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]
        features["congestion_diff"] = features["home_matches_last_10"] - features["away_matches_last_10"]
        # La API omite frecuentemente el nivel de los equipos; conviértelo a NaN
        # antes de restar, ya que None - None provoca TypeError durante el pipeline.
        away_level = pd.to_numeric(row.get("away_level"), errors="coerce")
        home_level = pd.to_numeric(row.get("home_level"), errors="coerce")
        features["level_diff"] = away_level - home_level
        features["neutral_venue"] = row.get("neutral_venue", np.nan)
        features["market_home_prob"] = row.get("market_home_prob", np.nan)
        features["market_overround"] = row.get("market_overround", np.nan)
        features["market_home_odds"] = row.get("market_home_odds", np.nan)
        feature_rows.append(features)

        # Los datos del encuentro se agregan después de calcular sus features:
        # nunca se filtra el resultado ni xG del propio partido al modelo.
        for side, team, gf, ga in (
            ("home", teams[0], row["home_goals"], row["away_goals"]),
            ("away", teams[1], row["away_goals"], row["home_goals"]),
        ):
            points = 3 if gf > ga else (1 if gf == ga else 0)
            xg_for = row.get(f"{side}_xg")
            xg_against = row.get("away_xg" if side == "home" else "home_xg")
            item = {
                "date": date, "venue_role": side, "points": points,
                "gf": float(gf), "ga": float(ga),
                "xg_for": float(xg_for) if pd.notna(xg_for) else None,
                "xg_against": float(xg_against) if pd.notna(xg_against) else None,
            }
            history.setdefault(team, []).append(item)

    feature_frame = pd.DataFrame(feature_rows, index=df.index)
    result = pd.concat([df, feature_frame], axis=1)
    return result


# ============================================================
# ENTRENAMIENTO
# ============================================================
def train_models(df: pd.DataFrame):
    # Home Win vs No Home Win: se excluyen H features objetivo multi-clase.
    feature_cols = [c for c in df.columns if c.startswith(("home_", "away_", "form_diff_", "xg_diff_"))]
    feature_cols += ["rest_days_diff", "congestion_diff", "level_diff", "neutral_venue",
                     "market_home_prob", "market_overround", "market_home_odds"]
    feature_cols = list(dict.fromkeys(c for c in feature_cols if c in df.columns))
    data = df.dropna(subset=["result"]).copy().sort_values("utc_date")
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    y = (data["result"] == "H").astype(int)
    if len(data) < 30 or y.nunique() < 2:
        raise ValueError(f"Datos insuficientes para entrenar clasificación binaria ({len(data)} partidos).")

    split = max(int(len(data) * 0.8), 1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_test) == 0 or y_train.nunique() < 2:
        raise ValueError("El split temporal no permite evaluar ambas clases; descarga más temporadas.")

    n_splits = min(3, max(2, len(X_train) // 20))
    cv = TimeSeriesSplit(n_splits=n_splits)
    models = {
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        )
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            LGBMClassifier(n_estimators=250, max_depth=5, learning_rate=0.035,
                           num_leaves=15, class_weight="balanced", random_state=RANDOM_STATE,
                           verbosity=-1)
        )
    else:
        warnings.warn("LightGBM no está instalado; solo se entrenará Logistic Regression.")

    calibrated = {}
    probabilities = {}
    results = {}
    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        print(f"\nEntrenando y calibrando {name}...")
        # Si hay muy pocos datos, evita una calibración CV inestable.
        if len(X_train) >= 45:
            fitted = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=cv)
        else:
            warnings.warn("Pocos ejemplos: se omite calibración cruzada para este dataset.")
            fitted = model
        fitted.fit(X_train, y_train)
        prob = fitted.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "log_loss": log_loss(y_test, prob, labels=[0, 1]),
            "brier_score": brier_score_loss(y_test, prob),
        }
        print(f"  Accuracy: {metrics['accuracy']:.4f} | LogLoss: {metrics['log_loss']:.4f} | Brier: {metrics['brier_score']:.4f}")
        calibrated[name], probabilities[name], results[name] = fitted, prob, metrics
        joblib.dump(fitted, f"models/{name.lower()}.joblib")

    if len(probabilities) >= 2:
        ensemble_prob = np.mean(np.column_stack(list(probabilities.values())), axis=1)
        results["Ensemble"] = {
            "accuracy": accuracy_score(y_test, ensemble_prob >= 0.5),
            "log_loss": log_loss(y_test, ensemble_prob, labels=[0, 1]),
            "brier_score": brier_score_loss(y_test, ensemble_prob),
        }
        print("\nEnsemble (promedio de probabilidades calibradas): "
              f"Accuracy={results['Ensemble']['accuracy']:.4f} | "
              f"LogLoss={results['Ensemble']['log_loss']:.4f} | "
              f"Brier={results['Ensemble']['brier_score']:.4f}")
        joblib.dump(calibrated, "models/ensemble_components.joblib")
        joblib.dump("mean calibrated probabilities", "models/best_model.joblib")
    else:
        best_name = next(iter(calibrated))
        joblib.dump(calibrated[best_name], "models/best_model.joblib")

    joblib.dump(feature_cols, "models/feature_cols.joblib")
    joblib.dump({"target": "home_win", "positive_class": 1}, "models/target_metadata.joblib")
    print("\nModelos, columnas de features y metadatos guardados en /models")
    return results


if __name__ == "__main__":
    print("=== FA Cup ML Trainer (TheStatsAPI) ===")
    print(f"Fecha: {datetime.utcnow().isoformat()}Z | MAX_SEASONS={MAX_SEASONS}\n")
    dataset = build_dataset()
    if dataset.empty:
        raise SystemExit("No se obtuvieron partidos. Revisa API key y cobertura de FA Cup.")
    dataset = engineer_features(dataset)
    dataset.to_csv("fa_cup_dataset.csv", index=False)
    print("Dataset guardado: fa_cup_dataset.csv")
    results = train_models(dataset)
    print("\nResumen final:")
    for name, metrics in results.items():
        print(f"  {name}: Acc={metrics['accuracy']:.4f} | LL={metrics['log_loss']:.4f} | Brier={metrics['brier_score']:.4f}")
