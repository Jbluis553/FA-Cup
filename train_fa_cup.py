"""
Entrenamiento de modelos ML para predecir partidos de la FA Cup
usando TheStatsAPI (https://www.thestatsapi.com)
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime

# ============================================================
# CONFIGURACIÓN
# ============================================================
API_KEY = os.getenv("THESTATSAPI_API_KEY")
BASE_URL = "https://api.thestatsapi.com/api"
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

# Cuántas temporadas históricas descargar (ajusta según tu plan/quota)
MAX_SEASONS = 6
PER_PAGE = 100
SLEEP_BETWEEN_CALLS = 0.35   # respeta rate limits

# ============================================================
# HELPERS DE LA API
# ============================================================
def api_get(endpoint: str, params: dict = None):
    """Hace GET a TheStatsAPI con manejo básico de errores y rate-limit."""
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)

    if response.status_code == 429:
        retry = int(response.headers.get("Retry-After", 60))
        print(f"Rate limit. Esperando {retry}s...")
        time.sleep(retry)
        return api_get(endpoint, params)

    if response.status_code != 200:
        print(f"Error {response.status_code} en {endpoint}: {response.text[:300]}")
        return None

    return response.json()

def find_fa_cup_competition():
    """Busca el competition_id de la FA Cup."""
    data = api_get("/football/competitions", {
        "search": "FA Cup",
        "country_code": "GB-ENG",
        "type": "cup",
        "per_page": 10
    })
    if not data or not data.get("data"):
        # Fallback más amplio
        data = api_get("/football/competitions", {"search": "FA Cup", "per_page": 10})

    if not data or not data.get("data"):
        raise ValueError("No se encontró la FA Cup. Revisa tu API key o cobertura.")

    # Preferimos el que contenga "FA Cup" y sea de Inglaterra
    for comp in data["data"]:
        name = comp.get("name", "").lower()
        if "fa cup" in name and "women" not in name:
            print(f"Encontrada: {comp['name']} → {comp['id']}")
            return comp["id"]

    # Si no, cogemos el primero
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
    all_matches = []
    page = 1

    while True:
        data = api_get("/football/matches", {
            "competition_id": competition_id,
            "season_id": season_id,
            "status": status,
            "per_page": PER_PAGE,
            "page": page,
            "sort": "utc_date"
        })
        if not data:
            break

        matches = data.get("data", [])
        if not matches:
            break

        all_matches.extend(matches)
        meta = data.get("meta", {})
        total_pages = meta.get("total_pages", 1)
        print(f"  Página {page}/{total_pages} → {len(matches)} partidos")

        if page >= total_pages:
            break
        page += 1
        time.sleep(SLEEP_BETWEEN_CALLS)

    return all_matches

def fetch_match_stats(match_id: str):
    """Obtiene estadísticas del partido (si están disponibles)."""
    data = api_get(f"/football/matches/{match_id}/stats")
    if not data or "data" not in data:
        return None
    return data["data"]

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
        season_name = season.get("name") or season.get("year")
        print(f"\nDescargando temporada {season_name} ({season_id})...")

        matches = fetch_all_matches(competition_id, season_id)
        print(f"  Total partidos: {len(matches)}")

        for m in matches:
            score = m.get("score") or {}
            home_goals = score.get("home")
            away_goals = score.get("away")
            winner = score.get("winner")  # "home", "away", "draw"

            if home_goals is None or away_goals is None:
                continue

            # Resultado final
            if winner == "home" or (winner is None and home_goals > away_goals):
                result = "H"
            elif winner == "away" or (winner is None and away_goals > home_goals):
                result = "A"
            else:
                result = "D"

            home = m.get("home_team") or {}
            away = m.get("away_team") or {}

            row = {
                "match_id": m.get("id"),
                "season_id": season_id,
                "utc_date": m.get("utc_date"),
                "home_team_id": home.get("id"),
                "home_team": home.get("name"),
                "away_team_id": away.get("id"),
                "away_team": away.get("name"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "went_to_extra_time": score.get("went_to_extra_time", False),
                "went_to_penalties": score.get("went_to_penalties", False),
            }
            rows.append(row)

        time.sleep(SLEEP_BETWEEN_CALLS)

    df = pd.DataFrame(rows)
    print(f"\nDataset bruto: {len(df)} partidos")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features sencillas pero útiles para FA Cup."""
    df = df.copy()
    df["utc_date"] = pd.to_datetime(df["utc_date"])
    df = df.sort_values("utc_date").reset_index(drop=True)

    # Forma reciente (últimos 5 partidos de cada equipo)
    def compute_form(team_col, goals_for_col, goals_against_col, window=5):
        form_pts = []
        form_gf = []
        form_ga = []
        for i, row in df.iterrows():
            team = row[team_col]
            past = df[(df["home_team"] == team) | (df["away_team"] == team)].loc[:i-1].tail(window)
            if len(past) == 0:
                form_pts.append(1.0)
                form_gf.append(1.2)
                form_ga.append(1.2)
                continue
            pts = 0
            gf = 0
            ga = 0
            for _, p in past.iterrows():
                if p["home_team"] == team:
                    gf += p["home_goals"]
                    ga += p["away_goals"]
                    if p["result"] == "H":
                        pts += 3
                    elif p["result"] == "D":
                        pts += 1
                else:
                    gf += p["away_goals"]
                    ga += p["home_goals"]
                    if p["result"] == "A":
                        pts += 3
                    elif p["result"] == "D":
                        pts += 1
            n = len(past)
            form_pts.append(pts / n)
            form_gf.append(gf / n)
            form_ga.append(ga / n)
        return form_pts, form_gf, form_ga

    print("Calculando forma de equipos...")
    df["home_form_pts"], df["home_gf_avg"], df["home_ga_avg"] = compute_form(
        "home_team", "home_goals", "away_goals"
    )
    df["away_form_pts"], df["away_gf_avg"], df["away_ga_avg"] = compute_form(
        "away_team", "away_goals", "home_goals"
    )

    df["form_diff"] = df["home_form_pts"] - df["away_form_pts"]
    df["gf_diff"] = df["home_gf_avg"] - df["away_gf_avg"]
    df["ga_diff"] = df["away_ga_avg"] - df["home_ga_avg"]  # positivo = local concede menos

    # Features de interacción
    df["form_x_gf"] = df["form_diff"] * df["gf_diff"]

    return df

# ============================================================
# ENTRENAMIENTO
# ============================================================
def train_models(df: pd.DataFrame):
    feature_cols = [
        "home_form_pts", "away_form_pts", "form_diff",
        "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
        "gf_diff", "ga_diff", "form_x_gf"
    ]

    # Eliminar filas con NaN en features
    df = df.dropna(subset=feature_cols + ["result"]).copy()
    print(f"Partidos con features válidas: {len(df)}")

    X = df[feature_cols]
    y = df["result"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split temporal (últimos 20% como test)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_enc[:split], y_enc[split:]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            objective="multi:softprob", num_class=3, random_state=42, n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            objective="multiclass", num_class=3, random_state=42, verbose=-1
        )
    }

    results = {}
    best_model = None
    best_name = None
    best_acc = 0

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        print(f"\nEntrenando {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, proba)

        print(f"  Accuracy: {acc:.4f} | LogLoss: {ll:.4f}")
        print(classification_report(y_test, preds, target_names=le.classes_))

        results[name] = {"accuracy": acc, "log_loss": ll}
        joblib.dump(model, f"models/{name.lower()}.joblib")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # Guardar el mejor y el encoder
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(le, "models/label_encoder.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")

    print("\n" + "="*50)
    print(f"Mejor modelo: {best_name} (Accuracy {best_acc:.4f})")
    print("Modelos guardados en carpeta /models")
    print("="*50)

    return results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=== FA Cup ML Trainer (TheStatsAPI) ===")
    print(f"Fecha: {datetime.utcnow().isoformat()}Z\n")

    df = build_dataset()
    if df.empty:
        raise SystemExit("No se obtuvieron partidos. Revisa API key y cobertura de FA Cup.")

    df = engineer_features(df)
    df.to_csv("fa_cup_dataset.csv", index=False)
    print("Dataset guardado: fa_cup_dataset.csv")

    results = train_models(df)
    print("\nResumen final:")
    for name, metrics in results.items():
        print(f"  {name}: Acc={metrics['accuracy']:.4f} | LL={metrics['log_loss']:.4f}")
