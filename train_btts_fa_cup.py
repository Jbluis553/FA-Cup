"""Entrena modelos para el mercado BTTS Yes (ambos equipos marcan)."""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_fa_cup import LGBMClassifier, build_dataset, engineer_features

RANDOM_STATE = 42
MODEL_DIR = os.path.join("models", "btts")


def train_btts_models(df: pd.DataFrame):
    """Entrena un clasificador binario cuyo positivo es que ambos marquen."""
    feature_cols = [
        c for c in df.columns
        if c.startswith(("home_", "away_", "form_diff_", "xg_diff_"))
    ]
    feature_cols += [
        "rest_days_diff", "congestion_diff", "level_diff", "neutral_venue",
        "market_home_prob", "market_overround", "market_home_odds",
    ]
    feature_cols = list(dict.fromkeys(c for c in feature_cols if c in df.columns))

    data = df.dropna(subset=["home_goals", "away_goals"]).copy().sort_values("utc_date")
    X = data[feature_cols].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    y = ((pd.to_numeric(data["home_goals"], errors="coerce") > 0) &
         (pd.to_numeric(data["away_goals"], errors="coerce") > 0)).astype(int)
    if len(data) < 30 or y.nunique() < 2:
        raise ValueError(f"Datos insuficientes para entrenar BTTS Yes ({len(data)} partidos).")

    split = max(int(len(data) * 0.8), 1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_test) == 0 or y_train.nunique() < 2:
        raise ValueError("El split temporal no permite evaluar ambas clases; descarga más temporadas.")

    cv = TimeSeriesSplit(n_splits=min(3, max(2, len(X_train) // 20)))
    models = {
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        )
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            LGBMClassifier(
                n_estimators=250, max_depth=5, learning_rate=0.035, num_leaves=15,
                class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1,
            ),
        )
    else:
        warnings.warn("LightGBM no está instalado; solo se entrenará Logistic Regression.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    fitted_models, probabilities, results = {}, {}, {}
    for name, estimator in models.items():
        print(f"\nEntrenando modelo BTTS Yes: {name}...")
        if len(X_train) >= 45:
            model = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
        else:
            warnings.warn("Pocos ejemplos: se omite calibración cruzada para este dataset.")
            model = estimator
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        metrics = {
            "accuracy": accuracy_score(y_test, prob >= 0.5),
            "log_loss": log_loss(y_test, prob, labels=[0, 1]),
            "brier_score": brier_score_loss(y_test, prob),
        }
        print(f"  Accuracy: {metrics['accuracy']:.4f} | LogLoss: {metrics['log_loss']:.4f} | Brier: {metrics['brier_score']:.4f}")
        fitted_models[name] = model
        probabilities[name] = prob
        results[name] = metrics
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name.lower()}.joblib"))

    if len(fitted_models) > 1:
        ensemble_prob = np.mean(np.column_stack(list(probabilities.values())), axis=1)
        results["Ensemble"] = {
            "accuracy": accuracy_score(y_test, ensemble_prob >= 0.5),
            "log_loss": log_loss(y_test, ensemble_prob, labels=[0, 1]),
            "brier_score": brier_score_loss(y_test, ensemble_prob),
        }
        joblib.dump(fitted_models, os.path.join(MODEL_DIR, "ensemble_components.joblib"))
        joblib.dump("mean calibrated probabilities", os.path.join(MODEL_DIR, "best_model.joblib"))
    else:
        joblib.dump(next(iter(fitted_models.values())), os.path.join(MODEL_DIR, "best_model.joblib"))

    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.joblib"))
    joblib.dump(
        {"target": "btts_yes", "positive_class": 1,
         "definition": "home_goals > 0 and away_goals > 0"},
        os.path.join(MODEL_DIR, "target_metadata.joblib"),
    )
    print(f"\nModelos BTTS Yes guardados en {MODEL_DIR}/")
    return results


if __name__ == "__main__":
    dataset = build_dataset()
    if dataset.empty:
        raise SystemExit("No se obtuvieron partidos. Revisa API key y cobertura de FA Cup.")
    dataset = engineer_features(dataset)
    dataset.to_csv("fa_cup_dataset.csv", index=False)
    print("Dataset guardado: fa_cup_dataset.csv")
    metrics = train_btts_models(dataset)
    print("\nResumen BTTS Yes:")
    for name, values in metrics.items():
        print(f"  {name}: Acc={values['accuracy']:.4f} | LL={values['log_loss']:.4f} | Brier={values['brier_score']:.4f}")
