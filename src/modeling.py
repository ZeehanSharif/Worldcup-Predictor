"""
Baseline model (midpoint):
- Predict match_result_1x2 (win/draw/loss from home POV)
- Features: elo_home_pre, elo_away_pre, elo_delta_pre, is_knockout
- Model: simple logistic-style classifier (we'll just use multinomial LogisticRegression)

No calibration, no SHAP, no hyperparam search yet.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from .config import TrainConfig
from .features import engineer_features, temporal_train_test_split

FEATURE_COLS = [
    "elo_home_pre",
    "elo_away_pre",
    "elo_delta_pre",
    "is_knockout",
]

TARGET_COL = "match_result_1x2"

def train_baseline(matches_clean: pd.DataFrame, cfg: TrainConfig):
    feat_df = engineer_features(matches_clean)
    train_df, test_df = temporal_train_test_split(feat_df, cfg.test_year)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(str)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].astype(str)

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    preproc = ColumnTransformer(
        transformers=[("num", numeric_transformer, FEATURE_COLS)],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=500,
        random_state=cfg.random_state
    )

    pipe = Pipeline(
        steps=[
            ("prep", preproc),
            ("logreg", clf),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["win","draw","loss"])

    model_bundle = {
        "model": pipe,
        "feature_cols": FEATURE_COLS,
        "classes": list(pipe.classes_),
        "holdout_report": report,
        "holdout_confusion": cm,
        "holdout_year": cfg.test_year,
    }

    return model_bundle
