"""
Evaluate baseline World Cup model on the actual 2022 results.
"""

from pathlib import Path
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import TrainConfig, DATA_RAW_DIR
from src.cleaning import run_clean_and_save
from src.modeling import train_baseline
from src.elo import build_elo_timeseries


def load_actual_2022_results():
    """Load real match-level results for the 2022 World Cup using correct Fjelstul columns."""
    df = pd.read_csv(DATA_RAW_DIR / "matches.csv")

    df_2022 = df[df["tournament_id"] == "WC-2022"].copy()

    # Create labels using correct column names
    def label(row):
        if row["home_team_score"] > row["away_team_score"]:
            return "win"
        elif row["home_team_score"] < row["away_team_score"]:
            return "loss"
        else:
            return "draw"

    df_2022["label"] = df_2022.apply(label, axis=1)
    return df_2022


def build_feature_frame(model_bundle, df_2022, matches_clean):
    """Build features using raw Fjelstul team_code columns."""
    elo_df = build_elo_timeseries(matches_clean)

    home_elos = elo_df[["team_home", "elo_home_pre", "date"]].rename(
        columns={"team_home": "team", "elo_home_pre": "elo_pre"}
    )
    away_elos = elo_df[["team_away", "elo_away_pre", "date"]].rename(
        columns={"team_away": "team", "elo_away_pre": "elo_pre"}
    )

    elos = pd.concat([home_elos, away_elos], ignore_index=True)

    final_elo = (
        elos.sort_values("date")
            .groupby("team")["elo_pre"]
            .last()
            .to_dict()
    )

    rows = []
    for _, row in df_2022.iterrows():
        home = row["home_team_code"]
        away = row["away_team_code"]

        r_home = final_elo.get(home, 1500)
        r_away = final_elo.get(away, 1500)

        rows.append({
            "elo_home_pre": r_home,
            "elo_away_pre": r_away,
            "elo_delta_pre": r_home - r_away,
            "is_knockout": int(row["knockout_stage"]),
        })

    return pd.DataFrame(rows)


def main():
    matches_clean = run_clean_and_save()
    cfg = TrainConfig()

    model_bundle = train_baseline(matches_clean, cfg)
    model = model_bundle["model"]
    classes = model_bundle["classes"]

    df_2022 = load_actual_2022_results()

    X_2022 = build_feature_frame(model_bundle, df_2022, matches_clean)
    y_true = df_2022["label"]

    preds = model.predict(X_2022)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average="macro", zero_division=0)
    rec = recall_score(y_true, preds, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=classes)

    print("\n=== MODEL EVALUATION ON 2022 WORLD CUP ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("\nLabels:", classes)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
