"""
Simple script to predict top 10 contenders for 2026 World Cup using our
simple logistic-style classifier model.

Uses model predictions (win probability - loss probability) across all
historical matches to compute team strength scores and rank teams.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.cleaning import run_clean_and_save
from src.config import TrainConfig, TEAMS_CSV
from src.modeling import train_baseline
from src.features import engineer_features

def compute_team_strengths(model_bundle, matches_clean):
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    class_order = model_bundle["classes"]  # e.g. ["draw","loss","win"]

    # build feature dataframe
    feat_df = engineer_features(matches_clean)
    X = feat_df[feature_cols]

    # predicted class probabilities for every historical match
    probs = model.predict_proba(X)

    # Map class indices → probabilities
    # Example mapping: {"win": 2, "draw": 0, "loss": 1}
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    # strength metric:
    #   win_prob - loss_prob
    # gives a simple continuous strength score
    win_prob = probs[:, class_to_idx["win"]]
    loss_prob = probs[:, class_to_idx["loss"]]
    strength = win_prob - loss_prob 

    team_strength = {}

    for (_, row), s in zip(feat_df.iterrows(), strength):
        home = row["team_home"]
        away = row["team_away"]

        # add strength contribution to home/away
        team_strength[home] = team_strength.get(home, 0) + s
        team_strength[away] = team_strength.get(away, 0) - s

    return team_strength


if __name__ == "__main__":
    # clean data
    matches_clean = run_clean_and_save()

    # train model
    cfg = TrainConfig(test_year=2022)
    model_bundle = train_baseline(matches_clean, cfg)
    team_strengths = compute_team_strengths(model_bundle, matches_clean)

    teams_df = pd.read_csv(TEAMS_CSV)
    name_map = dict(zip(teams_df["team_code"], teams_df["team_name"]))

    df = pd.DataFrame([
        {"team_code": t, "strength": s, "team_name": name_map.get(t, t)}
        for t, s in team_strengths.items()
    ])
    df = df.sort_values("strength", ascending=False)

    # find top 10 contenders
    top10 = df.head(10)

    # visualization graph
    top10_display = top10
    print("\nTop 10 Contenders Based on Model Predictions:")
    print("-" * 50)
    for rank, (idx, row) in enumerate(top10.iterrows(), 1):
        print(f"{rank:2d}. {row['team_name']:20s} - Strength: {row['strength']:.3f}")
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top10, x="strength", y="team_name")
    plt.title("Top 10 World Cup Contenders (Model-Based Ranking)")
    plt.xlabel("Predicted Strength Score (strength = win_prob - loss_prob)")
    plt.ylabel("Team")
    plt.show()

