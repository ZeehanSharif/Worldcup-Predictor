"""
Very simple rolling Elo for national teams.

We're using Elo because prior work and football analytics communities use Elo-like ratings to measure team strength for match prediction. 

This gives us 'elo_home_pre' and 'elo_away_pre' as features.
"""

import pandas as pd
import numpy as np

BASE_ELO = 1500.0
K = 20.0  # modest sensitivity

def _expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** (-(r_a - r_b) / 400.0))

def _score(home_goals, away_goals):
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5

def build_elo_timeseries(df_matches: pd.DataFrame) -> pd.DataFrame:
    ratings = {}
    elo_home_pre = []
    elo_away_pre = []

    elo_home_post = []
    elo_away_post = []

    for _, row in df_matches.iterrows():
        h = row["team_home"]
        a = row["team_away"]

        r_h = ratings.get(h, BASE_ELO)
        r_a = ratings.get(a, BASE_ELO)

        elo_home_pre.append(r_h)
        elo_away_pre.append(r_a)

        exp_h = _expected(r_h, r_a)
        score_h, score_a = _score(row["score_home"], row["score_away"])

        delta = K * (score_h - exp_h)
        r_h_new = r_h + delta
        r_a_new = r_a - delta

        elo_home_post.append(r_h_new)
        elo_away_post.append(r_a_new)

        ratings[h] = r_h_new
        ratings[a] = r_a_new

    out = df_matches.copy()
    out["elo_home_pre"] = elo_home_pre
    out["elo_away_pre"] = elo_away_pre
    out["elo_delta_pre"] = out["elo_home_pre"] - out["elo_away_pre"]
    return out
