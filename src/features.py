"""
Midpoint feature engineering.

Right now we only use:
- Elo before the match
- Elo delta
- Whether match is knockout vs group (context)
- The label (win/draw/loss)

We are NOT yet adding:
- recent form windows
- rest days
- injuries
- travel / climate

Those will be "future work" for final project.
"""

import pandas as pd
from .elo import build_elo_timeseries

def engineer_features(matches_clean: pd.DataFrame) -> pd.DataFrame:
    df = build_elo_timeseries(matches_clean)

    # knockout flag
    df["is_knockout"] = df["stage"].str.contains(
        "round|quarter|semi|final",
        case=False,
        na=False
    ).astype(int)
    
    # final label column already exists as 'match_result_1x2'
    feat_df = df.copy()

    return feat_df

def temporal_train_test_split(feat_df: pd.DataFrame, test_year: int):
    train_df = feat_df[feat_df["year"] != test_year].copy()
    test_df  = feat_df[feat_df["year"] == test_year].copy()
    return train_df, test_df
