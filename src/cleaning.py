"""
Cleaning step for the World Cup forecasting project.

This cleaned table is the single source of truth for downstream steps (elo.py, features.py, modeling.py).
"""

import pandas as pd
from . import config, data_loader


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw CSV columns into the internal names we use everywhere else. makes things easy.
    """

    rename_map = {
        "home_team_code": "team_home",
        "away_team_code": "team_away",
        "home_team_score": "score_home",
        "away_team_score": "score_away",
        "stage_name": "stage",
        "group_name": "group",
        "match_date": "date",
        "city_name": "city",
        "country_name": "country",
        "stadium_name": "stadium",
        
        # pass through flags for later use if needed
        "home_team_win": "home_team_win",
        "away_team_win": "away_team_win",
        "draw": "draw",
        "extra_time": "extra_time",
        "penalty_shootout": "penalty_shootout",
    }

    # only rename what actually exists in the incoming df
    apply_map = {old: new for old, new in rename_map.items() if old in df.columns}
    out = df.rename(columns=apply_map).copy()

    # parse the match date into a real datetime
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # derive numeric year from the date
    if "date" in out.columns:
        out["year"] = out["date"].dt.year

    return out


def _add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add outcome info needed for modeling:
    - goal_diff_home = score_home - score_away
    - match_result_1x2 in {"win", "loss", "draw"} from the home team's perspective

    If those are missing or NaN for any row, we fall back to score comparison.
    """

    out = df.copy()

    # compute margin
    out["goal_diff_home"] = out["score_home"] - out["score_away"]

    def infer_result(row):
        # try explicit flags first if they exist
        if "home_team_win" in row and pd.notna(row["home_team_win"]) and row["home_team_win"] == 1:
            return "win"
        if "away_team_win" in row and pd.notna(row["away_team_win"]) and row["away_team_win"] == 1:
            return "loss"
        if "draw" in row and pd.notna(row["draw"]) and row["draw"] == 1:
            return "draw"

        # fallback: infer from score
        if row["score_home"] > row["score_away"]:
            return "win"
        elif row["score_home"] < row["score_away"]:
            return "loss"
        else:
            return "draw"

    out["match_result_1x2"] = out.apply(infer_result, axis=1)

    return out


def clean_matches() -> pd.DataFrame:
    """
    Core cleaning routine:
    - load raw matches.csv
    - standardize column names and types
    - require key fields to exist
    - filter to men's World Cup finals era: 1930–2022
    - add derived outcomes
    - sort chronologically
    """

    raw = data_loader.load_matches()
    df = _standardize(raw)

    # required columns for downstream steps
    required_cols = [
        "team_home",
        "team_away",
        "score_home",
        "score_away",
        "date",
        "year",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"After standardization, missing required columns: {missing}. "
            "Check that matches.csv column names match what cleaning._standardize() expects."
        )
    # removing west germany
    df = df[df["team_home"] != "DEU"]
    df = df[df["team_away"] != "DEU"]
    
    # drop rows with no valid date/year/team/score
    df = df.dropna(subset=["year", "team_home", "team_away", "score_home", "score_away"])
    df["year"] = df["year"].astype(int)

    # World Cup men's finals era range
    df = df[(df["year"] >= 1930) & (df["year"] <= 2022)]

    # outcome labels
    df = _add_outcomes(df)

    # sort in chronological order
    df = df.sort_values(["year", "date"]).reset_index(drop=True)
    
    # removing women tournaments
    df = df[~df["tournament_name"].str.contains("Women")]
    
    # removing teams that don't exist
    out_teams = ["West Germany", "Czechoslovakia", "Yugoslavia", "Soviet Union", "USSR", "Zaire", "Dutch East Indies", "Serbia and Montenegro", "Bohemia", "United Arab Republic","East Germany", "Northern Rhodesia", "Rhodesia","Republic of China"]
    df = df[~df['home_team_name'].isin(out_teams) & ~df['away_team_name'].isin(out_teams)]
    return df




def run_clean_and_save() -> pd.DataFrame:
    """
    Convenience entry point:
    - runs clean_matches()
    - writes data/processed/matches_clean.parquet
    - returns the cleaned DataFrame
    """

    df = clean_matches()

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MATCHES_CLEAN_PARQUET, index=False)

    return df
