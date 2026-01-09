"""
Utility script to print the *actual* 2022 World Cup results
from the Fjelstul World Cup Database (group standings + knockout bracket).

This does NOT use the model — it just reads the raw data and formats it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW_DIR  


def load_2022_group_standings() -> pd.DataFrame:
    path = DATA_RAW_DIR / "group_standings.csv"
    df = pd.read_csv(path)

    df_2022 = df[df["tournament_id"] == "WC-2022"].copy()
    df_2022 = df_2022.sort_values(["group_name", "position"]).reset_index(drop=True)
    return df_2022


def load_2022_knockout_matches() -> pd.DataFrame:
    """
    Load knockout-stage matches for the 2022 World Cup from matches.csv.
    Sort them in the correct bracket order.
    """
    path = DATA_RAW_DIR / "matches.csv"
    df = pd.read_csv(path)

    # 2022 knockout matches only
    df_2022 = df[(df["tournament_id"] == "WC-2022") & (df["knockout_stage"] == 1)].copy()

    keep_cols = [
        "stage_name",
        "match_date",
        "home_team_name",
        "away_team_name",
        "home_team_score",
        "away_team_score",
        "extra_time",
        "penalty_shootout",
        "score_penalties",
    ]
    keep_cols = [c for c in keep_cols if c in df_2022.columns]
    df_2022 = df_2022[keep_cols]

    # Normalize stage names
    df_2022["stage_name"] = df_2022["stage_name"].str.lower().str.strip()

    # Fully robust stage-order mapping
    STAGE_ORDER = {
        "round of 16": 1,
        "round-of-16": 1,
        "round_of_16": 1,

        "quarter-finals": 2,
        "quarter-final": 2,
        "quarterfinal": 2,
        "quarterfinals": 2,

        "semi-finals": 3,
        "semi-final": 3,
        "semifinals": 3,
        "semifinal": 3,

        "third place": 4,
        "third-place": 4,
        "third-place match": 4,
        "third place playoff": 4,

        "final": 5
    }

    df_2022["stage_sort"] = df_2022["stage_name"].map(STAGE_ORDER)

    # Sort in correct knockout order
    df_2022 = df_2022.sort_values(["stage_sort", "match_date"]).reset_index(drop=True)
    df_2022.drop(columns=["stage_sort"], inplace=True)

    return df_2022


def print_group_standings(groups_2022: pd.DataFrame) -> None:
    print("\n=== 2022 WORLD CUP GROUP STANDINGS ===")
    for group_name, sub in groups_2022.groupby("group_name"):
        sub = sub.sort_values("position")
        print(f"\n{group_name}")
        print("-" * len(group_name))

        cols = [
            "position",
            "team_name",
            "team_code",
            "played",
            "wins",
            "draws",
            "losses",
            "points",
        ]
        existing_cols = [c for c in cols if c in sub.columns]
        print(sub[existing_cols].to_string(index=False))


def print_bracket(df):
    """
    Pretty-print the knockout bracket in correct order.
    Winner is computed directly from the dataset.
    """
    print("\n=== 2022 WORLD CUP KNOCKOUT STAGE ===")

    def get_winner(row):
        if row['home_team_score'] > row['away_team_score']:
            return row['home_team_name']
        elif row['home_team_score'] < row['away_team_score']:
            return row['away_team_name']
        else:
            if row['penalty_shootout'] == 1:
                pen = str(row['score_penalties']).replace("–", "-")
                home_pen, away_pen = map(int, pen.split("-"))
                return row['home_team_name'] if home_pen > away_pen else row['away_team_name']
            return "Draw"

    for idx, row in df.iterrows():
        winner = get_winner(row)

        print(
            f"{idx:<2}  "
            f"{row['stage_name']:<18} "
            f"{row['home_team_name']:<15} "
            f"{row['away_team_name']:<15} "
            f"{winner}"
        )

    final_row = df[df["stage_name"] == "final"].iloc[0]
    print("\nChampion:", get_winner(final_row))



if __name__ == "__main__":
    groups_2022 = load_2022_group_standings()
    knockouts_2022 = load_2022_knockout_matches()

    print_group_standings(groups_2022)
    print_bracket(knockouts_2022)
