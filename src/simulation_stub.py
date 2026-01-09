"""
Simple tournament simulation for the World Cup forecasting project.

This is intentionally lightweight and designed to plug into the existing
pipeline without changing other modules. It uses:
- the baseline logistic model trained on Elo-based features, and
- a simple Elo snapshot per team at the end of our historical data.

From these pieces we can simulate:
- group stages (round robin, 3/1/0 points),
- a knockout bracket built from the group winners.

The design is deliberately minimal and easy to extend in future work.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running this file directly as a script:
#   python -m src.simulation_stub
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import run_clean_and_save
from src.config import TrainConfig, DATA_RAW_DIR, TEAMS_CSV
from src.modeling import train_baseline
from src.elo import BASE_ELO, K, _expected, _score  # reuse Elo helpers


@dataclass
class MatchSimulationResult:
    """Container for a single simulated match."""
    home_team: str
    away_team: str
    result: str          # "win" / "loss" / "draw" from home-team POV
    winner: Optional[str]
    loser: Optional[str]
    probs: Dict[str, float]


def compute_final_elo_ratings(matches_clean: pd.DataFrame) -> Dict[str, float]:
    """
    Compute per-team Elo rating after the last historical match.

    This mirrors the Elo update logic in src.elo.build_elo_timeseries but
    returns only the final rating for each team, which we then treat as a
    "current strength" prior for tournament simulation.
    """
    ratings: Dict[str, float] = {}

    for _, row in matches_clean.iterrows():
        home = row["team_home"]
        away = row["team_away"]

        r_h = ratings.get(home, BASE_ELO)
        r_a = ratings.get(away, BASE_ELO)

        exp_h = _expected(r_h, r_a)
        score_h, score_a = _score(row["score_home"], row["score_away"])

        delta = K * (score_h - exp_h)
        ratings[home] = r_h + delta
        ratings[away] = r_a - delta

    return ratings


def simulate_match(
    model_bundle: Dict,
    elo_ratings: Dict[str, float],
    home_team: str,
    away_team: str,
    is_knockout: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> MatchSimulationResult:
    """
    Simulate a single match using the trained baseline model.

    Parameters
    ----------
    model_bundle:
        Dict returned by modeling.train_baseline (contains "model" and "classes").
    elo_ratings:
        Mapping team_code -> Elo rating prior to the tournament.
    home_team, away_team:
        Team codes of the sides.
    is_knockout:
        If True, draws are resolved into a winner using only the
        model's win/loss probabilities.
    rng:
        Optional NumPy Generator for reproducibility.

    Returns
    -------
    MatchSimulationResult with the sampled result and probabilities.
    """
    if rng is None:
        rng = np.random.default_rng()

    model = model_bundle["model"]
    classes: List[str] = model_bundle["classes"]

    r_home = elo_ratings.get(home_team, BASE_ELO)
    r_away = elo_ratings.get(away_team, BASE_ELO)

    feat_row = pd.DataFrame(
        [
            {
                "elo_home_pre": r_home,
                "elo_away_pre": r_away,
                "elo_delta_pre": r_home - r_away,
                "is_knockout": 1 if is_knockout else 0,
            }
        ]
    )

    probs_arr = model.predict_proba(feat_row)[0]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    probs = {c: float(probs_arr[i]) for c, i in class_to_idx.items()}

    # sample outcome from the model probabilities
    result = rng.choice(classes, p=probs_arr)

    # For knockout matches, we cannot end in a draw; re-sample between win/loss.
    if is_knockout and result == "draw":
        p_win = probs["win"]
        p_loss = probs["loss"]
        total = p_win + p_loss
        if total <= 0:
            # degenerate case: fall back to 50/50
            p_win = p_loss = 0.5
        else:
            p_win /= total
            p_loss /= total
        result = rng.choice(["win", "loss"], p=[p_win, p_loss])

    if result == "win":
        winner = home_team
        loser = away_team
    elif result == "loss":
        winner = away_team
        loser = home_team
    else:
        winner = None
        loser = None

    return MatchSimulationResult(
        home_team=home_team,
        away_team=away_team,
        result=result,
        winner=winner,
        loser=loser,
        probs=probs,
    )


def load_2022_groups() -> Dict[str, List[str]]:
    """
    Load 2022 group compositions from the Fjelstul group_standings.csv file.

    Returns a mapping: group name (e.g. "Group A") -> list of team codes.
    """
    path = DATA_RAW_DIR / "group_standings.csv"
    df = pd.read_csv(path)
    df_2022 = df[df["tournament_id"] == "WC-2022"].copy()
    groups: Dict[str, List[str]] = {}
    for group_name, sub in df_2022.groupby("group_name"):
        # sort by official position so 1st/2nd are meaningful
        sub = sub.sort_values("position")
        groups[group_name] = sub["team_code"].tolist()
    return groups


def load_team_name_mapping() -> Dict[str, str]:
    """
    Load the mapping from team codes to full team names.

    Returns a mapping: team_code -> team_name.
    """
    df = pd.read_csv(TEAMS_CSV)
    team_map = dict(zip(df["team_code"], df["team_name"]))
    # fix because there is 2 different DEU's
    team_map["DEU"] = "Germany"
    return team_map


def simulate_group(
    group_name: str,
    teams: Iterable[str],
    model_bundle: Dict,
    elo_ratings: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Simulate a single round-robin group.

    Each pair of teams plays once. We keep a simple table with:
    played, wins, draws, losses, and points (3/1/0).
    """
    if rng is None:
        rng = np.random.default_rng()

    teams = list(teams)
    records = {
        team: {
            "team": team,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
        }
        for team in teams
    }

    for i, home in enumerate(teams):
        for away in teams[i + 1 :]:
            res = simulate_match(
                model_bundle=model_bundle,
                elo_ratings=elo_ratings,
                home_team=home,
                away_team=away,
                is_knockout=False,
                rng=rng,
            )
            records[home]["played"] += 1
            records[away]["played"] += 1

            if res.result == "win":
                records[home]["wins"] += 1
                records[away]["losses"] += 1
                records[home]["points"] += 3
            elif res.result == "loss":
                records[away]["wins"] += 1
                records[home]["losses"] += 1
                records[away]["points"] += 3
            else:
                records[home]["draws"] += 1
                records[away]["draws"] += 1
                records[home]["points"] += 1
                records[away]["points"] += 1

    table = pd.DataFrame.from_records(list(records.values()))
    table["group"] = group_name

    # sort by points, then wins as a simple tie-breaker
    table = table.sort_values(
        ["group", "points", "wins"], ascending=[True, False, False]
    ).reset_index(drop=True)
    return table


def simulate_group_stage(
    groups: Dict[str, List[str]],
    model_bundle: Dict,
    elo_ratings: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Simulate all groups and return:
    - full group table, and
    - list of teams that advance (top two from each group).
    """
    all_tables = []
    advancing: List[str] = []

    for group_name, team_list in sorted(groups.items()):
        table = simulate_group(
            group_name=group_name,
            teams=team_list,
            model_bundle=model_bundle,
            elo_ratings=elo_ratings,
            rng=rng,
        )
        all_tables.append(table)
        advancing.extend(table.head(2)["team"].tolist())

    full_table = pd.concat(all_tables, ignore_index=True)
    return full_table, advancing


def simulate_knockout_bracket(
    teams: List[str],
    model_bundle: Dict,
    elo_ratings: Dict[str, float],
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Simulate a simple single-elimination bracket.

    Teams are paired in the given order (0 vs 1, 2 vs 3, ...).
    This is not trying to exactly match FIFA's official bracket logic;
    it is only a transparent example of using the model to simulate
    a sequence of knockout rounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    round_names = ["Round of 16", "Quarterfinals", "Semifinals", "Final"]
    current = list(teams)
    rows = []

    for round_name in round_names:
        next_round: List[str] = []
        for i in range(0, len(current), 2):
            if i + 1 >= len(current):
                # odd team count, carry forward automatically
                next_round.append(current[i])
                continue
            home = current[i]
            away = current[i + 1]
            res = simulate_match(
                model_bundle=model_bundle,
                elo_ratings=elo_ratings,
                home_team=home,
                away_team=away,
                is_knockout=True,
                rng=rng,
            )
            rows.append(
                {
                    "round": round_name,
                    "home_team": res.home_team,
                    "away_team": res.away_team,
                    "result": res.result,
                    "winner": res.winner,
                    "loser": res.loser,
                    **{f"p_{k}": v for k, v in res.probs.items()},
                }
            )
            if res.winner is not None:
                next_round.append(res.winner)

        current = next_round
        if len(current) <= 1:
            break

    return pd.DataFrame(rows)


def simulate_world_cup_2022(
    model_bundle: Dict,
    matches_clean: pd.DataFrame,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level helper that wires together:
    - final Elo ratings from the historical data
    - 2022 group composition
    - group + knockout simulation

    Returns
    -------
    group_table:
        Combined group stage table for all 8 groups.
    knockout_results:
        DataFrame of simulated knockout matches.
    """
    rng = np.random.default_rng(random_state)
    elo_ratings = compute_final_elo_ratings(matches_clean)
    groups = load_2022_groups()

    group_table, advancers = simulate_group_stage(
        groups=groups,
        model_bundle=model_bundle,
        elo_ratings=elo_ratings,
        rng=rng,
    )

    knockout_results = simulate_knockout_bracket(
        teams=advancers,
        model_bundle=model_bundle,
        elo_ratings=elo_ratings,
        rng=rng,
    )

    return group_table, knockout_results


if __name__ == "__main__":
    # End-to-end demo when run as a script.
    matches_clean = run_clean_and_save()
    cfg = TrainConfig()
    model_bundle = train_baseline(matches_clean, cfg)

    print(f"Trained baseline model with holdout year {model_bundle['holdout_year']}.")

    groups_table, ko = simulate_world_cup_2022(
        model_bundle=model_bundle,
        matches_clean=matches_clean,
        random_state=cfg.random_state,
    )

    # Load team name mapping for better display
    team_names = load_team_name_mapping()
    groups_table["team_name"] = groups_table["team"].map(team_names)
    ko["home_team_name"] = ko["home_team"].map(team_names)
    ko["away_team_name"] = ko["away_team"].map(team_names)
    ko["winner_name"] = ko["winner"].map(team_names)

    print("\n=== Simulated 2022 Group Stage (top 2 advance) ===")
    for group_name in sorted(groups_table["group"].unique()):
        sub = groups_table[groups_table["group"] == group_name].copy()
        print(f"\n{group_name}")
        print(sub[["team_name", "played", "wins", "draws", "losses", "points"]])

    if not ko.empty:
        print("\n=== Simulated Knockout Bracket ===")
        print(ko[["round", "home_team_name", "away_team_name", "winner_name"]])

        final_match = ko[ko["round"] == "Final"].tail(1)
        if not final_match.empty:
            champion = final_match.iloc[0]["winner_name"]
            print(f"\nPredicted champion (single simulation): {champion}")