from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MATCHES_CSV = DATA_RAW_DIR / "matches.csv"
TEAMS_CSV = DATA_RAW_DIR / "teams.csv"

MATCHES_CLEAN_PARQUET = DATA_PROCESSED_DIR / "matches_clean.parquet"

@dataclass
class TrainConfig:
    label_col: str = "match_result_1x2"
    test_year: int = 2022 # hold out the most recent WC we have
    random_state: int = 42
