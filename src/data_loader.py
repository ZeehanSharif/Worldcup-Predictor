import pandas as pd
from . import config

def load_matches() -> pd.DataFrame:
    """
    Load raw historical World Cup matches (Fjelstul csv).
    Expected columns include team codes, goals, stage, date, etc.
    """
    return pd.read_csv(config.MATCHES_CSV)
