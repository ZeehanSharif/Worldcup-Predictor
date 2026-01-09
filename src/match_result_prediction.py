from cleaning import run_clean_and_save
from src.config import TrainConfig
from modeling import train_baseline
import pandas as pd

def predict_match(pipe, home_elo, away_elo, is_knockout):
    df = pd.DataFrame([{
        "elo_home_pre": home_elo,
        "elo_away_pre": away_elo,
        "elo_delta_pre": home_elo - away_elo,
        "is_knockout": int(is_knockout),
    }])
    result = pipe.predict(df)[0]
    proba = pd.DataFrame(pipe.predict_proba(df), columns=pipe.classes_)
    return result, proba


if __name__ == "__main__":
    # Load and clean data
    matches_clean = run_clean_and_save()

    # Train model
    cfg = TrainConfig(test_year=2022, random_state=42)
    model_bundle = train_baseline(matches_clean, cfg)
    pipe = model_bundle["model"]

    # Predict a match
    result, proba = predict_match(pipe, 1800, 1750, True)
    print("Predicted result:", result)
    print("Prediction probabilities:\n", proba)

