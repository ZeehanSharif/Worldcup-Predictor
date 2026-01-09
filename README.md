# World Cup Outcome Forecasting  

## Overview

This project builds a reproducible data -> model pipeline to forecast the FIFA Men's World Cup.

## How to Run and Test

- Clone this repo locally.
- Create a new branch (not master) and switch over to it.
- Create a virtual environment in the project root and install requirements:
    ```bash
    python -m venv .venv
    .venv\Scripts\Activate.ps1    # Windows
    # source .venv/bin/activate   # macOS/Linux
    pip install -r requirements.txt
    ```
- Open python interpereter and run:
    ```python
    import sys
    from pathlib import Path
    sys.path.append(str(Path(".").resolve()))

    from src.cleaning import run_clean_and_save
    from src.config import TrainConfig
    from src.modeling import train_baseline

    # clean + label historical data
    matches_clean = run_clean_and_save()
    print(matches_clean.shape)
    print(matches_clean.head())

    # train + evaluate baseline model
    cfg = TrainConfig(test_year=2022)
    model_bundle = train_baseline(matches_clean, cfg)
    print(model_bundle["holdout_report"])
    print(model_bundle["holdout_confusion"])
    ```
- If you need to make a change, open a PR from your new branch with the changes.
