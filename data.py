"""Load the Formula 1 Tyre Strategy competition data as pandas DataFrames.

Usage from any other .py file in the project:

    from data import load_data
    train, test, sample_submission = load_data()

Or load a single dataset:

    from data import load_train, load_test, load_sample_submission
    train = load_train()
"""

from pathlib import Path

import kagglehub
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

COMPETITION = "s-403011-ml-project-formula-1-tyre-strategy"


def get_data_dir() -> Path:
    """Download the competition files (cached after the first call) and return their folder."""
    return Path(kagglehub.competition_download(COMPETITION))


def load_train() -> pd.DataFrame:
    return pd.read_parquet(get_data_dir() / "train_laps.parquet")


def load_test() -> pd.DataFrame:
    return pd.read_parquet(get_data_dir() / "test_laps.parquet")


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(get_data_dir() / "sample_submission.csv")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, sample_submission) DataFrames."""
    data_dir = get_data_dir()
    train = pd.read_parquet(data_dir / "train_laps.parquet")
    test = pd.read_parquet(data_dir / "test_laps.parquet")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sample_submission


if __name__ == "__main__":
    train, test, sample_submission = load_data()
    print("Data folder:", get_data_dir())
    for name, df in [("train", train), ("test", test), ("sample_submission", sample_submission)]:
        print(f"{name}: {df.shape}")
