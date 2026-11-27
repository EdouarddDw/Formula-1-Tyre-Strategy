"""Project-wide constants. Every other module imports from here.

Sanity check: `uv run python -m src.config`
"""

from pathlib import Path

# --- Reproducibility --------------------------------------------------------------
SEED = 42

# --- Target -------------------------------------------------------------------------
# Fixed order = submission column order. Use it for model outputs and metrics too.
CLASSES = ["SOFT", "MEDIUM", "HARD", "WET_OR_INTER", "NONE"]
TARGET = "next_compound"  # train only; absent from test_laps

# --- Identifiers -------------------------------------------------------------------
RACE_KEYS = ["race_id"]  # CV groups, e.g. "2018_01"
DRIVER_KEYS = ["race_id", "driver_code"]  # one car in one race

# Submission id = race_id + "_" + driver_code, e.g. "2018_02_ALO".
# Checked: sample_submission ids are a subset of the test ids (1740 of 1814; the 74
# missing cars retired before the cut). sample_submission is sorted by id.
ID_COL = "id"
ID_FORMAT = "{race_id}_{driver_code}"


def make_id(df):
    """Submission id per row: race_id + "_" + driver_code."""
    return df["race_id"] + "_" + df["driver_code"]


# --- Scoring ------------------------------------------------------------------------
# Scored rows = target is not NA (official definition in src/target.py, which counts
# "Finished", "+n Lap(s)" and "Lapped" as finished). No separate status pattern here.

# --- Official test cut ---------------------------------------------------------------
# Each test race is cut when the leader finishes a lap between 30% and 50% of the race,
# fraction u ~ U(CUT_FRAC_MIN, CUT_FRAC_MAX) per race (see src/validation.py).
CUT_FRAC_MIN, CUT_FRAC_MAX = 0.3, 0.5

# Reference only (not used for the cut): 5th / 95th percentiles of the last observed
# lap_number of the 1740 sample_submission cars in test_laps (range 14-36, median 23).
CUTOFF_LAP_MIN = 15
CUTOFF_LAP_MAX = 32

# --- Validation / probabilities -----------------------------------------------------
N_FOLDS = 5
PROB_EPS = 1e-4  # clip probabilities to [PROB_EPS, 1] before renormalising

# --- Paths --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SUBMISSIONS_DIR = ROOT / "submissions"

for _d in (RESULTS_DIR, SUBMISSIONS_DIR):
    _d.mkdir(exist_ok=True)


if __name__ == "__main__":
    for _name, _value in list(globals().items()):
        if _name.isupper():
            print(f"{_name:24} = {_value!r}")
