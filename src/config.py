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
RACE_KEYS = ["year", "round"]  # CV groups (equivalent to the `race_id` column, "2018_01")
DRIVER_KEYS = RACE_KEYS + ["driver_code"]  # one car in one race

# Submission id: f"{year}_{round:02d}_{driver_code}", e.g. "2018_02_ALO".
# Checked: built this way, sample_submission ids are a subset of the test ids
# (1740 of 1814; the 74 missing cars all retired before the cutoff).
# sample_submission is sorted by id.
ID_COL = "id"
ID_FORMAT = "{year}_{round:02d}_{driver_code}"

# --- Scoring ------------------------------------------------------------------------
# Scored cars: result_status "Finished" or "+n Lap"/"+n Laps" (n up to 15 seen in the data).
# Use with `df["result_status"].str.match(FINISHED_STATUS_PATTERN)`.
# Note: "Lapped" (1 car, 2026_09 SAI) is not matched, following the official rule.
FINISHED_STATUS_PATTERN = r"^(Finished|\+\d+ Laps?)$"

# --- Test-like lap window ----------------------------------------------------------
# 5th and 95th percentiles of each scored driver's last observed lap_number in
# test_laps (1740 ids from sample_submission). Full range 14-36, median 23.
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
