"""Fit a registered model on all train races and write a Kaggle submission.

    uv run python predict.py --model transition [--params '{"alpha": 1}']
"""

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from src.config import CLASSES, DRIVER_KEYS, ID_COL, SUBMISSIONS_DIR, TARGET, make_id
from src.data import load_data
from src.metrics import clip_and_normalise
from src.models import build
from src.target import add_next_compound


def last_test_laps(test: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """Each test car's last visible lap, in the order of `ids`."""
    last = test.sort_values("lap_number").groupby(DRIVER_KEYS).tail(1)
    last = last.set_index(make_id(last).rename(ID_COL))
    missing = set(ids) - set(last.index)
    assert not missing, f"{len(missing)} submission ids not found in test_laps"
    return last.loc[ids]


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Assert the submission matches sample_submission: ids, order, columns, valid rows."""
    assert list(sub.columns) == [ID_COL] + CLASSES, f"columns {list(sub.columns)}"
    assert list(sample.columns) == list(sub.columns), "column order differs from sample_submission"
    assert sub[ID_COL].tolist() == sample[ID_COL].tolist(), "ids / order differ from sample_submission"
    p = sub[CLASSES].to_numpy()
    assert np.isfinite(p).all(), "non-finite probabilities"
    assert (p > 0).all(), "zero probabilities"
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), "rows do not sum to 1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", default="{}", help="JSON dict of model parameters")
    args = ap.parse_args()
    params = json.loads(args.params)

    train, test, sample = load_data()
    train = add_next_compound(train, name=TARGET)
    train = train[train[TARGET].notna()].reset_index(drop=True)
    model = build(args.model, **params).fit(train, train[TARGET])

    X = last_test_laps(test, sample[ID_COL])
    p = clip_and_normalise(model.predict_proba(X))
    sub = pd.DataFrame(p, columns=CLASSES)
    sub.insert(0, ID_COL, sample[ID_COL].to_numpy())
    check_submission(sub, sample)

    path = SUBMISSIONS_DIR / f"{args.model}_{date.today().isoformat()}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path.relative_to(SUBMISSIONS_DIR.parent)}  ({len(sub)} rows)")
    print("mean predicted probabilities:", sub[CLASSES].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
