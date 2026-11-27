"""Baselines: constant class prior and the compound/stint transition table.

Both take X = a laps DataFrame (all columns) and y = next_compound.
fit_on="cut": fit on simulated official-cut rows of the training races (test-like
window, one row per car per cut) instead of all laps; the priors differ a lot.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import CLASSES, SEED, TARGET


class _LapsWindowMixin:
    """Selects the fitting rows: all laps, or simulated cut rows."""

    def _fit_rows(self, X: pd.DataFrame, y) -> pd.DataFrame:
        df = X.assign(**{TARGET: np.asarray(y, dtype=object)})
        if self.fit_on == "all":
            return df
        if self.fit_on == "cut":
            from src.validation import cut_rows

            return cut_rows(df, np.random.default_rng(SEED), n_cuts=self.n_cuts)
        raise ValueError(f"fit_on must be 'all' or 'cut', got {self.fit_on!r}")


def _class_freq(y) -> np.ndarray:
    freq = pd.Series(np.asarray(y, dtype=object)).value_counts(normalize=True)
    return freq.reindex(CLASSES, fill_value=0.0).to_numpy()


class PriorBaseline(_LapsWindowMixin, ClassifierMixin, BaseEstimator):
    """Constant prediction = class frequencies of the training targets."""

    def __init__(self, fit_on: str = "cut", n_cuts: int = 20):
        self.fit_on = fit_on
        self.n_cuts = n_cuts

    def fit(self, X, y):
        self.classes_ = np.array(CLASSES)
        self.prior_ = _class_freq(self._fit_rows(X, y)[TARGET])
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))


class TransitionBaseline(_LapsWindowMixin, ClassifierMixin, BaseEstimator):
    """P(next | current compound, stint capped at max_stint), Laplace-smoothed (alpha).

    Unseen cells (incl. missing compound / stint) fall back to the prior.
    """

    def __init__(self, fit_on: str = "cut", n_cuts: int = 20, alpha: float = 1.0, max_stint: int = 3):
        self.fit_on = fit_on
        self.n_cuts = n_cuts
        self.alpha = alpha
        self.max_stint = max_stint

    def _cell(self, X: pd.DataFrame) -> pd.Series:
        stint = X["stint"].clip(upper=self.max_stint).astype("string").fillna("NA")
        return X["compound"].astype("string").fillna("UNKNOWN") + "|" + stint

    def fit(self, X, y):
        self.classes_ = np.array(CLASSES)
        df = self._fit_rows(X, y)
        self.prior_ = _class_freq(df[TARGET])
        counts = pd.crosstab(self._cell(df), df[TARGET]).reindex(columns=CLASSES, fill_value=0)
        smoothed = counts + self.alpha
        self.table_ = smoothed.div(smoothed.sum(axis=1), axis=0)
        self.cell_counts_ = counts.sum(axis=1)
        return self

    def predict_proba(self, X):
        p = self.table_.reindex(self._cell(X).to_numpy())
        p[p.isna().any(axis=1)] = self.prior_
        return p.to_numpy(dtype=float)
