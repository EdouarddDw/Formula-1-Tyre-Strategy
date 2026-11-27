"""Official metric (multi-class log loss) and diagnostics. Class order = config.CLASSES."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix

from src.config import CLASSES, PROB_EPS


def clip_and_normalise(p, eps: float = PROB_EPS) -> np.ndarray:
    """Clip probabilities to [eps, 1] and renormalise each row to sum to 1."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def log_loss(y_true, p) -> float:
    """Official metric: multi-class log loss, columns of p in CLASSES order, rows renormalised.

    Computed directly: sklearn's log_loss assumes alphabetical column order, not CLASSES.
    """
    idx = pd.Index(CLASSES).get_indexer(np.asarray(y_true, dtype=object))
    if (idx < 0).any():
        raise ValueError("y_true contains labels outside CLASSES (NA target rows must be dropped)")
    p = clip_and_normalise(p)
    return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))


def accuracy(y_true, p) -> float:
    """Share of rows whose argmax class equals the true class."""
    pred = np.asarray(CLASSES)[np.asarray(p).argmax(axis=1)]
    return float(np.mean(pred == np.asarray(y_true, dtype=object)))


def confusion_matrix(y_true, p, normalize: str | None = "true") -> pd.DataFrame:
    """Confusion matrix of argmax predictions (rows = true, columns = predicted)."""
    pred = np.asarray(CLASSES)[np.asarray(p).argmax(axis=1)]
    cm = _sk_confusion_matrix(np.asarray(y_true, dtype=object), pred, labels=CLASSES, normalize=normalize)
    return pd.DataFrame(cm, index=pd.Index(CLASSES, name="true"), columns=pd.Index(CLASSES, name="pred"))


def reference_log_losses(y_true, y_prior) -> dict[str, float]:
    """Uniform (ln 5) and constant-prior log loss; prior = class frequencies of y_prior."""
    prior = pd.Series(np.asarray(y_prior, dtype=object)).value_counts(normalize=True)
    prior = prior.reindex(CLASSES, fill_value=0).to_numpy()
    n = len(y_true)
    return {
        "uniform": float(np.log(len(CLASSES))),
        "prior": log_loss(y_true, np.tile(prior, (n, 1))),
    }


def plot_reliability(y_true, p, n_bins: int = 10, ax=None):
    """One-vs-rest reliability curve per class (quantile bins). Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    y = np.asarray(y_true, dtype=object)
    p = clip_and_normalise(p)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect")
    for k, cls in enumerate(CLASSES):
        df = pd.DataFrame({"p": p[:, k], "hit": y == cls})
        bins = pd.qcut(df["p"], q=n_bins, duplicates="drop")
        cal = df.groupby(bins, observed=True).agg(p=("p", "mean"), hit=("hit", "mean"))
        ax.plot(cal["p"], cal["hit"], marker="o", ms=3, lw=1.2, label=cls)
    ax.set(xlabel="predicted probability", ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    return ax


def plot_confusion(cm: pd.DataFrame, ax=None):
    """Heatmap of a confusion matrix DataFrame. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="true")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax
