"""Cross-validate a registered model on simulated official cuts.

    uv run python evaluate.py --model transition [--folds 5] [--n-cuts 5] [--save-plots]
    uv run python evaluate.py --model prior --params '{"fit_on": "all"}'
"""

import argparse
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import CLASSES, N_FOLDS, RESULTS_DIR, SEED, TARGET
from src.data import load_train
from src.metrics import (
    accuracy,
    clip_and_normalise,
    confusion_matrix,
    log_loss,
    plot_confusion,
    plot_reliability,
)
from src.models import build
from src.validation import DEFAULT_MODE, DEFAULT_N_CUTS, get_folds

RESULTS_CSV = RESULTS_DIR / "results.csv"


def git_commit() -> str:
    """Short hash of HEAD, with '+dirty' if tracked files have uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0
        return sha + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", default="{}", help="JSON dict of model parameters")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--n-cuts", type=int, default=DEFAULT_N_CUTS, help="simulated cuts per eval race")
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=["laps", "time"])
    ap.add_argument("--save-plots", action="store_true")
    args = ap.parse_args()
    params = json.loads(args.params)

    fold_ll, fold_ref, oof_y, oof_p = [], [], [], []
    for k, (tr_df, ev_df) in enumerate(get_folds(load_train(), args.folds, SEED, args.mode, args.n_cuts)):
        model = build(args.model, **params).fit(tr_df, tr_df[TARGET])
        p = clip_and_normalise(model.predict_proba(ev_df))
        y = ev_df[TARGET].to_numpy(dtype=object)
        prior_all = tr_df[TARGET].value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_numpy()
        fold_ll.append(log_loss(y, p))
        fold_ref.append(log_loss(y, np.tile(prior_all, (len(y), 1))))
        oof_y.append(y)
        oof_p.append(p)
        print(f"fold {k}: log loss {fold_ll[-1]:.4f}  (n = {len(y)}, {ev_df['race_id'].nunique()} races)")

    y, p = np.concatenate(oof_y), np.vstack(oof_p)
    mean, std = float(np.mean(fold_ll)), float(np.std(fold_ll, ddof=1))
    acc = accuracy(y, p)
    print(f"\n{args.model} {params}: CV log loss {mean:.4f} ± {std:.4f}  |  accuracy {acc:.3f}")
    print(f"reference: uniform ln 5 = {np.log(len(CLASSES)):.4f}  |  "
          f"all-laps prior = {np.mean(fold_ref):.4f}")
    print("\nconfusion matrix (row-normalised, pooled over folds)")
    print(confusion_matrix(y, p).round(2).to_string())

    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "params": json.dumps(params, sort_keys=True),
        "mode": args.mode,
        "folds": args.folds,
        "n_cuts": args.n_cuts,
        "cv_logloss_mean": round(mean, 5),
        "cv_logloss_std": round(std, 5),
        "accuracy": round(acc, 4),
        "git_commit": git_commit(),
    }
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)
    print(f"\nappended to {RESULTS_CSV.relative_to(RESULTS_DIR.parent)}")

    if args.save_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for name, draw in [
            ("confusion", lambda ax: plot_confusion(confusion_matrix(y, p), ax=ax)),
            ("calibration", lambda ax: plot_reliability(y, p, ax=ax)),
        ]:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            draw(ax)
            ax.set_title(f"{args.model}: {name} (CV log loss {mean:.3f})", fontsize=9)
            path = RESULTS_DIR / f"{args.model}_{name}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=130)
            plt.close(fig)
            print(f"saved {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
