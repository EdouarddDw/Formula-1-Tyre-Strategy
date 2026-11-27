# Formula-1-Tyre-Strategy
S403011 Machine Learning Project: Formula 1 Tyre Strategy
([Kaggle competition](https://www.kaggle.com/competitions/s-403011-ml-project-formula-1-tyre-strategy),
deadline **22 December 2026, 23:59**).

The data comes from the Kaggle competition.
It is **not** stored in this repo: `src/data.py` downloads it from Kaggle with your own API token
and caches it on your machine.

**Contents:** [Quick start](#quick-start) · [The task](#the-task) ·
[How the pipeline works](#how-the-pipeline-works) · [Repository layout](#repository-layout) ·
[Validation](#validation-srcvalidationpy) · [Metrics](#metrics-srcmetricspy) ·
[Models](#models-srcmodels) · [Evaluate and submit](#evaluate-and-submit) ·
[Current results](#current-results) · [Team rules](#team-rules) · [Next steps](#next-steps) ·
[Setup](#setup) · [Troubleshooting](#troubleshooting)

## Quick start

Once [Setup](#setup) is done:

```bash
uv run python -m src.data                    # download + print dataset shapes
uv run python -m src.validation              # cut-rule diagnostics + fold check (~70 s)
uv run python evaluate.py --model transition --save-plots   # CV log loss + plots
uv run python predict.py --model transition  # writes submissions/transition_<date>.csv
```

## The task

Lap-by-lap data of F1 races 2018–2026: **95 training races** (complete) and **92 test races**
(cut part-way through). The two sets contain different races.
For each car still running at the test cut, predict the probability of the compound it fits at
its **next** pit stop after its last visible lap:

| Class | Meaning |
|---|---|
| `SOFT`, `MEDIUM`, `HARD` | next dry compound, **relative to that race** (softest / middle / hardest of the three brought) |
| `WET_OR_INTER` | next tyre is intermediate or wet |
| `NONE` | no further pit stop: drives to the flag on the current tyres |

**Metric: multi-class log loss** (lower is better). We are scored on probabilities, not on the
predicted class, so calibration matters as much as ranking, and a single confident miss
(p ≈ 0 for the true class) dominates the score.

**Scored rows:** only cars that *finish* the race and have a known target. Cars that retire
after the cut still need a row in the submission but are ignored. There is a single public
leaderboard (no private split), so it is easy to overfit: **trust CV, use the leaderboard only
as a sanity check.**

Key facts we checked in the data:

- **Official test cut:** each test race is cut when the leader finishes a lap between 30% and
  50% of the race (fraction `u ~ U(0.3, 0.5)` per race). Every car shows only the laps it had
  finished by then, so lapped cars show fewer laps.
- `*_Lplus14` columns (weather / safety car 14 laps ahead) are look-ahead features that
  Kaggle provides on purpose, even past the cut. Using them is allowed.
- `result_status` / `classified` (final race outcome) are present in test. 157 of the 1,740
  submission cars did not finish, so their rows are not scored.
- Submission `id = race_id + "_" + driver_code` (e.g. `2018_02_ALO`), 1,740 rows sorted by
  id. The 74 test cars missing from `sample_submission` retired before the cut.

## How the pipeline works

```
train_laps.parquet ──► add_next_compound (src/target.py, official)   target on the FULL race
                         │
                         ▼
               get_folds (src/validation.py)        GroupKFold by race_id, 5 folds
               ├─ train_df: all laps of the training races with a known target
               └─ eval_df : held-out races, cut at random u ∈ [0.3, 0.5] like the test set,
                            one row per car at its last visible lap (× n_cuts)
                         │
                         ▼
               model = build(name, **params).fit(train_df, y)        (src/models/)
               p = model.predict_proba(eval_df)                      columns in CLASSES order
                         │
                         ▼
               log_loss(y, p)  (src/metrics.py)  ──►  results/results.csv (+ plots)

predict.py: fit on all train races ──► each test car's last visible lap ──► predict
            ──► check against sample_submission ──► submissions/<model>_<date>.csv
```

Three rules hold everywhere:

1. **Class order** is `CLASSES = ["SOFT", "MEDIUM", "HARD", "WET_OR_INTER", "NONE"]`
   (`src/config.py`): model outputs, metrics and submission all use it.
2. **The target is computed on the full race first, then the race is cut.** Never the other
   way around: a cut race does not show the future stop.
3. **Never split one race across train and evaluation.** Laps of the same car are nearly
   identical, so a random split leaks.

## Repository layout

```
├── src/
│   ├── config.py        constants: SEED, CLASSES, keys, cut fractions, PROB_EPS, paths, make_id()
│   ├── data.py          Kaggle download (kagglehub) + load_train / load_test / load_sample_submission
│   ├── target.py        official add_next_compound (copied from Kaggle, do not edit)
│   ├── validation.py    simulated official cut + grouped CV folds + diagnostics
│   ├── metrics.py       official log loss, clipping, accuracy, confusion matrix, plots
│   └── models/
│       ├── __init__.py  MODELS registry: name -> estimator class, build(name, **params)
│       └── baselines.py PriorBaseline, TransitionBaseline
├── evaluate.py          CV of one model → prints scores, appends to results/results.csv
├── predict.py           fit on all train → submissions/<model>_<date>.csv
├── results/             results.csv (run log, committed) + confusion / calibration plots
├── submissions/         generated CSVs (not committed: *.csv is ignored)
├── EDA.ipynb            exploration only
└── pyproject.toml, uv.lock
```

`src/config.py` constants you will use:

| Name | Value | Use |
|---|---|---|
| `SEED` | 42 | the only random seed (folds, cuts) |
| `CLASSES` | `["SOFT", "MEDIUM", "HARD", "WET_OR_INTER", "NONE"]` | fixed column order |
| `TARGET` | `"next_compound"` | target column name |
| `RACE_KEYS`, `DRIVER_KEYS` | `["race_id"]`, `["race_id", "driver_code"]` | grouping keys |
| `CUT_FRAC_MIN`, `CUT_FRAC_MAX` | 0.3, 0.5 | official cut range |
| `CUTOFF_LAP_MIN`, `CUTOFF_LAP_MAX` | 15, 32 | reference only: 5–95% of last visible lap in test |
| `N_FOLDS` | 5 | CV folds |
| `PROB_EPS` | 1e-4 | probabilities are clipped to `[PROB_EPS, 1]` and renormalised |
| `make_id(df)` | | `race_id + "_" + driver_code` |

## Validation (`src/validation.py`)

The CV setup copies how Kaggle builds the test set, so that the CV score predicts the
leaderboard.

**Simulated cut** for a race and a fraction `u`:

1. Leader lap `k` = first lap with `k ≥ u · scheduled_laps` (`mode="laps"`).
2. `T_cut` = race time (`race_time_s`, cumulative at the end of a lap) of the car in
   position 1 at the end of lap `k`.
3. Each car keeps its laps with `race_time_s ≤ T_cut`. Its last kept lap is the row we predict.
4. Cars with no finished lap after `T_cut` retired before the cut and are dropped (like test).
   Cars whose target is NA (non-finishers without another stop) are dropped too (not scored).

`mode="laps"` (fraction of `scheduled_laps`) is the default because it matches the test set.
In test, the leader's last visible lap is always between 30.0% and 50.0% of `scheduled_laps`.
The alternative `mode="time"` (fraction of the winner's race time) would put ≥ 5% of races
outside that range (red flags, safety cars stretch the time), which never happens in test.

| Function | Returns |
|---|---|
| `cut_time(race_df, u, mode)` | `T_cut` in seconds |
| `cut_race(race_df, u, mode)` | the visible laps after the cut |
| `last_visible_laps(race_df, u, mode)` | one row per car still running, at its last visible lap |
| `make_eval_rows(race_df, rng, mode, n_cuts)` | target on the full race → cut → known-target rows, with `cut_id` |
| `cut_rows(df, rng, mode, n_cuts)` | same for every race of a DataFrame that already has the target (used to fit on test-like rows) |
| `get_folds(train, n_folds, seed, mode, n_cuts)` | yields `(train_df, eval_df)` per fold |

`get_folds` details:

- `GroupKFold(shuffle=True, random_state=SEED)` on `race_id`: 76 training / 19 evaluation
  races per fold, each race held out exactly once.
- `train_df` = **all laps** of the training races with a known target (≈ 80k rows). Models
  decide themselves whether to use all laps or only test-like rows (see `fit_on` below).
- `eval_df` = `n_cuts` (default 5) random cuts per held-out race, ≈ 1,700 rows per fold.
  The cuts are drawn once per race with `SEED`, so they do not depend on the fold split.
  The 5 cuts of one car are correlated: the per-fold std understates the real noise.

**Class priors are very different at the cut than over all laps** (simulated, 50 cuts per race):

| | SOFT | MEDIUM | HARD | WET_OR_INTER | NONE |
|---|---|---|---|---|---|
| eval rows at the simulated cut | 16.4% | 23.8% | 34.6% | 2.9% | 22.3% |
| `sample_submission` default values | 16.2% | 23.7% | 33.3% | 2.7% | 24.1% |
| all known-target train laps | 12.1% | 16.7% | 25.5% | 2.0% | 43.7% |

The simulated priors almost reproduce the `sample_submission` values, which independently
confirms the cut and the scored-row filter.

`uv run python -m src.validation` prints these diagnostics: test vs simulated last-lap
distributions for both modes (quantiles + KS), the class distribution, and the fold check.

## Metrics (`src/metrics.py`)

| Function | Does |
|---|---|
| `clip_and_normalise(p, eps=PROB_EPS)` | clip to `[eps, 1]`, rows sum to 1 (applied before scoring and in the submission) |
| `log_loss(y, p)` | **official metric**, `p` columns in `CLASSES` order |
| `accuracy(y, p)` | argmax accuracy (secondary) |
| `confusion_matrix(y, p)` | row-normalised `DataFrame` |
| `plot_reliability(y, p)`, `plot_confusion(cm)` | calibration curves (one-vs-rest) and heatmap |

⚠️ **Do not use `sklearn.metrics.log_loss` directly.** It assumes the probability columns are
in *alphabetical* label order (`HARD, MEDIUM, NONE, SOFT, WET_OR_INTER`), even with
`labels=CLASSES`, and silently gives wrong scores for our column order. Use
`src.metrics.log_loss`.

## Models (`src/models/`)

Every model is an sklearn-style estimator:

- `fit(X, y)`: `X` = laps `DataFrame` with **all columns** (the model picks its features),
  `y` = `next_compound` (no NA).
- `predict_proba(X)` → array `(n_rows, 5)` in `CLASSES` order; `classes_ = CLASSES`.
- Registered by name in `MODELS` in `src/models/__init__.py`.

Current models:

| Name | Class | Idea | Parameters |
|---|---|---|---|
| `prior` | `PriorBaseline` | constant class frequencies | `fit_on="cut"`, `n_cuts=20` |
| `transition` | `TransitionBaseline` | P(next \| current `compound`, `stint` capped at 3), Laplace-smoothed; unseen cells / missing values → prior | `fit_on="cut"`, `n_cuts=20`, `alpha=1.0`, `max_stint=3` |

`fit_on="cut"` (default) fits on simulated cut rows of the training races (`cut_rows`, 20 cuts
per race) instead of all laps, so the model sees the same lap window and class priors as the
test set. `fit_on="all"` uses every lap. The cut version is ≈ 0.1 better in log loss.

**Adding a model** (no change to `evaluate.py` / `predict.py` needed):

1. Create `src/models/my_model.py` with a class implementing `fit` / `predict_proba` as above
   (inherit `ClassifierMixin, BaseEstimator`; all constructor arguments as attributes).
2. Register it: `MODELS["my_model"] = MyModel` in `src/models/__init__.py`.
3. `uv run python evaluate.py --model my_model --params '{"some_param": 3}'`.

Guidelines for probability models:

- No class weighting / resampling: it distorts probabilities and hurts log loss.
- Never output exact 0: probabilities are clipped, but a clipped 1e-4 on the true class still
  costs −ln(1e-4) ≈ 9.2 for that row.
- Calibrate tree models (Platt / isotonic / temperature) inside the training folds only.
- Only use information visible up to a car's last lap: forward-fill within a car, never
  backward-fill. `*_Lplus14` columns are the allowed exception.

## Evaluate and submit

```bash
uv run python evaluate.py --model NAME [--params JSON] [--folds 5] [--n-cuts 5] [--mode laps] [--save-plots]
```

- Prints per-fold log loss, mean ± std, accuracy, the uniform (ln 5 = 1.609) and all-laps
  prior reference values, and the pooled confusion matrix.
- Appends one row to `results/results.csv`: date, model, params, mode, folds, n_cuts,
  mean/std log loss, accuracy, git commit (`+dirty` = uncommitted changes, so commit before
  runs you want to cite).
- `--save-plots` writes `results/<model>_confusion.png` and `results/<model>_calibration.png`.
- Takes ≈ 30–40 s for the baselines (building the folds is ≈ 15 s).

```bash
uv run python predict.py --model NAME [--params JSON]
```

- Fits on all 95 training races (known-target laps), predicts each test car's last visible lap.
- Checks before writing: same ids and order as `sample_submission`, columns
  `["id"] + CLASSES`, finite, no zeros, rows sum to 1.
- Writes `submissions/<model>_<YYYY-MM-DD>.csv`. Upload it on Kaggle (web, or
  `kaggle competitions submit -c s-403011-ml-project-formula-1-tyre-strategy -f <file> -m "<msg>"`).

## Current results

CV: 5 folds grouped by race, 5 simulated cuts per held-out race, `mode="laps"`.

| Model | Params | CV log loss | Accuracy |
|---|---|---|---|
| uniform | | 1.6094 | – |
| prior | `fit_on="all"` | 1.5630 ± 0.137 | 0.224 |
| prior | `fit_on="cut"` | 1.4650 ± 0.129 | 0.347 |
| transition | `fit_on="all"` | 1.4014 ± 0.072 | 0.444 |
| transition | `fit_on="cut"` | 1.2888 ± 0.156 | 0.497 |
| **transition** | `alpha=5` | **1.2661 ± 0.123** | 0.497 |

First submission: `transition` with `alpha=5`. The full run log is `results/results.csv`.

## Team rules

- Everything runs with `uv run`; add packages with `uv add <pkg>` and commit
  `pyproject.toml` + `uv.lock`.
- Work on your own branch; merge into `main` through small pull requests;
  `git pull origin main` regularly.
- `src/validation.py`, `src/config.py` and (future) `src/features.py` are shared:
  change them only in focused PRs, since every score depends on them.
- All models use the same folds (`get_folds`) and are compared only by CV log loss.
- Reusable code goes in `src/`; notebooks are for exploration only (clear outputs before
  committing).
- Never commit data files or `.env`.

## Next steps

- **WET_OR_INTER overconfidence:** the transition table gives dry cells ≈ 0.0004 on
  `WET_OR_INTER`, so an unexpected switch to inters costs ≈ 7.8 on that row. Smooth towards
  the prior instead of a flat count, and use `rainfall_Lplus14`.
- **NONE depends on race progress:** add `race_progress` / `laps_remaining` (NONE is poorly
  calibrated around p ≈ 0.45).
- `src/features.py`: shared `build_features(df)` (compounds used so far, two-compound rule,
  stint length vs typical max stint, field context, oracle `*_Lplus14` features).
- Then: multinomial logistic regression → gradient boosting (+ calibration) → blend.
- Scored rows are finishers only, and test shows the final `result_status`: training or
  validating on finishers only may match the scored population better.

## Setup

### 1. Install uv

We use [uv](https://docs.astral.sh/uv/) to manage Python and the packages, so everyone
gets exactly the same versions (they are pinned in `uv.lock`).

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

(Or `brew install uv` / `pip install uv`.)

### 2. Install the dependencies

From the repo folder:

```bash
uv sync
```

This creates a `.venv/` folder with the right Python packages. You don't need to activate it:
prefix commands with `uv run` (e.g. `uv run python -m src.data`). If you prefer, activate it
with `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows), and
select `.venv` as the interpreter in VS Code / Jupyter.

To add a new package, use `uv add <package>` (not `pip install`) so it ends up in
`pyproject.toml` and `uv.lock` for everyone. Commit both files.

### 3. Get access to the Kaggle data

1. Create a [Kaggle](https://www.kaggle.com) account (or log in).
2. Open the [competition page](https://www.kaggle.com/competitions/s-403011-ml-project-formula-1-tyre-strategy)
   and **join the competition / accept the rules**. Without this the download fails with a
   403 / permission error.
3. Go to **Settings → API** ([kaggle.com/settings](https://www.kaggle.com/settings)) and click
   **Generate New Token**. Copy the token.

### 4. Add your API token

Copy the template and paste your token in it:

```bash
cp .env.example .env
```

`.env` should look like:

```
KAGGLE_API_TOKEN=your_kaggle_api_token_here
```

`.env` is in `.gitignore`. **Never commit it** — the token is personal.

### 5. Check that it works

```bash
uv run python -m src.data
```

The first run downloads the files (cached afterwards, under `~/.cache/kagglehub/`) and prints
the shape of each dataset.

## Loading the data in your code

```python
from src.data import load_data

train, test, sample_submission = load_data()
```

Or load a single dataset:

```python
from src.data import load_train, load_test, load_sample_submission

train = load_train()
```

| Function | File | Returns |
| --- | --- | --- |
| `load_train()` | `train_laps.parquet` | training laps `DataFrame` |
| `load_test()` | `test_laps.parquet` | test laps `DataFrame` |
| `load_sample_submission()` | `sample_submission.csv` | submission format `DataFrame` |
| `get_data_dir()` | — | `Path` of the folder with the raw files |

## Troubleshooting

- **401 / unauthenticated**: `.env` is missing, is not in the repo root, or the token is wrong.
  Generate a new token and try again.
- **403 / forbidden**: you haven't joined the competition / accepted the rules on Kaggle.
- **`ModuleNotFoundError`**: run `uv sync`, and make sure you run with `uv run` or with
  `.venv` activated / selected as the interpreter.
- **Not using uv?** In a virtual environment, run
  `pip install kagglehub pandas pyarrow python-dotenv matplotlib scikit-learn scipy` (the list in `pyproject.toml`).
  The versions won't be pinned like they are with `uv.lock`.
