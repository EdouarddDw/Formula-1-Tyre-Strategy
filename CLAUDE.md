# CLAUDE.md — Formula 1 Tyre Strategy (S403011)

## Project

Course project for **S403011 (Machine Learning, UNIGE, Prof. Sebastian Engelke)**:
a private Kaggle class competition.
Team of 4–5, working via Git branches and pull requests.

- **Runs:** 29 October → **22 December 2026, 23:59:59** (Kaggle close = Moodle deadline)
- **Deliverable:** one zip with report + code on Moodle, named
  `KaggleTeamName-LastName1_LastName2_LastName3.zip` (one member submits for the team).
  Kaggle team name must start with the Moodle team number, e.g. `T14_PitWizards`.
- Competition page: https://www.kaggle.com/competitions/s-403011-ml-project-formula-1-tyre-strategy

**Task:** multiclass probabilistic classification. Lap-by-lap data of F1 races 2018–2026:
**95 training races, 92 test races** (disjoint). For each car still running at the test
cut, predict the compound it fits at its **next** pit stop after its last visible lap:

| Class | Meaning |
|---|---|
| `SOFT`, `MEDIUM`, `HARD` | next dry compound |
| `WET_OR_INTER` | next tyre is wet or intermediate |
| `NONE` | no further pit stop this race |

## How the test races are cut (official)

- The cut is placed when the **leader finishes a lap between 30% and 50% of the race**,
  the fraction drawn **uniformly at random** per race, using `race_time_s`.
- Every car shows only the laps **it had finished by then** (lapped cars show fewer laps).
- Oracle / look-ahead columns (`*_Lplus14`) are given **even past the cut** in test.
- → `validation.py` reproduces this rule on held-out train races: draw `u ~ U(0.3, 0.5)`
  per race, pick the leader's lap at fraction `u` of the race, take the leader's
  `race_time_s` at the end of that lap as `T_cut`, keep each car's laps with
  `race_time_s ≤ T_cut`.
  **Open question:** "% of the race" = of laps or of the leader's total race time?
  Check on Kaggle / the Q&A forum, or test which version matches the test set's
  distribution of last visible laps.

## Data files

| File | Content |
|---|---|
| `train_laps.parquet` | 95 full races, one row per driver-lap, with `next_compound` |
| `test_laps.parquet` | 92 races up to the cut, no target |
| `sample_submission.csv` | the exact 1,740 rows (cars) to predict |
| `target.py` | official `add_next_compound` → copied to `src/target.py` |
| `*_telemetry.parquet` | large, read per race with `filters=[("race_id", "==", "2023_01")]` (not downloaded yet — check Kaggle Data tab) |

Key columns: `lap_id` (`2023_01_VER_007`), `race_id` (`2023_01`), `driver_code`, `circuit_id`,
`result_status`, `classified`, `official_stop`, `stint`, `compound`, `compound_hardness`,
`hardness_scale`, `pit_lane_start`, `track_status_code`.

## Target definition (`src/target.py`, official)

For a row with last visible lap L:
- next **official** stop on lap S > L → tyre on lap S + 1 (a stop on lap L itself does not count;
  non-official tyre changes, e.g. under red flag, don't count)
- no further stop **and finished** → `NONE`
- no further stop and **did not finish**, or tyre after the stop unknown → **NA**
- Finished regex: `^(?:Finished|\+\d+ Laps?|Lapped)$` — lapped cars count as finishers.
- SOFT/MEDIUM/HARD are **relative to each race**, never `compound_hardness`.
- Train counts: NONE 44,251 · HARD 25,846 · MEDIUM 16,888 · SOFT 12,197 · WET_OR_INTER 2,009 · NA 3,566.
- `add_next_compound(test, complete_races=False)` labels early test laps whose next stop is
  already visible → possible extra training data, **biased towards early stops**, never for CV.

## Evaluation (official)

**Multi-class log loss**, lower is better:

$$\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik}\ln \hat p_{ik},
\qquad \hat p_{ik} = \frac{p_{ik}}{\sum_{j} p_{ij}}$$

- Rows are renormalised to sum to 1 by Kaggle.
- **Only finishers with a known target are scored** (target not NA).
  Cars that retire *after* the cut still need a row but are ignored.
  Cars that retired *before* the cut must **not** have a row (it errors) —
  predict exactly the ids in `sample_submission.csv`.
- **Only one public leaderboard**, no private split → it is easy to overfit the LB.
  Trust CV; use the LB as a sanity check, not as the tuning target.

### Consequences for modelling

- We optimise **probabilities, not argmax**: calibration matters as much as ranking.
- **No class weighting / resampling** — it distorts probabilities and hurts log loss.
- Match the **test-window class priors** (HARD ≈ 47%, NONE ≈ 12%), not the all-laps priors.
- **Never output exact 0 (or 1):** clip, e.g. `p = clip(p, 1e-4, 1)` then renormalise.
  One confident miss dominates the score.
- Calibrate tree models (Platt / isotonic / temperature scaling on CV folds) and check
  reliability curves. Random forest and naive Bayes are typically poorly calibrated.
- Averaging probabilities of several models (ensembling / blending) usually lowers log loss.
- Report the **uniform-prior and majority-prior log loss** as reference points.

## Submission format

`submissions/*.csv` with header, one row per `id` in the test set
(check against `sample_submission.csv` — same ids, same order, same columns):

```
id,SOFT,MEDIUM,HARD,WET_OR_INTER,NONE
2018_02_ALO,0.1618,0.2374,0.3331,0.0266,0.2411
```

`id` = `race_id + "_" + driver_code` (e.g. `2018_02_ALO`). Each row = prediction at that
car's last observed test lap. 1,740 rows.

## Key findings from EDA (drive every modelling decision)

1. **Train/test shift.** Train contains full races (laps 1 → ~80). Test races are
   cut at 30–50% of race distance; each driver's last observed lap is ≈ 15–35.
2. **Class priors differ massively.**
   - All train laps: NONE 44%, HARD 26%, MEDIUM 17%, SOFT 12%, WET/INTER 2%
   - Test-like window (rough EDA approximation, laps 15–32): HARD 47%, MEDIUM 26%,
     NONE 12%, SOFT 11%, WET/INTER 3% → **recompute with the exact cut rule**.
3. **Effective sample size ≈ 95 races × ~20 cars ≈ 1,900 cars**, not ~100k laps. Laps within
   a driver-race are near-duplicates → regularise, and never split randomly.
4. **Signal lives in interactions**, not single features:
   - current compound → next: MEDIUM→HARD 0.48, SOFT→MEDIUM 0.32, WET→INTER 0.87
   - stint 1 → mostly HARD/MEDIUM; stint ≥ 2 → mostly NONE
   - `race_progress` is the main driver of NONE (corr 0.72)
   - `rainfall_Lplus14`, humidity, low track/air temp → WET_OR_INTER
   - strong circuit effects (e.g. istanbul/hockenheimring wet, jeddah high NONE), some season effects
5. **Collinear block:** `lap_number`, `race_time_s`, `race_progress`, `laps_remaining`
   (|r| ≈ 0.9). Keep only `race_progress` + `laps_remaining` for linear models / NB.
6. **Missingness** (verify each cause before coding the fix):
   - `*_Lplus14` ≈ 21% missing in train only: empty when the race is over by lap L + 14
     (official). They are **legitimate oracle features**, given past the cut in test — use them.
   - `pit_lane_time_s` ≈ 97% missing in both → binary `pitted_this_lap`, don't impute.
   - `gap_to_car_ahead_s` / `_behind_s` ≈ 5% → probably leader / last car (structural).
   - `compound_hardness`, `hardness_scale` ≈ 5% train, 7% test → probably INTER/WET laps
     (no C-number) → encode as a category, not imputed. Note `compound_hardness` is C1
     (hardest) to C6 (softest), 2018 uses old names (SUPERSOFT…); `hardness_scale` only
     says which naming is used → map 2018 names onto the C-scale.
   - sector/speed columns ≈ 2–5% → lap 1, in/out laps, timing gaps.
   - `*_so_far`, `*_pending` ≈ 1% → probably lap 1 / before any event → fill 0.

## Validation rules (non-negotiable)

- Test races are **disjoint** from train races → **GroupKFold grouped by `race_id`**.
  Never random K-fold, never split laps of one race across folds.
- **Simulate the official cut** (see "How the test races are cut") on each validation race,
  and score **one row per car** at its last visible lap. Several random cuts per race
  (e.g. 5 seeds) reduce noise. Training rows can be restricted/weighted to the same window.
- Compute the target on the **full race first** (`add_next_compound`), then cut.
- **Score only rows with a known target** (drops non-finishers, like Kaggle).
  Drop cars retired before the simulated cut entirely.
- Train only on rows with a known target (NA rows dropped).
- Metric in CV = the official log loss (`sklearn.metrics.log_loss` with the fixed class order).
- All models use the **same folds** from `src/validation.py` and the **same features**
  from `src/features.py`.
- The Kaggle test set has no labels: it is only used to create submissions.

## Feature engineering priorities

Encode the strategy rules directly rather than hoping models find them:
- compounds used so far; number of distinct dry compounds used (two-compound rule)
- feasibility: `laps_remaining` vs typical max stint of current compound at that circuit
- race context from other drivers up to the cutoff (e.g. share of field already on HARD)
- `compound_hardness` (actual C-number, harmonised for 2018) × current compound
- oracle features: `rainfall_Lplus14`, `safety_car_Lplus14`, `red_flag_Lplus14`, …
- pit penalties pending (`drive_through_pending`, `stop_go_pending`, `black_orange_flag`)
- per-driver lag/trend features over the lap history (lap time degradation)
- Only use information available up to the cutoff: forward-fill within a driver-race,
  never backward-fill.

## Model plan (baseline ladder, all scored by CV log loss)

1. Prior baseline: constant test-window class frequencies
2. Transition-matrix rule: P(next | current compound, stint), smoothed
3. Naive Bayes (CategoricalNB on binned features, one feature per collinear block) — expect overconfidence
4. Multinomial logistic regression
5. Random forest (+ calibration)
6. XGBoost / LightGBM / CatBoost — expected main model (+ calibration check)
7. Optional: small GRU over each driver's lap sequence, only once 1–6 and validation are solid
8. Final: probability blend of the best models, weights chosen on CV

Possible structure for tree models:
`P(c) = P(stop) · P(c | stop)`, `P(NONE) = 1 − P(stop)`.

## Repository structure

```
Formula-1-Tyre-Strategy/
├── data/raw/            # optional local copies (gitignored); data.py downloads via kagglehub
├── notebooks/           # exploration only (EDA)
├── src/
│   ├── __init__.py
│   ├── config.py        # SEED, CLASSES (fixed order), cut fractions (0.3, 0.5), clip eps
│   ├── data.py          # kagglehub download + loading + deterministic cleaning
│   ├── target.py        # official add_next_compound (from Kaggle, unchanged)
│   ├── features.py      # build_features(df) — shared by train and test
│   ├── validation.py    # grouped CV folds + simulated official cut + known-target filter
│   ├── metrics.py       # log loss (official), accuracy, confusion matrix, calibration
│   └── models/
│       ├── __init__.py  # MODELS registry: name -> build()
│       ├── baselines.py
│       ├── naive_bayes.py
│       ├── logreg.py
│       ├── random_forest.py
│       └── xgboost_model.py
├── evaluate.py          # python evaluate.py --model logreg  → CV log loss + plots
├── predict.py           # fit on all train → submissions/<model>_<date>.csv
├── results/             # results.csv (one row per run), confusion matrices, calibration plots
├── submissions/
├── pyproject.toml / uv.lock   # dependencies via uv (`uv add <pkg>`, never pip)
└── README.md
```

## Conventions

- Run code with `uv run` (e.g. `uv run python -m src.data`). Add packages with `uv add`.
- Kaggle token lives in `.env` (gitignored) — never commit it.
- `CLASSES = ["SOFT", "MEDIUM", "HARD", "WET_OR_INTER", "NONE"]` — this exact order
  everywhere (model output columns, metrics, submission).
- Every model module exposes `build(**params)` returning an sklearn-style estimator
  with `fit(X, y)` and `predict_proba(X)`; register it in `src/models/__init__.py`.
  Adding a model must never require editing `evaluate.py`.
- `predict.py` must check the submission against `sample_submission.csv` (ids, order,
  columns, rows sum to 1, no zeros) before writing.
- CLI uses `argparse` with double-dash flags (`--model`, `--folds`, `--save-plots`).
- One random seed (`config.SEED`) used everywhere.
- Append each evaluation run to `results/results.csv` (model, date, params, CV log loss).
- Reusable code goes in `src/` as `.py`; notebooks only for exploration.
  Clear notebook outputs before committing.

## Git workflow

- Personal branches; merge into `main` via small, frequent pull requests.
- Regularly `git pull origin main` into your branch.
- `src/features.py` and `src/validation.py` are shared — change them only via focused PRs.
- Never commit data files or `.env`.

## How Claude should help

- Integrate code into the existing scripts/modules rather than giving standalone snippets.
- When debugging, prefer hints and directional questions over handing over corrected code.
- Keep explanations concise; use tables and equations where they carry the point.
- Flag anything that could leak future information, break the train/test-window match,
  or produce overconfident probabilities.
