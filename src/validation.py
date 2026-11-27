"""Grouped CV folds with a simulated official test cut.

Official rule: each test race is cut when the leader finishes a lap between 30% and 50%
of the race (u ~ U(0.3, 0.5) per race), using race_time_s. Every car shows only the laps
it had finished by then.

Diagnostics: `uv run python -m src.validation`
"""

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import CLASSES, CUT_FRAC_MAX, CUT_FRAC_MIN, DRIVER_KEYS, N_FOLDS, SEED, TARGET
from src.target import add_next_compound

DEFAULT_MODE = "laps"  # chosen from the diagnostics below (matches the test set)
DEFAULT_N_CUTS = 5


def leader_lap_times(race_df: pd.DataFrame) -> pd.Series:
    """race_time_s of the leader (position 1) at the end of each lap, indexed by lap_number."""
    lead = race_df[race_df["position"] == 1].groupby("lap_number")["race_time_s"].min()
    fastest = race_df.groupby("lap_number")["race_time_s"].min()  # fallback if no position 1
    return lead.reindex(fastest.index).fillna(fastest).dropna().sort_index()


def cut_time(race_df: pd.DataFrame, u: float, mode: str = DEFAULT_MODE) -> float:
    """T_cut: leader's race_time_s at the end of the first leader lap reaching fraction u.

    mode="laps": lap k >= u * scheduled_laps.  mode="time": leader time >= u * winner's total time.
    """
    lead = leader_lap_times(race_df)
    if mode == "laps":
        k = math.ceil(u * race_df["scheduled_laps"].iloc[0] - 1e-9)
        crossed = lead[lead.index >= k]
    elif mode == "time":
        crossed = lead[lead >= u * lead.iloc[-1]]
    else:
        raise ValueError(f"mode must be 'laps' or 'time', got {mode!r}")
    return float(crossed.iloc[0])


def cut_race(race_df: pd.DataFrame, u: float, mode: str = DEFAULT_MODE) -> pd.DataFrame:
    """Laps visible after cutting the race at fraction u (race_time_s <= T_cut)."""
    t_cut = cut_time(race_df, u, mode)
    return race_df[race_df["race_time_s"] <= t_cut]


def last_visible_laps(race_df: pd.DataFrame, u: float, mode: str = DEFAULT_MODE) -> pd.DataFrame:
    """One row per car still running at the cut, at its last visible lap (like the test set).

    Cars that retired before the cut (no finished lap after T_cut) are dropped.
    """
    t_cut = cut_time(race_df, u, mode)
    running = race_df.groupby(DRIVER_KEYS)["race_time_s"].transform("max") > t_cut
    visible = race_df[(race_df["race_time_s"] <= t_cut) & running]
    last = visible.sort_values("lap_number").groupby(DRIVER_KEYS, sort=False).tail(1)
    return last.assign(cut_u=u, cut_time_s=t_cut).sort_values(DRIVER_KEYS)


def make_eval_rows(
    race_df: pd.DataFrame, rng: np.random.Generator, mode: str = DEFAULT_MODE, n_cuts: int = 1
) -> pd.DataFrame:
    """Simulated eval rows of one full race: target on the full race, then cut, one row per car.

    Drops cars with an NA target (not scored on Kaggle) and cars retired before the cut.
    With n_cuts > 1, repeats with different u; `cut_id` identifies the cut.
    """
    race_df = add_next_compound(race_df, name=TARGET)
    out = []
    for cut_id, u in enumerate(rng.uniform(CUT_FRAC_MIN, CUT_FRAC_MAX, size=n_cuts)):
        rows = last_visible_laps(race_df, u, mode)
        out.append(rows[rows[TARGET].notna()].assign(cut_id=cut_id))
    return pd.concat(out, ignore_index=True)


def get_folds(
    train: pd.DataFrame,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
    mode: str = DEFAULT_MODE,
    n_cuts: int = DEFAULT_N_CUTS,
):
    """Yield (train_df, eval_df) per fold, grouped by race_id.

    train_df: all laps of the training races with a known target.
    eval_df: simulated official-cut rows of the held-out races (one per car per cut).
    """
    train = add_next_compound(train, name=TARGET)
    rng = np.random.default_rng(seed)
    races = sorted(train["race_id"].unique())
    eval_rows = pd.concat(  # cuts drawn once per race, independent of the fold split
        [make_eval_rows(train[train["race_id"] == r], rng, mode, n_cuts) for r in races],
        ignore_index=True,
    )
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, ev_idx in gkf.split(train, groups=train["race_id"]):
        ev_races = train["race_id"].iloc[ev_idx].unique()
        train_df = train.iloc[tr_idx]
        yield (
            train_df[train_df[TARGET].notna()].reset_index(drop=True),
            eval_rows[eval_rows["race_id"].isin(ev_races)].reset_index(drop=True),
        )


if __name__ == "__main__":
    from scipy.stats import ks_2samp

    from src.config import make_id
    from src.data import load_data

    train, test, sample_submission = load_data()
    train = add_next_compound(train, name=TARGET)
    races = [g for _, g in train.groupby("race_id")]

    # Test: each submission car's last visible lap; leader's lap fraction per race
    test_last = test.sort_values("lap_number").groupby(DRIVER_KEYS).tail(1)
    test_last = test_last[make_id(test_last).isin(sample_submission["id"])]
    test_lead_frac = test.groupby("race_id").apply(
        lambda g: leader_lap_times(g).index[-1] / g["scheduled_laps"].iloc[0]
    )

    # 1. Simulated last visible lap (all cars running at the cut, like sample_submission)
    n_sim = 20
    qs = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    sims, fracs = {}, {}
    for mode in ("laps", "time"):
        rng = np.random.default_rng(SEED)
        rows, lead = [], []
        for g in races:
            for u in rng.uniform(CUT_FRAC_MIN, CUT_FRAC_MAX, size=n_sim):
                rows.append(last_visible_laps(g, u, mode))
                k = leader_lap_times(g).loc[lambda s: s <= cut_time(g, u, mode)].index[-1]
                lead.append(k / g["scheduled_laps"].iloc[0])
        sims[mode] = pd.concat(rows)
        fracs[mode] = pd.Series(lead)

    print(f"1. Last visible lap per car: test ({len(test_last)} cars) vs simulated "
          f"({len(races)} train races x {n_sim} cuts)\n")
    for label, f in [
        ("last lap_number", lambda d: d["lap_number"]),
        ("last lap / scheduled_laps", lambda d: d["lap_number"] / d["scheduled_laps"]),
    ]:
        tab = pd.DataFrame({"test": f(test_last).quantile(qs).values}, index=qs)
        for mode in sims:
            tab[mode] = f(sims[mode]).quantile(qs).values
        print(f"   {label} (quantiles)")
        print(tab.round(3).to_string(), "\n")
        for mode in sims:
            ks = ks_2samp(f(test_last).astype(float), f(sims[mode]).astype(float))
            print(f"   KS {mode:5}: D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}")
        print()

    tab = pd.DataFrame({"test": test_lead_frac.quantile(qs).values}, index=qs)
    for mode in fracs:
        tab[mode] = fracs[mode].quantile(qs).values
    print("   leader's last visible lap / scheduled_laps, one per race (quantiles)")
    print(tab.round(3).to_string(), "\n")
    for mode in fracs:
        ks = ks_2samp(test_lead_frac.astype(float), fracs[mode])
        print(f"   KS {mode:5}: D = {ks.statistic:.3f}, p = {ks.pvalue:.3g}")

    # 2. Class distribution of eval rows (default mode, many cuts)
    rng = np.random.default_rng(SEED)
    ev = pd.concat([make_eval_rows(g, rng, DEFAULT_MODE, n_cuts=50) for g in races])
    print(f"\n2. Eval-row class distribution, mode={DEFAULT_MODE!r}, 50 cuts/race "
          f"({len(ev)} rows, {len(ev) / 50:.0f} cars per cut)")
    dist = ev[TARGET].value_counts(normalize=True).reindex(CLASSES)
    print((100 * dist).round(1).astype(str).add(" %").to_string())
    lap_all = train[train[TARGET].notna()][TARGET].value_counts(normalize=True).reindex(CLASSES)
    print("   (all known-target train laps: "
          + ", ".join(f"{c} {100 * p:.1f}%" for c, p in lap_all.items()) + ")")

    # 3. No race in both train_df and eval_df of any fold
    print("\n3. Fold check")
    seen = []
    for i, (tr_df, ev_df) in enumerate(get_folds(load_data()[0])):
        overlap = set(tr_df["race_id"]) & set(ev_df["race_id"])
        seen += list(ev_df["race_id"].unique())
        print(f"   fold {i}: train {tr_df['race_id'].nunique()} races / {len(tr_df)} laps, "
              f"eval {ev_df['race_id'].nunique()} races / {len(ev_df)} rows, overlap {len(overlap)}")
        assert not overlap
    assert len(seen) == len(set(seen)) == len(races), "each race must be held out exactly once"
    print("   OK: disjoint races, every race held out exactly once")
