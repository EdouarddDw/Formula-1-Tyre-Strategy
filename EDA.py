"""Basic exploratory data analysis for the F1 tyre strategy task.

Target: `next_compound` (SOFT / MEDIUM / HARD / WET_OR_INTER / NONE), i.e. the compound of
the driver's next stint, or NONE when there is no further pit stop.

Run with:  uv run python EDA.py
Prints summaries to the terminal and saves every chart in `eda_plots/`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_data

OUT = Path("eda_plots")
OUT.mkdir(exist_ok=True)

TARGET = "next_compound"
CLASSES = ["SOFT", "MEDIUM", "HARD", "WET_OR_INTER", "NONE"]
# One fixed colour per class, used in every chart.
CLASS_COLORS = {
    "SOFT": "#eb6834",
    "MEDIUM": "#eda100",
    "HARD": "#2a78d6",
    "WET_OR_INTER": "#1baf7a",
    "NONE": "#8c8c8c",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {OUT / name}.png")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def stacked_share(ax, table, title, xlabel):
    """Stacked bar of class shares; `table` = rows are groups, columns are classes."""
    share = table.reindex(columns=CLASSES, fill_value=0)
    share = share.div(share.sum(axis=1), axis=0)
    bottom = np.zeros(len(share))
    x = np.arange(len(share))
    for cls in CLASSES:
        ax.bar(x, share[cls], bottom=bottom, color=CLASS_COLORS[cls], label=cls,
               edgecolor="white", linewidth=1)
        bottom += share[cls].to_numpy()
    ax.set_xticks(x, share.index.astype(str), rotation=45, ha="right")
    ax.set_ylabel("share of laps")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
train, test, sample_submission = load_data()
train["driver_race"] = train["race_id"] + "_" + train["driver_code"]
test["driver_race"] = test["race_id"] + "_" + test["driver_code"]

section("Shapes")
print(f"train: {train.shape}   test: {test.shape}   sample_submission: {sample_submission.shape}")
print(f"train races: {train.race_id.nunique()}   test races: {test.race_id.nunique()}   "
      f"shared races: {len(set(train.race_id) & set(test.race_id))}")
print(f"train driver-races: {train.driver_race.nunique()}   "
      f"test driver-races: {test.driver_race.nunique()}   submission rows: {len(sample_submission)}")

# ---------------------------------------------------------------------------
# 1. Target distribution
# ---------------------------------------------------------------------------
section("1. Target distribution")
counts = train[TARGET].value_counts(dropna=False)
print(pd.DataFrame({"count": counts, "share": (counts / len(train)).round(3)}))
print(f"\nMissing target: {train[TARGET].isna().sum()} laps "
      f"(result_status of these: {train.loc[train[TARGET].isna(), 'result_status'].value_counts().head(5).to_dict()})")

labelled = train[train[TARGET].notna()].copy()
labelled[TARGET] = labelled[TARGET].astype(str)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
lap_counts = labelled[TARGET].value_counts().reindex(CLASSES)
axes[0].bar(lap_counts.index, lap_counts.values, color=[CLASS_COLORS[c] for c in CLASSES])
axes[0].set_title("Target per lap (all train laps)")
axes[0].set_ylabel("laps")
for i, v in enumerate(lap_counts.values):
    axes[0].text(i, v, f"{v / lap_counts.sum():.0%}", ha="center", va="bottom", fontsize=9)

# The test set is scored once per driver-race, so the per-lap view over-weights long stints.
# Look at the target on the last observed lap of each driver in a test-like window.
last_lap_test = test.groupby("driver_race")["lap_number"].max()
lo, hi = last_lap_test.quantile([0.05, 0.95])
window = labelled[labelled.lap_number.between(lo, hi)]
per_driver = window.groupby("driver_race")[TARGET].agg(lambda s: s.sample(1, random_state=0).iloc[0])
dr_counts = per_driver.value_counts().reindex(CLASSES, fill_value=0)
axes[1].bar(dr_counts.index, dr_counts.values, color=[CLASS_COLORS[c] for c in CLASSES])
axes[1].set_title(f"Target per driver-race, one random lap in laps {lo:.0f}–{hi:.0f}\n(test-like window)")
axes[1].set_ylabel("driver-races")
for i, v in enumerate(dr_counts.values):
    axes[1].text(i, v, f"{v / dr_counts.sum():.0%}", ha="center", va="bottom", fontsize=9)
save(fig, "01_target_distribution")
print("\nClass shares in test-like window (one lap per driver-race):")
print((dr_counts / dr_counts.sum()).round(3).to_string())

# ---------------------------------------------------------------------------
# 2. Train vs test: where are the laps?
# ---------------------------------------------------------------------------
section("2. Train vs test lap coverage")
print("Laps observed per test driver-race:")
print(test.groupby("driver_race").size().describe().round(1).to_string())
fig, ax = plt.subplots(figsize=(10, 4))
bins = np.arange(0, train.lap_number.max() + 2)
ax.hist(train.lap_number, bins=bins, alpha=0.6, density=True, label="train laps", color="#2a78d6")
ax.hist(test.lap_number, bins=bins, alpha=0.6, density=True, label="test laps", color="#eb6834")
ax.hist(last_lap_test, bins=bins, histtype="step", linewidth=2, density=True,
        label="test: last observed lap per driver", color="#1a1a19")
ax.set_title("Lap number coverage: train has full races, test is truncated")
ax.set_xlabel("lap_number")
ax.set_ylabel("density")
ax.legend(frameon=False)
save(fig, "02_train_vs_test_lap_coverage")

# ---------------------------------------------------------------------------
# 3. Target vs race progress
# ---------------------------------------------------------------------------
section("3. Target vs race progress")
labelled["race_progress"] = labelled["lap_number"] / labelled["scheduled_laps"]
labelled["progress_bin"] = pd.cut(labelled["race_progress"], bins=np.linspace(0, 1.05, 22)).astype(str)
fig, ax = plt.subplots(figsize=(12, 4.5))
stacked_share(ax, pd.crosstab(labelled["progress_bin"], labelled[TARGET]).sort_index(
    key=lambda idx: idx.str.extract(r"\(([-\d.]+)")[0].astype(float)),
    "Next compound by race progress (lap / scheduled laps)", "race progress bin")
save(fig, "03_target_by_race_progress")

# ---------------------------------------------------------------------------
# 4. Current compound and stint vs next compound
# ---------------------------------------------------------------------------
section("4. Current compound / stint vs target")
ct = pd.crosstab(labelled["compound"].astype(str), labelled[TARGET], normalize="index").reindex(columns=CLASSES)
print("Row-normalised P(next_compound | current compound):")
print(ct.round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
im = axes[0].imshow(ct.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
axes[0].set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
axes[0].set_yticks(range(len(ct)), ct.index)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.iat[i, j]
        axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if v > 0.5 else "#1a1a19")
axes[0].set_title("P(next compound | current compound)")
axes[0].set_xlabel("next_compound")
axes[0].set_ylabel("current compound")
axes[0].grid(False)
fig.colorbar(im, ax=axes[0], fraction=0.04)

stacked_share(axes[1], pd.crosstab(labelled["stint"].clip(upper=5), labelled[TARGET]),
              "Next compound by stint number (5 = 5+)", "stint")
save(fig, "04_target_by_current_compound_and_stint")

# ---------------------------------------------------------------------------
# 5. Target over seasons and circuits
# ---------------------------------------------------------------------------
section("5. Target over years and circuits")
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 2]})
stacked_share(axes[0], pd.crosstab(labelled["year"], labelled[TARGET]), "Next compound by season", "year")
by_circuit = pd.crosstab(labelled["circuit_id"], labelled[TARGET])
by_circuit = by_circuit.loc[(by_circuit.div(by_circuit.sum(axis=1), axis=0)).get("NONE", 0).sort_values().index]
stacked_share(axes[1], by_circuit, "Next compound by circuit (sorted by NONE share)", "circuit")
save(fig, "05_target_by_year_and_circuit")

# ---------------------------------------------------------------------------
# 6. Key numeric features by target class
# ---------------------------------------------------------------------------
section("6. Numeric features by target")
key_feats = ["tyre_life_laps", "laps_remaining", "race_progress", "position",
             "track_temp_c", "air_temp_c", "humidity_pct", "gap_to_car_ahead_s"]
print(labelled.groupby(TARGET)[key_feats].median().reindex(CLASSES).round(2).to_string())

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, feat in zip(axes.flat, key_feats):
    data = [labelled.loc[labelled[TARGET] == c, feat].dropna().astype(float) for c in CLASSES]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, cls in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color("#1a1a19")
    ax.set_xticks(range(1, len(CLASSES) + 1), ["SOFT", "MED", "HARD", "WET/I", "NONE"], fontsize=8)
    ax.set_title(feat, fontsize=10)
fig.suptitle("Key numeric features by next_compound (outliers hidden)", fontweight="bold")
save(fig, "06_numeric_features_by_target")

# Weather: does rain now / in 14 laps predict a switch to wets?
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["rainfall", "rainfall_Lplus14"]):
    stacked_share(ax, pd.crosstab(labelled[col].astype(str), labelled[TARGET]),
                  f"Next compound by {col}", col)
save(fig, "07_target_by_rainfall")

# ---------------------------------------------------------------------------
# 7. Feature–target association (one-vs-rest correlation per class)
# ---------------------------------------------------------------------------
section("7. Feature-target correlation (one-vs-rest)")
num = labelled.select_dtypes(include=["number", "bool", "boolean"]).astype(float)
num = num.loc[:, num.nunique() > 1]
onehot = pd.get_dummies(labelled[TARGET]).reindex(columns=CLASSES).astype(float)
assoc = pd.DataFrame({c: num.corrwith(onehot[c]) for c in CLASSES})
assoc = assoc.loc[assoc.abs().max(axis=1).sort_values(ascending=False).index]
top = assoc.head(25)
print("Top 25 features by max |corr| with any class:")
print(top.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(top.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
ax.set_yticks(range(len(top)), top.index, fontsize=9)
for i in range(top.shape[0]):
    for j in range(top.shape[1]):
        ax.text(j, i, f"{top.iat[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
ax.set_title("Correlation of numeric features with each class\n(one-vs-rest, top 25)")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.04)
save(fig, "08_feature_target_correlation")

# ---------------------------------------------------------------------------
# 8. Feature–feature correlation (redundancy)
# ---------------------------------------------------------------------------
section("8. Feature-feature correlation")
corr = num.corr()
pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
         .rename("corr").reset_index())
pairs = pairs.loc[pairs["corr"].abs() > 0.9].sort_values("corr", key=abs, ascending=False)
print(f"Highly correlated pairs (|r| > 0.9): {len(pairs)}  -> candidates to drop/merge")
print(pairs.head(30).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
ax.set_title("Correlation between numeric features")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.03)
save(fig, "09_feature_correlation_matrix")

# ---------------------------------------------------------------------------
# 9. Missing values
# ---------------------------------------------------------------------------
section("9. Missing values")
miss = pd.DataFrame({"train": train.drop(columns=[TARGET]).isna().mean(),
                     "test": test.isna().mean()}).dropna()
miss = miss.loc[miss.max(axis=1) > 0].sort_values("train")
print(miss.round(3).to_string())
fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(miss))))
y = np.arange(len(miss))
ax.barh(y - 0.2, miss["train"], height=0.4, label="train", color="#2a78d6")
ax.barh(y + 0.2, miss["test"], height=0.4, label="test", color="#eb6834")
ax.set_yticks(y, miss.index, fontsize=8)
ax.set_xlabel("share missing")
ax.set_title("Missing values per column")
ax.legend(frameon=False)
save(fig, "10_missing_values")

print(f"\nDone. Charts are in {OUT.resolve()}")
