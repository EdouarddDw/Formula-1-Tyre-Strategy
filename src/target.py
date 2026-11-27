"""target.py: add the prediction target `next_compound` to a laps table (one row = one driver on one lap).

What the target is
------------------
For a row with lap_number L (think of L as "the last lap we have seen of this car"):
- the car makes another official pit stop after lap L, on lap S:   target = the tyre it drives on lap S + 1
  (a stop on lap L itself is already visible and does not count):
      SOFT, MEDIUM or HARD (dry tyres), WET_OR_INTER (INTERMEDIATE or WET rain tyre);
  SOFT/MEDIUM/HARD = softest/middle/hardest of the three dry compounds brought to that race (the relative `compound`
  column, never `compound_hardness`; e.g. the C3 rubber is SOFT at one race and HARD at another);
- the car does not stop again and finished the race:   target = NONE (it drives to the flag on its current tyres).
  "Finished" = result_status 'Finished', '+1 Lap', '+2 Laps', ... or 'Lapped', so lapped cars count as finishers;
- otherwise NA (missing): the car does not stop again and did not finish (retired, disqualified, ...), or the tyre after
  the stop is unknown (no lap S + 1, e.g. the car retired in the pits, or no tyre recorded on it).

In the competition, only cars that finish the race and have a known target are scored.

Tiny tutorial
-------------
    import pandas as pd
    from target import add_next_compound      # this file must be in your working folder (or on sys.path)

    train = pd.read_parquet("train_laps.parquet")
    train = add_next_compound(train)           # (re)computes the column next_compound
    train.next_compound.value_counts(dropna=False)

    # one car: Alonso, 2018 Bahrain (stops on laps 14 and 39)
    car = train[(train.race_id == "2018_02") & (train.driver_code == "ALO")]
    car[["lap_number", "compound", "official_stop", "next_compound"]]

The test races are cut, so their future is missing. add_next_compound(test, complete_races=False) fills in only the rows
whose next stop and the lap after it are already visible (all others NA): handy for using the early laps of test races, but
these rows are biased towards early stops.

Run `python target.py` in the folder with train_laps.parquet to see the example car.
Needs the columns race_id, driver_code, lap_number, official_stop, compound, result_status.
"""
import pandas as pd

CLASSES = ["SOFT", "MEDIUM", "HARD", "WET_OR_INTER", "NONE"]
TYRE = {"SOFT": "SOFT", "MEDIUM": "MEDIUM", "HARD": "HARD", "INTERMEDIATE": "WET_OR_INTER", "WET": "WET_OR_INTER"}
FINISHED = r"^(?:Finished|\+\d+ Laps?|Lapped)$"     # result_status texts of cars that finished (also laps down)
CAR = ["race_id", "driver_code"]


def add_next_compound(df: pd.DataFrame, name: str = "next_compound", complete_races: bool = True) -> pd.DataFrame:
    """Return a copy of df with the target column `name` added (text, NA where unknown). Row order is kept.

    complete_races=True: races with all laps (train_laps). False: cut races (test_laps), only rows whose next stop and the
    lap after it are visible.
    """
    d = df.reset_index(drop=True).sort_values(CAR + ["lap_number"], kind="stable")
    car = [d[c] for c in CAR]                                                  # group by car: one race, one driver

    # 1. the next official stop strictly after lap L (lap number), and the tyre on the lap after that stop
    stop_lap = d.lap_number.where(d.official_stop.astype(bool))
    next_stop = stop_lap.groupby(car).transform(lambda s: s.shift(-1).bfill())   # shift(-1): skip this lap; bfill: next stop below
    tyre = d.set_index(CAR + ["lap_number"]).compound.map(TYRE)                  # tyre of every (car, lap); NaN if unknown
    tyre = tyre[~tyre.index.duplicated()]
    key = pd.MultiIndex.from_arrays([d.race_id, d.driver_code, next_stop + 1])
    y = pd.Series(tyre.reindex(key).values, index=d.index, dtype="object")

    # 2. no further stop: NONE for cars that finished (NA for the others)
    if complete_races:
        finished = d.result_status.str.match(FINISHED).fillna(False).astype(bool)
        y = y.where(next_stop.notna(), pd.Series("NONE", index=d.index).where(finished))

    out = df.copy()
    out[name] = pd.array(y.sort_index().values, dtype="string")              # back to the original row order
    return out


if __name__ == "__main__":
    train = add_next_compound(pd.read_parquet("train_laps.parquet"))
    car = train[(train.race_id == "2018_02") & (train.driver_code == "ALO")]
    print(car[["lap_number", "compound", "official_stop", "next_compound"]].to_string(index=False))
    print("\n", train.next_compound.value_counts(dropna=False).to_string())
