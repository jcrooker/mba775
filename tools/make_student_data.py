"""Build the student-facing data files from the fred_tools cache.

INSTRUCTOR TOOL. Students never run this.

`seed_fred_cache.R` fills data/raw/ with one CSV per FRED request, named for
the request rather than for a human. This script turns that cache into a small
number of tidy, clearly named files that students can load in one line:

    data/raw/DFF__none__none.csv          ->  data/dff.csv
    data/raw/<ST>UR__1976-01-01__none.csv ->  data/state_unemployment.csv

Run from the repository root:

    python tools/make_student_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data")

STATE_ABB = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def read_cached(name: str) -> pd.Series:
    """Read one cache file into a Series indexed by date."""
    path = RAW / name
    frame = pd.read_csv(path, na_values=["."], keep_default_na=True)
    date_col, value_col = frame.columns[0], frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    series = frame.set_index(date_col)[value_col].astype("float64").sort_index()
    series.index.name = "date"
    return series


def build_dff() -> None:
    series = read_cached("DFF__none__none.csv")
    out = series.rename("federal_funds_rate").to_frame()
    out.to_csv(OUT / "dff.csv")
    print(f"  dff.csv                  {len(out):>7,} rows  "
          f"{out.index.min().date()} to {out.index.max().date()}  "
          f"({out['federal_funds_rate'].isna().sum()} missing)")


def build_state_unemployment() -> None:
    frames = []
    missing = []
    for abb in STATE_ABB:
        name = f"{abb}UR__1976-01-01__none.csv"
        if not (RAW / name).exists():
            missing.append(abb)
            continue
        s = read_cached(name)
        frames.append(pd.DataFrame({
            "date": s.index,
            "state": abb,
            "unemployment_rate": s.to_numpy(),
        }))

    if missing:
        print(f"  WARNING: no cache file for {', '.join(missing)}", file=sys.stderr)

    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.sort_values(["date", "state"]).reset_index(drop=True)
    tidy.to_csv(OUT / "state_unemployment.csv", index=False)

    print(f"  state_unemployment.csv   {len(tidy):>7,} rows  "
          f"{tidy['state'].nunique()} states  "
          f"{tidy['date'].min().date()} to {tidy['date'].max().date()}")


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(
            f"{RAW} not found. Run seed_fred_cache.R first, from the folder "
            f"that should contain data/raw/."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    print("Building student data files:")
    build_dff()
    build_state_unemployment()
    print("\nDone. Commit these to the repository so students can download them.")


if __name__ == "__main__":
    main()
