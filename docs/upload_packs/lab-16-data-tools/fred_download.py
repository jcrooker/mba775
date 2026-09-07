"""Download a FRED series for the MBA 775 forecasting challenge.

    python fred_download.py CPIAUCSL            # writes data/raw/CPIAUCSL.csv
    python fred_download.py ICSA HOUST NVSTHPI  # several at once
    python fred_download.py CPIAUCSL --start 2015-01-01

Two ways to reach FRED, tried in this order:

  1. The public CSV endpoint, no key needed:
         https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL
     This is the same file the "Download > CSV" button on a FRED series page
     gives you. If your network blocks it, use the button and save the file
     into data/raw/ yourself -- the rest of the course works the same way.

  2. The FRED API through the `fredapi` package, if you have set a free API
     key in the environment variable FRED_API_KEY:
         setx FRED_API_KEY "your-key"        (Windows, then open a new terminal)
         export FRED_API_KEY="your-key"      (Mac / Linux)
     Never paste the key into a script, a notebook, or a chat message.

Either way the output is the same tidy CSV:

    date,value
    2015-01-01,234.747
    ...

with a companion <ID>.provenance.txt recording the series ID, retrieval
date, units and frequency (which the API supplies and the CSV endpoint does
not -- for the CSV route, copy them from the FRED series page).

This file DOWNLOADS. It is meant for your own computer, or for Claude when
your account has web access. The course's analysis scripts never download;
they read the CSV this produces. If you cannot download at all, ask Claude to
say so plainly rather than inventing a series.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (MBA775 coursework)", "Accept": "text/csv,*/*"}

# What a student needs to know about each challenge series. Frequencies and
# units are as FRED states them; check the series page if in doubt.
CHALLENGE_SERIES = {
    "CPIAUCSL": ("Consumer Price Index, all urban consumers (SA)", "Index 1982-84=100", "Monthly"),
    "ICSA":     ("Initial claims for unemployment insurance (SA)", "Number of claims", "Weekly, Saturday-ended"),
    "HOUST":    ("New privately-owned housing units started (SAAR)", "Thousands of units", "Monthly"),
    "NVSTHPI":  ("All-transactions house price index for Nevada (NSA)", "Index 1980Q1=100", "Quarterly"),
}


def via_csv_endpoint(series_id: str, start: str | None = None) -> pd.DataFrame:
    url = f"{FRED_CSV}?id={series_id}" + (f"&cosd={start}" if start else "")
    with urlopen(Request(url, headers=_HEADERS), timeout=60) as resp:
        text = resp.read().decode("utf-8")
    frame = pd.read_csv(StringIO(text), na_values=["."])
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def via_api(series_id: str, start: str | None = None) -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set in the environment.")
    from fredapi import Fred                    # pip install fredapi
    fred = Fred(api_key=key)
    s = fred.get_series(series_id, observation_start=start)
    frame = s.rename("value").rename_axis("date").reset_index()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def download(series_id: str, start: str | None = None, out_dir: str | Path = "data/raw") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    frame, route = None, None
    for name, fn in (("csv endpoint", via_csv_endpoint), ("fredapi", via_api)):
        try:
            frame, route = fn(series_id, start), name
            break
        except Exception as exc:                # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if frame is None:
        raise SystemExit(
            f"Could not download {series_id}.\n  " + "\n  ".join(errors) +
            "\nUse the Download > CSV button on the FRED series page and save the "
            f"file as {out_dir / (series_id + '.csv')}.")

    path = out_dir / f"{series_id}.csv"
    frame.to_csv(path, index=False)
    desc, units, freq = CHALLENGE_SERIES.get(series_id, ("", "see FRED page", "see FRED page"))
    missing = int(frame["value"].isna().sum())
    prov = (
        f"series_id: {series_id}\n"
        f"description: {desc}\n"
        f"source: https://fred.stlouisfed.org/series/{series_id}\n"
        f"retrieved: {date.today().isoformat()} via {route}\n"
        f"units: {units}\nfrequency: {freq}\n"
        f"observations: {len(frame)}  first: {frame['date'].min().date()}  "
        f"last: {frame['date'].max().date()}  missing: {missing}\n"
    )
    (out_dir / f"{series_id}.provenance.txt").write_text(prov)
    print(prov)
    if missing:
        print(f"NOTE: {missing} observation(s) are missing ('.' on FRED). They are kept as "
              f"blanks in the CSV. Decide how to treat them before forecasting.")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("series", nargs="+", help="FRED series IDs, e.g. CPIAUCSL ICSA")
    ap.add_argument("--start", default=None, help="first observation date, YYYY-MM-DD")
    ap.add_argument("--out", default="data/raw", help="output folder (default data/raw)")
    args = ap.parse_args()
    for sid in args.series:
        download(sid.upper(), args.start, args.out)
