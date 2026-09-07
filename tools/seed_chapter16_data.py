"""Build the student-facing data files for Chapter 16 (the forecasting challenge).

Produces, in data/:

    cpi.csv           CPIAUCSL, monthly from 1970-01, missing months KEPT as blanks
    icsa.csv          ICSA, weekly from 2010-01
    houst.csv         HOUST, monthly from 1990-01
    nvsthpi.csv       NVSTHPI, quarterly, full history
    lv_visitors.csv   LVCVA Visitor Volume, monthly, every Year-End workbook + YTD

Run from the repository root, with internet access:

    python tools/seed_chapter16_data.py            # refresh everything
    python tools/seed_chapter16_data.py --no-lvcva  # FRED only

FRED is read through tools/fred_tools.py (the keyless CSV endpoint, with the
transport fallbacks that file already knows about). If FRED refuses Python's
HTTPS client on your network, run `Rscript tools/seed_fred_cache.R` style
downloads or use the Download > CSV button and drop the files in data/raw/
under their series IDs; this script will pick them up from the cache.

The LVCVA workbooks are read through scripts/lvcva_tools.py. Unlike the
earlier seeders this one does NOT drop missing observations: a hole in a
series is something a forecaster has to decide about, so it stays visible.

After running, update the date ranges and retrieval dates in data/README.md
(the Chapter 16 section) and rebuild the upload packs:

    python tools/build_upload_packs.py --verify
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

import fred_tools                                                   # noqa: E402

DATA = REPO / "data"
RAW = DATA / "raw"

SERIES = {
    # file            FRED id      column            start
    "cpi.csv":      ("CPIAUCSL", "cpi",            "1970-01-01"),
    "icsa.csv":     ("ICSA",     "initial_claims", "2010-01-01"),
    "houst.csv":    ("HOUST",    "housing_starts", "1990-01-01"),
    "nvsthpi.csv":  ("NVSTHPI",  "nv_hpi",         None),
}


def seed_fred() -> None:
    DATA.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)
    fred_tools.set_cache_dir(str(RAW))
    for fname, (sid, col, start) in SERIES.items():
        s = fred_tools.fred_series(sid, start=start, refresh=True)
        frame = s.rename(col).rename_axis("date").reset_index()
        frame["date"] = pd.to_datetime(frame["date"])
        # Re-insert calendar gaps as blank rows so a missing month is visible.
        freq = {"CPIAUCSL": "MS", "HOUST": "MS", "NVSTHPI": "QS", "ICSA": "W-SAT"}[sid]
        full = pd.DataFrame({"date": pd.date_range(frame["date"].min(), frame["date"].max(), freq=freq)})
        frame = full.merge(frame, on="date", how="left")
        frame.to_csv(DATA / fname, index=False)
        miss = int(frame[col].isna().sum())
        print(f"{fname:<16} {sid:<9} {len(frame):>5} rows  "
              f"{frame['date'].min().date()} to {frame['date'].max().date()}  missing {miss}")


def seed_lvcva() -> None:
    import lvcva_tools
    lvcva_tools.write_visitors_csv(DATA / "lv_visitors.csv")


if __name__ == "__main__":
    print(f"Seeding Chapter 16 data, {date.today().isoformat()}\n")
    seed_fred()
    if "--no-lvcva" not in sys.argv:
        print()
        seed_lvcva()
    print("\nNow update the Chapter 16 section of data/README.md and run "
          "python tools/build_upload_packs.py --verify")
