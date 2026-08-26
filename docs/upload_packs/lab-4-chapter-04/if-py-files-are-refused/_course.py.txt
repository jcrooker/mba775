"""Shared helpers for MBA 775 scripts.

The only job of this file is to find a data file no matter where you happen to
have put it, so that a script works whether you are running it in Claude's
sandbox with the CSV uploaded alongside it, in a folder you cloned from
GitHub, or from a notebook somewhere else entirely.

You do not need to read this file to do the coursework, but you are welcome
to. It is short.
"""

from pathlib import Path

import pandas as pd

# Places a data file might reasonably live, relative to this file and to
# wherever Python happens to be running.
# Order matters. A dedicated data/ folder is checked BEFORE the working
# directory, so a stray file left at the project root by some other process
# cannot silently shadow the canonical data file. Loading the wrong file with
# the right name is the kind of error that produces confident wrong answers.
_SEARCH_DIRS = [
    Path.cwd() / "data",
    Path(__file__).resolve().parent / "data",
    Path(__file__).resolve().parent.parent / "data",
    Path.cwd(),
    Path(__file__).resolve().parent,
    Path("/mnt/user-data/uploads"),
    Path("/mnt/data"),
]


def find_data(filename: str) -> Path:
    """Return the path to `filename`, searching the usual places.

    Raises FileNotFoundError with a clear, actionable message if it is
    missing, rather than letting pandas fail with a bare path error.
    """
    tried = []
    for folder in _SEARCH_DIRS:
        candidate = folder / filename
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate

    # Browsers rename repeat downloads: dff.csv becomes "dff (1).csv", and on
    # some systems "dff-1.csv" or "dff copy.csv". A student who downloads twice
    # would otherwise hit a file-not-found error for a file they are looking at.
    # Accept those variants -- but SAY SO, loudly. Silently loading a file the
    # script did not ask for is exactly the failure this course warns about.
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for folder in _SEARCH_DIRS:
        if not folder.is_dir():
            continue
        variants = sorted(
            f for f in folder.glob(f"{stem}*{suffix}")
            if f.is_file() and f.name != filename
        )
        if variants:
            chosen = variants[0]
            print(f"NOTE: '{filename}' was not found. Using '{chosen.name}' "
                  f"instead, which appears to be a renamed copy.")
            if len(variants) > 1:
                others = ", ".join(f.name for f in variants[1:])
                print(f"      Other candidates were also present: {others}")
                print(f"      Check that the file used is the one you intended.")
            return chosen

    raise FileNotFoundError(
        f"Could not find '{filename}'.\n"
        f"Upload it alongside this script, or download it from the course "
        f"repository at https://github.com/jcrooker/mba775 (data/ folder).\n"
        f"Looked in:\n" + "".join(f"    {p}\n" for p in tried)
    )


def load_dff(filename: str = "dff.csv") -> pd.Series:
    """Load the Federal Funds Effective Rate as a Series indexed by date."""
    path = find_data(filename)
    frame = pd.read_csv(path, parse_dates=["date"])
    series = frame.set_index("date")["federal_funds_rate"].astype("float64")
    series = series.sort_index()
    series.name = "federal_funds_rate"
    return series


def load_state_unemployment(filename: str = "state_unemployment.csv") -> pd.DataFrame:
    """Load the tidy state unemployment table (date, state, unemployment_rate)."""
    path = find_data(filename)
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.sort_values(["date", "state"]).reset_index(drop=True)


def month_coverage(series: pd.Series) -> pd.DataFrame:
    """Days observed in each month, days the month actually has, and whether
    the month is complete.

    Resampling silently produces a bucket for the current month whatever the
    date is. A month with three days of data appears beside months with thirty,
    formatted identically, and any average or change computed from it is not
    comparable. This makes that visible instead of leaving it to be discovered.
    """
    observed = series.resample("MS").count()
    calendar_days = observed.index.days_in_month
    return pd.DataFrame({
        "observed": observed.astype(int),
        "in_month": calendar_days.astype(int),
        "complete": (observed.to_numpy() >= calendar_days.to_numpy()),
    }, index=observed.index)


def banner(title: str) -> None:
    """Print a section heading, so long output stays readable."""
    print()
    print(title)
    print("-" * len(title))
