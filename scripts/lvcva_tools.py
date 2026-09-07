"""Las Vegas visitor volume from the LVCVA, for the MBA 775 forecasting challenge.

Python translation of the course's LIB_LVCVA_Utilities.R. The Las Vegas
Convention and Visitors Authority does not publish visitor volume on FRED; it
posts an Excel workbook for each year on its research page:

    https://www.lvcva.com/research/

One "Year-to-Date Summary" workbook for the current year and one "Year-End
Summary" workbook for each completed year. Row 7 of the first sheet holds the
month headers, row 8 holds "Visitor Volume", one value every other column
(the columns between carry an 'r' flag when a figure has been revised).

The functions below do what the R versions did:

    identify_current_year_spreadsheet()       URL of the Year-to-Date workbook
    identify_available_annual_spreadsheets()  URLs of the Year-End workbooks
    download_most_recent_data(target_dir)     save today's copy of the YTD file
    load_las_vegas_visitor_data_from_xlsx()   one workbook -> Year, Month, value
    recent_annual_visitors_data()             every Year-End workbook, stacked
    recent_timepath_visitors_data()           full monthly series as a Series

Run it as a script to write data/lv_visitors.csv:

    python lvcva_tools.py

Needs: pandas, openpyxl, and internet access. This file DOWNLOADS. Use it on
your own computer (or in Claude if your account has web access). The course
scripts themselves never download; they read the CSV this produces.

Be polite to the LVCVA's server: the functions pause a few seconds between
workbook downloads, as the R version did. Do not remove the pause.
"""

from __future__ import annotations

import io
import random
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

__all__ = [
    "identify_current_year_spreadsheet",
    "identify_available_annual_spreadsheets",
    "download_most_recent_data",
    "load_las_vegas_visitor_data_from_xlsx",
    "recent_annual_visitors_data",
    "recent_timepath_visitors_data",
    "write_visitors_csv",
]

LVCVA_RESEARCH_URL = "https://www.lvcva.com/research/"

# Older link text for the archived executive-summary page; kept so the
# functions still find files if the LVCVA moves them back.
LVCVA_ARCHIVE_URL = ("https://www.lvcva.com/research/reports/post/"
                     "lvcva-executive-summary-of-southern-nevada-tourism-indicators/")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "*/*",
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def _get(url: str, timeout: float = 60.0) -> bytes:
    req = Request(url, headers=_HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _page_links(url: str = LVCVA_RESEARCH_URL) -> list[tuple[str, str]]:
    """(link text, href) for every .xlsx link on the page."""
    html = _get(url).decode("utf-8", errors="replace")
    out = []
    for m in re.finditer(r'<a\s[^>]*href="([^"]+\.xlsx)"[^>]*>(.*?)</a>', html,
                         flags=re.S | re.I):
        href = m.group(1)
        text = re.sub(r"<[^>]+>", " ", m.group(2))
        text = re.sub(r"\s+", " ", text).strip()
        out.append((text, href))
    return out


def identify_current_year_spreadsheet() -> str | None:
    """URL of the 'Year-to-Date Summary' workbook, or None if not found."""
    for text, href in _page_links():
        if re.search(r"Year[- ]to[- ]Date", text, flags=re.I):
            return href
    return None


def identify_available_annual_spreadsheets() -> list[str]:
    """URLs of every 'Year-End Summary' workbook, newest first as listed."""
    return [href for text, href in _page_links()
            if re.search(r"Year[- ]End", text, flags=re.I)]


def download_most_recent_data(target_dir: str | Path = "./lv_visitors") -> Path | None:
    """Save today's copy of the Year-to-Date workbook. Skips if it exists."""
    url = identify_current_year_spreadsheet()
    if url is None:
        print("Could not find a Year-to-Date workbook on the LVCVA research page.")
        return None
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / f"lv-visitors-{date.today().isoformat()}.xlsx"
    if not dest.exists():
        dest.write_bytes(_get(url))
    return dest


def load_las_vegas_visitor_data_from_xlsx(variable: str = "Visitor Volume",
                                          xlsxFile: str | Path | None = None,
                                          ) -> pd.DataFrame | None:
    """Read one LVCVA summary workbook and return Year, Month, value.

    `xlsxFile` may be a URL or a local path. Defaults to the current YTD file.
    Months not yet published come back as NaN.
    """
    if xlsxFile is None:
        xlsxFile = identify_current_year_spreadsheet()
        if xlsxFile is None:
            return None
    src = str(xlsxFile)
    raw = _get(src) if src.lower().startswith("http") else Path(src).read_bytes()

    # Row 7 (1-based) is the header row; read with no header so nothing is
    # guessed, then find the rows ourselves.
    sheet = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=None)
    first_col = sheet.iloc[:, 0].astype(str)
    hdr_rows = first_col[first_col.str.match(r"^\s*Tourism Indicators", case=False, na=False)].index
    var_rows = first_col[first_col.str.match(r"^\s*" + re.escape(variable), case=False, na=False)].index
    if len(hdr_rows) == 0 or len(var_rows) == 0:
        raise ValueError(f"Could not find the header row or '{variable}' in {src}")
    hdr = sheet.iloc[hdr_rows[0]]
    row = sheet.iloc[var_rows[0]]

    # The year comes from the "<year> YTD" cell in the header row.
    ytd_cells = [str(v) for v in hdr if re.search(r"\d{4}\s*YTD", str(v))]
    if not ytd_cells:
        raise ValueError(f"No '<year> YTD' cell found in the header row of {src}")
    year = int(re.search(r"(\d{4})", ytd_cells[0]).group(1))

    # Month columns are the ones whose header is a date (Excel serial or
    # datetime). Values sit in the same columns on the variable's row.
    values = []
    for col in sheet.columns:
        h = hdr[col]
        is_date = isinstance(h, (pd.Timestamp, datetime)) or (
            isinstance(h, (int, float)) and not pd.isna(h) and 30000 < float(h) < 80000)
        if is_date:
            v = pd.to_numeric(row[col], errors="coerce")
            values.append(v)
    if len(values) != 12:
        raise ValueError(f"Expected 12 month columns, found {len(values)} in {src}")

    return pd.DataFrame({"Year": year, "Month": MONTHS, "value": values})


def recent_annual_visitors_data(variable: str = "Visitor Volume") -> pd.DataFrame | None:
    """Stack every Year-End workbook into one table."""
    frames = []
    for url in identify_available_annual_spreadsheets():
        try:
            frames.append(load_las_vegas_visitor_data_from_xlsx(variable, url))
        except Exception as exc:                        # noqa: BLE001
            print(f"Skipping {url}: {exc}")
        time.sleep(random.uniform(3, 5))                # be polite; keep this
    return pd.concat(frames, ignore_index=True) if frames else None


def recent_timepath_visitors_data(variable: str = "Visitor Volume") -> pd.Series:
    """Full monthly series: every Year-End file plus the current YTD file."""
    annual = recent_annual_visitors_data(variable)
    current = load_las_vegas_visitor_data_from_xlsx(variable)
    parts = [p for p in (current, annual) if p is not None]
    if not parts:
        raise RuntimeError("No LVCVA workbooks could be read.")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"] + "-01",
                                format="%Y-%B-%d")
    s = df.drop_duplicates("Date").set_index("Date")["value"].sort_index()
    s.name = variable.lower().replace(" ", "_")
    return s


def write_visitors_csv(path: str | Path = "data/lv_visitors.csv") -> Path:
    """Write the monthly series as date,visitors and print provenance."""
    s = recent_timepath_visitors_data("Visitor Volume")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "visitors": s.values.astype(int)})
    out.to_csv(path, index=False)
    print(f"Wrote {path}: {len(out)} months, {out['date'].iloc[0]} to {out['date'].iloc[-1]}")
    print(f"Source: {LVCVA_RESEARCH_URL}  Retrieved: {date.today().isoformat()}")
    print("Units: visitors per month (LVCVA 'Visitor Volume'). Recent months are subject to revision.")
    return path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/lv_visitors.csv"
    write_visitors_csv(target)
