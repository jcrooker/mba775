"""Minimal FRED download helpers for MBA 775.

FRED publishes every series at a public CSV endpoint that requires no API key:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF

That is a plain URL returning plain CSV, so `pandas` alone is enough to read
it. These helpers wrap that endpoint with the error handling, retries, caching,
and type conversion an analysis should not have to repeat in every script.

The keyed REST API at api.stlouisfed.org is a different service. It returns
richer metadata and requires a personal API key. Use it for the Group Forecast
Project; use this module for coursework and lecture demonstrations.

Caching
-------
Set a cache directory once and every download is saved as raw CSV and reused
on later runs:

    import fred_tools
    fred_tools.set_cache_dir("data/raw")

This makes a document render quickly and identically after the first run, and
it keeps an unmodified copy of the source data, which is what a reproducible
analysis is supposed to preserve. Pass ``refresh=True`` to force a new
download.

Requires: pandas. Nothing else.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

__all__ = [
    "fred_series",
    "fred_frame",
    "set_cache_dir",
    "get_cache_dir",
    "set_transport",
    "get_transport",
    "FredError",
]

FRED_CSV_ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# FRED has used both spellings for the date column over the years.
_DATE_HEADERS = ("date", "observation_date")

# FRED can be slow to first byte, especially for long daily series. A generous
# timeout costs nothing when the server is responsive.
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2

# Some networks and CDNs treat unfamiliar clients differently. Presenting a
# conventional browser User-Agent avoids a class of silent stalls.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Accept-Encoding": "identity",
    "Connection": "close",
}

_CACHE_DIR: Path | None = None

# Which transport actually works on this machine. Resolved on first use.
# Python's own HTTPS client is fingerprinted differently from a browser, and
# some CDNs in front of FRED accept the TLS handshake and then never answer.
# curl.exe (Windows 10+) and PowerShell use the Windows schannel stack and are
# treated like ordinary clients.
_TRANSPORT: str | None = None
_TRANSPORT_ORDER = ("urllib", "curl", "powershell")
_PROBE_TIMEOUT = 20.0


class FredError(RuntimeError):
    """Raised when FRED cannot supply the requested series."""


def set_transport(name):
    """Force a transport: 'urllib', 'curl', 'powershell', or None to auto-detect."""
    global _TRANSPORT
    if name is not None and name not in _TRANSPORT_ORDER:
        raise ValueError(f"transport must be one of {_TRANSPORT_ORDER} or None")
    _TRANSPORT = name
    return _TRANSPORT


def get_transport() -> str | None:
    """Return the transport in use, or None if not yet resolved."""
    return _TRANSPORT


def set_cache_dir(path) -> Path:
    """Store downloaded CSVs under `path` and reuse them on later runs."""
    global _CACHE_DIR
    _CACHE_DIR = Path(path)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def get_cache_dir() -> Path | None:
    """Return the active cache directory, or None if caching is off."""
    return _CACHE_DIR


def _cache_path(series_id: str, start, end) -> Path | None:
    if _CACHE_DIR is None:
        return None
    part = lambda v: "none" if v is None else str(pd.Timestamp(v).date())
    return _CACHE_DIR / f"{series_id}__{part(start)}__{part(end)}.csv"


def _build_url(series_id: str, start=None, end=None) -> str:
    params = {"id": series_id}
    if start is not None:
        params["cosd"] = str(pd.Timestamp(start).date())
    if end is not None:
        params["coed"] = str(pd.Timestamp(end).date())
    return f"{FRED_CSV_ENDPOINT}?{urlencode(params)}"


class _HttpRefused(Exception):
    """The server answered with an HTTP error status."""

    def __init__(self, code):
        super().__init__(str(code))
        self.code = code


def _via_urllib(url: str, timeout: float) -> str:
    try:
        with urlopen(Request(url, headers=_HEADERS), timeout=timeout) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        raise _HttpRefused(exc.code) from None


def _run(argv, timeout):
    exe = shutil.which(argv[0])
    if exe is None:
        raise FileNotFoundError(f"{argv[0]} is not available")
    return subprocess.run([exe, *argv[1:]], capture_output=True, timeout=timeout + 10)


def _via_curl(url: str, timeout: float) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "fred.csv"
        proc = _run(
            ["curl", "-sS", "-f", "-L", "--max-time", str(int(timeout)),
             "-A", _HEADERS["User-Agent"], "-o", str(out), url],
            timeout,
        )
        if proc.returncode == 22:
            raise _HttpRefused("4xx/5xx")
        if proc.returncode != 0:
            raise OSError(proc.stderr.decode("utf-8", "replace").strip()
                          or f"curl exit {proc.returncode}")
        return out.read_text(encoding="utf-8-sig")


def _via_powershell(url: str, timeout: float) -> str:
    exe = "powershell" if shutil.which("powershell") else "pwsh"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "fred.csv"
        proc = _run(
            [exe, "-NoProfile", "-Command",
             f"$ProgressPreference='SilentlyContinue'; "
             f"Invoke-WebRequest -Uri '{url}' -UseBasicParsing "
             f"-TimeoutSec {int(timeout)} -OutFile '{out}'"],
            timeout,
        )
        if proc.returncode != 0:
            raise OSError(proc.stderr.decode("utf-8", "replace").strip()
                          or f"powershell exit {proc.returncode}")
        return out.read_text(encoding="utf-8-sig")


_TRANSPORTS = {
    "urllib": _via_urllib,
    "curl": _via_curl,
    "powershell": _via_powershell,
}


def _attempt(name: str, url: str, timeout: float) -> str:
    body = _TRANSPORTS[name](url, timeout)
    if not body.strip():
        raise OSError("empty response")
    return body


def _fetch(url: str, timeout: float, retries: int = DEFAULT_RETRIES) -> str:
    """Fetch `url`, choosing a transport that works on this machine."""
    global _TRANSPORT
    problems = []

    if _TRANSPORT is None:
        # Probe each transport once with a short timeout, then remember the winner.
        for name in _TRANSPORT_ORDER:
            try:
                body = _attempt(name, url, min(timeout, _PROBE_TIMEOUT))
            except _HttpRefused as exc:
                raise FredError(
                    f"FRED refused the request (HTTP {exc.code}). "
                    f"The series ID is probably misspelled or retired."
                ) from None
            except FileNotFoundError as exc:
                problems.append(f"{name}: {exc}")
            except TimeoutError:
                problems.append(f"{name}: timed out")
            except subprocess.TimeoutExpired:
                problems.append(f"{name}: timed out")
            except (URLError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                problems.append(f"{name}: {reason}")
            else:
                _TRANSPORT = name
                return body

        raise FredError(
            "Could not retrieve data from FRED with any available transport.\n"
            f"    URL: {url}\n"
            + "".join(f"    tried {p}\n" for p in problems)
            + "    If this URL loads in your browser, try again off the "
              "university network, or call set_cache_dir() and place a "
              "manually downloaded CSV there."
        )

    last = ""
    for attempt in range(retries + 1):
        try:
            return _attempt(_TRANSPORT, url, timeout)
        except _HttpRefused as exc:
            raise FredError(
                f"FRED refused the request (HTTP {exc.code}). "
                f"The series ID is probably misspelled or retired."
            ) from None
        except TimeoutError:
            last = f"timed out after {timeout:.0f}s"
        except subprocess.TimeoutExpired:
            last = f"timed out after {timeout:.0f}s"
        except (URLError, OSError) as exc:
            last = str(getattr(exc, "reason", exc))
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))

    raise FredError(
        f"Could not retrieve data from FRED via {_TRANSPORT} ({last}) after "
        f"{retries + 1} attempts.\n    URL: {url}"
    )


def fred_series(
    series_id,
    start=None,
    end=None,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
    refresh=False,
) -> pd.Series:
    """Download one FRED series as a pandas Series indexed by date.

    Parameters
    ----------
    series_id : str
        A FRED series identifier, for example ``"DFF"`` or ``"NVUR"``.
    start, end : optional
        Anything ``pd.Timestamp`` accepts. Omit for the full published history.
    timeout : float
        Seconds to wait for each attempt.
    retries : int
        Additional attempts after a timeout or connection problem.
    refresh : bool
        Ignore any cached copy and download again.

    Returns
    -------
    pandas.Series
        Float values named after the series ID, indexed by a DatetimeIndex
        named ``date``. FRED's ``.`` missing marker becomes ``NaN``; missing
        observations are preserved, never dropped.

    Raises
    ------
    FredError
        If FRED is unreachable, refuses the request, or returns something that
        is not the expected CSV.

    Notes
    -----
    ``start`` is a *request*, not a guarantee. Some series are distributed with
    a limited window (``SP500`` carries roughly ten years) and FRED will
    silently return less history than you asked for. Always check the first
    index value against what you expected.
    """
    series_id = str(series_id).strip().upper()
    if not series_id:
        raise ValueError("series_id must be a non-empty string.")

    cache_file = _cache_path(series_id, start, end)

    payload = None
    if cache_file is not None and cache_file.exists() and not refresh:
        payload = cache_file.read_text(encoding="utf-8-sig")

    if payload is None:
        payload = _fetch(_build_url(series_id, start, end), timeout, retries)
        if cache_file is not None:
            cache_file.write_text(payload, encoding="utf-8")

    first_line = payload.lstrip().split("\n", 1)[0].strip().lower()
    if not first_line.startswith(_DATE_HEADERS):
        raise FredError(
            f"FRED did not return observations for {series_id!r}. "
            f"It replied with: {first_line[:80]!r}"
        )

    frame = pd.read_csv(StringIO(payload), na_values=["."], keep_default_na=True)
    if frame.shape[1] < 2:
        raise FredError(f"FRED returned no value column for {series_id!r}.")

    date_col, value_col = frame.columns[0], frame.columns[1]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    series = frame.set_index(date_col)[value_col].astype("float64").sort_index()
    series.index.name = "date"
    series.name = series_id
    return series


def fred_frame(series_ids, start=None, end=None, verbose=True, **kwargs):
    """Download several FRED series and report what succeeded.

    Unlike a single bulk request, one bad identifier does not sink the whole
    download. Each series is fetched independently and failures are reported
    rather than raised. Extra keyword arguments are passed to `fred_series`.

    Returns
    -------
    (frame, status) : tuple
        ``frame`` is a DataFrame with one column per series that downloaded,
        outer-joined on date. Series published at different frequencies will
        therefore produce many ``NaN`` cells -- that is expected, and means the
        frequencies still need to be reconciled before modeling.

        ``status`` is a DataFrame indexed by series ID recording, for each
        request, whether it succeeded, how many non-missing observations came
        back, and the first and last dates covered.
    """
    series_ids = [str(s).strip().upper() for s in series_ids]

    collected = {}
    status_rows = []

    for series_id in series_ids:
        try:
            values = fred_series(series_id, start=start, end=end, **kwargs)
        except FredError as exc:
            status_rows.append({
                "series_id": series_id,
                "result": "FAILED",
                "n_obs": 0,
                "first": pd.NaT,
                "last": pd.NaT,
                "detail": str(exc).split("\n")[0],
            })
            if verbose:
                print(f"  {series_id}: FAILED - {str(exc).splitlines()[0]}")
            continue

        collected[series_id] = values
        status_rows.append({
            "series_id": series_id,
            "result": "ok",
            "n_obs": int(values.notna().sum()),
            "first": values.first_valid_index(),
            "last": values.last_valid_index(),
            "detail": "",
        })

    frame = pd.DataFrame(collected)
    status = pd.DataFrame(status_rows).set_index("series_id")

    if verbose:
        n_ok = int((status["result"] == "ok").sum())
        print(f"Retrieved {n_ok} of {len(series_ids)} series.")

    return frame, status
