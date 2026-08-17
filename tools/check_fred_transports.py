"""Find a way to fetch FRED that works on this machine.

DNS, TCP, and TLS all succeed but the HTTP response never arrives. That
pattern points at the server (or something in front of it) deciding not to
answer this particular client. Different tools present different TLS
fingerprints and header sets, so one of them may well succeed where Python's
urllib does not.

This tries four transports and reports which work:

    1. urllib, minimal headers
    2. urllib, full browser-style headers
    3. curl.exe        (ships with Windows 10+, uses schannel)
    4. PowerShell      (uses .NET / schannel)

Run it with the same interpreter Quarto uses:

    "%LOCALAPPDATA%\\venvs\\mba775\\Scripts\\python.exe" check_fred_transports.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

URL = os.environ.get(
    "FRED_TEST_URL",
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF&cosd=2020-01-01",
)
TIMEOUT = 30

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

results = []


def report(name, ok, seconds, detail):
    status = "OK     " if ok else "FAILED "
    print(f"  {status} {name:<28} {seconds:6.2f}s  {detail}")
    results.append((name, ok))


def summarize(body: str) -> str:
    if not body:
        return "empty response"
    head = body.splitlines()[0][:60]
    return f"{len(body):,} bytes, header={head!r}"


def try_urllib(name, headers):
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(URL, headers=headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8", "replace")
        report(name, True, time.perf_counter() - t0, summarize(body))
    except Exception as exc:
        report(name, False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")


def try_subprocess(name, argv, outfile=None):
    exe = shutil.which(argv[0])
    if exe is None:
        report(name, False, 0.0, f"{argv[0]} not found on PATH")
        return
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=TIMEOUT + 15)
        elapsed = time.perf_counter() - t0
        if outfile is not None:
            body = ""
            if os.path.exists(outfile):
                with open(outfile, "r", encoding="utf-8", errors="replace") as fh:
                    body = fh.read()
                os.unlink(outfile)
        else:
            body = proc.stdout.decode("utf-8", "replace")

        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            report(name, False, elapsed,
                   f"exit {proc.returncode}: {err[0] if err else '(no message)'}")
        elif not body.strip():
            report(name, False, elapsed, "exit 0 but empty response")
        else:
            report(name, True, elapsed, summarize(body))
    except subprocess.TimeoutExpired:
        report(name, False, time.perf_counter() - t0, "timed out")
    except Exception as exc:
        report(name, False, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")


def main():
    print(f"Python : {sys.version.split()[0]}")
    print(f"URL    : {URL}\n")
    print("Transport tests")
    print("-" * 78)

    try_urllib("urllib (minimal headers)", {"User-Agent": "Python-urllib/3"})
    try_urllib("urllib (browser headers)", BROWSER_HEADERS)

    tmp = os.path.join(tempfile.gettempdir(), "fred_curl_test.csv")
    try_subprocess(
        "curl",
        ["curl", "-sSL", "--max-time", str(TIMEOUT), "-A", BROWSER_HEADERS["User-Agent"],
         "-o", tmp, URL],
        outfile=tmp,
    )

    ps = shutil.which("powershell") or shutil.which("pwsh")
    if ps:
        try_subprocess(
            "PowerShell Invoke-WebRequest",
            [os.path.basename(ps), "-NoProfile", "-Command",
             f"(Invoke-WebRequest -Uri '{URL}' -UseBasicParsing "
             f"-TimeoutSec {TIMEOUT}).Content"],
        )
    else:
        report("PowerShell Invoke-WebRequest", False, 0.0, "powershell not found")

    print("-" * 78)
    working = [n for n, ok in results if ok]
    if working:
        print(f"\nWORKING TRANSPORT(S): {', '.join(working)}")
        print("fred_tools can be pointed at whichever of these succeeds.")
    else:
        print("\nNo transport succeeded. The block is not client-specific --")
        print("something on the network is dropping requests to this host.")
        print("Try again off the university network (phone hotspot) to confirm.")


if __name__ == "__main__":
    main()
