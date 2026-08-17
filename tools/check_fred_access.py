"""Diagnose why Python cannot reach FRED when a browser can.

Run it with the same interpreter Quarto uses:

    "%LOCALAPPDATA%\\venvs\\mba775\\Scripts\\python.exe" check_fred_access.py

Each stage is timed and reported separately, so the output tells you which
layer is failing: name resolution, TCP, TLS, or the HTTP response itself.
"""

import os
import socket
import ssl
import sys
import time
import urllib.request
from urllib.error import HTTPError, URLError

HOST = "fred.stlouisfed.org"
URL = f"https://{HOST}/graph/fredgraph.csv?id=DFF&cosd=2020-01-01"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print(f"Python     : {sys.version.split()[0]}")
    print(f"Executable : {sys.executable}")

    section("1. Proxy configuration")
    env_proxies = {
        k: v for k, v in os.environ.items()
        if k.lower() in {"http_proxy", "https_proxy", "no_proxy", "all_proxy"}
    }
    print(f"Proxy environment variables : {env_proxies or 'none set'}")
    try:
        detected = urllib.request.getproxies()
    except Exception as exc:  # pragma: no cover - platform dependent
        detected = f"lookup failed: {exc}"
    print(f"Proxies Python detected     : {detected or 'none'}")
    print("\nIf your browser reaches FRED through a proxy or PAC/auto-config")
    print("script and the line above says 'none', that is very likely the")
    print("cause: Python does not evaluate PAC files.")

    section("2. DNS resolution")
    t0 = time.perf_counter()
    try:
        infos = socket.getaddrinfo(HOST, 443, proto=socket.IPPROTO_TCP)
        addrs = sorted({i[4][0] for i in infos})
        print(f"OK  ({time.perf_counter() - t0:.2f}s)  {HOST} -> {addrs}")
    except OSError as exc:
        print(f"FAILED ({time.perf_counter() - t0:.2f}s): {exc}")
        return

    section("3. TCP connect on port 443")
    t0 = time.perf_counter()
    try:
        with socket.create_connection((HOST, 443), timeout=15) as sock:
            print(f"OK  ({time.perf_counter() - t0:.2f}s)  peer={sock.getpeername()[0]}")
    except OSError as exc:
        print(f"FAILED ({time.perf_counter() - t0:.2f}s): {exc}")
        return

    section("4. TLS handshake")
    t0 = time.perf_counter()
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((HOST, 443), timeout=15) as raw:
            with ctx.wrap_socket(raw, server_hostname=HOST) as tls:
                cert = tls.getpeercert()
                issuer = dict(x[0] for x in cert.get("issuer", ()))
                print(f"OK  ({time.perf_counter() - t0:.2f}s)  {tls.version()}")
                print(f"Certificate issuer : {issuer.get('organizationName', '?')}")
                print("\nIf that issuer is your university or a security")
                print("appliance rather than a public CA, TLS inspection is in")
                print("play and may be what is stalling the response.")
    except Exception as exc:
        print(f"FAILED ({time.perf_counter() - t0:.2f}s): {type(exc).__name__}: {exc}")
        return

    section("5. HTTP GET with different User-Agent values")
    for label, ua in [("browser-style", BROWSER_UA), ("custom", "MBA775-coursework")]:
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(URL, headers={
                "User-Agent": ua,
                "Accept": "text/csv,text/plain,*/*",
                "Accept-Encoding": "identity",
                "Connection": "close",
            })
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = resp.read().decode("utf-8")
            first = body.splitlines()[0] if body else "(empty)"
            print(f"{label:<14} OK  ({time.perf_counter() - t0:5.2f}s)  "
                  f"{len(body):,} bytes  header={first!r}")
        except HTTPError as exc:
            print(f"{label:<14} HTTP {exc.code} ({time.perf_counter() - t0:5.2f}s)")
        except TimeoutError:
            print(f"{label:<14} TIMED OUT ({time.perf_counter() - t0:5.2f}s)")
        except URLError as exc:
            print(f"{label:<14} FAILED ({time.perf_counter() - t0:5.2f}s): {exc.reason}")
        except Exception as exc:
            print(f"{label:<14} FAILED ({time.perf_counter() - t0:5.2f}s): "
                  f"{type(exc).__name__}: {exc}")

    section("Interpretation")
    print("Stages 2-4 pass but stage 5 times out : the connection is being")
    print("    opened and then stalled -- proxy, VPN, or TLS inspection.")
    print("browser-style works, custom does not  : the User-Agent was the")
    print("    problem; fred_tools already sends the browser-style header.")
    print("Both succeed                          : the earlier failure was")
    print("    transient; the added retries should absorb it.")


if __name__ == "__main__":
    main()
