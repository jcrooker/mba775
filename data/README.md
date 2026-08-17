# Data

Every file here comes from the Federal Reserve Economic Data (FRED) service of
the Federal Reserve Bank of St. Louis, downloaded unmodified and reshaped only
into a tidy layout. No values have been edited, interpolated, or filled.

Rebuild these files with `tools/seed_fred_cache.R` followed by
`python tools/make_student_data.py`.

## dff.csv

| | |
|---|---|
| Series | `DFF` — Federal Funds Effective Rate |
| Source | <https://fred.stlouisfed.org/series/DFF> |
| Units | Percent, not seasonally adjusted |
| Frequency | Daily, 7-day (a value on **every calendar day**, weekends and holidays included) |
| Date range | 1954-07-01 to 2026-08-13 |
| Rows | 26,342 |
| Missing values | 0 |
| Retrieved | 2026-08-17 |

Columns: `date`, `federal_funds_rate`.

The 7-day calendar matters. Selecting an arbitrary date from this series will
find a value; doing the same on a business-day series such as `DGS10` or
`SP500` will not.

## state_unemployment.csv

| | |
|---|---|
| Series | `<ST>UR` — unemployment rate for each of the 50 states (e.g. `NVUR` for Nevada) |
| Source | <https://fred.stlouisfed.org/> (search the series ID) |
| Units | Percent, seasonally adjusted |
| Frequency | Monthly |
| Date range | 1976-01-01 to 2026-06-01 |
| Rows | 30,300 (50 states x 606 months) |
| Missing values | 50 |
| Retrieved | 2026-08-17 |

Columns: `date`, `state`, `unemployment_rate` — one row per state per month
(a "long" or "tidy" layout).

**Note the gap.** Every state is missing a value for 2025-10. That is
how FRED distributes it; nothing has been dropped here. Missing months in
official statistics usually have a reason, and finding out what it was is a
better exercise than filling the hole.

## A caution about "latest"

These files are a snapshot. FRED revises published data, and the most recent
month of any series is preliminary and will change. Re-download before making
a claim that depends on the newest observation, and record the retrieval date
alongside any result.
