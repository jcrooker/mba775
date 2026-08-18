# Data

Most files here come from the Federal Reserve Economic Data (FRED) service of
the Federal Reserve Bank of St. Louis, downloaded unmodified and reshaped only
into a tidy layout. No values have been edited, interpolated, or filled. Where
a file comes from somewhere else, or is simulated rather than observed, the
entry below says so — read the entry before you cite the file.

Rebuild these files with the seeding scripts in `tools/`:

| Chapter | Script |
|---|---|
| 1 | `tools/seed_fred_cache.R`, then `python tools/make_student_data.py` |
| 2 | `Rscript tools/seed_chapter02_data.R` |
| 3 | `Rscript tools/seed_chapter03_data.R` |

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


# Chapter 3 files

## state_population.csv

| | |
|---|---|
| Series | `<ST>POP` — resident population for each of the 50 states |
| Source | <https://fred.stlouisfed.org/> (search the series ID) |
| Units | **Thousands of people.** Multiply by 1,000 before quoting a headcount |
| Frequency | Annual |
| Rows | 50, one per state |

Columns: `Member`, `Population`, `as_of`. Derived from the Chapter 2 panel
(`state_hpi_ur_pop.csv`) rather than re-downloaded.

## unemployment_rate.csv

| | |
|---|---|
| Series | `UNRATE` — civilian unemployment rate |
| Source | <https://fred.stlouisfed.org/series/UNRATE> |
| Units | Percent, seasonally adjusted, **rounded to one decimal place** |
| Frequency | Monthly, from 1948-01 |

Columns: `date`, `unemployment_rate`. The rounding is why this series has a
mode at all; ties here are an artifact of publication, not of the economy.

## cpi_inflation.csv

| | |
|---|---|
| Series | `CPIAUCSL` — Consumer Price Index for All Urban Consumers |
| Source | <https://fred.stlouisfed.org/series/CPIAUCSL> |
| Units | Index 1982-1984 = 100, seasonally adjusted |
| Frequency | Monthly, from 1970-01 |

Columns: `date`, `cpi`, `inflation`. The `inflation` column is **computed
here**, not downloaded: it is the 12-month log change in `cpi`, so the first
twelve rows are blank by construction. A log change and a percent change are
close at low rates and diverge as rates rise.

## patents_by_origin.csv

| | |
|---|---|
| Series | `PATENTUSALLTOTAL`, `PATENT4NCNTOTAL`, `PATENT4NJPTOTAL`, `PATENT4NDETOTAL`, `PATENT4NGBTOTAL` |
| Source | <https://fred.stlouisfed.org/> (search the series ID) |
| Units | Count of patents granted |
| Frequency | Annual |

Columns: `date`, `US`, `China`, `Japan`, `Germany`, `UK`.

**This series is discontinued.** It stops in 2020 and nothing on the FRED page
announces that. Any statement built on it must say "through 2020."

## grad_student_gpa.csv

| | |
|---|---|
| Source | <https://crooker.faculty.unlv.edu/mba775/data/grad_student_gpa.csv> |
| Units | Grade point average, 0.0 to 4.0 |

Columns include `GPA`. Anonymised graduate student records. The hard ceiling
at 4.0 is what makes this distribution left-skewed.

## annual_returns.csv

| | |
|---|---|
| Symbols | `^GSPC` (S&P 500 index) and `WMT` (Walmart) |
| Source | Yahoo Finance, via the R `quantmod` package |
| Units | Annual return as a decimal (0.10 = 10%) |
| Frequency | Annual, the last 50 years |

Columns: `year`, `sp500`, `wmt`. Each return is computed from the first
adjusted close of one year against the first adjusted close of the next.
Adjusted closes account for dividends and splits, so these are total returns,
not price changes. Yahoo revises its history; re-seed before quoting a figure
that depends on the newest year.

## nevada_agi_2016_sample.csv

| | |
|---|---|
| Source | Simulated from IRS Statistics of Income zip-code counts, tax year 2016 |
| IRS portal | <https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi> |
| Units | Adjusted gross income in **tens of thousands of dollars** |
| Rows | 100,000 |

Columns: `Obs`, `AGI`.

**Two things to state whenever you use this file.**

First, the observations are **simulated**. The IRS publishes counts of returns
by zip code and income bracket, never household-level records. Individual
rows are draws consistent with those published counts; the distribution's
shape is real, no single row is.

Second, this is a **fixed-seed random sample** of 100,000 drawn from roughly
1.3 million simulated households. The full file is about 28 MB, too large to
distribute here. The seed is fixed, so every student computes identical
numbers from it.
