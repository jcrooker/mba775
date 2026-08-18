# seed_chapter04_data.R ---------------------------------------------------
#
# Build the student-facing data file for Chapter 4 (Introducing
# Probabilities).
#
# Produces, in data/:
#   nevada_economy.csv   Nevada real GDP and unemployment, quarterly, with
#                        the four-quarter log growth rate in real GDP
#
# Run from the repository root:
#     Rscript tools/seed_chapter04_data.R
#   or, from the R console:
#     source("tools/seed_chapter04_data.R")
#
# Two series, both from FRED:
#   NVRQGSP  Real GDP, all industry total, Nevada. Quarterly, SAAR,
#            millions of chained dollars. BEA's state quarterly series
#            begins in 2005, so this is where the sample starts.
#   NVUR     Unemployment rate, Nevada. Monthly, seasonally adjusted.
#
# The unemployment series is monthly and the GDP series is quarterly. We
# keep the month that opens each quarter rather than averaging, because the
# note classifies quarters into economic states and an averaged rate would
# not correspond to any month anyone can look up.

force     <- FALSE
cache_dir <- file.path("data", "raw")
out_dir   <- "data"
pause     <- 0.4

dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir,   recursive = TRUE, showWarnings = FALSE)

# --- shared helpers (same as chapters 2 and 3) ---------------------------

fred_csv <- function(id, start = NA_character_) {
  dest <- file.path(cache_dir, sprintf("%s__%s__none.csv", id,
                                       ifelse(is.na(start), "none", start)))
  if (!force && file.exists(dest) && file.size(dest) > 20) return(dest)

  url <- sprintf("https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s", id)
  if (!is.na(start)) url <- paste0(url, "&cosd=", start)

  ok <- tryCatch({
    suppressWarnings(download.file(url, dest, mode = "wb", quiet = TRUE)); TRUE
  }, error = function(e) FALSE)
  if (!ok) { if (file.exists(dest)) unlink(dest); return(NA_character_) }

  first <- tryCatch(readLines(dest, n = 1, warn = FALSE), error = function(e) "")
  if (!length(first) || !grepl("^(observation_date|DATE)\\s*,", first, ignore.case = TRUE)) {
    unlink(dest); return(NA_character_)
  }
  Sys.sleep(pause)
  dest
}

read_series <- function(path) {
  d <- read.csv(path, stringsAsFactors = FALSE, na.strings = c(".", "NA", ""))
  names(d) <- c("date", "value")
  d$date  <- as.Date(d$date)
  d$value <- suppressWarnings(as.numeric(d$value))
  d[!is.na(d$value), ]
}

# --- Nevada real GDP ------------------------------------------------------

cat("1. Nevada real GDP (NVRQGSP)\n")
p_gdp <- fred_csv("NVRQGSP")
if (is.na(p_gdp)) {
  cat("   FAILED. Nothing further can be built; stopping.\n")
} else {
  gdp <- read_series(p_gdp)
  names(gdp) <- c("date", "real_gdp")
  cat(sprintf("   %s quarters, %s to %s\n",
              format(nrow(gdp), big.mark = ","), min(gdp$date), max(gdp$date)))

  # --- Nevada unemployment ------------------------------------------------

  cat("2. Nevada unemployment rate (NVUR)\n")
  p_ur <- fred_csv("NVUR")
  if (is.na(p_ur)) {
    cat("   FAILED. Nothing further can be built; stopping.\n")
  } else {
    ur <- read_series(p_ur)
    names(ur) <- c("date", "unemployment_rate")
    cat(sprintf("   %s months, %s to %s\n",
                format(nrow(ur), big.mark = ","), min(ur$date), max(ur$date)))

    # --- merge and derive -------------------------------------------------

    cat("3. Merging and computing four-quarter growth\n")
    d <- merge(gdp, ur, by = "date")           # inner join keeps quarter starts

    # Four-quarter log change, the definition used in the lecture note. The
    # first four quarters are blank by construction, not by omission.
    d$gdp_growth <- c(rep(NA_real_, 4),
                      diff(log(d$real_gdp), lag = 4))

    write.csv(d, file.path(out_dir, "nevada_economy.csv"), row.names = FALSE)

    ok <- d[!is.na(d$gdp_growth), ]
    cat(sprintf("   %s rows written, %s to %s\n",
                format(nrow(d), big.mark = ","), min(d$date), max(d$date)))
    cat(sprintf("   %s usable quarters (four-quarter growth defined)\n",
                format(nrow(ok), big.mark = ",")))
    cat(sprintf("   growth      mean %+.4f  sd %.4f  min %+.4f  max %+.4f\n",
                mean(ok$gdp_growth), sd(ok$gdp_growth),
                min(ok$gdp_growth), max(ok$gdp_growth)))
    cat(sprintf("   unemployment mean %.2f  sd %.2f  min %.1f  max %.1f\n",
                mean(ok$unemployment_rate), sd(ok$unemployment_rate),
                min(ok$unemployment_rate), max(ok$unemployment_rate)))

    worst <- ok[which.min(ok$gdp_growth), ]
    peak  <- ok[which.max(ok$unemployment_rate), ]
    as_quarter <- function(d) sprintf("%s Q%d", format(d, "%Y"),
                                      (as.integer(format(d, "%m")) - 1) %/% 3 + 1)
    cat(sprintf("   worst growth quarter: %s (%+.2f%%)\n",
                as_quarter(worst$date), 100 * worst$gdp_growth))
    cat(sprintf("   highest unemployment: %s (%.1f%%)\n",
                format(peak$date, "%B %Y"), peak$unemployment_rate))

    # The note classifies quarters into Low / Middle / High by quartiles and
    # then pairs each quarter with the state two quarters ahead. Report the
    # count that survives that pairing, because that is the denominator every
    # probability in the note is built on.
    n_pairs <- nrow(ok) - 2
    cat(sprintf("   quarters with a six-month-forward state: %s\n",
                format(n_pairs, big.mark = ",")))
  }
}

# --- summary --------------------------------------------------------------

cat("\n----------------------------------------------------------------\n")
f <- file.path(out_dir, "nevada_economy.csv")
cat(sprintf("  %-26s %s\n", "nevada_economy.csv",
            if (file.exists(f)) sprintf("%8s bytes", format(file.size(f), big.mark = ","))
            else "   missing"))
cat("\nCommit this so students can download it.\n")
