# seed_chapter03_data.R ---------------------------------------------------
#
# Build the student-facing data files for Chapter 3 (Calculating Descriptive
# Statistics).
#
# Produces, in data/:
#   state_population.csv     50 states, latest population (mean vs median)
#   unemployment_rate.csv    UNRATE monthly, 1948 to today (symmetric-ish)
#   cpi_inflation.csv        CPIAUCSL with 12-month log inflation (z-scores)
#   patents_by_origin.csv    patents granted by country of origin (the mode)
#   grad_student_gpa.csv     graduate GPA (left skew) - from Skip's site
#   annual_returns.csv       S&P 500 and Walmart annual returns (range, CV)
#
# Run from the repository root:
#     Rscript tools/seed_chapter03_data.R
#
# Requires the quantmod package for the stock series. Everything else is a
# plain download.

force     <- FALSE
cache_dir <- file.path("data", "raw")
out_dir   <- "data"
pause     <- 0.4

dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir,   recursive = TRUE, showWarnings = FALSE)

# --- shared helpers (same as chapter 2) ----------------------------------

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

# --- 1. State populations -------------------------------------------------
#
# Reuse the panel built for Chapter 2 rather than re-downloading 50 series.

cat("1. State populations\n")
panel_file <- file.path(out_dir, "state_hpi_ur_pop.csv")
if (file.exists(panel_file)) {
  panel <- read.csv(panel_file, stringsAsFactors = FALSE)
  pops <- data.frame(Member = panel$Member,
                     Population = panel$POPN,
                     as_of = panel$POP_date,
                     stringsAsFactors = FALSE)
  write.csv(pops, file.path(out_dir, "state_population.csv"), row.names = FALSE)
  west <- c("WA","ID","MT","WY","OR","NV","CA","UT","CO","NM","AZ")
  ne   <- c("CT","ME","MA","NH","RI","VT")
  w <- pops$Population[pops$Member %in% west]
  n <- pops$Population[pops$Member %in% ne]
  cat(sprintf("   %d states written (as of %s)\n", nrow(pops), pops$as_of[1]))
  cat(sprintf("   West (n=%d):        mean %s   median %s\n", length(w),
              format(round(mean(w)), big.mark = ","), format(round(median(w)), big.mark = ",")))
  cat(sprintf("   New England (n=%d): mean %s   median %s\n", length(n),
              format(round(mean(n)), big.mark = ","), format(round(median(n)), big.mark = ",")))
} else {
  cat("   SKIPPED: run tools/seed_chapter02_data.R first\n")
}

# --- 2. Unemployment rate -------------------------------------------------

cat("2. Unemployment rate (UNRATE)\n")
p <- fred_csv("UNRATE")
if (!is.na(p)) {
  d <- read_series(p); names(d) <- c("date", "unemployment_rate")
  write.csv(d, file.path(out_dir, "unemployment_rate.csv"), row.names = FALSE)
  cat(sprintf("   %s rows, %s to %s | mean %.2f, median %.2f\n",
              format(nrow(d), big.mark = ","), min(d$date), max(d$date),
              mean(d$unemployment_rate), median(d$unemployment_rate)))
} else cat("   FAILED\n")

# --- 3. Inflation ---------------------------------------------------------

cat("3. Consumer price index (CPIAUCSL)\n")
p <- fred_csv("CPIAUCSL", "1970-01-01")
if (!is.na(p)) {
  d <- read_series(p); names(d) <- c("date", "cpi")
  # Twelve-month log change, the definition used in the lecture note.
  d$inflation <- c(rep(NA_real_, 12), diff(log(d$cpi), lag = 12))
  write.csv(d, file.path(out_dir, "cpi_inflation.csv"), row.names = FALSE)
  ok <- d[!is.na(d$inflation), ]
  cat(sprintf("   %s rows, %s to %s | inflation mean %.4f, sd %.4f\n",
              format(nrow(d), big.mark = ","), min(d$date), max(d$date),
              mean(ok$inflation), sd(ok$inflation)))
  pk <- ok[which.max(ok$inflation), ]
  cat(sprintf("   Highest 12-month inflation: %.2f%% in %s\n",
              100 * pk$inflation, format(pk$date, "%B %Y")))
} else cat("   FAILED\n")

# --- 4. Patents by country of origin -------------------------------------

cat("4. Patents granted by country of origin\n")
patent_ids <- c(US = "PATENTUSALLTOTAL", China = "PATENT4NCNTOTAL",
                Japan = "PATENT4NJPTOTAL", Germany = "PATENT4NDETOTAL",
                UK = "PATENT4NGBTOTAL")
frames <- list()
for (nm in names(patent_ids)) {
  p <- fred_csv(patent_ids[[nm]])
  if (is.na(p)) { cat(sprintf("   %-8s %-20s FAILED\n", nm, patent_ids[[nm]])); next }
  d <- read_series(p); names(d) <- c("date", nm)
  frames[[nm]] <- d
}
if (length(frames)) {
  pat <- Reduce(function(a, b) merge(a, b, by = "date", all = TRUE), frames)
  write.csv(pat, file.path(out_dir, "patents_by_origin.csv"), row.names = FALSE)
  latest <- pat[nrow(pat), -1, drop = FALSE]
  cat(sprintf("   %d rows, %s to %s | %d countries\n",
              nrow(pat), min(pat$date), max(pat$date), ncol(pat) - 1))
  cat(sprintf("   Most recent year (%s) leader: %s\n",
              format(max(pat$date), "%Y"),
              names(latest)[which.max(as.numeric(latest))]))
} else cat("   FAILED for all patent series\n")

# --- 5. Graduate student GPA ---------------------------------------------

cat("5. Graduate student GPA\n")
gpa_dest <- file.path(out_dir, "grad_student_gpa.csv")
if (force || !file.exists(gpa_dest)) {
  ok <- tryCatch({
    suppressWarnings(download.file(
      "https://crooker.faculty.unlv.edu/mba775/data/grad_student_gpa.csv",
      gpa_dest, mode = "wb", quiet = TRUE)); TRUE
  }, error = function(e) FALSE)
} else ok <- TRUE
if (ok && file.exists(gpa_dest)) {
  g <- read.csv(gpa_dest, stringsAsFactors = FALSE)
  cat(sprintf("   %s rows | mean %.3f, median %.3f (%s skew)\n",
              format(nrow(g), big.mark = ","), mean(g$GPA), median(g$GPA),
              if (mean(g$GPA) < median(g$GPA)) "left" else "right"))
} else cat("   FAILED to download\n")

# --- 6. Annual stock returns ---------------------------------------------

cat("6. Annual returns, S&P 500 and Walmart\n")
if (!requireNamespace("quantmod", quietly = TRUE)) {
  cat("   SKIPPED: install.packages('quantmod')\n")
} else {
  suppressPackageStartupMessages(library(quantmod))
  from_date <- seq(Sys.Date(), length = 51, by = "-1 years")[51]

  first_close_by_year <- function(sym, label) {
    x <- tryCatch(getSymbols(sym, src = "yahoo", auto.assign = FALSE,
                             from = from_date, to = Sys.Date()),
                  error = function(e) NULL)
    if (is.null(x)) { cat(sprintf("   %-8s FAILED\n", label)); return(NULL) }
    adj <- x[, grep("Adjusted", colnames(x))[1]]
    yr  <- as.integer(format(index(adj), "%Y"))
    first <- tapply(as.numeric(adj), yr, function(v) v[1])
    data.frame(year = as.integer(names(first)),
               price = as.numeric(first), stringsAsFactors = FALSE)
  }

  sp  <- first_close_by_year("^GSPC", "S&P 500")
  wmt <- first_close_by_year("WMT", "Walmart")

  if (!is.null(sp) && !is.null(wmt)) {
    ret <- function(d, nm) {
      out <- data.frame(year = d$year[-1],
                        r = d$price[-1] / d$price[-nrow(d)] - 1)
      names(out)[2] <- nm
      out
    }
    returns <- merge(ret(sp, "sp500"), ret(wmt, "wmt"), by = "year", all = TRUE)
    write.csv(returns, file.path(out_dir, "annual_returns.csv"), row.names = FALSE)
    cat(sprintf("   %d years, %d to %d\n", nrow(returns),
                min(returns$year), max(returns$year)))
    for (nm in c("sp500", "wmt")) {
      v <- returns[[nm]][!is.na(returns[[nm]])]
      cat(sprintf("   %-6s mean %+.4f  sd %.4f  range %.4f  CV %.1f\n",
                  nm, mean(v), sd(v), max(v) - min(v), 100 * sd(v) / mean(v)))
    }
  }
}

# --- 7. Nevada AGI (supplied by the instructor) ---------------------------

cat("7. Nevada adjusted gross income\n")
agi_dest <- file.path(out_dir, "nevada_agi_2016.csv")
if (!file.exists(agi_dest)) {
  # Accept the name the file carries on the instructor's drive, so a copy-in
  # does not have to be renamed by hand.
  alias <- file.path(out_dir, "agi-in-tenthousands-nv-2016.csv")
  if (file.exists(alias)) {
    cat("   Using agi-in-tenthousands-nv-2016.csv (same file, other name).\n")
    agi_dest <- alias
  }
}
agi_samp <- file.path(out_dir, "nevada_agi_2016_sample.csv")
if (file.exists(agi_dest)) {
  a <- read.csv(agi_dest, stringsAsFactors = FALSE)
  v <- a[[ncol(a)]]
  cat(sprintf("   %s rows | mean %.2f, median %.2f (right skew)\n",
              format(nrow(a), big.mark = ","), mean(v), median(v)))

  # The full file is ~28 MB. That is too large to put in a repository that
  # students fetch one file at a time through a chat window, so the committed
  # version is a fixed-seed random sample. The seed is fixed so that every
  # student and this note compute identical numbers.
  if (force || !file.exists(agi_samp)) {
    set.seed(775)
    n_keep <- min(100000L, nrow(a))
    keep <- sort(sample.int(nrow(a), n_keep))
    samp <- data.frame(Obs = seq_len(n_keep), AGI = v[keep])
    write.csv(samp, agi_samp, row.names = FALSE)
  }
  s <- read.csv(agi_samp, stringsAsFactors = FALSE)
  cat(sprintf("   sample: %s rows | mean %.2f, median %.2f  (full file: %.2f / %.2f)\n",
              format(nrow(s), big.mark = ","), mean(s$AGI), median(s$AGI),
              mean(v), median(v)))
  cat("   Commit nevada_agi_2016_sample.csv. The full file stays local.\n")
} else {
  cat("   NOT FOUND. This file is not on a public URL.\n")
  cat("   Copy it in and re-run:\n")
  cat("     G:/My Drive/mba775/data/agi-in-tenthousands-nv-2016.csv\n")
  cat("        ->  data/nevada_agi_2016.csv\n")
  cat("   The right-skew example will be skipped until it is present.\n")
}

# --- summary --------------------------------------------------------------

cat("\n----------------------------------------------------------------\n")
for (f in c("state_population.csv", "unemployment_rate.csv", "cpi_inflation.csv",
            "patents_by_origin.csv", "grad_student_gpa.csv",
            "annual_returns.csv", "nevada_agi_2016_sample.csv")) {
  p <- file.path(out_dir, f)
  cat(sprintf("  %-26s %s\n", f,
              if (file.exists(p)) sprintf("%8s bytes", format(file.size(p), big.mark = ","))
              else "   missing"))
}
cat("\nCommit these so students can download them.\n")
