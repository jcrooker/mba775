# seed_chapter02_data.R ---------------------------------------------------
#
# Build the student-facing data files for Chapter 2 (Displaying Descriptive
# Statistics), using R's downloader.
#
# Why R: on this machine Python's HTTPS clients are refused by the CDN in
# front of FRED, while R's download.file() succeeds. Fetch once with the tool
# that works; the notes and student scripts then read local CSVs and never
# depend on a live network.
#
# Produces, in data/:
#   state_hpi_ur_pop.csv   50 states: house price index (level AND ten-year
#                          change), unemployment, population, population growth
#   us_recession_daily.csv NBER recession indicator, daily since 1854
#   consumer_confidence.csv  US consumer confidence index, monthly
#   monmouth_poll.csv      Monmouth University Poll #240, questions 16 and 38
#
# Run from the repository root:
#     Rscript tools/seed_chapter02_data.R
#
# Roughly 150 requests, so allow two or three minutes. Re-running only
# fetches what is missing; set force <- TRUE to refresh everything.

force     <- FALSE
cache_dir <- file.path("data", "raw")
out_dir   <- "data"
pause     <- 0.4

dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir,   recursive = TRUE, showWarnings = FALSE)

state_abb <- c(
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
  "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
  "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
  "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
)

# --- helpers --------------------------------------------------------------

fred_csv <- function(id, start = NA_character_) {
  dest <- file.path(cache_dir, sprintf("%s__%s__none.csv", id,
                                       ifelse(is.na(start), "none", start)))
  if (!force && file.exists(dest) && file.size(dest) > 20) return(dest)

  url <- sprintf("https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s", id)
  if (!is.na(start)) url <- paste0(url, "&cosd=", start)

  ok <- tryCatch({
    suppressWarnings(download.file(url, dest, mode = "wb", quiet = TRUE))
    TRUE
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

last_value <- function(d) if (nrow(d)) d$value[nrow(d)] else NA_real_
last_date  <- function(d) if (nrow(d)) d$date[nrow(d)]  else as.Date(NA)

value_near <- function(d, target) {
  # Value of the observation closest to `target`, for computing growth over a
  # fixed span even when series have different publication calendars.
  if (!nrow(d)) return(NA_real_)
  d$value[which.min(abs(as.numeric(d$date - target)))]
}

date_near <- function(d, target) {
  # The DATE of that same observation. Reported so the note can state the
  # window each growth rate is actually measured over, rather than implying
  # the two windows line up exactly. They do not: population runs to the most
  # recent Census vintage, the house price index to the most recent quarter.
  if (!nrow(d)) return(as.Date(NA))
  d$date[which.min(abs(as.numeric(d$date - target)))]
}

# --- 1. State panel -------------------------------------------------------

cat("1. State housing, unemployment, and population\n")

rows <- list()
failed <- character(0)

for (abb in state_abb) {
  ids <- c(hpi = paste0(abb, "STHPI"), ur = paste0(abb, "UR"), pop = paste0(abb, "POP"))
  paths <- vapply(ids, fred_csv, character(1))

  if (any(is.na(paths))) {
    failed <- c(failed, abb)
    cat(sprintf("   %-3s FAILED (%s)\n", abb,
                paste(names(ids)[is.na(paths)], collapse = ", ")))
    next
  }

  hpi <- read_series(paths[["hpi"]])
  ur  <- read_series(paths[["ur"]])
  pop <- read_series(paths[["pop"]])

  pop_now  <- last_value(pop)
  pop_then <- value_near(pop, last_date(pop) - 3653)   # ten years earlier

  # The HPI is an INDEX with 1980Q1 = 100, so its level measures cumulative
  # appreciation since 1980 and is not comparable across states as a price.
  # What IS comparable, and what matches the ten-year population growth
  # window, is the ten-year CHANGE in the index. Carry both: the level so the
  # note can show what goes wrong with it, the change so it can show the fix.
  hpi_now  <- last_value(hpi)
  hpi_then <- value_near(hpi, last_date(hpi) - 3653)
  hpi_then_date <- date_near(hpi, last_date(hpi) - 3653)
  # Also the FIVE-year change. The note uses it to show that a relationship
  # can be strong over one window and absent over another, so the window
  # belongs in the sentence. Computed here so the note never asserts it.
  hpi_five <- value_near(hpi, last_date(hpi) - 1826)
  # And a COUNT: quarters in the most recent forty in which the index fell.
  # The note needs one genuinely discrete variable -- something counted, not
  # measured -- so that the ungrouped frequency distribution, the gaps between
  # histogram bars, and the empty class all have somewhere to live.
  hpi_recent <- hpi[hpi$date >= last_date(hpi) - 3653, ]
  hpi_down   <- sum(diff(hpi_recent$value) < 0)

  rows[[abb]] <- data.frame(
    Member = abb,
    HPI    = hpi_now,
    HPIO   = hpi_then,
    gHPI   = round(100 * (hpi_now - hpi_then) / hpi_then, 1),
    gHPI5  = round(100 * (hpi_now - hpi_five) / hpi_five, 1),
    nDOWN  = hpi_down,
    UR     = last_value(ur),
    POPN   = pop_now,
    POPO   = pop_then,
    gPOP   = round(100 * (pop_now - pop_then) / pop_then, 1),
    HPI_date  = last_date(hpi),
    HPIO_date = hpi_then_date,
    UR_date   = last_date(ur),
    POP_date  = last_date(pop),
    stringsAsFactors = FALSE
  )
}

state_data <- do.call(rbind, rows)
rownames(state_data) <- NULL
write.csv(state_data, file.path(out_dir, "state_hpi_ur_pop.csv"), row.names = FALSE)

cat(sprintf("   %d states written", nrow(state_data)))
if (length(failed)) cat(sprintf("  (failed: %s)", paste(failed, collapse = ", ")))
cat("\n")
if (nrow(state_data)) {
  cat(sprintf("   HPI as of %s | UR as of %s | POP as of %s\n",
              max(state_data$HPI_date), max(state_data$UR_date),
              max(state_data$POP_date)))
  cat(sprintf("   Population growth measured over the ten years to %s\n",
              max(state_data$POP_date)))
  cat(sprintf("   House price growth measured over the ten years to %s\n",
              max(state_data$HPI_date)))
  cat(sprintf("   nDOWN (quarters with a falling index, last ten years): %d to %d, median %g\n",
              min(state_data$nDOWN), max(state_data$nDOWN),
              median(state_data$nDOWN)))
  cat(sprintf("   r(house price growth, population growth) = %+.3f\n",
              cor(state_data$gHPI, state_data$gPOP)))
  cat(sprintf("   r(house price LEVEL,  population growth) = %+.3f  <- the index trap\n",
              cor(state_data$HPI, state_data$gPOP)))
}

# --- 2. Recession indicator ----------------------------------------------

cat("2. NBER recession indicator (USRECD)\n")
p <- fred_csv("USRECD")
if (!is.na(p)) {
  rec <- read_series(p)
  names(rec) <- c("date", "recession")
  rec$recession <- as.integer(rec$recession)
  write.csv(rec, file.path(out_dir, "us_recession_daily.csv"), row.names = FALSE)
  cat(sprintf("   %s rows, %s to %s, %.1f%% of days in recession\n",
              format(nrow(rec), big.mark = ","), min(rec$date), max(rec$date),
              100 * mean(rec$recession)))
} else {
  cat("   FAILED\n")
}

# --- 3. Consumer confidence ----------------------------------------------
#
# TWO series, deliberately.
#
#   USACSCICP02STSAM  - currently published
#   CSCICP03USM665S   - the same concept, discontinued in early 2024
#
# The discontinued one is kept because it teaches something no live series
# can: a retired series downloads without error, parses cleanly, and contains
# only correct values. The ONLY way to notice is to look at the last
# observation date. That check belongs in the course.

cat("3. Consumer confidence\n")

fetch_cci <- function(id, outfile, label) {
  p <- fred_csv(id)
  if (is.na(p)) { cat(sprintf("   %-16s FAILED\n", id)); return(invisible(NULL)) }
  d <- read_series(p)
  names(d) <- c("date", "cci")
  write.csv(d, file.path(out_dir, outfile), row.names = FALSE)
  stale_days <- as.numeric(Sys.Date() - max(d$date))
  cat(sprintf("   %-16s %4s rows, %s to %s  [%s]\n",
              id, format(nrow(d), big.mark = ","),
              min(d$date), max(d$date),
              if (stale_days > 200) sprintf("STALE: %.0f months", stale_days / 30.4)
              else "current"))
  invisible(d)
}

fetch_cci("USACSCICP02STSAM", "consumer_confidence.csv", "current")
fetch_cci("CSCICP03USM665S", "consumer_confidence_discontinued.csv", "discontinued")

# --- 4. Monmouth poll -----------------------------------------------------

cat("4. Monmouth University Poll #240\n")
mon_raw <- file.path(cache_dir, "MUP240_NATL_archive_full.tab")
ok <- TRUE
if (force || !file.exists(mon_raw)) {
  ok <- tryCatch({
    suppressWarnings(download.file(
      "https://crooker.faculty.unlv.edu/mba775/data/MUP240_NATL_archive_full.tab",
      mon_raw, mode = "wb", quiet = TRUE))
    TRUE
  }, error = function(e) FALSE)
}

if (ok && file.exists(mon_raw)) {
  mon <- read.table(mon_raw, header = TRUE, sep = "\t",
                    stringsAsFactors = FALSE, quote = "", comment.char = "")

  q16_labels <- c("1. No changes", "2. Some improvement",
                  "3. Many improvements", "4. Significant changes")
  q38_labels <- c("1. Yes, go out", "2. Yes, stay in", "3. No, nothing special")

  decode <- function(v, labels) {
    out <- rep(NA_character_, length(v))
    for (i in seq_along(labels)) out[v == i] <- labels[i]
    out[v == 9] <- "9. Do not know / Refused"
    out
  }

  poll <- data.frame(
    respondent = seq_len(nrow(mon)),
    q16_system_of_government = decode(mon$Q16, q16_labels),
    q38_valentines_plans     = decode(mon$Q38, q38_labels),
    stringsAsFactors = FALSE
  )
  write.csv(poll, file.path(out_dir, "monmouth_poll.csv"), row.names = FALSE)
  cat(sprintf("   %s respondents; Q16 answered by %s, Q38 by %s\n",
              format(nrow(poll), big.mark = ","),
              sum(!is.na(poll$q16_system_of_government)),
              sum(!is.na(poll$q38_valentines_plans))))
} else {
  cat("   FAILED to download the poll file\n")
}

# --- summary --------------------------------------------------------------

cat("\n----------------------------------------------------------------\n")
made <- list.files(out_dir, pattern = "\\.csv$")
for (f in made) {
  cat(sprintf("  %-26s %8s bytes\n", f,
              format(file.size(file.path(out_dir, f)), big.mark = ",")))
}
cat("\nCommit these so students can download them.\n")
