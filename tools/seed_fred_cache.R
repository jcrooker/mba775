# seed_fred_cache.R -------------------------------------------------------
#
# Populate data/raw/ with every FRED series the MBA 775 lecture notes need,
# using R's downloader.
#
# Why R: on this machine Python's HTTPS clients (urllib, curl.exe, PowerShell)
# are refused by the CDN in front of FRED, while R's download.file() succeeds.
# Rather than fight that, we fetch once with the tool that works and let the
# notes read from the cache. That is also better practice: the notes stop
# depending on a live network at render time, and data/raw/ holds an
# unmodified copy of every source file.
#
# Run from the qmd-migration-test folder:
#
#     "C:/Program Files/R/R-4.x.x/bin/Rscript.exe" seed_fred_cache.R
#
# or just source() it in the RStudio Console with that folder as the working
# directory. Re-running only fetches what is missing; use force = TRUE to
# refresh everything.

force     <- FALSE          # TRUE re-downloads files that already exist
cache_dir <- file.path("data", "raw")
pause     <- 0.5            # seconds between requests, to stay polite

dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

# --- the series each note needs ------------------------------------------

state_abb <- c(
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
  "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
  "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
  "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
)

indicators_4050 <- c(
  "SP500", "T10Y3M", "M2REAL", "PERMIT", "HOUST", "DGORDER",
  "UCOGNO", "AWHMAN", "ICSA", "TEMPHELPS", "UMCSENT", "USSLIND"
)

# Each request is (series id, start date). NA start means full history.
requests <- rbind(
  # chapter 01
  data.frame(id = "DFF", start = "2020-01-01", stringsAsFactors = FALSE),
  data.frame(id = "DFF", start = NA_character_,  stringsAsFactors = FALSE),
  data.frame(id = paste0(state_abb, "UR"), start = "1976-01-01",
             stringsAsFactors = FALSE),
  # note 4050
  data.frame(id = indicators_4050, start = "1990-01-01",
             stringsAsFactors = FALSE)
)

# --- helpers --------------------------------------------------------------

cache_name <- function(id, start) {
  # Must match fred_tools._cache_path(): <ID>__<start|none>__<end|none>.csv
  file.path(cache_dir,
            sprintf("%s__%s__none.csv", id, ifelse(is.na(start), "none", start)))
}

fred_url <- function(id, start) {
  u <- sprintf("https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s", id)
  if (!is.na(start)) u <- paste0(u, "&cosd=", start)
  u
}

looks_like_csv <- function(path) {
  if (!file.exists(path) || file.size(path) < 20) return(FALSE)
  first <- tryCatch(readLines(path, n = 1, warn = FALSE), error = function(e) "")
  length(first) == 1 &&
    grepl("^(observation_date|DATE)\\s*,", first, ignore.case = TRUE)
}

# --- fetch loop -----------------------------------------------------------

ok <- character(0)
skipped <- character(0)
failed <- list()

cat(sprintf("Seeding %d series into %s\n\n", nrow(requests), cache_dir))

for (i in seq_len(nrow(requests))) {
  id    <- requests$id[i]
  start <- requests$start[i]
  dest  <- cache_name(id, start)
  label <- sprintf("%s (%s)", id, ifelse(is.na(start), "full history", start))

  if (!force && looks_like_csv(dest)) {
    skipped <- c(skipped, label)
    next
  }

  result <- tryCatch({
    suppressWarnings(
      download.file(fred_url(id, start), dest, mode = "wb", quiet = TRUE)
    )
    "downloaded"
  }, error = function(e) conditionMessage(e))

  if (result == "downloaded" && looks_like_csv(dest)) {
    n <- length(readLines(dest, warn = FALSE)) - 1L
    cat(sprintf("  OK      %-28s %6d rows\n", label, n))
    ok <- c(ok, label)
  } else {
    # Do not leave a truncated or HTML file behind: the notes would try to
    # parse it and fail confusingly.
    if (file.exists(dest)) unlink(dest)
    reason <- if (result == "downloaded") "response was not FRED CSV" else result
    cat(sprintf("  FAILED  %-28s %s\n", label, reason))
    failed[[label]] <- reason
  }

  Sys.sleep(pause)
}

# --- summary --------------------------------------------------------------

cat("\n----------------------------------------------------------------\n")
cat(sprintf("downloaded : %d\n", length(ok)))
cat(sprintf("already ok : %d\n", length(skipped)))
cat(sprintf("failed     : %d\n", length(failed)))

if (length(failed)) {
  cat("\nFailures:\n")
  for (nm in names(failed)) cat(sprintf("  %-28s %s\n", nm, failed[[nm]]))
  cat("\nSome FRED series are genuinely retired (USSLIND may be one).\n")
  cat("A failure here is reported by the notes rather than crashing them.\n")
}

cat(sprintf("\nCache now holds %d CSV files.\n",
            length(list.files(cache_dir, pattern = "\\.csv$"))))
cat("The notes will read from this cache and no longer need live access.\n")
