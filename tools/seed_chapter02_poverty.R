# seed_chapter02_poverty.R -------------------------------------------------
#
# Download the long-run global poverty and income data for Chapter 2 from
# Our World in Data, using R's downloader.
#
# Why R, and why on your machine: the same reason as seed_chapter02_data.R.
# OWID serves everything from ourworldindata.org, which the assistant's
# sandbox cannot reach at all, and Python's HTTPS client is refused by some
# CDNs on this machine. R's download.file() is the transport that works.
# Fetch once, commit the CSV, and the note reads a local file forever after.
#
# WHY TWO SERIES INSTEAD OF ONE CHART
#
# The obvious chart -- OWID's "Share of population living in extreme poverty
# vs. GDP per capita" -- still exists at that title, but its data has been
# swapped underneath it. It now runs on the World Bank's PIP survey series
# (1963 onward, $3/day at 2021 prices) joined to World Bank GDP per capita
# (1990 onward). The two overlap only from 1990, so that chart can no longer
# draw the two-century line it used to.
#
# The long-run picture has to be rebuilt from the two series that DO still
# publish back to 1820:
#
#   world-population-in-extreme-poverty-absolute
#       Ravallion (2016) updated with World Bank (2019). 1820-2015.
#       Gives COUNTS of people in and not in extreme poverty, at the
#       $1.90/day line in 2011 prices. The share is computed from the two.
#
#   gdp-per-capita-maddison
#       Maddison Project Database 2023 (Bolt and van Zanden).
#       Constant international-$ at 2011 prices.
#
# Both are on the 2011 PPP basis, which is why they can be put on the same
# chart -- and which is exactly what has to be stated in the caption, because
# the World Bank's CURRENT line is $3/day at 2021 prices and puts a different
# number of people below it.
#
# Run from the repository root:
#     Rscript tools/seed_chapter02_poverty.R
#
# It prints what it found. Read that output before trusting anything.

force     <- FALSE
cache_dir <- file.path("data", "raw")
out_dir   <- "data"

dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_dir,   recursive = TRUE, showWarnings = FALSE)

SLUGS <- c(
  poverty = "world-population-in-extreme-poverty-absolute",
  gdp     = "gdp-per-capita-maddison"
)

owid_csv <- function(slug) {
  dest <- file.path(cache_dir, sprintf("owid__%s.csv", slug))
  if (!force && file.exists(dest) && file.size(dest) > 200) {
    cat(sprintf("   cached  %s\n", basename(dest)))
    return(dest)
  }
  url <- sprintf("https://ourworldindata.org/grapher/%s.csv?csvType=full&useColumnShortNames=false",
                 slug)
  ok <- tryCatch({
    suppressWarnings(download.file(url, dest, mode = "wb", quiet = TRUE)); TRUE
  }, error = function(e) { cat("   ERROR: ", conditionMessage(e), "\n"); FALSE })
  if (!ok) { if (file.exists(dest)) unlink(dest); return(NA_character_) }

  first <- tryCatch(readLines(dest, n = 1, warn = FALSE), error = function(e) "")
  if (!length(first) || !grepl("^Entity,", first)) {
    cat(sprintf("   REJECTED %s -- first line was not an OWID CSV header:\n     %s\n",
                slug, substr(paste(first, collapse = ""), 1, 120)))
    unlink(dest); return(NA_character_)
  }
  cat(sprintf("   fetched %s  (%.0f KB)\n", basename(dest), file.size(dest) / 1024))
  dest
}

describe <- function(d, label) {
  cat(sprintf("\n   %s\n", label))
  cat(sprintf("     rows      %s\n", format(nrow(d), big.mark = ",")))
  cat(sprintf("     columns   %s\n", paste(names(d), collapse = " | ")))
  cat(sprintf("     years     %s to %s\n", min(d$Year, na.rm = TRUE),
              max(d$Year, na.rm = TRUE)))
  cat(sprintf("     entities  %d\n", length(unique(d$Entity))))
  cat(sprintf("     World?    %s\n",
              if ("World" %in% d$Entity) "yes" else "NO -- see note below"))
}

cat("Chapter 2: long-run poverty and income (Our World in Data)\n\n")

cat("1. Downloading\n")
paths <- vapply(SLUGS, owid_csv, character(1))
if (any(is.na(paths))) {
  cat("\nAt least one download failed. Nothing was written.\n")
  quit(status = 1)
}

cat("\n2. What arrived\n")
pov <- read.csv(paths[["poverty"]], stringsAsFactors = FALSE, check.names = FALSE)
gdp <- read.csv(paths[["gdp"]],     stringsAsFactors = FALSE, check.names = FALSE)
describe(pov, "world-population-in-extreme-poverty-absolute")
describe(gdp, "gdp-per-capita-maddison")

# --- locate the columns, and REFUSE to guess -------------------------------
#
# The first version of this script warned when a pattern matched more than one
# column and then used the first match anyway. It picked "not in extreme
# poverty" for both the in-poverty and not-in-poverty roles, which made every
# year come out at exactly 50% and the correlation NA. Warning and continuing
# is worse than stopping: the output looked like data.
#
# So: each column is matched by a rule that must select EXACTLY ONE column.
# Zero matches or more than one and the script stops and prints everything.

pick_one <- function(d, want, reject = NULL, what = "") {
  cand <- setdiff(names(d), c("Entity", "Code", "Year"))
  cand <- cand[grepl(want, cand, ignore.case = TRUE)]
  if (!is.null(reject)) cand <- cand[!grepl(reject, cand, ignore.case = TRUE)]
  if (length(cand) != 1) {
    cat(sprintf("\n   STOPPING: expected exactly one %s column, found %d.\n",
                what, length(cand)))
    if (length(cand)) cat(sprintf("     matched: %s\n", paste(cand, collapse = " | ")))
    cat(sprintf("     all columns: %s\n", paste(names(d), collapse = " | ")))
    cat("   Send this output and the rule will be corrected rather than guessed.\n")
    quit(status = 1)
  }
  cand[1]
}

cat("\n3. Choosing columns (each rule must match exactly one)\n")
col_notin <- pick_one(pov, "not in extreme poverty|not in poverty",
                      NULL, "not-in-poverty count")
col_in    <- pick_one(pov, "in extreme poverty|in poverty",
                      "\\bnot\\b", "in-poverty count")
col_gdp   <- pick_one(gdp, "gdp per capita",
                      "annotation", "GDP per capita")

cat(sprintf("   in poverty      <- %s\n", col_in))
cat(sprintf("   not in poverty  <- %s\n", col_notin))
cat(sprintf("   GDP per capita  <- %s\n", col_gdp))

if (identical(col_in, col_notin)) {
  cat("\n   STOPPING: the two poverty columns resolved to the same column.\n")
  quit(status = 1)
}

# --- world-level series ---------------------------------------------------
cat("\n4. Building the world series\n")

pw <- pov[pov$Entity == "World", c("Year", col_in, col_notin)]
names(pw) <- c("year", "n_poor", "n_notpoor")
pw$share_poor <- round(100 * pw$n_poor / (pw$n_poor + pw$n_notpoor), 2)

gw <- gdp[gdp$Entity == "World", c("Year", col_gdp)]
names(gw) <- c("year", "gdp_pc")

world <- merge(pw, gw, by = "year")
world <- world[order(world$year), ]

if (!nrow(world)) {
  cat("   No overlapping World rows. Check the 'World?' lines above.\n")
  cat("   If Maddison has no World entity, say so and we will aggregate\n")
  cat("   from countries weighted by population instead.\n")
  quit(status = 1)
}

write.csv(world, file.path(out_dir, "world_poverty_income.csv"), row.names = FALSE)
cat(sprintf("   world_poverty_income.csv  %d rows, %d to %d\n",
            nrow(world), min(world$year), max(world$year)))

# --- what did NOT survive the join ----------------------------------------
missing <- setdiff(pw$year, world$year)
if (length(missing))
  cat(sprintf("   %d poverty years had no Maddison World GDP and were dropped: %s\n",
              length(missing), paste(missing, collapse = ", ")))

# --- the Maddison country panel, kept because it IS unreadable ------------
# 21,586 rows across 178 countries. This is the table that cannot be read,
# for the opening of the section. The poverty series is World-only, so it
# cannot supply one.
cg <- gdp[, c("Entity", "Code", "Year", col_gdp)]
names(cg) <- c("entity", "code", "year", "gdp_pc")
cg <- cg[!is.na(cg$gdp_pc), ]
write.csv(cg, file.path(out_dir, "gdp_per_capita_panel.csv"), row.names = FALSE)
cat(sprintf("   gdp_per_capita_panel.csv  %s rows, %d entities, %d to %d\n",
            format(nrow(cg), big.mark = ","), length(unique(cg$entity)),
            min(cg$year), max(cg$year)))

# --- the numbers the note will quote, printed so they can be checked ------
cat("\n5. The world series in full\n\n")
print(world, row.names = FALSE)

if (sd(world$share_poor) == 0) {
  cat("\n   STOPPING: the poverty share is identical in every year, which means\n")
  cat("   the two count columns are the same column. Send this output.\n")
  quit(status = 1)
}

cat("\n6. The headline numbers -- check these against the chart you remember\n")
first <- world[1, ]; last <- world[nrow(world), ]
cat(sprintf("   %d: %.1f%% in extreme poverty, GDP per capita $%s\n",
            first$year, first$share_poor,
            format(round(first$gdp_pc), big.mark = ",")))
cat(sprintf("   %d: %.1f%% in extreme poverty, GDP per capita $%s\n",
            last$year, last$share_poor,
            format(round(last$gdp_pc), big.mark = ",")))
cat(sprintf("   correlation of share with log GDP per capita: %+.3f\n",
            cor(world$share_poor, log(world$gdp_pc))))

cat("\nDone. Files are in data/. Paste this output back.\n")
