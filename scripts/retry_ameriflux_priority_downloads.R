USER_EMAIL <- "akulkumar02008@gmail.com"
USER_ID <- "akulkumar02008@gmail.com"

out_dir <- "data/raw/towers/_downloads/ameriflux_base"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create("results/tower_downloads", recursive = TRUE, showWarnings = FALSE)

options(timeout = 1200)
options(repos = c(CRAN = "https://cloud.r-project.org"))

library(amerifluxr)

priority_sites <- c(
  "US-Var",
  "US-SRG",
  "US-SRM",
  "US-Wkg",
  "US-AR1",
  "US-AR2",
  "US-ARb",
  "US-ARM",
  "US-Cop",
  "US-Kon",
  "US-KFS",
  "US-Whs",
  "US-Seg",
  "US-Ses"
)

log_path <- "results/tower_downloads/ameriflux_retry_download_log.csv"

download_one <- function(sid, attempt) {
  cat("\n===== site", sid, "| attempt", attempt, "=====\n")
  flush.console()

  ans <- tryCatch({
    amf_download_base(
      user_id = USER_ID,
      user_email = USER_EMAIL,
      site_id = sid,
      data_product = "BASE-BADM",
      data_policy = "CCBY4.0",
      agree_policy = TRUE,
      intended_use = "remote_sensing",
      intended_use_text = "Tower validation of satellite-derived grassland water-use efficiency response under compound atmospheric and soil-moisture stress.",
      out_dir = out_dir,
      verbose = TRUE
    )
  }, error = function(e) e)

  if (inherits(ans, "error")) {
    cat("FAILED:", conditionMessage(ans), "\n")
    return(data.frame(
      site_id = sid,
      attempt = attempt,
      status = "FAIL",
      file = "",
      error = conditionMessage(ans),
      stringsAsFactors = FALSE
    ))
  } else {
    cat("SUCCESS\n")
    print(ans)
    return(data.frame(
      site_id = sid,
      attempt = attempt,
      status = "PASS",
      file = paste(ans, collapse = ";"),
      error = "",
      stringsAsFactors = FALSE
    ))
  }
}

log <- data.frame(
  site_id = character(),
  attempt = integer(),
  status = character(),
  file = character(),
  error = character(),
  stringsAsFactors = FALSE
)

for (sid in priority_sites) {
  success <- FALSE

  for (attempt in 1:3) {
    row <- download_one(sid, attempt)
    log <- rbind(log, row)
    write.csv(log, log_path, row.names = FALSE)

    if (row$status[1] == "PASS") {
      success <- TRUE
      break
    }

    cat("Sleeping 20 seconds before retry...\n")
    Sys.sleep(20)
  }

  if (!success) {
    cat("Giving up on", sid, "after 3 attempts.\n")
  }
}

cat("\n===== FINAL DOWNLOAD SUMMARY =====\n")
print(table(log$status))
write.csv(log, log_path, row.names = FALSE)

cat("\n===== FILES IN OUTPUT DIR =====\n")
print(list.files(out_dir, recursive = TRUE, full.names = TRUE))
