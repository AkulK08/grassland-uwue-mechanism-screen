USER_EMAIL <- "akulkumar02008@gmail.com"
USER_ID <- "akulkumar02008@gmail.com"

out_dir <- "data/raw/towers/_downloads/ameriflux_base"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create("results/tower_downloads", recursive = TRUE, showWarnings = FALSE)

options(repos = c(CRAN = "https://cloud.r-project.org"))

cat("===== R version =====\n")
print(R.version.string)

cat("\n===== installing dependencies =====\n")
needed <- c("remotes", "curl", "httr", "jsonlite", "dplyr", "readr", "lubridate")
for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p)
  }
}

cat("\n===== installing amerifluxr from GitHub =====\n")
if (!requireNamespace("amerifluxr", quietly = TRUE)) {
  remotes::install_github("chuhousen/amerifluxr", upgrade = "never", dependencies = TRUE)
}

cat("\n===== loading amerifluxr =====\n")
library(amerifluxr)

cat("Using user_id:", USER_ID, "\n")
cat("Using user_email:", USER_EMAIL, "\n")
cat("Output directory:", out_dir, "\n")

site_id <- "US-Var"

cat("\n===== trying AmeriFlux BASE-BADM download for", site_id, "=====\n")

ans <- tryCatch({
  amf_download_base(
    user_id = USER_ID,
    user_email = USER_EMAIL,
    site_id = site_id,
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
  cat("\nDOWNLOAD_FAILED\n")
  cat("Error message:\n")
  cat(conditionMessage(ans), "\n")
  writeLines(conditionMessage(ans), "results/tower_downloads/ameriflux_one_site_error.txt")
  quit(status = 1)
} else {
  cat("\nDOWNLOAD_SUCCESS\n")
  print(ans)
  write.csv(
    data.frame(site_id = site_id, file = paste(ans, collapse = ";")),
    "results/tower_downloads/ameriflux_one_site_success.csv",
    row.names = FALSE
  )
}
