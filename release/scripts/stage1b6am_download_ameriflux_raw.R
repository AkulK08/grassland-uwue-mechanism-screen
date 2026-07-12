options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("amerifluxr", quietly = TRUE)) {
  install.packages("amerifluxr")
}

library(amerifluxr)

user_id <- Sys.getenv("AMF_USER")
user_email <- Sys.getenv("AMF_EMAIL")
out_dir <- "data/raw/towers/_reza_raw_exports/ameriflux_base"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sites <- c(
  "CA-SF3",
  "US-CMW", "US-Cop", "US-Dk1", "US-Ne1", "US-Ne2", "US-Ne3",
  "US-SP1", "US-Ton", "US-Var"
)

log_rows <- list()

for (site in sites) {
  for (policy in c("CCBY4.0", "LEGACY")) {
    message("Trying ", site, " policy=", policy)
    status <- "not_run"
    file_out <- NA_character_
    err <- NA_character_

    tryCatch({
      file_out <- amf_download_base(
        user_id = user_id,
        user_email = user_email,
        site_id = site,
        data_product = "BASE-BADM",
        data_policy = policy,
        agree_policy = TRUE,
        intended_use = "remote_sensing",
        intended_use_text = "Tower validation of satellite WUE/uWUE products for grassland compound drought stress analysis",
        out_dir = out_dir,
        verbose = TRUE
      )
      status <- "downloaded"
    }, error = function(e) {
      status <<- "failed"
      err <<- as.character(e$message)
    })

    log_rows[[length(log_rows) + 1]] <- data.frame(
      site_id = site,
      data_policy = policy,
      status = status,
      file_out = ifelse(length(file_out) == 0, NA_character_, as.character(file_out)[1]),
      error = ifelse(is.na(err), NA_character_, err),
      stringsAsFactors = FALSE
    )

    if (status == "downloaded") break
  }
}

log_df <- do.call(rbind, log_rows)
write.csv(log_df, "results/stage1b6am_raw_tower_download_ingest/tables/Table_PRODUCT03bs_ameriflux_download_log.csv", row.names = FALSE)
print(log_df)
