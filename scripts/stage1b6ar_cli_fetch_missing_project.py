from pathlib import Path
import os, re, json, csv, zipfile, subprocess, shutil
from urllib.parse import urlparse, unquote

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6ar_cli_fetch_missing_project"
TAB = OUT / "tables"
TXT = OUT / "text"
DISC = ROOT / "data/raw/towers/_project_raw_exports/cli_download_discovery"
DEST = ROOT / "data/raw/towers/_project_raw_exports/manual_fluxnet"

for p in [TAB, TXT, DISC, DEST]:
    p.mkdir(parents=True, exist_ok=True)

TARGETS = ["NL-Hrw", "US-SP1", "US-Ne1", "US-Ne2", "US-Ne3"]
PRIORITY_DOWNLOAD_TARGETS = ["NL-Hrw", "US-SP1"]

def run(cmd, cwd=None, check=False):
    print("\n$ " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
        print(r.stdout[-4000:])
        if r.stderr.strip():
            print("STDERR:", r.stderr[-4000:])
        if check and r.returncode != 0:
            raise RuntimeError("Command failed")
        return r
    except Exception as e:
        print("FAILED:", repr(e))
        return None

def curl(url, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "--connect-timeout", "30", "--max-time", "900", "-o", str(out), url]
    r = run(cmd)
    return r is not None and r.returncode == 0 and out.exists() and out.stat().st_size > 1000

def safe_name_from_url(url, fallback):
    p = urlparse(url)
    name = Path(unquote(p.path)).name
    if not name or "." not in name:
        name = fallback
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name

# 1) Download metadata/code snapshots that may contain direct URLs.
sources = {
    "fluxnet_data_explorer_main.zip": "https://github.com/trevorkeenan/fluxnet-data-explorer/archive/refs/heads/main.zip",
    "fluxnet_shuttle_main.zip": "https://github.com/fluxnet/shuttle/archive/refs/heads/main.zip",
}

download_log = []
for fn, url in sources.items():
    out = DISC / fn
    ok = curl(url, out)
    download_log.append({"kind": "metadata_repo", "url": url, "out": str(out), "ok": ok})
    if ok:
        try:
            with zipfile.ZipFile(out) as z:
                exdir = DISC / fn.replace(".zip", "")
                if exdir.exists():
                    shutil.rmtree(exdir)
                z.extractall(exdir)
                download_log.append({"kind": "extract_metadata_repo", "url": url, "out": str(exdir), "ok": True})
        except Exception as e:
            download_log.append({"kind": "extract_metadata_repo", "url": url, "out": str(out), "ok": False, "error": repr(e)})

# 2) Search every text-like file for target site rows and URLs.
site_hits = []
url_hits = []
text_ext = {".csv", ".json", ".txt", ".md", ".html", ".js", ".ts", ".yml", ".yaml"}

for p in DISC.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in text_ext:
        continue
    try:
        txt = p.read_text(errors="ignore")
    except Exception:
        continue

    for site in TARGETS:
        if site in txt or site.replace("-", "_") in txt:
            # capture surrounding lines
            lines = txt.splitlines()
            for i, line in enumerate(lines):
                if site in line or site.replace("-", "_") in line:
                    site_hits.append({
                        "site_id": site,
                        "file": str(p),
                        "line_no": i + 1,
                        "line": line[:2000],
                    })

    urls = re.findall(r"https?://[^\s\"'<>),]+", txt)
    for u in urls:
        for site in TARGETS:
            if site in u or site.replace("-", "_") in u or site.replace("-", "") in u:
                url_hits.append({
                    "site_id": site,
                    "file": str(p),
                    "url": u,
                    "download_candidate": bool(re.search(r"\.(zip|csv|txt|gz)(\?|$)", u, flags=re.I)),
                })

# 3) Also infer URLs from hit lines containing URLs.
for h in site_hits:
    for u in re.findall(r"https?://[^\s\"'<>),]+", h["line"]):
        url_hits.append({
            "site_id": h["site_id"],
            "file": h["file"],
            "url": u,
            "download_candidate": bool(re.search(r"\.(zip|csv|txt|gz)(\?|$)", u, flags=re.I)),
        })

# De-duplicate URL hits.
seen = set()
dedup_urls = []
for row in url_hits:
    key = (row["site_id"], row["url"])
    if key not in seen:
        seen.add(key)
        dedup_urls.append(row)
url_hits = dedup_urls

# 4) Try direct downloads from candidate URLs.
direct_downloads = []
for row in url_hits:
    site = row["site_id"]
    url = row["url"]
    if site not in PRIORITY_DOWNLOAD_TARGETS:
        continue

    # Try only plausible data URLs first, not random docs.
    plausible = (
        row.get("download_candidate")
        or "download" in url.lower()
        or "fluxnet" in url.lower()
        or "icos" in url.lower()
        or "ameriflux" in url.lower()
    )
    if not plausible:
        continue

    name = safe_name_from_url(url, f"{site}_cli_download.dat")
    if site not in name:
        name = f"{site}_{name}"
    out = DEST / name

    ok = curl(url, out)
    direct_downloads.append({
        "site_id": site,
        "url": url,
        "out": str(out),
        "ok": ok,
        "bytes": out.stat().st_size if out.exists() else 0,
    })

# 5) Try AmeriFlux BASE download for US-SP1 again via amerifluxr if R exists.
# Note: amerifluxr only supports BASE-BADM, not AmeriFlux FLUXNET. This may still help if a newer BASE zip has better columns.
amf_user = os.environ.get("AMF_USER", "")
amf_email = os.environ.get("AMF_EMAIL", "")

r_script = DISC / "try_us_sp1_amerifluxr.R"
r_script.write_text(r'''
args <- commandArgs(trailingOnly=TRUE)
out_dir <- args[1]
user_id <- Sys.getenv("AMF_USER")
user_email <- Sys.getenv("AMF_EMAIL")
if (user_id == "" || user_email == "") {
  cat("AMF_USER or AMF_EMAIL not set; skipping amerifluxr retry\n")
  quit(status=0)
}
if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org")
if (!requireNamespace("amerifluxr", quietly=TRUE)) remotes::install_github("chuhousen/amerifluxr", upgrade="never")
library(amerifluxr)
for (policy in c("CCBY4.0", "LEGACY")) {
  cat("Trying US-SP1 BASE-BADM", policy, "\n")
  try({
    f <- amf_download_base(
      user_id=user_id,
      user_email=user_email,
      site_id="US-SP1",
      data_product="BASE-BADM",
      data_policy=policy,
      agree_policy=TRUE,
      intended_use="remote_sensing",
      intended_use_text="tower validation and product comparison for WUE research",
      out_dir=out_dir,
      verbose=TRUE
    )
    print(f)
  }, silent=FALSE)
}
''', encoding="utf-8")

if shutil.which("Rscript"):
    r = run(["Rscript", str(r_script), str(DEST)])
    direct_downloads.append({
        "site_id": "US-SP1",
        "url": "amerifluxr::amf_download_base",
        "out": str(DEST),
        "ok": r is not None and r.returncode == 0,
        "note": "amerifluxr supports BASE-BADM only; AmeriFlux FLUXNET may still require website/session.",
    })

# 6) Save discovery tables.
with open(TAB / "Table_PRODUCT03cz_cli_metadata_download_log.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for row in download_log for k in row.keys()}))
    w.writeheader()
    w.writerows(download_log)

with open(TAB / "Table_PRODUCT03da_cli_site_hits.csv", "w", newline="") as f:
    fieldnames = ["site_id", "file", "line_no", "line"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in site_hits:
        w.writerow(row)

with open(TAB / "Table_PRODUCT03db_cli_url_hits.csv", "w", newline="") as f:
    fieldnames = ["site_id", "file", "url", "download_candidate"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in url_hits:
        w.writerow(row)

with open(TAB / "Table_PRODUCT03dc_cli_direct_download_attempts.csv", "w", newline="") as f:
    fieldnames = sorted({k for row in direct_downloads for k in row.keys()}) if direct_downloads else ["site_id", "url", "out", "ok"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in direct_downloads:
        w.writerow(row)

decision = {
    "metadata_repos_downloaded": sum(1 for x in download_log if x.get("kind") == "metadata_repo" and x.get("ok")),
    "site_hit_rows": len(site_hits),
    "url_hit_rows": len(url_hits),
    "direct_download_attempts": len(direct_downloads),
    "direct_download_successes": sum(1 for x in direct_downloads if x.get("ok")),
    "priority_targets": PRIORITY_DOWNLOAD_TARGETS,
    "manual_fluxnet_files_now": [str(p) for p in DEST.glob("*")],
    "next_action": "rerun stage1b6aq repair after any successful downloads",
}
(TAB / "STAGE1B6AR_CLI_FETCH_MISSING_project_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

report = []
report.append("# Stage 1B.6AR CLI fetch missing project data")
report.append("")
report.append("## Decision")
report.append("```json")
report.append(json.dumps(decision, indent=2))
report.append("```")
report.append("")
report.append("## Direct download attempts")
report.append("```text")
if direct_downloads:
    report.append(pd.DataFrame(direct_downloads).to_string(index=False))
else:
    report.append("No direct download attempts found.")
report.append("```")
report.append("")
report.append("## Site hit preview")
report.append("```text")
if site_hits:
    report.append(pd.DataFrame(site_hits).head(80).to_string(index=False))
else:
    report.append("No site hits.")
report.append("```")
(TXT / "STAGE1B6AR_CLI_FETCH_MISSING_project_REPORT.md").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
