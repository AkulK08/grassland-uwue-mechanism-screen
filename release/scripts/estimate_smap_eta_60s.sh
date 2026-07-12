#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

echo "===== SMAP ETA estimator ====="
echo "This will watch progress for 60 seconds and estimate remaining time if measurable."
echo ""

python - <<'PY'
from pathlib import Path
from glob import glob
import os, time, math
import pandas as pd

ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")

TARGET_ROWS = None
TARGET_RAW_FILES = None

# Best target estimate: expected point-date pairs from existing local GEE data, 2015-2024.
gee_files = sorted(glob(str(ROOT / "data/raw/gee/wue_timeseries_*.csv")))
pairs = set()
dates = set()
points = set()

for f in gee_files:
    try:
        df = pd.read_csv(f, usecols=["point_id", "date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].dt.year >= 2015]
        for pid, dt in zip(df["point_id"].astype(str), df["date"].dt.strftime("%Y-%m-%d")):
            if dt != "NaT":
                pairs.add((pid, dt))
                points.add(pid)
                dates.add(dt)
    except Exception:
        pass

if pairs:
    TARGET_ROWS = len(pairs)
else:
    # fallback: 2015-2024 = 10 years × 46 eight-day periods × 216 points
    TARGET_ROWS = 10 * 46 * 216

# Expected 8-day SMAP periods: 2015-2024 = about 460.
TARGET_RAW_FILES = 10 * 46

paths = {
    "matched": ROOT / "data/processed/smap_era5_matched_points.csv",
    "comparison": ROOT / "results/stress/smap_era5_comparison.csv",
    "manifest": ROOT / "data/raw/smap_l4/smap_l4_8day_download_manifest.csv",
}

def file_rows(p: Path):
    if not p.exists():
        return 0
    try:
        # Fast enough for this scale.
        return max(sum(1 for _ in open(p, "rb")) - 1, 0)
    except Exception:
        return 0

def raw_smap_files():
    d = ROOT / "data/raw/smap_l4"
    if not d.exists():
        return 0
    exts = ["*.h5", "*.hdf5", "*.nc", "*.nc4", "*.he5"]
    files = []
    for e in exts:
        files.extend(d.glob(e))
    return len(files)

def total_smap_bytes():
    d = ROOT / "data/raw/smap_l4"
    if not d.exists():
        return 0
    total = 0
    for p in d.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total

def snapshot():
    matched_rows = file_rows(paths["matched"])
    manifest_rows = file_rows(paths["manifest"])
    comparison_rows = file_rows(paths["comparison"])
    raw_files = raw_smap_files()
    raw_bytes = total_smap_bytes()
    return {
        "matched_rows": matched_rows,
        "manifest_rows": manifest_rows,
        "comparison_rows": comparison_rows,
        "raw_files": raw_files,
        "raw_bytes": raw_bytes,
        "time": time.time(),
    }

def fmt_seconds(sec):
    if sec is None or not math.isfinite(sec) or sec < 0:
        return "unknown"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def pct(a, b):
    if not b:
        return "unknown"
    return f"{100*a/b:.2f}%"

print("Target estimate:")
print(f"  expected matched point-date rows: {TARGET_ROWS}")
print(f"  expected raw 8-day SMAP files:    {TARGET_RAW_FILES}")
print("")
print("Taking snapshot 1...")
s1 = snapshot()
print(s1)
print("")
print("Waiting 60 seconds...")
time.sleep(60)
print("")
print("Taking snapshot 2...")
s2 = snapshot()
print(s2)
print("")

dt = s2["time"] - s1["time"]

candidates = []

# Best ETA: matched rows against expected point-date rows.
delta_rows = s2["matched_rows"] - s1["matched_rows"]
if s2["matched_rows"] > 0 and TARGET_ROWS:
    rate = delta_rows / dt if delta_rows > 0 else None
    if rate:
        remaining = max(TARGET_ROWS - s2["matched_rows"], 0)
        candidates.append(("matched_rows", s2["matched_rows"], TARGET_ROWS, rate, remaining / rate))

# Second best: raw SMAP files against 460 expected windows.
delta_files = s2["raw_files"] - s1["raw_files"]
if s2["raw_files"] > 0 and TARGET_RAW_FILES:
    rate = delta_files / dt if delta_files > 0 else None
    if rate:
        remaining = max(TARGET_RAW_FILES - s2["raw_files"], 0)
        candidates.append(("raw_smap_files", s2["raw_files"], TARGET_RAW_FILES, rate, remaining / rate))

# Third: manifest rows if it is being written.
delta_manifest = s2["manifest_rows"] - s1["manifest_rows"]
if s2["manifest_rows"] > 0 and TARGET_RAW_FILES:
    rate = delta_manifest / dt if delta_manifest > 0 else None
    if rate:
        remaining = max(TARGET_RAW_FILES - s2["manifest_rows"], 0)
        candidates.append(("manifest_rows", s2["manifest_rows"], TARGET_RAW_FILES, rate, remaining / rate))

print("===== ETA RESULT =====")

if candidates:
    name, current, target, rate, eta = candidates[0]
    print(f"progress metric used: {name}")
    print(f"current progress:     {current} / {target} ({pct(current, target)})")
    print(f"rate over 60 sec:     {rate:.4f} units/sec")
    print(f"estimated time left:  {fmt_seconds(eta)}")
else:
    print("ETA unavailable from this 60-second window.")
    print("Reason: no measurable increase in matched rows, raw SMAP files, or manifest rows.")
    print("")
    print("This usually means one of these:")
    print("  1. the task is still running but in a long step with no file writes yet;")
    print("  2. the task is downloading/processing one large file;")
    print("  3. the task is stuck;")
    print("  4. the output paths are different from the expected ones.")
    print("")
    print("Check the process/log below.")

print("")
print("===== CURRENT SMAP FILES =====")
for p in paths.values():
    print(str(p), "exists=", p.exists(), "rows=", file_rows(p))
print("raw_smap_files:", raw_smap_files())
print("raw_smap_bytes:", total_smap_bytes())
PY

echo ""
echo "===== SMAP process currently running ====="
ps -axo pid,lstart,etime,%cpu,%mem,command | grep -E "run_true_smap_validation_8day|smap" | grep -v grep || echo "No SMAP process found."

echo ""
echo "===== Latest SMAP log if any ====="
LOG=$(ls -t logs/*smap* logs/*validation* 2>/dev/null | head -1 || true)
if [ -n "${LOG:-}" ]; then
  echo "LOG: $LOG"
  tail -60 "$LOG"
else
  echo "No SMAP log found."
fi
