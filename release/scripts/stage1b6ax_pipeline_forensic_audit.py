from pathlib import Path
import re
import json
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6ax_pipeline_forensic_audit"
TAB = OUT / "tables"
TXT = OUT / "text"
OUT.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

KEY_TERMS = [
    "latent_slope_change",
    "c4_fraction",
    "mean_vpd",
    "baseline_vpd",
    "rooting_depth",
    "aridity",
    "mean_annual_temperature",
    "mean_annual_precipitation",
    "growing_season_mean_lai",
    "mean_soil_moisture",
    "dropna",
    "complete_case",
    "notna",
    "isna",
    "merge",
    "join",
    "inner",
    "left",
    "right",
    "crop",
    "grassland",
    "no_crop",
    "sahel",
    "ols",
    "formula",
    "r2",
    "aic",
    "bic",
    "vif",
    "bootstrap",
]

SCRIPT_EXTS = [".py", ".sh", ".R", ".ipynb"]

# ----------------------------
# 1. Inventory files
# ----------------------------

script_rows = []
data_rows = []

for p in ROOT.rglob("*"):
    if any(part.startswith(".") for part in p.relative_to(ROOT).parts):
        continue
    if p.is_file() and p.suffix in SCRIPT_EXTS:
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            txt = ""
        script_rows.append({
            "path": str(p.relative_to(ROOT)),
            "suffix": p.suffix,
            "size_bytes": p.stat().st_size,
            "n_lines": txt.count("\n") + 1,
            "mentions_key_terms": sum(term in txt for term in KEY_TERMS),
            "mentions_dropna": "dropna" in txt,
            "mentions_merge": "merge" in txt or ".join" in txt,
            "mentions_ols": "ols" in txt.lower() or "statsmodels" in txt.lower(),
        })

    if p.is_file() and p.suffix.lower() in [".csv", ".parquet", ".pq", ".json"]:
        data_rows.append({
            "path": str(p.relative_to(ROOT)),
            "suffix": p.suffix,
            "size_bytes": p.stat().st_size,
        })

pd.DataFrame(script_rows).sort_values(
    ["mentions_key_terms", "size_bytes"], ascending=[False, False]
).to_csv(TAB / "script_inventory_ranked.csv", index=False)

pd.DataFrame(data_rows).sort_values("size_bytes", ascending=False).to_csv(
    TAB / "data_inventory.csv", index=False
)

# ----------------------------
# 2. Grep key lines from scripts
# ----------------------------

grep_rows = []

for row in script_rows:
    p = ROOT / row["path"]
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except Exception:
        continue

    for i, line in enumerate(lines, start=1):
        low = line.lower()
        matched = [term for term in KEY_TERMS if term.lower() in low]
        if matched:
            grep_rows.append({
                "script": row["path"],
                "line": i,
                "matched_terms": "; ".join(matched),
                "text": line.strip()[:500],
            })

grep = pd.DataFrame(grep_rows)
grep.to_csv(TAB / "key_term_script_grep.csv", index=False)

# Make focused greps.
for label, terms in {
    "dropna_filtering": ["dropna", "complete_case", "notna", "isna"],
    "joins_merges": ["merge", "join", "inner", "left", "right"],
    "c4_creation": ["c4_fraction", "c4"],
    "vpd_response_creation": ["latent_slope_change", "mean_vpd", "baseline_vpd"],
    "modeling": ["ols", "formula", "r2", "aic", "bic", "statsmodels"],
    "landcover_crop_filters": ["crop", "grassland", "no_crop", "irrig"],
}.items():
    sub = grep[grep["matched_terms"].str.contains("|".join(terms), case=False, na=False)]
    sub.to_csv(TAB / f"grep_{label}.csv", index=False)

# ----------------------------
# 3. Inspect all CSV tables for columns / row counts
# ----------------------------

table_rows = []

def read_head_and_cols(p):
    try:
        if p.suffix.lower() == ".csv":
            d = pd.read_csv(p, nrows=5, low_memory=False)
        elif p.suffix.lower() in [".parquet", ".pq"]:
            d = pd.read_parquet(p)
            d = d.head(5)
        else:
            return None
        return d
    except Exception:
        return None

for row in data_rows:
    p = ROOT / row["path"]
    if p.suffix.lower() not in [".csv", ".parquet", ".pq"]:
        continue

    d_head = read_head_and_cols(p)
    if d_head is None:
        continue

    try:
        if p.suffix.lower() == ".csv":
            n = sum(1 for _ in open(p, errors="ignore")) - 1
        else:
            n = len(pd.read_parquet(p))
    except Exception:
        n = np.nan

    cols = list(d_head.columns)
    lowcols = [c.lower() for c in cols]
    table_rows.append({
        "path": row["path"],
        "n_rows_est": n,
        "n_cols": len(cols),
        "has_lat_lon": any(c in lowcols for c in ["lat", "latitude"]) and any(c in lowcols for c in ["lon", "longitude"]),
        "has_point_id": any("point" in c and "id" in c for c in lowcols),
        "has_c4_fraction": "c4_fraction" in lowcols,
        "has_latent_slope_change": "latent_slope_change" in lowcols,
        "has_mean_vpd": "mean_vpd" in lowcols,
        "has_rooting_depth": "rooting_depth" in lowcols,
        "has_crop": any("crop" in c for c in lowcols),
        "has_grassland": any("grass" in c for c in lowcols),
        "columns": "; ".join(cols[:120]),
    })

tables = pd.DataFrame(table_rows).sort_values(
    ["has_c4_fraction", "has_latent_slope_change", "has_mean_vpd", "n_rows_est"],
    ascending=[False, False, False, False]
)
tables.to_csv(TAB / "table_column_inventory_ranked.csv", index=False)

# ----------------------------
# 4. Find lineage-ish tables for point data
# ----------------------------

candidate_tables = tables[
    (tables["has_lat_lon"] | tables["has_point_id"]) &
    (
        tables["has_c4_fraction"] |
        tables["has_latent_slope_change"] |
        tables["has_mean_vpd"] |
        tables["has_crop"] |
        tables["has_grassland"]
    )
].copy()

candidate_tables.to_csv(TAB / "candidate_point_lineage_tables.csv", index=False)

lineage_rows = []
for _, r in candidate_tables.iterrows():
    p = ROOT / r["path"]
    try:
        df = pd.read_csv(p, low_memory=False) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
    except Exception as e:
        lineage_rows.append({"path": r["path"], "error": str(e)})
        continue

    out = {"path": r["path"], "n": len(df), "n_cols": len(df.columns)}
    for col in [
        "point_id", "lat", "lon",
        "c4_fraction", "latent_slope_change", "mean_vpd",
        "rooting_depth", "aridity",
        "mean_annual_temperature", "mean_annual_precipitation",
        "growing_season_mean_lai", "mean_soil_moisture",
    ]:
        if col in df.columns:
            out[f"{col}_nonmissing"] = int(df[col].replace([np.inf, -np.inf], np.nan).notna().sum())
            out[f"{col}_missing"] = int(df[col].replace([np.inf, -np.inf], np.nan).isna().sum())
        else:
            out[f"{col}_nonmissing"] = np.nan
            out[f"{col}_missing"] = np.nan

    if "lat" in df.columns and "lon" in df.columns:
        out["lat_min"] = df["lat"].min()
        out["lat_max"] = df["lat"].max()
        out["lon_min"] = df["lon"].min()
        out["lon_max"] = df["lon"].max()
        out["n_sahel_broad"] = int((df["lat"].between(10,20) & df["lon"].between(-20,40)).sum())

    lineage_rows.append(out)

pd.DataFrame(lineage_rows).to_csv(TAB / "candidate_point_lineage_missingness.csv", index=False)

# ----------------------------
# 5. Detect suspicious script patterns
# ----------------------------

suspicious_patterns = {
    "complete_case_all_numeric_or_all_cols": [
        r"\.dropna\(\s*\)",
        r"dropna\(\s*axis\s*=",
    ],
    "inner_join": [
        r"how\s*=\s*['\"]inner['\"]",
        r"\.merge\([^)]*\)",
    ],
    "same_sample_risk": [
        r"rsquared",
        r"\.aic",
        r"\.bic",
    ],
    "hard_region_filter": [
        r"lat.*between",
        r"lon.*between",
        r"sahel",
    ],
    "hard_crop_grass_filter": [
        r"crop",
        r"grassland",
        r"igbp",
    ],
}

suspicious_rows = []
for row in script_rows:
    p = ROOT / row["path"]
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except Exception:
        continue

    for i, line in enumerate(lines, start=1):
        for label, pats in suspicious_patterns.items():
            for pat in pats:
                if re.search(pat, line, flags=re.IGNORECASE):
                    suspicious_rows.append({
                        "issue_type": label,
                        "script": row["path"],
                        "line": i,
                        "pattern": pat,
                        "text": line.strip()[:500],
                    })

pd.DataFrame(suspicious_rows).to_csv(TAB / "suspicious_script_patterns.csv", index=False)

# ----------------------------
# 6. Make readable summary
# ----------------------------

def top_csv(path, n=20):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    d = pd.read_csv(p)
    return d.head(n).to_string(index=False)

memo = []
memo.append("Pipeline forensic audit")
memo.append("=" * 80)
memo.append("")
memo.append("Purpose:")
memo.append("Trace how the WUE/C4/VPD analysis tables are made and flag internal issues.")
memo.append("")
memo.append("Top scripts by key-term relevance:")
memo.append(top_csv("script_inventory_ranked.csv", 25))
memo.append("")
memo.append("Candidate point-lineage tables:")
memo.append(top_csv("candidate_point_lineage_missingness.csv", 30))
memo.append("")
memo.append("Suspicious script patterns:")
memo.append(top_csv("suspicious_script_patterns.csv", 60))
memo.append("")
memo.append("Important files written:")
for name in [
    "script_inventory_ranked.csv",
    "data_inventory.csv",
    "key_term_script_grep.csv",
    "grep_dropna_filtering.csv",
    "grep_joins_merges.csv",
    "grep_c4_creation.csv",
    "grep_vpd_response_creation.csv",
    "grep_modeling.csv",
    "grep_landcover_crop_filters.csv",
    "table_column_inventory_ranked.csv",
    "candidate_point_lineage_tables.csv",
    "candidate_point_lineage_missingness.csv",
    "suspicious_script_patterns.csv",
]:
    memo.append(f"- {TAB / name}")

(TXT / "READ_ME_pipeline_forensic_audit.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_pipeline_forensic_audit.txt'}")
