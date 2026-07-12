from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6ay_c4_and_latent_response_deep_audit"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

TARGET_SCRIPTS = [
    ROOT / "scripts/stage1b6ai_project_final_lock_with_c4.py",
    ROOT / "scripts/phase8_latent_product_adjusted_response.py",
]

KEY_TABLE = ROOT / "results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"

# -----------------------------
# helpers
# -----------------------------

def read_text(path):
    return Path(path).read_text(errors="ignore").splitlines()

def extract_window(path, center_line, radius=25):
    lines = read_text(path)
    lo = max(1, center_line - radius)
    hi = min(len(lines), center_line + radius)
    return "\n".join(
        f"{i:05d}: {lines[i-1]}"
        for i in range(lo, hi + 1)
    )

def grep_script(path, terms):
    rows = []
    lines = read_text(path)
    for i, line in enumerate(lines, start=1):
        low = line.lower()
        hit = [t for t in terms if t.lower() in low]
        if hit:
            rows.append({
                "script": str(path.relative_to(ROOT)),
                "line": i,
                "matched": "; ".join(hit),
                "text": line.strip()[:700],
            })
    return rows

# -----------------------------
# 1. Extract code windows
# -----------------------------

windows = []

# Known important regions from earlier grep.
known_windows = {
    "stage1b6ai_C4_file_and_sampling": ("scripts/stage1b6ai_project_final_lock_with_c4.py", 512),
    "stage1b6ai_C4_raw_to_fraction": ("scripts/stage1b6ai_project_final_lock_with_c4.py", 642),
    "stage1b6ai_C4_model": ("scripts/stage1b6ai_project_final_lock_with_c4.py", 727),
    "phase8_response_aliasing": ("scripts/phase8_latent_product_adjusted_response.py", 793),
    "phase8_response_validation_1": ("scripts/phase8_latent_product_adjusted_response.py", 924),
    "phase8_response_validation_2": ("scripts/phase8_latent_product_adjusted_response.py", 1024),
    "phase8_response_correlation": ("scripts/phase8_latent_product_adjusted_response.py", 1602),
}

for label, (rel, line) in known_windows.items():
    path = ROOT / rel
    if path.exists():
        text = extract_window(path, line, radius=35)
        windows.append({"label": label, "script": rel, "center_line": line, "window_text": text})
        (TXT / f"CODE_WINDOW_{label}.txt").write_text(text)

pd.DataFrame(windows).to_csv(TAB / "important_code_windows_index.csv", index=False)

# -----------------------------
# 2. Grep target scripts deeply
# -----------------------------

terms = [
    "c4_fraction",
    "c4_fraction_raw",
    "C4_distribution_NUS",
    "sample_vals",
    "raster",
    "netcdf",
    "xarray",
    "lat",
    "lon",
    "nearest",
    "interp",
    "merge",
    "join",
    "dropna",
    "latent_slope_change",
    "latent_post_slope",
    "latent_satbreak_probability",
    "slope_change",
    "breakpoint",
    "weighted",
    "weight",
    "product",
    "gpp_product",
    "et_product",
    "median",
    "mean",
    "groupby",
    "corr",
]

grep_rows = []
for path in TARGET_SCRIPTS:
    if path.exists():
        grep_rows.extend(grep_script(path, terms))

grep = pd.DataFrame(grep_rows)
grep.to_csv(TAB / "target_script_deep_grep.csv", index=False)

# focused subsets
for name, pats in {
    "c4_sampling_join": ["c4_fraction", "c4_fraction_raw", "sample_vals", "C4_distribution_NUS", "raster", "xarray", "nearest", "interp"],
    "latent_response_creation": ["latent_slope_change", "latent_post_slope", "latent_satbreak_probability", "slope_change", "breakpoint"],
    "product_weighting": ["weighted", "weight", "product", "gpp_product", "et_product"],
    "dropna_merge": ["dropna", "merge", "join"],
}.items():
    sub = grep[grep["matched"].str.contains("|".join(pats), case=False, na=False)]
    sub.to_csv(TAB / f"grep_{name}.csv", index=False)

# -----------------------------
# 3. Inspect key point table
# -----------------------------

if not KEY_TABLE.exists():
    raise SystemExit(f"Missing key table: {KEY_TABLE}")

df = pd.read_csv(KEY_TABLE, low_memory=False).replace([np.inf, -np.inf], np.nan)

summary_rows = []
for col in df.columns:
    s = df[col]
    row = {
        "column": col,
        "dtype": str(s.dtype),
        "nonmissing": int(s.notna().sum()),
        "missing": int(s.isna().sum()),
        "missing_rate": float(s.isna().mean()),
        "unique": int(s.nunique(dropna=True)),
    }
    xnum = pd.to_numeric(s, errors="coerce")
    if xnum.notna().any() and not pd.api.types.is_bool_dtype(s):
        row.update({
            "min": float(xnum.min(skipna=True)),
            "p01": float(xnum.quantile(0.01)),
            "p05": float(xnum.quantile(0.05)),
            "median": float(xnum.median(skipna=True)),
            "p95": float(xnum.quantile(0.95)),
            "p99": float(xnum.quantile(0.99)),
            "max": float(xnum.max(skipna=True)),
        })
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(TAB / "key_table_column_profile.csv", index=False)

# -----------------------------
# 4. C4-specific diagnostics
# -----------------------------

c4_cols = [c for c in df.columns if "c4" in c.lower()]
c4_diag = []

for c in c4_cols:
    x = pd.to_numeric(df[c], errors="coerce")
    c4_diag.append({
        "column": c,
        "nonmissing": int(x.notna().sum()),
        "missing": int(x.isna().sum()),
        "min": float(x.min(skipna=True)) if x.notna().any() else np.nan,
        "median": float(x.median(skipna=True)) if x.notna().any() else np.nan,
        "max": float(x.max(skipna=True)) if x.notna().any() else np.nan,
        "n_gt_1": int((x > 1).sum()) if x.notna().any() else 0,
        "n_gt_100": int((x > 100).sum()) if x.notna().any() else 0,
        "n_lt_0": int((x < 0).sum()) if x.notna().any() else 0,
    })

pd.DataFrame(c4_diag).to_csv(TAB / "c4_column_diagnostics.csv", index=False)

if "c4_fraction" in df.columns:
    c4 = pd.to_numeric(df["c4_fraction"], errors="coerce")
    bins = [-np.inf, 0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, np.inf]
    labels = ["<0", "0", "0-0.01", "0.01-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-0.9", "0.9-1.0", ">1"]
    tmp = pd.DataFrame({"c4_fraction": c4})
    tmp["c4_bin"] = pd.cut(c4, bins=bins, labels=labels)
    tmp["c4_missing"] = c4.isna()
    tmp["c4_bin_plus_missing"] = tmp["c4_bin"].astype(str)
    tmp.loc[tmp["c4_missing"], "c4_bin_plus_missing"] = "MISSING"
    tmp["c4_bin_plus_missing"].value_counts(dropna=False).reset_index().rename(
        columns={"index": "bin", "c4_bin_plus_missing": "n"}
    ).to_csv(TAB / "c4_fraction_distribution_bins.csv", index=False)

# -----------------------------
# 5. Missing C4 geography / environment differences
# -----------------------------

if "c4_fraction" in df.columns:
    df["_missing_c4"] = pd.to_numeric(df["c4_fraction"], errors="coerce").isna()

    compare_cols = [
        "lat", "lon",
        "latent_slope_change",
        "mean_vpd",
        "aridity",
        "mean_annual_temperature",
        "mean_annual_precipitation",
        "growing_season_mean_lai",
        "mean_soil_moisture",
        "rooting_depth",
    ]
    compare_cols = [c for c in compare_cols if c in df.columns]

    rows = []
    for c in compare_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        for missing_flag, group in [(True, "missing_c4"), (False, "has_c4")]:
            vals = x[df["_missing_c4"].eq(missing_flag)]
            rows.append({
                "variable": c,
                "group": group,
                "n": int(vals.notna().sum()),
                "mean": float(vals.mean(skipna=True)) if vals.notna().any() else np.nan,
                "median": float(vals.median(skipna=True)) if vals.notna().any() else np.nan,
                "sd": float(vals.std(skipna=True)) if vals.notna().any() else np.nan,
                "min": float(vals.min(skipna=True)) if vals.notna().any() else np.nan,
                "max": float(vals.max(skipna=True)) if vals.notna().any() else np.nan,
            })

    pd.DataFrame(rows).to_csv(TAB / "missing_c4_vs_has_c4_environment_summary.csv", index=False)

    id_cols = [c for c in ["point_id", "lat", "lon"] if c in df.columns]
    keep = id_cols + compare_cols + c4_cols
    keep = list(dict.fromkeys([c for c in keep if c in df.columns]))
    df[df["_missing_c4"]][keep].to_csv(TAB / "EXACT_POINTS_missing_c4.csv", index=False)
    df[~df["_missing_c4"]][keep].to_csv(TAB / "EXACT_POINTS_has_c4.csv", index=False)

# -----------------------------
# 6. Latent response diagnostics
# -----------------------------

response_cols = [c for c in df.columns if any(k in c.lower() for k in [
    "latent",
    "slope_change",
    "post_slope",
    "satbreak",
    "threshold",
    "breakpoint",
    "response",
])]
pd.DataFrame({"response_like_column": response_cols}).to_csv(TAB / "response_like_columns.csv", index=False)

resp_diag = []
for c in response_cols:
    x = pd.to_numeric(df[c], errors="coerce")
    resp_diag.append({
        "column": c,
        "nonmissing": int(x.notna().sum()),
        "missing": int(x.isna().sum()),
        "unique": int(x.nunique(dropna=True)),
        "min": float(x.min(skipna=True)) if x.notna().any() else np.nan,
        "median": float(x.median(skipna=True)) if x.notna().any() else np.nan,
        "max": float(x.max(skipna=True)) if x.notna().any() else np.nan,
        "mean": float(x.mean(skipna=True)) if x.notna().any() else np.nan,
        "sd": float(x.std(skipna=True)) if x.notna().any() else np.nan,
    })

pd.DataFrame(resp_diag).to_csv(TAB / "response_column_diagnostics.csv", index=False)

# Pairwise correlations between response-like variables.
num_resp = []
for c in response_cols:
    x = pd.to_numeric(df[c], errors="coerce")
    if x.notna().sum() >= 20 and x.nunique(dropna=True) > 1:
        num_resp.append(c)

corr_rows = []
for i, a in enumerate(num_resp):
    for b in num_resp[i+1:]:
        ok = pd.to_numeric(df[a], errors="coerce").notna() & pd.to_numeric(df[b], errors="coerce").notna()
        if ok.sum() >= 20:
            corr_rows.append({
                "x": a,
                "y": b,
                "n": int(ok.sum()),
                "pearson": float(pd.to_numeric(df.loc[ok,a], errors="coerce").corr(pd.to_numeric(df.loc[ok,b], errors="coerce"))),
                "spearman": float(pd.to_numeric(df.loc[ok,a], errors="coerce").corr(pd.to_numeric(df.loc[ok,b], errors="coerce"), method="spearman")),
            })

pd.DataFrame(corr_rows).sort_values("pearson", ascending=False).to_csv(
    TAB / "response_like_pairwise_correlations.csv",
    index=False
)

# -----------------------------
# 7. Inspect possible source tables from phase8 outputs
# -----------------------------

phase8_tables = []
for p in (ROOT / "results").rglob("*.csv"):
    rel = str(p.relative_to(ROOT))
    low = rel.lower()
    if any(k in low for k in ["phase8", "latent", "product_adjusted", "slope_change"]):
        try:
            head = pd.read_csv(p, nrows=5, low_memory=False)
            n = sum(1 for _ in open(p, errors="ignore")) - 1
            phase8_tables.append({
                "path": rel,
                "n": n,
                "n_cols": len(head.columns),
                "has_point_id": "point_id" in head.columns,
                "has_latent_slope_change": "latent_slope_change" in head.columns,
                "has_gpp_product": "gpp_product" in head.columns,
                "has_et_product": "et_product" in head.columns,
                "columns": "; ".join(head.columns[:80]),
            })
        except Exception as e:
            phase8_tables.append({
                "path": rel,
                "error": str(e),
            })

pd.DataFrame(phase8_tables).sort_values(
    ["has_latent_slope_change", "n"], ascending=[False, False]
).to_csv(TAB / "phase8_latent_source_tables_inventory.csv", index=False)

# -----------------------------
# 8. Memo
# -----------------------------

def show_csv(name, n=30):
    p = TAB / name
    if not p.exists():
        return "MISSING"
    d = pd.read_csv(p)
    return d.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6AY: C4 and latent-response deep audit")
memo.append("=" * 80)
memo.append("")
memo.append(f"Key table: {KEY_TABLE}")
memo.append(f"Rows: {len(df)}")
memo.append("")
memo.append("C4 diagnostics:")
memo.append(show_csv("c4_column_diagnostics.csv", 20))
memo.append("")
memo.append("C4 distribution bins:")
memo.append(show_csv("c4_fraction_distribution_bins.csv", 20))
memo.append("")
memo.append("Missing C4 vs has C4 environmental/geographic summary:")
memo.append(show_csv("missing_c4_vs_has_c4_environment_summary.csv", 40))
memo.append("")
memo.append("Response column diagnostics:")
memo.append(show_csv("response_column_diagnostics.csv", 40))
memo.append("")
memo.append("Response-like pairwise correlations:")
memo.append(show_csv("response_like_pairwise_correlations.csv", 40))
memo.append("")
memo.append("Phase8 latent source tables:")
memo.append(show_csv("phase8_latent_source_tables_inventory.csv", 30))
memo.append("")
memo.append("Important code windows were written as CODE_WINDOW_*.txt in:")
memo.append(str(TXT))
memo.append("")
memo.append("Most important files:")
for name in [
    "important_code_windows_index.csv",
    "target_script_deep_grep.csv",
    "grep_c4_sampling_join.csv",
    "grep_latent_response_creation.csv",
    "grep_product_weighting.csv",
    "grep_dropna_merge.csv",
    "key_table_column_profile.csv",
    "c4_column_diagnostics.csv",
    "c4_fraction_distribution_bins.csv",
    "missing_c4_vs_has_c4_environment_summary.csv",
    "EXACT_POINTS_missing_c4.csv",
    "response_column_diagnostics.csv",
    "response_like_pairwise_correlations.csv",
    "phase8_latent_source_tables_inventory.csv",
]:
    memo.append(f"- {TAB / name}")

(TXT / "READ_ME_c4_and_latent_response_deep_audit.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_c4_and_latent_response_deep_audit.txt'}")
