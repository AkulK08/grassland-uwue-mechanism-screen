from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()
OUT = ROOT / "results" / "n112_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Edit/add paths here if your relevant table has a different name.
CANDIDATE_FILES = [
    ROOT / "results/stage1b6as_final_full_reza_rigor/tables/STAGE1B6AS_FINAL_FULL_REZA_RIGOR_DATA.csv",
    ROOT / "results/stage1b6as_final_full_reza_rigor/tables/STAGE1B6AS_FINAL_FULL_REZA_RIGOR_INPUT.csv",
    ROOT / "results/stage1b6au_strict_real_trait_flux/tables/Table_PRODUCT03dx_strict_real_candidate_trait_flux_theses.csv",
    ROOT / "results/stage1b6au_strict_real_trait_flux/tables/Table_PRODUCT03dw_strict_real_trait_to_flux_tests.csv",
    ROOT / "results/mediation/tables/grassland_like_no_crop_mediation_input.csv",
]

existing = [p for p in CANDIDATE_FILES if p.exists()]

if not existing:
    print("No candidate files found. Searching all results/*.csv for likely files...")
    existing = list((ROOT / "results").rglob("*.csv"))

print("\nCandidate files:")
for i, p in enumerate(existing):
    try:
        df0 = pd.read_csv(p, nrows=5)
        print(f"[{i}] {p}  shape_preview_cols={len(df0.columns)}")
    except Exception as e:
        print(f"[{i}] {p}  READ ERROR: {e}")

# Pick the file most likely to contain the complete-case n=112 analysis.
# If this chooses wrong, set CHOSEN_INDEX manually after seeing the printed list.
CHOSEN_INDEX = None

def score_file(p):
    try:
        d = pd.read_csv(p, nrows=100)
        cols = set(d.columns)
        score = 0
        keywords = [
            "c4", "vpd", "uwue", "wue", "rooting", "aridity",
            "temperature", "precipitation", "soil", "lai", "moisture",
            "crop", "lat", "lon"
        ]
        joined = " ".join(cols).lower()
        for k in keywords:
            if k in joined:
                score += 1
        return score
    except Exception:
        return -1

if CHOSEN_INDEX is None:
    scores = [(score_file(p), i, p) for i, p in enumerate(existing)]
    scores.sort(reverse=True)
    chosen = scores[0][2]
else:
    chosen = existing[CHOSEN_INDEX]

print(f"\nUsing file:\n{chosen}\n")
df = pd.read_csv(chosen)
print("Raw shape:", df.shape)
print("\nColumns:")
print(list(df.columns))

def find_col(possible_contains, required=False):
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}
    for pattern in possible_contains:
        matches = [c for c in cols if pattern.lower() in low[c]]
        if matches:
            return matches[0]
    if required:
        raise ValueError(f"Could not find column matching any of: {possible_contains}")
    return None

# Try to auto-detect columns. The script will skip any it cannot find.
candidate_controls = {
    "c4_fraction": ["c4_fraction", "c4_frac", "c4"],
    "baseline_vpd": ["baseline_vpd", "mean_vpd", "vpd"],
    "uwue_response": ["uwue_response", "uwue", "wue_response", "latent_slope", "slope_change", "response"],
    "rooting_depth": ["rooting_depth", "root"],
    "aridity": ["aridity"],
    "mean_annual_temperature": ["mean_annual_temperature", "mat", "temperature"],
    "mean_annual_precipitation": ["mean_annual_precipitation", "map", "precipitation"],
    "soil_texture_pc1": ["soil_texture_pc1", "texture_pc1", "soil_texture"],
    "growing_season_mean_lai": ["growing_season_mean_lai", "mean_lai", "lai"],
    "mean_soil_moisture": ["mean_soil_moisture", "soil_moisture", "sm"],
}

detected = {}
used_cols = set()
for logical, patterns in candidate_controls.items():
    for pattern in patterns:
        matches = [c for c in df.columns if pattern.lower() in c.lower() and c not in used_cols]
        if matches:
            detected[logical] = matches[0]
            used_cols.add(matches[0])
            break
    if logical not in detected:
        detected[logical] = None

print("\nDetected analysis columns:")
for k, v in detected.items():
    print(f"{k:30s} -> {v}")

analysis_cols = [v for v in detected.values() if v is not None]

# Optional common filters
filter_candidates = {
    "crop_clean": ["crop_clean", "no_crop", "crop"],
    "grassland_like": ["grassland_like", "grassland", "open_ecosystem"],
    "sahel": ["sahel"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude"],
}

detected_filters = {}
for logical, patterns in filter_candidates.items():
    detected_filters[logical] = None
    for pattern in patterns:
        matches = [c for c in df.columns if pattern.lower() in c.lower()]
        if matches:
            detected_filters[logical] = matches[0]
            break

print("\nDetected filter/geography columns:")
for k, v in detected_filters.items():
    print(f"{k:20s} -> {v}")

# Missingness summary
missing_rows = []
for c in analysis_cols:
    missing_rows.append({
        "column": c,
        "non_missing_n": int(df[c].notna().sum()),
        "missing_n": int(df[c].isna().sum()),
        "missing_pct": float(df[c].isna().mean()),
        "unique_n": int(df[c].nunique(dropna=True)),
    })

missing_df = pd.DataFrame(missing_rows).sort_values("non_missing_n")
missing_df.to_csv(OUT / "missingness_by_column.csv", index=False)

print("\nMissingness by analysis column:")
print(missing_df.to_string(index=False))

# Sequential complete-case loss
tmp = df.copy()
seq_rows = [{"step": "raw", "required_column": "", "n_remaining": len(tmp), "lost_at_step": 0}]

for c in analysis_cols:
    before = len(tmp)
    tmp = tmp[tmp[c].notna()].copy()
    after = len(tmp)
    seq_rows.append({
        "step": f"require {c}",
        "required_column": c,
        "n_remaining": after,
        "lost_at_step": before - after,
    })

seq_df = pd.DataFrame(seq_rows)
seq_df.to_csv(OUT / "sequential_complete_case_loss.csv", index=False)

print("\nSequential complete-case loss:")
print(seq_df.to_string(index=False))

# Order-independent contribution: how many rows are lost uniquely because of each variable?
full_cc_mask = df[analysis_cols].notna().all(axis=1)
full_cc_n = int(full_cc_mask.sum())

drop_rows = []
for c in analysis_cols:
    other_cols = [x for x in analysis_cols if x != c]
    other_cc = df[other_cols].notna().all(axis=1) if other_cols else pd.Series(True, index=df.index)
    recoverable_if_drop_c = int(other_cc.sum())
    lost_due_to_c_given_others = recoverable_if_drop_c - full_cc_n
    drop_rows.append({
        "dropped_column": c,
        "n_if_this_column_not_required": recoverable_if_drop_c,
        "gain_vs_full_complete_case": lost_due_to_c_given_others,
    })

drop_df = pd.DataFrame(drop_rows).sort_values("gain_vs_full_complete_case", ascending=False)
drop_df.to_csv(OUT / "leave_one_control_out_n_gain.csv", index=False)

print(f"\nFull complete-case n across detected analysis columns: {full_cc_n}")
print("\nHow much n increases if each column is NOT required:")
print(drop_df.to_string(index=False))

# Pairwise complete-case matrix
pair = pd.DataFrame(index=analysis_cols, columns=analysis_cols, dtype=int)
for a in analysis_cols:
    for b in analysis_cols:
        pair.loc[a, b] = int(df[[a, b]].notna().all(axis=1).sum())
pair.to_csv(OUT / "pairwise_complete_case_n.csv")

print("\nPairwise complete-case n matrix saved.")

# Geography / Sahel check
lat_col = detected_filters.get("lat")
lon_col = detected_filters.get("lon")

if lat_col and lon_col:
    broad_sahel = (
        df[lat_col].between(10, 20, inclusive="both")
        & df[lon_col].between(-20, 40, inclusive="both")
    )
    core_sahel = (
        df[lat_col].between(12, 18, inclusive="both")
        & df[lon_col].between(-17, 35, inclusive="both")
    )

    geo_rows = [
        {"subset": "all rows", "n": len(df)},
        {"subset": "broad_sahel_lat10_20_lon-20_40", "n": int(broad_sahel.sum())},
        {"subset": "core_sahel_lat12_18_lon-17_35", "n": int(core_sahel.sum())},
        {"subset": "full_complete_case", "n": full_cc_n},
        {"subset": "full_complete_case_AND_broad_sahel", "n": int((full_cc_mask & broad_sahel).sum())},
        {"subset": "full_complete_case_AND_core_sahel", "n": int((full_cc_mask & core_sahel).sum())},
    ]
    geo_df = pd.DataFrame(geo_rows)
    geo_df.to_csv(OUT / "sahel_geography_check.csv", index=False)

    print("\nSahel/geography check:")
    print(geo_df.to_string(index=False))
else:
    print("\nNo lat/lon columns detected, so Sahel check skipped.")

# Value counts for filters
for name, c in detected_filters.items():
    if c and c not in [lat_col, lon_col]:
        vc = df[c].value_counts(dropna=False).head(30)
        vc.to_csv(OUT / f"value_counts_{name}_{c}.csv")
        print(f"\nTop values for {name} column {c}:")
        print(vc.to_string())

# Save complete-case and incomplete rows for inspection
df.loc[full_cc_mask].to_csv(OUT / "full_complete_case_rows.csv", index=False)
df.loc[~full_cc_mask].to_csv(OUT / "incomplete_rows_excluded_from_complete_case.csv", index=False)

print("\nSaved audit outputs to:")
print(OUT)
print("\nKey files:")
print(" - missingness_by_column.csv")
print(" - sequential_complete_case_loss.csv")
print(" - leave_one_control_out_n_gain.csv")
print(" - pairwise_complete_case_n.csv")
print(" - sahel_geography_check.csv, if lat/lon detected")
print(" - full_complete_case_rows.csv")
print(" - incomplete_rows_excluded_from_complete_case.csv")
