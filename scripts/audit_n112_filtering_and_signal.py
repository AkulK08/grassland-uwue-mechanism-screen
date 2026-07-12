from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results" / "n112_audit"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------

def read_any(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def find_candidate_tables():
    pats = [
        "results/**/*.csv",
        "data/processed/**/*.csv",
        "data/**/*.csv",
    ]
    hits = []
    for pat in pats:
        for p in ROOT.glob(pat):
            name = p.name.lower()
            if any(k in name for k in [
                "candidate", "c4", "trait", "flux", "reza", "full", "control",
                "point", "wue", "uwue", "mediation", "pathway"
            ]):
                hits.append(p)
    return sorted(set(hits), key=lambda x: str(x))

def norm_col(c):
    return c.lower().replace(" ", "_").replace("-", "_")

def find_col(df, candidates):
    lookup = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        if norm_col(cand) in lookup:
            return lookup[norm_col(cand)]
    for c in df.columns:
        nc = norm_col(c)
        if any(norm_col(cand) in nc for cand in candidates):
            return c
    return None

def safe_ols(df, y, xs):
    try:
        import statsmodels.api as sm
        d = df[[y] + xs].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < len(xs) + 5:
            return None
        X = sm.add_constant(d[xs], has_constant="add")
        model = sm.OLS(d[y], X).fit()
        rows = []
        for x in xs:
            rows.append({
                "model_y": y,
                "term": x,
                "n": int(model.nobs),
                "coef": model.params.get(x, np.nan),
                "p": model.pvalues.get(x, np.nan),
                "r2": model.rsquared,
                "aic": model.aic,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame([{
            "model_y": y,
            "term": "ERROR",
            "n": np.nan,
            "coef": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "aic": np.nan,
            "error": str(e),
        }])

# ---------- locate best input table ----------

tables = find_candidate_tables()
inventory = []

for p in tables:
    try:
        df0 = read_any(p)
        cols_norm = [norm_col(c) for c in df0.columns]
        score = 0
        for key in ["c4", "vpd", "wue", "uwue", "rooting", "aridity", "latitude", "longitude", "lat", "lon"]:
            if any(key in c for c in cols_norm):
                score += 1
        inventory.append({
            "path": str(p),
            "rows": len(df0),
            "cols": len(df0.columns),
            "score": score,
            "columns": "; ".join(df0.columns[:60]),
        })
    except Exception as e:
        inventory.append({
            "path": str(p),
            "rows": None,
            "cols": None,
            "score": -1,
            "columns": f"READ ERROR: {e}",
        })

inv = pd.DataFrame(inventory).sort_values(["score", "rows"], ascending=[False, False])
inv.to_csv(OUT / "candidate_input_tables_ranked.csv", index=False)

if inv.empty:
    raise SystemExit("No candidate CSV tables found.")

best_path = Path(inv.iloc[0]["path"])
df = read_any(best_path)

print("\nUsing input table:")
print(best_path)
print("rows:", len(df))
print("cols:", len(df.columns))
print("\nTop candidate tables written to:")
print(OUT / "candidate_input_tables_ranked.csv")

# ---------- identify columns ----------

lat_col = find_col(df, ["lat", "latitude"])
lon_col = find_col(df, ["lon", "longitude"])
site_col = find_col(df, ["site", "site_id", "point_id", "id"])
c4_col = find_col(df, ["c4_fraction", "c4", "c4_frac"])
vpd_col = find_col(df, ["baseline_vpd", "mean_vpd", "vpd"])
uwue_col = find_col(df, [
    "uwue_response", "uwue_latent_slope_change", "latent_slope_change",
    "slope_change", "wue_response", "response"
])

control_candidates = {
    "rooting_depth": ["rooting_depth", "root_depth"],
    "aridity": ["aridity", "aridity_index"],
    "mean_annual_temperature": ["mean_annual_temperature", "mat", "temperature"],
    "mean_annual_precipitation": ["mean_annual_precipitation", "map", "precipitation"],
    "soil_texture_pc1": ["soil_texture_pc1", "texture_pc1", "soil_texture"],
    "growing_season_mean_lai": ["growing_season_mean_lai", "mean_lai", "lai"],
    "mean_vpd": ["mean_vpd", "baseline_vpd", "vpd"],
    "mean_soil_moisture": ["mean_soil_moisture", "soil_moisture", "sm"],
}

controls = {}
for label, cands in control_candidates.items():
    col = find_col(df, cands)
    if col is not None:
        controls[label] = col

id_cols = [c for c in [site_col, lat_col, lon_col] if c is not None]

identified = {
    "input_table": str(best_path),
    "n_raw": len(df),
    "site_col": site_col,
    "lat_col": lat_col,
    "lon_col": lon_col,
    "c4_col": c4_col,
    "vpd_col": vpd_col,
    "uwue_col": uwue_col,
    "controls": controls,
}
(OUT / "identified_columns.json").write_text(json.dumps(identified, indent=2))

print("\nIdentified columns:")
print(json.dumps(identified, indent=2))

# ---------- missingness audit ----------

needed_signal = [x for x in [c4_col, vpd_col, uwue_col] if x is not None]
full_control_cols = needed_signal + list(dict.fromkeys(controls.values()))

missing_rows = []
for c in df.columns:
    missing_rows.append({
        "column": c,
        "nonmissing_n": int(df[c].replace([np.inf, -np.inf], np.nan).notna().sum()),
        "missing_n": int(df[c].replace([np.inf, -np.inf], np.nan).isna().sum()),
        "missing_rate": float(df[c].replace([np.inf, -np.inf], np.nan).isna().mean()),
    })

pd.DataFrame(missing_rows).sort_values("missing_n", ascending=False).to_csv(
    OUT / "all_column_missingness.csv", index=False
)

key_missing = pd.DataFrame([
    {
        "role": "signal/control",
        "column": c,
        "nonmissing_n": int(df[c].replace([np.inf, -np.inf], np.nan).notna().sum()),
        "missing_n": int(df[c].replace([np.inf, -np.inf], np.nan).isna().sum()),
        "missing_rate": float(df[c].replace([np.inf, -np.inf], np.nan).isna().mean()),
    }
    for c in full_control_cols
])
key_missing.to_csv(OUT / "key_variable_missingness.csv", index=False)

# ---------- sequential complete-case deletion audit ----------

seq = []
tmp = df.copy()
seq.append({"step": "raw", "required_column": "", "n_remaining": len(tmp), "n_lost_at_step": 0})

for c in full_control_cols:
    before = len(tmp)
    tmp = tmp[tmp[c].replace([np.inf, -np.inf], np.nan).notna()]
    after = len(tmp)
    seq.append({
        "step": f"require {c}",
        "required_column": c,
        "n_remaining": after,
        "n_lost_at_step": before - after,
    })

seq_df = pd.DataFrame(seq)
seq_df.to_csv(OUT / "sequential_filter_audit_original_order.csv", index=False)

# Also rank each variable by how much it alone would delete from the signal sample.
if needed_signal:
    signal_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed_signal)
else:
    signal_df = df.copy()

drop_impact = []
for c in list(dict.fromkeys(controls.values())):
    drop_impact.append({
        "control_column": c,
        "signal_sample_n_before_control": len(signal_df),
        "n_with_control_available": int(signal_df[c].replace([np.inf, -np.inf], np.nan).notna().sum()),
        "n_lost_if_required": int(signal_df[c].replace([np.inf, -np.inf], np.nan).isna().sum()),
        "lost_rate_if_required": float(signal_df[c].replace([np.inf, -np.inf], np.nan).isna().mean()),
    })

pd.DataFrame(drop_impact).sort_values("n_lost_if_required", ascending=False).to_csv(
    OUT / "control_variable_drop_impact_from_signal_sample.csv", index=False
)

# ---------- define samples that avoid unnecessary complete-case deletion ----------

sample_defs = {
    "signal_only": needed_signal,
    "c4_vpd_only": [x for x in [c4_col, vpd_col, uwue_col] if x is not None],
    "climate_core": [x for x in [
        c4_col, uwue_col, vpd_col,
        controls.get("aridity"),
        controls.get("mean_annual_temperature"),
        controls.get("mean_annual_precipitation"),
    ] if x is not None],
    "climate_plus_lai_sm": [x for x in [
        c4_col, uwue_col, vpd_col,
        controls.get("aridity"),
        controls.get("mean_annual_temperature"),
        controls.get("mean_annual_precipitation"),
        controls.get("growing_season_mean_lai"),
        controls.get("mean_soil_moisture"),
    ] if x is not None],
    "full_controls_including_rooting_depth": full_control_cols,
    "full_controls_EXCLUDING_rooting_depth": [
        c for c in full_control_cols
        if c != controls.get("rooting_depth")
    ],
    "full_controls_EXCLUDING_rooting_depth_and_soil_texture": [
        c for c in full_control_cols
        if c not in [controls.get("rooting_depth"), controls.get("soil_texture_pc1")]
    ],
}

sample_summary = []
for name, cols in sample_defs.items():
    cols = list(dict.fromkeys([c for c in cols if c is not None]))
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    sample_summary.append({
        "sample": name,
        "n": len(d),
        "required_columns": "; ".join(cols),
    })
    keep_cols = list(dict.fromkeys(id_cols + cols))
    if keep_cols:
        d[keep_cols].to_csv(OUT / f"points_{name}.csv", index=False)
    else:
        d.to_csv(OUT / f"points_{name}.csv", index=False)

pd.DataFrame(sample_summary).to_csv(OUT / "sample_size_by_control_set.csv", index=False)

# ---------- identify exactly which points are in n=112 vs dropped ----------

full_cols = sample_defs["full_controls_including_rooting_depth"]
full_cols = list(dict.fromkeys([c for c in full_cols if c is not None]))

lean_cols = sample_defs["full_controls_EXCLUDING_rooting_depth"]
lean_cols = list(dict.fromkeys([c for c in lean_cols if c is not None]))

full_mask = df.replace([np.inf, -np.inf], np.nan)[full_cols].notna().all(axis=1) if full_cols else pd.Series(True, index=df.index)
lean_mask = df.replace([np.inf, -np.inf], np.nan)[lean_cols].notna().all(axis=1) if lean_cols else pd.Series(True, index=df.index)

diagnostic = df.copy()
diagnostic["_row_index"] = diagnostic.index
diagnostic["_in_full_complete_case"] = full_mask
diagnostic["_in_lean_no_rooting_depth_complete_case"] = lean_mask

for c in full_cols:
    diagnostic[f"_missing_{c}"] = diagnostic[c].replace([np.inf, -np.inf], np.nan).isna()

front_cols = ["_row_index", "_in_full_complete_case", "_in_lean_no_rooting_depth_complete_case"] + id_cols
front_cols = list(dict.fromkeys([c for c in front_cols if c in diagnostic.columns]))
diagnostic = diagnostic[front_cols + [c for c in diagnostic.columns if c not in front_cols]]

diagnostic.to_csv(OUT / "every_point_full_vs_lean_membership_and_missingness.csv", index=False)

full_points = diagnostic[diagnostic["_in_full_complete_case"]]
dropped_by_full_but_kept_lean = diagnostic[
    (~diagnostic["_in_full_complete_case"]) &
    (diagnostic["_in_lean_no_rooting_depth_complete_case"])
]

full_points.to_csv(OUT / "EXACT_POINTS_IN_FULL_COMPLETE_CASE_n112_or_equivalent.csv", index=False)
dropped_by_full_but_kept_lean.to_csv(
    OUT / "POINTS_DROPPED_BY_FULL_BUT_RECOVERED_WITHOUT_ROOTING_DEPTH.csv",
    index=False
)

# ---------- signal/model checks with tiered controls ----------

model_rows = []

if c4_col and uwue_col:
    model_specs = {
        "M0_signal_C4_only": [c4_col],
        "M1_C4_plus_VPD": [c4_col, vpd_col],
        "M2_climate_core": [
            c4_col, vpd_col,
            controls.get("aridity"),
            controls.get("mean_annual_temperature"),
            controls.get("mean_annual_precipitation"),
        ],
        "M3_climate_plus_LAI_SM": [
            c4_col, vpd_col,
            controls.get("aridity"),
            controls.get("mean_annual_temperature"),
            controls.get("mean_annual_precipitation"),
            controls.get("growing_season_mean_lai"),
            controls.get("mean_soil_moisture"),
        ],
        "M4_full_no_rooting_depth": [
            c4_col, vpd_col,
            controls.get("aridity"),
            controls.get("mean_annual_temperature"),
            controls.get("mean_annual_precipitation"),
            controls.get("soil_texture_pc1"),
            controls.get("growing_season_mean_lai"),
            controls.get("mean_soil_moisture"),
        ],
        "M5_full_with_rooting_depth": [
            c4_col, vpd_col,
            controls.get("rooting_depth"),
            controls.get("aridity"),
            controls.get("mean_annual_temperature"),
            controls.get("mean_annual_precipitation"),
            controls.get("soil_texture_pc1"),
            controls.get("growing_season_mean_lai"),
            controls.get("mean_soil_moisture"),
        ],
    }

    all_model_tables = []
    for model_name, xs in model_specs.items():
        xs = list(dict.fromkeys([x for x in xs if x is not None and x != uwue_col]))
        tab = safe_ols(df, uwue_col, xs)
        if tab is not None:
            tab.insert(0, "model", model_name)
            all_model_tables.append(tab)

    if all_model_tables:
        pd.concat(all_model_tables, ignore_index=True).to_csv(
            OUT / "tiered_control_model_results.csv", index=False
        )

# ---------- simple mediation components, without forcing full controls ----------

med_tables = []

if c4_col and vpd_col:
    tab = safe_ols(df, vpd_col, [c4_col])
    if tab is not None:
        tab.insert(0, "pathway_model", "A_baseline_VPD_on_C4")
        med_tables.append(tab)

if uwue_col and vpd_col:
    tab = safe_ols(df, uwue_col, [vpd_col])
    if tab is not None:
        tab.insert(0, "pathway_model", "B_uWUE_on_VPD_only")
        med_tables.append(tab)

if uwue_col and c4_col:
    tab = safe_ols(df, uwue_col, [c4_col])
    if tab is not None:
        tab.insert(0, "pathway_model", "C_total_uWUE_on_C4")
        med_tables.append(tab)

if uwue_col and c4_col and vpd_col:
    tab = safe_ols(df, uwue_col, [c4_col, vpd_col])
    if tab is not None:
        tab.insert(0, "pathway_model", "D_direct_uWUE_on_C4_plus_VPD")
        med_tables.append(tab)

if med_tables:
    pd.concat(med_tables, ignore_index=True).to_csv(
        OUT / "basic_pathway_signal_tests_no_unnecessary_controls.csv",
        index=False
    )

# ---------- console summary ----------

print("\nWrote audit outputs to:")
print(OUT)

print("\nSample sizes by control set:")
print(pd.read_csv(OUT / "sample_size_by_control_set.csv").to_string(index=False))

print("\nVariables causing complete-case deletion from signal sample:")
drop_file = OUT / "control_variable_drop_impact_from_signal_sample.csv"
if drop_file.exists():
    print(pd.read_csv(drop_file).to_string(index=False))

print("\nSequential deletion audit:")
print(pd.read_csv(OUT / "sequential_filter_audit_original_order.csv").to_string(index=False))

print("\nExact points are in:")
print(OUT / "EXACT_POINTS_IN_FULL_COMPLETE_CASE_n112_or_equivalent.csv")

print("\nPoints recovered when rooting depth is not required are in:")
print(OUT / "POINTS_DROPPED_BY_FULL_BUT_RECOVERED_WITHOUT_ROOTING_DEPTH.csv")

print("\nModel results, if columns were found, are in:")
print(OUT / "tiered_control_model_results.csv")
print(OUT / "basic_pathway_signal_tests_no_unnecessary_controls.csv")
