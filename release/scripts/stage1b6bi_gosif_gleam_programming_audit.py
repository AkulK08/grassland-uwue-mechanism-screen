from pathlib import Path
import re
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bi_gosif_gleam_programming_audit"
TAB = OUT / "tables"
TXT = OUT / "text"
for p in [TAB, TXT]:
    p.mkdir(parents=True, exist_ok=True)

POINT = ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv"
OBS = ROOT / "results/trait_framework/phase8/table_latent_model_observations.csv"
LC = ROOT / "results/stage1b6be_FULL_STRICT_lai_artifact_screen/tables/POINT_LEVEL_LANDCOVER_CROPLAND_FLAGS.csv"

if not POINT.exists():
    raise SystemExit(f"Missing point table: {POINT}")
if not OBS.exists():
    raise SystemExit(f"Missing observation table: {OBS}")

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def first_existing(cols, *names):
    for n in names:
        if n in cols:
            return n
    low = {norm(c): c for c in cols}
    for n in names:
        if norm(n) in low:
            return low[norm(n)]
    return None

def zscore(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd

def fit(data, formula, focal):
    toks = sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula)) - {"C", "I", "Q"})
    use = data[toks].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(25, len(toks) + 8):
        return {
            "status": "N_TOO_SMALL",
            "n": len(use),
            "coef": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "aic": np.nan,
        }
    try:
        m = smf.ols(formula, data=use).fit(cov_type="HC3")
        ci = m.conf_int()
        return {
            "status": "FIT_OK",
            "n": int(m.nobs),
            "coef": float(m.params.get(focal, np.nan)),
            "se_hc3": float(m.bse.get(focal, np.nan)),
            "p": float(m.pvalues.get(focal, np.nan)),
            "ci_low": float(ci.loc[focal, 0]) if focal in ci.index else np.nan,
            "ci_high": float(ci.loc[focal, 1]) if focal in ci.index else np.nan,
            "r2": float(m.rsquared),
            "aic": float(m.aic),
        }
    except Exception as e:
        return {
            "status": f"FIT_FAIL: {e}",
            "n": len(use),
            "coef": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "aic": np.nan,
        }

# --------------------------
# Read point table
# --------------------------

raw = pd.read_csv(POINT, low_memory=False).replace([np.inf, -np.inf], np.nan)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()
cols = list(raw.columns)

src = {
    "point_id": first_existing(cols, "point_id"),
    "latent_y": first_existing(cols, "latent_slope_change"),
    "lai": first_existing(cols, "growing_season_mean_lai", "mean_lai"),
    "vpd": first_existing(cols, "mean_vpd", "mean_obs_vpd"),
    "mat": first_existing(cols, "mean_annual_temperature", "mean_temperature"),
    "map": first_existing(cols, "mean_annual_precipitation", "mean_precipitation"),
    "arid": first_existing(cols, "aridity"),
    "sm": first_existing(cols, "mean_soil_moisture", "mean_obs_soil_moisture"),
    "lat": first_existing(cols, "lat", "latitude"),
    "lon": first_existing(cols, "lon", "longitude"),
    "sand": first_existing(cols, "soil_sand", "sand"),
    "clay": first_existing(cols, "soil_clay", "clay"),
    "silt": first_existing(cols, "soil_silt", "silt"),
}

missing = [k for k in ["point_id", "latent_y", "lai", "vpd", "mat", "map", "arid", "sm", "lat", "lon"] if src[k] is None]
if missing:
    raise SystemExit(f"Missing point variables: {missing}")

d = pd.DataFrame()
for k, c in src.items():
    if c is None:
        continue
    d[k] = raw[c].astype(str) if k == "point_id" else pd.to_numeric(raw[c], errors="coerce")

# soil texture PC1
if all(c in d.columns for c in ["sand", "clay", "silt"]):
    soil = d[["sand", "clay", "silt"]].apply(zscore)
    use = soil.dropna()
    pc1 = pd.Series(np.nan, index=d.index)
    if len(use) >= 40:
        X = use.values
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        scores = X @ vt[0, :]
        pc1.loc[use.index] = scores
        if pc1.corr(d["clay"]) < 0:
            pc1 = -pc1
        d["soil_texture_pc1"] = pc1

if LC.exists():
    lc = pd.read_csv(LC, low_memory=False)
    if "point_id" in lc.columns:
        keep = ["point_id"]
        for c in ["any_cropland_managed_irrigation_flag", "any_natural_grassland_indicator"]:
            if c in lc.columns:
                keep.append(c)
        lc = lc[keep].copy()
        lc["point_id"] = lc["point_id"].astype(str)
        lc = lc.groupby("point_id").first().reset_index()
        d = d.merge(lc, on="point_id", how="left")

if "any_cropland_managed_irrigation_flag" not in d.columns:
    d["any_cropland_managed_irrigation_flag"] = False
if "any_natural_grassland_indicator" not in d.columns:
    d["any_natural_grassland_indicator"] = True

d["any_cropland_managed_irrigation_flag"] = d["any_cropland_managed_irrigation_flag"].fillna(False).astype(bool)
d["any_natural_grassland_indicator"] = d["any_natural_grassland_indicator"].fillna(False).astype(bool)
d["abs_lat"] = d["lat"].abs()
d["sahel_broad"] = d["lat"].between(10, 20) & d["lon"].between(-20, 40)

for c in list(d.columns):
    if c not in ["point_id", "any_cropland_managed_irrigation_flag", "any_natural_grassland_indicator", "sahel_broad"]:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c + "_z"] = zscore(d[c])

controls = [c for c in ["vpd_z", "arid_z", "mat_z", "map_z", "sm_z", "soil_texture_pc1_z", "lat_z", "lon_z"] if c in d.columns]
controls_no_mat = [c for c in controls if c != "mat_z"]

# --------------------------
# Read observations and audit product filter
# --------------------------

obs = pd.read_csv(OBS, low_memory=False).replace([np.inf, -np.inf], np.nan)
obs = obs.loc[:, ~obs.columns.duplicated()].copy()
oc = list(obs.columns)

point_col = first_existing(oc, "point_id")
gpp_col = first_existing(oc, "gpp_product", "gpp")
et_col = first_existing(oc, "et_product", "et")
slope_col = first_existing(oc, "slope_change", "latent_slope_change")

if point_col is None or gpp_col is None or et_col is None or slope_col is None:
    raise SystemExit(f"Missing observation cols: point={point_col}, gpp={gpp_col}, et={et_col}, slope={slope_col}")

obs[point_col] = obs[point_col].astype(str)
obs[gpp_col] = obs[gpp_col].astype(str)
obs[et_col] = obs[et_col].astype(str)
obs[slope_col] = pd.to_numeric(obs[slope_col], errors="coerce")

combos = obs.groupby([gpp_col, et_col]).size().reset_index(name="n_rows")
combos.to_csv(TAB / "AUDIT_product_combo_counts.csv", index=False)

gg = obs[
    obs[gpp_col].str.lower().str.contains("gosif|sif", regex=True, na=False)
    & obs[et_col].str.lower().str.contains("gleam", regex=True, na=False)
].copy()

gg.to_csv(TAB / "AUDIT_raw_gosif_gleam_rows_HEAD.csv", index=False)

# --------------------------
# Hidden dimension audit
# --------------------------

hidden_rows = []
for c in gg.columns:
    if c in [point_col, gpp_col, et_col, slope_col]:
        continue
    nun = gg[c].nunique(dropna=True)
    if 1 < nun <= 30:
        vals = gg[c].dropna().astype(str).value_counts().head(20).to_dict()
        hidden_rows.append({
            "column": c,
            "n_unique": int(nun),
            "top_values": json.dumps(vals),
        })

hidden = pd.DataFrame(hidden_rows).sort_values("n_unique")
hidden.to_csv(TAB / "AUDIT_possible_hidden_definition_columns.csv", index=False)

per_point = gg.groupby(point_col).size().reset_index(name="n_gosif_gleam_rows_per_point")
per_point.to_csv(TAB / "AUDIT_gosif_gleam_rows_per_point.csv", index=False)

# --------------------------
# Build point outcomes: mean, median, winsorized mean
# --------------------------

def winsor_mean(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    lo, hi = np.nanpercentile(x, [1, 99])
    return np.clip(x, lo, hi).mean()

point_out = gg.groupby(point_col)[slope_col].agg(
    gosif_gleam_mean="mean",
    gosif_gleam_median="median",
    gosif_gleam_sd="std",
    gosif_gleam_n="size",
    gosif_gleam_min="min",
    gosif_gleam_max="max",
).reset_index().rename(columns={point_col: "point_id"})

point_win = gg.groupby(point_col)[slope_col].apply(winsor_mean).reset_index().rename(
    columns={point_col: "point_id", slope_col: "gosif_gleam_winsor_mean"}
)

point_out = point_out.merge(point_win, on="point_id", how="left")
dd = d.merge(point_out, on="point_id", how="left")

for c in ["gosif_gleam_mean", "gosif_gleam_median", "gosif_gleam_winsor_mean", "latent_y"]:
    dd[c + "_z_global"] = zscore(dd[c])

# --------------------------
# Test mean vs median vs winsorized, global vs within-mask scaling
# --------------------------

masks = {
    "all_available": pd.Series(True, index=dd.index),
    "cropland_clean": ~dd["any_cropland_managed_irrigation_flag"],
    "no_broad_sahel": ~dd["sahel_broad"],
    "cropland_clean_no_broad_sahel": (~dd["any_cropland_managed_irrigation_flag"]) & (~dd["sahel_broad"]),
    "warm_mat_gt_2p08C": dd["mat"] > 2.08,
    "abs_lat_le_48": dd["abs_lat"] <= 48,
}

rows = []

for outcome_base in ["gosif_gleam_mean", "gosif_gleam_median", "gosif_gleam_winsor_mean"]:
    for scale_mode in ["global_z", "within_mask_z"]:
        for mask_name, mask in masks.items():
            sub = dd.loc[mask].copy()

            if scale_mode == "global_z":
                ycol = outcome_base + "_z_global"
            else:
                ycol = outcome_base + "_z_within"
                sub[ycol] = zscore(sub[outcome_base])

            full = f"{ycol} ~ lai_z + " + " + ".join(controls)
            red = f"{ycol} ~ " + " + ".join(controls)
            r = fit(sub, full, "lai_z")
            r.update({
                "mask": mask_name,
                "outcome_aggregation": outcome_base,
                "scale_mode": scale_mode,
                "model": "LAI_main_full_controls",
            })
            rows.append(r)

            if len(controls_no_mat):
                full_int = f"{ycol} ~ lai_z * mat_z + " + " + ".join(controls_no_mat)
                r2 = fit(sub, full_int, "lai_z:mat_z")
                r2.update({
                    "mask": mask_name,
                    "outcome_aggregation": outcome_base,
                    "scale_mode": scale_mode,
                    "model": "LAI_x_MAT_full_controls",
                })
                rows.append(r2)

pd.DataFrame(rows).to_csv(TAB / "AUDIT_gosif_gleam_mean_median_winsor_scaling_tests.csv", index=False)

# --------------------------
# Stratify by hidden definition columns
# --------------------------

strat_rows = []

candidate_cols = hidden["column"].tolist() if len(hidden) else []
# Keep only plausible experimental-definition columns, not coordinates/diagnostics.
priority_patterns = [
    "stress", "metric", "season", "growing", "response", "window", "threshold",
    "definition", "model", "variant", "scenario", "method", "time", "bin"
]
candidate_cols = [
    c for c in candidate_cols
    if any(p in norm(c) for p in priority_patterns)
]

for dim in candidate_cols:
    for val, gsub in gg.groupby(dim, dropna=True):
        if len(gsub) < 500:
            continue
        point_dim = (
            gsub.dropna(subset=[point_col, slope_col])
                .groupby(point_col)[slope_col]
                .mean()
                .reset_index()
                .rename(columns={point_col: "point_id", slope_col: "y_dim"})
        )
        tmp = d.merge(point_dim, on="point_id", how="left")
        tmp["y_dim_z"] = zscore(tmp["y_dim"])

        full = "y_dim_z ~ lai_z + " + " + ".join(controls)
        r = fit(tmp, full, "lai_z")
        r.update({
            "dimension_column": dim,
            "dimension_value": str(val),
            "n_obs_rows": len(gsub),
            "n_point_rows": point_dim["point_id"].nunique(),
            "model": "LAI_main_full_controls_by_hidden_dimension",
        })
        strat_rows.append(r)

        if len(controls_no_mat):
            full_int = "y_dim_z ~ lai_z * mat_z + " + " + ".join(controls_no_mat)
            r2 = fit(tmp, full_int, "lai_z:mat_z")
            r2.update({
                "dimension_column": dim,
                "dimension_value": str(val),
                "n_obs_rows": len(gsub),
                "n_point_rows": point_dim["point_id"].nunique(),
                "model": "LAI_x_MAT_full_controls_by_hidden_dimension",
            })
            strat_rows.append(r2)

if strat_rows:
    pd.DataFrame(strat_rows).sort_values(["dimension_column", "model", "p"]).to_csv(
        TAB / "AUDIT_gosif_gleam_by_hidden_definition_subsets.csv", index=False
    )
else:
    pd.DataFrame([{"status": "NO_USABLE_HIDDEN_DEFINITION_SUBSETS_FOUND"}]).to_csv(
        TAB / "AUDIT_gosif_gleam_by_hidden_definition_subsets.csv", index=False
    )

# --------------------------
# Outlier audit
# --------------------------

outlier_rows = []
base = dd.copy()
base["y"] = base["gosif_gleam_mean_z_global"]

for mask_name, mask in masks.items():
    sub = base.loc[mask].copy()
    zabs = sub["y"].abs()
    for rule, keep in {
        "all": pd.Series(True, index=sub.index),
        "drop_abs_z_gt_3": zabs <= 3,
        "drop_abs_z_gt_2p5": zabs <= 2.5,
        "drop_top_1pct_abs_y": zabs <= zabs.quantile(0.99),
        "drop_top_5pct_abs_y": zabs <= zabs.quantile(0.95),
    }.items():
        ss = sub.loc[keep].copy()
        full = "y ~ lai_z + " + " + ".join(controls)
        r = fit(ss, full, "lai_z")
        r.update({
            "mask": mask_name,
            "outlier_rule": rule,
            "model": "LAI_main_full_controls",
            "n_dropped": int(len(sub) - len(ss)),
        })
        outlier_rows.append(r)

pd.DataFrame(outlier_rows).to_csv(TAB / "AUDIT_gosif_gleam_outlier_sensitivity.csv", index=False)

# --------------------------
# Memo
# --------------------------

def show(path, n=60):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    x = pd.read_csv(p)
    if len(x) == 0:
        return "EMPTY"
    return x.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6BI GOSIF x GLEAM programming / interpretation audit")
memo.append("=" * 90)
memo.append("")
memo.append("1. Product combo counts")
memo.append(show("AUDIT_product_combo_counts.csv", 30))
memo.append("")
memo.append("2. Possible hidden definition columns inside GOSIF x GLEAM rows")
memo.append(show("AUDIT_possible_hidden_definition_columns.csv", 80))
memo.append("")
memo.append("3. Rows per point")
x = pd.read_csv(TAB / "AUDIT_gosif_gleam_rows_per_point.csv")
memo.append(x["n_gosif_gleam_rows_per_point"].describe().to_string())
memo.append("")
memo.append("4. Main audit: mean vs median vs winsorized; global vs within-mask z-scoring")
audit = pd.read_csv(TAB / "AUDIT_gosif_gleam_mean_median_winsor_scaling_tests.csv")
keep = [c for c in ["mask", "outcome_aggregation", "scale_mode", "model", "status", "n", "coef", "se_hc3", "p", "ci_low", "ci_high", "r2", "aic"] if c in audit.columns]
memo.append(audit[keep].to_string(index=False))
memo.append("")
memo.append("5. Hidden-definition subset tests")
memo.append(show("AUDIT_gosif_gleam_by_hidden_definition_subsets.csv", 100))
memo.append("")
memo.append("6. Outlier sensitivity")
memo.append(show("AUDIT_gosif_gleam_outlier_sensitivity.csv", 100))
memo.append("")
memo.append("Important files:")
for f in [
    "AUDIT_product_combo_counts.csv",
    "AUDIT_possible_hidden_definition_columns.csv",
    "AUDIT_gosif_gleam_rows_per_point.csv",
    "AUDIT_gosif_gleam_mean_median_winsor_scaling_tests.csv",
    "AUDIT_gosif_gleam_by_hidden_definition_subsets.csv",
    "AUDIT_gosif_gleam_outlier_sensitivity.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_gosif_gleam_programming_audit.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_gosif_gleam_programming_audit.txt'}")
