#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path("results/reviewer_paper_feasibility")
OUT.mkdir(parents=True, exist_ok=True)

RAW = Path("results/project_final_nature_boot50/fullspec_response_results_raw.csv")
if not RAW.exists():
    raise SystemExit(f"Missing {RAW}")

df = pd.read_csv(RAW, low_memory=False)
df.columns = [str(c).strip() for c in df.columns]

df["gpp_product_norm"] = df["gpp_product"].astype(str).str.upper()
df["et_product_norm"] = df["et_product"].astype(str).str.upper()

u = df[df["metric"].astype(str).str.lower().eq("uwue")].copy()
u["sat_or_breakdown"] = u["response_class_strict"].isin(["saturation", "breakdown"]).astype(int)
u["breakdown_only"] = u["response_class_strict"].eq("breakdown").astype(int)
u["saturation_only"] = u["response_class_strict"].eq("saturation").astype(int)

report = []
def add(x=""):
    print(x)
    report.append(str(x))

add("# reviewer-paper feasibility check")
add("")

# ---------------------------------------------------------
# 1. Gate 1/Gate 2 raw structure
# ---------------------------------------------------------
expected = sorted((g,e) for g in ["MODIS","GOSIF","PML"] for e in ["MODIS","GLEAM","PML"])
found = sorted(set(zip(u["gpp_product_norm"], u["et_product_norm"])))
missing = sorted(set(expected) - set(found))

add("## 1. Raw 3x3 infrastructure")
add(f"- Raw result shape: `{df.shape}`")
add(f"- Unique points: `{df['point_id'].nunique()}`")
add(f"- Product combos found: `{len(found)}/9`")
add(f"- Missing product combos: `{missing}`")
add(f"- Metrics: `{sorted(df['metric'].dropna().unique().tolist())}`")
add(f"- Stress definitions: `{sorted(df['stress_definition'].dropna().unique().tolist())}`")
add(f"- Growing seasons: `{sorted(df['growing_season'].dropna().unique().tolist())}`")
add("")

# ---------------------------------------------------------
# 2. What raw result says biologically
# ---------------------------------------------------------
class_counts = u["response_class_strict"].value_counts(dropna=False).rename_axis("class").reset_index(name="n")
class_counts["frac"] = class_counts["n"] / class_counts["n"].sum()
class_counts.to_csv(OUT / "uwue_class_counts.csv", index=False)

prod = (
    u.groupby(["gpp_product_norm","et_product_norm"])
    .agg(
        n=("point_id","size"),
        points=("point_id","nunique"),
        sat_or_breakdown_frac=("sat_or_breakdown","mean"),
        breakdown_frac=("breakdown_only","mean"),
        saturation_frac=("saturation_only","mean"),
    )
    .reset_index()
    .sort_values("sat_or_breakdown_frac", ascending=False)
)
prod.to_csv(OUT / "uwue_product_signal.csv", index=False)

add("## 2. Raw result biological interpretation")
add("")
add("### uWUE strict class counts")
add(class_counts.to_string(index=False))
add("")
add("### uWUE product signal")
add(prod.to_string(index=False))
add("")

overall_break = float(u["breakdown_only"].mean())
overall_satbreak = float(u["sat_or_breakdown"].mean())
best_combo = prod.iloc[0].to_dict()

if overall_break < 0.05:
    add("- Interpretation: strict breakdown is rare globally. Do not frame the paper as universal WUE breakdown.")
if overall_satbreak < 0.10:
    add("- Interpretation: saturation/breakdown is a minority response. The trait paper must use continuous response strength, not just binary breakdown class.")
add(f"- Strongest product combo: `{best_combo['gpp_product_norm']}/{best_combo['et_product_norm']}` with sat/break fraction `{best_combo['sat_or_breakdown_frac']:.3f}`.")
add("")

# ---------------------------------------------------------
# 3. Detect slope columns for trait endpoint
# ---------------------------------------------------------
slope_cols = [c for c in u.columns if "slope" in c.lower()]
add("## 3. Continuous response metric availability")
add(f"- Slope-like columns: `{slope_cols}`")

pre_cols = [c for c in slope_cols if "pre" in c.lower()]
post_cols = [c for c in slope_cols if "post" in c.lower()]
pre_col = pre_cols[0] if pre_cols else None
post_col = post_cols[0] if post_cols else None

if pre_col and post_col:
    u[pre_col] = pd.to_numeric(u[pre_col], errors="coerce")
    u[post_col] = pd.to_numeric(u[post_col], errors="coerce")
    u["slope_change"] = u[post_col] - u[pre_col]
    add(f"- Using pre-slope column: `{pre_col}`")
    add(f"- Using post-slope column: `{post_col}`")
    add("- Continuous slope-change endpoint: `AVAILABLE`")
else:
    add("- Continuous slope-change endpoint: `NOT DETECTED`; binary response fractions can still be used, but that is weaker.")
add("")

# ---------------------------------------------------------
# 4. Point-level response outcomes
# ---------------------------------------------------------
agg_dict = {
    "n_fits": ("point_id","size"),
    "sat_or_breakdown_rate": ("sat_or_breakdown","mean"),
    "breakdown_rate": ("breakdown_only","mean"),
    "saturation_rate": ("saturation_only","mean"),
}
if "slope_change" in u.columns:
    agg_dict["median_slope_change"] = ("slope_change","median")
    agg_dict["median_post_slope"] = (post_col,"median")
    agg_dict["median_pre_slope"] = (pre_col,"median")

point_resp = (
    u.groupby(["point_id","gpp_product_norm","et_product_norm"])
    .agg(**agg_dict)
    .reset_index()
)

point_resp.to_csv(OUT / "point_product_response_outcomes.csv", index=False)

variation = (
    point_resp.groupby(["gpp_product_norm","et_product_norm"])
    .agg(
        points=("point_id","nunique"),
        satbreak_mean=("sat_or_breakdown_rate","mean"),
        satbreak_sd=("sat_or_breakdown_rate","std"),
        breakdown_mean=("breakdown_rate","mean"),
        breakdown_sd=("breakdown_rate","std"),
    )
    .reset_index()
)
if "median_slope_change" in point_resp.columns:
    extra = (
        point_resp.groupby(["gpp_product_norm","et_product_norm"])
        .agg(
            slope_change_finite=("median_slope_change", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            slope_change_sd=("median_slope_change", "std"),
        )
        .reset_index()
    )
    variation = variation.merge(extra, on=["gpp_product_norm","et_product_norm"], how="left")

variation.to_csv(OUT / "response_outcome_variation_by_combo.csv", index=False)

add("## 4. Does the response have enough spatial variation for traits?")
add(variation.to_string(index=False))
add("")

# ---------------------------------------------------------
# 5. Trait sampling full 199 points
# ---------------------------------------------------------
pts = pd.DataFrame({"point_id": sorted(u["point_id"].astype(str).unique())})

def parse_pid(pid):
    s = str(pid).replace(",", "_").split("_")
    if len(s) < 2:
        return np.nan, np.nan
    try:
        return float(s[0]), float(s[1])
    except Exception:
        return np.nan, np.nan

pts[["lon","lat"]] = pts["point_id"].apply(lambda x: pd.Series(parse_pid(x)))
pts = pts.dropna(subset=["lon","lat"]).copy()

trait_status = {}
try:
    import xarray as xr

    paths = {
        "psi50": Path("data/external/liu_2021_psi50_0p1deg.nc"),
        "isohydricity": Path("data/external/konings_gentine_isohydricity_0p1deg.nc"),
        "rooting_depth": Path("data/external/stocker_2023_rooting_depth_0p1deg.nc"),
    }

    for name, path in paths.items():
        if not path.exists():
            pts[name] = np.nan
            trait_status[name] = {"exists": False}
            continue

        ds = xr.open_dataset(path)
        lat_name = next((c for c in ds.coords if c.lower() in ["lat","latitude","y"]), None)
        lon_name = next((c for c in ds.coords if c.lower() in ["lon","longitude","x"]), None)
        var_name = next(iter(ds.data_vars))

        vals = []
        for _, r in pts.iterrows():
            try:
                val = ds[var_name].sel({lat_name: r["lat"], lon_name: r["lon"]}, method="nearest").values
                vals.append(float(np.asarray(val).squeeze()))
            except Exception:
                vals.append(np.nan)

        pts[name] = vals
        trait_status[name] = {
            "exists": True,
            "var": var_name,
            "finite": int(np.isfinite(pd.to_numeric(pts[name], errors="coerce")).sum()),
            "total": int(len(pts)),
            "coverage": float(np.isfinite(pd.to_numeric(pts[name], errors="coerce")).mean()),
        }

except Exception as e:
    trait_status["error"] = f"{type(e).__name__}: {e}"

# aridity
aridity_path = Path("data/external/aridity_by_point.csv")
if aridity_path.exists():
    ar = pd.read_csv(aridity_path, low_memory=False)
    ar.columns = [str(c).strip() for c in ar.columns]
    if "point_id" in ar.columns:
        ar["point_id"] = ar["point_id"].astype(str)
        pts = pts.merge(ar, on="point_id", how="left", suffixes=("","_aridity"))

pts.to_csv(OUT / "trait_values_at_global_points.csv", index=False)

add("## 5. Full trait coverage across global points")
for k,v in trait_status.items():
    add(f"- {k}: `{v}`")
add("")

# ---------------------------------------------------------
# 6. Minimal trait-model feasibility
# ---------------------------------------------------------
trait_ready = point_resp.merge(pts, on="point_id", how="left")
trait_ready.to_csv(OUT / "trait_model_ready_raw_not_final.csv", index=False)

predictors = [c for c in ["psi50","isohydricity","rooting_depth"] if c in trait_ready.columns]

# add one aridity-like covariate if present
for c in trait_ready.columns:
    if "aridity" in c.lower() or c.lower() in ["ai","arid_index"]:
        if c not in predictors:
            predictors.append(c)
        break

outcomes = ["sat_or_breakdown_rate", "breakdown_rate", "saturation_rate"]
if "median_slope_change" in trait_ready.columns:
    outcomes.insert(0, "median_slope_change")
if "median_post_slope" in trait_ready.columns:
    outcomes.insert(1, "median_post_slope")

feas_rows = []
for combo, d0 in trait_ready.groupby(["gpp_product_norm","et_product_norm"]):
    for outcome in outcomes:
        cols = [outcome] + predictors
        d = d0[cols].copy()
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.replace([np.inf,-np.inf], np.nan).dropna()
        feas_rows.append({
            "gpp_product": combo[0],
            "et_product": combo[1],
            "outcome": outcome,
            "usable_n_complete_cases": len(d),
            "outcome_sd": float(d[outcome].std()) if len(d) > 1 else np.nan,
            "predictors_used": ",".join(predictors),
            "meets_min_n_50": len(d) >= 50,
            "has_outcome_variation": bool(len(d) > 1 and pd.to_numeric(d[outcome], errors="coerce").std() > 0),
        })

feas = pd.DataFrame(feas_rows)
feas.to_csv(OUT / "trait_model_feasibility_by_combo_outcome.csv", index=False)

add("## 6. Trait-model feasibility by combo/outcome")
add(feas.to_string(index=False))
add("")

# ---------------------------------------------------------
# 7. Final verdict
# ---------------------------------------------------------
gate12_ok = (len(missing) == 0)
trait_any_ok = bool((feas["meets_min_n_50"] & feas["has_outcome_variation"]).any()) if not feas.empty else False
slope_available = "median_slope_change" in point_resp.columns

add("## 7. Go / no-go interpretation")
if gate12_ok:
    add("- Gate 1/Gate 2 infrastructure: `YES`")
else:
    add("- Gate 1/Gate 2 infrastructure: `NO`")

if slope_available:
    add("- Continuous response endpoint for trait paper: `YES`")
else:
    add("- Continuous response endpoint for trait paper: `WEAK/NOT FOUND`")

if trait_any_ok:
    add("- Trait model has at least one combo/outcome with usable N >= 50 and nonzero outcome variation: `YES`")
else:
    add("- Trait model complete-case feasibility is weak with current 199 points: `NO_OR_WEAK`")

add("- Tower validation: `NOT YET`; current blocker is pending tower-centered GEE extraction.")

add("")
add("### Bottom line")
if gate12_ok and slope_available and trait_any_ok:
    add("The reviewer-style paper is possible in principle, but still conditional on tower validation. The raw data support moving toward a trait-response paper, not a universal breakdown paper.")
elif gate12_ok and not trait_any_ok:
    add("The response pipeline is strong, but the trait endpoint may be underpowered/sparse with only 199 points. If tower validation succeeds, you may still need a larger trait-covered satellite sample before the exact reviewer thesis is defensible.")
else:
    add("The exact reviewer-style paper is not yet supported by the available results.")

Path(OUT / "README_reviewer_paper_feasibility.md").write_text("\n".join(report))

summary = {
    "gate12_ok": gate12_ok,
    "slope_available": slope_available,
    "trait_any_complete_case_ok": trait_any_ok,
    "tower_validation_done": False,
    "raw_breakdown_frac": overall_break if "overall_break" in globals() else None,
    "raw_sat_or_breakdown_frac": overall_satbreak if "overall_satbreak" in globals() else None,
}
with open(OUT / "reviewer_paper_feasibility_verdict.json", "w") as f:
    json.dump(summary, f, indent=2)

add("")
add(f"WROTE `{OUT / 'README_reviewer_paper_feasibility.md'}`")
add(f"WROTE `{OUT / 'reviewer_paper_feasibility_verdict.json'}`")
