#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path("results/thesis_feasibility_no_tower")
OUT.mkdir(parents=True, exist_ok=True)

RAW = Path("results/reza_final_nature_boot50/fullspec_response_results_raw.csv")
CO2 = Path("results/reza_final_nature_boot50/fullspec_response_results_co2corrected.csv")

if not RAW.exists():
    raise SystemExit(f"Missing {RAW}")
if not CO2.exists():
    raise SystemExit(f"Missing {CO2}")

raw = pd.read_csv(RAW, low_memory=False)
co2 = pd.read_csv(CO2, low_memory=False)

for df in [raw, co2]:
    df.columns = [str(c).strip() for c in df.columns]
    df["gpp_product"] = df["gpp_product"].astype(str).str.upper()
    df["et_product"] = df["et_product"].astype(str).str.upper()
    df["metric"] = df["metric"].astype(str).str.lower()
    df["point_id"] = df["point_id"].astype(str)

def summarize_signal(df, label):
    u = df[df["metric"].eq("uwue")].copy()
    u["sat_or_breakdown"] = u["response_class_strict"].isin(["saturation", "breakdown"])
    u["breakdown"] = u["response_class_strict"].eq("breakdown")
    u["saturation"] = u["response_class_strict"].eq("saturation")

    prod = (
        u.groupby(["gpp_product","et_product"])
        .agg(
            n=("point_id","size"),
            points=("point_id","nunique"),
            sat_or_breakdown_frac=("sat_or_breakdown","mean"),
            breakdown_frac=("breakdown","mean"),
            saturation_frac=("saturation","mean"),
        )
        .reset_index()
    )
    prod["version"] = label
    return prod

prod_raw = summarize_signal(raw, "raw")
prod_co2 = summarize_signal(co2, "co2corrected")
prod_both = pd.concat([prod_raw, prod_co2], ignore_index=True)
prod_both.to_csv(OUT / "product_signal_raw_vs_co2.csv", index=False)

wide = prod_both.pivot_table(
    index=["gpp_product","et_product"],
    columns="version",
    values=["sat_or_breakdown_frac","breakdown_frac","saturation_frac"]
)
wide.columns = ["_".join(c).strip() for c in wide.columns]
wide = wide.reset_index()
for c in ["sat_or_breakdown_frac", "breakdown_frac", "saturation_frac"]:
    r = f"{c}_raw"
    k = f"{c}_co2corrected"
    if r in wide.columns and k in wide.columns:
        wide[f"{c}_co2_minus_raw"] = wide[k] - wide[r]
wide.to_csv(OUT / "product_signal_co2_stability.csv", index=False)

# Compare slope_change raw vs CO2 for identical fit keys.
key_cols = ["point_id","metric","gpp_product","et_product","stress_definition","growing_season"]
slope_cols = ["pre_slope","post_slope","slope_change"]
raw_s = raw[key_cols + [c for c in slope_cols if c in raw.columns]].copy()
co2_s = co2[key_cols + [c for c in slope_cols if c in co2.columns]].copy()

m = raw_s.merge(co2_s, on=key_cols, suffixes=("_raw","_co2"))
rows = []
for combo, d in m[m["metric"].eq("uwue")].groupby(["gpp_product","et_product"]):
    row = {"gpp_product": combo[0], "et_product": combo[1], "n": len(d)}
    for c in slope_cols:
        a = f"{c}_raw"
        b = f"{c}_co2"
        if a in d.columns and b in d.columns:
            x = pd.to_numeric(d[a], errors="coerce")
            y = pd.to_numeric(d[b], errors="coerce")
            ok = x.notna() & y.notna()
            row[f"{c}_corr_raw_co2"] = float(x[ok].corr(y[ok])) if ok.sum() > 2 else np.nan
            row[f"{c}_median_abs_diff"] = float((x[ok] - y[ok]).abs().median()) if ok.sum() > 0 else np.nan
    rows.append(row)
slope_stability = pd.DataFrame(rows)
slope_stability.to_csv(OUT / "slope_endpoint_raw_vs_co2_stability.csv", index=False)

# Build point-level response outcome from CO2-corrected version, because that is safer for long-period thesis.
u = co2[co2["metric"].eq("uwue")].copy()
u["sat_or_breakdown"] = u["response_class_strict"].isin(["saturation","breakdown"]).astype(int)
u["breakdown"] = u["response_class_strict"].eq("breakdown").astype(int)
u["saturation"] = u["response_class_strict"].eq("saturation").astype(int)

agg = {
    "sat_or_breakdown_rate": ("sat_or_breakdown","mean"),
    "breakdown_rate": ("breakdown","mean"),
    "saturation_rate": ("saturation","mean"),
    "n_fits": ("point_id","size"),
}
if "slope_change" in u.columns:
    agg["median_slope_change"] = ("slope_change","median")
if "post_slope" in u.columns:
    agg["median_post_slope"] = ("post_slope","median")
if "pre_slope" in u.columns:
    agg["median_pre_slope"] = ("pre_slope","median")

resp = (
    u.groupby(["point_id","gpp_product","et_product"])
    .agg(**agg)
    .reset_index()
)

def parse_pid(pid):
    s = str(pid).replace(",", "_").split("_")
    if len(s) < 2:
        return np.nan, np.nan
    try:
        return float(s[0]), float(s[1])
    except Exception:
        return np.nan, np.nan

pts = pd.DataFrame({"point_id": sorted(u["point_id"].unique())})
pts[["lon","lat"]] = pts["point_id"].apply(lambda x: pd.Series(parse_pid(x)))
pts = pts.dropna(subset=["lon","lat"]).copy()

# Sample trait maps.
trait_status = {}
try:
    import xarray as xr
    trait_paths = {
        "psi50": Path("data/external/liu_2021_psi50_0p1deg.nc"),
        "isohydricity": Path("data/external/konings_gentine_isohydricity_0p1deg.nc"),
        "rooting_depth": Path("data/external/stocker_2023_rooting_depth_0p1deg.nc"),
    }

    for name, path in trait_paths.items():
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

# Add aridity if point_id merge works.
aridity_path = Path("data/external/aridity_by_point.csv")
if aridity_path.exists():
    ar = pd.read_csv(aridity_path, low_memory=False)
    ar.columns = [str(c).strip() for c in ar.columns]
    if "point_id" in ar.columns:
        ar["point_id"] = ar["point_id"].astype(str)
        pts = pts.merge(ar, on="point_id", how="left", suffixes=("","_aridity"))

# Identify usable aridity columns.
arid_cols = [c for c in pts.columns if "aridity" in c.lower() or c.lower() in ["ai","arid_index"]]

trait_ready = resp.merge(pts, on="point_id", how="left")
trait_ready.to_csv(OUT / "trait_model_ready_co2corrected.csv", index=False)

# Test predictor sets.
predictor_sets = {
    "rooting_only": ["rooting_depth"],
    "psi50_rooting": ["psi50", "rooting_depth"],
    "psi50_rooting_aridity": ["psi50", "rooting_depth"] + arid_cols[:1],
    "full_traits_with_isohydricity": ["psi50", "rooting_depth", "isohydricity"] + arid_cols[:1],
}

outcomes = [c for c in ["median_slope_change", "median_post_slope", "sat_or_breakdown_rate", "breakdown_rate", "saturation_rate"] if c in trait_ready.columns]

def ols_r2(y, X):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ beta
    ssr = np.sum((y - pred) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if sst == 0 else 1 - ssr/sst
    return beta, r2

model_rows = []
for (gpp, et), d0 in trait_ready.groupby(["gpp_product","et_product"]):
    for outcome in outcomes:
        for set_name, preds in predictor_sets.items():
            preds = [p for p in preds if p in d0.columns]
            if not preds:
                continue
            cols = [outcome] + preds
            d = d0[cols].copy()
            for c in cols:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.replace([np.inf,-np.inf], np.nan).dropna()
            row = {
                "gpp_product": gpp,
                "et_product": et,
                "outcome": outcome,
                "predictor_set": set_name,
                "predictors": ",".join(preds),
                "n_complete": len(d),
                "outcome_sd": float(d[outcome].std()) if len(d) > 1 else np.nan,
                "meets_min_n_50": len(d) >= 50,
                "has_variation": bool(len(d) > 1 and d[outcome].std() > 0),
                "r2_screen": np.nan,
            }
            if len(d) >= max(10, len(preds) + 5) and d[outcome].std() > 0:
                try:
                    beta, r2 = ols_r2(d[outcome], d[preds])
                    row["r2_screen"] = r2
                    for p, b in zip(preds, beta[1:]):
                        row[f"beta_{p}"] = b
                except Exception:
                    pass
            model_rows.append(row)

models = pd.DataFrame(model_rows)
models.to_csv(OUT / "trait_model_screen_co2corrected_NOT_FINAL.csv", index=False)

# Verdict.
report = []
def add(x=""):
    report.append(str(x))
    print(x)

add("# Thesis feasibility without tower GEE")
add("")
add("## 1. Raw vs CO₂ availability")
add(f"- Raw exists: `{RAW.exists()}`")
add(f"- CO₂-corrected exists: `{CO2.exists()}`")
add(f"- Raw shape: `{raw.shape}`")
add(f"- CO₂ shape: `{co2.shape}`")
add("")
add("## 2. Product signal stability")
add(wide.to_string(index=False))
add("")
add("## 3. Slope endpoint raw-vs-CO₂ stability")
add(slope_stability.to_string(index=False))
add("")
add("## 4. Trait coverage")
for k,v in trait_status.items():
    add(f"- {k}: `{v}`")
add("")
add("## 5. Trait model screen")
if len(models):
    add(models.sort_values(["predictor_set","n_complete"], ascending=[True,False]).to_string(index=False))
else:
    add("No model rows.")
add("")
add("## 6. Interpretation")

core = models[
    models["predictor_set"].isin(["rooting_only","psi50_rooting","psi50_rooting_aridity"]) &
    models["meets_min_n_50"] &
    models["has_variation"]
].copy()

full = models[
    models["predictor_set"].eq("full_traits_with_isohydricity") &
    models["meets_min_n_50"] &
    models["has_variation"]
].copy()

if len(core) > 0:
    add("- Core physiology thesis using P50/rooting/aridity is computationally possible with current points.")
else:
    add("- Core physiology thesis is weak with current points.")

if len(full) > 0:
    add("- Full mentor trait thesis including isohydricity is computationally possible with current points.")
else:
    add("- Full mentor trait thesis including isohydricity is NOT supported by current 199 points because complete-case coverage is too low.")

add("- Without tower-centered GEE, product-family validation is still missing.")
add("- Therefore: the thesis is plausible only as an unvalidated feasibility result right now; final paper still needs either tower extraction or a clear statement that tower validation is pending/impossible.")

verdict = {
    "raw_co2_available": True,
    "core_trait_models_possible_current_points": bool(len(core) > 0),
    "full_isohydricity_trait_model_possible_current_points": bool(len(full) > 0),
    "tower_validation_available": False,
    "final_mentor_thesis_supported_now": False,
    "final_mentor_thesis_possible_if_tower_validates_and_trait_sample_expanded": True,
}

with open(OUT / "thesis_feasibility_verdict.json", "w") as f:
    json.dump(verdict, f, indent=2)

Path(OUT / "README_thesis_feasibility_no_tower.md").write_text("\n".join(report))
print("")
print("WROTE", OUT / "README_thesis_feasibility_no_tower.md")
print("WROTE", OUT / "thesis_feasibility_verdict.json")
