#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path("results/paper_path_preflight")
OUT.mkdir(parents=True, exist_ok=True)

report = []
verdict = {}

def add(x=""):
    print(x)
    report.append(str(x))

add("# Paper-path preflight: raw 3x3 + tower + trait readiness")
add("")

# ----------------------------
# 1. Global raw 3x3 result
# ----------------------------
raw = Path("results/project_final_nature_boot50/fullspec_response_results_raw.csv")
if not raw.exists():
    add("## GLOBAL RAW 3x3: FAIL")
    add(f"Missing `{raw}`.")
    verdict["global_raw_3x3"] = "fail_missing"
else:
    df = pd.read_csv(raw, low_memory=False)
    df["gpp_product_norm"] = df["gpp_product"].astype(str).str.lower()
    df["et_product_norm"] = df["et_product"].astype(str).str.lower()

    combos = sorted(set(zip(df["gpp_product_norm"], df["et_product_norm"])))
    expected = sorted((g,e) for g in ["modis","gosif","pml"] for e in ["modis","gleam","pml"])
    missing = sorted(set(expected) - set(combos))

    add("## 1. Global raw 3x3 result")
    add(f"- File exists: `{raw}`")
    add(f"- Shape: `{df.shape}`")
    add(f"- Unique points: `{df['point_id'].nunique()}`")
    add(f"- Product combos found: `{len(combos)}` / 9")
    add(f"- Missing combos after case-normalization: `{missing}`")
    add(f"- Metrics: `{sorted(df['metric'].dropna().unique().tolist())}`")
    add(f"- Stress definitions: `{sorted(df['stress_definition'].dropna().unique().tolist())}`")
    add(f"- Growing seasons: `{sorted(df['growing_season'].dropna().unique().tolist())}`")

    if not missing:
        verdict["global_raw_3x3"] = "pass"
    else:
        verdict["global_raw_3x3"] = "fail_missing_combos"

    u = df[df["metric"].eq("uwue")].copy()
    u["sat_or_breakdown"] = u["response_class_strict"].isin(["saturation","breakdown"])
    u["breakdown_only"] = u["response_class_strict"].eq("breakdown")
    u["saturation_only"] = u["response_class_strict"].eq("saturation")

    product_signal = (
        u.groupby(["gpp_product","et_product"])
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
    product_signal.to_csv(OUT / "global_raw_uwue_product_signal.csv", index=False)

    add("")
    add("### Primary uWUE product signal")
    add(product_signal.to_string(index=False))

    class_counts = (
        u["response_class_strict"]
        .value_counts(dropna=False)
        .rename_axis("response_class")
        .reset_index(name="n")
    )
    class_counts["fraction"] = class_counts["n"] / len(u)
    class_counts.to_csv(OUT / "global_raw_uwue_class_counts.csv", index=False)

    add("")
    add("### Primary uWUE class counts")
    add(class_counts.to_string(index=False))

# ----------------------------
# 2. Current tower data readiness
# ----------------------------
add("")
add("## 2. Tower validation readiness")

tower_points = Path("data/raw/towers/tower_validation_points_agent.csv")
tower_gosif = Path("data/raw/towers/tower_agents/tower_gosif_point_timeseries.csv")
tower_gleam = Path("data/raw/towers/tower_agents/tower_gleam_point_timeseries.csv")
tower_gee_dir = Path("data/raw/towers/tower_gee")

for p in [tower_points, tower_gosif, tower_gleam]:
    add(f"- `{p}` exists: `{p.exists()}`")

tower_ready_local = tower_points.exists() and tower_gosif.exists() and tower_gleam.exists()

if tower_gosif.exists():
    tg = pd.read_csv(tower_gosif)
    add(f"- Tower GOSIF shape: `{tg.shape}`, points: `{tg['point_id'].nunique()}`, dates: `{tg['date'].nunique()}`")
if tower_gleam.exists():
    te = pd.read_csv(tower_gleam)
    add(f"- Tower GLEAM shape: `{te.shape}`, points: `{te['point_id'].nunique()}`, dates: `{te['date'].nunique()}`")

gee_files = sorted(tower_gee_dir.glob("tower_wue_timeseries_*.csv")) if tower_gee_dir.exists() else []
add(f"- Tower-centered GEE files downloaded: `{len(gee_files)}`")

if tower_ready_local and len(gee_files) == 0:
    verdict["tower_validation"] = "blocked_waiting_for_tower_centered_GEE"
    add("- Tower status: `LOCAL_GOSIF_GLEAM_READY_BUT_GEE_CORE_MISSING`")
elif tower_ready_local and len(gee_files) > 0:
    verdict["tower_validation"] = "partial_gee_files_present"
    add("- Tower status: `PARTIAL_GEE_READY`")
else:
    verdict["tower_validation"] = "fail_local_tower_inputs_missing"
    add("- Tower status: `MISSING_LOCAL_TOWER_INPUTS`")

# ----------------------------
# 3. Trait map/data readiness
# ----------------------------
add("")
add("## 3. Trait/covariate readiness")

trait_files = {
    "psi50": Path("data/external/liu_2021_psi50_0p1deg.nc"),
    "isohydricity": Path("data/external/konings_gentine_isohydricity_0p1deg.nc"),
    "rooting_depth": Path("data/external/stocker_2023_rooting_depth_0p1deg.nc"),
    "aridity": Path("data/external/aridity_by_point.csv"),
}
for name, path in trait_files.items():
    add(f"- {name}: `{path}` exists: `{path.exists()}`")

verdict["trait_files_present"] = all(p.exists() for p in trait_files.values())

# Optional xarray sampling smoke test
trait_sample_status = {}
try:
    import xarray as xr

    # Use global 3x3 point IDs as target sample points.
    if raw.exists():
        pts = pd.DataFrame({"point_id": sorted(df["point_id"].astype(str).unique())})
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

        for name in ["psi50","isohydricity","rooting_depth"]:
            path = trait_files[name]
            if not path.exists():
                trait_sample_status[name] = "missing"
                continue
            try:
                ds = xr.open_dataset(path)
                lat_name = next((c for c in ds.coords if c.lower() in ["lat","latitude","y"]), None)
                lon_name = next((c for c in ds.coords if c.lower() in ["lon","longitude","x"]), None)
                var_name = next((v for v in ds.data_vars), None)

                if lat_name is None or lon_name is None or var_name is None:
                    trait_sample_status[name] = f"could_not_detect_dims vars={list(ds.data_vars)} coords={list(ds.coords)}"
                    continue

                sample_vals = []
                for _, r in pts.head(25).iterrows():
                    val = ds[var_name].sel({lat_name: r["lat"], lon_name: r["lon"]}, method="nearest").values
                    sample_vals.append(float(np.asarray(val).squeeze()))

                ok = np.isfinite(sample_vals).sum()
                trait_sample_status[name] = f"sampled_first_25 finite={ok}/25 var={var_name}"
            except Exception as e:
                trait_sample_status[name] = f"sample_error: {type(e).__name__}: {e}"
except Exception as e:
    trait_sample_status["xarray"] = f"xarray_unavailable_or_error: {type(e).__name__}: {e}"

add("")
add("### Trait sampling smoke test")
for k,v in trait_sample_status.items():
    add(f"- {k}: `{v}`")

verdict["trait_sampling_smoke"] = trait_sample_status

# ----------------------------
# 4. Overall path verdict
# ----------------------------
add("")
add("## 4. Overall reviewer-paper path verdict")

if verdict.get("global_raw_3x3") == "pass":
    add("- Gate 1/Gate 2 raw computational infrastructure: `PASS_FOR_DRAFTING_METHODS_AND_PRELIM_RESULTS`")
else:
    add("- Gate 1/Gate 2 raw computational infrastructure: `NOT_READY`")

if verdict.get("tower_validation") == "blocked_waiting_for_tower_centered_GEE":
    add("- Gate 3 tower validation: `BLOCKED_ONLY_BY_TOWER_CENTERED_MODIS_PML_ERA5_EXPORT`")
elif verdict.get("tower_validation") == "partial_gee_files_present":
    add("- Gate 3 tower validation: `PARTIAL_READY_BUILD_MATRIX_NEXT`")
else:
    add("- Gate 3 tower validation: `NOT_READY`")

if verdict.get("trait_files_present"):
    add("- Trait phase data files: `PRESENT`")
else:
    add("- Trait phase data files: `MISSING_SOME_FILES`")

add("")
add("### Brutal conclusion")
add("The paper path is computationally alive, but the reviewer-level thesis is not unlocked until tower-centered MODIS/PML/ERA5 extraction completes and identifies a credible satellite product/metric family. Do not waste time rerunning global products. The next decisive bottleneck is the tower-centered GEE export now pending.")

Path(OUT / "README_paper_path_preflight.md").write_text("\n".join(report))
with open(OUT / "paper_path_preflight_verdict.json", "w") as f:
    json.dump(verdict, f, indent=2)

add("")
add(f"WROTE `{OUT / 'README_paper_path_preflight.md'}`")
add(f"WROTE `{OUT / 'paper_path_preflight_verdict.json'}`")
