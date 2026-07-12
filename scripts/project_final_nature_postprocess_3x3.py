#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path("results/project_final_nature")
OUT.mkdir(parents=True, exist_ok=True)
Path("docs").mkdir(exist_ok=True)

def alg_table():
    rows = [
        ["MOD17A2HGF.061","GPP","yes/indirect","yes","yes","no direct","yes/fPAR","light-use-efficiency"],
        ["GOSIF v2","GPP","no direct","yes","not primary","no direct","not shared MODIS LAI/fPAR","SIF-based"],
        ["PML-V2 GPP","GPP","yes","yes","yes","model dependent","yes","coupled carbon-water"],
        ["MOD16A2GF.061","ET","yes","yes","yes","no direct","yes/fPAR","Penman-Monteith"],
        ["GLEAM v4.3a","ET","structurally distinct","yes","yes","yes","vegetation dependent","Priestley-Taylor"],
        ["PML-V2 ET","ET","yes","yes","yes","model dependent","yes","coupled carbon-water"],
    ]
    df = pd.DataFrame(rows, columns=["product","flux","uses_vpd","uses_radiation","uses_temperature","uses_soil_moisture","vegetation_input","algorithm_family"])
    df["manual_source_verification_needed_before_manuscript"] = True
    df.to_csv("docs/algorithm_dependency_table.csv", index=False)

def dag():
    Path("docs/trait_causal_dag.md").write_text("""# Trait causal DAG and estimand

Target estimand: effect of hydraulic/stomatal strategy on validated uWUE response shape, holding climate and soil texture fixed.

Climate/aridity confounds traits, species composition, VPD regime, soil moisture regime, and response shape.
Soil texture confounds plant-available water and apparent soil-moisture thresholds.
Species composition confounds hydraulic traits, rooting depth, and response shape.

Required controls: aridity, MAP/MAT where available, growing-season LAI where available, sand/silt/clay or soil-texture descriptor.

Random forest + SHAP is descriptive only. Main trait claim requires partial-pooling / hierarchical model.
""")

def sample_traits():
    points_path = Path("data/raw/gee/stable_grassland_points.csv")
    if not points_path.exists():
        return pd.DataFrame([{"status":"not_run","reason":"missing stable_grassland_points.csv"}])
    pts = pd.read_csv(points_path)
    pts["point_id"] = pts["point_id"].astype(str)
    rows = pts[["point_id","lat","lon"]].copy()

    try:
        import xarray as xr
        specs = [
            ("data/external/liu_2021_psi50_0p1deg.nc","psi50","trait_psi50"),
            ("data/external/konings_gentine_isohydricity_0p1deg.nc","isohydricity","trait_isohydricity"),
            ("data/external/stocker_2023_rooting_depth_0p1deg.nc","rooting_depth","trait_rooting_depth"),
        ]
        for path, var, outcol in specs:
            p = Path(path)
            if not p.exists():
                rows[outcol] = np.nan
                continue
            ds = xr.open_dataset(p)
            vals = []
            for _, r in pts.iterrows():
                vals.append(float(ds[var].sel(lat=float(r["lat"]), lon=float(r["lon"]), method="nearest").values))
            ds.close()
            rows[outcol] = vals
        rows["status"] = "sampled"
    except Exception as e:
        rows["status"] = "error: " + str(e)
    rows.to_csv("data/external/trait_by_point.csv", index=False)
    return rows

def tower_status():
    paths = list(Path("data/raw/towers").glob("*.csv"))
    out = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            n = len(df)
            sites = df["tower_id"].nunique() if "tower_id" in df.columns else 0
            years = pd.to_datetime(df["date"], errors="coerce").dt.year.nunique() if "date" in df.columns else 0
            usable = False
            if n > 0 and {"tower_id","date","GPP_NT_VUT_REF","LE_F_MDS","VPD"}.issubset(df.columns):
                ok = df.dropna(subset=["GPP_NT_VUT_REF","LE_F_MDS","VPD"])
                usable = len(ok) > 500
            out.append({"file":str(p),"rows":n,"tower_sites":sites,"years":years,"usable_for_smoke":usable})
        except Exception as e:
            out.append({"file":str(p),"rows":0,"tower_sites":0,"years":0,"usable_for_smoke":False,"error":str(e)})
    pd.DataFrame(out).to_csv(OUT / "tower_arbiter_status.csv", index=False)

def aridity():
    ar = Path("data/external/aridity_by_point.csv")
    if not ar.exists():
        return
    ari = pd.read_csv(ar)
    ari["point_id"] = ari["point_id"].astype(str)
    for version in ["raw","co2corrected"]:
        p = OUT / f"fullspec_response_results_{version}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["point_id"] = df["point_id"].astype(str)
        m = df.merge(ari[["point_id","aridity_index","aridity_quartile"]], on="point_id", how="left")
        s = m.groupby(["metric","stress_definition","growing_season","aridity_quartile","response_class_strict"], dropna=False).size().reset_index(name="n")
        s.to_csv(OUT / f"fullspec_aridity_summary_{version}.csv", index=False)

def trait_proxy():
    traits = Path("data/external/trait_by_point.csv")
    res = OUT / "fullspec_response_results_raw.csv"
    if not traits.exists() or not res.exists():
        pd.DataFrame([{"status":"not_run","reason":"missing traits or strict results"}]).to_csv(OUT / "hierarchical_trait_proxy_results.csv", index=False)
        return
    tr = pd.read_csv(traits)
    df = pd.read_csv(res)
    tr["point_id"] = tr["point_id"].astype(str)
    df["point_id"] = df["point_id"].astype(str)
    m = df[df["metric"]=="uwue"].merge(tr, on="point_id", how="left")
    cols = [c for c in ["trait_psi50","trait_isohydricity","trait_rooting_depth"] if c in m.columns]
    if len(m) < 5 or not cols:
        pd.DataFrame([{"status":"not_run","reason":"insufficient trait rows"}]).to_csv(OUT / "hierarchical_trait_proxy_results.csv", index=False)
        return
    out = []
    for c in cols:
        out.append({"status":"smoke_proxy_complete","term":c,"n_nonmissing":int(m[c].notna().sum()),"note":"not final PyMC/Stan hierarchical model"})
    pd.DataFrame(out).to_csv(OUT / "hierarchical_trait_proxy_results.csv", index=False)

def manifest():
    required = [
        "results/project_final_nature/fullspec_response_results_raw.csv",
        "results/project_final_nature/fullspec_response_results_co2corrected.csv",
        "results/project_final_nature/fullspec_vpd_sm_surface_raw.csv",
        "results/project_final_nature/fullspec_vpd_sm_surface_co2corrected.csv",
        "results/project_final_nature/fullspec_aridity_summary_raw.csv",
        "results/project_final_nature/fullspec_aridity_summary_co2corrected.csv",
        "docs/algorithm_dependency_table.csv",
        "docs/trait_causal_dag.md",
        "data/external/trait_by_point.csv",
        "results/project_final_nature/tower_arbiter_status.csv",
        "results/project_final_nature/hierarchical_trait_proxy_results.csv",
        "results/project_final_nature/qc_audit_raw.json",
        "results/project_final_nature/qc_audit_co2corrected.json",
    ]
    rows = []
    for f in required:
        p = Path(f)
        rows.append({"file":f,"exists":p.exists(),"size":p.stat().st_size if p.exists() else 0})
    pd.DataFrame(rows).to_csv(OUT / "fullspec_implementation_manifest.csv", index=False)

alg_table()
dag()
sample_traits()
tower_status()
aridity()
trait_proxy()
manifest()
print("WROTE fullspec postprocess outputs")
