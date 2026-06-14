"""Point-table backend for GEE/AppEEARS/GLEAM/GOSIF extracted CSVs."""

from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import pandas as pd

from ..models.segmented import segmented_with_uncertainty, fit_interaction_model

GPP_COLS = {"MODIS": "gpp_modis", "PML": "gpp_pml", "GOSIF": "gpp_gosif"}
ET_COLS = {"MODIS": "et_modis", "PML": "et_pml", "GLEAM": "et_gleam"}


def load_point_exports(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = Path(f).name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    for c in df.columns:
        if c not in {"point_id", "date", "source_file", ".geo"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Earth Engine CSVs from this Drive-export script use lat/lon.
    # The downstream point backend expects latitude/longitude.
    if "latitude" not in df.columns and "lat" in df.columns:
        df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
    if "longitude" not in df.columns and "lon" in df.columns:
        df["longitude"] = pd.to_numeric(df["lon"], errors="coerce")

    df = df.replace([-9999, -9999.0, -32767, -32768], np.nan)

    if "point_id" not in df:
        df["point_id"] = df["longitude"].round(4).astype(str) + "_" + df["latitude"].round(4).astype(str)

    return df


def add_stress_definitions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(["point_id", "date"])
    def z(s):
        sd = s.std(skipna=True)
        return (s - s.mean(skipna=True)) / sd if sd and np.isfinite(sd) else np.nan
    d["vpd_z"] = d.groupby("point_id")["vpd"].transform(z)
    d["sm_z"] = d.groupby("point_id")["soil_moisture"].transform(z)
    d["csi_zscore"] = 0.5 * d["vpd_z"] - 0.5 * d["sm_z"]
    d["vpd_rank"] = d.groupby("point_id")["vpd"].rank(pct=True)
    d["sm_rank"] = d.groupby("point_id")["soil_moisture"].rank(pct=True)
    d["csi_percentile"] = (d["vpd_rank"] - 0.75).clip(lower=0) + (0.25 - d["sm_rank"]).clip(lower=0)
    d["csi_copula"] = d["vpd_rank"] * (1 - d["sm_rank"])
    d["interaction"] = d["vpd_z"] * d["sm_z"]
    return d


def add_growing_masks(df: pd.DataFrame, gpp_reference: str = "gpp_pml") -> pd.DataFrame:
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    annual_peak = d.groupby(["point_id", "year"])[gpp_reference].transform("max")
    d["gs_gpp_threshold"] = d[gpp_reference] >= 0.2 * annual_peak
    d["gs_climate_threshold"] = (d["temperature"] > 5.0) & (d["precipitation"].rolling(4, min_periods=1).sum().fillna(0) > 1.0)
    d["gs_month_fixed_effects"] = True
    return d


def add_log_wue(df: pd.DataFrame, gpp_products: list[str], et_products: list[str], gpp_floor=0.05, et_floor=0.1) -> pd.DataFrame:
    d = df.copy()
    for g in gpp_products:
        gcol = GPP_COLS[g]
        if gcol not in d: continue
        for e in et_products:
            ecol = ET_COLS[e]
            if ecol not in d: continue
            name = f"log_wue_{g}_{e}"
            ok = (d[gcol] > gpp_floor) & (d[ecol] > et_floor)
            d[name] = np.where(ok, np.log(d[gcol]) - np.log(d[ecol]), np.nan)
            d[f"log_gpp_{g}"] = np.where(d[gcol] > gpp_floor, np.log(d[gcol]), np.nan)
            d[f"log_et_{e}"] = np.where(d[ecol] > et_floor, np.log(d[ecol]), np.nan)
    return d


def prepare_point_table(input_glob: str, out_csv: str, gpp_products=None, et_products=None) -> Path:
    gpp_products = gpp_products or ["MODIS", "PML"]
    et_products = et_products or ["MODIS", "PML"]
    df = load_point_exports(input_glob)
    df = add_stress_definitions(df)
    df = add_growing_masks(df)
    df = add_log_wue(df, gpp_products, et_products)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def _mask_for_growing(df: pd.DataFrame, growing: str) -> pd.Series:
    return df[f"gs_{growing}"] if f"gs_{growing}" in df else pd.Series(True, index=df.index)


def fit_point_matrix(prepared_csv: str, out_csv: str, gpp_products=None, et_products=None,
                     stress_defs=None, growing_defs=None, min_obs=50, n_boot=1000, seed=42) -> Path:
    gpp_products = gpp_products or ["MODIS", "PML"]
    et_products = et_products or ["MODIS", "PML"]
    stress_defs = stress_defs or ["zscore", "percentile", "copula", "interaction"]
    growing_defs = growing_defs or ["gpp_threshold", "climate_threshold", "month_fixed_effects"]
    df = pd.read_csv(prepared_csv, parse_dates=["date"])
    rows = []
    for g in gpp_products:
        if GPP_COLS.get(g) not in df: continue
        for e in et_products:
            if ET_COLS.get(e) not in df: continue
            ycol = f"log_wue_{g}_{e}"
            if ycol not in df: continue
            for stress in stress_defs:
                xcol = {"zscore":"csi_zscore", "percentile":"csi_percentile", "copula":"csi_copula", "interaction":"interaction"}[stress]
                for growing in growing_defs:
                    mask = _mask_for_growing(df, growing)
                    for pid, sub in df.loc[mask].groupby("point_id"):
                        s = sub[["date", "latitude", "longitude", ycol, xcol, "vpd_z", "sm_z"]].dropna(subset=[ycol, xcol])
                        if stress == "interaction":
                            tmp = s.rename(columns={ycol:"log_wue"})
                            res = fit_interaction_model(tmp)
                            row = {"n": res.get("n"), "breakpoint": np.nan, "pre_slope": np.nan, "post_slope": np.nan,
                                   "slope_change": res.get("interaction_coef"), "response_class": res.get("response_class"),
                                   "interaction_p": res.get("interaction_p")}
                        else:
                            fit = segmented_with_uncertainty(s[xcol].values, s[ycol].values, min_obs=min_obs, n_boot=n_boot, seed=seed)
                            row = fit.to_dict()
                        row.update({
                            "point_id": pid,
                            "lat": s["latitude"].iloc[0] if len(s) else np.nan,
                            "lon": s["longitude"].iloc[0] if len(s) else np.nan,
                            "gpp_product": g,
                            "et_product": e,
                            "stress_definition": stress,
                            "growing_season": growing,
                        })
                        rows.append(row)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def summarize_point_matrix(results_csv: str, out_csv: str) -> Path:
    df = pd.read_csv(results_csv)
    group = ["gpp_product", "et_product", "stress_definition", "growing_season"]
    rows = []
    for keys, sub in df.groupby(group):
        row = dict(zip(group, keys))
        vc = sub["response_class"].value_counts(normalize=True)
        for cls in ["enhancement", "saturation", "reversal", "inconclusive", "insufficient_data", "interaction_negative", "interaction_not_negative"]:
            row[f"frac_{cls}"] = float(vc.get(cls, 0.0))
        row["median_pre_slope"] = sub["pre_slope"].median(skipna=True)
        row["median_post_slope"] = sub["post_slope"].median(skipna=True)
        row["median_slope_change"] = sub["slope_change"].median(skipna=True)
        row["n_points"] = sub["point_id"].nunique()
        rows.append(row)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
