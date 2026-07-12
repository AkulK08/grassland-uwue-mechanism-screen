#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 17: Broad tower validation from AmeriFlux/FLUXNET candidate inventory.

This script replaces the older narrow FLUXNET2015 checker.

Inputs:
  results/tower_file_check/ameriflux_candidate_wue_files.csv
  results/tower_file_check/ameriflux_zip_member_inventory.csv

Primary goals:
  1. Use broad AmeriFlux/FLUXNET zip inventory.
  2. Prefer daily FLUXMET_DD files.
  3. Extract GPP, LE, VPD, SWC/P variables.
  4. Compute daily and 8-day WUE/uWUE tower flux metrics.
  5. Estimate tower response phenotype under high VPD / compound stress.
  6. Compare to existing satellite latent points only as provisional nearest-neighbor diagnostic.
  7. Produce tower-centered satellite extraction targets for rigorous validation.
  8. Produce manuscript-safe tower-validation interpretation.

Important:
  If existing satellite points are not close to towers, this script will say so.
  Rigorous tower validation then requires extracting satellite products at tower coordinates.
"""

from pathlib import Path
import os
import re
import io
import json
import math
import zipfile
import warnings
import subprocess
import sys
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")


def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
        return True
    except Exception:
        print(f"Installing missing package: {pip_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            __import__(import_name)
            return True
        except Exception as e:
            print(f"WARNING: could not install {pip_name}: {e}")
            return False


for import_name, pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
]:
    ensure_package(import_name, pip_name)

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(".")
CANDIDATE_FILE = Path("results/tower_file_check/ameriflux_candidate_wue_files.csv")
INVENTORY_FILE = Path("results/tower_file_check/ameriflux_zip_member_inventory.csv")

PH8_LATENT = Path("results/trait_framework/phase8/table_latent_response_by_point.csv")
PH16_POINTS = Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv")

OUT = Path("results/tower_validation_broad_inventory")
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"
TMP = OUT / "tmp"

for p in [OUT, TAB, FIG, TXT, TMP]:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def die(msg):
    raise SystemExit(f"\nERROR: {msg}\n")


def read_csv(path: Path, required=True, **kwargs):
    if not path.exists():
        if required:
            die(f"Missing required file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False, **kwargs)
    return df.loc[:, ~df.columns.duplicated()].copy()


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")


def save_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"WROTE {path}")


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()
    print(f"WROTE {path}")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def fmt(x, d=3):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{d}f}"
    except Exception:
        return "NA"


def pct(x):
    try:
        if pd.isna(x):
            return "NA"
        return f"{100*float(x):.1f}%"
    except Exception:
        return "NA"


def clean_missing(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = out[c].replace([-9999, -9999.0, -6999, -6999.0], np.nan)
        else:
            out[c] = out[c].replace(["-9999", "-9999.0", "-6999", "-6999.0"], np.nan)
    return out


def choose_col(columns, preferred):
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    for p in preferred:
        if p in cols:
            return p
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    return None


def cols_matching(columns, patterns):
    out = []
    for c in columns:
        cl = c.lower()
        for pat in patterns:
            if re.search(pat, cl):
                out.append(c)
                break
    return out


def parse_site_id_from_zip_or_member(zip_name, member=""):
    text = f"{zip_name} {member}"

    patterns = [
        r"AMF_([A-Z]{2}-[A-Za-z0-9]{3})_",
        r"FLX_([A-Z]{2}-[A-Za-z0-9]{3})_",
        r"([A-Z]{2}-[A-Za-z0-9]{3})",
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)

    return ""


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def safe_read_member_csv(zip_path, member, nrows=None, usecols=None):
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(member) as f:
            try:
                return pd.read_csv(f, nrows=nrows, usecols=usecols, low_memory=False)
            except UnicodeDecodeError:
                f.seek(0)
                return pd.read_csv(f, nrows=nrows, usecols=usecols, low_memory=False, encoding="latin1")


def parse_timestamp(df):
    if "TIMESTAMP" in df.columns:
        s = df["TIMESTAMP"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        # Daily TIMESTAMP is YYYYMMDD; monthly/yearly are excluded upstream.
        dt = pd.to_datetime(s.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        return dt

    if "TIMESTAMP_START" in df.columns:
        s = df["TIMESTAMP_START"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        dt = pd.to_datetime(s.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        return dt

    if "date" in [c.lower() for c in df.columns]:
        col = [c for c in df.columns if c.lower() == "date"][0]
        return pd.to_datetime(df[col], errors="coerce")

    return pd.Series(pd.NaT, index=df.index)


# ---------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------

def regex_find_float(text, keys):
    for key in keys:
        # handles KEY,value / KEY: value / KEY = value
        pat = rf"{re.escape(key)}[^-\d\.]{{0,40}}(-?\d+\.?\d*)"
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return np.nan


def regex_find_igbp(text):
    patterns = [
        r"\bIGBP\b[^A-Za-z0-9]{1,40}([A-Z]{3})",
        r"\bLAND_COVER_IGBP\b[^A-Za-z0-9]{1,40}([A-Z]{3})",
        r"\bIGBP_CLASS\b[^A-Za-z0-9]{1,40}([A-Z]{3})",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def extract_metadata_from_zip(zip_path, site_id):
    """
    Tries several metadata formats:
      1. CSV with direct columns LOCATION_LAT / LOCATION_LONG / IGBP.
      2. BADM/BIF long table with VARIABLE / DATAVALUE fields.
      3. Raw regex search through metadata-like files.
    """
    meta = {
        "site": site_id,
        "tower_lat": np.nan,
        "tower_lon": np.nan,
        "igbp": "",
        "metadata_source": "",
    }

    zip_path = Path(zip_path)
    if not zip_path.exists():
        return meta

    metadata_name_patterns = [
        "BADM", "BIF", "META", "SITE", "VARIABLES", "AUX"
    ]

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            metadata_members = [
                n for n in names
                if any(p.lower() in n.lower() for p in metadata_name_patterns)
                and n.lower().endswith((".csv", ".txt"))
            ]

            # Put BADM/BIF first.
            metadata_members = sorted(
                metadata_members,
                key=lambda x: (0 if ("badm" in x.lower() or "bif" in x.lower()) else 1, x)
            )

            for member in metadata_members:
                try:
                    raw = z.open(member).read(2_000_000)
                    text = raw.decode("utf-8", errors="ignore")

                    # First regex, because it works on many metadata formats.
                    lat = regex_find_float(text, ["LOCATION_LAT", "LATITUDE", "SITE_LAT", "LAT"])
                    lon = regex_find_float(text, ["LOCATION_LONG", "LOCATION_LON", "LONGITUDE", "SITE_LON", "LON"])
                    igbp = regex_find_igbp(text)

                    if pd.notna(lat) and pd.notna(lon):
                        meta["tower_lat"] = lat
                        meta["tower_lon"] = lon
                        meta["igbp"] = igbp or meta["igbp"]
                        meta["metadata_source"] = member + "::regex"
                        return meta

                    # Try structured CSV.
                    try:
                        dfm = pd.read_csv(io.BytesIO(raw), low_memory=False)
                        dfm = clean_missing(dfm)
                        cols = list(dfm.columns)
                        colmap = {c.lower(): c for c in cols}

                        lat_col = choose_col(cols, ["LOCATION_LAT", "LATITUDE", "SITE_LAT", "LAT"])
                        lon_col = choose_col(cols, ["LOCATION_LONG", "LOCATION_LON", "LONGITUDE", "SITE_LON", "LON"])
                        igbp_col = choose_col(cols, ["IGBP", "LAND_COVER_IGBP", "IGBP_CLASS"])

                        if lat_col and lon_col:
                            lat_val = num(dfm[lat_col]).dropna()
                            lon_val = num(dfm[lon_col]).dropna()
                            if len(lat_val) and len(lon_val):
                                meta["tower_lat"] = float(lat_val.iloc[0])
                                meta["tower_lon"] = float(lon_val.iloc[0])
                                if igbp_col:
                                    vals = dfm[igbp_col].dropna().astype(str)
                                    meta["igbp"] = vals.iloc[0].strip().upper() if len(vals) else ""
                                meta["metadata_source"] = member + "::wide_csv"
                                return meta

                        # Long BADM style.
                        var_col = choose_col(cols, ["VARIABLE", "VAR", "BADM_VARIABLE", "DATAVARIABLE"])
                        val_col = choose_col(cols, ["DATAVALUE", "VALUE", "DATA_VALUE", "VAR_VALUE"])
                        if var_col and val_col:
                            dlong = dfm[[var_col, val_col]].dropna()
                            dlong[var_col] = dlong[var_col].astype(str).str.upper()
                            dlong[val_col] = dlong[val_col].astype(str)

                            def find_long(keys):
                                keys = [k.upper() for k in keys]
                                sub = dlong[dlong[var_col].isin(keys)]
                                if len(sub):
                                    return sub[val_col].iloc[0]
                                for k in keys:
                                    sub = dlong[dlong[var_col].str.contains(k, regex=False, na=False)]
                                    if len(sub):
                                        return sub[val_col].iloc[0]
                                return None

                            lat_v = find_long(["LOCATION_LAT", "LATITUDE", "SITE_LAT"])
                            lon_v = find_long(["LOCATION_LONG", "LOCATION_LON", "LONGITUDE", "SITE_LON"])
                            igbp_v = find_long(["IGBP", "LAND_COVER_IGBP", "IGBP_CLASS"])

                            if lat_v is not None and lon_v is not None:
                                meta["tower_lat"] = float(lat_v)
                                meta["tower_lon"] = float(lon_v)
                                meta["igbp"] = str(igbp_v).strip().upper()[:3] if igbp_v is not None else ""
                                meta["metadata_source"] = member + "::long_csv"
                                return meta

                    except Exception:
                        pass

                except Exception:
                    continue

    except Exception as e:
        meta["metadata_source"] = f"zip_error:{e}"

    return meta


def load_repo_site_lists():
    """
    Fallback metadata from existing repo site lists if available.
    """
    possible = [
        Path("data/raw/towers/ameriflux_grassland_sites.csv"),
        Path("data/raw/towers/fluxnet2015_grassland_sites.csv"),
        Path("data/raw/towers/ameriflux_sites.csv"),
        Path("data/raw/towers/fluxnet_sites.csv"),
        Path("data/processed/tower_sites.csv"),
        Path("data/processed/ameriflux_grassland_sites.csv"),
        Path("data/processed/fluxnet2015_grassland_sites.csv"),
    ]

    rows = []
    for p in possible:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
            cols = list(df.columns)
            site_col = choose_col(cols, ["site", "site_id", "SITE_ID", "tower_id", "id"])
            lat_col = choose_col(cols, ["lat", "latitude", "LOCATION_LAT", "tower_lat"])
            lon_col = choose_col(cols, ["lon", "longitude", "LOCATION_LONG", "tower_lon"])
            igbp_col = choose_col(cols, ["igbp", "IGBP", "landcover", "land_cover"])
            if site_col and lat_col and lon_col:
                tmp = pd.DataFrame({
                    "site": df[site_col].astype(str),
                    "tower_lat": num(df[lat_col]),
                    "tower_lon": num(df[lon_col]),
                    "igbp": df[igbp_col].astype(str).str.upper() if igbp_col else "",
                    "metadata_source": str(p),
                })
                rows.append(tmp)
        except Exception as e:
            print(f"WARNING: could not read fallback site list {p}: {e}")

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out = out.dropna(subset=["tower_lat", "tower_lon"])
        out = out.drop_duplicates("site")
        return out

    return pd.DataFrame(columns=["site", "tower_lat", "tower_lon", "igbp", "metadata_source"])


# ---------------------------------------------------------------------
# Tower daily parsing
# ---------------------------------------------------------------------

def read_tower_daily_from_member(row):
    zip_path = Path(row["zip_path"])
    member = row["member"]
    site = row["site"]

    try:
        df = safe_read_member_csv(zip_path, member)
    except Exception as e:
        return pd.DataFrame(), f"read_error:{e}"

    df = clean_missing(df)
    df["date"] = parse_timestamp(df)
    df = df.dropna(subset=["date"]).copy()

    cols = list(df.columns)

    gpp_col = choose_col(cols, [
        "GPP_NT_VUT_REF",
        "GPP_NT_VUT_MEAN",
        "GPP_DT_VUT_REF",
        "GPP_DT_VUT_MEAN",
        "GPP_NT_CUT_REF",
        "GPP_DT_CUT_REF",
        "GPP",
    ])

    le_col = choose_col(cols, [
        "LE_F_MDS",
        "LE_CORR",
        "LE",
        "LE_F",
        "LE_PI",
    ])

    vpd_col = choose_col(cols, [
        "VPD_F_MDS",
        "VPD_F",
        "VPD_ERA",
        "VPD",
        "VPD_PI",
    ])

    p_col = choose_col(cols, [
        "P_F",
        "P",
        "P_ERA",
        "P_F_MDS",
        "PRECIP",
    ])

    swc_cols = cols_matching(cols, [
        r"^swc",
        r"soil.*water",
        r"soil.*moist",
    ])

    # Avoid QC columns for SWC.
    swc_cols = [c for c in swc_cols if not c.lower().endswith("_qc") and "qc" not in c.lower()]
    swc_col = swc_cols[0] if swc_cols else None

    le_qc_col = choose_col(cols, ["LE_F_MDS_QC", "LE_QC", "LE_F_QC"])
    vpd_qc_col = choose_col(cols, ["VPD_F_MDS_QC", "VPD_QC", "VPD_F_QC"])

    if not gpp_col or not le_col or not vpd_col:
        return pd.DataFrame(), f"missing_required_cols:gpp={gpp_col},le={le_col},vpd={vpd_col}"

    out = pd.DataFrame()
    out["site"] = site
    out["date"] = df["date"]
    out["year"] = out["date"].dt.year
    out["doy"] = out["date"].dt.dayofyear
    out["source_zip"] = str(zip_path)
    out["source_member"] = member

    out["gpp_raw"] = num(df[gpp_col])
    out["le_wm2"] = num(df[le_col])
    out["vpd_raw"] = num(df[vpd_col])

    if p_col:
        out["precip_mm_day"] = num(df[p_col])
    else:
        out["precip_mm_day"] = np.nan

    if swc_col:
        out["swc"] = num(df[swc_col])
    else:
        out["swc"] = np.nan

    if le_qc_col:
        out["le_qc"] = num(df[le_qc_col])
    else:
        out["le_qc"] = np.nan

    if vpd_qc_col:
        out["vpd_qc"] = num(df[vpd_qc_col])
    else:
        out["vpd_qc"] = np.nan

    out["gpp_col"] = gpp_col
    out["le_col"] = le_col
    out["vpd_col"] = vpd_col
    out["swc_col"] = swc_col or ""
    out["p_col"] = p_col or ""

    # FLUXNET daily GPP is usually already gC m-2 d-1.
    # If values look like micromol m-2 s-1, convert using 12.0107e-6 gC/umol and seconds/day.
    med_gpp = out["gpp_raw"].median(skipna=True)
    if pd.notna(med_gpp) and med_gpp > 80:
        out["gpp_gC_m2_day"] = out["gpp_raw"] * 12.0107e-6 * 86400.0
        out["gpp_unit_inferred"] = "umol_m2_s_to_gC_m2_day"
    else:
        out["gpp_gC_m2_day"] = out["gpp_raw"]
        out["gpp_unit_inferred"] = "assumed_gC_m2_day"

    # Convert latent heat W/m2 to ET mm/day.
    # 1 kg/m2 water = 1 mm water. lambda ≈ 2.45e6 J/kg.
    out["et_mm_day"] = out["le_wm2"] * 86400.0 / 2.45e6

    # FLUXNET/AmeriFlux VPD is commonly hPa; convert to kPa.
    # If the raw median is already tiny, keep as-is.
    med_vpd = out["vpd_raw"].median(skipna=True)
    if pd.notna(med_vpd) and med_vpd > 0:
        if med_vpd > 0.2:
            out["vpd_kpa"] = out["vpd_raw"] / 10.0
            out["vpd_unit_inferred"] = "hPa_to_kPa"
        else:
            out["vpd_kpa"] = out["vpd_raw"]
            out["vpd_unit_inferred"] = "assumed_kPa"
    else:
        out["vpd_kpa"] = np.nan
        out["vpd_unit_inferred"] = "unknown"

    # Basic physical filters.
    out.loc[out["gpp_gC_m2_day"] <= 0, "gpp_gC_m2_day"] = np.nan
    out.loc[out["et_mm_day"] <= 0, "et_mm_day"] = np.nan
    out.loc[out["vpd_kpa"] < 0, "vpd_kpa"] = np.nan

    out["wue_gC_per_mm"] = out["gpp_gC_m2_day"] / out["et_mm_day"]
    out["uwue_gC_kPa05_per_mm"] = out["gpp_gC_m2_day"] * np.sqrt(out["vpd_kpa"].clip(lower=0)) / out["et_mm_day"]

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(out)
    out = out.dropna(subset=["gpp_gC_m2_day", "et_mm_day", "vpd_kpa", "wue_gC_per_mm", "uwue_gC_kPa05_per_mm"]).copy()
    after = len(out)

    if after == 0:
        return pd.DataFrame(), "no_valid_daily_rows_after_basic_filters"

    return out, ""


def aggregate_to_8day(daily):
    d = daily.copy()
    d["year_start"] = pd.to_datetime(d["year"].astype(str) + "-01-01")
    d["modis_8day_index"] = ((d["doy"] - 1) // 8).astype(int)
    d["period_start"] = d["year_start"] + pd.to_timedelta(d["modis_8day_index"] * 8, unit="D")
    d["period_end"] = d["period_start"] + pd.to_timedelta(7, unit="D")

    rows = []
    group_cols = ["site", "period_start", "period_end", "year", "modis_8day_index"]

    for keys, g in d.groupby(group_cols):
        site, period_start, period_end, year, idx = keys
        n_days = len(g)
        if n_days < 4:
            continue

        gpp_sum = g["gpp_gC_m2_day"].sum(skipna=True)
        et_sum = g["et_mm_day"].sum(skipna=True)

        if not np.isfinite(gpp_sum) or not np.isfinite(et_sum) or et_sum <= 0 or gpp_sum <= 0:
            continue

        row = {
            "site": site,
            "period_start": period_start,
            "period_end": period_end,
            "year": int(year),
            "modis_8day_index": int(idx),
            "n_valid_days": int(n_days),
            "gpp_8day_gC_m2": float(gpp_sum),
            "et_8day_mm": float(et_sum),
            "vpd_8day_kpa_mean": float(g["vpd_kpa"].mean(skipna=True)),
            "vpd_8day_kpa_max": float(g["vpd_kpa"].max(skipna=True)),
            "swc_8day_mean": float(g["swc"].mean(skipna=True)) if g["swc"].notna().sum() else np.nan,
            "precip_8day_mm": float(g["precip_mm_day"].sum(skipna=True)) if g["precip_mm_day"].notna().sum() else np.nan,
            "wue_8day": float(gpp_sum / et_sum),
            "uwue_8day": float(gpp_sum * math.sqrt(max(g["vpd_kpa"].mean(skipna=True), 0)) / et_sum)
                if pd.notna(g["vpd_kpa"].mean(skipna=True)) else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["log_wue_8day"] = np.log(out["wue_8day"].where(out["wue_8day"] > 0))
    out["log_uwue_8day"] = np.log(out["uwue_8day"].where(out["uwue_8day"] > 0))
    return out


# ---------------------------------------------------------------------
# Stress construction and response fits
# ---------------------------------------------------------------------

def site_zscore(x):
    x = num(pd.Series(x))
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean(skipna=True)) / sd


def add_stress_indices(eight):
    out = eight.copy()
    out["vpd_z_site"] = np.nan
    out["swc_z_site"] = np.nan
    out["dry_swc_z_site"] = np.nan
    out["precip_z_site"] = np.nan
    out["dry_precip_z_site"] = np.nan
    out["stress_vpd_only"] = np.nan
    out["stress_compound_vpd_swc"] = np.nan
    out["stress_vpd_precip_proxy"] = np.nan
    out["growing_season_proxy"] = False

    for site, idx in out.groupby("site").groups.items():
        loc = list(idx)
        g = out.loc[loc].copy()

        vpd_z = site_zscore(g["vpd_8day_kpa_mean"])
        out.loc[loc, "vpd_z_site"] = vpd_z.values
        out.loc[loc, "stress_vpd_only"] = vpd_z.values

        if g["swc_8day_mean"].notna().sum() >= 12:
            swc_z = site_zscore(g["swc_8day_mean"])
            out.loc[loc, "swc_z_site"] = swc_z.values
            out.loc[loc, "dry_swc_z_site"] = (-swc_z).values
            out.loc[loc, "stress_compound_vpd_swc"] = ((vpd_z - swc_z) / 2.0).values

        if g["precip_8day_mm"].notna().sum() >= 12:
            p_z = site_zscore(g["precip_8day_mm"])
            out.loc[loc, "precip_z_site"] = p_z.values
            out.loc[loc, "dry_precip_z_site"] = (-p_z).values
            out.loc[loc, "stress_vpd_precip_proxy"] = ((vpd_z - p_z) / 2.0).values

        # Growing-season proxy: keep periods above site-specific low GPP threshold.
        # Avoid using all dormant-season windows.
        if g["gpp_8day_gC_m2"].notna().sum() >= 8:
            thr = g["gpp_8day_gC_m2"].quantile(0.25)
            out.loc[loc, "growing_season_proxy"] = g["gpp_8day_gC_m2"] > thr

    return out


def fit_piecewise_grid(x, y, min_n=18, min_side=6, n_boot=300, seed=42):
    x = num(pd.Series(x)).values
    y = num(pd.Series(y)).values

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < min_n:
        return None

    if np.nanmax(x) - np.nanmin(x) < 1.0:
        return None

    qs = np.linspace(0.35, 0.75, 17)
    knots = np.unique(np.nanquantile(x, qs))

    best = None

    for k in knots:
        left = x <= k
        right = x > k
        if left.sum() < min_side or right.sum() < min_side:
            continue

        X = np.column_stack([
            np.ones(len(x)),
            x,
            np.maximum(0, x - k)
        ])

        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            rss = float(np.sum((y - pred) ** 2))
            tss = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - rss / tss if tss > 0 else np.nan
        except Exception:
            continue

        pre = float(beta[1])
        change = float(beta[2])
        post = float(beta[1] + beta[2])

        candidate = {
            "breakpoint": float(k),
            "pre_slope": pre,
            "slope_change": change,
            "post_slope": post,
            "rss": rss,
            "r2": r2,
            "n_obs": int(len(x)),
            "n_left": int(left.sum()),
            "n_right": int(right.sum()),
        }

        if best is None or candidate["rss"] < best["rss"]:
            best = candidate

    if best is None:
        return None

    rng = np.random.default_rng(seed)
    boots = []

    n = len(x)
    for b in range(n_boot):
        ii = rng.choice(np.arange(n), size=n, replace=True)
        xb = x[ii]
        yb = y[ii]

        try:
            knots_b = np.unique(np.nanquantile(xb, qs))
        except Exception:
            continue

        best_b = None
        for k in knots_b:
            left = xb <= k
            right = xb > k
            if left.sum() < min_side or right.sum() < min_side:
                continue

            Xb = np.column_stack([
                np.ones(len(xb)),
                xb,
                np.maximum(0, xb - k)
            ])

            try:
                beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
                pred = Xb @ beta
                rss = float(np.sum((yb - pred) ** 2))
            except Exception:
                continue

            row = {
                "breakpoint": float(k),
                "pre_slope": float(beta[1]),
                "slope_change": float(beta[2]),
                "post_slope": float(beta[1] + beta[2]),
                "rss": rss,
            }

            if best_b is None or row["rss"] < best_b["rss"]:
                best_b = row

        if best_b:
            boots.append(best_b)

    boot = pd.DataFrame(boots)

    if len(boot) >= 20:
        for col in ["breakpoint", "pre_slope", "slope_change", "post_slope"]:
            best[col + "_ci_low"] = float(boot[col].quantile(0.025))
            best[col + "_ci_high"] = float(boot[col].quantile(0.975))
            best[col + "_boot_sd"] = float(boot[col].std())
    else:
        for col in ["breakpoint", "pre_slope", "slope_change", "post_slope"]:
            best[col + "_ci_low"] = np.nan
            best[col + "_ci_high"] = np.nan
            best[col + "_boot_sd"] = np.nan

    # Class from median fit and uncertainty.
    pre = best["pre_slope"]
    post = best["post_slope"]
    change = best["slope_change"]

    post_low = best.get("post_slope_ci_low", np.nan)
    post_high = best.get("post_slope_ci_high", np.nan)
    change_high = best.get("slope_change_ci_high", np.nan)

    if np.isfinite(post_low) and post_low > 0:
        response_class = "enhancement"
    elif np.isfinite(post_high) and post_high < 0 and np.isfinite(change_high) and change_high < 0:
        response_class = "breakdown"
    elif change < 0 and post < 0.25:
        response_class = "saturation"
    elif post > 0 and change >= 0:
        response_class = "enhancement"
    else:
        response_class = "inconclusive"

    best["response_class"] = response_class

    if len(boot):
        boot_classes = []
        for _, r in boot.iterrows():
            if r["post_slope"] < 0 and r["slope_change"] < 0:
                boot_classes.append("breakdown")
            elif r["slope_change"] < 0 and r["post_slope"] < 0.25:
                boot_classes.append("saturation")
            elif r["post_slope"] > 0:
                boot_classes.append("enhancement")
            else:
                boot_classes.append("inconclusive")
        best["p_tower_saturation_breakdown"] = float(np.mean([c in ["saturation", "breakdown"] for c in boot_classes]))
        best["p_tower_enhancement"] = float(np.mean([c == "enhancement" for c in boot_classes]))
    else:
        best["p_tower_saturation_breakdown"] = np.nan
        best["p_tower_enhancement"] = np.nan

    return best


def classify_tower_sites(eight, n_boot=300):
    rows = []
    stress_methods = [
        ("compound_vpd_swc", "stress_compound_vpd_swc"),
        ("vpd_precip_proxy", "stress_vpd_precip_proxy"),
        ("vpd_only", "stress_vpd_only"),
    ]

    metrics = [
        ("uwue", "log_uwue_8day"),
        ("wue", "log_wue_8day"),
    ]

    for site, g0 in eight.groupby("site"):
        g0 = g0.copy()

        # Primary fits use growing-season proxy if enough records remain.
        g_gs = g0[g0["growing_season_proxy"]].copy()
        if len(g_gs) >= 18:
            g_use = g_gs
            growing_season_used = "gpp_top75_percent"
        else:
            g_use = g0
            growing_season_used = "all_valid_periods"

        for metric_name, metric_col in metrics:
            for stress_name, stress_col in stress_methods:
                if stress_col not in g_use.columns:
                    continue

                usable = g_use[[stress_col, metric_col]].dropna()
                if len(usable) < 18:
                    continue

                fit = fit_piecewise_grid(
                    usable[stress_col],
                    usable[metric_col],
                    min_n=18,
                    min_side=6,
                    n_boot=n_boot,
                    seed=abs(hash(site + metric_name + stress_name)) % (2**32 - 1)
                )

                if fit is None:
                    continue

                row = {
                    "site": site,
                    "tower_metric": metric_name,
                    "stress_method": stress_name,
                    "stress_col": stress_col,
                    "metric_col": metric_col,
                    "growing_season_used": growing_season_used,
                    "n_site_8day_total": int(len(g0)),
                    "n_fit_8day": int(len(usable)),
                    "n_years": int(g0["year"].nunique()),
                    "has_swc": bool(g0["swc_8day_mean"].notna().sum() >= 12),
                    "has_precip": bool(g0["precip_8day_mm"].notna().sum() >= 12),
                    "vpd_mean_kpa": float(g0["vpd_8day_kpa_mean"].mean(skipna=True)),
                    "vpd_p90_kpa": float(g0["vpd_8day_kpa_mean"].quantile(0.90)),
                    "gpp_sum": float(g0["gpp_8day_gC_m2"].sum(skipna=True)),
                    "et_sum": float(g0["et_8day_mm"].sum(skipna=True)),
                }
                row.update(fit)
                rows.append(row)

    fits = pd.DataFrame(rows)

    if fits.empty:
        return fits

    # Choose one primary fit per site:
    # prefer uWUE, then compound_vpd_swc, then precip proxy, then VPD-only.
    metric_rank = {"uwue": 0, "wue": 1}
    stress_rank = {"compound_vpd_swc": 0, "vpd_precip_proxy": 1, "vpd_only": 2}

    fits["metric_rank"] = fits["tower_metric"].map(metric_rank).fillna(99)
    fits["stress_rank"] = fits["stress_method"].map(stress_rank).fillna(99)
    fits["primary_rank"] = fits["metric_rank"] * 10 + fits["stress_rank"]

    fits["is_primary_tower_fit"] = False
    primary_idx = fits.sort_values(["site", "primary_rank", "n_fit_8day"], ascending=[True, True, False]).groupby("site").head(1).index
    fits.loc[primary_idx, "is_primary_tower_fit"] = True

    return fits


# ---------------------------------------------------------------------
# Existing satellite nearest-match diagnostic
# ---------------------------------------------------------------------

def load_satellite_points():
    if PH16_POINTS.exists():
        sat = read_csv(PH16_POINTS, required=True)
        source = str(PH16_POINTS)
    elif PH8_LATENT.exists():
        sat = read_csv(PH8_LATENT, required=True)
        source = str(PH8_LATENT)
    else:
        return pd.DataFrame(), ""

    if "point_id" not in sat.columns:
        sat["point_id"] = sat.index.astype(str)

    for c in ["lat", "lon"]:
        if c not in sat.columns:
            return pd.DataFrame(), source

    return sat, source


def nearest_satellite_matches(site_meta, sat):
    rows = []
    if site_meta.empty or sat.empty:
        return pd.DataFrame()

    sat2 = sat.dropna(subset=["lat", "lon"]).copy()
    if sat2.empty:
        return pd.DataFrame()

    sat_lat = num(sat2["lat"]).values
    sat_lon = num(sat2["lon"]).values

    for _, r in site_meta.dropna(subset=["tower_lat", "tower_lon"]).iterrows():
        dist = haversine_km(float(r["tower_lat"]), float(r["tower_lon"]), sat_lat, sat_lon)
        if len(dist) == 0:
            continue
        j = int(np.nanargmin(dist))
        sr = sat2.iloc[j]

        row = {
            "site": r["site"],
            "tower_lat": r["tower_lat"],
            "tower_lon": r["tower_lon"],
            "nearest_point_id": sr.get("point_id", ""),
            "nearest_point_lat": sr.get("lat", np.nan),
            "nearest_point_lon": sr.get("lon", np.nan),
            "distance_km": float(dist[j]),
            "match_within_25km": bool(dist[j] <= 25),
            "match_within_50km": bool(dist[j] <= 50),
            "match_within_100km": bool(dist[j] <= 100),
        }

        for c in [
            "latent_response_class",
            "latent_satbreak_probability",
            "latent_post_slope",
            "latent_slope_change",
            "event_limitation_hotspot",
            "eco_ecoregion",
            "eco_biome",
            "hydroclimatic_regime",
            "high_vpd_gt_2p26",
            "region_Sahel",
        ]:
            if c in sr.index:
                row["satellite_" + c] = sr[c]

        rows.append(row)

    return pd.DataFrame(rows)


def compare_tower_satellite(tower_fits, matches):
    if tower_fits.empty or matches.empty:
        return pd.DataFrame()

    primary = tower_fits[tower_fits["is_primary_tower_fit"]].copy()
    comp = primary.merge(matches, on="site", how="left")

    if "satellite_latent_post_slope" in comp.columns:
        comp["tower_post_slope_sign"] = np.sign(num(comp["post_slope"]))
        comp["sat_post_slope_sign"] = np.sign(num(comp["satellite_latent_post_slope"]))
        comp["post_slope_sign_agreement"] = comp["tower_post_slope_sign"].eq(comp["sat_post_slope_sign"])

    if "satellite_latent_slope_change" in comp.columns:
        comp["tower_slope_change_sign"] = np.sign(num(comp["slope_change"]))
        comp["sat_slope_change_sign"] = np.sign(num(comp["satellite_latent_slope_change"]))
        comp["slope_change_sign_agreement"] = comp["tower_slope_change_sign"].eq(comp["sat_slope_change_sign"])

    comp["usable_for_rigorous_validation_now"] = comp["match_within_50km"].fillna(False)

    return comp


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not CANDIDATE_FILE.exists():
        die(f"Missing candidate inventory: {CANDIDATE_FILE}")

    cand = read_csv(CANDIDATE_FILE)
    inv = read_csv(INVENTORY_FILE, required=False)

    print("")
    print("Loaded broad tower candidate inventory")
    print("candidate rows:", len(cand))
    print("unique sites:", cand["site"].nunique() if "site" in cand.columns else "unknown")

    required = ["zip_path", "zip_name", "site", "member", "candidate_wue_file"]
    for c in required:
        if c not in cand.columns:
            die(f"Candidate file missing column: {c}")

    # Fix missing site IDs if needed.
    cand["site"] = cand.apply(
        lambda r: r["site"] if isinstance(r["site"], str) and r["site"].strip()
        else parse_site_id_from_zip_or_member(str(r["zip_name"]), str(r["member"])),
        axis=1
    )

    cand["member"] = cand["member"].astype(str)
    cand["zip_path"] = cand["zip_path"].astype(str)

    cand["is_daily"] = cand["member"].str.contains("_FLUXMET_DD_", case=False, regex=False) | cand["member"].str.contains("_DD_", case=False, regex=False)
    cand["is_subdaily"] = cand["member"].str.contains("_HH_", case=False, regex=False) | cand["member"].str.contains("_HR_", case=False, regex=False)
    cand["is_weekly"] = cand["member"].str.contains("_WW_", case=False, regex=False)
    cand["is_monthly"] = cand["member"].str.contains("_MM_", case=False, regex=False)
    cand["is_yearly"] = cand["member"].str.contains("_YY_", case=False, regex=False)

    usable = cand[
        cand["candidate_wue_file"].astype(str).str.lower().isin(["true", "1", "yes"]) |
        (cand["candidate_wue_file"] == True)
    ].copy()

    daily = usable[usable["is_daily"]].copy()

    if daily.empty:
        die("No daily candidate files found. The scanner found candidates but none matched DD/daily.")

    # Prefer files with more columns and soil moisture.
    if "has_soil_moisture" in daily.columns:
        daily["has_soil_moisture_sort"] = daily["has_soil_moisture"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        daily["has_soil_moisture_sort"] = False

    if "n_columns" not in daily.columns:
        daily["n_columns"] = 0

    selected = (
        daily.sort_values(["site", "has_soil_moisture_sort", "n_columns"], ascending=[True, False, False])
        .drop_duplicates("site", keep="first")
        .reset_index(drop=True)
    )

    selected["daily_file_selected_reason"] = "preferred_daily_FLUXMET_DD_with_required_WUE_columns"

    save_csv(cand, TAB / "Table80_all_candidate_wue_members_loaded.csv")
    save_csv(selected, TAB / "Table81_selected_daily_tower_files.csv")

    # Metadata extraction.
    fallback_meta = load_repo_site_lists()
    fallback_map = fallback_meta.set_index("site").to_dict(orient="index") if not fallback_meta.empty else {}

    meta_rows = []
    for _, r in selected.iterrows():
        site = r["site"]
        meta = extract_metadata_from_zip(r["zip_path"], site)

        if (pd.isna(meta["tower_lat"]) or pd.isna(meta["tower_lon"])) and site in fallback_map:
            fb = fallback_map[site]
            meta["tower_lat"] = fb.get("tower_lat", np.nan)
            meta["tower_lon"] = fb.get("tower_lon", np.nan)
            meta["igbp"] = fb.get("igbp", meta.get("igbp", ""))
            meta["metadata_source"] = "fallback_site_list:" + str(fb.get("metadata_source", ""))

        meta["zip_path"] = r["zip_path"]
        meta["zip_name"] = r["zip_name"]
        meta["member"] = r["member"]
        meta_rows.append(meta)

    site_meta = pd.DataFrame(meta_rows)

    # IGBP class flags.
    site_meta["igbp"] = site_meta["igbp"].fillna("").astype(str).str.upper().str.slice(0, 3)
    grassland_like = {"GRA", "SAV", "WSA", "OSH", "CSH"}
    strict_grass = {"GRA"}
    savanna_grass_extension = {"SAV", "WSA", "OSH", "CSH"}

    site_meta["is_strict_grassland_gra"] = site_meta["igbp"].isin(strict_grass)
    site_meta["is_grassland_savanna_extension"] = site_meta["igbp"].isin(grassland_like)
    site_meta["igbp_missing"] = site_meta["igbp"].eq("") | site_meta["igbp"].eq("NAN")

    # Keep unknowns for now; they are flagged and can be removed later.
    site_meta["passes_landcover_screen_lenient"] = site_meta["is_grassland_savanna_extension"] | site_meta["igbp_missing"]

    save_csv(site_meta, TAB / "Table82_tower_site_metadata_extracted.csv")

    # Read daily tower data.
    daily_rows = []
    read_errors = []

    for _, r in selected.iterrows():
        site = r["site"]
        df, err = read_tower_daily_from_member(r)
        if err:
            read_errors.append({
                "site": site,
                "zip_path": r["zip_path"],
                "member": r["member"],
                "error": err,
            })
            continue
        daily_rows.append(df)
        print(f"READ {site}: daily rows {len(df)}")

    if daily_rows:
        tower_daily = pd.concat(daily_rows, ignore_index=True)
    else:
        tower_daily = pd.DataFrame()

    read_errors_df = pd.DataFrame(read_errors)

    save_csv(read_errors_df, TAB / "Table83_tower_read_errors.csv")

    if tower_daily.empty:
        die("No usable tower daily data after parsing candidate files.")

    tower_daily = tower_daily.merge(
        site_meta[[
            "site", "tower_lat", "tower_lon", "igbp",
            "is_strict_grassland_gra", "is_grassland_savanna_extension",
            "igbp_missing", "passes_landcover_screen_lenient",
            "metadata_source"
        ]],
        on="site",
        how="left"
    )

    save_csv(tower_daily, TAB / "Table84_tower_daily_wue_fluxes.csv")

    # Site-year quality summary.
    site_year = (
        tower_daily.groupby(["site", "year"])
        .agg(
            n_daily=("date", "size"),
            gpp_mean=("gpp_gC_m2_day", "mean"),
            et_sum=("et_mm_day", "sum"),
            vpd_mean=("vpd_kpa", "mean"),
            swc_coverage=("swc", lambda x: float(x.notna().mean())),
            precip_coverage=("precip_mm_day", lambda x: float(x.notna().mean())),
        )
        .reset_index()
    )

    site_year["passes_min_daily_rows"] = site_year["n_daily"] >= 90

    site_summary = (
        site_year.groupby("site")
        .agg(
            years_total=("year", "nunique"),
            years_with_90_daily_rows=("passes_min_daily_rows", "sum"),
            n_daily_total=("n_daily", "sum"),
            mean_daily_rows_per_year=("n_daily", "mean"),
            mean_vpd_kpa=("vpd_mean", "mean"),
            mean_swc_coverage=("swc_coverage", "mean"),
            mean_precip_coverage=("precip_coverage", "mean"),
        )
        .reset_index()
    )

    site_summary = site_summary.merge(
        site_meta[[
            "site", "tower_lat", "tower_lon", "igbp",
            "is_strict_grassland_gra", "is_grassland_savanna_extension",
            "igbp_missing",
            "passes_landcover_screen_lenient",
            "metadata_source"
        ]],
        on="site",
        how="left"
    )

    site_summary["passes_record_length_3yr_lenient"] = site_summary["years_with_90_daily_rows"] >= 3
    site_summary["usable_for_tower_response_lenient"] = site_summary["passes_record_length_3yr_lenient"] & site_summary["passes_landcover_screen_lenient"]

    save_csv(site_year, TAB / "Table85_tower_site_year_quality.csv")
    save_csv(site_summary, TAB / "Table86_tower_usable_site_summary.csv")

    # Build 8-day.
    tower_8day = aggregate_to_8day(tower_daily)
    if tower_8day.empty:
        die("No 8-day tower records produced.")

    tower_8day = add_stress_indices(tower_8day)

    tower_8day = tower_8day.merge(
        site_summary[[
            "site", "tower_lat", "tower_lon", "igbp",
            "is_strict_grassland_gra", "is_grassland_savanna_extension",
            "igbp_missing",
            "passes_landcover_screen_lenient",
            "passes_record_length_3yr_lenient",
            "usable_for_tower_response_lenient",
            "metadata_source"
        ]],
        on="site",
        how="left"
    )

    save_csv(tower_8day, TAB / "Table87_tower_8day_wue_stress.csv")

    # Fit response shapes.
    tower_fits_all = classify_tower_sites(tower_8day, n_boot=300)

    if tower_fits_all.empty:
        save_csv(tower_fits_all, TAB / "Table88_tower_response_phenotypes_all_fits.csv")
        die("No tower response fits produced. Check Table87 for record counts and stress coverage.")

    tower_fits_all = tower_fits_all.merge(
        site_summary[[
            "site", "tower_lat", "tower_lon", "igbp",
            "is_strict_grassland_gra", "is_grassland_savanna_extension",
            "igbp_missing",
            "usable_for_tower_response_lenient",
            "metadata_source"
        ]],
        on="site",
        how="left"
    )

    save_csv(tower_fits_all, TAB / "Table88_tower_response_phenotypes_all_fits.csv")

    tower_primary = tower_fits_all[tower_fits_all["is_primary_tower_fit"]].copy()
    save_csv(tower_primary, TAB / "Table89_tower_response_phenotypes_primary_by_site.csv")

    # Summary by IGBP/regime.
    by_class = (
        tower_primary.groupby(["response_class"])
        .agg(
            n_sites=("site", "nunique"),
            median_post_slope=("post_slope", "median"),
            median_slope_change=("slope_change", "median"),
            median_p_satbreak=("p_tower_saturation_breakdown", "median"),
        )
        .reset_index()
        .sort_values("n_sites", ascending=False)
    )
    save_csv(by_class, TAB / "Table90_tower_response_class_summary.csv")

    by_igbp = (
        tower_primary.groupby(["igbp"])
        .agg(
            n_sites=("site", "nunique"),
            n_grassland_extension=("is_grassland_savanna_extension", "sum"),
            median_post_slope=("post_slope", "median"),
            median_slope_change=("slope_change", "median"),
            median_p_satbreak=("p_tower_saturation_breakdown", "median"),
        )
        .reset_index()
        .sort_values("n_sites", ascending=False)
    )
    save_csv(by_igbp, TAB / "Table91_tower_response_summary_by_igbp.csv")

    # Nearest satellite diagnostic.
    sat, sat_source = load_satellite_points()
    matches = nearest_satellite_matches(site_meta, sat)
    save_csv(matches, TAB / "Table92_existing_satellite_point_nearest_tower_matches.csv")

    comp = compare_tower_satellite(tower_primary, matches)
    save_csv(comp, TAB / "Table93_existing_satellite_tower_provisional_comparison.csv")

    # Product arbitration only possible if real close matches exist.
    close_comp = comp[comp.get("usable_for_rigorous_validation_now", False) == True].copy() if not comp.empty else pd.DataFrame()
    if not close_comp.empty:
        agreement_cols = [c for c in ["post_slope_sign_agreement", "slope_change_sign_agreement"] if c in close_comp.columns]
        arbitration_rows = []
        for c in agreement_cols:
            arbitration_rows.append({
                "comparison_metric": c,
                "n_close_matches": int(close_comp[c].notna().sum()),
                "agreement_fraction": float(close_comp[c].mean()) if close_comp[c].notna().sum() else np.nan,
            })
        arbitration = pd.DataFrame(arbitration_rows)
    else:
        arbitration = pd.DataFrame([{
            "comparison_metric": "none",
            "n_close_matches": 0,
            "agreement_fraction": np.nan,
            "interpretation": "No rigorous tower-satellite arbitration from existing 199 satellite sample; tower-centered extraction required."
        }])

    save_csv(arbitration, TAB / "Table94_product_arbitration_status_from_existing_matches.csv")

    # Tower-centered satellite extraction target list.
    extraction_targets = site_summary[
        site_summary["usable_for_tower_response_lenient"]
    ].copy()

    if extraction_targets.empty:
        extraction_targets = site_summary.copy()

    extraction_targets = extraction_targets.dropna(subset=["tower_lat", "tower_lon"]).copy()

    extraction_targets["target_id"] = extraction_targets["site"]
    extraction_targets["extract_lat"] = extraction_targets["tower_lat"]
    extraction_targets["extract_lon"] = extraction_targets["tower_lon"]
    extraction_targets["recommended_satellite_window"] = "tower_pixel_plus_3x3_sensitivity"
    extraction_targets["target_reason"] = np.where(
        extraction_targets["usable_for_tower_response_lenient"],
        "usable_tower_response_site",
        "metadata_site_needs_quality_review"
    )

    target_cols = [
        "target_id", "site", "extract_lat", "extract_lon", "igbp",
        "is_strict_grassland_gra", "is_grassland_savanna_extension",
        "years_total", "years_with_90_daily_rows", "n_daily_total",
        "mean_vpd_kpa", "mean_swc_coverage", "mean_precip_coverage",
        "recommended_satellite_window", "target_reason"
    ]
    target_cols = [c for c in target_cols if c in extraction_targets.columns]

    save_csv(extraction_targets[target_cols], TAB / "Table95_tower_centered_satellite_extraction_targets.csv")

    # Also write an AppEEARS-style coordinate CSV.
    appeears = extraction_targets[["site", "extract_lat", "extract_lon"]].rename(
        columns={"site": "id", "extract_lat": "latitude", "extract_lon": "longitude"}
    )
    save_csv(appears := appeears, TAB / "tower_centered_coordinates_for_satellite_extraction.csv")

    # Figures.
    try:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        counts = tower_primary["response_class"].value_counts()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("Tower sites")
        ax.set_title("Tower response classes from broad AmeriFlux/FLUXNET inventory")
        ax.tick_params(axis="x", rotation=25)
        savefig(FIG / "Figure1_tower_response_class_counts.png")
    except Exception as e:
        print("WARNING: Figure1 failed:", e)

    try:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(tower_primary["post_slope"], tower_primary["slope_change"], s=55, alpha=0.8)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Tower high-stress/post-transition slope")
        ax.set_ylabel("Tower slope change")
        ax.set_title("Tower WUE response phenotype")
        savefig(FIG / "Figure2_tower_post_slope_vs_slope_change.png")
    except Exception as e:
        print("WARNING: Figure2 failed:", e)

    try:
        if not matches.empty:
            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.scatter(matches["tower_lon"], matches["tower_lat"], label="Tower sites", s=55)
            if not sat.empty:
                ax.scatter(sat["lon"], sat["lat"], label="Existing satellite sample points", s=20, alpha=0.5)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Tower sites vs existing 199 satellite sample points")
            ax.legend(frameon=False)
            savefig(FIG / "Figure3_tower_sites_vs_existing_satellite_points.png")
    except Exception as e:
        print("WARNING: Figure3 failed:", e)

    try:
        if not comp.empty and "satellite_latent_post_slope" in comp.columns:
            plot = comp.dropna(subset=["post_slope", "satellite_latent_post_slope"]).copy()
            if len(plot):
                fig, ax = plt.subplots(figsize=(5.5, 5))
                ax.scatter(plot["satellite_latent_post_slope"], plot["post_slope"], s=55)
                ax.axhline(0, linestyle="--", linewidth=1)
                ax.axvline(0, linestyle="--", linewidth=1)
                ax.set_xlabel("Nearest satellite latent post-slope")
                ax.set_ylabel("Tower post-slope")
                ax.set_title("Provisional nearest-neighbor tower–satellite comparison")
                savefig(FIG / "Figure4_provisional_tower_satellite_post_slope.png")
    except Exception as e:
        print("WARNING: Figure4 failed:", e)

    # Verdict.
    n_candidate_sites = int(cand["site"].nunique())
    n_selected_daily = int(selected["site"].nunique())
    n_sites_with_metadata = int(site_meta.dropna(subset=["tower_lat", "tower_lon"])["site"].nunique())
    n_sites_daily = int(tower_daily["site"].nunique())
    n_sites_8day = int(tower_8day["site"].nunique())
    n_fit_sites = int(tower_primary["site"].nunique())
    n_close_50 = int(matches["match_within_50km"].sum()) if not matches.empty else 0
    n_extraction_targets = int(len(extraction_targets))

    class_counts = tower_primary["response_class"].value_counts().to_dict()

    verdict = {
        "candidate_sites_from_broad_inventory": n_candidate_sites,
        "selected_daily_sites": n_selected_daily,
        "sites_with_metadata_coordinates": n_sites_with_metadata,
        "sites_with_daily_wue": n_sites_daily,
        "sites_with_8day_wue": n_sites_8day,
        "sites_with_primary_tower_response_fit": n_fit_sites,
        "primary_tower_response_class_counts": class_counts,
        "existing_satellite_source": sat_source,
        "existing_satellite_points": int(len(sat)) if not sat.empty else 0,
        "existing_satellite_matches_within_50km": n_close_50,
        "tower_centered_satellite_extraction_targets": n_extraction_targets,
        "can_claim_fully_tower_backed_satellite_phenotype_now": bool(n_close_50 >= 5),
        "can_claim_tower_flux_phenotype_now": bool(n_fit_sites >= 10),
        "requires_tower_centered_satellite_extraction": bool(n_close_50 < 5),
    }

    (OUT / "phase17_tower_validation_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    # Manuscript text.
    if verdict["requires_tower_centered_satellite_extraction"]:
        validation_status = (
            "The broad tower inventory successfully supports construction of an independent tower-observed "
            "ecosystem WUE response phenotype, but the existing 199 satellite sample is not sufficient for "
            "rigorous tower-satellite validation because too few tower sites are colocated with sampled satellite points. "
            "The next required step is tower-centered satellite extraction at the tower coordinates listed in "
            "`Table95_tower_centered_satellite_extraction_targets.csv`."
        )
    else:
        validation_status = (
            "The broad tower inventory produced colocated tower-satellite comparisons from the existing satellite sample. "
            "These comparisons can be used as provisional tower validation, but tower-centered extraction is still preferred "
            "for final manuscript-grade validation."
        )

    class_counts_text = pd.Series(class_counts).to_string() if class_counts else "No tower classes."

    text_lines = []
    text_lines.append("# Phase 17 broad tower validation verdict")
    text_lines.append("")
    text_lines.append("## What changed")
    text_lines.append("")
    text_lines.append("The old tower checker was too narrow. This phase used the broad AmeriFlux/FLUXNET candidate inventory and preferred daily `_FLUXMET_DD_` members with GPP, LE, and VPD columns.")
    text_lines.append("")
    text_lines.append("## Summary numbers")
    text_lines.append("")
    text_lines.append(f"- Candidate sites from broad inventory: `{n_candidate_sites}`")
    text_lines.append(f"- Selected daily files/sites: `{n_selected_daily}`")
    text_lines.append(f"- Sites with metadata coordinates: `{n_sites_with_metadata}`")
    text_lines.append(f"- Sites with parsed daily tower WUE: `{n_sites_daily}`")
    text_lines.append(f"- Sites with 8-day tower WUE: `{n_sites_8day}`")
    text_lines.append(f"- Sites with primary tower response fits: `{n_fit_sites}`")
    text_lines.append(f"- Existing satellite sample points: `{verdict['existing_satellite_points']}`")
    text_lines.append(f"- Existing tower-satellite matches within 50 km: `{n_close_50}`")
    text_lines.append(f"- Tower-centered satellite extraction targets: `{n_extraction_targets}`")
    text_lines.append("")
    text_lines.append("## Tower response class counts")
    text_lines.append("")
    text_lines.append("```text")
    text_lines.append(class_counts_text)
    text_lines.append("```")
    text_lines.append("")
    text_lines.append("## Validation status")
    text_lines.append("")
    text_lines.append(validation_status)
    text_lines.append("")
    text_lines.append("## What this means for the thesis")
    text_lines.append("")
    text_lines.append("The tower data allow a real tower-observed ecosystem flux phenotype to be constructed. The safest thesis is not that all grasslands show universal WUE breakdown. Instead, the defensible thesis is that satellite-derived WUE response is heterogeneous across grasslands, partially structured by hydraulic/rooting traits and specific grassland regimes, and now testable against tower-observed carbon-water flux behavior.")
    text_lines.append("")
    text_lines.append("## Manuscript-safe claim")
    text_lines.append("")
    text_lines.append("Tower data provide an independent ecosystem-flux check on the satellite-derived WUE response phenotype. If tower and satellite response signs/classes agree at colocated or tower-centered extraction points, the claim can be strengthened from a satellite-only trait association to a tower-checked, trait-consistent ecosystem response framework.")
    text_lines.append("")
    text_lines.append("## Claim to avoid")
    text_lines.append("")
    text_lines.append("Do not claim that xylem vulnerability causally proves universal WUE breakdown. The tower validation should be used to test whether the satellite-derived trait-consistent phenotype corresponds to observed tower flux behavior.")
    text_lines.append("")
    text_lines.append("## Required next step")
    text_lines.append("")
    text_lines.append("If the existing 199 satellite sample is not close enough to the tower sites, use `Table95_tower_centered_satellite_extraction_targets.csv` to extract satellite products at tower coordinates.")
    text = chr(10).join(text_lines) + chr(10)

    save_text(text, TXT / "phase17_broad_tower_validation_verdict.md")
    save_text(text, OUT / "README_phase17_broad_tower_validation.md")

    summary_rows = [
        {"metric": "candidate_sites_from_broad_inventory", "value": n_candidate_sites},
        {"metric": "selected_daily_sites", "value": n_selected_daily},
        {"metric": "sites_with_metadata_coordinates", "value": n_sites_with_metadata},
        {"metric": "sites_with_daily_wue", "value": n_sites_daily},
        {"metric": "sites_with_8day_wue", "value": n_sites_8day},
        {"metric": "sites_with_primary_tower_response_fit", "value": n_fit_sites},
        {"metric": "existing_satellite_points", "value": verdict["existing_satellite_points"]},
        {"metric": "existing_tower_satellite_matches_within_50km", "value": n_close_50},
        {"metric": "tower_centered_satellite_extraction_targets", "value": n_extraction_targets},
        {"metric": "can_claim_tower_flux_phenotype_now", "value": verdict["can_claim_tower_flux_phenotype_now"]},
        {"metric": "requires_tower_centered_satellite_extraction", "value": verdict["requires_tower_centered_satellite_extraction"]},
    ]
    save_csv(pd.DataFrame(summary_rows), TAB / "Table96_phase17_verdict_summary.csv")

    print("")
    print("DONE Phase 17 broad tower validation.")
    print("")
    print("SUMMARY:")
    for k, v in verdict.items():
        print(f"{k}: {v}")
    print("")
    print("MAIN OUTPUTS:")
    print(TAB / "Table89_tower_response_phenotypes_primary_by_site.csv")
    print(TAB / "Table93_existing_satellite_tower_provisional_comparison.csv")
    print(TAB / "Table95_tower_centered_satellite_extraction_targets.csv")
    print(OUT / "README_phase17_broad_tower_validation.md")


if __name__ == "__main__":
    main()
