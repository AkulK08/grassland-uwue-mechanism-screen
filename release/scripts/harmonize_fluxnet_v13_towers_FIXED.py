from pathlib import Path
import zipfile
import re
import numpy as np
import pandas as pd

ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")
ZIP_DIR = ROOT / "data/raw/towers/_downloads/fluxnet2015"
OUT_DIR = ROOT / "data/raw/towers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA = [
    "tower_id", "source_network", "date", "lat", "lon", "igbp",
    "GPP_NT_VUT_REF", "LE_F_MDS", "ET", "VPD", "soil_moisture",
    "energy_balance_closure", "gapfill_fraction"
]

GRASSLIKE = {"GRA", "SAV", "WSA", "OSH", "CSH", "CRO"}

def site_id_from_zip(name):
    m = re.search(r'^(AMF|ICOS|EUF|JPF|FLX)_([A-Z]{2}-[A-Za-z0-9]+)_', name)
    if m:
        return m.group(2), m.group(1)
    m = re.search(r'([A-Z]{2}-[A-Za-z0-9]+)', name)
    if m:
        return m.group(1), "UNKNOWN"
    return Path(name).stem[:30], "UNKNOWN"

def numeric(x):
    return pd.to_numeric(x, errors="coerce").replace(-9999, np.nan).replace(-9999.0, np.nan)

def find_col(cols, exact=(), contains_any=()):
    lower = {c.lower(): c for c in cols}
    for e in exact:
        if e.lower() in lower:
            return lower[e.lower()]
    for c in cols:
        cl = c.lower()
        if any(s.lower() in cl for s in contains_any):
            if "qc" not in cl and "randunc" not in cl and "se" not in cl:
                return c
    return None

def parse_bif_metadata(z, bif_member):
    meta = {"lat": np.nan, "lon": np.nan, "igbp": ""}
    if bif_member is None:
        return meta

    try:
        with z.open(bif_member) as f:
            bif = pd.read_csv(f, low_memory=False)
    except Exception:
        return meta

    if "VARIABLE" not in bif.columns or "DATAVALUE" not in bif.columns:
        return meta

    var = bif["VARIABLE"].astype(str).str.upper()

    def get_var(names, contains=None):
        hits = []
        for n in names:
            hits.append(bif[var == n.upper()])
        if contains:
            hits.append(bif[var.str.contains(contains.upper(), na=False)])
        hits = [h for h in hits if len(h)]
        if not hits:
            return None
        s = hits[0]["DATAVALUE"].dropna()
        return s.iloc[0] if len(s) else None

    lat = get_var(["LOCATION_LAT", "LATITUDE"], contains="LAT")
    lon = get_var(["LOCATION_LONG", "LOCATION_LON", "LONGITUDE"], contains="LON")
    igbp = get_var(["IGBP", "LOCATION_IGBP"], contains="IGBP")

    if lat is not None:
        meta["lat"] = pd.to_numeric(lat, errors="coerce")
    if lon is not None:
        meta["lon"] = pd.to_numeric(lon, errors="coerce")
    if igbp is not None:
        meta["igbp"] = str(igbp).strip().upper()

    return meta

def choose_members(z):
    names = z.namelist()

    data_candidates = [
        n for n in names
        if n.endswith(".csv")
        and "_FLUXMET_DD_" in n
        and "BIF" not in n
        and "ERA5" not in n
    ]

    bif_candidates = [
        n for n in names
        if n.endswith(".csv")
        and "_BIF_" in n
        and "BIFVARINFO" not in n
    ]

    return (
        data_candidates[0] if data_candidates else None,
        bif_candidates[0] if bif_candidates else None,
    )

def parse_timestamp(df):
    col = find_col(df.columns, exact=["TIMESTAMP", "TIMESTAMP_START", "date", "Date"])
    if col is None:
        return None

    s = df[col].astype(str)
    if s.str.match(r"^\d{8}$").mean() > 0.5:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    if s.str.match(r"^\d{12}$").mean() > 0.5:
        return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def harmonize_zip(zip_path):
    site_id, network = site_id_from_zip(zip_path.name)

    try:
        z = zipfile.ZipFile(zip_path)
    except Exception as e:
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": f"bad_zip:{e}"}

    data_member, bif_member = choose_members(z)
    if data_member is None:
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": "no_fluxmet_dd"}

    meta = parse_bif_metadata(z, bif_member)

    try:
        with z.open(data_member) as f:
            df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": f"csv_read_error:{e}"}

    dt = parse_timestamp(df)
    if dt is None:
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": "no_timestamp"}

    gpp_col = find_col(
        df.columns,
        exact=["GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP_NT_CUT_REF", "GPP_DT_CUT_REF", "GPP"],
        contains_any=["GPP_NT", "GPP_DT", "GPP"]
    )
    le_col = find_col(
        df.columns,
        exact=["LE_F_MDS", "LE_CORR", "LE"],
        contains_any=["LE_F_MDS", "LE_CORR"]
    )
    vpd_col = find_col(
        df.columns,
        exact=["VPD_F", "VPD_F_MDS", "VPD"],
        contains_any=["VPD"]
    )
    swc_col = find_col(
        df.columns,
        exact=["SWC_F_MDS", "SWC_F_MDS_1", "SWC_1", "SWC"],
        contains_any=["SWC"]
    )
    qc_col = find_col(
        df.columns,
        exact=["GPP_NT_VUT_REF_QC", "NEE_VUT_REF_QC", "LE_F_MDS_QC"]
    )

    # IMPORTANT: make dataframe with index/date first, then assign scalar metadata.
    out = pd.DataFrame({"date": pd.to_datetime(dt).dt.floor("D")})
    out["tower_id"] = site_id
    out["source_network"] = network
    out["lat"] = meta["lat"]
    out["lon"] = meta["lon"]
    out["igbp"] = meta["igbp"]

    out["GPP_NT_VUT_REF"] = numeric(df[gpp_col]) if gpp_col else np.nan
    out["LE_F_MDS"] = numeric(df[le_col]) if le_col else np.nan
    out["ET"] = out["LE_F_MDS"] * 86400.0 / 2.45e6
    out["VPD"] = numeric(df[vpd_col]) if vpd_col else np.nan
    out["soil_moisture"] = numeric(df[swc_col]) if swc_col else np.nan
    out["energy_balance_closure"] = np.nan
    out["gapfill_fraction"] = numeric(df[qc_col]) if qc_col else np.nan

    out = out.dropna(subset=["date"])
    out = out[out["date"].dt.year.between(1990, 2026)]

    out = out.groupby(
        ["tower_id", "source_network", "date", "lat", "lon", "igbp"],
        dropna=False
    ).agg({
        "GPP_NT_VUT_REF": "mean",
        "LE_F_MDS": "mean",
        "ET": "mean",
        "VPD": "mean",
        "soil_moisture": "mean",
        "energy_balance_closure": "mean",
        "gapfill_fraction": "mean",
    }).reset_index()

    stat = {
        "zip": zip_path.name,
        "site": site_id,
        "network": network,
        "status": "ok",
        "data_member": data_member,
        "bif_member": bif_member,
        "igbp": meta["igbp"],
        "lat": meta["lat"],
        "lon": meta["lon"],
        "rows": len(out),
        "start": out["date"].min() if len(out) else None,
        "end": out["date"].max() if len(out) else None,
        "gpp_col": gpp_col,
        "le_col": le_col,
        "vpd_col": vpd_col,
        "swc_col": swc_col,
    }

    return out, stat

zips = sorted(ZIP_DIR.glob("*.zip"))
print("Found zip files:", len(zips))

all_rows = []
summary = []

for zp in zips:
    df, stat = harmonize_zip(zp)
    summary.append(stat)
    if df is not None and len(df):
        all_rows.append(df)
        print(f"OK {stat['site']} {stat['network']}: rows={len(df)} IGBP={stat['igbp']} years={pd.to_datetime(stat['start']).year}-{pd.to_datetime(stat['end']).year}")
    else:
        print(f"SKIP {stat.get('site')} {stat.get('network')}: {stat.get('status')}")

summary_df = pd.DataFrame(summary)
summary_path = OUT_DIR / "fluxnet_v13_harmonization_summary_FIXED.csv"
summary_df.to_csv(summary_path, index=False)

if not all_rows:
    raise SystemExit("No usable tower rows found.")

all_df = pd.concat(all_rows, ignore_index=True)
all_df = all_df.drop_duplicates(subset=["tower_id", "date"])
all_df = all_df.sort_values(["tower_id", "date"])

all_unfiltered_path = OUT_DIR / "fluxnet_v13_all_sites_unfiltered.csv"
all_df.reindex(columns=SCHEMA).to_csv(all_unfiltered_path, index=False)

grass_df = all_df[all_df["igbp"].astype(str).str.upper().isin(GRASSLIKE)].copy()

if len(grass_df) < 100:
    raise SystemExit(
        f"Grass-like subset too small: {len(grass_df)} rows. "
        "Need to inspect IGBP metadata/download more grassland sites."
    )

outputs = {
    "fluxnet2015_grassland_sites.csv": grass_df,
    "ameriflux_grassland_sites.csv": grass_df[grass_df["source_network"] == "AMF"].copy(),
    "icos_grassland_sites.csv": grass_df[grass_df["source_network"] == "ICOS"].copy(),
    "ozflux_grassland_sites.csv": grass_df.iloc[0:0].copy(),
}

for name, df in outputs.items():
    path = OUT_DIR / name
    df = df.reindex(columns=SCHEMA)
    df.to_csv(path, index=False)
    print(f"Wrote {name}: rows={len(df)} towers={df['tower_id'].nunique() if len(df) else 0}")

print("\nDONE")
print("All unfiltered rows:", len(all_df))
print("All unfiltered towers:", all_df["tower_id"].nunique())
print("Grass-like rows:", len(grass_df))
print("Grass-like towers:", grass_df["tower_id"].nunique())
print("Grass-like years:", grass_df["date"].dt.year.min(), "-", grass_df["date"].dt.year.max())

print("\nGrass-like IGBP counts:")
print(grass_df["igbp"].value_counts(dropna=False))

print("\nGrass-like network counts:")
print(grass_df["source_network"].value_counts(dropna=False))

print("\nGrass-like missingness:")
print(grass_df[["GPP_NT_VUT_REF", "LE_F_MDS", "ET", "VPD", "soil_moisture"]].isna().mean())

print("\nWrote summary:", summary_path)
print("Wrote unfiltered:", all_unfiltered_path)
