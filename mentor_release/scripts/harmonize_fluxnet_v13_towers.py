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

def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
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

    cols = list(bif.columns)

    # Long BIF format: VARIABLE / DATAVALUE
    if "VARIABLE" in cols and "DATAVALUE" in cols:
        var = bif["VARIABLE"].astype(str).str.upper()
        val = bif["DATAVALUE"]

        def get_by_var(possible, contains=None):
            hits = []
            for p in possible:
                hits.append(bif[var == p.upper()])
            if contains:
                hits.append(bif[var.str.contains(contains.upper(), na=False)])
            hits = [h for h in hits if len(h)]
            if not hits:
                return None
            s = hits[0]["DATAVALUE"].dropna()
            if len(s) == 0:
                return None
            return s.iloc[0]

        lat = get_by_var(["LOCATION_LAT", "LATITUDE"], contains="LAT")
        lon = get_by_var(["LOCATION_LONG", "LOCATION_LON", "LONGITUDE"], contains="LON")
        igbp = get_by_var(["IGBP", "LOCATION_IGBP"], contains="IGBP")

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

    if not data_candidates:
        data_candidates = [
            n for n in names
            if n.endswith(".csv")
            and "_DD_" in n
            and "FLUXMET" in n
            and "BIF" not in n
            and "ERA5" not in n
        ]

    bif_candidates = [
        n for n in names
        if n.endswith(".csv")
        and "_BIF_" in n
        and "BIFVARINFO" not in n
    ]

    data_member = data_candidates[0] if data_candidates else None
    bif_member = bif_candidates[0] if bif_candidates else None

    return data_member, bif_member

def parse_timestamp(df):
    col = find_col(df.columns, ["TIMESTAMP", "TIMESTAMP_START", "date", "Date"])
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
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": f"bad_zip: {e}"}

    data_member, bif_member = choose_members(z)

    if data_member is None:
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": "no_fluxmet_dd"}

    meta = parse_bif_metadata(z, bif_member)

    try:
        with z.open(data_member) as f:
            df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": f"csv_read_error: {e}"}

    dt = parse_timestamp(df)
    if dt is None:
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": "no_timestamp"}

    gpp_col = find_col(df.columns, [
        "GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP_NT_CUT_REF", "GPP_DT_CUT_REF", "GPP"
    ])
    le_col = find_col(df.columns, [
        "LE_F_MDS", "LE_CORR", "LE"
    ])
    vpd_col = find_col(df.columns, [
        "VPD_F", "VPD_F_MDS", "VPD"
    ])
    swc_col = find_col(df.columns, [
        "SWC_F_MDS", "SWC_F_MDS_1", "SWC_1", "SWC"
    ])
    nee_qc_col = find_col(df.columns, [
        "NEE_VUT_REF_QC", "GPP_NT_VUT_REF_QC", "LE_F_MDS_QC"
    ])

    if gpp_col is None and le_col is None:
        return None, {"zip": zip_path.name, "site": site_id, "network": network, "status": "no_gpp_or_le"}

    out = pd.DataFrame()
    out["tower_id"] = site_id
    out["source_network"] = network
    out["date"] = pd.to_datetime(dt).dt.floor("D")
    out["lat"] = meta["lat"]
    out["lon"] = meta["lon"]
    out["igbp"] = meta["igbp"]

    out["GPP_NT_VUT_REF"] = numeric(df[gpp_col]) if gpp_col else np.nan
    out["LE_F_MDS"] = numeric(df[le_col]) if le_col else np.nan

    # Convert daily mean latent heat flux W/m2 to ET mm/day.
    out["ET"] = out["LE_F_MDS"] * 86400.0 / 2.45e6

    out["VPD"] = numeric(df[vpd_col]) if vpd_col else np.nan
    out["soil_moisture"] = numeric(df[swc_col]) if swc_col else np.nan

    out["energy_balance_closure"] = np.nan
    out["gapfill_fraction"] = numeric(df[nee_qc_col]) if nee_qc_col else np.nan

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

    status = {
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

    return out, status

zips = sorted(ZIP_DIR.glob("*.zip"))
print("Found zip files:", len(zips))

all_rows = []
summary = []

for zp in zips:
    df, stat = harmonize_zip(zp)
    summary.append(stat)
    if df is not None and len(df):
        all_rows.append(df)
        print(f"OK {stat['site']}: {len(df)} rows, {stat.get('start')} to {stat.get('end')}, IGBP={stat.get('igbp')}")
    else:
        print(f"SKIP {stat['site']}: {stat['status']}")

summary_df = pd.DataFrame(summary)
summary_path = OUT_DIR / "fluxnet_v13_harmonization_summary.csv"
summary_df.to_csv(summary_path, index=False)

if not all_rows:
    print("No usable rows found.")
    raise SystemExit(1)

all_df = pd.concat(all_rows, ignore_index=True)
all_df = all_df.drop_duplicates(subset=["tower_id", "date"])
all_df = all_df.sort_values(["tower_id", "date"])

# Make grass-like subset where IGBP metadata exists.
grass_df = all_df[all_df["igbp"].astype(str).str.upper().isin(GRASSLIKE)].copy()

if len(grass_df) >= 100:
    final_df = grass_df
    print("\nUsing grass-like IGBP subset.")
else:
    final_df = all_df
    print("\nWARNING: Grass-like IGBP subset too small or metadata missing. Keeping all rows for now.")

outputs = {
    "fluxnet2015_grassland_sites.csv": final_df,
    "ameriflux_grassland_sites.csv": final_df[final_df["source_network"] == "AMF"].copy(),
    "icos_grassland_sites.csv": final_df[final_df["source_network"] == "ICOS"].copy(),
    "ozflux_grassland_sites.csv": final_df.iloc[0:0].copy(),
}

for name, df in outputs.items():
    path = OUT_DIR / name
    df = df.reindex(columns=SCHEMA)
    df.to_csv(path, index=False)
    print(f"Wrote {path} rows={len(df)} towers={df['tower_id'].nunique() if len(df) else 0}")

print("\nFinal full dataset:")
print("rows:", len(final_df))
print("towers:", final_df["tower_id"].nunique())
print("years:", final_df["date"].dt.year.min(), "-", final_df["date"].dt.year.max())
print("\nIGBP counts:")
print(final_df["igbp"].value_counts(dropna=False).head(30))
print("\nNetwork counts:")
print(final_df["source_network"].value_counts(dropna=False))
print("\nMissingness:")
print(final_df[["GPP_NT_VUT_REF", "LE_F_MDS", "ET", "VPD", "soil_moisture"]].isna().mean())
print("\nSummary file:", summary_path)
