from pathlib import Path
import zipfile
import pandas as pd
import re

ROOT = Path(".")
SEARCH_DIRS = [
    ROOT / "data/raw/towers/_downloads/ameriflux_base",
    ROOT / "data/raw/towers/_downloads/fluxnet2015",
    Path.home() / "Downloads",
]

OUT_DIR = ROOT / "results/tower_file_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

zip_paths = []
for d in SEARCH_DIRS:
    if d.exists():
        zip_paths.extend(sorted(d.rglob("*.zip")))

rows = []

def infer_site(path_name, member_name=""):
    text = path_name + " " + member_name
    m = re.search(r"(AMF|FLX)_([A-Z]{2,3}-[A-Za-z0-9]+)", text)
    if m:
        return m.group(2)
    m = re.search(r"([A-Z]{2,3}-[A-Za-z0-9]+)", text)
    if m:
        return m.group(1)
    return ""

def classify_member(name):
    upper = name.upper()
    if "FLUXNET" in upper and ("_DD_" in upper or upper.endswith("_DD.CSV")):
        return "fluxnet_daily"
    if "FLUXNET" in upper and ("_HH_" in upper or "_HR_" in upper or "_FULLSET" in upper):
        return "fluxnet_subdaily_or_fullset"
    if "BASE" in upper or "BADM" in upper or "AMF_" in upper:
        return "ameriflux_base_or_badm"
    return "other_csv"

def read_header_from_zip(zp, member):
    try:
        with zipfile.ZipFile(zp) as z:
            with z.open(member) as f:
                df = pd.read_csv(f, nrows=5, low_memory=False)
        return list(df.columns), ""
    except Exception as e:
        return [], str(e)

for zp in zip_paths:
    try:
        with zipfile.ZipFile(zp) as z:
            members = [m for m in z.namelist() if m.lower().endswith((".csv", ".csv.gz"))]
    except Exception as e:
        rows.append({
            "zip_path": str(zp),
            "zip_name": zp.name,
            "site": infer_site(zp.name),
            "member": "",
            "member_type": "ZIP_READ_ERROR",
            "n_columns": 0,
            "has_gpp_nt": False,
            "has_gpp_any": False,
            "has_le": False,
            "has_vpd": False,
            "has_soil_moisture": False,
            "candidate_wue_file": False,
            "columns_found": "",
            "read_error": str(e),
        })
        continue

    for m in members:
        cols, err = read_header_from_zip(zp, m)
        colset = set(cols)

        gpp_cols = [c for c in cols if c.upper().startswith("GPP") or "GPP" in c.upper()]
        le_cols = [c for c in cols if c.upper() in {"LE_F_MDS", "LE_CORR", "LE", "LE_PI", "LE_F"} or c.upper().startswith("LE_")]
        vpd_cols = [c for c in cols if "VPD" in c.upper()]
        sm_cols = [c for c in cols if c.upper().startswith(("SWC", "SW_IN", "SW_OUT", "TS_", "PA_", "P_F")) or "SOIL" in c.upper() or "SWC" in c.upper()]

        has_gpp_nt = "GPP_NT_VUT_REF" in colset
        has_gpp_any = len(gpp_cols) > 0
        has_le = len(le_cols) > 0
        has_vpd = len(vpd_cols) > 0
        has_sm = len([c for c in cols if "SWC" in c.upper() or "SOIL" in c.upper()]) > 0

        candidate = has_gpp_any and has_le and has_vpd

        rows.append({
            "zip_path": str(zp),
            "zip_name": zp.name,
            "site": infer_site(zp.name, m),
            "member": m,
            "member_type": classify_member(m),
            "n_columns": len(cols),
            "has_gpp_nt": has_gpp_nt,
            "has_gpp_any": has_gpp_any,
            "has_le": has_le,
            "has_vpd": has_vpd,
            "has_soil_moisture": has_sm,
            "candidate_wue_file": candidate,
            "columns_found": ",".join(gpp_cols[:5] + le_cols[:5] + vpd_cols[:5] + sm_cols[:5]),
            "read_error": err,
        })

df = pd.DataFrame(rows)
out_all = OUT_DIR / "ameriflux_zip_member_inventory.csv"
df.to_csv(out_all, index=False)

if len(df):
    candidates = df[df["candidate_wue_file"]].copy()
else:
    candidates = pd.DataFrame()

out_candidates = OUT_DIR / "ameriflux_candidate_wue_files.csv"
candidates.to_csv(out_candidates, index=False)

print("===== AMERIFLUX ZIP SCAN SUMMARY =====")
print("zip files scanned:", len(zip_paths))
print("csv members scanned:", len(df))
print("candidate WUE members:", len(candidates))

if len(candidates):
    print("unique candidate sites:", candidates["site"].nunique())
    print("")
    print("===== candidate sites =====")
    print(sorted(candidates["site"].dropna().unique().tolist()))
    print("")
    print("===== candidate preview =====")
    preview_cols = [
        "site", "zip_name", "member", "member_type",
        "has_gpp_nt", "has_gpp_any", "has_le", "has_vpd", "has_soil_moisture",
        "columns_found"
    ]
    print(candidates[preview_cols].head(80).to_string(index=False))
else:
    print("No candidate WUE files found by broad scan.")

print("")
print("WROTE:")
print(out_all)
print(out_candidates)
