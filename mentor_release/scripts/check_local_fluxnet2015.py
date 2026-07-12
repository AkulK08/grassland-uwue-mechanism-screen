#!/usr/bin/env python3
import os
import re
import zipfile
import pandas as pd
from pathlib import Path

ROOTS = [
    Path("/Users/me/Downloads/grassland_wue_nature_repo"),
    Path("/Users/me/Downloads"),
    Path("/Users/me/Desktop"),
    Path("/Users/me/Documents"),
]

OUTDIR = Path("results/tower_file_check")
OUTDIR.mkdir(parents=True, exist_ok=True)

NAME_PATTERNS = [
    "FLUXNET2015",
    "FLX_",
    "FULLSET",
    "SUBSET",
    "AUXMETEO",
]

KEY_COLS = [
    "TIMESTAMP_START",
    "TIMESTAMP_END",
    "TIMESTAMP",
    "GPP_NT_VUT_REF",
    "GPP_DT_VUT_REF",
    "LE_F_MDS",
    "LE_CORR",
    "H_F_MDS",
    "NETRAD",
    "VPD_F",
    "VPD",
    "TA_F",
    "P_F",
    "SW_IN_F",
    "SWC_F_MDS_1",
    "SWC_F_MDS_2",
    "SWC_1",
    "SWC_2",
    "NEE_VUT_REF",
    "RECO_NT_VUT_REF",
]

def looks_relevant(path: Path) -> bool:
    s = str(path).upper()
    return any(p.upper() in s for p in NAME_PATTERNS)

def site_from_name(name: str):
    m = re.search(r"FLX_([A-Z]{2}-[A-Za-z0-9]+)", name)
    if m:
        return m.group(1)
    m = re.search(r"([A-Z]{2}-[A-Za-z0-9]+)", name)
    if m:
        return m.group(1)
    return ""

def read_csv_header(path: Path):
    try:
        df = pd.read_csv(path, nrows=5)
        return list(df.columns), len(df.columns), None
    except Exception as e:
        return [], 0, str(e)

def inspect_zip(path: Path):
    rows = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            members = [m for m in z.namelist() if m.lower().endswith(".csv")]
            relevant_members = [m for m in members if looks_relevant(Path(m))]
            for m in relevant_members[:20]:
                try:
                    with z.open(m) as f:
                        df = pd.read_csv(f, nrows=5)
                    cols = list(df.columns)
                    rows.append({
                        "container": str(path),
                        "file": m,
                        "site": site_from_name(m),
                        "kind": "zip_member_csv",
                        "n_columns": len(cols),
                        "has_GPP_NT_VUT_REF": "GPP_NT_VUT_REF" in cols,
                        "has_LE_F_MDS": "LE_F_MDS" in cols,
                        "has_VPD": ("VPD_F" in cols) or ("VPD" in cols),
                        "has_soil_moisture": any(c.startswith("SWC") for c in cols),
                        "key_columns_found": ",".join([c for c in KEY_COLS if c in cols]),
                        "read_error": "",
                    })
                except Exception as e:
                    rows.append({
                        "container": str(path),
                        "file": m,
                        "site": site_from_name(m),
                        "kind": "zip_member_csv",
                        "n_columns": 0,
                        "has_GPP_NT_VUT_REF": False,
                        "has_LE_F_MDS": False,
                        "has_VPD": False,
                        "has_soil_moisture": False,
                        "key_columns_found": "",
                        "read_error": str(e),
                    })
    except Exception as e:
        rows.append({
            "container": str(path),
            "file": "",
            "site": "",
            "kind": "zip_error",
            "n_columns": 0,
            "has_GPP_NT_VUT_REF": False,
            "has_LE_F_MDS": False,
            "has_VPD": False,
            "has_soil_moisture": False,
            "key_columns_found": "",
            "read_error": str(e),
        })
    return rows

rows = []
candidate_paths = []

for root in ROOTS:
    if not root.exists():
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip huge/noisy folders
        parts = set(Path(dirpath).parts)
        if "Library" in parts or ".git" in parts or "__pycache__" in parts:
            continue

        for fn in filenames:
            p = Path(dirpath) / fn
            if looks_relevant(p) and p.suffix.lower() in [".csv", ".zip", ".gz"]:
                candidate_paths.append(p)

print(f"Candidate FLUXNET-like files found: {len(candidate_paths)}")

for p in candidate_paths:
    suffixes = "".join(p.suffixes).lower()

    if p.suffix.lower() == ".zip":
        rows.extend(inspect_zip(p))
        continue

    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz") or p.suffix.lower() == ".gz":
        cols, ncols, err = read_csv_header(p)
        rows.append({
            "container": "",
            "file": str(p),
            "site": site_from_name(p.name),
            "kind": "csv_or_gz",
            "n_columns": ncols,
            "has_GPP_NT_VUT_REF": "GPP_NT_VUT_REF" in cols,
            "has_LE_F_MDS": "LE_F_MDS" in cols,
            "has_VPD": ("VPD_F" in cols) or ("VPD" in cols),
            "has_soil_moisture": any(c.startswith("SWC") for c in cols),
            "key_columns_found": ",".join([c for c in KEY_COLS if c in cols]),
            "read_error": err or "",
        })

df = pd.DataFrame(rows)

if df.empty:
    print("\nNO FLUXNET2015-like readable CSV files found in repo/Downloads/Desktop/Documents.")
    print("Try searching external drives or tell me where you downloaded FLUXNET.")
else:
    df.to_csv(OUTDIR / "local_fluxnet2015_file_inventory.csv", index=False)

    usable = df[
        (df["has_GPP_NT_VUT_REF"] == True)
        & (df["has_LE_F_MDS"] == True)
    ].copy()

    usable.to_csv(OUTDIR / "usable_fluxnet2015_tower_files.csv", index=False)

    print("\n===== SUMMARY =====")
    print("Readable candidate files/members:", len(df))
    print("Usable WUE files with GPP_NT_VUT_REF + LE_F_MDS:", len(usable))
    print("Unique usable sites:", usable["site"].replace("", pd.NA).dropna().nunique())

    print("\n===== USABLE FILES PREVIEW =====")
    if usable.empty:
        print("None found with both GPP_NT_VUT_REF and LE_F_MDS.")
    else:
        cols = ["site", "kind", "file", "container", "has_VPD", "has_soil_moisture", "key_columns_found"]
        print(usable[cols].head(50).to_string(index=False))

    print("\nWrote:")
    print(OUTDIR / "local_fluxnet2015_file_inventory.csv")
    print(OUTDIR / "usable_fluxnet2015_tower_files.csv")
