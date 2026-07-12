from pathlib import Path
from datetime import datetime
import json
import pandas as pd

OUT = Path("results/stage1b6o_product_matrix_scope_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

paths = {
    "MODIS_GPP": Path("data/raw_local/no_gee_direct_point_extract_full/MODIS_GPP_MOD17_FULL_direct_earthdata_point_samples.csv"),
    "MODIS_ET": Path("data/raw_local/no_gee_direct_point_extract_full/MODIS_ET_MOD16_FULL_direct_earthdata_point_samples.csv"),
    "MODIS_LAI": Path("data/raw_local/no_gee_direct_point_extract_full/MODIS_LAI_MCD15_FULL_direct_earthdata_point_samples.csv"),
    "MCD64A1_BURNED_AREA": Path("data/raw_local/no_gee_direct_point_extract_full/MCD64A1_BURNED_AREA_FULL_direct_earthdata_point_samples.csv"),
    "GLEAM_ET_POINT": Path("data/raw/agents/gleam_point_timeseries.csv"),
    "PML_RELOCK": Path("results/stage1b6n2_pml_spatial_strict_relock/tables/Table_PRODUCT02bz_pml_spatial_strict_relock_decision.csv"),
    "PML_GPP": Path("data/raw_local/pml_final13_point_extract/PML_GPP_FINAL13_point_samples.csv"),
    "PML_ET": Path("data/raw_local/pml_final13_point_extract/PML_ET_FINAL13_point_samples.csv"),
}

# GOSIF may exist in several forms, so search likely local point/sample outputs plus raw files.
gosif_candidates = sorted(
    list(Path("data").glob("**/*GOSIF*point*.csv")) +
    list(Path("data").glob("**/*gosif*point*.csv")) +
    list(Path("data").glob("**/*GOSIF*.csv")) +
    list(Path("data").glob("**/*gosif*.csv")) +
    list(Path("data/raw/gosif").glob("GOSIF_GPP_*_Mean.tif.gz"))
)

rows = []
for key, p in paths.items():
    rows.append({
        "source": key,
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
    })

for p in gosif_candidates[:30]:
    rows.append({
        "source": "GOSIF_CANDIDATE",
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
    })

source_inventory = pd.DataFrame(rows)
source_inventory.to_csv(TAB / "Table_PRODUCT02ca_product_matrix_source_inventory.csv", index=False)

def csv_profile(path, source):
    if not path.exists():
        return {
            "source": source,
            "path": str(path),
            "exists": False,
            "read_ok": False,
        }
    try:
        df = pd.read_csv(path, nrows=200000)
        cols = list(df.columns)
        out = {
            "source": source,
            "path": str(path),
            "exists": True,
            "read_ok": True,
            "columns": ";".join(cols),
            "sampled_rows": len(df),
        }
        for c in ["id", "point_id", "site_id"]:
            if c in cols:
                out["n_sites_sample"] = df[c].nunique()
                out["site_col"] = c
                break
        for c in ["date", "time"]:
            if c in cols:
                dates = pd.to_datetime(df[c], errors="coerce")
                out["date_col"] = c
                out["date_min_sample"] = str(dates.min().date()) if dates.notna().any() else ""
                out["date_max_sample"] = str(dates.max().date()) if dates.notna().any() else ""
                out["n_dates_sample"] = dates.nunique()
                break
        return out
    except Exception as e:
        return {
            "source": source,
            "path": str(path),
            "exists": True,
            "read_ok": False,
            "error": repr(e),
        }

profiles = []
for key, p in paths.items():
    if p.suffix.lower() == ".csv":
        profiles.append(csv_profile(p, key))

# Profile strongest GOSIF CSV candidates, if any.
for p in [x for x in gosif_candidates if x.suffix.lower() == ".csv"][:20]:
    profiles.append(csv_profile(p, "GOSIF_CANDIDATE"))

profile_df = pd.DataFrame(profiles)
profile_df.to_csv(TAB / "Table_PRODUCT02cb_product_matrix_source_profiles.csv", index=False)

# Read PML relock.
if paths["PML_RELOCK"].exists():
    pml_decision = pd.read_csv(paths["PML_RELOCK"])
    pml_role = str(pml_decision["pml_role"].iloc[0])
    pml_verdict = str(pml_decision["verdict"].iloc[0])
    pml_strict_ok = not bool(pml_decision["blocking_for_strict_3x3"].iloc[0])
else:
    pml_role = " notUNKNOWN"
    pml_verdict = "MISSING_PML_RELOCK"
    pml_strict_ok = False

has_modis_gpp = paths["MODIS_GPP"].exists()
has_modis_et = paths["MODIS_ET"].exists()
has_gleam_et = paths["GLEAM_ET_POINT"].exists()
has_lai = paths["MODIS_LAI"].exists()
has_burn = paths["MCD64A1_BURNED_AREA"].exists()
has_pml_gpp_et = paths["PML_GPP"].exists() and paths["PML_ET"].exists()

# GOSIF is allowed if either raw GOSIF rasters exist or a point CSV exists.
has_gosif = len(gosif_candidates) > 0

matrix_rows = [
    {
        "matrix_role": "STRICT_PRIMARY",
        "gpp_product": "MODIS_GPP_MOD17",
        "et_product": "MODIS_ET_MOD16",
        "allowed": has_modis_gpp and has_modis_et,
        "reason": "Both direct Earthdata final-13 point extractions complete.",
    },
    {
        "matrix_role": "STRICT_PRIMARY",
        "gpp_product": "MODIS_GPP_MOD17",
        "et_product": "GLEAM_ET",
        "allowed": has_modis_gpp and has_gleam_et,
        "reason": "MODIS GPP direct final-13 available; GLEAM ET local point timeseries available.",
    },
    {
        "matrix_role": "STRICT_PRIMARY",
        "gpp_product": "GOSIF_GPP",
        "et_product": "MODIS_ET_MOD16",
        "allowed": has_gosif and has_modis_et,
        "reason": "GOSIF local files/candidates available; MODIS ET direct final-13 available.",
    },
    {
        "matrix_role": "STRICT_PRIMARY",
        "gpp_product": "GOSIF_GPP",
        "et_product": "GLEAM_ET",
        "allowed": has_gosif and has_gleam_et,
        "reason": "GOSIF local files/candidates available; GLEAM ET local point timeseries available.",
    },
    {
        "matrix_role": "COARSE_SENSITIVITY_ONLY",
        "gpp_product": "PML_GPP",
        "et_product": "PML_ET",
        "allowed": has_pml_gpp_et and pml_role == "COARSE_PML_SENSITIVITY_ONLY",
        "reason": f"PML relock verdict={pml_verdict}; role={pml_role}. Not strict tower-centered because spatial mismatch is several degrees.",
    },
]

matrix_df = pd.DataFrame(matrix_rows)
matrix_df.to_csv(TAB / "Table_PRODUCT02cc_allowed_product_matrix_scope.csv", index=False)

all_strict_2x2 = bool(matrix_df[matrix_df["matrix_role"].eq("STRICT_PRIMARY")]["allowed"].all())
qa_ready = has_lai and has_burn

if all_strict_2x2 and qa_ready:
    verdict = "STRICT_2X2_PRODUCT_MATRIX_READY_WITH_PML_COARSE_SENSITIVITY"
    blocking_for_next_analysis = False
else:
    verdict = "PRODUCT_MATRIX_NOT_READY"
    blocking_for_next_analysis = True

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "strict_3x3_with_pml_allowed": bool(pml_strict_ok),
    "strict_2x2_allowed": all_strict_2x2,
    "pml_role": pml_role,
    "qa_lai_available": has_lai,
    "qa_burned_area_available": has_burn,
    "verdict": verdict,
    "blocking_for_next_analysis": blocking_for_next_analysis,
    "next_stage": "BUILD_STRICT_2X2_RESPONSE_TABLE_AND_PML_COARSE_SENSITIVITY" if not blocking_for_next_analysis else "RESOLVE_SOURCE_GAPS",
}])
decision.to_csv(TAB / "Table_PRODUCT02cd_product_matrix_scope_decision.csv", index=False)

report = []
report.append("# Stage 1B.6O product-matrix scope lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Allowed product matrix scope")
report.append("")
report.append("```text")
report.append(matrix_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Source inventory")
report.append("")
report.append("```text")
report.append(source_inventory.to_string(index=False))
report.append("```")
report.append("")
report.append("## Source profiles")
report.append("")
report.append("```text")
report.append(profile_df.to_string(index=False) if len(profile_df) else "No profiles.")
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Use a strict 2x2 matrix for tower-centered primary inference: MODIS/GOSIF GPP by MODIS/GLEAM ET. PML is allowed only as coarse-grid sensitivity unless a better spatially matched PML extraction is found.")
report.append("")

(TXT / "STAGE1B6O_PRODUCT_MATRIX_SCOPE_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6O_product_matrix_scope_lock",
    "status": verdict,
    "blocking_for_next_analysis": bool(blocking_for_next_analysis),
    "outputs": {
        "source_inventory": str(TAB / "Table_PRODUCT02ca_product_matrix_source_inventory.csv"),
        "source_profiles": str(TAB / "Table_PRODUCT02cb_product_matrix_source_profiles.csv"),
        "matrix_scope": str(TAB / "Table_PRODUCT02cc_allowed_product_matrix_scope.csv"),
        "decision": str(TAB / "Table_PRODUCT02cd_product_matrix_scope_decision.csv"),
        "report": str(TXT / "STAGE1B6O_PRODUCT_MATRIX_SCOPE_LOCK_REPORT.md"),
    }
}
(TAB / "STAGE1B6O_PRODUCT_MATRIX_SCOPE_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02ca_product_matrix_source_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT02cb_product_matrix_source_profiles.csv")
print("WROTE", TAB / "Table_PRODUCT02cc_allowed_product_matrix_scope.csv")
print("WROTE", TAB / "Table_PRODUCT02cd_product_matrix_scope_decision.csv")
print("WROTE", TXT / "STAGE1B6O_PRODUCT_MATRIX_SCOPE_LOCK_REPORT.md")
