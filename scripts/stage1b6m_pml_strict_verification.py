from pathlib import Path
from datetime import datetime
import json
import glob
import pandas as pd

OUT = Path("results/stage1b6m_pml_strict_verification")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

patterns = [
    "data/**/*PML*.nc",
    "data/**/*pml*.nc",
    "data/**/*PML*.csv",
    "data/**/*pml*.csv",
    "data/processed/components_PML_*.nc",
    "data/processed/*PML*.nc",
    "data/demo/*PML*.nc",
]

files = sorted(set(sum([glob.glob(p, recursive=True) for p in patterns], [])))

rows = []
for f in files:
    p = Path(f)
    rows.append({
        "path": str(p),
        "name": p.name,
        "suffix": p.suffix,
        "size_bytes": p.stat().st_size if p.exists() else None,
        "is_demo": "demo" in str(p).lower(),
        "is_processed": "processed" in str(p).lower(),
        "looks_component": "components_" in p.name.lower(),
    })

candidates = pd.DataFrame(rows)
candidates.to_csv(TAB / "Table_PRODUCT02bp_pml_candidate_files.csv", index=False)

inspect_rows = []
for f in files:
    p = Path(f)
    if p.suffix.lower() == ".nc":
        try:
            import xarray as xr
            ds = xr.open_dataset(p)
            inspect_rows.append({
                "path": str(p),
                "open_status": "OPEN_OK",
                "dims": json.dumps({k:int(v) for k,v in ds.sizes.items()}),
                "variables": ";".join(list(ds.data_vars)),
                "coords": ";".join(list(ds.coords)),
            })
            ds.close()
        except Exception as e:
            inspect_rows.append({
                "path": str(p),
                "open_status": "OPEN_ERROR",
                "error": str(e)[:1000],
            })
    elif p.suffix.lower() == ".csv":
        try:
            head = pd.read_csv(p, nrows=5)
            inspect_rows.append({
                "path": str(p),
                "open_status": "OPEN_OK_CSV",
                "dims": "",
                "variables": ";".join(head.columns),
                "coords": "",
            })
        except Exception as e:
            inspect_rows.append({
                "path": str(p),
                "open_status": "OPEN_ERROR_CSV",
                "error": str(e)[:1000],
            })

inspect = pd.DataFrame(inspect_rows)
inspect.to_csv(TAB / "Table_PRODUCT02bq_pml_file_inspection.csv", index=False)

usable = []
for _, r in inspect.iterrows():
    path_lower = str(r.get("path", "")).lower()
    vars_lower = str(r.get("variables", "")).lower()
    dims_lower = str(r.get("dims", "")).lower()
    coords_lower = str(r.get("coords", "")).lower()

    has_gpp = "gpp" in vars_lower
    has_et = ("et" in vars_lower) or ("transpiration" in vars_lower) or ("evap" in vars_lower)
    has_time = ("time" in dims_lower) or ("date" in dims_lower) or ("time" in coords_lower) or ("date" in coords_lower)
    not_demo = "demo" not in path_lower

    usable.append({
        "path": r["path"],
        "has_gpp": has_gpp,
        "has_et": has_et,
        "has_time_or_date": has_time,
        "not_demo_path": not_demo,
        "strict_candidate": bool((has_gpp or has_et) and has_time and not_demo),
    })

usable_df = pd.DataFrame(usable)
usable_df.to_csv(TAB / "Table_PRODUCT02br_pml_strict_candidate_decision.csv", index=False)

n_gpp_candidates = int(((usable_df["has_gpp"]) & usable_df["strict_candidate"]).sum()) if len(usable_df) else 0
n_et_candidates = int(((usable_df["has_et"]) & usable_df["strict_candidate"]).sum()) if len(usable_df) else 0

if n_gpp_candidates > 0 and n_et_candidates > 0:
    verdict = "PML_GPP_AND_ET_STRICT_CANDIDATES_EXIST"
    blocking_for_3x3 = False
elif n_gpp_candidates > 0 or n_et_candidates > 0:
    verdict = "PML_PARTIAL_CANDIDATE_ONLY"
    blocking_for_3x3 = True
else:
    verdict = "PML_NOT_STRICTLY_VERIFIED"
    blocking_for_3x3 = True

summary = pd.DataFrame([{
    "n_pml_files_found": len(files),
    "n_opened_or_checked": len(inspect),
    "n_strict_gpp_candidates": n_gpp_candidates,
    "n_strict_et_candidates": n_et_candidates,
    "verdict": verdict,
    "blocking_for_full_3x3": blocking_for_3x3,
}])
summary.to_csv(TAB / "Table_PRODUCT02bs_pml_verification_summary.csv", index=False)

report = []
report.append("# Stage 1B.6M PML strict verification")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## Strict candidate decision")
report.append("")
report.append("```text")
report.append(usable_df.to_string(index=False) if len(usable_df) else "No usable PML candidate rows.")
report.append("```")
report.append("")
report.append("## File inspection")
report.append("")
report.append("```text")
report.append(inspect.head(80).to_string(index=False) if len(inspect) else "No PML files found.")
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("PML can enter the 3x3 product matrix only if both PML GPP and PML ET have non-demo time-series candidates.")
report.append("")

(TXT / "STAGE1B6M_PML_STRICT_VERIFICATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02bp_pml_candidate_files.csv")
print("WROTE", TAB / "Table_PRODUCT02bq_pml_file_inspection.csv")
print("WROTE", TAB / "Table_PRODUCT02br_pml_strict_candidate_decision.csv")
print("WROTE", TAB / "Table_PRODUCT02bs_pml_verification_summary.csv")
print("WROTE", TXT / "STAGE1B6M_PML_STRICT_VERIFICATION_REPORT.md")
