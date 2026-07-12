from pathlib import Path
from datetime import datetime
import json
import shutil
import pandas as pd

OUT = Path("results/stage1b6e_external_storage_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

FEAS = Path("results/stage1b6c_download_feasibility/tables/Table_PRODUCT02v_disk_space_feasibility.csv")
DOWNLOAD_SUM = Path("results/stage1b6b_bbox_modis_search/tables/Table_PRODUCT02s_bbox_search_download_summary.csv")

if not FEAS.exists():
    raise FileNotFoundError("Missing Stage 1B.6C disk feasibility table.")
if not DOWNLOAD_SUM.exists():
    raise FileNotFoundError("Missing Stage 1B.6B download summary table.")

feas = pd.read_csv(FEAS)
download_sum = pd.read_csv(DOWNLOAD_SUM)

estimated_gb = float(feas["estimated_download_gb"].iloc[0])
recommended_gb = float(feas["recommended_free_gb_25pct_buffer"].iloc[0])

candidates = []

paths = [Path("."), Path("/Users/me/Downloads"), Path("/Volumes")]
for base in paths:
    if not base.exists():
        continue
    if base == Path("/Volumes"):
        for p in sorted(base.iterdir()):
            if p.exists() and p.is_dir():
                try:
                    u = shutil.disk_usage(p)
                    candidates.append({
                        "candidate_path": str(p),
                        "kind": "EXTERNAL_OR_MOUNTED_VOLUME",
                        "total_gb": u.total / 1e9,
                        "used_gb": u.used / 1e9,
                        "free_gb": u.free / 1e9,
                    })
                except Exception as e:
                    candidates.append({
                        "candidate_path": str(p),
                        "kind": "EXTERNAL_OR_MOUNTED_VOLUME",
                        "total_gb": None,
                        "used_gb": None,
                        "free_gb": None,
                        "error": str(e),
                    })
    else:
        try:
            u = shutil.disk_usage(base)
            candidates.append({
                "candidate_path": str(base.resolve()),
                "kind": "LOCAL_PATH",
                "total_gb": u.total / 1e9,
                "used_gb": u.used / 1e9,
                "free_gb": u.free / 1e9,
            })
        except Exception as e:
            candidates.append({
                "candidate_path": str(base),
                "kind": "LOCAL_PATH",
                "total_gb": None,
                "used_gb": None,
                "free_gb": None,
                "error": str(e),
            })

df = pd.DataFrame(candidates)
df["estimated_download_gb"] = estimated_gb
df["recommended_free_gb_25pct_buffer"] = recommended_gb
df["passes_strict_storage_requirement"] = pd.to_numeric(df["free_gb"], errors="coerce") >= recommended_gb
df = df.sort_values(["passes_strict_storage_requirement", "free_gb"], ascending=[False, False])
df.to_csv(TAB / "Table_PRODUCT02ad_storage_candidates.csv", index=False)

passing = df[df["passes_strict_storage_requirement"].eq(True)].copy()

if len(passing):
    selected_root = Path(passing.iloc[0]["candidate_path"]) / "grassland_wue_strict_raw_downloads"
    selected_root.mkdir(parents=True, exist_ok=True)
    status = "STRICT_STORAGE_AVAILABLE"
    next_action = "Run the full download using this selected external/local storage root."
else:
    selected_root = Path("")
    status = "STRICT_STORAGE_NOT_AVAILABLE"
    next_action = "Attach an external drive with at least 500 GB free, then rerun this stage."

products = []
for _, r in download_sum.iterrows():
    product_group = r["product_group"]
    short_name = r["short_name"]
    if status == "STRICT_STORAGE_AVAILABLE":
        if product_group == "MODIS_GPP_MOD17":
            target_dir = selected_root / "modis_mod17_gpp"
        elif product_group == "MODIS_ET_MOD16":
            target_dir = selected_root / "modis_mod16_et"
        elif product_group == "MODIS_LAI_MCD15":
            target_dir = selected_root / "modis_mcd15_lai"
        elif product_group == "MCD64A1_BURNED_AREA":
            target_dir = selected_root / "modis_mcd64_burned_area"
        else:
            target_dir = selected_root / product_group.lower()
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = ""

    products.append({
        "product_group": product_group,
        "short_name": short_name,
        "n_to_download": int(r["n_to_download"]),
        "estimated_gb": float(r["estimated_size_mb_all_found"]) / 1024.0,
        "target_dir": str(target_dir),
    })

prod_df = pd.DataFrame(products)
prod_df.to_csv(TAB / "Table_PRODUCT02ae_selected_download_targets.csv", index=False)

download_command = ""
if status == "STRICT_STORAGE_AVAILABLE":
    download_command = (
        "STRICT_DOWNLOAD_ROOT='" + str(selected_root) + "' "
        "DO_DOWNLOAD=1 TARGET_SCOPE=main13 START_YEAR=2001 END_YEAR=2024 "
        "python -u scripts/stage1b6f_full_strict_modis_download.py "
        "2>&1 | tee logs/stage1b6f_full_strict_modis_download.log"
    )

cmd_df = pd.DataFrame([{
    "status": status,
    "selected_root": str(selected_root) if status == "STRICT_STORAGE_AVAILABLE" else "",
    "download_command": download_command,
    "next_action": next_action,
}])
cmd_df.to_csv(TAB / "Table_PRODUCT02af_storage_lock_decision.csv", index=False)

report = []
report.append("# Stage 1B.6E external storage lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Storage candidates")
report.append("")
report.append("```text")
report.append(df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Selected download targets")
report.append("")
report.append("```text")
report.append(prod_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Storage lock decision")
report.append("")
report.append("```text")
report.append(cmd_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("This stage is complete only when a storage root with enough free space is selected for the full raw gridded MODIS/MCD download.")
report.append("")

(TXT / "STAGE1B6E_EXTERNAL_STORAGE_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6E_external_storage_lock",
    "status": status,
    "estimated_download_gb": estimated_gb,
    "recommended_free_gb": recommended_gb,
    "selected_root": str(selected_root) if status == "STRICT_STORAGE_AVAILABLE" else "",
    "outputs": {
        "storage_candidates": str(TAB / "Table_PRODUCT02ad_storage_candidates.csv"),
        "selected_targets": str(TAB / "Table_PRODUCT02ae_selected_download_targets.csv"),
        "decision": str(TAB / "Table_PRODUCT02af_storage_lock_decision.csv"),
        "report": str(TXT / "STAGE1B6E_EXTERNAL_STORAGE_LOCK_REPORT.md"),
    }
}
(TAB / "STAGE1B6E_EXTERNAL_STORAGE_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02ad_storage_candidates.csv")
print("WROTE", TAB / "Table_PRODUCT02ae_selected_download_targets.csv")
print("WROTE", TAB / "Table_PRODUCT02af_storage_lock_decision.csv")
print("WROTE", TXT / "STAGE1B6E_EXTERNAL_STORAGE_LOCK_REPORT.md")
