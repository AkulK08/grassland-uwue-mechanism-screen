from pathlib import Path
from datetime import datetime
import json
import pandas as pd

IN_SUMMARY = Path("results/stage1b6n_pml_final13_coverage_lock/tables/Table_PRODUCT02bx_pml_final13_coverage_summary.csv")
IN_SITE = Path("results/stage1b6n_pml_final13_coverage_lock/tables/Table_PRODUCT02bu_pml_final13_site_coverage.csv")

OUT = Path("results/stage1b6n2_pml_spatial_strict_relock")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(IN_SUMMARY)
site = pd.read_csv(IN_SITE)

# Strict threshold for tower-centered product matrix:
# nearest grid point should be close enough to represent the tower/product point.
# 0.25 degrees is already generous for a point-level tower analysis.
STRICT_MAX_DEG = 0.25
COARSE_SENSITIVITY_MAX_DEG = 8.0

max_lat = float(summary["max_abs_lat_diff"].iloc[0])
max_lon = float(summary["max_abs_lon_diff"].iloc[0])
n_targets = int(summary["n_targets"].iloc[0])
min_times = int(summary["min_times_across_products"].iloc[0])
min_nonmissing = int(summary["min_nonmissing_site_product_rows"].iloc[0])

site["max_abs_coord_diff"] = site[["max_abs_lat_diff", "max_abs_lon_diff"]].max(axis=1)
site["strict_spatial_pass"] = site["max_abs_coord_diff"] <= STRICT_MAX_DEG
site["coarse_sensitivity_pass"] = site["max_abs_coord_diff"] <= COARSE_SENSITIVITY_MAX_DEG

n_strict_site_product_pass = int(site["strict_spatial_pass"].sum())
n_site_product_rows = int(len(site))
n_coarse_site_product_pass = int(site["coarse_sensitivity_pass"].sum())

all_strict_spatial = n_strict_site_product_pass == n_site_product_rows
all_coarse_spatial = n_coarse_site_product_pass == n_site_product_rows
has_coverage = (
    bool(summary["has_both_pml_gpp_and_et"].iloc[0])
    and int(summary["min_sites_across_products"].iloc[0]) == 13
    and min_times >= 300
    and min_nonmissing >= 300
    and int(summary["n_errors"].iloc[0]) == 0
)

if has_coverage and all_strict_spatial:
    verdict = "PML_STRICT_TOWER_CENTERED_MATRIX_PASS"
    role = "STRICT_3X3_PRODUCT_MATRIX"
    blocking_for_strict_3x3 = False
elif has_coverage and all_coarse_spatial:
    verdict = "PML_COVERAGE_PASS_BUT_SPATIALLY_COARSE"
    role = "COARSE_PML_SENSITIVITY_ONLY"
    blocking_for_strict_3x3 = True
else:
    verdict = "PML_FAIL"
    role = "DO_NOT_USE_FOR_PRODUCT_MATRIX"
    blocking_for_strict_3x3 = True

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_targets": n_targets,
    "min_times_across_products": min_times,
    "min_nonmissing_site_product_rows": min_nonmissing,
    "max_abs_lat_diff": max_lat,
    "max_abs_lon_diff": max_lon,
    "strict_max_degree_threshold": STRICT_MAX_DEG,
    "coarse_sensitivity_max_degree_threshold": COARSE_SENSITIVITY_MAX_DEG,
    "site_product_rows_strict_pass": n_strict_site_product_pass,
    "site_product_rows_total": n_site_product_rows,
    "site_product_rows_coarse_pass": n_coarse_site_product_pass,
    "coverage_ok": has_coverage,
    "verdict": verdict,
    "pml_role": role,
    "blocking_for_strict_3x3": blocking_for_strict_3x3,
    "recommended_matrix_now": "STRICT_2x2_MODIS_GOSIF_BY_MODIS_GLEAM_PLUS_PML_COARSE_SENSITIVITY" if role == "COARSE_PML_SENSITIVITY_ONLY" else "STRICT_3x3_OK",
}])

site.to_csv(TAB / "Table_PRODUCT02by_pml_site_spatial_strictness.csv", index=False)
decision.to_csv(TAB / "Table_PRODUCT02bz_pml_spatial_strict_relock_decision.csv", index=False)

report = []
report.append("# Stage 1B.6N.2 PML spatial strict relock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Site/product spatial strictness")
report.append("")
report.append("```text")
report.append(site[[
    "point_id", "product", "n_nonmissing", "date_min", "date_max",
    "max_abs_lat_diff", "max_abs_lon_diff", "max_abs_coord_diff",
    "strict_spatial_pass", "coarse_sensitivity_pass"
]].to_string(index=False))
report.append("```")
report.append("")
report.append("## Scientific interpretation")
report.append("")
report.append("The local PML files contain usable GPP/ET time series, but the nearest-grid spatial mismatch is too large for a strict tower-centered product-matrix claim. PML should be treated as a coarse sensitivity product unless a better-resolution or properly point-extracted PML source is located.")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Do not present PML as equivalent to MODIS/GOSIF/GLEAM in a strict tower-centered 3x3 matrix when nearest-grid mismatch is several degrees.")
report.append("")

(TXT / "STAGE1B6N2_PML_SPATIAL_STRICT_RELOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6N.2_pml_spatial_strict_relock",
    "status": verdict,
    "pml_role": role,
    "blocking_for_strict_3x3": bool(blocking_for_strict_3x3),
    "outputs": {
        "decision": str(TAB / "Table_PRODUCT02bz_pml_spatial_strict_relock_decision.csv"),
        "site_strictness": str(TAB / "Table_PRODUCT02by_pml_site_spatial_strictness.csv"),
        "report": str(TXT / "STAGE1B6N2_PML_SPATIAL_STRICT_RELOCK_REPORT.md"),
    }
}
(TAB / "STAGE1B6N2_PML_SPATIAL_STRICT_RELOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02bz_pml_spatial_strict_relock_decision.csv")
print("WROTE", TAB / "Table_PRODUCT02by_pml_site_spatial_strictness.csv")
print("WROTE", TXT / "STAGE1B6N2_PML_SPATIAL_STRICT_RELOCK_REPORT.md")
