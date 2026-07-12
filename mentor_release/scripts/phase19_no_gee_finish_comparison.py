from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/tower_centered_phase19_no_gee")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

FITS = TAB / "Table121_no_gee_gosif_gleam_satellite_response_by_site.csv"
MERGED = TAB / "Table120_no_gee_tower13_gosif_gleam_merged_timeseries.csv"
TOWERFIT = Path("results/tower_validation_broad_inventory/tables/Table89_tower_response_phenotypes_primary_by_site.csv")
TARGETS = Path("results/tower_satellite_extraction_targets_FINAL/MAIN_expanded_grassland_savanna_open_coordinates.csv")

def die(msg):
    raise SystemExit("\nERROR: " + msg + "\n")

def read(path):
    if not path.exists():
        die(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)

def first_existing(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

fits = read(FITS)
merged_ts = read(MERGED)
towerfit = read(TOWERFIT)
targets = read(TARGETS)

fits["site"] = fits["site"].astype(str)
towerfit["site"] = towerfit["site"].astype(str)

# Prefer no-GEE uWUE-like satellite metric; fallback to WUE.
preferred_metric = "log_uwue_gosif_gleam_tower_vpd"
primary = fits[fits["satellite_metric"].eq(preferred_metric)].copy()
if primary.empty:
    preferred_metric = "log_wue_gosif_gleam"
    primary = fits[fits["satellite_metric"].eq(preferred_metric)].copy()

if primary.empty:
    die("No usable satellite fits in Table121.")

tower_cols = [
    "site", "response_class", "post_slope", "slope_change",
    "p_tower_saturation_breakdown", "tower_metric", "stress_method",
    "n_fit_8day", "n_years"
]
tower_cols = [c for c in tower_cols if c in towerfit.columns]

comp = primary.merge(
    towerfit[tower_cols],
    on="site",
    how="left",
    suffixes=("_satellite", "_tower")
)

# Robust column naming after merge.
sat_post_col = first_existing(comp, ["post_slope_satellite", "post_slope"])
sat_change_col = first_existing(comp, ["slope_change_satellite", "slope_change"])
tower_post_col = first_existing(comp, ["post_slope_tower", "tower_post_slope"])
tower_change_col = first_existing(comp, ["slope_change_tower", "tower_slope_change"])
tower_class_col = first_existing(comp, ["response_class", "tower_response_class"])

if tower_class_col is None:
    die("Could not find tower response class column after merge. Columns=" + str(list(comp.columns)))
if sat_post_col is None:
    die("Could not find satellite post-slope column after merge. Columns=" + str(list(comp.columns)))
if tower_post_col is None:
    die("Could not find tower post-slope column after merge. Columns=" + str(list(comp.columns)))

comp = comp.rename(columns={
    tower_class_col: "tower_response_class",
    sat_post_col: "satellite_post_slope",
    sat_change_col: "satellite_slope_change" if sat_change_col else sat_change_col,
    tower_post_col: "tower_post_slope",
    tower_change_col: "tower_slope_change" if tower_change_col else tower_change_col,
})

# If rename with None inserted nothing, clean columns are okay.
comp["class_agreement_exact"] = comp["satellite_response_class"].astype(str).eq(
    comp["tower_response_class"].astype(str)
)

sat_limited = comp["satellite_response_class"].isin(["saturation", "breakdown"])
tower_limited = comp["tower_response_class"].isin(["saturation", "breakdown"])
comp["class_agreement_limited_vs_enhanced"] = sat_limited.eq(tower_limited)

comp["slope_direction_agreement"] = np.sign(
    pd.to_numeric(comp["satellite_post_slope"], errors="coerce")
).eq(
    np.sign(pd.to_numeric(comp["tower_post_slope"], errors="coerce"))
)

# Add target scope label.
scope_map = targets.rename(columns={"target_id": "site"}).copy()
scope_map["site"] = scope_map["site"].astype(str)
scope_map["validation_scope"] = "expanded_grassland_savanna_open"
comp = comp.merge(scope_map[["site", "validation_scope"]], on="site", how="left")

# Order useful columns.
front = [
    "site", "validation_scope",
    "tower_response_class", "satellite_response_class",
    "tower_post_slope", "satellite_post_slope",
    "tower_slope_change", "satellite_slope_change",
    "p_tower_saturation_breakdown", "p_satellite_saturation_breakdown",
    "p_satellite_breakdown", "p_satellite_enhancement",
    "class_agreement_exact",
    "class_agreement_limited_vs_enhanced",
    "slope_direction_agreement",
    "n_fit", "n_fit_8day", "n_years",
    "satellite_metric", "satellite_product_combo",
]
front = [c for c in front if c in comp.columns]
rest = [c for c in comp.columns if c not in front]
comp = comp[front + rest]

out_comp = TAB / "Table122_no_gee_tower_vs_satellite_gosif_gleam_comparison.csv"
comp.to_csv(out_comp, index=False)

summary = {
    "mode": "NO_GEE",
    "main_scope": "expanded_grassland_savanna_open",
    "main_coordinate_sites": int(targets.shape[0]),
    "satellite_product_combo": "GOSIF_GPP + GLEAM_ET",
    "stress_source": "tower VPD + tower SWC/precip proxy",
    "satellite_metric_used": preferred_metric,
    "merged_timeseries_rows": int(len(merged_ts)),
    "sites_with_satellite_fit": int(comp["site"].nunique()),
    "exact_class_agreement_fraction": float(comp["class_agreement_exact"].mean()) if len(comp) else None,
    "limited_vs_enhanced_agreement_fraction": float(comp["class_agreement_limited_vs_enhanced"].mean()) if len(comp) else None,
    "slope_direction_agreement_fraction": float(comp["slope_direction_agreement"].mean()) if len(comp) else None,
    "tower_class_counts": comp["tower_response_class"].value_counts().to_dict(),
    "satellite_class_counts": comp["satellite_response_class"].value_counts().to_dict(),
}

(TAB / "Table123_no_gee_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
pd.DataFrame([summary]).to_csv(TAB / "Table123_no_gee_validation_summary.csv", index=False)

# Figures.
try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    comp["satellite_response_class"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("No-GEE GOSIF/GLEAM satellite response classes")
    ax.set_ylabel("Sites")
    plt.tight_layout()
    fig.savefig(FIG / "Figure1_no_gee_gosif_gleam_satellite_class_counts.png", dpi=300)
    plt.close(fig)
except Exception as e:
    print("WARNING: Figure1 failed:", e)

try:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, g in comp.groupby("tower_response_class"):
        ax.scatter(g["tower_post_slope"], g["satellite_post_slope"], label=cls, s=70)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Tower post-slope")
    ax.set_ylabel("Satellite GOSIF/GLEAM post-slope")
    ax.set_title("No-GEE tower vs satellite post-slope")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(FIG / "Figure2_no_gee_tower_vs_satellite_post_slope.png", dpi=300)
    plt.close(fig)
except Exception as e:
    print("WARNING: Figure2 failed:", e)

show_cols = [
    "site", "tower_response_class", "satellite_response_class",
    "tower_post_slope", "satellite_post_slope",
    "tower_slope_change", "satellite_slope_change",
    "class_agreement_exact", "class_agreement_limited_vs_enhanced",
    "slope_direction_agreement", "n_fit"
]
show_cols = [c for c in show_cols if c in comp.columns]

report = []
report.append("# Phase 19 no-GEE GOSIF/GLEAM tower-centered validation")
report.append("")
report.append("## Summary")
report.append("")
for k, v in summary.items():
    report.append(f"- {k}: `{v}`")
report.append("")
report.append("## Main comparison table")
report.append("")
report.append("```text")
report.append(comp[show_cols].to_string(index=False))
report.append("```")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("This is the no-GEE tower-centered satellite validation slice. It tests whether locally sampled GOSIF GPP and GLEAM ET reproduce the tower-observed response phenotype at the 13 expanded grassland/savanna/open tower sites.")
report.append("")
report.append("This does not complete the full 3x3 product matrix because MODIS/PML/ERA5 extraction was blocked by Earth Engine. It is still useful as the first direct tower-centered satellite product check.")
report.append("")

out_report = TXT / "PHASE19_NO_GEE_GOSIF_GLEAM_VALIDATION_REPORT.md"
out_report.write_text("\n".join(report), encoding="utf-8")

print("WROTE", out_comp, comp.shape)
print("WROTE", TAB / "Table123_no_gee_validation_summary.csv")
print("WROTE", out_report)
print("")
print("\n".join(report))
