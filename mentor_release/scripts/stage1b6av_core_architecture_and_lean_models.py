from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6av_core_architecture_lean_models"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

POINTS = ROOT / "results/stage1b6ai_reza_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"

df = pd.read_csv(POINTS, low_memory=False)
df = df.replace([np.inf, -np.inf], np.nan)

# -----------------------------
# Core variables
# -----------------------------

y = "latent_slope_change"
c4 = "c4_fraction"
vpd = "mean_vpd"

core_controls = [
    "aridity",
    "mean_annual_temperature",
    "mean_annual_precipitation",
    "growing_season_mean_lai",
    "mean_soil_moisture",
]

extra_controls = [
    "rooting_depth",
]

all_possible = [y, c4, vpd] + core_controls + extra_controls

missing = [col for col in all_possible if col not in df.columns]
if missing:
    raise SystemExit(f"Missing expected columns: {missing}")

# -----------------------------
# Optional natural/no-crop filter detection
# -----------------------------

filter_cols = {
    "crop_clean": None,
    "grassland_like": None,
    "natural_grassland_like": None,
    "no_crop": None,
}

for target in list(filter_cols):
    for col in df.columns:
        cl = col.lower()
        if target in cl:
            filter_cols[target] = col
            break

base = df.copy()

# Do NOT force crop/grassland filter unless columns exist and are interpretable.
# Save value counts so you can decide if prior pipeline was filtering too harshly.
vc_rows = []
for label, col in filter_cols.items():
    if col is not None:
        counts = base[col].value_counts(dropna=False)
        for val, n in counts.items():
            vc_rows.append({"filter_label": label, "column": col, "value": val, "n": n})
pd.DataFrame(vc_rows).to_csv(TAB / "detected_filter_value_counts.csv", index=False)

# Current best primary subset:
# use all points with y, C4, and VPD.
signal = base.dropna(subset=[y, c4, vpd]).copy()

# -----------------------------
# Total point audit
# -----------------------------

audit_rows = []

def add_audit(name, d, required_cols=None):
    required_cols = required_cols or []
    audit_rows.append({
        "stage": name,
        "n": len(d),
        "required_cols": "; ".join(required_cols),
    })

add_audit("raw_point_table", base)
add_audit("core_signal_complete_y_c4_vpd", signal, [y, c4, vpd])

for controls_name, controls in [
    ("core_climate_controls_no_rooting", core_controls),
    ("core_plus_rooting_depth", core_controls + extra_controls),
]:
    cols = [y, c4, vpd] + controls
    add_audit(
        f"complete_case_{controls_name}",
        base.dropna(subset=cols),
        cols
    )

# Leave-one-variable-out loss from core signal sample.
loss_rows = []
for col in core_controls + extra_controls:
    before = len(signal)
    after = len(signal.dropna(subset=[col]))
    loss_rows.append({
        "control": col,
        "n_before": before,
        "n_after_requiring_control": after,
        "n_lost": before - after,
        "lost_rate": (before - after) / before if before else np.nan,
    })

pd.DataFrame(audit_rows).to_csv(TAB / "sample_size_architecture.csv", index=False)
pd.DataFrame(loss_rows).sort_values("n_lost", ascending=False).to_csv(
    TAB / "control_missingness_loss_from_core_signal.csv", index=False
)

# Save exact point membership.
membership = base.copy()
membership["_row_index"] = membership.index
membership["_in_core_signal_y_c4_vpd"] = membership[[y, c4, vpd]].notna().all(axis=1)
membership["_in_core_climate_no_rooting"] = membership[[y, c4, vpd] + core_controls].notna().all(axis=1)
membership["_in_full_plus_rooting"] = membership[[y, c4, vpd] + core_controls + extra_controls].notna().all(axis=1)

id_cols = [c for c in ["point_id", "lat", "lon"] if c in membership.columns]
front = ["_row_index"] + id_cols + [
    "_in_core_signal_y_c4_vpd",
    "_in_core_climate_no_rooting",
    "_in_full_plus_rooting",
]
front = [c for c in front if c in membership.columns]
membership[front + [c for c in membership.columns if c not in front]].to_csv(
    TAB / "point_membership_all_samples.csv",
    index=False
)

membership[membership["_in_core_signal_y_c4_vpd"]].to_csv(
    TAB / "POINTS_primary_core_signal_n142_expected.csv",
    index=False
)

membership[membership["_in_core_climate_no_rooting"]].to_csv(
    TAB / "POINTS_primary_core_climate_no_rooting.csv",
    index=False
)

membership[membership["_in_full_plus_rooting"]].to_csv(
    TAB / "POINTS_strict_full_plus_rooting_n112_expected.csv",
    index=False
)

# -----------------------------
# Model fitting
# -----------------------------

def zscore_col(d, col):
    s = d[col]
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        d[col + "_z"] = np.nan
    else:
        d[col + "_z"] = (s - s.mean()) / sd

model_df = base.copy()
for col in [y, c4, vpd] + core_controls + extra_controls:
    zscore_col(model_df, col)

zy = y + "_z"
zc4 = c4 + "_z"
zvpd = vpd + "_z"

z_core_controls = [col + "_z" for col in core_controls]
z_extra_controls = [col + "_z" for col in extra_controls]

specs = [
    {
        "model": "M0_C4_only",
        "formula": f"{zy} ~ {zc4}",
        "sample_cols": [zy, zc4],
        "interpretation": "raw C4 signal only",
    },
    {
        "model": "M1_VPD_only",
        "formula": f"{zy} ~ {zvpd}",
        "sample_cols": [zy, zvpd],
        "interpretation": "baseline VPD climate signal only",
    },
    {
        "model": "M2_C4_plus_VPD",
        "formula": f"{zy} ~ {zc4} + {zvpd}",
        "sample_cols": [zy, zc4, zvpd],
        "interpretation": "does C4 contribute after VPD control?",
    },
    {
        "model": "M3_C4xVPD_primary_interaction",
        "formula": f"{zy} ~ {zc4} * {zvpd}",
        "sample_cols": [zy, zc4, zvpd],
        "interpretation": "primary biological interaction: does C4 modify VPD response?",
    },
    {
        "model": "M4_interaction_plus_core_climate_no_rooting",
        "formula": f"{zy} ~ {zc4} * {zvpd} + " + " + ".join(z_core_controls),
        "sample_cols": [zy, zc4, zvpd] + z_core_controls,
        "interpretation": "primary controlled model without rooting-depth sample loss",
    },
    {
        "model": "M5_interaction_plus_core_plus_rooting_sensitivity",
        "formula": f"{zy} ~ {zc4} * {zvpd} + " + " + ".join(z_core_controls + z_extra_controls),
        "sample_cols": [zy, zc4, zvpd] + z_core_controls + z_extra_controls,
        "interpretation": "strict sensitivity model only",
    },
]

coef_rows = []
fit_rows = []

for spec in specs:
    d = model_df.dropna(subset=spec["sample_cols"]).copy()
    if len(d) < 20:
        continue

    fit = smf.ols(spec["formula"], data=d).fit(cov_type="HC3")

    fit_rows.append({
        "model": spec["model"],
        "n": int(fit.nobs),
        "r2": fit.rsquared,
        "adj_r2": fit.rsquared_adj,
        "aic": fit.aic,
        "bic": fit.bic,
        "formula": spec["formula"],
        "interpretation": spec["interpretation"],
    })

    for term in fit.params.index:
        coef_rows.append({
            "model": spec["model"],
            "n": int(fit.nobs),
            "term": term,
            "coef": fit.params[term],
            "se_hc3": fit.bse[term],
            "t": fit.tvalues[term],
            "p": fit.pvalues[term],
            "ci_low": fit.conf_int().loc[term, 0],
            "ci_high": fit.conf_int().loc[term, 1],
            "r2": fit.rsquared,
            "aic": fit.aic,
        })

fits = pd.DataFrame(fit_rows)
coefs = pd.DataFrame(coef_rows)

fits.to_csv(TAB / "tiered_model_fit_comparison.csv", index=False)
coefs.to_csv(TAB / "tiered_model_coefficients_hc3.csv", index=False)

# Delta R2/AIC relative to VPD-only and C4+VPD.
if not fits.empty:
    base_vpd_r2 = fits.loc[fits["model"] == "M1_VPD_only", "r2"]
    base_c4vpd_r2 = fits.loc[fits["model"] == "M2_C4_plus_VPD", "r2"]

    fits2 = fits.copy()
    if len(base_vpd_r2):
        fits2["delta_r2_vs_VPD_only"] = fits2["r2"] - float(base_vpd_r2.iloc[0])
    if len(base_c4vpd_r2):
        fits2["delta_r2_vs_C4_plus_VPD"] = fits2["r2"] - float(base_c4vpd_r2.iloc[0])
    fits2.to_csv(TAB / "tiered_model_fit_comparison_with_delta_r2.csv", index=False)

# -----------------------------
# Region / geography diagnostics
# -----------------------------

if "lat" in base.columns and "lon" in base.columns:
    geo = base.copy()
    geo["sahel_broad_lat10_20_lon-20_40"] = (
        geo["lat"].between(10, 20) & geo["lon"].between(-20, 40)
    )
    geo["sahel_core_lat12_18_lon-17_35"] = (
        geo["lat"].between(12, 18) & geo["lon"].between(-17, 35)
    )

    geo_summary = []
    for sample_name, mask in [
        ("raw", pd.Series(True, index=geo.index)),
        ("core_signal", geo[[y, c4, vpd]].notna().all(axis=1)),
        ("core_climate_no_rooting", geo[[y, c4, vpd] + core_controls].notna().all(axis=1)),
        ("strict_plus_rooting", geo[[y, c4, vpd] + core_controls + extra_controls].notna().all(axis=1)),
    ]:
        d = geo[mask]
        geo_summary.append({
            "sample": sample_name,
            "n": len(d),
            "n_sahel_broad": int(d["sahel_broad_lat10_20_lon-20_40"].sum()),
            "rate_sahel_broad": float(d["sahel_broad_lat10_20_lon-20_40"].mean()) if len(d) else np.nan,
            "n_sahel_core": int(d["sahel_core_lat12_18_lon-17_35"].sum()),
            "rate_sahel_core": float(d["sahel_core_lat12_18_lon-17_35"].mean()) if len(d) else np.nan,
            "lat_min": d["lat"].min() if len(d) else np.nan,
            "lat_max": d["lat"].max() if len(d) else np.nan,
            "lon_min": d["lon"].min() if len(d) else np.nan,
            "lon_max": d["lon"].max() if len(d) else np.nan,
        })

    pd.DataFrame(geo_summary).to_csv(TAB / "geographic_scope_and_sahel_check.csv", index=False)

# -----------------------------
# Human-readable memo
# -----------------------------

sample_arch = pd.read_csv(TAB / "sample_size_architecture.csv")
loss = pd.read_csv(TAB / "control_missingness_loss_from_core_signal.csv")

memo = []
memo.append("WUE core architecture / lean-model audit")
memo.append("=" * 60)
memo.append("")
memo.append(f"Input table: {POINTS}")
memo.append("")
memo.append("Sample architecture:")
memo.append(sample_arch.to_string(index=False))
memo.append("")
memo.append("Control loss from core signal sample:")
memo.append(loss.to_string(index=False))
memo.append("")
if (TAB / "tiered_model_fit_comparison_with_delta_r2.csv").exists():
    memo.append("Tiered model fit:")
    memo.append(pd.read_csv(TAB / "tiered_model_fit_comparison_with_delta_r2.csv").to_string(index=False))
    memo.append("")
memo.append("Interpretation rule:")
memo.append("- Use M3/M4 as the primary C4 x VPD test.")
memo.append("- Treat M5 with rooting depth as sensitivity only, because rooting depth causes extra complete-case loss.")
memo.append("- If C4 x VPD is significant in M3/M4, frame as functional composition modifying VPD response.")
memo.append("- If C4 x VPD is null and C4 adds little ΔR² after VPD, frame C4 mostly as a marker of VPD/climate regime.")

(TXT / "READ_ME_architecture_lean_model_audit.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nKey files:")
print(TAB / "sample_size_architecture.csv")
print(TAB / "control_missingness_loss_from_core_signal.csv")
print(TAB / "tiered_model_fit_comparison_with_delta_r2.csv")
print(TAB / "tiered_model_coefficients_hc3.csv")
print(TAB / "point_membership_all_samples.csv")
print(TAB / "geographic_scope_and_sahel_check.csv")
print(TXT / "READ_ME_architecture_lean_model_audit.txt")
