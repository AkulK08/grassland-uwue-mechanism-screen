from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6p_strict_2x2_response_table")
TAB = OUT / "tables"
TXT = OUT / "text"
DATA = Path("data/processed/stage1b6p")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

TARGET_FILE = Path("data/raw_local/no_gee_point_requests/FINAL_STRICT_no_gee_product_points_for_appeears.csv")

MODIS_GPP = Path("data/raw_local/no_gee_direct_point_extract_full/MODIS_GPP_MOD17_FULL_direct_earthdata_point_samples.csv")
MODIS_ET = Path("data/raw_local/no_gee_direct_point_extract_full/MODIS_ET_MOD16_FULL_direct_earthdata_point_samples.csv")
MODIS_LAI = Path("data/raw_local/no_gee_direct_point_extract_full/MODIS_LAI_MCD15_FULL_direct_earthdata_point_samples.csv")
BURN = Path("data/raw_local/no_gee_direct_point_extract_full/MCD64A1_BURNED_AREA_FULL_direct_earthdata_point_samples.csv")

GOSIF_CANDIDATES = [
    Path("data/raw/tower_centered_phase19/agents/gosif_tower13_point_timeseries.csv"),
    Path("data/raw/agents/gosif_point_timeseries.csv"),
]

GLEAM_CANDIDATES = [
    Path("data/raw/tower_centered_phase19/agents/gleam_tower13_point_timeseries.csv"),
    Path("data/raw/agents/gleam_point_timeseries.csv"),
]

PML_GPP = Path("data/raw_local/pml_final13_point_extract/PML_GPP_FINAL13_point_samples.csv")
PML_ET = Path("data/raw_local/pml_final13_point_extract/PML_ET_FINAL13_point_samples.csv")
PML_RELOCK = Path("results/stage1b6n2_pml_spatial_strict_relock/tables/Table_PRODUCT02bz_pml_spatial_strict_relock_decision.csv")

def read_targets():
    df = pd.read_csv(TARGET_FILE)
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[c] for c in ["id", "point_id", "site_id", "site"] if c in cols), None)
    lat_col = next((cols[c] for c in ["lat", "latitude"] if c in cols), None)
    lon_col = next((cols[c] for c in ["lon", "longitude"] if c in cols), None)
    if not id_col or not lat_col or not lon_col:
        raise ValueError(f"Could not parse target columns: {list(df.columns)}")
    out = pd.DataFrame({
        "point_id": df[id_col].astype(str),
        "lat": pd.to_numeric(df[lat_col]),
        "lon": pd.to_numeric(df[lon_col]),
    }).drop_duplicates()
    return out

TARGETS = read_targets()
TARGET_IDS = set(TARGETS["point_id"])

def load_modis_layer(path, layer, value_name, qc_layer=None, qc_name=None):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["point_id"] = df["id"].astype(str)

    val = df[df["layer"].eq(layer)].copy()
    val = val[val["point_id"].isin(TARGET_IDS)].copy()
    val[value_name] = pd.to_numeric(val["scaled_value"], errors="coerce")
    val[f"{value_name}_sample_status"] = val["sample_status"].astype(str)
    val = val[["point_id", "date", value_name, f"{value_name}_sample_status"]]

    if qc_layer and qc_name:
        qc = df[df["layer"].eq(qc_layer)].copy()
        qc = qc[qc["point_id"].isin(TARGET_IDS)].copy()
        qc[qc_name] = pd.to_numeric(qc["raw_value"], errors="coerce")
        qc[f"{qc_name}_sample_status"] = qc["sample_status"].astype(str)
        qc = qc[["point_id", "date", qc_name, f"{qc_name}_sample_status"]]
        val = val.merge(qc, on=["point_id", "date"], how="left")

    return val.drop_duplicates(["point_id", "date"])

def find_existing(cands, label):
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"No {label} candidate exists: {[str(x) for x in cands]}")

def load_agent_product(candidates, value_col, product_name):
    path = find_existing(candidates, product_name)
    df = pd.read_csv(path)
    if "date" not in df.columns or "point_id" not in df.columns:
        raise ValueError(f"{path} missing date or point_id. Columns={list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"{path} missing {value_col}. Columns={list(df.columns)}")

    df["point_id"] = df["point_id"].astype(str)
    df = df[df["point_id"].isin(TARGET_IDS)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df[["point_id", "date", value_col]].drop_duplicates(["point_id", "date"]), path

def load_lai():
    if not MODIS_LAI.exists():
        return pd.DataFrame(columns=["point_id", "date", "lai_modis", "lai_qc"])
    lai = load_modis_layer(MODIS_LAI, "Lai_500m", "lai_modis", "FparLai_QC", "lai_qc")
    return lai

def load_burn():
    if not BURN.exists():
        return pd.DataFrame(columns=["point_id", "year_month", "burn_date_raw", "burn_qa", "burned_this_month"])

    df = pd.read_csv(BURN)
    df["date"] = pd.to_datetime(df["date"])
    df["point_id"] = df["id"].astype(str)
    df = df[df["point_id"].isin(TARGET_IDS)].copy()
    df["year_month"] = df["date"].dt.strftime("%Y-%m")

    bd = df[df["layer"].eq("Burn_Date")].copy()
    qa = df[df["layer"].eq("QA")].copy()

    bd["burn_date_raw"] = pd.to_numeric(bd["raw_value"], errors="coerce")
    qa["burn_qa"] = pd.to_numeric(qa["raw_value"], errors="coerce")

    out = bd[["point_id", "year_month", "burn_date_raw"]].merge(
        qa[["point_id", "year_month", "burn_qa"]],
        on=["point_id", "year_month"],
        how="left"
    )
    out["burned_this_month"] = out["burn_date_raw"].fillna(0).gt(0)
    return out.drop_duplicates(["point_id", "year_month"])

def build_combo(gpp_df, et_df, gpp_col, et_col, gpp_product, et_product, role):
    x = gpp_df[["point_id", "date", gpp_col]].copy()
    y = et_df[["point_id", "date", et_col]].copy()
    out = x.merge(y, on=["point_id", "date"], how="inner")
    out = out.rename(columns={gpp_col: "gpp", et_col: "et"})
    out["gpp_product"] = gpp_product
    out["et_product"] = et_product
    out["matrix_role"] = role
    out["gpp"] = pd.to_numeric(out["gpp"], errors="coerce")
    out["et"] = pd.to_numeric(out["et"], errors="coerce")
    out["wue"] = np.where((out["gpp"] > 0) & (out["et"] > 0), out["gpp"] / out["et"], np.nan)
    out["log_gpp"] = np.where(out["gpp"] > 0, np.log(out["gpp"]), np.nan)
    out["log_et"] = np.where(out["et"] > 0, np.log(out["et"]), np.nan)
    out["log_wue"] = np.where(out["wue"] > 0, np.log(out["wue"]), np.nan)
    return out

errors = []
sources = []

try:
    modis_gpp = load_modis_layer(MODIS_GPP, "Gpp_500m", "gpp_modis", "Psn_QC_500m", "gpp_modis_qc")
    sources.append({"source": "MODIS_GPP", "path": str(MODIS_GPP), "rows": len(modis_gpp), "sites": modis_gpp["point_id"].nunique(), "dates": modis_gpp["date"].nunique()})
except Exception as e:
    errors.append({"source": "MODIS_GPP", "error": repr(e)})

try:
    modis_et = load_modis_layer(MODIS_ET, "ET_500m", "et_modis", "ET_QC_500m", "et_modis_qc")
    sources.append({"source": "MODIS_ET", "path": str(MODIS_ET), "rows": len(modis_et), "sites": modis_et["point_id"].nunique(), "dates": modis_et["date"].nunique()})
except Exception as e:
    errors.append({"source": "MODIS_ET", "error": repr(e)})

try:
    gosif, gosif_path = load_agent_product(GOSIF_CANDIDATES, "gpp_gosif", "GOSIF_GPP")
    sources.append({"source": "GOSIF_GPP", "path": str(gosif_path), "rows": len(gosif), "sites": gosif["point_id"].nunique(), "dates": gosif["date"].nunique()})
except Exception as e:
    errors.append({"source": "GOSIF_GPP", "error": repr(e)})

try:
    gleam, gleam_path = load_agent_product(GLEAM_CANDIDATES, "et_gleam", "GLEAM_ET")
    sources.append({"source": "GLEAM_ET", "path": str(gleam_path), "rows": len(gleam), "sites": gleam["point_id"].nunique(), "dates": gleam["date"].nunique()})
except Exception as e:
    errors.append({"source": "GLEAM_ET", "error": repr(e)})

try:
    lai = load_lai()
    sources.append({"source": "MODIS_LAI", "path": str(MODIS_LAI), "rows": len(lai), "sites": lai["point_id"].nunique() if len(lai) else 0, "dates": lai["date"].nunique() if len(lai) else 0})
except Exception as e:
    errors.append({"source": "MODIS_LAI", "error": repr(e)})
    lai = pd.DataFrame(columns=["point_id", "date", "lai_modis", "lai_qc"])

try:
    burn = load_burn()
    sources.append({"source": "MCD64A1_BURNED_AREA", "path": str(BURN), "rows": len(burn), "sites": burn["point_id"].nunique() if len(burn) else 0, "dates": burn["year_month"].nunique() if len(burn) else 0})
except Exception as e:
    errors.append({"source": "MCD64A1_BURNED_AREA", "error": repr(e)})
    burn = pd.DataFrame(columns=["point_id", "year_month", "burn_date_raw", "burn_qa", "burned_this_month"])

strict_tables = []
if "modis_gpp" in locals() and "modis_et" in locals():
    strict_tables.append(build_combo(modis_gpp, modis_et, "gpp_modis", "et_modis", "MODIS_GPP_MOD17", "MODIS_ET_MOD16", "STRICT_PRIMARY"))
if "modis_gpp" in locals() and "gleam" in locals():
    strict_tables.append(build_combo(modis_gpp, gleam, "gpp_modis", "et_gleam", "MODIS_GPP_MOD17", "GLEAM_ET", "STRICT_PRIMARY"))
if "gosif" in locals() and "modis_et" in locals():
    strict_tables.append(build_combo(gosif, modis_et, "gpp_gosif", "et_modis", "GOSIF_GPP", "MODIS_ET_MOD16", "STRICT_PRIMARY"))
if "gosif" in locals() and "gleam" in locals():
    strict_tables.append(build_combo(gosif, gleam, "gpp_gosif", "et_gleam", "GOSIF_GPP", "GLEAM_ET", "STRICT_PRIMARY"))

if strict_tables:
    strict = pd.concat(strict_tables, ignore_index=True)
else:
    strict = pd.DataFrame()

# Add QA/covariates.
if len(strict):
    strict = strict.merge(lai[["point_id", "date", "lai_modis", "lai_qc"]], on=["point_id", "date"], how="left")
    strict["year_month"] = pd.to_datetime(strict["date"]).dt.strftime("%Y-%m")
    strict = strict.merge(burn, on=["point_id", "year_month"], how="left")
    strict["burned_this_month"] = strict["burned_this_month"].fillna(False)
    strict["qa_keep_unburned"] = ~strict["burned_this_month"].astype(bool)
    strict["year"] = pd.to_datetime(strict["date"]).dt.year
    strict["doy"] = pd.to_datetime(strict["date"]).dt.dayofyear

# PML coarse sensitivity.
pml = pd.DataFrame()
try:
    if PML_GPP.exists() and PML_ET.exists():
        pg = pd.read_csv(PML_GPP)
        pe = pd.read_csv(PML_ET)
        pg["date"] = pd.to_datetime(pg["date"]).dt.strftime("%Y-%m-%d")
        pe["date"] = pd.to_datetime(pe["date"]).dt.strftime("%Y-%m-%d")
        pml = build_combo(pg, pe, "gpp", "et", "PML_GPP", "PML_ET", "COARSE_SENSITIVITY_ONLY")
        pml["year"] = pd.to_datetime(pml["date"]).dt.year
        pml["doy"] = pd.to_datetime(pml["date"]).dt.dayofyear

        if PML_RELOCK.exists():
            relock = pd.read_csv(PML_RELOCK)
            pml["pml_role_lock"] = relock["pml_role"].iloc[0]
            pml["pml_spatial_strict_verdict"] = relock["verdict"].iloc[0]
            pml["pml_max_abs_lat_diff"] = relock["max_abs_lat_diff"].iloc[0]
            pml["pml_max_abs_lon_diff"] = relock["max_abs_lon_diff"].iloc[0]
except Exception as e:
    errors.append({"source": "PML_COARSE_SENSITIVITY", "error": repr(e)})

strict_out = DATA / "strict_2x2_response_table_final13.csv"
pml_out = DATA / "pml_coarse_sensitivity_response_table_final13.csv"
combined_out = DATA / "strict_2x2_plus_pml_coarse_response_table_final13.csv"

strict.to_csv(strict_out, index=False)
pml.to_csv(pml_out, index=False)
pd.concat([strict, pml], ignore_index=True, sort=False).to_csv(combined_out, index=False)

def summarize_table(df, label):
    if len(df) == 0:
        return []
    g = (
        df.groupby(["matrix_role", "gpp_product", "et_product"], dropna=False)
        .agg(
            n_rows=("date", "size"),
            n_sites=("point_id", "nunique"),
            n_dates=("date", "nunique"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            n_valid_wue=("wue", lambda s: int(pd.notna(s).sum())),
            n_positive_gpp=("gpp", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            n_positive_et=("et", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        )
        .reset_index()
    )
    g["table"] = label
    return g.to_dict("records")

coverage = pd.DataFrame(
    summarize_table(strict, "strict_2x2") + summarize_table(pml, "pml_coarse")
)
coverage.to_csv(TAB / "Table_PRODUCT02ce_response_table_coverage_summary.csv", index=False)

source_df = pd.DataFrame(sources)
source_df.to_csv(TAB / "Table_PRODUCT02cf_response_table_source_summary.csv", index=False)

err_df = pd.DataFrame(errors)
err_df.to_csv(TAB / "Table_PRODUCT02cg_response_table_errors.csv", index=False)

expected_strict_combos = {
    ("MODIS_GPP_MOD17", "MODIS_ET_MOD16"),
    ("MODIS_GPP_MOD17", "GLEAM_ET"),
    ("GOSIF_GPP", "MODIS_ET_MOD16"),
    ("GOSIF_GPP", "GLEAM_ET"),
}
got_strict_combos = set()
if len(strict):
    got_strict_combos = set(zip(strict["gpp_product"], strict["et_product"]))

strict_ok = expected_strict_combos.issubset(got_strict_combos)
strict_sites_ok = len(strict) and strict.groupby(["gpp_product", "et_product"])["point_id"].nunique().min() == 13
strict_dates_ok = len(strict) and strict.groupby(["gpp_product", "et_product"])["date"].nunique().min() >= 1000
pml_ok = len(pml) > 0 and pml["point_id"].nunique() == 13 and pml["date"].nunique() >= 300
no_errors = len(err_df) == 0

if strict_ok and strict_sites_ok and strict_dates_ok and pml_ok and no_errors:
    verdict = "STRICT_2X2_RESPONSE_TABLE_READY_WITH_PML_COARSE_SENSITIVITY"
    blocking_next = False
else:
    verdict = "RESPONSE_TABLE_INCOMPLETE_OR_NEEDS_REVIEW"
    blocking_next = True

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "strict_rows": int(len(strict)),
    "pml_coarse_rows": int(len(pml)),
    "strict_combo_count": int(len(got_strict_combos)),
    "strict_all_four_combos_present": bool(strict_ok),
    "strict_sites_ok_13": bool(strict_sites_ok),
    "strict_min_dates_ok_1000": bool(strict_dates_ok),
    "pml_coarse_ok": bool(pml_ok),
    "n_errors": int(len(err_df)),
    "verdict": verdict,
    "blocking_next_stage": bool(blocking_next),
    "next_stage": "BUILD_STRESS_AND_GROWING_SEASON_DESIGN" if not blocking_next else "FIX_RESPONSE_TABLE_INPUTS",
}])
decision.to_csv(TAB / "Table_PRODUCT02ch_response_table_decision.csv", index=False)

report = []
report.append("# Stage 1B.6P strict 2x2 response table")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Coverage summary")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False) if len(coverage) else "No coverage.")
report.append("```")
report.append("")
report.append("## Source summary")
report.append("")
report.append("```text")
report.append(source_df.to_string(index=False) if len(source_df) else "No sources.")
report.append("```")
report.append("")
report.append("## Errors")
report.append("")
report.append("```text")
report.append(err_df.to_string(index=False) if len(err_df) else "No errors.")
report.append("```")
report.append("")
report.append("## Outputs")
report.append("")
report.append(f"- Strict 2x2 response table: `{strict_out}`")
report.append(f"- PML coarse sensitivity response table: `{pml_out}`")
report.append(f"- Combined response table: `{combined_out}`")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Primary inference uses the strict 2x2 tower-centered matrix only. PML rows are included only as coarse-grid sensitivity and must not be described as strict tower-centered evidence.")
report.append("")

(TXT / "STAGE1B6P_STRICT_2X2_RESPONSE_TABLE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6P_strict_2x2_response_table",
    "status": verdict,
    "blocking_next_stage": bool(blocking_next),
    "outputs": {
        "strict_response": str(strict_out),
        "pml_coarse_response": str(pml_out),
        "combined_response": str(combined_out),
        "coverage_summary": str(TAB / "Table_PRODUCT02ce_response_table_coverage_summary.csv"),
        "decision": str(TAB / "Table_PRODUCT02ch_response_table_decision.csv"),
        "report": str(TXT / "STAGE1B6P_STRICT_2X2_RESPONSE_TABLE_REPORT.md"),
    }
}
(TAB / "STAGE1B6P_STRICT_2X2_RESPONSE_TABLE_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", strict_out)
print("WROTE", pml_out)
print("WROTE", combined_out)
print("WROTE", TAB / "Table_PRODUCT02ce_response_table_coverage_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02ch_response_table_decision.csv")
print("WROTE", TXT / "STAGE1B6P_STRICT_2X2_RESPONSE_TABLE_REPORT.md")
