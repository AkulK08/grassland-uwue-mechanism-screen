#!/usr/bin/env python
from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd

GPP_PRODUCTS = ["modis", "gosif", "pml"]
ET_PRODUCTS = ["modis", "gleam", "pml"]
METRICS = ["uwue", "iwue", "raw_wue"]
ET_FLOOR = 0.1

OUT = Path("results/project_final_nature_stage")
OUT.mkdir(parents=True, exist_ok=True)
STAGE = Path("data/processed/final_nature")
STAGE.mkdir(parents=True, exist_ok=True)

QA_CANDIDATES = [
    Path("data/processed/modis_qa_by_point_8day_wide.csv"),
    Path("data/processed/modis_qa_by_point_8day.csv"),
]

PAIRS = [
    ("data/processed/project_metric_matrix_raw.csv", STAGE / "project_metric_matrix_raw_FINAL3x3.csv"),
    ("data/processed/project_metric_matrix_co2corrected.csv", STAGE / "project_metric_matrix_co2corrected_FINAL3x3.csv"),
]

def parse_two_floats(x):
    vals = re.findall(r"-?\d+(?:\.\d+)?", str(x))
    if len(vals) < 2:
        return None
    return float(vals[0]), float(vals[1])

def coord_keys_from_point_id(series, decimals):
    out_lonlat = []
    out_latlon = []
    for v in series.astype(str):
        parsed = parse_two_floats(v)
        if parsed is None:
            out_lonlat.append(None)
            out_latlon.append(None)
        else:
            a, b = parsed
            out_lonlat.append(f"{round(a, decimals)}_{round(b, decimals)}")
            out_latlon.append(f"{round(b, decimals)}_{round(a, decimals)}")
    return pd.Series(out_lonlat, index=series.index), pd.Series(out_latlon, index=series.index)

def normalize_date(s):
    return pd.to_datetime(s, errors="coerce").dt.floor("D")

def bool_from_existing(s):
    if s.dtype == bool:
        return s.fillna(False)
    num = pd.to_numeric(s, errors="coerce")
    out = num.eq(1)
    st = s.astype(str).str.strip().str.lower()
    out = out | st.isin(["true", "1", "1.0", "yes", "y", "t"])
    return out.fillna(False)

def good_from_modis_qc_bits(s):
    q = pd.to_numeric(s, errors="coerce")
    q_int = q.fillna(-999999).astype(int)
    return ((q_int & 3) == 0) & q.notna()

def load_qa_table():
    loaded = []
    for p in QA_CANDIDATES:
        if not p.exists():
            continue
        try:
            q = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print("Could not read QA candidate", p, e)
            continue

        q.columns = [str(c).strip() for c in q.columns]
        q["__source_file"] = str(p)

        # If this is already wide, keep it.
        if {"point_id", "date"}.issubset(q.columns):
            loaded.append(q)
            continue

        # Try common AppEEARS-style names.
        point_col = None
        date_col = None
        for c in q.columns:
            cl = c.lower()
            if point_col is None and cl in ["point_id", "id", "site_id"]:
                point_col = c
            if date_col is None and ("date" in cl or "time" in cl):
                date_col = c

        if point_col and date_col:
            q = q.rename(columns={point_col: "point_id", date_col: "date"})
            loaded.append(q)

    if not loaded:
        raise SystemExit("No readable MODIS QA candidate table found.")

    qa = pd.concat(loaded, ignore_index=True, sort=False)

    if "point_id" not in qa.columns or "date" not in qa.columns:
        raise SystemExit(f"QA table has no point_id/date columns. Columns: {list(qa.columns)}")

    # Keep possible QA columns.
    wanted = ["point_id", "date", "__source_file"]
    for c in qa.columns:
        cl = c.lower()
        if c in ["Psn_QC_500m", "ET_QC_500m", "modis_gpp_qc_good", "modis_et_qc_good"]:
            wanted.append(c)
        elif "psn" in cl and "qc" in cl:
            wanted.append(c)
        elif "et" in cl and "qc" in cl:
            wanted.append(c)

    wanted = list(dict.fromkeys(wanted))
    qa = qa[wanted].copy()
    qa["point_id"] = qa["point_id"].astype(str)
    qa["date_norm"] = normalize_date(qa["date"])

    # Standardize QC column names if variants exist.
    if "Psn_QC_500m" not in qa.columns:
        psn_cands = [c for c in qa.columns if "psn" in c.lower() and "qc" in c.lower()]
        if psn_cands:
            qa = qa.rename(columns={psn_cands[0]: "Psn_QC_500m"})

    if "ET_QC_500m" not in qa.columns:
        et_cands = [c for c in qa.columns if "et" in c.lower() and "qc" in c.lower()]
        et_cands = [c for c in et_cands if c != "modis_et_qc_good"]
        if et_cands:
            qa = qa.rename(columns={et_cands[0]: "ET_QC_500m"})

    # Collapse duplicates after standardization.
    keep_cols = ["point_id", "date_norm", "__source_file"]
    for c in ["Psn_QC_500m", "ET_QC_500m", "modis_gpp_qc_good", "modis_et_qc_good"]:
        if c in qa.columns:
            keep_cols.append(c)
    qa = qa[keep_cols].drop_duplicates(["point_id", "date_norm"])

    return qa

def build_keys(df, prefix):
    df = df.copy()
    df[f"{prefix}_point_exact"] = df["point_id"].astype(str)
    for dec in [6, 5, 4, 3, 2]:
        lonlat, latlon = coord_keys_from_point_id(df["point_id"], dec)
        df[f"{prefix}_lonlat_{dec}"] = lonlat
        df[f"{prefix}_latlon_{dec}"] = latlon
    return df

def score_exact_join(m, qa, mkey, qkey):
    left = m[["__rowid", mkey, "date_norm"]].copy()
    right_cols = [qkey, "date_norm"]
    for c in ["Psn_QC_500m", "ET_QC_500m", "modis_gpp_qc_good", "modis_et_qc_good"]:
        if c in qa.columns:
            right_cols.append(c)
    right = qa[right_cols].copy()
    merged = left.merge(right, left_on=[mkey, "date_norm"], right_on=[qkey, "date_norm"], how="left")
    psn_match = merged["Psn_QC_500m"].notna().sum() if "Psn_QC_500m" in merged.columns else 0
    et_match = merged["ET_QC_500m"].notna().sum() if "ET_QC_500m" in merged.columns else 0
    flag_match = 0
    if "modis_gpp_qc_good" in merged.columns:
        flag_match += merged["modis_gpp_qc_good"].notna().sum()
    if "modis_et_qc_good" in merged.columns:
        flag_match += merged["modis_et_qc_good"].notna().sum()
    return int(max(psn_match, et_match, flag_match)), merged

def score_asof_join(m, qa, mkey, qkey, tolerance_days):
    left = m[["__rowid", mkey, "date_norm"]].dropna().copy()
    right_cols = [qkey, "date_norm"]
    for c in ["Psn_QC_500m", "ET_QC_500m", "modis_gpp_qc_good", "modis_et_qc_good"]:
        if c in qa.columns:
            right_cols.append(c)
    right = qa[right_cols].dropna(subset=[qkey, "date_norm"]).copy()

    parts = []
    for key, left_g in left.groupby(mkey):
        right_g = right[right[qkey] == key]
        if right_g.empty:
            continue
        left_g = left_g.sort_values("date_norm")
        right_g = right_g.sort_values("date_norm")
        out = pd.merge_asof(
            left_g,
            right_g,
            on="date_norm",
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
        )
        parts.append(out)

    if not parts:
        return 0, pd.DataFrame({"__rowid": m["__rowid"]})

    merged = pd.concat(parts, ignore_index=True)
    psn_match = merged["Psn_QC_500m"].notna().sum() if "Psn_QC_500m" in merged.columns else 0
    et_match = merged["ET_QC_500m"].notna().sum() if "ET_QC_500m" in merged.columns else 0
    flag_match = 0
    if "modis_gpp_qc_good" in merged.columns:
        flag_match += merged["modis_gpp_qc_good"].notna().sum()
    if "modis_et_qc_good" in merged.columns:
        flag_match += merged["modis_et_qc_good"].notna().sum()
    return int(max(psn_match, et_match, flag_match)), merged

def find_best_qa_match(matrix_df, qa):
    m = matrix_df[["point_id", "date"]].copy()
    m["__rowid"] = np.arange(len(m))
    m["date_norm"] = normalize_date(m["date"])
    m = build_keys(m, "m")

    q = qa.copy()
    q = q.rename(columns={"point_id": "qa_point_id"})
    q = q.rename(columns={"qa_point_id": "point_id"})
    q = build_keys(q, "q")

    candidates = []

    key_pairs = [("m_point_exact", "q_point_exact")]
    for dec in [6, 5, 4, 3, 2]:
        key_pairs.extend([
            (f"m_lonlat_{dec}", f"q_lonlat_{dec}"),
            (f"m_lonlat_{dec}", f"q_latlon_{dec}"),
            (f"m_latlon_{dec}", f"q_lonlat_{dec}"),
            (f"m_latlon_{dec}", f"q_latlon_{dec}"),
        ])

    report_rows = []

    for mkey, qkey in key_pairs:
        score, merged = score_exact_join(m, q, mkey, qkey)
        report_rows.append({"mode": "exact", "mkey": mkey, "qkey": qkey, "tolerance_days": 0, "matched_rows": score})
        candidates.append((score, "exact", mkey, qkey, 0, merged))

        for tol in [4, 8, 16]:
            score, merged = score_asof_join(m, q, mkey, qkey, tol)
            report_rows.append({"mode": "nearest", "mkey": mkey, "qkey": qkey, "tolerance_days": tol, "matched_rows": score})
            candidates.append((score, "nearest", mkey, qkey, tol, merged))

    report = pd.DataFrame(report_rows).sort_values("matched_rows", ascending=False)
    report.to_csv(OUT / "modis_qa_match_attempts.csv", index=False)
    print("Top QA match attempts:")
    print(report.head(20).to_string(index=False))

    best = max(candidates, key=lambda x: x[0])
    score, mode, mkey, qkey, tol, merged = best
    print("BEST QA MATCH:", {"matched_rows": score, "mode": mode, "mkey": mkey, "qkey": qkey, "tolerance_days": tol})

    if score == 0:
        diag = {
            "matrix_point_sample": matrix_df["point_id"].astype(str).head(10).tolist(),
            "matrix_date_sample": matrix_df["date"].astype(str).head(10).tolist(),
            "qa_point_sample": qa["point_id"].astype(str).head(10).tolist(),
            "qa_date_sample": qa["date_norm"].astype(str).head(10).tolist(),
            "qa_columns": list(qa.columns),
        }
        pd.Series(diag).to_json(OUT / "modis_qa_no_match_diagnostics.json", indent=2)
        raise SystemExit("Could not match AppEEARS QA to metric matrix. See results/project_final_nature_stage/modis_qa_match_attempts.csv and modis_qa_no_match_diagnostics.json")

    return merged

def apply_matched_qa(df, matched):
    df = df.copy()
    matched = matched.copy()

    qa_cols = ["__rowid"]
    for c in ["Psn_QC_500m", "ET_QC_500m", "modis_gpp_qc_good", "modis_et_qc_good"]:
        if c in matched.columns:
            qa_cols.append(c)

    q = matched[qa_cols].drop_duplicates("__rowid")
    df["__rowid"] = np.arange(len(df))
    df = df.drop(columns=[c for c in ["Psn_QC_500m", "ET_QC_500m"] if c in df.columns])
    df = df.merge(q, on="__rowid", how="left")

    if "Psn_QC_500m" in df.columns and df["Psn_QC_500m"].notna().sum() > 0:
        df["modis_gpp_qc_good"] = good_from_modis_qc_bits(df["Psn_QC_500m"])
        print("RECOMPUTED modis_gpp_qc_good from matched Psn_QC_500m")
    elif "modis_gpp_qc_good" in df.columns and df["modis_gpp_qc_good"].notna().sum() > 0:
        df["modis_gpp_qc_good"] = bool_from_existing(df["modis_gpp_qc_good"])
        print("USED matched/existing modis_gpp_qc_good")
    else:
        raise SystemExit("No matched MODIS GPP QA available")

    if "ET_QC_500m" in df.columns and df["ET_QC_500m"].notna().sum() > 0:
        df["modis_et_qc_good"] = good_from_modis_qc_bits(df["ET_QC_500m"])
        print("RECOMPUTED modis_et_qc_good from matched ET_QC_500m")
    elif "modis_et_qc_good" in df.columns and df["modis_et_qc_good"].notna().sum() > 0:
        df["modis_et_qc_good"] = bool_from_existing(df["modis_et_qc_good"])
        print("USED matched/existing modis_et_qc_good")
    else:
        raise SystemExit("No matched MODIS ET QA available")

    df = df.drop(columns=["__rowid"])
    return df

def repair_one(src, dst, qa):
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise SystemExit(f"Missing source matrix: {src}")

    shutil.copy2(src, dst)
    print()
    print("STAGED", src, "->", dst)

    df = pd.read_csv(dst, low_memory=False)
    matched = find_best_qa_match(df, qa)
    df = apply_matched_qa(df, matched)

    print("modis_gpp_qc_good fraction:", float(df["modis_gpp_qc_good"].mean()))
    print("modis_et_qc_good fraction:", float(df["modis_et_qc_good"].mean()))

    vpd_col = "vpd_for_metric" if "vpd_for_metric" in df.columns else "vpd"
    vpd = pd.to_numeric(df[vpd_col], errors="coerce")
    vpd_safe = vpd.where(vpd > 0)

    summary = []

    for gpp in GPP_PRODUCTS:
        gcol = f"gpp_{gpp}"
        if gcol not in df.columns:
            raise SystemExit(f"Missing {gcol} in {dst}")
        g = pd.to_numeric(df[gcol], errors="coerce")

        for et in ET_PRODUCTS:
            ecol = f"et_{et}"
            if ecol not in df.columns:
                raise SystemExit(f"Missing {ecol} in {dst}")
            e = pd.to_numeric(df[ecol], errors="coerce")

            valid = g.notna() & e.notna() & vpd_safe.notna() & (e > ET_FLOOR)

            if gpp == "modis":
                valid &= df["modis_gpp_qc_good"]
            if et == "modis":
                valid &= df["modis_et_qc_good"]

            combo = f"{gpp}_{et}"

            raw = (g / e).where(valid)
            uwue = (g * np.sqrt(vpd_safe) / e).where(valid)
            iwue = (g * vpd_safe / e).where(valid)

            df[f"raw_wue_{combo}"] = raw
            df[f"uwue_{combo}"] = uwue
            df[f"iwue_{combo}"] = iwue

            df[f"log_raw_wue_{combo}"] = np.log(raw.where(raw > 0))
            df[f"log_uwue_{combo}"] = np.log(uwue.where(uwue > 0))
            df[f"log_iwue_{combo}"] = np.log(iwue.where(iwue > 0))

            for metric in METRICS:
                c = f"log_{metric}_{combo}"
                usable_points_50 = int(df.groupby("point_id")[c].apply(lambda s: s.notna().sum() >= 50).sum())
                summary.append({
                    "matrix": dst.name,
                    "combo": combo,
                    "metric": metric,
                    "nonnull": int(df[c].notna().sum()),
                    "usable_points_min50": usable_points_50,
                })

    df.to_csv(dst, index=False)
    print("WROTE", dst, df.shape)

    sm = pd.DataFrame(summary)
    count_path = OUT / f"metric_matrix_combo_counts_{dst.stem}.csv"
    sm.to_csv(count_path, index=False)
    print("WROTE", count_path)

    piv = sm.pivot_table(index="combo", columns="metric", values="usable_points_min50", aggfunc="first")
    print(piv.to_string())

    expected = {f"{g}_{e}" for g in GPP_PRODUCTS for e in ET_PRODUCTS}
    if set(piv.index) != expected:
        raise SystemExit(f"3x3 staging failed. Expected {sorted(expected)}, got {sorted(piv.index)}")

    bad = []
    for combo in expected:
        for metric in METRICS:
            val = int(piv.loc[combo, metric])
            if val <= 0:
                bad.append((combo, metric, val))

    if bad:
        raise SystemExit(f"Some combos have zero usable points: {bad}")

qa = load_qa_table()
print("Loaded QA table:", qa.shape)
print("QA columns:", list(qa.columns))
print("QA point sample:", qa["point_id"].astype(str).head(5).tolist())
print("QA date sample:", qa["date_norm"].astype(str).head(5).tolist())

for src, dst in PAIRS:
    repair_one(src, dst, qa)

print()
print("FINAL 3x3 STAGING PASSED")
