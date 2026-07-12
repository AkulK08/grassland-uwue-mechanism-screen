from pathlib import Path

p = Path("scripts/phase18_grassland_spatial_trait_lock.py")
s = p.read_text()

marker = '    recommendation = f"""# Phase 18 grassland + spatial/trait lock'
if marker not in s:
    raise SystemExit("Could not find broken recommendation block. Run: tail -80 scripts/phase18_grassland_spatial_trait_lock.py")

s = s[:s.index(marker)]

tail = r'''
    # Recommendation text.
    strict_n = int(df["is_strict_grassland_tower"].sum())
    expanded_n = int(df["is_expanded_grassland_savanna_tower"].sum())
    open_n = int(df["is_open_nonforest_tower"].sum())
    all_n = int(len(df))

    strict_sat = float(df.loc[df["is_strict_grassland_tower"], "tower_satbreak_event"].mean()) if strict_n else np.nan
    expanded_sat = float(df.loc[df["is_expanded_grassland_savanna_tower"], "tower_satbreak_event"].mean()) if expanded_n else np.nan
    open_sat = float(df.loc[df["is_open_nonforest_tower"], "tower_satbreak_event"].mean()) if open_n else np.nan
    all_sat = float(df["tower_satbreak_event"].mean()) if all_n else np.nan

    top_tests_txt = "No characteristic tests available."
    if not tests.empty:
        top_tests_txt = tests.head(30).to_string(index=False)

    top_groups_txt = "No spatial enrichment tests available."
    if not groups.empty:
        top_groups_txt = groups.head(30).to_string(index=False)

    top_rules_txt = "No rules available."
    if not rules.empty:
        top_rules_txt = rules.head(30).to_string(index=False)

    top_tree_txt = tree_text if tree_text.strip() else "No decision-tree rules available."

    verdict = {
        "all_tower_response_sites": all_n,
        "strict_grassland_GRA_sites": strict_n,
        "expanded_grassland_savanna_open_sites": expanded_n,
        "open_nonforest_sites": open_n,
        "all_tower_satbreak_fraction": all_sat,
        "strict_grassland_satbreak_fraction": strict_sat,
        "expanded_grassland_savanna_satbreak_fraction": expanded_sat,
        "open_nonforest_satbreak_fraction": open_sat,
        "satellite_proxy_source": sat_source,
        "geopandas_spatial_annotation_available": bool(GEOPANDAS_OK),
        "strict_grassland_main_analysis_ready": bool(strict_n >= 8),
        "expanded_open_main_analysis_ready": bool(expanded_n >= 10),
        "recommended_tower_validation_scope": (
            "strict_grassland_GRA" if strict_n >= 8
            else "expanded_grassland_savanna_open" if expanded_n >= 10
            else "all_tower_ecosystems_with_grassland_subset_as_sensitivity"
        ),
    }

    (OUT / "phase18_grassland_spatial_trait_lock_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str),
        encoding="utf-8"
    )

    recommendation_lines = [
        "# Phase 18 grassland + spatial/trait lock",
        "",
        "## Core counts",
        "",
        f"- All tower response sites: `{all_n}`",
        f"- Strict IGBP grassland sites, GRA only: `{strict_n}`",
        f"- Expanded grassland/savanna/open sites: `{expanded_n}`",
        f"- Open nonforest sites: `{open_n}`",
        "",
        "## Tower saturation/breakdown fractions",
        "",
        f"- All tower sites: `{pct(all_sat)}`",
        f"- Strict grassland sites: `{pct(strict_sat)}`",
        f"- Expanded grassland/savanna/open sites: `{pct(expanded_sat)}`",
        f"- Open nonforest sites: `{pct(open_sat)}`",
        "",
        "## Interpretation",
        "",
        "This phase separates the mentor thesis into two levels.",
        "",
        "1. General ecosystem-flux thesis: the all-site tower phenotype can support a broad ecosystem WUE/uWUE response paper because it uses many eddy-covariance sites and captures saturation, breakdown, enhancement, and inconclusive response classes.",
        "",
        "2. Grassland-specific thesis: the strict grassland version depends on how many valid GRA sites remain after metadata repair. If strict GRA has too few sites, use expanded grassland/savanna/open ecosystems as the tower validation scope and present strict GRA as a sensitivity check.",
        "",
        "## Recommended hierarchy for the paper",
        "",
        "- Main tower validation: expanded grassland/savanna/open tower subset if n is large enough.",
        "- Sensitivity: strict GRA-only towers.",
        "- Contrast: forest towers or all ecosystem towers.",
        "- Final validation: extract satellite products at the tower coordinates listed in the Table108 coordinate files.",
        "",
        "## Top characteristic/trait association tests",
        "",
        "```text",
        top_tests_txt,
        "```",
        "",
        "## Top spatial/ecoregion enrichment tests",
        "",
        "```text",
        top_groups_txt,
        "```",
        "",
        "## Top candidate rules",
        "",
        "```text",
        top_rules_txt,
        "```",
        "",
        "## Decision-tree rules",
        "",
        "```text",
        top_tree_txt,
        "```",
        "",
        "## Manuscript-safe conclusion",
        "",
        "The tower results support a tower-observed ecosystem flux phenotype, but the final grassland-specific thesis depends on the size and quality of the repaired grassland/savanna/open-tower subset. The next required step is tower-centered satellite extraction for the selected target files, followed by direct tower-vs-satellite response-class comparison.",
        "",
    ]

    recommendation = "\n".join(recommendation_lines)
    save_text(recommendation, TXT / "PHASE18_GRASSLAND_SPATIAL_TRAIT_LOCK_VERDICT.md")
    save_text(recommendation, OUT / "README_phase18_grassland_spatial_trait_lock.md")

    methods_lines = [
        "# Methods: Phase 18 grassland spatial trait lock",
        "",
        "Phase 18 repaired tower land-cover labels, annotated tower sites with spatial/ecoregion information when local GIS layers were available, and separated tower validation into strict grassland, expanded grassland/savanna/open, open nonforest, forest-contrast, and all-site scopes.",
        "",
        "Tower saturation/breakdown was defined as tower response class equal to saturation or breakdown. Enhancement was used as the main contrast class. Characteristic tests compared saturation/breakdown sites against enhancement sites using Mann-Whitney tests for continuous variables and Fisher exact tests for categorical groups.",
        "",
        "Nearest satellite-point traits and environmental variables are treated only as provisional proxies. They do not replace tower-centered satellite extraction.",
        "",
    ]
    save_text("\n".join(methods_lines), TXT / "METHODS_PHASE18_GRASSLAND_SPATIAL_TRAIT_LOCK.md")

    print("")
    print("==============================")
    print("PHASE 18 GRASSLAND SPATIAL TRAIT LOCK VERDICT")
    print("==============================")
    print(json.dumps(verdict, indent=2, default=str))
    print("")
    print(recommendation)
    print("")
    print("MAIN OUTPUTS:")
    print(TAB / "Table101_tower_landcover_spatial_trait_annotation.csv")
    print(TAB / "Table102_tower_response_summary_by_validation_scope.csv")
    print(TAB / "Table105_characteristic_trait_association_tests.csv")
    print(TAB / "Table108_satellite_extraction_targets_expanded_grassland_savanna_open_coordinates_only.csv")
    print(TXT / "PHASE18_GRASSLAND_SPATIAL_TRAIT_LOCK_VERDICT.md")


if __name__ == "__main__":
    main()
'''
p.write_text(s + tail)
print("PATCHED", p)
