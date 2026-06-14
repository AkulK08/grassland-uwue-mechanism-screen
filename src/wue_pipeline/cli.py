"""Command-line interface for the WUE pipeline."""

from __future__ import annotations
import click

from .config import load_config
from .utils import setup_logging
from .io.demo_data import make_demo_data
from .workflows.preprocess import run_preprocess
from .workflows.gate1 import run_gate1
from .workflows.gate2 import run_gate2
from .workflows.gate3 import run_gate3
from .workflows.phase4 import run_phase4
from .figures.main_figures import generate_all_figures
from .reporting.manuscript import write_manuscript_files
from .remote.gee_extract import submit_points_export, submit_year_exports, list_tasks
from .workflows.point_tables import prepare_point_table, fit_point_matrix, summarize_point_matrix


def _cfg(config):
    c = load_config(config)
    setup_logging(c.file("logs", "pipeline.log"))
    return c


@click.group()
def main():
    """Grassland WUE Nature Pipeline CLI."""


@main.command("make-demo")
@click.option("--config", default="configs/demo.yaml", show_default=True)
def make_demo(config):
    cfg = _cfg(config)
    make_demo_data(cfg)
    click.echo("Demo data written.")


@main.command("preprocess")
@click.option("--config", default="configs/demo.yaml", show_default=True)
def preprocess(config):
    run_preprocess(_cfg(config))


@main.command("gate1")
@click.option("--config", default="configs/demo.yaml", show_default=True)
def gate1(config):
    run_gate1(_cfg(config))


@main.command("gate2")
@click.option("--config", default="configs/demo.yaml", show_default=True)
def gate2(config):
    run_gate2(_cfg(config))


@main.command("gate3")
@click.option("--config", default="configs/demo.yaml", show_default=True)
def gate3(config):
    run_gate3(_cfg(config))


@main.command("phase4")
@click.option("--config", default="configs/demo.yaml", show_default=True)
def phase4(config):
    run_phase4(_cfg(config))


@main.command("figures")
@click.option("--config", default="configs/demo.yaml", show_default=True)
def figures(config):
    generate_all_figures(_cfg(config))


@main.command("manuscript")
@click.option("--config", default="configs/demo.yaml", show_default=True)
def manuscript(config):
    write_manuscript_files(_cfg(config))


@main.command("run-all")
@click.option("--config", default="configs/demo.yaml", show_default=True)
@click.option("--make-demo/--no-make-demo", default=False, show_default=True)
def run_all(config, make_demo):
    cfg = _cfg(config)
    if make_demo or cfg.mode == "demo":
        make_demo_data(cfg)
    run_preprocess(cfg)
    run_gate1(cfg)
    run_gate2(cfg)
    run_gate3(cfg)
    run_phase4(cfg)
    generate_all_figures(cfg)
    write_manuscript_files(cfg)
    click.echo("Pipeline completed.")


@main.group("remote")
def remote():
    """Command-line-only remote extraction agents."""


@remote.command("gee-submit-points")
@click.option("--project", required=True, help="Google Cloud project enabled for Earth Engine.")
@click.option("--bucket", required=True, help="Google Cloud Storage bucket name, without gs://.")
@click.option("--prefix", default="wue_remote", show_default=True, help="GCS prefix/folder.")
@click.option("--start-year", default=2001, show_default=True)
@click.option("--end-year", default=2024, show_default=True)
@click.option("--n-points", default=50000, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--include-savanna/--grassland-only", default=False, show_default=True)
def gee_submit_points(project, bucket, prefix, start_year, end_year, n_points, seed, include_savanna):
    """Submit stable grassland point export to GCS."""
    info = submit_points_export(project, bucket, prefix, start_year, end_year, n_points, seed, include_savanna)
    click.echo(f"Submitted {info.description}: task={info.task_id}, state={info.state}")


@remote.command("gee-submit-years")
@click.option("--project", required=True, help="Google Cloud project enabled for Earth Engine.")
@click.option("--bucket", required=True, help="Google Cloud Storage bucket name, without gs://.")
@click.option("--prefix", default="wue_remote", show_default=True, help="GCS prefix/folder.")
@click.option("--points-asset", required=True, help="Earth Engine asset id containing stable points.")
@click.option("--start-year", default=2001, show_default=True)
@click.option("--end-year", default=2024, show_default=True)
@click.option("--scale", default=10000, show_default=True, help="Sampling scale in meters.")
def gee_submit_years(project, bucket, prefix, points_asset, start_year, end_year, scale):
    """Submit annual GEE point-time extraction tasks to GCS."""
    infos = submit_year_exports(project, bucket, prefix, points_asset, start_year, end_year, scale=scale)
    for info in infos:
        click.echo(f"Submitted {info.description}: task={info.task_id}, state={info.state}")


@remote.command("gee-tasks")
@click.option("--project", required=True)
def gee_tasks(project):
    """Print Earth Engine batch task status."""
    for st in list_tasks(project):
        click.echo(f"{st.get('description')}\t{st.get('state')}\t{st.get('id')}")


@main.group("points")
def points():
    """Analyze compact point-time CSVs exported by remote agents."""


@points.command("prepare")
@click.option("--input-glob", required=True, help="CSV glob, e.g. 'data/raw/gee/*.csv'.")
@click.option("--output", default="data/processed/point_timeseries_prepared.csv", show_default=True)
@click.option("--gpp-products", default="MODIS,PML", show_default=True)
@click.option("--et-products", default="MODIS,PML", show_default=True)
def points_prepare(input_glob, output, gpp_products, et_products):
    out = prepare_point_table(input_glob, output, gpp_products.split(','), et_products.split(','))
    click.echo(f"Prepared point table: {out}")


@points.command("fit-matrix")
@click.option("--prepared", default="data/processed/point_timeseries_prepared.csv", show_default=True)
@click.option("--output", default="results/tables/point_gate2_pixel_results.csv", show_default=True)
@click.option("--gpp-products", default="MODIS,PML", show_default=True)
@click.option("--et-products", default="MODIS,PML", show_default=True)
@click.option("--min-obs", default=50, show_default=True)
@click.option("--n-boot", default=1000, show_default=True)
@click.option("--seed", default=42, show_default=True)
def points_fit_matrix(prepared, output, gpp_products, et_products, min_obs, n_boot, seed):
    out = fit_point_matrix(prepared, output, gpp_products.split(','), et_products.split(','), min_obs=min_obs, n_boot=n_boot, seed=seed)
    click.echo(f"Point matrix fit results: {out}")


@points.command("summarize")
@click.option("--results", default="results/tables/point_gate2_pixel_results.csv", show_default=True)
@click.option("--output", default="results/tables/point_gate2_robustness_matrix.csv", show_default=True)
def points_summarize(results, output):
    out = summarize_point_matrix(results, output)
    click.echo(f"Point robustness summary: {out}")


@points.command("run-all")
@click.option("--input-glob", required=True, help="CSV glob from GEE/GCS downloads.")
@click.option("--gpp-products", default="MODIS,PML", show_default=True)
@click.option("--et-products", default="MODIS,PML", show_default=True)
@click.option("--min-obs", default=50, show_default=True)
@click.option("--n-boot", default=1000, show_default=True)
def points_run_all(input_glob, gpp_products, et_products, min_obs, n_boot):
    prep = prepare_point_table(input_glob, "data/processed/point_timeseries_prepared.csv", gpp_products.split(','), et_products.split(','))
    res = fit_point_matrix(str(prep), "results/tables/point_gate2_pixel_results.csv", gpp_products.split(','), et_products.split(','), min_obs=min_obs, n_boot=n_boot)
    summ = summarize_point_matrix(str(res), "results/tables/point_gate2_robustness_matrix.csv")
    click.echo(f"Prepared: {prep}\nFits: {res}\nSummary: {summ}")


if __name__ == "__main__":
    main()
