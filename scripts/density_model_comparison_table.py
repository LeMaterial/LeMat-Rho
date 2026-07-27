"""Aggregate D7 per-arm eval outputs into a cross-arm comparison (D8).

Reads one or more parquet files produced by
``scripts/density_model_eval.py`` and writes:

* A CSV with one row per arm: ``model``, ``n_structures``,
   ``nmape_mean``, ``nmape_std``, ``nmape_median`` and the same for
   ``rmse`` / ``nrmse``.
* A GitHub-flavoured markdown table for paste-into-PR consumption.

Each input parquet may carry rows from one arm (typical) or
multiple arms; rows are grouped by the ``model`` column so it
works either way. Multiple input files for the same arm are
concatenated before aggregation, which is the right behaviour
when a sharded eval run writes per-chunk outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_METRIC_COLS = ("nmape", "rmse", "nrmse")


def aggregate_per_arm(inputs: list[str | Path]) -> pd.DataFrame:
    """Concatenate the per-row eval parquets and aggregate per arm.

    Parameters
    ----------
    inputs :
        Paths to D7-shaped per-row eval parquets.

    Returns
    -------
    pd.DataFrame with one row per arm and columns:
    ``model``, ``n_structures``, ``{nmape,rmse,nrmse}_{mean,std,median}``.
    """
    frames = [pd.read_parquet(p) for p in inputs]
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for model_name, group in df.groupby("model", sort=True):
        row = {"model": model_name, "n_structures": len(group)}
        for metric in _METRIC_COLS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_median"] = float(group[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def render_markdown_table(agg: pd.DataFrame) -> str:
    """Render the aggregated table as a GitHub-flavoured markdown table.

    Format::

        | Model | N | NMAPE (%) | RMSE (e/A^3) | NRMSE (%) |
        | --- | --- | --- | --- | --- |
        | salted | 1500 | 32.10 +/- 8.42 | 0.0120 +/- 0.0050 | 28.70 +/- 7.20 |
    """
    header = "| Model | N | NMAPE (%) | RMSE (e/A^3) | NRMSE (%) |"
    sep = "| --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for _, row in agg.iterrows():
        lines.append(
            "| {model} | {n} | {nmape:.2f} +/- {nmape_s:.2f} | "
            "{rmse:.4f} +/- {rmse_s:.4f} | {nrmse:.2f} +/- {nrmse_s:.2f} |".format(
                model=row["model"],
                n=int(row["n_structures"]),
                nmape=row["nmape_mean"],
                nmape_s=row["nmape_std"],
                rmse=row["rmse_mean"],
                rmse_s=row["rmse_std"],
                nrmse=row["nrmse_mean"],
                nrmse_s=row["nrmse_std"],
            )
        )
    return "\n".join(lines) + "\n"


def build_comparison_table(
    inputs: list[str | Path],
    csv_path: str | Path,
    markdown_path: str | Path,
) -> pd.DataFrame:
    """End-to-end: aggregate + write CSV and markdown."""
    agg = aggregate_per_arm(inputs)
    Path(csv_path).write_text(agg.to_csv(index=False))
    Path(markdown_path).write_text(render_markdown_table(agg))
    return agg


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate per-arm density eval parquets into a comparison table."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more D7-output parquets.",
    )
    parser.add_argument("--csv", required=True, type=Path, help="Output CSV path.")
    parser.add_argument(
        "--markdown", required=True, type=Path, help="Output markdown path."
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    agg = build_comparison_table(
        inputs=args.inputs, csv_path=args.csv, markdown_path=args.markdown
    )
    print(render_markdown_table(agg))


if __name__ == "__main__":
    main()
