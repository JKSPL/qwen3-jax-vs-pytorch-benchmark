from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "qwen3-jax-vs-pytorch-benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a W&B report for the synthetic Qwen3 benchmark group."
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY, required=DEFAULT_ENTITY is None)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--group", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def load_local_results(output_dir: Path, group: str) -> list[dict[str, Any]]:
    results = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name.startswith("latest_"):
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if payload.get("group") == group:
            results.append(payload)
    return results


def summarize_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return (
            "No local JSON summaries were found for this group. "
            "The report panels still query W&B runs."
        )

    by_framework: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_framework.setdefault(result["framework"], []).append(result)

    lines = [
        "| Framework | Runs | Tokens/sec mean | Seconds/step mean | Parameters | Devices |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for framework in sorted(by_framework):
        framework_results = by_framework[framework]
        tokens_per_second = mean(float(item["tokens_per_second"]) for item in framework_results)
        seconds_per_step = mean(float(item["seconds_per_step"]) for item in framework_results)
        parameters = framework_results[-1].get("parameter_count", "")
        devices = framework_results[-1].get("device_count", "")
        lines.append(
            f"| {framework} | {len(framework_results)} | {tokens_per_second:,.0f} | "
            f"{seconds_per_step:.4f} | {parameters:,} | {devices} |"
        )
    return "\n".join(lines)


def save_report(report) -> None:
    try:
        report.save()
        return
    except Exception as exc:
        if "relogin required" not in str(exc):
            raise

    import wandb_workspaces.reports.v2.interface as wr_interface

    model = report._to_model()
    response = wr_interface.execute_graphql(
        wr_interface._get_api(),
        wr_interface.gql.upsert_view,
        {
            "id": None if not model.id else model.id,
            "name": wr_interface.internal._generate_name() if not model.name else model.name,
            "entityName": model.project.entity_name,
            "projectName": model.project.name,
            "description": model.description,
            "displayName": model.display_name,
            "type": "runs",
            "spec": model.spec.model_dump_json(by_alias=True, exclude_none=True),
        },
    )
    report.id = response["upsertView"]["view"]["id"]


def write_local_markdown(
    output_dir: Path, title: str, group: str, summary_table: str, results: list[dict[str, Any]]
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"Group: `{group}`", "", summary_table, ""]
    if results:
        lines.extend(["## Runs", ""])
        for result in sorted(results, key=lambda item: item["framework"]):
            url = result.get("wandb_url", "")
            suffix = f" - {url}" if url else ""
            lines.append(
                f"- {result['framework']}: {float(result['tokens_per_second']):,.0f} tokens/sec, "
                f"{float(result['seconds_per_step']):.4f} seconds/step{suffix}"
            )
    path = output_dir / f"report_{group}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results = load_local_results(output_dir, args.group)
    summary_table = summarize_results(results)

    import wandb_workspaces.reports.v2 as wr

    title = args.title or f"Synthetic Qwen3 JAX vs PyTorch - {args.group}"
    runset = wr.Runset(
        entity=args.entity,
        project=args.project,
        name="Benchmark runs",
        filters=f"group == '{args.group}'",
        groupby=["config.framework"],
        visible_columns=[
            "Name",
            "Job Type",
            "Summary.tokens_per_second",
            "Summary.seconds_per_step",
            "Summary.loss",
            "Summary.parameter_count",
            "Summary.device_count",
        ],
    )
    report = wr.Report(
        project=args.project,
        entity=args.entity,
        title=title,
        description="Synthetic fixed-batch Qwen3 training throughput comparison.",
        width="fluid",
        blocks=[
            wr.H1("Synthetic Qwen3 JAX vs PyTorch"),
            wr.MarkdownBlock(
                "This report compares steady-state training throughput on fixed synthetic data. "
                "Tokens/sec counts input tokens processed during forward, backward, "
                "and optimizer steps."
            ),
            wr.H2("Local Summary"),
            wr.MarkdownBlock(summary_table),
            wr.H2("W&B Panels"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[
                    wr.BarPlot(
                        title="Tokens/sec by framework",
                        metrics=[wr.SummaryMetric("tokens_per_second")],
                        groupby=wr.Config("framework"),
                        groupby_aggfunc="mean",
                        title_x="tokens/sec",
                    ),
                    wr.BarPlot(
                        title="Seconds/step by framework",
                        metrics=[wr.SummaryMetric("seconds_per_step")],
                        groupby=wr.Config("framework"),
                        groupby_aggfunc="mean",
                        title_x="seconds/step",
                    ),
                    wr.BarPlot(
                        title="Final loss by framework",
                        metrics=[wr.SummaryMetric("loss")],
                        groupby=wr.Config("framework"),
                        groupby_aggfunc="mean",
                        title_x="loss",
                    ),
                ],
            ),
        ],
    )
    markdown_path = write_local_markdown(output_dir, title, args.group, summary_table, results)
    save_report(report)
    print(
        json.dumps(
            {
                "report_url": report.url,
                "local_markdown": str(markdown_path),
                "group": args.group,
                "project": args.project,
                "entity": args.entity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
