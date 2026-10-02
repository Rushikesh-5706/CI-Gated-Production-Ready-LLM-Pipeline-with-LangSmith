import json
import os
from pathlib import Path

MARKER = "<!-- llm-regression-report -->"


def build_report(
    results: dict,
    baseline: dict | None,
    costs: dict[str, float],
    usage_by_model: dict[str, dict[str, int]],
    git_sha: str,
    experiment_urls: dict[str, str],
    failing_examples: list[dict],
) -> str:
    lines = [MARKER, ""]

    gate_pass = results.get("gate_pass", False)
    failed_reasons = results.get("failed_reasons", [])

    if gate_pass:
        lines.append("Status: PASSED")
    else:
        lines.append("Status: FAILED")
        for reason in failed_reasons:
            lines.append(f"  - {reason}")

    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Prompt | Version | Metric | Result | Threshold | Baseline | Delta | Status |")
    lines.append("|--------|---------|--------|--------|-----------|----------|-------|--------|")

    for entry in results.get("metrics", []):
        prompt = entry["prompt"]
        version = entry["version"]
        metric = entry["metric"]
        value = entry["value"]
        threshold = entry.get("threshold", "")
        baseline_val = ""
        delta = ""

        if baseline:
            bkey = f"{prompt}_{metric}"
            bval = baseline.get(bkey)
            if bval is not None:
                baseline_val = f"{bval:.3f}"
                delta = f"{value - bval:+.3f}"

        threshold_str = f"{threshold}" if threshold else ""
        status = entry.get("status", "")
        lines.append(
            f"| {prompt} | {version} | {metric} | {value:.3f} | "
            f"{threshold_str} | {baseline_val} | {delta} | {status} |"
        )

    lines.append("")
    lines.append("## Example Counts")
    lines.append("")
    lines.append("| Dataset | Expected | Scored |")
    lines.append("|---------|----------|--------|")
    for ds in results.get("datasets", []):
        lines.append(f"| {ds['name']} | {ds['expected']} | {ds['scored']} |")

    lines.append("")
    lines.append("## Token Usage and Cost Estimate")
    lines.append("")
    lines.append("| Model | Prompt Tokens | Completion Tokens | Total Tokens | Est. Cost (USD) |")
    lines.append("|-------|---------------|-------------------|--------------|-----------------|")
    for model, counts in usage_by_model.items():
        cost = costs.get(model, 0.0)
        lines.append(
            f"| {model} | {counts['prompt_tokens']} | {counts['completion_tokens']} | "
            f"{counts['total_tokens']} | ${cost:.6f} |"
        )
    lines.append(f"| **Total** | | | | **${costs.get('total', 0.0):.6f}** |")
    lines.append("")
    lines.append("Costs are estimates at list prices. The Groq free tier incurs no charge.")

    lines.append("")
    lines.append("## Run Details")
    lines.append("")
    lines.append(f"- Commit: {git_sha}")
    workflow_url = os.environ.get("GITHUB_SERVER_URL", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if workflow_url and repo and run_id:
        lines.append(f"- Workflow run: {workflow_url}/{repo}/actions/runs/{run_id}")
    for name, url in experiment_urls.items():
        lines.append(f"- Experiment ({name}): {url}")

    if failing_examples:
        lines.append("")
        lines.append("## Failing Examples (up to 10)")
        lines.append("")
        for fe in failing_examples[:10]:
            lines.append(f"- **{fe['evaluator']}** on `{fe['dataset']}`: {fe['comment']}")

    lines.append("")
    return "\n".join(lines)


def write_results(output_dir: Path, results: dict, report: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "report.md").write_text(report)


def load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_baseline(path: Path, results: dict) -> None:
    baseline = {}
    for entry in results.get("metrics", []):
        bkey = f"{entry['prompt']}_{entry['metric']}"
        baseline[bkey] = entry["value"]
    path.write_text(json.dumps(baseline, indent=2) + "\n")
