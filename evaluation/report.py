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

    sample_size = results.get("sample_size", 0)
    if sample_size and sample_size > 0:
        lines.append(f"  (sampled {sample_size} examples per dataset)")

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

        threshold_str = str(threshold) if threshold else ""
        status = entry.get("status", "")
        lines.append(
            f"| {prompt} | {version} | {metric} | {value:.3f} | "
            f"{threshold_str} | {baseline_val} | {delta} | {status} |"
        )

    lines.append("")
    lines.append("## Example Counts")
    lines.append("")
    lines.append("| Dataset | Expected | Scored | Complete |")
    lines.append("|---------|----------|--------|----------|")
    for ds in results.get("datasets", []):
        exp = ds["expected"]
        scored = ds["scored"]
        pct = f"{100 * scored / exp:.1f}%" if exp > 0 else "n/a"
        lines.append(f"| {ds['name']} | {exp} | {scored} | {pct} |")

    lines.append("")
    lines.append("## Token Usage and Cost Estimate")
    lines.append("")
    lines.append(
        "| Model | Prompt Tokens | Completion Tokens | Total Tokens | Est. Cost (USD) |"
    )
    lines.append(
        "|-------|---------------|-------------------|--------------|-----------------|"
    )
    total_cost = costs.get("total", 0.0)
    for model, counts in usage_by_model.items():
        cost = costs.get(model, 0.0)
        # Mark as unpriced when no pricing entry exists.
        cost_str = "unpriced" if costs.get(f"{model}_unpriced") is not None else f"${cost:.6f}"
        lines.append(
            f"| {model} | {counts['prompt_tokens']} | {counts['completion_tokens']} | "
            f"{counts['total_tokens']} | {cost_str} |"
        )
    lines.append(f"| **Total** | | | | **${total_cost:.6f}** |")
    lines.append("")
    lines.append(
        "Rates sourced from https://groq.com/pricing as of 2026-10-03. "
        "The Groq free tier incurs no charge."
    )

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
        if url:
            lines.append(f"- Experiment ({name}): {url}")

    if failing_examples:
        lines.append("")
        lines.append("## Failing Examples (up to 10)")
        lines.append("")
        lines.append("| # | Evaluator | Dataset | Example ID | Input Snippet | Score | Comment |")
        lines.append("|---|-----------|---------|------------|---------------|-------|---------|")
        sorted_failures = sorted(failing_examples, key=lambda x: x.get("score", 0))
        for i, fe in enumerate(sorted_failures[:10], 1):
            snippet = str(fe.get("input_snippet", ""))[:60].replace("|", "\\|")
            comment = str(fe.get("comment", ""))[:80].replace("|", "\\|")
            ex_id = str(fe.get("example_id", ""))[:16]
            score = fe.get("score", "")
            lines.append(
                f"| {i} | {fe['evaluator']} | {fe['dataset']} | "
                f"{ex_id} | {snippet} | {score} | {comment} |"
            )

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
        if entry.get("status") == "INFO":
            continue
        bkey = f"{entry['prompt']}_{entry['metric']}"
        baseline[bkey] = entry["value"]
    path.write_text(json.dumps(baseline, indent=2) + "\n")
