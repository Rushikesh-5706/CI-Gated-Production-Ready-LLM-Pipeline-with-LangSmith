"""Static and live audit of the task contract requirements."""

import json
import sys
from pathlib import Path

import yaml
from langsmith import Client

ROOT = Path(__file__).resolve().parent.parent
PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    status = "OK" if ok else "FAIL"
    if not ok:
        FAIL += 1
    else:
        PASS += 1
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def main():
    print("=== Contract Verification ===\n")

    # Required files
    print("[Files and directories]")
    required_files = [
        ".gitignore",
        ".dockerignore",
        ".env.example",
        "Dockerfile",
        "docker-compose.yml",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "app/__init__.py",
        "app/config.py",
        "app/errors.py",
        "app/schemas.py",
        "app/registry.py",
        "app/limiter.py",
        "app/llm.py",
        "app/parsing.py",
        "app/pipeline.py",
        "app/main.py",
        "prompts/summarize_text/v1.json",
        "prompts/summarize_text/v2.json",
        "prompts/extract_entities/v1.json",
        "prompts/extract_entities/v2.json",
        "data/summarization_regression.jsonl",
        "data/extraction_regression.jsonl",
        "evaluation/__init__.py",
        "evaluation/config.py",
        "evaluation/evaluators.py",
        "evaluation/judge.py",
        "evaluation/usage.py",
        "evaluation/pricing.py",
        "evaluation/run_regression.py",
        "evaluation/report.py",
        "evaluation/pr_comment.py",
        "evaluation/schemas/entities.schema.json",
        "scripts/seed_datasets.py",
        "scripts/verify_contract.py",
        "scripts/verify_traces.py",
        "scripts/smoke_test.sh",
        "tests/test_registry.py",
        "tests/test_parsing.py",
        "tests/test_evaluators.py",
        "tests/test_judge_parsing.py",
        "tests/test_api.py",
        "tests/test_report.py",
        "tests/test_gate.py",
        ".github/workflows/llm-regression.yml",
    ]
    for f in required_files:
        check(f, (ROOT / f).exists())

    # Prompt file validation
    print("\n[Prompt files]")
    for prompt_name in ["summarize_text", "extract_entities"]:
        for ver in ["v1", "v2"]:
            p = ROOT / "prompts" / prompt_name / f"{ver}.json"
            if not p.exists():
                check(f"{prompt_name}/{ver}.json", False, "missing")
                continue
            try:
                data = json.loads(p.read_text())
                has_template = (
                    isinstance(data.get("template"), str) and "{input_text}" in data["template"]
                )
                has_changelog = (
                    isinstance(data.get("changelog"), str) and len(data["changelog"]) > 0
                )
                has_temp = isinstance(
                    data.get("model_parameters", {}).get("temperature"), (int, float)
                )
                check(f"{prompt_name}/{ver}.json", has_template and has_changelog and has_temp)
            except json.JSONDecodeError:
                check(f"{prompt_name}/{ver}.json", False, "invalid JSON")

    # .env.example variables
    print("\n[.env.example variables]")
    env_example = ROOT / ".env.example"
    if env_example.exists():
        content = env_example.read_text()
        required_vars = [
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_PROJECT",
            "GROQ_API_KEY",
            "PORT",
        ]
        for var in required_vars:
            check(f"{var} in .env.example", var in content)
    else:
        check(".env.example exists", False)

    # Workflow YAML
    print("\n[Workflow]")
    wf_path = ROOT / ".github" / "workflows" / "llm-regression.yml"
    if wf_path.exists():
        try:
            wf = yaml.safe_load(wf_path.read_text())
            triggers = wf.get("on", {})
            has_pr = "pull_request" in triggers
            pr_config = triggers.get("pull_request", {})
            targets_main = "main" in pr_config.get("branches", [])
            check("triggers on pull_request", has_pr)
            check("targets main branch", targets_main)
        except yaml.YAMLError as exc:
            check("workflow YAML valid", False, str(exc))
    else:
        check("workflow file exists", False)

    # Docker compose healthcheck
    print("\n[Docker compose]")
    compose_path = ROOT / "docker-compose.yml"
    if compose_path.exists():
        try:
            dc = yaml.safe_load(compose_path.read_text())
            services = dc.get("services", {})
            for svc_name, svc in services.items():
                has_hc = "healthcheck" in svc
                check(f"service '{svc_name}' has healthcheck", has_hc)
        except yaml.YAMLError:
            check("compose YAML valid", False)

    # LangSmith datasets (live)
    print("\n[LangSmith datasets]")
    try:
        client = Client()
        for ds_name in ["summarization-regression", "extraction-regression"]:
            try:
                ds = client.read_dataset(dataset_name=ds_name)
                examples = list(client.list_examples(dataset_id=ds.id))
                count = len(examples)
                check(f"'{ds_name}' exists with >= 25 examples", count >= 25, f"count={count}")
            except Exception as exc:
                check(f"'{ds_name}' exists", False, str(exc))
    except Exception as exc:
        check("LangSmith client", False, str(exc))

    # Evaluation results
    print("\n[Evaluation results]")
    results_path = ROOT / "results" / "results.json"
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text())
            metrics = results.get("metrics", [])
            metric_keys = {m["metric"] for m in metrics}
            for key in ["json_schema_validity", "summary_length_bounds", "llm_as_judge_quality"]:
                check(f"evaluator '{key}' in results", key in metric_keys)
        except json.JSONDecodeError:
            check("results.json valid", False)
    else:
        check("results/results.json exists", False, "run evaluation first")

    print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
