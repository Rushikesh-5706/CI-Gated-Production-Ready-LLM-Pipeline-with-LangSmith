"""Entry point: python -m evaluation.run_regression"""

import argparse
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

from langsmith import Client, evaluate, get_current_run_tree

from app import config, registry
from app.llm import set_usage_callback
from app.pipeline import execute
from evaluation import config as eval_config
from evaluation.evaluators import (
    entity_match_f1,
    json_schema_validity,
    summary_length_bounds,
)
from evaluation.gate import evaluate_gate
from evaluation.judge import llm_as_judge_quality
from evaluation.pricing import estimate_total
from evaluation.report import build_report, load_baseline, save_baseline, write_results
from evaluation.usage import start_accumulator, stop_accumulator

log = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get("GITHUB_SHA", "unknown")[:8]


def _make_target(prompt_name: str, prompt_version: str, environment: str):
    def target(inputs: dict) -> dict:
        input_text = inputs.get("input_text", "")
        rt = get_current_run_tree()
        if rt:
            rt.add_metadata(
                {
                    "prompt_name": prompt_name,
                    "prompt_version": prompt_version,
                    "environment": environment,
                }
            )
            rt.add_tags([prompt_name, prompt_version, environment])

        result = execute(prompt_name, prompt_version, input_text)
        return {"output": result.output}

    return target


def _find_latest_version(prompt_name: str) -> str:
    v = registry.latest_version(prompt_name)
    if v is None:
        print(f"no versions found for prompt '{prompt_name}'", file=sys.stderr)
        sys.exit(1)
    return v


def _get_experiment_url(ls_client: Client, results_obj) -> str:
    """Reliably extract the LangSmith experiment URL from an evaluate() result."""
    # Prefer a direct attribute.
    for attr in ("experiment_url", "url"):
        url = getattr(results_obj, attr, None)
        if url:
            return url
    # Fall back to reading the project by experiment_name.
    exp_name = getattr(results_obj, "experiment_name", None)
    if exp_name:
        try:
            project = ls_client.read_project(project_name=exp_name)
            url = getattr(project, "url", None) or ""
            if url:
                return url
        except Exception as exc:
            log.debug("could not read project %r: %s", exp_name, exc)
    return ""


def _sample_examples(examples: list, size: int) -> list:
    if size <= 0 or size >= len(examples):
        return examples
    rng = random.Random(42)
    return rng.sample(examples, size)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-version", default=None)
    parser.add_argument("--extract-version", default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=config.EVAL_SAMPLE_SIZE,
        help="number of examples to sample per dataset; 0 means all",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    registry.load_all()

    sum_version = args.summarize_version or _find_latest_version("summarize_text")
    ext_version = args.extract_version or _find_latest_version("extract_entities")

    client = Client()

    datasets = {
        eval_config.SUMMARIZATION_DATASET: None,
        eval_config.EXTRACTION_DATASET: None,
    }
    for ds_name in list(datasets):
        try:
            ds = client.read_dataset(dataset_name=ds_name)
            datasets[ds_name] = ds
        except Exception as exc:
            print(f"dataset '{ds_name}' not found: {exc}", file=sys.stderr)
            sys.exit(1)

    sum_examples_all = list(client.list_examples(dataset_id=datasets[eval_config.SUMMARIZATION_DATASET].id))
    ext_examples_all = list(client.list_examples(dataset_id=datasets[eval_config.EXTRACTION_DATASET].id))

    for ds_name, examples in [
        (eval_config.SUMMARIZATION_DATASET, sum_examples_all),
        (eval_config.EXTRACTION_DATASET, ext_examples_all),
    ]:
        if len(examples) < 25:
            print(
                f"dataset '{ds_name}' has {len(examples)} examples, need at least 25",
                file=sys.stderr,
            )
            sys.exit(1)
        log.info("dataset '%s' has %d examples", ds_name, len(examples))

    sum_examples = _sample_examples(sum_examples_all, args.sample_size)
    ext_examples = _sample_examples(ext_examples_all, args.sample_size)

    sample_note = (
        f" (sampled {args.sample_size} of {len(sum_examples_all)})"
        if args.sample_size > 0
        else ""
    )
    log.info(
        "using %d summarization and %d extraction examples%s",
        len(sum_examples),
        len(ext_examples),
        sample_note,
    )

    git_sha = _git_sha()
    environment = config.EVAL_ENVIRONMENT

    accumulator = start_accumulator()
    set_usage_callback(accumulator.record)

    all_metrics: list[dict] = []
    all_datasets_info: list[dict] = []
    experiment_urls: dict[str, str] = {}
    failing_examples: list[dict] = []

    # -- Summarization --
    sum_prefix = f"ci-summarize_text-{sum_version}-{git_sha}"
    sum_target = _make_target("summarize_text", sum_version, environment)

    log.info("running summarization evaluation with %s", sum_version)
    sum_results = evaluate(
        sum_target,
        data=sum_examples,
        evaluators=[summary_length_bounds, llm_as_judge_quality],
        experiment_prefix=sum_prefix,
        max_concurrency=config.EVAL_MAX_CONCURRENCY,
        metadata={"git_sha": git_sha, "prompt_version": sum_version},
    )

    length_scores: list[float] = []
    judge_scores: list[float] = []
    # Track completeness: each example is complete only if both evaluators scored.
    sum_scored = 0
    sum_expected = len(sum_examples)

    for result in sum_results:
        eval_results = result.get("evaluation_results", {})
        results_list = (
            eval_results.get("results", []) if isinstance(eval_results, dict) else []
        )

        length_scored = False
        judge_scored = False
        example_input = result.get("example")
        if example_input:
            input_snippet = str(getattr(example_input, "inputs", {}).get("input_text", ""))[:60]
            example_id = str(getattr(example_input, "id", "unknown"))[:16]
        else:
            input_snippet = ""
            example_id = "unknown"

        for er in results_list:
            key = getattr(er, "key", "")
            score = getattr(er, "score", None)
            if score is None:
                continue

            if key == eval_config.EVALUATOR_LENGTH:
                length_scores.append(float(score))
                length_scored = True
                if score == 0:
                    failing_examples.append(
                        {
                            "evaluator": key,
                            "dataset": eval_config.SUMMARIZATION_DATASET,
                            "example_id": example_id,
                            "input_snippet": input_snippet,
                            "score": score,
                            "comment": getattr(er, "comment", ""),
                        }
                    )
            elif key == eval_config.EVALUATOR_JUDGE:
                judge_scores.append(float(score))
                judge_scored = True
                if score < eval_config.JUDGE_MEAN_MIN:
                    failing_examples.append(
                        {
                            "evaluator": key,
                            "dataset": eval_config.SUMMARIZATION_DATASET,
                            "example_id": example_id,
                            "input_snippet": input_snippet,
                            "score": score,
                            "comment": getattr(er, "comment", ""),
                        }
                    )

        if length_scored and judge_scored:
            sum_scored += 1

    length_pass_rate = sum(length_scores) / len(length_scores) if length_scores else 0.0
    judge_mean = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    all_metrics.append(
        {
            "prompt": "summarize_text",
            "version": sum_version,
            "metric": eval_config.EVALUATOR_LENGTH,
            "value": length_pass_rate,
            "threshold": "",
            "status": "PASS" if length_pass_rate > 0 else "WARN",
        }
    )
    all_metrics.append(
        {
            "prompt": "summarize_text",
            "version": sum_version,
            "metric": eval_config.EVALUATOR_JUDGE,
            "value": judge_mean,
            "threshold": eval_config.JUDGE_MEAN_MIN,
            "status": "PASS" if judge_mean >= eval_config.JUDGE_MEAN_MIN else "FAIL",
        }
    )

    all_datasets_info.append(
        {
            "name": eval_config.SUMMARIZATION_DATASET,
            "expected": sum_expected,
            "scored": sum_scored,
        }
    )
    experiment_urls["summarize_text"] = _get_experiment_url(client, sum_results)

    # -- Extraction --
    ext_prefix = f"ci-extract_entities-{ext_version}-{git_sha}"
    ext_target = _make_target("extract_entities", ext_version, environment)

    log.info("running extraction evaluation with %s", ext_version)
    ext_results = evaluate(
        ext_target,
        data=ext_examples,
        evaluators=[json_schema_validity, entity_match_f1],
        experiment_prefix=ext_prefix,
        max_concurrency=config.EVAL_MAX_CONCURRENCY,
        metadata={"git_sha": git_sha, "prompt_version": ext_version},
    )

    validity_scores: list[float] = []
    f1_scores: list[float] = []
    ext_scored = 0
    ext_expected = len(ext_examples)

    for result in ext_results:
        eval_results = result.get("evaluation_results", {})
        results_list = (
            eval_results.get("results", []) if isinstance(eval_results, dict) else []
        )

        validity_scored = False
        f1_scored = False
        example_input = result.get("example")
        if example_input:
            input_snippet = str(getattr(example_input, "inputs", {}).get("input_text", ""))[:60]
            example_id = str(getattr(example_input, "id", "unknown"))[:16]
        else:
            input_snippet = ""
            example_id = "unknown"

        for er in results_list:
            key = getattr(er, "key", "")
            score = getattr(er, "score", None)
            if score is None:
                continue

            if key == eval_config.EVALUATOR_JSON_SCHEMA:
                validity_scores.append(float(score))
                validity_scored = True
                if score == 0:
                    failing_examples.append(
                        {
                            "evaluator": key,
                            "dataset": eval_config.EXTRACTION_DATASET,
                            "example_id": example_id,
                            "input_snippet": input_snippet,
                            "score": score,
                            "comment": getattr(er, "comment", ""),
                        }
                    )
            elif key == "entity_match_f1":
                f1_scores.append(float(score))
                f1_scored = True

        if validity_scored and f1_scored:
            ext_scored += 1

    validity_rate = (
        sum(validity_scores) / len(validity_scores) if validity_scores else 0.0
    )
    f1_mean = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    all_metrics.append(
        {
            "prompt": "extract_entities",
            "version": ext_version,
            "metric": eval_config.EVALUATOR_JSON_SCHEMA,
            "value": validity_rate,
            "threshold": eval_config.JSON_VALIDITY_MIN,
            "status": "PASS" if validity_rate >= eval_config.JSON_VALIDITY_MIN else "FAIL",
        }
    )
    all_metrics.append(
        {
            "prompt": "extract_entities",
            "version": ext_version,
            "metric": "entity_match_f1",
            "value": f1_mean,
            "threshold": "informational",
            "status": "INFO",
        }
    )

    all_datasets_info.append(
        {
            "name": eval_config.EXTRACTION_DATASET,
            "expected": ext_expected,
            "scored": ext_scored,
        }
    )
    experiment_urls["extract_entities"] = _get_experiment_url(client, ext_results)

    # -- Gate --
    total_scored = sum_scored + ext_scored
    total_expected = sum_expected + ext_expected

    gate = evaluate_gate(
        json_validity=validity_rate if validity_scores else None,
        judge_mean=judge_mean if judge_scores else None,
        scored=total_scored,
        total=total_expected,
    )

    # -- Finalize --
    acc = stop_accumulator()
    set_usage_callback(None)
    usage_by_model = acc.totals() if acc else {}
    costs = estimate_total(usage_by_model)

    baseline_path = Path(__file__).resolve().parent / "baseline.json"
    baseline = load_baseline(baseline_path)

    results_dict = {
        "gate_pass": gate.passed,
        "failed_reasons": gate.reasons,
        "metrics": all_metrics,
        "datasets": all_datasets_info,
        "sample_size": args.sample_size,
    }

    report = build_report(
        results=results_dict,
        baseline=baseline,
        costs=costs,
        usage_by_model=usage_by_model,
        git_sha=git_sha,
        experiment_urls=experiment_urls,
        failing_examples=failing_examples,
    )

    output_dir = Path(args.output_dir)
    write_results(output_dir, results_dict, report)

    print(report)

    if args.update_baseline:
        if not gate.passed:
            print("gate failed; refusing to update baseline", file=sys.stderr)
            sys.exit(1)
        save_baseline(baseline_path, results_dict)
        print(f"baseline updated at {baseline_path}")

    if not gate.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
