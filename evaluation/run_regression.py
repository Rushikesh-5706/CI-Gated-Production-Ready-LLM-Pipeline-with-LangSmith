"""Entry point: python -m evaluation.run_regression"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from langsmith import Client, evaluate, get_current_run_tree

from app import config, registry
from app.llm import set_usage_callback
from app.pipeline import execute
from evaluation import config as eval_config
from evaluation.evaluators import json_schema_validity, summary_length_bounds
from evaluation.judge import llm_as_judge_quality
from evaluation.pricing import estimate_total
from evaluation.report import build_report, load_baseline, save_baseline, write_results
from evaluation.usage import start_accumulator, stop_accumulator

log = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
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

        result = execute(prompt_name, prompt_version, input_text, environment)
        return {"output": result.output}

    return target


def _find_latest_version(prompt_name: str) -> str:
    v = registry.latest_version(prompt_name)
    if v is None:
        print(f"no versions found for prompt '{prompt_name}'", file=sys.stderr)
        sys.exit(1)
    return v


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-version", default=None)
    parser.add_argument("--extract-version", default=None)
    parser.add_argument("--sample-size", type=int, default=config.EVAL_SAMPLE_SIZE)
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
    for ds_name in datasets:
        try:
            ds = client.read_dataset(dataset_name=ds_name)
            datasets[ds_name] = ds
        except Exception as exc:
            print(f"dataset '{ds_name}' not found: {exc}", file=sys.stderr)
            sys.exit(1)

    for ds_name, ds in datasets.items():
        examples = list(client.list_examples(dataset_id=ds.id))
        count = len(examples)
        if count < 25:
            print(f"dataset '{ds_name}' has {count} examples, need at least 25", file=sys.stderr)
            sys.exit(1)
        log.info("dataset '%s' has %d examples", ds_name, count)

    git_sha = _git_sha()
    environment = config.EVAL_ENVIRONMENT

    accumulator = start_accumulator()
    set_usage_callback(accumulator.record)

    all_metrics = []
    all_datasets_info = []
    experiment_urls = {}
    failing_examples = []
    gate_pass = True
    failed_reasons = []

    # -- Summarization --
    sum_prefix = f"ci-summarize_text-{sum_version}-{git_sha}"
    sum_target = _make_target("summarize_text", sum_version, environment)

    log.info("running summarization evaluation with %s", sum_version)
    sum_results = evaluate(
        sum_target,
        data=eval_config.SUMMARIZATION_DATASET,
        evaluators=[summary_length_bounds, llm_as_judge_quality],
        experiment_prefix=sum_prefix,
        max_concurrency=config.EVAL_MAX_CONCURRENCY,
        metadata={"git_sha": git_sha, "prompt_version": sum_version},
    )

    length_scores = []
    judge_scores = []
    sum_scored = 0
    sum_expected = 0

    for result in sum_results:
        sum_expected += 1
        eval_results = result.get("evaluation_results", {})
        results_list = eval_results.get("results", []) if isinstance(eval_results, dict) else []

        for er in results_list:
            key = getattr(er, "key", "")
            score = getattr(er, "score", None)
            if score is None:
                continue

            if key == eval_config.EVALUATOR_LENGTH:
                length_scores.append(score)
            elif key == eval_config.EVALUATOR_JUDGE:
                judge_scores.append(score)
                if score == 0:
                    failing_examples.append(
                        {
                            "evaluator": key,
                            "dataset": eval_config.SUMMARIZATION_DATASET,
                            "comment": getattr(er, "comment", ""),
                        }
                    )

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

    if judge_mean < eval_config.JUDGE_MEAN_MIN:
        gate_pass = False
        failed_reasons.append(
            f"summarize_text {eval_config.EVALUATOR_JUDGE} mean {judge_mean:.3f} < {eval_config.JUDGE_MEAN_MIN}"
        )

    all_datasets_info.append(
        {
            "name": eval_config.SUMMARIZATION_DATASET,
            "expected": sum_expected,
            "scored": sum_scored,
        }
    )

    try:
        exp_url = sum_results.experiment_url if hasattr(sum_results, "experiment_url") else ""
        experiment_urls["summarize_text"] = exp_url or ""
    except Exception:
        experiment_urls["summarize_text"] = ""

    # -- Extraction --
    ext_prefix = f"ci-extract_entities-{ext_version}-{git_sha}"
    ext_target = _make_target("extract_entities", ext_version, environment)

    log.info("running extraction evaluation with %s", ext_version)
    ext_results = evaluate(
        ext_target,
        data=eval_config.EXTRACTION_DATASET,
        evaluators=[json_schema_validity],
        experiment_prefix=ext_prefix,
        max_concurrency=config.EVAL_MAX_CONCURRENCY,
        metadata={"git_sha": git_sha, "prompt_version": ext_version},
    )

    validity_scores = []
    ext_scored = 0
    ext_expected = 0

    for result in ext_results:
        ext_expected += 1
        eval_results = result.get("evaluation_results", {})
        results_list = eval_results.get("results", []) if isinstance(eval_results, dict) else []

        for er in results_list:
            key = getattr(er, "key", "")
            score = getattr(er, "score", None)
            if score is None:
                continue

            if key == eval_config.EVALUATOR_JSON_SCHEMA:
                validity_scores.append(score)
                if score == 0:
                    failing_examples.append(
                        {
                            "evaluator": key,
                            "dataset": eval_config.EXTRACTION_DATASET,
                            "comment": getattr(er, "comment", ""),
                        }
                    )

        ext_scored += 1

    validity_rate = sum(validity_scores) / len(validity_scores) if validity_scores else 0.0

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

    if validity_rate < eval_config.JSON_VALIDITY_MIN:
        gate_pass = False
        failed_reasons.append(
            f"extract_entities {eval_config.EVALUATOR_JSON_SCHEMA} rate {validity_rate:.3f} < {eval_config.JSON_VALIDITY_MIN}"
        )

    all_datasets_info.append(
        {
            "name": eval_config.EXTRACTION_DATASET,
            "expected": ext_expected,
            "scored": ext_scored,
        }
    )

    try:
        exp_url = ext_results.experiment_url if hasattr(ext_results, "experiment_url") else ""
        experiment_urls["extract_entities"] = exp_url or ""
    except Exception:
        experiment_urls["extract_entities"] = ""

    # -- finalize --
    acc = stop_accumulator()
    set_usage_callback(None)
    usage_by_model = acc.totals() if acc else {}
    costs = estimate_total(usage_by_model)

    baseline_path = Path(__file__).resolve().parent / "baseline.json"
    baseline = load_baseline(baseline_path)

    results_dict = {
        "gate_pass": gate_pass,
        "failed_reasons": failed_reasons,
        "metrics": all_metrics,
        "datasets": all_datasets_info,
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
        if not gate_pass:
            print("gate failed; refusing to update baseline", file=sys.stderr)
            sys.exit(1)
        save_baseline(baseline_path, results_dict)
        print(f"baseline updated at {baseline_path}")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
