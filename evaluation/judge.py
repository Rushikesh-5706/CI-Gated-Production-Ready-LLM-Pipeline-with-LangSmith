import json
import logging

from langsmith.schemas import Example, Run

from app import config
from app.llm import chat_completion
from evaluation.config import EVALUATOR_JUDGE

log = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are an expert quality assessor evaluating AI-generated summaries. "
    "Respond only with a single JSON object."
)

JUDGE_PROMPT = (
    "Evaluate the summary below against the original text.\n\n"
    "Original text:\n{original_text}\n\n"
    "Generated summary:\n{generated_summary}\n\n"
    "Return a JSON object with exactly these keys:\n"
    '  "relevance_score": integer 1-5 (how well the summary captures the main points)\n'
    '  "conciseness_score": integer 1-5 (how concise and free of padding the summary is)\n'
    '  "justification": string (one sentence explaining the scores)\n'
    "Do not include any text outside the JSON object."
)


def _call_judge(original_text: str, summary: str) -> dict | None:
    prompt = JUDGE_PROMPT.replace("{original_text}", original_text).replace(
        "{generated_summary}", summary
    )

    result = chat_completion(
        prompt=prompt,
        system_prompt=JUDGE_SYSTEM,
        model=config.JUDGE_MODEL,
        temperature=0,
        max_tokens=config.JUDGE_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    llm_result = result["result"]
    text = llm_result.text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    rel = parsed.get("relevance_score")
    con = parsed.get("conciseness_score")

    if not isinstance(rel, (int, float)) or not isinstance(con, (int, float)):
        return None
    if not (1 <= rel <= 5) or not (1 <= con <= 5):
        return None

    return {
        "relevance_score": rel,
        "conciseness_score": con,
        "justification": parsed.get("justification", ""),
    }


def llm_as_judge_quality(run: Run, example: Example) -> dict:
    output = run.outputs.get("output") if run.outputs else None

    if output is None or not isinstance(output, str):
        return {
            "key": EVALUATOR_JUDGE,
            "score": 0,
            "comment": "no string output to judge",
        }

    input_text = ""
    if example and example.inputs:
        input_text = example.inputs.get("input_text", "")

    if not input_text:
        return {
            "key": EVALUATOR_JUDGE,
            "score": 0,
            "comment": "no input_text in example to compare against",
        }

    for _attempt in range(2):
        try:
            result = _call_judge(input_text, output)
            if result is not None:
                score = (result["relevance_score"] + result["conciseness_score"]) / 2.0
                comment = (
                    f"relevance={result['relevance_score']} "
                    f"conciseness={result['conciseness_score']} "
                    f"justification={result['justification']}"
                )
                return {"key": EVALUATOR_JUDGE, "score": score, "comment": comment}
        except Exception as exc:
            log.warning("judge evaluator failed on attempt %d: %s", _attempt + 1, exc)

    return {
        "key": EVALUATOR_JUDGE,
        "score": 0,
        "comment": "judge failed to produce valid scores after 2 attempts",
    }
