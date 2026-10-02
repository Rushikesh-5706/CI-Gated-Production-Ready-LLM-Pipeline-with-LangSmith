import json
import logging
import os

import requests

from evaluation.report import MARKER

log = logging.getLogger(__name__)


def _get_pr_number() -> int | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    try:
        with open(event_path) as f:
            event = json.load(f)
        return event.get("pull_request", {}).get("number")
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def upsert_comment(report: str) -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr_number = _get_pr_number()

    if not pr_number:
        pr_ref = os.environ.get("PR_NUMBER", "")
        if pr_ref:
            pr_number = int(pr_ref)

    if not token or not repo or not pr_number:
        log.warning("missing GITHUB_TOKEN, GITHUB_REPOSITORY, or PR number; skipping comment")
        _write_step_summary(report)
        return

    api = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    try:
        resp = requests.get(api, headers=headers, timeout=30)
        resp.raise_for_status()
        comments = resp.json()
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            log.warning(
                "403 reading comments (fork PR with read-only token); writing to step summary"
            )
            _write_step_summary(report)
            return
        raise

    existing_id = None
    for c in comments:
        if MARKER in c.get("body", ""):
            existing_id = c["id"]
            break

    try:
        if existing_id:
            url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_id}"
            resp = requests.patch(url, headers=headers, json={"body": report}, timeout=30)
        else:
            resp = requests.post(api, headers=headers, json={"body": report}, timeout=30)
        resp.raise_for_status()
        log.info("comment posted on PR #%d", pr_number)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            log.warning("403 posting comment; writing to step summary instead")
            _write_step_summary(report)
        else:
            raise


def _write_step_summary(report: str) -> None:
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(report + "\n")
        log.info("report appended to GITHUB_STEP_SUMMARY")
    else:
        print(report)
