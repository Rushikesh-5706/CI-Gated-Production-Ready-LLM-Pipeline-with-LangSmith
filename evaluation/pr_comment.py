"""Post or update a pull-request comment with the regression report.

Usage:
    python -m evaluation.pr_comment --report results/report.md
"""

import argparse
import logging
import os
import sys

import requests

from evaluation.report import MARKER

log = logging.getLogger(__name__)


def _get_pr_number() -> int | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    try:
        import json

        with open(event_path) as f:
            event = json.load(f)
        return event.get("pull_request", {}).get("number")
    except (OSError, ValueError, KeyError):
        return None


def _list_all_comments(api: str, headers: dict) -> list[dict]:
    """Page through all PR comments; GitHub returns 30 per page by default."""
    comments: list[dict] = []
    url = f"{api}?per_page=100&page=1"
    page = 1
    while url:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        comments.extend(batch)
        page += 1
        link_header = resp.headers.get("Link", "")
        if 'rel="next"' in link_header:
            # Parse next URL from Link header.
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
                    break
            else:
                break
        else:
            break
    return comments


def upsert_comment(report: str) -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr_number = _get_pr_number()

    if not pr_number:
        pr_ref = os.environ.get("PR_NUMBER", "")
        if pr_ref:
            try:
                pr_number = int(pr_ref)
            except ValueError:
                pass

    if not token or not repo or not pr_number:
        log.warning(
            "missing GITHUB_TOKEN, GITHUB_REPOSITORY, or PR number; skipping comment"
        )
        _write_step_summary(report)
        return

    api = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    try:
        comments = _list_all_comments(api, headers)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Post or update a PR comment with the regression report."
    )
    parser.add_argument(
        "--report",
        required=True,
        help="path to the report markdown file",
    )
    args = parser.parse_args()

    from pathlib import Path

    report_path = Path(args.report)
    if not report_path.exists():
        run_url = ""
        gh_server = os.environ.get("GITHUB_SERVER_URL", "")
        gh_repo = os.environ.get("GITHUB_REPOSITORY", "")
        gh_run = os.environ.get("GITHUB_RUN_ID", "")
        if gh_server and gh_repo and gh_run:
            run_url = f"{gh_server}/{gh_repo}/actions/runs/{gh_run}"
        notice = (
            f"{MARKER}\n\nStatus: FAILED\n\n"
            f"  - report file not found: {args.report}\n"
        )
        if run_url:
            notice += f"\nSee the run log for details: {run_url}\n"
        upsert_comment(notice)
        sys.exit(1)

    report = report_path.read_text()
    upsert_comment(report)


if __name__ == "__main__":
    main()
