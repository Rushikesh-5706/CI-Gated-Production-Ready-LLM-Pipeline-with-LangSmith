# CI-Gated Production-Ready LLM Pipeline with LangSmith

This project is a production-grade FastAPI application that serves LLM prompts, rigorously gated by a continuous integration pipeline using LangSmith.

## Features

- **FastAPI Endpoints**: High-performance HTTP server for invoking prompts (`/invoke/{prompt_name}`).
- **Dynamic Prompt Registry**: Prompt templates are separated from code and versioned (`prompts/`). You can deploy `v2` without modifying backend code.
- **Resilient LLM Caller**: Built-in Groq rate-limiting and backoff logic to prevent 429s.
- **Deep Observability**: Traces natively with LangSmith, adding rich metadata (`environment`, `prompt_version`, token metrics, latency).
- **Graceful Degradation**: Application starts even if the LLM API is down or keys are missing (routes correctly return `503 Service Unavailable`).
- **Regression Gating (CI)**: Includes a full evaluation suite that grades summaries with an LLM judge, and extracts with JSON schema validation. It gates GitHub PRs to prevent deploying degraded prompts.
- **Cost Tracking**: Computes LLM token costs dynamically for transparency over pipeline execution.

## Local Setup

### 1. Environment
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Fill in the `GROQ_API_KEY` and `LANGCHAIN_API_KEY`.

### 2. Run Locally (Python)
Install dependencies and run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 3. Run Locally (Docker)
```bash
docker compose up --build
```

## Testing and Verification

The project includes strict unit tests, linting, formatting, and a contract verifier.

```bash
# Run tests
pytest

# Verify architectural contract
python3 scripts/verify_contract.py

# Verify LangSmith Tracing works
python3 scripts/verify_traces.py
```

## Evaluation and CI

To test new prompt versions against the baseline dataset:
```bash
# Evaluate prompts against LangSmith datasets
python3 -m evaluation.run_regression --output-dir results
```
This outputs a `results/report.md` detailing pass/fail thresholds, LLM judge scores, and exact token costs. This script is run automatically on every Pull Request to `main`.
