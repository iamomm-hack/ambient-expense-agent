# Ambient Expense Agent

An event-driven "ambient" corporate expense approval agent built with Google ADK 2.2.0 and FastAPI.

This agent automatically processes incoming expense reports via a Pub/Sub webhook, applying intelligent routing, deterministic security constraints, and human-in-the-loop approvals before finalizing decisions.

## Features

- **Ambient Execution**: Runs headlessly as a FastAPI webhook receiver (`expense_agent/main.py`), automatically processing base64-encoded Pub/Sub payloads without human initiation.
- **Deterministic Security**: Implements a dedicated `security_checkpoint_node` to scrub PII (like SSNs) and flag prompt injection attempts *before* LLM evaluation.
- **Smart Routing**: Auto-approves expenses under $100 and routes high-value or flagged expenses to a manager for manual review.
- **Human-in-the-Loop**: Uses ADK's `RequestInput` to securely pause workflows and solicit `APPROVE`/`REJECT` decisions from human reviewers.
- **Local Evaluation Pipeline**: Includes a comprehensive local trace generator and grading suite to validate routing and security behaviors.

## Project Structure

```
ambient-expense-agent/
├── expense_agent/             # Core agent code
│   ├── agent.py               # Main agent LangGraph logic
│   ├── main.py                # FastAPI Webhook receiver for ambient execution
│   ├── config.py              # Agent configuration (thresholds, models)
│   └── app_utils/             # App utilities and telemetry config
├── tests/                     # Test suite and evaluation
│   ├── unit/                  # Pytest unit tests
│   ├── integration/           # Pytest integration tests
│   └── eval/                  # Evaluation loop
│       ├── datasets/          # Synthetic evaluation datasets
│       ├── eval_config.yaml   # Custom metrics config
│       ├── generate_traces.py # Trace generator script
│       └── grade_traces.py    # Local trace grader script
├── artifacts/traces/          # Generated evaluation traces
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

## Quick Start

Before you begin, ensure you have:
- **uv**: Python package manager
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`

### Running the Ambient Webhook Service

To start the FastAPI service that listens for incoming Pub/Sub push messages:

```bash
uv run python -m uvicorn expense_agent.main:app --host 0.0.0.0 --port 8080
```

You can simulate an incoming Pub/Sub message using PowerShell:

```powershell
$expense = @{
    amount = 45
    submitter = "bob@company.com"
    category = "meals"
    description = "Team lunch"
    date = "2026-06-21"
} | ConvertTo-Json -Compress

$base64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($expense))

$body = @{
    message = @{
        messageId = "test-pubsub-id-001"
        data = $base64
    }
    subscription = "projects/my-project/subscriptions/test-expense-topic"
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:8080/" -Method Post -Body $body -ContentType "application/json"
```

### Running the Evaluation Loop (Phase 3)

The agent includes a rigorous evaluation suite that generates and grades traces against synthetic threat vectors and standard use cases.

1. **Generate Traces:**
   ```bash
   uv run python tests/eval/generate_traces.py
   ```
2. **Grade Traces Locally:**
   ```bash
   uv run python tests/eval/grade_traces.py
   ```
   Or use the built-in agents-cli evaluation (if Vertex AI credentials are configured):
   ```bash
   uv tool run google-agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml
   ```

## Development and Testing

Test the agent interactively with the local ADK web UI:
```bash
uv run python -m google.adk.cli web .
```

Run unit and integration tests:
```bash
uv run pytest tests/unit tests/integration
```

## Deployment

To deploy the agent to Google Cloud:
```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging via OpenTelemetry (can be disabled locally via `otel_to_cloud=False` in `expense_agent/main.py`).
