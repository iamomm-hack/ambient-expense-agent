# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import re
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import google.auth
from google.auth.exceptions import DefaultCredentialsError

from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow, node
from google.genai import Client, types
from pydantic import BaseModel, Field


load_dotenv()

# Configure environment based on .env settings
use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"
if use_vertex:
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    if "GOOGLE_CLOUD_PROJECT" not in os.environ:
        try:
            _, project_id = google.auth.default()
            if project_id:
                os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        except DefaultCredentialsError:
            pass
else:
    # Disable Vertex AI and default to Developer API (e.g. AI Studio)
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


# --- Schemas ---

class ExpenseReport(BaseModel):
    employee_id: str
    amount: float
    category: str
    description: str
    receipt_text: str


class ComplianceResult(BaseModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    rationale: str
    recommendation: str


class WorkflowState(BaseModel):
    employee_id: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    description: Optional[str] = None
    receipt_text: Optional[str] = None
    status: Optional[Literal["AUTO_APPROVED", "SECURITY_REJECTED", "APPROVED", "REJECTED"]] = None
    risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    rationale: Optional[str] = None
    recommendation: Optional[str] = None
    manager_decision: Optional[str] = None
    requires_human_review: bool = False


class WorkflowOutput(BaseModel):
    employee_id: str
    amount: float
    status: Literal["AUTO_APPROVED", "SECURITY_REJECTED", "APPROVED", "REJECTED"]
    risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None
    rationale: Optional[str] = None
    manager_decision: Optional[str] = None
    requires_human_review: bool


# --- Helper Functions ---

def redact_pii(text: str) -> str:
    """Redacts obvious email, phone numbers, and SSN patterns from input text."""
    # Redact email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    # Redact phone numbers (various international and local formats)
    text = re.sub(r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b', '[REDACTED_PHONE]', text)
    # Redact Social Security Numbers (XXX-XX-XXXX)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)
    return text


def is_prompt_injection(text: str) -> bool:
    """Checks if the text contains obvious prompt-injection patterns."""
    patterns = [
        r"(?i)ignore\s+(?:all\s+)?prior\s+instructions",
        r"(?i)ignore\s+(?:all\s+)?previous\s+instructions",
        r"(?i)ignore\s+(?:all\s+)?system\s+prompts?",
        r"(?i)you\s+are\s+now\s+a",
        r"(?i)system\s+instruction\s+override",
        r"(?i)new\s+role\s*:",
        r"(?i)bypass\s+safety",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


# --- Workflow Nodes ---

def intake_node(node_input: ExpenseReport) -> Event:
    """Receives and saves the raw expense report details in the session state."""
    return Event(
        output=node_input,
        state={
            "employee_id": node_input.employee_id,
            "amount": node_input.amount,
            "category": node_input.category,
            "description": node_input.description,
            "receipt_text": node_input.receipt_text,
        }
    )


def routing_node(node_input: ExpenseReport) -> Event:
    """Routes the workflow based on the expense amount (auto-approve vs security screen)."""
    if node_input.amount < 100.0:
        return Event(
            output="auto_approve",
            route="auto_approve",
            state={"status": "AUTO_APPROVED", "requires_human_review": False}
        )
    else:
        return Event(
            output=node_input,
            route="security_check",
            state={"requires_human_review": True}
        )


def security_node(ctx: Context, node_input: ExpenseReport) -> Event:
    """Redacts PII and screens the expense text details for prompt injection attempts."""
    redacted_desc = redact_pii(node_input.description)
    redacted_receipt = redact_pii(node_input.receipt_text)

    if is_prompt_injection(node_input.description) or is_prompt_injection(node_input.receipt_text):
        return Event(
            output="reject",
            route="security_reject",
            state={
                "status": "SECURITY_REJECTED",
                "description": redacted_desc,
                "receipt_text": redacted_receipt,
            }
        )

    # Continue with redacted text
    return Event(
        output={
            "employee_id": node_input.employee_id,
            "amount": node_input.amount,
            "category": node_input.category,
            "description": redacted_desc,
            "receipt_text": redacted_receipt,
        },
        route="security_pass",
        state={
            "description": redacted_desc,
            "receipt_text": redacted_receipt,
        }
    )


@node
def compliance_node(ctx: Context, node_input: dict) -> Event:
    """Invokes Gemini to analyze compliance risk and policy violations."""
    client = Client()
    prompt = f"""
    Analyze this corporate expense report for compliance policy and risks:
    Employee ID: {node_input.get('employee_id')}
    Amount: {node_input.get('amount')} USD
    Category: {node_input.get('category')}
    Description: {node_input.get('description')}
    Receipt Text: {node_input.get('receipt_text')}
    """

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ComplianceResult,
            temperature=0.0,
        )
    )

    result = ComplianceResult.model_validate_json(response.text)

    return Event(
        output=result.model_dump(),
        state={
            "risk_level": result.risk_level,
            "rationale": result.rationale,
            "recommendation": result.recommendation,
        }
    )


@node(name="human_review", rerun_on_resume=True)
async def human_review_node(ctx: Context, node_input: dict):
    """Pauses the workflow to request manager approval, resuming once feedback is received."""
    if not ctx.resume_inputs or "manager_decision" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="manager_decision",
            message="Manager review required for expense. Please reply with 'APPROVE' or 'REJECT'."
        )
        return

    decision = ctx.resume_inputs["manager_decision"]
    decision_str = str(decision).upper().strip()
    if "APPROVE" in decision_str:
        status = "APPROVED"
    else:
        status = "REJECTED"

    yield Event(
        output=status,
        state={
            "manager_decision": decision_str,
            "status": status,
        }
    )


def finalize_node(ctx: Context) -> Event:
    """Builds and returns the final structured output matching the output schema."""
    output_data = {
        "employee_id": ctx.state.get("employee_id"),
        "amount": ctx.state.get("amount"),
        "status": ctx.state.get("status"),
        "risk_level": ctx.state.get("risk_level"),
        "rationale": ctx.state.get("rationale"),
        "manager_decision": ctx.state.get("manager_decision"),
        "requires_human_review": ctx.state.get("requires_human_review", False),
    }
    return Event(
        output=output_data
    )


# --- Workflow Graph Definition ---

root_agent = Workflow(
    name="root_agent",
    edges=[
        ('START', intake_node),
        (intake_node, routing_node),
        (routing_node, finalize_node, "auto_approve"),
        (routing_node, security_node, "security_check"),
        (security_node, finalize_node, "security_reject"),
        (security_node, compliance_node, "security_pass"),
        (compliance_node, human_review_node),
        (human_review_node, finalize_node),
    ],
    input_schema=ExpenseReport,
    state_schema=WorkflowState,
    output_schema=WorkflowOutput,
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)

