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

import os
import json
import base64
import re
from typing import Literal, Optional

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

from expense_agent.config import AUTO_APPROVE_THRESHOLD, LLM_MODEL

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

class RiskAssessmentResult(BaseModel):
    risk_alert: str


class WorkflowState(BaseModel):
    amount: Optional[float] = None
    submitter: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    date: Optional[str] = None
    status: Optional[str] = None
    risk_alert: Optional[str] = None
    manager_decision: Optional[str] = None
    is_security_flagged: Optional[bool] = None
    redacted_categories: Optional[list[str]] = None


class WorkflowOutput(BaseModel):
    amount: float
    submitter: str
    category: str
    description: str
    date: str
    status: str
    risk_alert: Optional[str] = None
    manager_decision: Optional[str] = None
    is_security_flagged: Optional[bool] = None
    redacted_categories: Optional[list[str]] = None


# --- Workflow Nodes ---

def intake_node(node_input) -> Event:
    """Receives and parses the raw Pub/Sub or direct JSON message payload."""
    # Handle Content objects from ADK playground
    if hasattr(node_input, "parts") and node_input.parts:
        text = ""
        for part in node_input.parts:
            if hasattr(part, "text") and part.text:
                text = part.text
                break
        if text:
            try:
                node_input = json.loads(text)
            except Exception:
                node_input = {"description": text}
        else:
            node_input = {}

    # Robust conversion/parsing if node_input is a string
    if isinstance(node_input, str):
        try:
            node_input = json.loads(node_input)
        except Exception:
            node_input = {"description": node_input}

    if not isinstance(node_input, dict):
        node_input = {}

    payload = node_input
    # Check for Pub/Sub data field
    if "data" in node_input:
        data_payload = node_input["data"]
        if isinstance(data_payload, str):
            try:
                # Try decoding base64
                decoded = base64.b64decode(data_payload).decode("utf-8")
                payload = json.loads(decoded)
            except Exception:
                # Fallback to plain JSON string parsing
                try:
                    payload = json.loads(data_payload)
                except Exception:
                    # Fallback to the raw string/dict
                    pass
        elif isinstance(data_payload, dict):
            payload = data_payload

    # Extract fields from the resolved payload
    amount_raw = payload.get("amount")
    try:
        amount = float(amount_raw) if amount_raw is not None else 0.0
    except (ValueError, TypeError):
        amount = 0.0

    submitter = payload.get("submitter") or payload.get("employee_id") or ""
    category = payload.get("category") or ""
    description = payload.get("description") or ""
    date = payload.get("date") or ""

    # Return event with updated state
    return Event(
        output=payload,
        state={
            "amount": amount,
            "submitter": submitter,
            "category": category,
            "description": description,
            "date": date,
        }
    )


def security_checkpoint_node(ctx: Context) -> Event:
    """Deterministically scrubs PII and checks for prompt injection."""
    description = ctx.state.get("description", "")
    redacted_categories = []
    
    # 1. Check for prompt injection
    injection_keywords = ["ignore previous", "auto-approve", "bypass", "system prompt", "forget", "you must approve", "override rules", "disregard"]
    lower_desc = description.lower()
    is_injection = any(kw in lower_desc for kw in injection_keywords)
    
    if is_injection:
        return Event(
            output="flagged",
            route="flagged",
            state={
                "is_security_flagged": True,
                "status": "SECURITY_FLAGGED",
                "risk_alert": "Prompt injection detected.",
            }
        )
    
    # 2. Scrub PII deterministically
    ssn_pattern = r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
    cc_pattern = r"\b(?:\d[-.\s]*?){13,16}\b"
    
    if re.search(ssn_pattern, description):
        description = re.sub(ssn_pattern, "[REDACTED_SSN]", description)
        redacted_categories.append("SSN")
        
    if re.search(cc_pattern, description):
        description = re.sub(cc_pattern, "[REDACTED_CC]", description)
        redacted_categories.append("Credit Card")
        
    return Event(
        output="clean",
        route="clean",
        state={
            "description": description,
            "redacted_categories": redacted_categories,
            "is_security_flagged": False,
        }
    )


def routing_node(ctx: Context) -> Event:
    """Routes the workflow based on the expense amount."""
    amount = ctx.state.get("amount", 0.0)
    if amount < AUTO_APPROVE_THRESHOLD:
        return Event(
            output="auto_approve",
            route="auto_approve",
            state={"status": "AUTO_APPROVED"}
        )
    else:
        return Event(
            output="risk_assessment",
            route="risk_assessment"
        )


@node
def risk_assessment_node(ctx: Context) -> Event:
    """Calls Gemini to evaluate expense details and generate a risk alert."""
    client = Client()
    
    amount = ctx.state.get("amount")
    submitter = ctx.state.get("submitter")
    category = ctx.state.get("category")
    description = ctx.state.get("description")
    date = ctx.state.get("date")

    prompt = f"""
    Analyze the following corporate expense details for any policy violations or risk:
    Submitter: {submitter}
    Amount: {amount} USD
    Category: {category}
    Description: {description}
    Date: {date}

    Identify any potential concerns or violations (e.g. unusually high amounts for the category, vague description, compliance risk).
    Provide a clear assessment or alert.
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RiskAssessmentResult,
            temperature=0.0,
        )
    )

    result = RiskAssessmentResult.model_validate_json(response.text)

    return Event(
        output=result.risk_alert,
        state={
            "risk_alert": result.risk_alert
        }
    )


@node(name="human_review", rerun_on_resume=True)
async def human_review_node(ctx: Context):
    """Pauses the workflow to request manager approval, resuming once feedback is received."""
    if not ctx.resume_inputs or "manager_decision" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="manager_decision",
            message="Manager review required for expense. Please reply with 'APPROVE' or 'REJECT'."
        )
        return

    decision = ctx.resume_inputs["manager_decision"]
    # ADK 2.2.0: resume_inputs values are FunctionResponse.response dicts
    if isinstance(decision, dict):
        decision = decision.get("manager_decision", next(iter(decision.values()), ""))
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
        "amount": ctx.state.get("amount"),
        "submitter": ctx.state.get("submitter"),
        "category": ctx.state.get("category"),
        "description": ctx.state.get("description"),
        "date": ctx.state.get("date"),
        "status": ctx.state.get("status"),
        "risk_alert": ctx.state.get("risk_alert"),
        "manager_decision": ctx.state.get("manager_decision"),
        "is_security_flagged": ctx.state.get("is_security_flagged"),
        "redacted_categories": ctx.state.get("redacted_categories"),
    }
    return Event(
        output=output_data
    )


# --- Workflow Graph Definition ---

root_agent = Workflow(
    name="root_agent",
    edges=[
        ('START', intake_node),
        (intake_node, security_checkpoint_node),
        (security_checkpoint_node, {"clean": routing_node, "flagged": human_review_node}),
        (routing_node, {"auto_approve": finalize_node, "risk_assessment": risk_assessment_node}),
        (risk_assessment_node, human_review_node),
        (human_review_node, finalize_node),
    ],

    state_schema=WorkflowState,
    output_schema=WorkflowOutput,
)

app = App(
    root_agent=root_agent,
    name="expense_agent",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
