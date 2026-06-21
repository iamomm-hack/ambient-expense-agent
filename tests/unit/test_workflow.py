import json
import base64
from unittest.mock import MagicMock, patch
import pytest

from google.adk.events.request_input import RequestInput
from google.adk.runners import InMemoryRunner
from google.genai import types

from expense_agent.agent import app


@pytest.mark.asyncio
async def test_auto_approval():
    """Test that expenses under $100 are automatically approved with base64 encoded Pub/Sub payload."""
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="expense_agent", user_id="test_user"
    )

    expense_data = {
        "submitter": "emp_123",
        "amount": 45.50,
        "category": "Meals",
        "description": "Business lunch with client",
        "date": "2026-06-21"
    }

    # Encode to base64 to simulate Pub/Sub payload
    expense_json = json.dumps(expense_data)
    encoded_data = base64.b64encode(expense_json.encode("utf-8")).decode("utf-8")
    pubsub_payload = {"data": encoded_data}

    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=json.dumps(pubsub_payload))]
    )

    events = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=new_message,
    ):
        events.append(event)

    final_output = None
    for e in events:
        if getattr(e, "output", None) is not None:
            final_output = e.output

    assert final_output is not None
    assert final_output["submitter"] == "emp_123"
    assert final_output["amount"] == 45.50
    assert final_output["category"] == "Meals"
    assert final_output["status"] == "AUTO_APPROVED"
    assert final_output["risk_alert"] is None
    assert final_output["manager_decision"] is None


@pytest.mark.asyncio
async def test_compliance_and_human_approval():
    """Test risk assessment and human approval workflow for expenses >= $100."""
    # Mock Gemini call
    with patch("expense_agent.agent.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.text = '{"risk_alert": "Low risk stay"}'
        mock_client.models.generate_content.return_value = mock_response

        runner = InMemoryRunner(app=app)
        session = await runner.session_service.create_session(
            app_name="expense_agent", user_id="test_user"
        )

        expense_data = {
            "submitter": "emp_007",
            "amount": 250.00,
            "category": "Lodging",
            "description": "Hotel stay for workshop",
            "date": "2026-06-21"
        }

        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=json.dumps(expense_data))]
        )

        # 1. Run the workflow until it yields RequestInput for manager decision
        events = []
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=new_message,
        ):
            events.append(event)

        # Assert RequestInput was yielded
        request_input_event = None
        for e in events:
            if getattr(e, "long_running_tool_ids", None):
                request_input_event = e

        assert request_input_event is not None

        # Verify LLM was called with correct config
        mock_client.models.generate_content.assert_called_once()
        args, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["model"] == "gemini-3.1-flash-lite"

        # 2. Resume the workflow with manager decision 'APPROVE'
        resume_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="manager_decision",
                        name="adk_request_input",
                        response={"manager_decision": "APPROVE"}
                    )
                )
            ]
        )

        events2 = []
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=resume_message,
        ):
            events2.append(event)

        final_output = None
        for e in events2:
            if getattr(e, "output", None) is not None:
                final_output = e.output

        assert final_output is not None
        assert final_output["submitter"] == "emp_007"
        assert final_output["amount"] == 250.00
        assert final_output["status"] == "APPROVED"
        assert final_output["risk_alert"] == "Low risk stay"
        assert final_output["manager_decision"] == "APPROVE"
