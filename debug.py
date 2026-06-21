import asyncio
import json
import base64
from unittest.mock import MagicMock, patch
from google.genai import types
from google.adk.runners import InMemoryRunner
from expense_agent.agent import app

async def main():
    print("Starting debug script")
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

        events = []
        async for event in runner.run_async(
            user_id="test_user",
            session_id=session.id,
            new_message=new_message,
        ):
            print(f"EVENT: {event}")
            events.append(event)
            
        print("EVENTS CAPTURED:", len(events))
        
        request_input_event = None
        for e in events:
            if type(e).__name__ == "RequestInput" or getattr(e, "interrupt_ids", None):
                request_input_event = e
                
        print("REQUEST_INPUT_EVENT:", request_input_event)

if __name__ == "__main__":
    asyncio.run(main())
