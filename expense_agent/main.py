import base64
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from google.genai import types
from google.adk.runners import InMemoryRunner

from expense_agent.agent import app as adk_app
from expense_agent.app_utils.telemetry import setup_telemetry

# Setup standard python logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ambient-expense-agent")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set otel_to_cloud=False
    setup_telemetry(otel_to_cloud=False)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/")
async def pubsub_push(request: Request):
    """
    Webhook for Pub/Sub push messages.
    Pub/Sub payload format:
    {
      "message": {
        "attributes": {"key": "value"},
        "data": "SGVsbG8gQ2xvdWQgUHViL1N1YiEgSGVyZSBpcyBteSBtZXNzYWdlIQ==",
        "messageId": "136969346945"
      },
      "subscription": "projects/myproject/subscriptions/mysubscription"
    }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    message = body.get("message", {})
    message_id = message.get("messageId", "unknown-id")
    subscription = body.get("subscription", "unknown-subscription")

    # Normalize subscription from 'projects/X/subscriptions/Y' to 'Y'
    short_sub_name = subscription.split("/")[-1]
    
    logger.info(f"Received Pub/Sub message {message_id} from {short_sub_name}")

    # Use the normalized subscription name as the user_id for tracking
    # We will use the message_id as the session ID to keep records readable
    runner = InMemoryRunner(app=adk_app)
    session = await runner.session_service.create_session(
        app_name="expense_agent", user_id=short_sub_name, session_id=message_id
    )

    # Explicitly decode base64 data from Pub/Sub
    raw_data = message.get("data", "")
    decoded_payload = "{}"
    if raw_data:
        try:
            decoded_payload = base64.b64decode(raw_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to decode base64 payload: {e}")

    # Wrap the decoded payload into a Gemini Content object for the ADK runner
    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=decoded_payload)]
    )

    events = []
    try:
        async for event in runner.run_async(
            user_id=short_sub_name,
            session_id=session.id,
            new_message=new_message,
        ):
            events.append(event)
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        # We return 200 even on workflow failure so Pub/Sub doesn't infinitely retry 
        # unless it's a transient infrastructure error. Here we assume bad data.
        return {"status": "error", "message": str(e)}

    # Inspect events to see if it reached a final output or requires human review
    final_status = "workflow_completed"
    for e in events:
        if getattr(e, "long_running_tool_ids", None):
            final_status = "awaiting_human_review"
            logger.info(f"Message {message_id} interrupted for human review.")
        if getattr(e, "output", None) is not None:
            final_status = "completed"
            logger.info(f"Message {message_id} completed with output: {e.output}")

    return {"status": "ok", "workflow_status": final_status, "session_id": session.id}
