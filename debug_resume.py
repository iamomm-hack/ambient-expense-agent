"""Debug script to trace the exact workflow execution for a high-value expense."""
import asyncio
import json
import uuid

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

from expense_agent.agent import app as adk_app


async def main():
    runner = InMemoryRunner(app=adk_app)
    app_name = adk_app.name
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name=app_name, user_id="debug", session_id=session_id
    )

    prompt = '{"submitter": "emp_002", "amount": 500.00, "category": "Lodging", "description": "Hotel stay", "date": "2026-06-21"}'
    msg = Content(role="user", parts=[Part(text=prompt)])

    print("=" * 70)
    print("PHASE 1: Initial run")
    print("=" * 70)
    events_phase1 = []
    async for event in runner.run_async(
        user_id="debug", session_id=session_id, new_message=msg
    ):
        events_phase1.append(event)
        node_path = getattr(event.node_info, "path", "?") if event.node_info else "?"
        route = getattr(event.actions, "route", None) if event.actions else None
        state_delta = getattr(event.actions, "state_delta", {}) if event.actions else {}
        lrt = getattr(event, "long_running_tool_ids", None)
        print(f"  Node: {node_path}")
        print(f"    output: {event.output}")
        print(f"    route: {route}")
        print(f"    state_delta: {state_delta}")
        print(f"    long_running_tool_ids: {lrt}")
        print()

    # Check session state after phase 1
    session = await runner.session_service.get_session(
        app_name=app_name, user_id="debug", session_id=session_id
    )
    print("Session state after phase 1:")
    print(json.dumps(dict(session.state), indent=2, default=str))
    print()

    # Check for interrupt
    interrupt = next((e for e in events_phase1 if getattr(e, "long_running_tool_ids", None)), None)
    if not interrupt:
        print("NO INTERRUPT - test scenario did not reach human_review")
        return

    print("=" * 70)
    print("PHASE 2: Resume with APPROVE")
    print("=" * 70)
    resume_msg = Content(
        role="user",
        parts=[Part.from_function_response(
            name="adk_request_input",
            response={"manager_decision": "APPROVE"},
        )],
    )

    events_phase2 = []
    async for event in runner.run_async(
        user_id="debug", session_id=session_id, new_message=resume_msg
    ):
        events_phase2.append(event)
        node_path = getattr(event.node_info, "path", "?") if event.node_info else "?"
        route = getattr(event.actions, "route", None) if event.actions else None
        state_delta = getattr(event.actions, "state_delta", {}) if event.actions else {}
        lrt = getattr(event, "long_running_tool_ids", None)
        print(f"  Node: {node_path}")
        print(f"    output: {event.output}")
        print(f"    route: {route}")
        print(f"    state_delta: {state_delta}")
        print(f"    long_running_tool_ids: {lrt}")
        print()

    # Check final session state
    session = await runner.session_service.get_session(
        app_name=app_name, user_id="debug", session_id=session_id
    )
    print("Session state after phase 2:")
    print(json.dumps(dict(session.state), indent=2, default=str))


asyncio.run(main())
