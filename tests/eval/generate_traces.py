"""Generate agent traces for evaluation grading.

Runs each scenario in the eval dataset through the local ADK workflow,
handles human-in-the-loop interrupts automatically, and serializes
traces in the EvaluationDataset format expected by `agents-cli eval grade`.
"""

import asyncio
import json
import os
import uuid

from google.adk.runners import InMemoryRunner
from google.genai import types

from expense_agent.agent import app as adk_app

DATASET_PATH = "tests/eval/datasets/basic-dataset.json"
OUTPUT_DIR = "artifacts/traces"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "generated_traces.json")


async def generate_traces():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    eval_cases = []

    for item in dataset:
        scenario = item.get("scenario_name", "unknown")
        prompt = item.get("prompt", "")
        print(f"Running scenario: {scenario}")

        runner = InMemoryRunner(app=adk_app)
        app_name = adk_app.name
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name=app_name, user_id="eval_runner", session_id=session_id
        )

        new_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )

        all_events = []
        interrupt_handled = False

        # --- Initial run ---
        async for event in runner.run_async(
            user_id="eval_runner",
            session_id=session_id,
            new_message=new_message,
        ):
            all_events.append(event)

        # --- Handle human-in-the-loop interrupt ---
        interrupt_event = next(
            (e for e in all_events if getattr(e, "long_running_tool_ids", None)),
            None,
        )
        if interrupt_event:
            interrupt_handled = True
            decision = "REJECT" if "injection" in scenario else "APPROVE"
            print(f"  -> Interrupted for manual review. Auto-{decision.lower()}ing...")

            resume_message = types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="adk_request_input",
                        response={"manager_decision": decision},
                    )
                ],
            )
            async for event in runner.run_async(
                user_id="eval_runner",
                session_id=session_id,
                new_message=resume_message,
            ):
                all_events.append(event)

        # --- Build the final response text from the last output event ---
        final_output = None
        for e in reversed(all_events):
            if getattr(e, "output", None) is not None:
                final_output = e.output
                break

        response_text = json.dumps(final_output) if final_output else ""

        # --- Build agent_data with structured turn info ---
        turns = []
        for e in all_events:
            turn_entry = {}
            if getattr(e, "node_info", None):
                turn_entry["node"] = e.node_info.path
            if getattr(e, "output", None) is not None:
                turn_entry["output"] = str(e.output)
            if getattr(e, "long_running_tool_ids", None):
                turn_entry["interrupt"] = list(e.long_running_tool_ids)
            if getattr(e, "actions", None) and getattr(e.actions, "route", None):
                turn_entry["route"] = e.actions.route
            if getattr(e, "actions", None) and getattr(e.actions, "state_delta", None):
                turn_entry["state_delta"] = e.actions.state_delta
            if turn_entry:
                turns.append(turn_entry)

        agent_data = {
            "turns": turns,
            "interrupt_handled": interrupt_handled,
            "final_output": final_output,
        }

        # --- Build EvalCase in the EvaluationDataset schema ---
        eval_cases.append(
            {
                "eval_case_id": scenario,
                "prompt": prompt,
                "responses": [response_text],
                "agent_data": agent_data,
            }
        )

    # Wrap in EvaluationDataset envelope
    output = {"eval_cases": eval_cases}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nGenerated {len(eval_cases)} traces -> {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(generate_traces())
