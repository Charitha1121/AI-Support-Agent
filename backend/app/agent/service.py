from __future__ import annotations

from typing import Any, Dict

from app.agent.graph import support_graph
from app.agent.memory import conversation_memory


def process_message(
    session_id: str,
    message: str,
) -> Dict[str, Any]:

    message = message.strip()

    if not message:
        return {
            "success": False,
            "error": "Message cannot be empty.",
        }

    if len(message) > 2000:
        return {
            "success": False,
            "error": "Message exceeds the 2000 character limit.",
        }

    history = conversation_memory.get_history(session_id)

    state = {
        "session_id": session_id,
        "message": message,
        "conversation_history": history,
    }

    try:
        result = support_graph.invoke(state)

        response = result.get(
            "response",
            "I'm sorry, but I couldn't process your request.",
        )

        conversation_memory.add_message(
            session_id=session_id,
            role="user",
            content=message,
        )

        conversation_memory.add_message(
            session_id=session_id,
            role="assistant",
            content=response,
        )

        return {
            "success": True,
            "session_id": session_id,
            "message": message,
            "response": response,
            "action": result.get("action"),
            "tool_name": result.get("tool_name"),
            "tool_result": result.get("tool_result"),
            "retrieved_documents": result.get(
                "retrieved_documents",
                [],
            ),
            "history_length": len(
                conversation_memory.get_history(session_id)
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "session_id": session_id,
            "error": "Agent execution failed.",
            "detail": str(exc),
        }