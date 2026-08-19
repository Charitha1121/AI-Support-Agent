from typing import Dict, List, Any

from app.agent.graph import run_agent
from app.agent.state import AgentState
from app.schemas.chat import ChatRequest, ChatResponse


class ConversationManager:
    """
    In-memory conversation manager for Day 5.

    Stores conversation history using session_id.
    """

    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(
        self,
        session_id: str,
    ) -> List[Dict[str, str]]:
        return self._sessions.get(
            session_id,
            [],
        ).copy()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:

        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append(
            {
                "role": role,
                "content": content,
            }
        )

    def clear_session(
        self,
        session_id: str,
    ) -> None:

        self._sessions.pop(
            session_id,
            None,
        )


conversation_manager = ConversationManager()


def process_chat_message(
    request: ChatRequest,
) -> ChatResponse:

    session_id = request.session_id
    user_message = request.message

    # ---------------------------------------------------------
    # Get previous conversation
    # ---------------------------------------------------------

    history = conversation_manager.get_history(
        session_id
    )

    # ---------------------------------------------------------
    # Execute LangGraph agent
    # ---------------------------------------------------------

    try:

        final_state: AgentState = run_agent(
            user_message=user_message,
            conversation_history=history,
        )

    except Exception as exc:

        print(
            f"[chat_service] Agent execution error: {exc}"
        )

        # Safe fallback
        final_state = {
            "message": user_message,
            "action": "escalate",
            "tool_name": None,
            "tool_args": {},
            "response": (
                "I'm unable to process your request right now. "
                "Your request has been flagged for human support."
            ),
            "conversation_history": history,
            "retrieved_documents": [],
            "tool_result": None,
        }

    # ---------------------------------------------------------
    # Extract state
    # ---------------------------------------------------------

    action = final_state.get(
        "action",
        "escalate",
    )

    response_text = final_state.get(
        "response",
        "Your request has been flagged for human support.",
    )

    tool_name = final_state.get(
        "tool_name"
    )

    tool_args = final_state.get(
        "tool_args",
        {},
    )

    tool_result = final_state.get(
        "tool_result"
    )

    retrieved_documents = final_state.get(
        "retrieved_documents",
        [],
    )

    # ---------------------------------------------------------
    # Save user message
    # ---------------------------------------------------------

    conversation_manager.add_message(
        session_id=session_id,
        role="user",
        content=user_message,
    )

    # ---------------------------------------------------------
    # Save assistant message
    # ---------------------------------------------------------

    conversation_manager.add_message(
        session_id=session_id,
        role="assistant",
        content=response_text,
    )

    # ---------------------------------------------------------
    # Return complete API response
    # ---------------------------------------------------------

    return ChatResponse(
        session_id=session_id,
        message=response_text,
        action=action,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        retrieved_documents=retrieved_documents,
    )