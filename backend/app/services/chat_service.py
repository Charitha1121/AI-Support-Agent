import logging
import uuid
from typing import Dict, List, Optional

from app.agent.graph import run_agent
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("novatech.chat_service")


class ConversationManager:
    """
    Manages in-memory multi-turn conversation threads indexed by conversation_id.
    """
    def __init__(self, max_history_per_conv: int = 20):
        self._conversations: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_per_conv = max_history_per_conv

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        return self._conversations.get(conversation_id, [])

    def append_turn(self, conversation_id: str, user_msg: str, assistant_msg: str):
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
        
        history = self._conversations[conversation_id]
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

        # Trim old turns if exceeding limit
        if len(history) > self.max_history_per_conv:
            self._conversations[conversation_id] = history[-self.max_history_per_conv:]

    def clear_conversation(self, conversation_id: str):
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]


# Global conversation manager instance
conversation_manager = ConversationManager()


def process_chat_message(request: ChatRequest) -> ChatResponse:
    """
    Execute the agent workflow for a customer message within a conversation thread.
    """
    conversation_id = request.conversation_id or f"conv-{uuid.uuid4().hex[:8]}"
    user_message = request.message.strip()

    # Retrieve prior conversation history for multi-turn context
    history = conversation_manager.get_history(conversation_id)

    try:
        # Run agent graph
        final_state = run_agent(
            user_message=user_message,
            conversation_id=conversation_id,
            history=history
        )

        response_text = final_state.get("response", "Thank you for contacting NovaTech Support.")
        action_taken = final_state.get("action") or "rag"
        tool_name = final_state.get("tool_name")
        sources = final_state.get("sources")
        escalation_id = final_state.get("escalation_id")

        # Save to conversation memory
        conversation_manager.append_turn(conversation_id, user_message, response_text)

        return ChatResponse(
            response=response_text,
            action_taken=action_taken,
            conversation_id=conversation_id,
            tool_name=tool_name,
            sources=sources,
            escalation_id=escalation_id
        )

    except Exception as e:
        logger.error(f"Error executing support agent graph for conv {conversation_id}: {str(e)}", exc_info=True)
        # Controlled fallback response to user
        return ChatResponse(
            response="I apologize, but an unexpected technical error occurred while processing your request. Please try again or request human assistance.",
            action_taken="escalate",
            conversation_id=conversation_id,
            escalation_id=None
        )
