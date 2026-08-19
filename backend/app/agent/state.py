from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    LangGraph agent state schema for NovaTech customer support.
    """
    message: str
    action: Optional[str]
    tool_name: Optional[str]
    tool_args: Optional[Dict[str, Any]]
    response: Optional[str]
    conversation_history: Optional[List[Dict[str, str]]]
    retrieved_documents: Optional[List[Dict[str, Any]]]
    tool_result: Optional[Dict[str, Any]]
