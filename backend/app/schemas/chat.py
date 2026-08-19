from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's message or question."
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Unique conversation thread identifier for multi-turn context."
    )


class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="The final customer-facing response text."
    )
    action_taken: Literal["rag", "tool", "escalate"] = Field(
        ...,
        description="The agentic action executed: 'rag', 'tool', or 'escalate'."
    )
    conversation_id: str = Field(
        ...,
        description="The active conversation thread ID."
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool executed (if action_taken is 'tool')."
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Knowledge base document titles/sections referenced (if action_taken is 'rag')."
    )
    escalation_id: Optional[str] = Field(
        default=None,
        description="Escalation reference ID (if action_taken is 'escalate')."
    )


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "NovaTech AI Support Agent"


class EscalationItem(BaseModel):
    id: str
    conversation_id: str
    user_message: str
    reason: str
    timestamp: str
    status: str
