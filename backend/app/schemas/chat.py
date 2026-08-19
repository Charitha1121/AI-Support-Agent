from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    session_id: str = Field(
        ...,
        min_length=1,
    )

    message: str = Field(
        ...,
        min_length=1,
    )


class ChatResponse(BaseModel):

    session_id: str

    message: str

    action: str

    tool_name: Optional[str] = None

    tool_args: Dict[str, Any] = Field(
        default_factory=dict
    )

    tool_result: Optional[Dict[str, Any]] = None

    retrieved_documents: List[Dict[str, Any]] = Field(
        default_factory=list
    )


class HealthResponse(BaseModel):

    status: str
    version: str
    service: str


class EscalationItem(BaseModel):

    id: int
    session_id: str
    message: str
    created_at: str