from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    success: bool

    session_id: str

    message: Optional[str] = None

    response: Optional[str] = None

    action: Optional[str] = None

    tool_name: Optional[str] = None

    tool_result: Optional[Dict[str, Any]] = None

    retrieved_documents: List[Dict[str, Any]] = []

    history_length: int = 0

    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    embedding_provider: str
    openai_enabled: bool


class SessionResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]