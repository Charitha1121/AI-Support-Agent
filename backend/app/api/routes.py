import os

from fastapi import APIRouter, HTTPException

from app.agent.memory import conversation_memory
from app.agent.service import process_message
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionResponse,
)


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():

    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        "local",
    ).lower()

    api_key = os.getenv("OPENAI_API_KEY")

    return {
        "status": "healthy",
        "service": "NovaTech AI Support Agent",
        "embedding_provider": provider,
        "openai_enabled": bool(api_key),
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    result = process_message(
        session_id=request.session_id,
        message=request.message,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=result,
        )

    return result


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
)
def get_session(session_id: str):

    history = conversation_memory.get_history(
        session_id
    )

    return {
        "session_id": session_id,
        "messages": history,
    }


@router.delete(
    "/sessions/{session_id}",
)
def delete_session(session_id: str):

    conversation_memory.clear(session_id)

    return {
        "success": True,
        "session_id": session_id,
        "message": "Conversation history cleared.",
    }