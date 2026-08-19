"""
NovaTech AI Support Agent
-------------------------
Production-oriented FastAPI application entry point.

Responsibilities:
- Application lifecycle management
- Database initialization/seeding
- Health/readiness endpoints
- Chat API
- Escalation API
- Request ID generation
- Structured request logging
- Latency measurement
- Safe exception handling
- CORS configuration
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.services.chat_service import (
    process_chat_message,
    conversation_manager,
)
from app.config import settings
from app.database.db import create_tables, seed_data
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    EscalationItem,
    HealthResponse,
)

from app.services.escalation_service import list_escalations


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "[%(levelname)s] "
        "[%(name)s] "
        "%(message)s"
    ),
    handlers=[
        logging.StreamHandler()
    ],
)

logger = logging.getLogger("novatech.api")


# ============================================================
# APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle.

    Startup:
        1. Create database tables.
        2. Seed demo data.
        3. Prepare application.

    Shutdown:
        Gracefully stop the application.
    """

    logger.info("=" * 70)
    logger.info("Starting NovaTech AI Support Agent")
    logger.info("Version: %s", settings.VERSION)
    logger.info("=" * 70)

    try:
        # ----------------------------------------------------
        # Database initialization
        # ----------------------------------------------------
        create_tables()
        logger.info("Database tables initialized.")

        # ----------------------------------------------------
        # Seed development/demo data
        # ----------------------------------------------------
        seed_data()
        logger.info("Database seed completed.")

        logger.info("NovaTech backend startup completed successfully.")

    except Exception:
        logger.exception("Application startup failed.")
        raise

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------
    logger.info("Shutting down NovaTech AI Support Agent...")
    logger.info("Shutdown completed.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Enterprise-oriented AI customer support agent using "
        "RAG retrieval, LangGraph orchestration, tool calling, "
        "and human escalation."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    # Development-friendly configuration.
    # For production, replace "*" with your frontend domain.
    allow_origins=["*"],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST LOGGING / REQUEST ID MIDDLEWARE
# ============================================================

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """
    Global middleware responsible for:

    - Generating a request ID
    - Measuring request latency
    - Logging request information
    - Adding X-Request-ID to responses
    - Capturing unexpected failures
    """

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    request.state.request_id = request_id

    start_time = time.perf_counter()

    method = request.method
    path = request.url.path

    logger.info(
        "Request started | request_id=%s | method=%s | path=%s",
        request_id,
        method,
        path,
    )

    try:
        response = await call_next(request)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            "Request completed | "
            "request_id=%s | method=%s | path=%s | "
            "status=%s | latency_ms=%.2f",
            request_id,
            method,
            path,
            response.status_code,
            elapsed_ms,
        )

        return response

    except Exception:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "Request failed | "
            "request_id=%s | method=%s | path=%s | "
            "latency_ms=%.2f",
            request_id,
            method,
            path,
            elapsed_ms,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers={
                "X-Request-ID": request_id
            },
            content={
                "detail": (
                    "An internal server error occurred. "
                    "Please try again later."
                ),
                "request_id": request_id,
            },
        )


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    """
    Final safety net for unexpected application exceptions.

    Internal exception details are intentionally not exposed
    to the client.
    """

    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4()),
    )

    logger.exception(
        "Unhandled application exception | request_id=%s",
        request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={
            "X-Request-ID": request_id
        },
        content={
            "detail": (
                "An unexpected error occurred. "
                "Please try again later."
            ),
            "request_id": request_id,
        },
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """
    Service health and readiness check.

    Supports both:
        GET /health
        GET /api/health
    """
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        service=settings.PROJECT_NAME,
    )
# ============================================================
# READINESS CHECK
# ============================================================

@app.get(
    "/ready",
    tags=["Health"],
)
def readiness_check():
    """
    Readiness endpoint.

    Verifies that the application's core infrastructure
    can be accessed before declaring the service ready.
    """

    try:
        # Import here to avoid unnecessary startup coupling.
        from app.database.db import create_tables

        # Database connectivity/initialization sanity check.
        create_tables()

        return {
            "status": "ready",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
        }

    except Exception:
        logger.exception("Readiness check failed.")

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "service": settings.PROJECT_NAME,
                "version": settings.VERSION,
            },
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.get(
    "/",
    tags=["Health"],
)
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
        "readiness": "/ready",
        "chat": "/api/chat",
        "escalations": "/api/escalations",
    }

# ============================================================
# CHAT ENDPOINT
# ============================================================
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
)
@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
)
def chat_endpoint(
    request: ChatRequest,
) -> ChatResponse:
    """
    Primary NovaTech AI Support endpoint.

    Flow:

        User Message
             |
             v
        LangGraph Router
             |
        +----+----+---------+
        |         |         |
       RAG      Tool    Escalation
        |         |         |
        v         v         v
     Response  Tool     Human Support
                |
                v
          Final Response
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not request.message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty.",
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty.",
        )

    # --------------------------------------------------------
    # Reasonable message-size protection
    # --------------------------------------------------------

    MAX_MESSAGE_LENGTH = 5000

    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Message is too long. "
                f"Maximum allowed length is "
                f"{MAX_MESSAGE_LENGTH} characters."
            ),
        )

    # --------------------------------------------------------
    # Normalize message
    # --------------------------------------------------------

    request.message = message

    request_id = getattr(
        getattr(request, "state", None),
        "request_id",
        None,
    )

    logger.info(
        "Processing chat request | request_id=%s",
        request_id,
    )

    # --------------------------------------------------------
    # Agent execution
    # --------------------------------------------------------

    try:
        response = process_chat_message(request)

        logger.info(
            "Chat request processed successfully | request_id=%s",
            request_id,
        )

        return response

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Chat processing failed | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to process your request right now. "
                "Please try again later."
            ),
        )


# ============================================================
# ESCALATIONS ENDPOINT
# ============================================================

@app.get(
    "/api/escalations",
    response_model=List[EscalationItem],
    tags=["Escalations"],
)
def get_escalations_endpoint(
    limit: int = 50,
) -> List[EscalationItem]:
    """
    Retrieve human-support escalation records.

    This endpoint is intended for the support/admin dashboard.
    """

    # --------------------------------------------------------
    # Validate limit
    # --------------------------------------------------------

    if limit < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be greater than 0.",
        )

    if limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit cannot exceed 100.",
        )

    try:
        items = list_escalations(limit=limit)

        return [
            EscalationItem(**item)
            for item in items
        ]

    except Exception:
        logger.exception(
            "Failed to retrieve escalation records."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve escalation records.",
        )

@app.get(
    "/api/sessions/{session_id}",
    tags=["Memory"],
)
def get_session(
    session_id: str,
):
    """
    Return conversation history for a session.
    """

    messages = conversation_manager.get_history(
        session_id
    )

    return {
        "session_id": session_id,
        "messages": messages,
    }


@app.delete(
    "/api/sessions/{session_id}",
    tags=["Memory"],
)
def clear_session(
    session_id: str,
):
    """
    Clear conversation memory for a session.
    """

    conversation_manager.clear_session(
        session_id
    )

    return {
        "session_id": session_id,
        "status": "cleared",
    }
# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )