import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database.db import create_tables, seed_data
from app.schemas.chat import ChatRequest, ChatResponse, EscalationItem, HealthResponse
from app.services.chat_service import process_chat_message
from app.services.escalation_service import list_escalations

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("novatech.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables, seed fake data, and warm up indexes on startup."""
    logger.info("Initializing NovaTech AI Support Agent backend...")
    create_tables()
    seed_data()
    logger.info("Database initialized and seeded.")
    yield
    logger.info("Shutting down NovaTech AI Support Agent backend.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise-grade AI Support Agent with Tool Calling, LangGraph orchestration, RAG retrieval, and human escalation.",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local dev and flexible frontend preview
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Middleware for structured request logging, latency measurement, and error capture."""
    start_time = time.time()
    path = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        logger.info(
            f"HTTP {method} {path} completed with status {response.status_code} in {process_time:.2f}ms"
        )
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"HTTP {method} {path} failed with error: {str(e)} in {process_time:.2f}ms",
            exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred. Please try again later."}
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Service health and readiness check."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        service=settings.PROJECT_NAME
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Primary chat endpoint. Orchestrates LangGraph agent, routes between
    RAG, SQLite tool calling, and escalation, and returns structured response.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty."
        )

    response = process_chat_message(request)
    return response


@app.get("/api/escalations", response_model=List[EscalationItem], tags=["Escalations"])
def get_escalations_endpoint(limit: int = 50) -> List[EscalationItem]:
    """Retrieve logged human escalations from SQLite."""
    items = list_escalations(limit=limit)
    return [EscalationItem(**item) for item in items]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
