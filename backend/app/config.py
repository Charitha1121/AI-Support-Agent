import os
from pathlib import Path
from typing import List


class Settings:
    PROJECT_NAME: str = "NovaTech AI Support Agent"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""

    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Database
    DATABASE_PATH: Path = Path(__file__).resolve().parent / "database" / "support.db"

    # CORS Settings
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
    ]


settings = Settings()
