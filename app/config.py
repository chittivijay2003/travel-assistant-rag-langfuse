"""Configuration management for the application."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    # Datadog Configuration
    dd_service: str = os.getenv("DD_SERVICE", "travel-assistant-rag")
    dd_env: str = os.getenv("DD_ENV", "development")
    dd_version: str = os.getenv("DD_VERSION", "1.0.0")
    dd_site: str = os.getenv("DD_SITE", "datadoghq.com")
    datadog_api_key: Optional[str] = os.getenv("DATADOG_API_KEY")
    datadog_app_key: Optional[str] = os.getenv("DATADOG_APP_KEY")

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection_name: str = os.getenv(
        "QDRANT_COLLECTION_NAME", "travel_destinations"
    )

    # Application Configuration
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # RAG Configuration
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))

    # Gemini Configuration
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    gemini_max_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env


settings = Settings()
