"""FastAPI application for RAG Travel Assistant with Datadog and LangFuse observability."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_config import setup_logging
from app.api.routes import router
from app.observability.langfuse import tracer as langfuse_tracer
from app.rag.vector_store import QdrantVectorStore

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Global vector store instance
_vector_store = None


def get_vector_store() -> QdrantVectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore()
    return _vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("=" * 80)
    logger.info("Starting RAG Travel Assistant API")
    logger.info(f"Service: {settings.dd_service}")
    logger.info(f"Environment: {settings.dd_env}")
    logger.info(f"Version: {settings.dd_version}")
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Gemini Model: {settings.gemini_model}")
    logger.info(f"LangFuse Enabled: {langfuse_tracer.is_enabled}")
    logger.info("=" * 80)

    # Initialize vector store and load data
    try:
        logger.info("Initializing vector store...")
        vector_store = get_vector_store()

        logger.info("Creating collection...")
        vector_store.create_collection(recreate=True)

        logger.info("Loading travel destination data...")
        data_path = Path(__file__).parent / "data" / "destinations.json"
        vector_store.load_data_from_json(str(data_path))

        info = vector_store.get_collection_info()
        logger.info(
            f"Data loaded successfully: {info.get('points_count', 0)} documents"
        )
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Failed to initialize data: {e}")
        logger.warning("Continuing without preloaded data...")

    yield

    # Shutdown
    logger.info("Shutting down RAG Travel Assistant API")

    # Flush LangFuse traces
    if langfuse_tracer.is_enabled:
        logger.info("Flushing LangFuse traces...")
        langfuse_tracer.flush()

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="RAG Travel Assistant API",
    description="""
    A RAG (Retrieval Augmented Generation) powered travel assistant that provides 
    information about travel destinations, visa requirements, attractions, and more.
    
    ## Features
    
    * **Hybrid Search**: Combines semantic and keyword search for better retrieval
    * **LangGraph Workflow**: Orchestrated multi-step RAG pipeline
    * **Gemini LLM**: Powered by Google's Gemini for natural language generation
    * **Observability**: Integrated with LangFuse and Datadog for tracing
    * **Travel Knowledge**: 15+ destinations with detailed visa and travel information
    
    ## Usage
    
    Send a POST request to `/rag-travel-assistant` with your travel query.
    """,
    version=settings.dd_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
            if settings.dd_env == "development"
            else "An error occurred",
        },
    )


# Include routers
app.include_router(router, prefix="/api/v1", tags=["Travel Assistant"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Travel Assistant API",
        "version": settings.dd_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


def main():
    """Run the application."""
    import uvicorn

    logger.info(f"Starting server on {settings.app_host}:{settings.app_port}")

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.dd_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
