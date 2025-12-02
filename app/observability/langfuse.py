"""LangFuse tracing integration for observability."""

import logging
import os
from typing import Any, Callable, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Configure LangFuse environment variables
if settings.langfuse_public_key and settings.langfuse_secret_key:
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host

try:
    from langfuse import Langfuse, observe

    LANGFUSE_AVAILABLE = True
    logger.info(f"LangFuse initialized with host: {settings.langfuse_host}")
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("LangFuse not available, tracing will be disabled")

    # Create dummy decorator
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator


class LangFuseTracer:
    """LangFuse tracing wrapper for RAG operations."""

    _instance = None
    _langfuse_client = None

    def __new__(cls):
        """Singleton pattern for LangFuse client."""
        if cls._instance is None:
            cls._instance = super(LangFuseTracer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize LangFuse client."""
        if not self._initialized:
            try:
                if settings.langfuse_public_key and settings.langfuse_secret_key:
                    self._langfuse_client = Langfuse(
                        public_key=settings.langfuse_public_key,
                        secret_key=settings.langfuse_secret_key,
                        host=settings.langfuse_host,
                    )
                    logger.info(
                        f"LangFuse client initialized: {settings.langfuse_host}"
                    )
                    self._initialized = True
                else:
                    logger.warning(
                        "LangFuse credentials not configured, tracing disabled"
                    )
                    self._initialized = False
            except Exception as e:
                logger.error(f"Failed to initialize LangFuse: {e}")
                self._initialized = False

    @property
    def client(self) -> Optional[Langfuse]:
        """Get LangFuse client."""
        return self._langfuse_client

    @property
    def is_enabled(self) -> bool:
        """Check if LangFuse is enabled."""
        return self._initialized and self._langfuse_client is not None

    def trace_retrieval(self, func: Callable) -> Callable:
        """
        Decorator to trace retrieval operations.

        Args:
            func: Function to trace

        Returns:
            Wrapped function with tracing
        """
        if not self.is_enabled or not LANGFUSE_AVAILABLE:
            return func

        return observe(name="hybrid_retrieval", as_type="retriever")(func)

    def trace_generation(self, func: Callable) -> Callable:
        """
        Decorator to trace LLM generation operations.

        Args:
            func: Function to trace

        Returns:
            Wrapped function with tracing
        """
        if not self.is_enabled or not LANGFUSE_AVAILABLE:
            return func

        return observe(
            name="llm_generation",
            as_type="generation",
        )(func)

    def trace_rag_pipeline(self, func: Callable) -> Callable:
        """
        Decorator to trace complete RAG pipeline.

        Args:
            func: Function to trace

        Returns:
            Wrapped function with tracing
        """
        if not self.is_enabled or not LANGFUSE_AVAILABLE:
            return func

        return observe(name="rag_pipeline", as_type="chain")(func)

    def create_trace(
        self,
        name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Manually create a trace.

        Args:
            name: Trace name
            input_data: Input data
            output_data: Output data
            metadata: Additional metadata
        """
        if not self.is_enabled:
            return

        try:
            self._langfuse_client.trace(
                name=name, input=input_data, output=output_data, metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Error creating manual trace: {e}")

    def flush(self) -> None:
        """Flush pending traces."""
        if self.is_enabled:
            try:
                self._langfuse_client.flush()
                logger.debug("LangFuse traces flushed")
            except Exception as e:
                logger.error(f"Error flushing LangFuse: {e}")


# Global tracer instance
tracer = LangFuseTracer()
