"""Datadog APM and tracing integration."""

import logging
from typing import Any, Callable, Dict, Optional
from functools import wraps
import os

logger = logging.getLogger(__name__)

# Datadog tracing
try:
    from ddtrace import tracer, patch

    DATADOG_AVAILABLE = True
except ImportError:
    DATADOG_AVAILABLE = False
    logger.warning("ddtrace not available, Datadog tracing disabled")


class DatadogTracer:
    """Datadog APM tracing wrapper."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(DatadogTracer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Datadog tracer."""
        if not self._initialized and DATADOG_AVAILABLE:
            try:
                # Patch supported libraries
                patch(fastapi=True, httpx=True)

                # Configure tracer
                tracer.configure(
                    hostname=os.getenv("DD_AGENT_HOST", "localhost"),
                    port=int(os.getenv("DD_TRACE_AGENT_PORT", "8126")),
                )

                self._initialized = True
                logger.info("Datadog tracer initialized")

            except Exception as e:
                logger.error(f"Failed to initialize Datadog tracer: {e}")
                self._initialized = False

    @property
    def is_enabled(self) -> bool:
        """Check if Datadog is enabled."""
        return DATADOG_AVAILABLE and self._initialized

    def trace_function(
        self,
        service_name: str = "travel-assistant-rag",
        span_name: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to trace functions with Datadog.

        Args:
            service_name: Service name for the trace
            span_name: Custom span name (defaults to function name)
            resource: Resource name

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            if not self.is_enabled:
                return func

            @wraps(func)
            def wrapper(*args, **kwargs):
                span_name_final = span_name or func.__name__
                resource_final = resource or func.__name__

                with tracer.trace(
                    span_name_final, service=service_name, resource=resource_final
                ) as span:
                    try:
                        # Add custom tags
                        span.set_tag("function", func.__name__)

                        result = func(*args, **kwargs)

                        # Tag success
                        span.set_tag("status", "success")

                        return result

                    except Exception as e:
                        # Tag error
                        span.set_tag("status", "error")
                        span.set_tag("error.message", str(e))
                        span.set_tag("error.type", type(e).__name__)
                        raise

            return wrapper

        return decorator

    def trace_retrieval(self, func: Callable) -> Callable:
        """
        Trace retrieval operations.

        Args:
            func: Function to trace

        Returns:
            Wrapped function
        """
        if not self.is_enabled:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace(
                "hybrid_search",
                service="travel-assistant-rag",
                resource="vector_retrieval",
            ) as span:
                try:
                    query = kwargs.get("query", args[0] if args else "unknown")
                    top_k = kwargs.get("top_k", 5)

                    span.set_tag("query", query[:100])  # Truncate long queries
                    span.set_tag("top_k", top_k)
                    span.set_tag("operation", "hybrid_search")

                    result = func(*args, **kwargs)

                    span.set_tag("num_results", len(result) if result else 0)
                    span.set_tag("status", "success")

                    return result

                except Exception as e:
                    span.set_tag("status", "error")
                    span.set_tag("error.message", str(e))
                    raise

        return wrapper

    def trace_generation(self, func: Callable) -> Callable:
        """
        Trace LLM generation operations.

        Args:
            func: Function to trace

        Returns:
            Wrapped function
        """
        if not self.is_enabled:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace(
                "llm_generation",
                service="travel-assistant-rag",
                resource="gemini_generate",
            ) as span:
                try:
                    query = kwargs.get("query", "unknown")

                    span.set_tag("query", query[:100])
                    span.set_tag("model", "gemini")
                    span.set_tag("operation", "generation")

                    result = func(*args, **kwargs)

                    if isinstance(result, dict):
                        span.set_tag("answer_length", len(result.get("answer", "")))
                        span.set_tag("sources_count", result.get("sources_count", 0))

                    span.set_tag("status", "success")

                    return result

                except Exception as e:
                    span.set_tag("status", "error")
                    span.set_tag("error.message", str(e))
                    raise

        return wrapper

    def trace_rag_pipeline(self, func: Callable) -> Callable:
        """
        Trace complete RAG pipeline.

        Args:
            func: Function to trace

        Returns:
            Wrapped function
        """
        if not self.is_enabled:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace(
                "rag_pipeline",
                service="travel-assistant-rag",
                resource="travel_assistant_query",
            ) as span:
                try:
                    query = kwargs.get("query", args[0] if args else "unknown")

                    span.set_tag("query", query[:100])
                    span.set_tag("pipeline", "travel_assistant")
                    span.set_tag("operation", "rag_query")

                    result = func(*args, **kwargs)

                    if isinstance(result, dict):
                        span.set_tag("has_error", "error" in result)
                        span.set_tag("sources_count", result.get("sources_count", 0))

                    span.set_tag("status", "success")

                    return result

                except Exception as e:
                    span.set_tag("status", "error")
                    span.set_tag("error.message", str(e))
                    span.set_tag("error.type", type(e).__name__)
                    raise

        return wrapper

    def add_tags(self, tags: Dict[str, Any]) -> None:
        """
        Add custom tags to current span.

        Args:
            tags: Dictionary of tags to add
        """
        if not self.is_enabled:
            return

        try:
            current_span = tracer.current_span()
            if current_span:
                for key, value in tags.items():
                    current_span.set_tag(key, value)
        except Exception as e:
            logger.error(f"Error adding tags: {e}")


# Global tracer instance
dd_tracer = DatadogTracer()
