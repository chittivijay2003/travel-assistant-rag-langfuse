"""FastAPI routes for the travel assistant."""

import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse
from typing import Dict, Any

from app.models import QueryRequest, QueryResponse, HealthResponse
from app.graph.travel_assistant import get_travel_assistant_graph
from app.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global graph instance
_graph = None


def get_graph():
    """Get or initialize the travel assistant graph."""
    global _graph
    if _graph is None:
        from main import get_vector_store

        logger.info("Initializing travel assistant graph...")
        _graph = get_travel_assistant_graph(vector_store=get_vector_store())
    return _graph


@router.post(
    "/rag-travel-assistant",
    response_model=QueryResponse,
    summary="Query the RAG Travel Assistant",
    description="Send a travel-related query and receive an AI-generated answer based on retrieved destination information.",
    response_description="AI-generated answer with source documents",
)
async def rag_travel_assistant(request: QueryRequest) -> QueryResponse:
    """
    Execute RAG query for travel assistance.

    Args:
        request: Query request with user question

    Returns:
        Query response with answer and sources

    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Received query: '{request.query}' (top_k={request.top_k})")

        # Get graph instance
        graph = get_graph()

        # Execute query
        result = await graph.ainvoke(query=request.query, top_k=request.top_k)

        # Prepare response
        response = QueryResponse(
            answer=result.get("answer", ""),
            query=request.query,
            sources_count=result.get("sources_count", 0),
            sources=result.get("sources") if request.return_sources else None,
            metadata=result.get("metadata"),
            error=result.get("error"),
        )

        logger.info(f"Query processed successfully (sources: {response.sources_count})")
        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )


@router.post(
    "/rag-travel-assistant/text",
    response_class=PlainTextResponse,
    summary="Query the RAG Travel Assistant (Plain Text)",
    description="Send a travel-related query and receive an AI-generated answer in plain text format with proper line breaks.",
)
async def rag_travel_assistant_text(request: QueryRequest) -> PlainTextResponse:
    """
    Execute RAG query for travel assistance - returns plain text.

    Args:
        request: Query request with user question

    Returns:
        Plain text answer with proper line breaks

    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"Received text query: '{request.query}' (top_k={request.top_k})")

        # Get graph instance
        graph = get_graph()

        # Execute query
        result = await graph.ainvoke(query=request.query, top_k=request.top_k)

        # Get the answer
        answer = result.get("answer", "No answer generated")

        # Add sources information if requested
        if request.return_sources and result.get("sources"):
            answer += "\n\n" + "=" * 50
            answer += f"\n📚 Sources: {result.get('sources_count', 0)}\n"
            answer += "=" * 50 + "\n"
            for i, source in enumerate(result.get("sources", []), 1):
                answer += (
                    f"{i}. {source['country']} (Relevance: {source['score']:.2%})\n"
                )

        logger.info(f"Text query processed successfully")
        return PlainTextResponse(content=answer)

    except Exception as e:
        logger.error(f"Error processing text query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the service and its dependencies.",
    response_description="Health status information",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        Health status information
    """
    try:
        # Import here to avoid circular dependency
        from main import get_vector_store

        # Check Qdrant connection
        vector_store = get_vector_store()
        qdrant_connected = False

        try:
            collections = vector_store.client.get_collections()
            qdrant_connected = True
            logger.debug(
                f"Qdrant connected, collections: {len(collections.collections)}"
            )
        except Exception as e:
            logger.warning(f"Qdrant connection check failed: {e}")

        return HealthResponse(
            status="healthy",
            version=settings.dd_version,
            qdrant_connected=qdrant_connected,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@router.get(
    "/collection-info",
    summary="Get Collection Information",
    description="Get information about the Qdrant collection.",
    response_description="Collection information",
)
async def get_collection_info() -> Dict[str, Any]:
    """
    Get Qdrant collection information.

    Returns:
        Collection information
    """
    try:
        from main import get_vector_store

        vector_store = get_vector_store()
        info = vector_store.get_collection_info()

        logger.info(f"Collection info retrieved: {info}")
        return info

    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting collection info: {str(e)}",
        )


@router.post(
    "/load-data",
    summary="Load Travel Data",
    description="Load travel destination data into Qdrant vector database.",
    response_description="Load status",
)
async def load_data() -> Dict[str, Any]:
    """
    Load travel destination data into Qdrant.

    Returns:
        Load status information
    """
    try:
        from main import get_vector_store
        from pathlib import Path

        logger.info("Loading travel destination data...")

        # Initialize vector store
        vector_store = get_vector_store()

        # Create collection
        vector_store.create_collection(recreate=True)
        logger.info("Collection created successfully")

        # Load data
        data_path = Path(__file__).parent.parent.parent / "data" / "destinations.json"
        vector_store.load_data_from_json(str(data_path))
        logger.info("Data loaded successfully")

        # Get collection info
        info = vector_store.get_collection_info()

        return {
            "status": "success",
            "message": "Data loaded successfully",
            "collection_info": info,
        }

    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading data: {str(e)}",
        )
