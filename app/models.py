"""Pydantic models for API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG query."""

    query: str = Field(
        ..., description="User query about travel destinations", min_length=1
    )
    top_k: int = Field(
        default=5, description="Number of documents to retrieve", ge=1, le=20
    )
    return_sources: bool = Field(
        default=True, description="Whether to return source documents"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are visa requirements for Indians traveling to Japan?",
                "top_k": 5,
                "return_sources": True,
            }
        }


class Source(BaseModel):
    """Source document information."""

    country: str = Field(..., description="Destination country")
    title: str = Field(default="", description="Document title")
    score: float = Field(..., description="Relevance score")
    id: Optional[Any] = Field(None, description="Document ID")


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    sources_count: int = Field(..., description="Number of source documents used")
    sources: Optional[List[Source]] = Field(None, description="Source documents")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Visa Requirements for Indian Citizens Traveling to Japan:\n- Tourist visa required\n- Processing time: 5-7 working days...",
                "query": "What are visa requirements for Indians traveling to Japan?",
                "sources_count": 3,
                "sources": [
                    {
                        "country": "Japan",
                        "title": "Japan Travel Guide",
                        "score": 0.95,
                        "id": "dest_001",
                    }
                ],
                "metadata": {"stage": "completed"},
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    qdrant_connected: bool = Field(..., description="Qdrant connection status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "qdrant_connected": True,
            }
        }
