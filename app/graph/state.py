"""LangGraph state definition for travel assistant."""

from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class TravelAssistantState(TypedDict):
    """State for the travel assistant graph."""

    # Messages in the conversation
    messages: Annotated[List[Dict[str, str]], add_messages]

    # Current user query
    query: str

    # Retrieved documents from RAG
    retrieved_documents: Optional[List[Dict[str, Any]]]

    # Number of documents to retrieve
    top_k: int

    # Generated answer
    answer: Optional[str]

    # Source information
    sources: Optional[List[Dict[str, Any]]]

    # Error information if any
    error: Optional[str]

    # Metadata
    metadata: Optional[Dict[str, Any]]

    # Route decision (for conditional edges)
    next_action: Optional[str]
