"""LangGraph travel assistant workflow."""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.graph.state import TravelAssistantState
from app.graph.nodes import (
    TravelAssistantNodes,
    route_after_input,
    route_after_retrieval,
    route_after_generation,
)
from app.observability.langfuse import tracer as langfuse_tracer
from app.observability.datadog import dd_tracer

logger = logging.getLogger(__name__)


class TravelAssistantGraph:
    """LangGraph-based travel assistant with RAG."""

    def __init__(self, vector_store=None):
        """Initialize the graph.

        Args:
            vector_store: Optional QdrantVectorStore instance to reuse
        """
        logger.info("Initializing TravelAssistantGraph...")

        # Initialize nodes with shared vector store
        self.nodes = TravelAssistantNodes(vector_store=vector_store)

        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

        logger.info("TravelAssistantGraph initialized successfully")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        logger.info("Building LangGraph workflow...")

        # Create graph
        workflow = StateGraph(TravelAssistantState)

        # Add nodes
        workflow.add_node("input", self.nodes.input_node)
        workflow.add_node("retrieve", self.nodes.retrieval_node)
        workflow.add_node("generate", self.nodes.generation_node)
        workflow.add_node("error", self.nodes.error_node)
        workflow.add_node("output", self.nodes.output_node)

        # Set entry point
        workflow.set_entry_point("input")

        # Add conditional edges
        workflow.add_conditional_edges(
            "input", route_after_input, {"retrieve": "retrieve", "error": "error"}
        )

        workflow.add_conditional_edges(
            "retrieve",
            route_after_retrieval,
            {"generate": "generate", "error": "error"},
        )

        workflow.add_conditional_edges(
            "generate", route_after_generation, {"complete": "output", "error": "error"}
        )

        # Add edges to end
        workflow.add_edge("error", "output")
        workflow.add_edge("output", END)

        logger.info("LangGraph workflow built successfully")
        return workflow

    @langfuse_tracer.trace_rag_pipeline
    @dd_tracer.trace_rag_pipeline
    def invoke(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the travel assistant workflow.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Result dictionary with answer and metadata
        """
        try:
            logger.info(f"Invoking graph with query: '{query}'")

            # Prepare initial state
            initial_state: TravelAssistantState = {
                "query": query,
                "top_k": top_k,
                "messages": [],
                "retrieved_documents": None,
                "answer": None,
                "sources": None,
                "error": None,
                "metadata": {},
                "next_action": None,
            }

            # Execute graph
            result = self.app.invoke(initial_state)

            # Format response
            response = {
                "answer": result.get("answer", ""),
                "query": query,
                "sources": result.get("sources", []),
                "sources_count": len(result.get("sources", [])),
                "metadata": result.get("metadata", {}),
            }

            if result.get("error"):
                response["error"] = result["error"]

            logger.info("Graph execution completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error executing graph: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "query": query,
                "sources": [],
                "sources_count": 0,
                "error": str(e),
                "metadata": {"stage": "error"},
            }

    async def ainvoke(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Asynchronously execute the travel assistant workflow.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Result dictionary with answer and metadata
        """
        try:
            logger.info(f"Async invoking graph with query: '{query}'")

            # Prepare initial state
            initial_state: TravelAssistantState = {
                "query": query,
                "top_k": top_k,
                "messages": [],
                "retrieved_documents": None,
                "answer": None,
                "sources": None,
                "error": None,
                "metadata": {},
                "next_action": None,
            }

            # Execute graph asynchronously
            result = await self.app.ainvoke(initial_state)

            # Format response
            response = {
                "answer": result.get("answer", ""),
                "query": query,
                "sources": result.get("sources", []),
                "sources_count": len(result.get("sources", [])),
                "metadata": result.get("metadata", {}),
            }

            if result.get("error"):
                response["error"] = result["error"]

            logger.info("Async graph execution completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error in async graph execution: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "query": query,
                "sources": [],
                "sources_count": 0,
                "error": str(e),
                "metadata": {"stage": "error"},
            }


# Global graph instance
_graph_instance = None


def get_travel_assistant_graph(vector_store=None) -> TravelAssistantGraph:
    """
    Get or create the global travel assistant graph instance.

    Args:
        vector_store: Optional QdrantVectorStore instance to reuse

    Returns:
        TravelAssistantGraph instance
    """
    global _graph_instance

    if _graph_instance is None:
        _graph_instance = TravelAssistantGraph(vector_store=vector_store)

    return _graph_instance
