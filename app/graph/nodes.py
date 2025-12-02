"""LangGraph nodes for travel assistant workflow."""

import logging
from typing import Dict, Any

from app.graph.state import TravelAssistantState
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import QdrantVectorStore
from app.observability.langfuse import tracer as langfuse_tracer
from app.observability.datadog import dd_tracer

logger = logging.getLogger(__name__)


class TravelAssistantNodes:
    """Nodes for the travel assistant LangGraph."""

    def __init__(self, vector_store: QdrantVectorStore = None):
        """Initialize nodes with RAG components.

        Args:
            vector_store: Optional QdrantVectorStore instance to reuse
        """
        logger.info("Initializing TravelAssistantNodes...")

        # Initialize RAG components
        if vector_store is None:
            self.vector_store = QdrantVectorStore()
        else:
            self.vector_store = vector_store

        self.retriever = HybridRetriever(self.vector_store)
        self.rag_pipeline = RAGPipeline(self.retriever)

        logger.info("TravelAssistantNodes initialized successfully")

    def input_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """
        Process input and prepare for RAG.

        Args:
            state: Current graph state

        Returns:
            Updated state
        """
        logger.info(f"Input node processing query: {state.get('query', 'N/A')}")

        # Set defaults
        if "top_k" not in state or state["top_k"] is None:
            state["top_k"] = 5

        if "metadata" not in state:
            state["metadata"] = {}

        state["metadata"]["stage"] = "input_processed"
        state["next_action"] = "retrieve"

        return state

    @langfuse_tracer.trace_retrieval
    @dd_tracer.trace_retrieval
    def retrieval_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """
        Retrieve relevant documents using hybrid search.

        Args:
            state: Current graph state

        Returns:
            Updated state with retrieved documents
        """
        try:
            query = state["query"]
            top_k = state.get("top_k", 5)

            logger.info(f"Retrieval node searching for: '{query}' (top_k={top_k})")

            # Perform hybrid search
            documents = self.retriever.hybrid_search(query=query, top_k=top_k)

            state["retrieved_documents"] = documents
            state["metadata"]["stage"] = "documents_retrieved"
            state["metadata"]["num_documents"] = len(documents)
            state["next_action"] = "generate"

            logger.info(f"Retrieved {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error in retrieval node: {e}")
            state["error"] = f"Retrieval error: {str(e)}"
            state["next_action"] = "error"

        return state

    @langfuse_tracer.trace_generation
    @dd_tracer.trace_generation
    def generation_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """
        Generate answer using RAG pipeline.

        Args:
            state: Current graph state

        Returns:
            Updated state with generated answer
        """
        try:
            query = state["query"]
            top_k = state.get("top_k", 5)

            logger.info(f"Generation node processing query: '{query}'")

            # Generate answer using RAG pipeline
            result = self.rag_pipeline.query(
                query=query, top_k=top_k, return_sources=True
            )

            state["answer"] = result.get("answer", "")
            state["sources"] = result.get("sources", [])
            state["metadata"]["stage"] = "answer_generated"
            state["metadata"]["sources_count"] = result.get("sources_count", 0)

            # Add to messages
            if "messages" not in state:
                state["messages"] = []

            state["messages"].append({"role": "user", "content": query})
            state["messages"].append({"role": "assistant", "content": state["answer"]})

            state["next_action"] = "complete"

            logger.info("Answer generated successfully")

        except Exception as e:
            logger.error(f"Error in generation node: {e}")
            state["error"] = f"Generation error: {str(e)}"
            state["next_action"] = "error"

        return state

    def error_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """
        Handle errors in the workflow.

        Args:
            state: Current graph state

        Returns:
            Updated state with error information
        """
        error_msg = state.get("error", "Unknown error occurred")
        logger.error(f"Error node: {error_msg}")

        state["answer"] = f"I apologize, but I encountered an error: {error_msg}"
        state["metadata"]["stage"] = "error"
        state["next_action"] = "complete"

        return state

    def output_node(self, state: TravelAssistantState) -> TravelAssistantState:
        """
        Prepare final output.

        Args:
            state: Current graph state

        Returns:
            Updated state with final output
        """
        logger.info("Output node preparing final response")

        state["metadata"]["stage"] = "completed"

        # Ensure answer exists
        if not state.get("answer"):
            state["answer"] = "I couldn't generate a response. Please try again."

        return state


def route_after_input(state: TravelAssistantState) -> str:
    """
    Route after input processing.

    Args:
        state: Current graph state

    Returns:
        Next node name
    """
    next_action = state.get("next_action", "retrieve")
    logger.debug(f"Routing from input to: {next_action}")
    return next_action


def route_after_retrieval(state: TravelAssistantState) -> str:
    """
    Route after retrieval.

    Args:
        state: Current graph state

    Returns:
        Next node name
    """
    next_action = state.get("next_action", "generate")
    logger.debug(f"Routing from retrieval to: {next_action}")
    return next_action


def route_after_generation(state: TravelAssistantState) -> str:
    """
    Route after generation.

    Args:
        state: Current graph state

    Returns:
        Next node name
    """
    next_action = state.get("next_action", "complete")
    logger.debug(f"Routing from generation to: {next_action}")
    return next_action
