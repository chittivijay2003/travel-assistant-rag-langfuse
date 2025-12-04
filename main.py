# GenAI Developer Assignment — RAG + LangGraph Travel Assistant
"""This notebook contains **instructions only**, with **TODO placeholders**.

## Objectives
- Set up Qdrant with travel destination data
- Implement hybrid search (semantic + keyword)
- Build RAG pipeline
- Integrate LangFuse for tracing
- Build FastAPI endpoint
- Use Gemini API keys
- Integrate into LangGraph Travel Assistant

---
"""

# Required imports
import os
import logging
from typing import List, Dict, Any, Optional, TypedDict
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseIndexParams,
    Prefetch,
    SparseVector,
)
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing_extensions import NotRequired

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangFuse (optional - will use fallback if not available)
try:
    from langfuse import Langfuse, observe

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

    def observe(*args, **kwargs):
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get port from environment
APP_PORT = int(os.getenv("APP_PORT", "8001"))

"""
## 📦 Task 1 — Set Up Qdrant With Travel Destination Data
Load sample destination documents into Qdrant.
"""

# TODO: Initialize Qdrant client and create collection
# TODO: Insert sample travel destination documents

# Initialize Qdrant client (in-memory for this example)
qdrant_client = QdrantClient(":memory:")
collection_name = "travel_destinations"

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vector_size = embedding_model.get_sentence_embedding_dimension()

logger.info(f"Embedding model loaded: dimension={vector_size}")

# Create collection with dense and sparse vectors for hybrid search
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
    },
)
logger.info(f"Collection '{collection_name}' created")

# Sample travel destination documents
sample_documents = [
    {
        "id": "dest_001",
        "country": "Japan",
        "title": "Japan Travel Guide",
        "content": "Japan is an island nation in East Asia, known for its rich culture, advanced technology, and beautiful landscapes. Popular cities include Tokyo, Kyoto, Osaka, and Hiroshima.",
        "visa_requirements": {
            "indian_citizens": {
                "visa_required": True,
                "visa_type": "Tourist visa",
                "processing_time": "5-7 working days",
                "stay_duration": "Up to 15 days",
                "documents_needed": [
                    "Valid passport (minimum 6 months validity)",
                    "Completed visa application form",
                    "Recent passport-size photograph",
                    "Flight itinerary",
                    "Hotel reservation or invitation letter",
                    "Bank statements (last 6 months)",
                ],
            }
        },
        "best_time_to_visit": "March to May (Spring) and September to November (Autumn)",
        "attractions": [
            "Mount Fuji",
            "Tokyo Tower",
            "Fushimi Inari Shrine",
            "Cherry blossom viewing",
        ],
        "currency": "Japanese Yen (JPY)",
    },
    {
        "id": "dest_002",
        "country": "Singapore",
        "title": "Singapore Travel Guide",
        "content": "Singapore is a modern city-state in Southeast Asia. Known for its cleanliness, efficiency, multicultural society, and world-class infrastructure.",
        "visa_requirements": {
            "indian_citizens": {
                "visa_required": True,
                "visa_type": "e-Visa",
                "processing_time": "2-3 working days",
                "stay_duration": "Up to 30 days",
                "documents_needed": [
                    "Valid passport",
                    "Photograph",
                    "Flight bookings",
                    "Hotel reservations",
                ],
            }
        },
        "best_time_to_visit": "February to April",
        "attractions": ["Marina Bay Sands", "Gardens by the Bay", "Sentosa Island"],
        "currency": "Singapore Dollar (SGD)",
    },
    {
        "id": "dest_003",
        "country": "Thailand",
        "title": "Thailand Travel Guide",
        "content": "Thailand, the Land of Smiles, is famous for tropical beaches, ornate temples, ancient ruins, and delicious cuisine. Bangkok, Chiang Mai, Phuket are popular destinations.",
        "visa_requirements": {
            "indian_citizens": {
                "visa_required": False,
                "visa_type": "Visa on Arrival",
                "processing_time": "On arrival",
                "stay_duration": "Up to 15 days",
                "documents_needed": [
                    "Valid passport",
                    "Return flight ticket",
                    "Hotel booking",
                ],
            }
        },
        "best_time_to_visit": "November to February",
        "attractions": ["Grand Palace", "Phi Phi Islands", "Floating markets"],
        "currency": "Thai Baht (THB)",
    },
]


# Helper function to create sparse vectors
def create_sparse_vector(text: str) -> Dict[str, Any]:
    """Create sparse vector from text for keyword matching."""
    country_keywords = ["japan", "singapore", "thailand", "dubai", "maldives"]
    words = text.lower().split()
    word_freq = {}

    for word in words:
        if len(word) > 2:
            word_hash = hash(word) % 100000
            boost = 3.0 if word in country_keywords else 1.0
            word_freq[word_hash] = word_freq.get(word_hash, 0) + boost

    return {"indices": list(word_freq.keys()), "values": list(word_freq.values())}


# Insert documents into Qdrant
points = []
for i, doc in enumerate(sample_documents):
    # Create text for embedding
    doc_text = f"{doc['title']}. {doc['content']}"

    # Generate dense vector
    dense_vector = embedding_model.encode(doc_text).tolist()

    # Generate sparse vector
    sparse_vec = create_sparse_vector(doc_text)

    # Create point
    point = PointStruct(
        id=i + 1,
        vector={
            "dense": dense_vector,
            "sparse": SparseVector(
                indices=sparse_vec["indices"], values=sparse_vec["values"]
            ),
        },
        payload=doc,
    )
    points.append(point)

# Upload points
qdrant_client.upsert(collection_name=collection_name, points=points)
logger.info(f"Inserted {len(points)} documents into Qdrant")

# Verify insertion
collection_info = qdrant_client.get_collection(collection_name)
logger.info(f"Collection stats: {collection_info.points_count} points")
"""
---
## 🔍 Task 2 — Implement Hybrid Search (Semantic + Keyword)
Combine vector similarity + keyword matching.
"""

# TODO: Implement hybrid search function


@observe(name="hybrid_retrieval", as_type="retriever")
def hybrid_retrieval(
    query: str, top_k: int = 5, min_score: float = 0.45
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic (dense) and keyword (sparse) search.

    Args:
        query: Search query
        top_k: Number of results to return
        min_score: Minimum relevance score threshold

    Returns:
        List of retrieved documents with scores
    """
    logger.info(f"Performing hybrid search for: '{query}' (top_k={top_k})")

    # Generate dense embedding for semantic search
    dense_vector = embedding_model.encode(query).tolist()

    # Generate sparse vector for keyword search
    sparse_vec = create_sparse_vector(query)

    try:
        # Perform hybrid search using query_points with RRF (Reciprocal Rank Fusion)
        results = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vec["indices"], values=sparse_vec["values"]
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query="rrf",  # Reciprocal Rank Fusion combines both searches
            limit=top_k,
        )

        # Format results and filter by minimum score
        documents = []
        for point in results.points:
            if point.score >= min_score:
                doc = {
                    "content": point.payload,
                    "score": point.score,
                    "id": point.id,
                }
                documents.append(doc)

        logger.info(
            f"Retrieved {len(documents)} documents (filtered by min_score={min_score})"
        )
        return documents

    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        # Fallback to semantic-only search
        logger.warning("Falling back to semantic-only search")
        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
        )

        documents = []
        for point in results.points:
            if point.score >= min_score:
                doc = {
                    "content": point.payload,
                    "score": point.score,
                    "id": point.id,
                }
                documents.append(doc)

        return documents


"""
---
"""
## 📚 Task 3 — Build RAG Pipeline
"""
Use hybrid search + Gemini LLM.
"""

# TODO: Implement RAG pipeline
# Step 1: Retrieve documents
# Step 2: Create final prompt
# Step 3: Call Gemini model

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")


def format_context(documents: List[Dict[str, Any]]) -> str:
    """Format retrieved documents into context string."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        content = doc["content"]
        country = content.get("country", "Unknown")

        # Format main content
        doc_text = f"[Source {i}: {country}]\n"
        doc_text += f"{content.get('content', '')}\n"

        # Format visa information
        if (
            "visa_requirements" in content
            and "indian_citizens" in content["visa_requirements"]
        ):
            visa = content["visa_requirements"]["indian_citizens"]
            doc_text += f"\nVisa Requirements for Indian Citizens:\n"
            doc_text += (
                f"- Visa Required: {'Yes' if visa.get('visa_required') else 'No'}\n"
            )
            doc_text += f"- Visa Type: {visa.get('visa_type', 'N/A')}\n"
            doc_text += f"- Processing Time: {visa.get('processing_time', 'N/A')}\n"
            doc_text += f"- Stay Duration: {visa.get('stay_duration', 'N/A')}\n"
            if "documents_needed" in visa:
                doc_text += (
                    f"- Documents Needed: {', '.join(visa['documents_needed'][:5])}\n"
                )

        # Format attractions
        if "attractions" in content and content["attractions"]:
            doc_text += f"\nTop Attractions: {', '.join(content['attractions'][:5])}\n"

        # Format other info
        if "best_time_to_visit" in content:
            doc_text += f"\nBest Time to Visit: {content['best_time_to_visit']}\n"
        if "currency" in content:
            doc_text += f"Currency: {content['currency']}\n"

        context_parts.append(doc_text)

    return "\n\n---\n\n".join(context_parts)


def create_prompt(query: str, context: str) -> str:
    """Create prompt for Gemini."""
    prompt = f"""You are a professional travel assistant providing accurate information.

Use the following context to answer the user's question.

Context:
{context}

User Question: {query}

IMPORTANT INSTRUCTIONS:
1. Answer directly and concisely
2. Use bullet points with dashes (-)
3. Format cleanly without extra symbols or emojis
4. Include only relevant information from the context

FORMAT EXAMPLE:

Visa Requirements for Indian Citizens Traveling to [Country]:
- Visa type: [Tourist visa/e-Visa/Visa on Arrival]
- Processing time: [X working days]
- Stay duration: [Up to X days]
- Documents needed: [List key documents]

Additional Information:
- Best time to visit: [Months]
- Top attractions: [List 2-3 places]
- Currency: [Currency name]

Keep answers factual and structured."""

    return prompt


@observe(
    name="llm_generation", as_type="generation", capture_input=True, capture_output=True
)
def generate_answer(prompt: str) -> Dict[str, Any]:
    """
    Generate answer using Gemini LLM.

    Args:
        prompt: Formatted prompt with query and context

    Returns:
        Dictionary with answer text and usage metadata
    """
    logger.info("Generating response with Gemini...")
    response = gemini_model.generate_content(
        prompt, generation_config={"temperature": 0.7, "max_output_tokens": 2048}
    )

    answer = response.text

    # Track token usage
    usage_metadata = {}
    if hasattr(response, "usage_metadata"):
        usage = response.usage_metadata
        usage_metadata = {
            "model": "gemini-2.0-flash-exp",
            "input_tokens": getattr(usage, "prompt_token_count", 0),
            "output_tokens": getattr(usage, "candidates_token_count", 0),
            "total_tokens": getattr(usage, "total_token_count", 0),
        }

        logger.info(
            f"Token usage - Input: {usage_metadata['input_tokens']}, "
            f"Output: {usage_metadata['output_tokens']}, "
            f"Total: {usage_metadata['total_tokens']}"
        )

    return {"answer": answer, "usage_metadata": usage_metadata}


@observe(name="rag_pipeline", as_type="chain")
def rag_pipeline(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Execute RAG query: retrieve documents and generate response.

    Args:
        query: User query
        top_k: Number of documents to retrieve

    Returns:
        Result dictionary with answer, sources, and metadata
    """
    try:
        logger.info(f"Executing RAG pipeline for: '{query}'")

        # Step 1: Retrieve documents using hybrid search
        documents = hybrid_retrieval(query, top_k=top_k)

        if not documents:
            logger.warning("No relevant documents found")
            return {
                "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different destination.",
                "query": query,
                "sources_count": 0,
                "sources": [],
            }

        # Step 2: Format context from retrieved documents
        context = format_context(documents)

        # Step 3: Create prompt
        prompt = create_prompt(query, context)

        # Step 4: Generate response with Gemini (traced separately)
        generation_result = generate_answer(prompt)
        answer = generation_result["answer"]
        usage_metadata = generation_result["usage_metadata"]

        logger.info("Response generated successfully")

        # Step 5: Prepare result
        result = {
            "answer": answer,
            "query": query,
            "sources_count": len(documents),
            "sources": [
                {
                    "country": doc["content"].get("country", "Unknown"),
                    "title": doc["content"].get("title", ""),
                    "score": doc.get("score", 0),
                    "id": doc.get("id"),
                }
                for doc in documents
            ],
            "generation_method": "Generated using retrieved documents + Gemini reasoning",
        }

        if usage_metadata:
            result["usage"] = usage_metadata

        return result

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {e}")
        return {
            "answer": f"I encountered an error while processing your request: {str(e)}",
            "query": query,
            "sources_count": 0,
            "sources": [],
            "error": str(e),
        }


"""
---
## 📊 Task 4 — Integrate LangFuse
Add tracing for retrieval + generation steps.
"""

# TODO: Initialize LangFuse
# TODO: Wrap RAG steps with tracing decorators

# Initialize LangFuse (if available)
langfuse_client = None
if LANGFUSE_AVAILABLE:
    try:
        LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
        LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            from langfuse import Langfuse

            langfuse_client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
            logger.info("LangFuse initialized successfully")
        else:
            logger.warning("LangFuse credentials not set, tracing disabled")
    except Exception as e:
        logger.warning(f"Could not initialize LangFuse: {e}")
else:
    logger.info("LangFuse not available - using fallback decorators")

# Note: The @observe decorators are already applied to hybrid_retrieval() and rag_pipeline()
# They will automatically trace these functions if LangFuse is available

"""
---
## 🔄 Task 5 — Add RAG Node to LangGraph Travel Assistant
Integrate RAG into the LangGraph workflow.
"""

# TODO: Add RAG node in LangGraph
# TODO: Update routing logic


# Define state for LangGraph
class TravelAssistantState(TypedDict):
    """State for travel assistant workflow."""

    query: str
    top_k: int
    documents: NotRequired[List[Dict[str, Any]]]
    answer: NotRequired[str]
    sources: NotRequired[List[Dict[str, Any]]]
    sources_count: NotRequired[int]
    generation_method: NotRequired[str]
    error: NotRequired[str]
    stage: NotRequired[str]


# Define workflow nodes
def input_node(state: TravelAssistantState) -> TravelAssistantState:
    """Process input and validate query."""
    logger.info(f"Input node: Processing query '{state['query']}'")
    state["stage"] = "input_processed"
    return state


def retrieval_node(state: TravelAssistantState) -> TravelAssistantState:
    """Retrieve relevant documents."""
    logger.info("Retrieval node: Searching for documents...")
    documents = hybrid_retrieval(state["query"], top_k=state.get("top_k", 5))
    state["documents"] = documents
    state["stage"] = "documents_retrieved"
    logger.info(f"Retrieved {len(documents)} documents")
    return state


def generation_node(state: TravelAssistantState) -> TravelAssistantState:
    """Generate answer from retrieved documents."""
    logger.info("Generation node: Generating answer...")
    documents = state.get("documents", [])

    if not documents:
        state["answer"] = "No relevant information found."
        state["sources"] = []
        state["sources_count"] = 0
    else:
        # Format and generate using traced function
        context = format_context(documents)
        prompt = create_prompt(state["query"], context)

        # Use the traced generate_answer function
        generation_result = generate_answer(prompt)
        state["answer"] = generation_result["answer"]
        state["sources"] = [
            {
                "country": doc["content"].get("country", "Unknown"),
                "score": doc.get("score", 0),
            }
            for doc in documents
        ]
        state["sources_count"] = len(documents)
        state["generation_method"] = (
            "Generated using retrieved documents + Gemini reasoning"
        )

    state["stage"] = "answer_generated"
    return state


def error_node(state: TravelAssistantState) -> TravelAssistantState:
    """Handle errors."""
    logger.error("Error node: Handling error")
    state["stage"] = "error"
    state["answer"] = state.get("error", "An error occurred")
    return state


def output_node(state: TravelAssistantState) -> TravelAssistantState:
    """Prepare final output."""
    logger.info("Output node: Preparing final output")
    state["stage"] = "completed"
    return state


# Routing functions
def route_after_input(state: TravelAssistantState) -> str:
    """Route after input processing."""
    if state.get("error"):
        return "error"
    return "retrieve"


def route_after_retrieval(state: TravelAssistantState) -> str:
    """Route after retrieval."""
    if state.get("error"):
        return "error"
    return "generate"


def route_after_generation(state: TravelAssistantState) -> str:
    """Route after generation."""
    if state.get("error"):
        return "error"
    return "complete"


# Build LangGraph workflow
workflow = StateGraph(TravelAssistantState)

# Add nodes
workflow.add_node("input", input_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)
workflow.add_node("error", error_node)
workflow.add_node("output", output_node)

# Set entry point
workflow.set_entry_point("input")

# Add conditional edges
workflow.add_conditional_edges(
    "input", route_after_input, {"retrieve": "retrieve", "error": "error"}
)

workflow.add_conditional_edges(
    "retrieve", route_after_retrieval, {"generate": "generate", "error": "error"}
)

workflow.add_conditional_edges(
    "generate", route_after_generation, {"complete": "output", "error": "error"}
)

# Add edges to end
workflow.add_edge("error", "output")
workflow.add_edge("output", END)

# Compile graph
travel_assistant_graph = workflow.compile()
logger.info("LangGraph workflow compiled successfully")

"""
---
## 🌐 Task 6 — Build FastAPI Endpoint `/rag-travel-assistant`
Expose full RAG workflow via API.
"""

# TODO: Build FastAPI endpoint
# TODO: Integrate LangGraph executor


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str
    top_k: int = 5
    return_sources: bool = True


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    answer: str
    query: str
    sources_count: int
    sources: Optional[List[Dict[str, Any]]] = None
    generation_method: Optional[str] = None
    error: Optional[str] = None


# Create FastAPI app
app = FastAPI(
    title="RAG Travel Assistant API",
    description="Travel assistant powered by RAG + LangGraph + Gemini",
    version="1.0.0",
)


@app.post("/rag-travel-assistant", response_model=QueryResponse)
async def rag_travel_assistant_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Execute RAG query for travel assistance.

    Args:
        request: Query request with user question

    Returns:
        Query response with answer and sources
    """
    try:
        logger.info(f"API: Received query: '{request.query}' (top_k={request.top_k})")

        # Execute via LangGraph
        state: TravelAssistantState = {"query": request.query, "top_k": request.top_k}

        result = travel_assistant_graph.invoke(state)

        # Prepare response
        response = QueryResponse(
            answer=result.get("answer", ""),
            query=request.query,
            sources_count=result.get("sources_count", 0),
            sources=result.get("sources") if request.return_sources else None,
            generation_method=result.get("generation_method"),
            error=result.get("error"),
        )

        logger.info(
            f"API: Query processed successfully (sources: {response.sources_count})"
        )
        return response

    except Exception as e:
        logger.error(f"API: Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/gemini-direct")
async def gemini_direct_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Direct Gemini generation without RAG retrieval.
    Uses only Gemini's built-in knowledge without document retrieval.

    Args:
        request: Query request with user question

    Returns:
        Query response with answer from Gemini only
    """
    try:
        logger.info(f"API: Direct Gemini query: '{request.query}'")

        # Create simple prompt without retrieved context
        prompt = f"""You are a travel assistant. Answer the following question based on your knowledge:

Question: {request.query}

Provide a concise, factual answer using bullet points with dashes (-).
Keep the response under 200 words."""

        # Generate answer directly from Gemini
        generation_result = generate_answer(prompt)

        response = QueryResponse(
            answer=generation_result["answer"],
            query=request.query,
            sources_count=0,
            sources=None,
            generation_method="Generated using Gemini reasoning only (no retrieval)",
            error=None,
        )

        logger.info("API: Direct Gemini query processed successfully")
        return response

    except Exception as e:
        logger.error(f"API: Error in direct Gemini query: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "qdrant": "in-memory",
            "gemini": "configured" if GEMINI_API_KEY else "not configured",
            "langfuse": "available" if LANGFUSE_AVAILABLE else "not available",
        },
    }


"""
---
# 📝 Sample Input
```
What are visa requirements for Indians traveling to Japan?
```

# ✅ Expected Output (High-Level)
```
Visa Requirements for Indian Citizens Traveling to Japan:
- Tourist visa required
- Passport, itinerary, bank statements, employment proof
- Processing time: 5–7 working days

(Generated using retrieved documents + Gemini reasoning)
```
---
"""
