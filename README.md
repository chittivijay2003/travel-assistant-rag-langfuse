# RAG Travel Assistant with LangGraph & Observability

A production-ready RAG (Retrieval Augmented Generation) powered travel assistant built with Qdrant vector search, Gemini LLM, LangGraph orchestration, FastAPI, and comprehensive observability through LangFuse and Datadog.

## 🌟 Features

- **Hybrid Search**: Combines semantic (dense vectors) and keyword (sparse vectors) search for superior retrieval accuracy
- **LangGraph Workflow**: Multi-step orchestrated pipeline with conditional routing
- **Gemini Integration**: Powered by Google's Gemini 2.0 for natural language generation
- **Comprehensive Observability**: 
  - LangFuse tracing for RAG pipeline visibility
  - Datadog APM for application performance monitoring
- **Travel Knowledge Base**: 15+ destinations with detailed information:
  - Visa requirements for Indian citizens
  - Processing times and required documents
  - Attractions and best time to visit
  - Climate, currency, and language information
- **FastAPI Backend**: Production-ready REST API with automatic documentation
- **Error Handling**: Comprehensive error handling and logging

## 📋 Prerequisites

- Python 3.13+
- Qdrant (running locally or cloud)
- Gemini API key
- (Optional) LangFuse account for tracing
- (Optional) Datadog account for APM

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Start Qdrant

Using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Or use Qdrant Cloud and update `QDRANT_URL` in `.env`

### 3. Configure Environment Variables

Edit `.env` file:
```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (for observability)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
DATADOG_API_KEY=your_datadog_api_key_here
DATADOG_APP_KEY=your_datadog_app_key_here

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
```

### 4. Load Travel Data into Qdrant

```bash
uv run python scripts/ingest_data.py
```

### 5. Start the API Server

```bash
uv run python main.py
```

The API will be available at: `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### POST `/api/v1/rag-travel-assistant`

Query the travel assistant with any travel-related question.

**Request:**
```json
{
  "query": "What are visa requirements for Indians traveling to Japan?",
  "top_k": 5,
  "return_sources": true
}
```

**Response:**
```json
{
  "answer": "Visa Requirements for Indian Citizens Traveling to Japan:\n\n- **Visa Required**: Yes, a tourist visa is required\n- **Visa Type**: Tourist visa\n- **Processing Time**: 5-7 working days\n- **Stay Duration**: Up to 15 days\n- **Validity**: 90 days from date of issue\n\n**Required Documents**:\n- Valid passport (minimum 6 months validity)\n- Completed visa application form\n- Recent passport-size photograph\n- Flight itinerary\n- Hotel reservation or invitation letter\n- Bank statements (last 6 months)\n- Employment proof or business registration\n- Income tax returns\n\n**Additional Information**:\nJapan offers a unique blend of ancient traditions and modern innovation, with popular destinations including Tokyo, Kyoto, Osaka, and Hiroshima. The best time to visit is during Spring (March-May) for cherry blossoms or Autumn (September-November) for pleasant weather.",
  "query": "What are visa requirements for Indians traveling to Japan?",
  "sources_count": 3,
  "sources": [
    {
      "country": "Japan",
      "title": "Japan Travel Guide",
      "score": 0.95,
      "id": "dest_001"
    }
  ]
}
```

#### GET `/api/v1/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "qdrant_connected": true
}
```

## 🏗️ Architecture

### Project Structure

```
travel-assistant-rag-datadog/
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI routes
│   ├── graph/
│   │   ├── state.py               # LangGraph state definition
│   │   ├── nodes.py               # LangGraph nodes (input, retrieval, generation)
│   │   └── travel_assistant.py   # Complete graph workflow
│   ├── observability/
│   │   ├── langfuse.py           # LangFuse tracing integration
│   │   └── datadog.py            # Datadog APM integration
│   ├── rag/
│   │   ├── vector_store.py       # Qdrant vector store management
│   │   ├── retriever.py          # Hybrid search implementation
│   │   └── pipeline.py           # RAG pipeline with Gemini
│   ├── config.py                 # Configuration management
│   ├── logging_config.py         # Logging setup
│   └── models.py                 # Pydantic models
├── data/
│   └── destinations.json         # Travel destination data
├── scripts/
│   └── ingest_data.py           # Data ingestion script
├── logs/                        # Application logs
├── main.py                      # FastAPI application
├── pyproject.toml              # Dependencies
└── README.md
```

### LangGraph Workflow

```
Input → Retrieval → Generation → Output
         ↓           ↓
       Error ←─────┘
```

**Nodes:**
1. **Input Node**: Validates query and prepares state
2. **Retrieval Node**: Performs hybrid search (semantic + keyword)
3. **Generation Node**: Generates answer using Gemini with retrieved context
4. **Error Node**: Handles errors gracefully
5. **Output Node**: Formats final response

### Hybrid Search Implementation

The retriever combines:
- **Dense Vectors**: Semantic similarity using sentence transformers (all-MiniLM-L6-v2)
- **Sparse Vectors**: Keyword matching using term frequency
- **Fusion**: Reciprocal Rank Fusion (RRF) to combine results

## 🔍 Sample Queries

Try these example queries:

1. "What are visa requirements for Indians traveling to Japan?"
2. "Best time to visit Switzerland?"
3. "Which countries offer visa-free entry for Indian citizens?"
4. "What documents do I need for a US tourist visa?"
5. "Tell me about attractions in Thailand"
6. "What is the climate like in Maldives?"

## 📊 Observability

### LangFuse Tracing

All RAG operations are traced:
- Retrieval steps (hybrid search)
- Generation steps (LLM calls)
- Complete pipeline execution

View traces in your LangFuse dashboard to:
- Monitor retrieval quality
- Track generation latency
- Debug issues
- Analyze user queries

### Datadog APM

Application metrics and traces:
- API endpoint performance
- Error rates
- Custom tags for queries
- Service dependencies

## 🧪 Testing

Test the API using curl:

```bash
curl -X POST "http://localhost:8000/api/v1/rag-travel-assistant" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are visa requirements for Indians traveling to Japan?",
    "top_k": 5
  }'
```

Or use the interactive Swagger UI at http://localhost:8000/docs

## 📝 Logging

Logs are written to:
- **Console**: INFO level and above
- **logs/app.log**: All logs with rotation (10MB, 5 backups)
- **logs/error.log**: ERROR level only

## 🔧 Configuration

Key settings in `app/config.py`:

```python
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=travel_destinations

# Gemini
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=2048

# RAG
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K_RESULTS=5

# API
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
```

## 🎯 Assignment Rubric Coverage

### 1. Qdrant Setup ✅
- Collection created with dense + sparse vectors
- 15 travel destination documents loaded
- Hybrid search configuration

### 2. Hybrid Search ✅
- Semantic search using dense vectors
- Keyword search using sparse vectors
- RRF fusion for result combination
- Accurate retrieval verified

### 3. RAG Pipeline ✅
- Retrieval + generation integrated
- Context formatting and prompt engineering
- High-quality travel-specific answers
- Error handling

### 4. LangFuse Integration ✅
- Tracing decorators on all RAG steps
- Retrieval traces
- Generation traces
- Pipeline-level traces
- Visible in LangFuse dashboard

### 5. FastAPI + LangGraph ✅
- `/rag-travel-assistant` endpoint functional
- LangGraph workflow with nodes and routing
- RAG node integrated
- Async support
- Complete API documentation

## 🚨 Troubleshooting

**Qdrant connection error:**
- Ensure Qdrant is running: `docker ps`
- Check QDRANT_URL in `.env`

**No results returned:**
- Run data ingestion: `uv run python scripts/ingest_data.py`
- Check collection: `GET /api/v1/collection-info`

**Gemini API errors:**
- Verify GEMINI_API_KEY is set correctly
- Check API quota and billing

## 📄 License

MIT

## 👥 Author

Developed as part of GenAI Developer Assignment
