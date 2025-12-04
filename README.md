# 🌍 RAG Travel Assistant API

A RAG (Retrieval Augmented Generation) powered travel assistant built with **Qdrant** vector search, **Gemini 2.0 Flash** LLM, **LangGraph** orchestration, **FastAPI**, and **LangFuse** observability.

## 🌟 Features

- **In-Memory Vector Search**: Qdrant with hybrid search (semantic + keyword)
- **Smart Retrieval**: Country name boosting and RRF (Reciprocal Rank Fusion)
- **Gemini 2.0 Flash**: Fast, accurate responses with token tracking
- **LangGraph Workflow**: Multi-step orchestration (Input → Retrieval → Generation → Output)
- **LangFuse Tracing**: Complete observability for RAG operations
- **FastAPI Backend**: REST API with automatic documentation
- **Travel Knowledge Base**: 3 sample destinations (Japan, Singapore, Thailand)

## 📋 Prerequisites

- Python 3.13+
- Gemini API key
- (Optional) LangFuse account for tracing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional - LangFuse Tracing
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Server Configuration
APP_PORT=8001
```

### 3. Start the Server

```bash
python3 main.py
```

Or with uvicorn:

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8001
```

The API will be available at: `http://localhost:8001`

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

### Main Endpoint

**POST** `/rag-travel-assistant`

Query the travel assistant with any travel-related question.

**Request:**
```json
{
  "query": "What are visa requirements for Indians traveling to Japan?",
  "top_k": 3,
  "return_sources": true
}
```

**Response:**
```json
{
  "answer": "Visa Requirements for Indian Citizens Traveling to Japan:\n- Visa type: Tourist visa\n- Processing time: 5-7 working days\n- Stay duration: Up to 15 days\n- Documents needed: Valid passport (minimum 6 months validity), Completed visa application form, Recent passport-size photograph, Flight itinerary, Hotel reservation or invitation letter\n\nAdditional Information:\n- Best time to visit: March to May (Spring) and September to November (Autumn)\n- Top attractions: Mount Fuji, Tokyo Tower\n- Currency: Japanese Yen (JPY)",
  "query": "What are visa requirements for Indians traveling to Japan?",
  "sources_count": 1,
  "sources": [
    {
      "country": "Japan",
      "score": 0.48249990243107377
    }
  ],
  "generation_method": "Generated using retrieved documents + Gemini reasoning",
  "error": null
}
```

### Gemini Direct Endpoint (No Retrieval)

**POST** `/gemini-direct`

Generate answers using only Gemini's built-in knowledge without document retrieval.

**Request:**
```json
{
  "query": "What are visa requirements for Indians traveling to Japan?",
  "top_k": 1
}
```

**Response:**
```json
{
  "answer": "Here's a summary of visa requirements for Indian citizens traveling to Japan:\n\n*   **Visa Required:** Yes, Indian citizens generally need a visa...",
  "query": "What are visa requirements for Indians traveling to Japan?",
  "sources_count": 0,
  "sources": null,
  "generation_method": "Generated using Gemini reasoning only (no retrieval)",
  "error": null
}
```

## 🧪 Testing

### Using cURL - RAG with Retrieval

```bash
curl -X POST "http://localhost:8001/rag-travel-assistant" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are visa requirements for Indians traveling to Japan?",
    "top_k": 3,
    "return_sources": true
  }' | python3 -m json.tool
```

### Using cURL - Gemini Direct (No Retrieval)

```bash
curl -X POST "http://localhost:8001/gemini-direct" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are visa requirements for Indians traveling to Japan?",
    "top_k": 1
  }' | python3 -m json.tool
```

### Using Test Script

Run comprehensive API tests:

```bash
python3 test_main.py
```

This will:
- Test all endpoints with 11 different scenarios (10 RAG + 1 Gemini Direct)
- Save results to `output.json`
- Display summary with success rate and response times

## 🏗️ Architecture

### Project Structure

```
travel-assistant-rag-datadog/
├── main.py                 # FastAPI app with RAG + LangGraph workflow
├── test_main.py           # Comprehensive API tests (11 scenarios)
├── requirements.txt       # Python dependencies (pinned versions)
├── .env                   # Environment configuration (git-ignored)
├── .env.example           # Environment template
├── .gitignore            # Git ignore rules
├── output.json           # Test results (auto-generated)
└── README.md             # Documentation
```

### LangGraph Workflow

```
Input → Retrieval → Generation → Output
         ↓           ↓
       Error ←─────┘
```

**Nodes:**
1. **Input**: Validates query and prepares state
2. **Retrieval**: Hybrid search (semantic + keyword with country boosting)
3. **Generation**: Gemini LLM with retrieved context
4. **Error**: Handles errors gracefully
5. **Output**: Formats final response

### Hybrid Search (hybrid_retrieval)

- **Semantic Search**: Dense vectors (384 dimensions) using all-MiniLM-L6-v2
- **Keyword Search**: Sparse vectors with country name boosting (3x weight)
- **Fusion**: RRF (Reciprocal Rank Fusion) for optimal ranking
- **Filtering**: Minimum score threshold (0.45) to filter irrelevant results
- **Fallback**: Semantic-only search if hybrid query fails

## 📊 Observability

### LangFuse Tracing

When configured, all RAG operations are automatically traced:

**Traces Created:**
- `hybrid_retrieval` - Retrieval operations with query and results
- `llm_generation` - LLM calls with token usage and model metadata
- `rag_pipeline` - Complete end-to-end pipeline

**Metrics Tracked:**
- Token usage (input, output, total)
- Query latency
- Retrieval scores
- Error rates

**Setup:**
1. Add LangFuse credentials to `.env`
2. Restart server
3. View traces at https://us.cloud.langfuse.com

## 🔧 Configuration

Environment variables in `.env`:

```env
# API Keys
GEMINI_API_KEY=your_key_here
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Server
APP_HOST=0.0.0.0
APP_PORT=8001
LOG_LEVEL=INFO

# Gemini
GEMINI_MODEL=gemini-2.0-flash-exp
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=2048
```

## 📝 Sample Queries

Try these examples:

### Visa Requirements
- "What are visa requirements for Indians traveling to Japan?"
- "Do I need a visa for Singapore?"
- "Tell me about Thailand visa requirements"

### Travel Information
- "Best time to visit Japan for cherry blossoms?"
- "What are the top tourist attractions in Singapore?"
- "What currency is used in Thailand?"

### Documents & Processing
- "What documents do I need for Thailand travel?"
- "How long does it take to get a Singapore visa?"

### Comparisons
- "Compare visa requirements for Japan, Singapore and Thailand"

## 🚨 Troubleshooting

**Server won't start:**
- Check if port 8001 is available: `lsof -i :8001`
- Kill existing process: `lsof -i :8001 | tail -n +2 | awk '{print $2}' | xargs kill -9`

**No traces in LangFuse:**
- Verify credentials in `.env`
- Check server logs for "LangFuse initialized successfully"
- Ensure `langfuse` package is installed

**Poor search results:**
- Check query spelling
- Try more specific questions
- Review `output.json` for source scores

## 📦 Dependencies

Core packages:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `google-generativeai` - Gemini LLM
- `langgraph` - Workflow orchestration
- `qdrant-client` - Vector database
- `sentence-transformers` - Embeddings
- `langfuse` - Observability
- `requests` - Testing HTTP calls

See `requirements.txt` for complete list.

## 📄 License

MIT

## 👥 Author

**Chitti Vijay**

GenAI Developer Assignment - Week 2
