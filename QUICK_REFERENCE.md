# ⚡ Quick Reference - RAG Travel Assistant

## 🚀 Quick Start (3 Steps)

```bash
# 1. Start Qdrant
docker-compose up -d

# 2. Load data
uv run python scripts/ingest_data.py

# 3. Start API
uv run python main.py
```

**Or use the convenience script:**
```bash
./start.sh
```

## 📍 Important URLs

- **API Root:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/api/v1/health
- **Qdrant UI:** http://localhost:6333/dashboard

## 🎯 Main Endpoint

**POST** `/api/v1/rag-travel-assistant`

```json
{
  "query": "Your travel question here",
  "top_k": 5,
  "return_sources": true
}
```

## 💡 Example Queries

1. "What are visa requirements for Indians traveling to Japan?"
2. "Which countries offer visa-free entry for Indian citizens?"
3. "Best time to visit Switzerland?"
4. "Tell me about attractions in Thailand"
5. "What documents do I need for a US tourist visa?"

## 📁 Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application |
| `app/api/routes.py` | API endpoints |
| `app/graph/travel_assistant.py` | LangGraph workflow |
| `app/rag/pipeline.py` | RAG pipeline |
| `app/rag/retriever.py` | Hybrid search |
| `app/rag/vector_store.py` | Qdrant operations |
| `data/destinations.json` | Travel data |
| `.env` | Configuration |

## 🔧 Common Commands

```bash
# Install dependencies
uv sync

# Load data into Qdrant
uv run python scripts/ingest_data.py

# Run tests
uv run python scripts/test_setup.py

# Run examples
uv run python scripts/example_usage.py

# Start server
uv run python main.py

# Start Qdrant
docker-compose up -d

# Stop Qdrant
docker-compose down

# View logs
tail -f logs/app.log
```

## 🔍 Health Checks

```bash
# API health
curl http://localhost:8000/api/v1/health

# Collection info
curl http://localhost:8000/api/v1/collection-info

# Qdrant collections
curl http://localhost:6333/collections
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | Change `APP_PORT` in `.env` |
| Qdrant not running | `docker-compose up -d` |
| No results | Run `scripts/ingest_data.py` |
| Gemini error | Check `GEMINI_API_KEY` in `.env` |

## 📊 Architecture

```
User Query
    ↓
FastAPI Endpoint
    ↓
LangGraph Workflow
    ↓
Input Node → Retrieval Node → Generation Node → Output Node
              (Hybrid Search)   (Gemini LLM)
              (Qdrant)
    ↓
Response with Sources
```

## 🔐 Required Environment Variables

**Minimum:**
```env
GEMINI_API_KEY=your_key_here
```

**Full Setup:**
```env
GEMINI_API_KEY=your_gemini_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
DATADOG_API_KEY=your_datadog_key
DATADOG_APP_KEY=your_datadog_app_key
```

## 📈 Performance Tips

- First request: ~3-5s (model loading)
- Later requests: ~1-2s
- Use `top_k=3` for faster responses
- Use `top_k=5-7` for better quality

## 🎯 Testing

```bash
# Quick test
curl -X POST http://localhost:8000/api/v1/rag-travel-assistant \
  -H "Content-Type: application/json" \
  -d '{"query": "Visa requirements for Japan?"}'

# Full test suite
uv run python scripts/test_setup.py
```

## 📦 Tech Stack

- **Framework:** FastAPI
- **Orchestration:** LangGraph
- **Vector DB:** Qdrant
- **LLM:** Google Gemini
- **Embeddings:** Sentence Transformers
- **Observability:** LangFuse + Datadog
- **Language:** Python 3.13

## 📚 Documentation

- `README.md` - Full documentation
- `SETUP.md` - Detailed setup guide
- `PROJECT_SUMMARY.md` - Implementation summary
- Swagger UI: http://localhost:8000/docs

## 🆘 Get Help

1. Check logs: `logs/app.log`, `logs/error.log`
2. Run tests: `scripts/test_setup.py`
3. Review setup: `SETUP.md`
4. Check API docs: http://localhost:8000/docs

---

**Quick Setup:** `docker-compose up -d && uv run python scripts/ingest_data.py && uv run python main.py`
