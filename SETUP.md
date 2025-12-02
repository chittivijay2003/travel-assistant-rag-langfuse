# 🚀 Setup Guide - RAG Travel Assistant

This guide will help you set up and run the RAG Travel Assistant application.

## Prerequisites

- Python 3.13+
- Docker (for Qdrant)
- uv package manager (or pip)
- Gemini API key from Google AI Studio

## Step-by-Step Setup

### 1. Install UV Package Manager (if not installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone/Navigate to Project

```bash
cd travel-assistant-rag-datadog
```

### 3. Install Dependencies

```bash
uv sync
```

This will install all required packages including:
- FastAPI and Uvicorn
- LangChain and LangGraph
- Qdrant client
- Sentence Transformers
- Google Generative AI (Gemini)
- LangFuse
- Datadog tracing
- And more...

### 4. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

### 5. Configure Environment Variables

Edit the `.env` file and update:

```env
# REQUIRED: Add your Gemini API key
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Optional: LangFuse (for tracing)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key

# Optional: Datadog (for APM)
DATADOG_API_KEY=your_datadog_api_key
DATADOG_APP_KEY=your_datadog_app_key
```

**Note:** The application will work with just the Gemini API key. LangFuse and Datadog are optional for observability.

### 6. Start Qdrant Vector Database

#### Option A: Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

This will:
- Start Qdrant on port 6333
- Create a persistent volume for data
- Run in the background

#### Option B: Using Docker Run

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

#### Verify Qdrant is Running

```bash
curl http://localhost:6333/collections
```

You should see an empty collections list: `{"result":{"collections":[]}}`

### 7. Load Travel Data into Qdrant

```bash
uv run python scripts/ingest_data.py
```

Expected output:
```
INFO - Starting data ingestion process...
INFO - Creating Qdrant collection...
INFO - Collection created successfully: travel_destinations
INFO - Loading data from: .../data/destinations.json
INFO - Loaded 15 documents
INFO - Inserted 15 documents into travel_destinations
INFO - Data ingestion completed successfully!
```

### 8. Test the Setup (Optional but Recommended)

```bash
uv run python scripts/test_setup.py
```

This will:
- Check configuration
- Test Qdrant connection
- Run sample queries through the RAG pipeline
- Verify everything is working

### 9. Start the API Server

#### Option A: Using the Start Script

```bash
./start.sh
```

#### Option B: Manually

```bash
uv run python main.py
```

The API will start on `http://localhost:8000`

Expected output:
```
================================================================================
Starting RAG Travel Assistant API
Service: travel-assistant-rag
Environment: development
Version: 1.0.0
Qdrant URL: http://localhost:6333
Gemini Model: gemini-2.0-flash-exp
LangFuse Enabled: False
================================================================================
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## 🎯 Verify Installation

### 1. Check Health Endpoint

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "qdrant_connected": true
}
```

### 2. Check Collection Info

```bash
curl http://localhost:8000/api/v1/collection-info
```

Expected response:
```json
{
  "name": "travel_destinations",
  "points_count": 15,
  "vectors_count": 15,
  "status": "green"
}
```

### 3. Test RAG Query

```bash
curl -X POST "http://localhost:8000/api/v1/rag-travel-assistant" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are visa requirements for Indians traveling to Japan?",
    "top_k": 5
  }'
```

You should receive a detailed answer about Japanese visa requirements!

## 📖 Using the API

### Interactive Documentation

Visit http://localhost:8000/docs for Swagger UI where you can:
- See all available endpoints
- Try queries interactively
- View request/response schemas

### Example Queries

Try these queries:

1. **Visa Requirements**
   ```json
   {"query": "What are visa requirements for Indians traveling to Japan?"}
   ```

2. **Visa-Free Countries**
   ```json
   {"query": "Which countries offer visa-free entry for Indian citizens?"}
   ```

3. **Best Time to Visit**
   ```json
   {"query": "Best time to visit Switzerland?"}
   ```

4. **Attractions**
   ```json
   {"query": "Tell me about attractions in Thailand"}
   ```

5. **Documents**
   ```json
   {"query": "What documents do I need for a US tourist visa?"}
   ```

## 🔧 Optional: Enable Observability

### LangFuse Setup

1. Sign up at [LangFuse](https://langfuse.com/)
2. Create a new project
3. Get your public and secret keys
4. Update `.env`:
   ```env
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```
5. Restart the server

You'll now see traces in your LangFuse dashboard!

### Datadog Setup

1. Sign up at [Datadog](https://www.datadoghq.com/)
2. Get your API and App keys
3. Update `.env`:
   ```env
   DATADOG_API_KEY=...
   DATADOG_APP_KEY=...
   ```
4. Install Datadog agent (optional for full APM)
5. Restart the server

## 🐛 Troubleshooting

### Qdrant Connection Error

**Problem:** `Error connecting to Qdrant`

**Solution:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker-compose up -d

# Check logs
docker logs qdrant
```

### No Results from Queries

**Problem:** RAG returns "No relevant information found"

**Solution:**
```bash
# Re-run data ingestion
uv run python scripts/ingest_data.py

# Verify data loaded
curl http://localhost:8000/api/v1/collection-info
```

### Gemini API Errors

**Problem:** `Error: Invalid API key`

**Solution:**
1. Verify your API key in `.env`
2. Check key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Ensure no extra spaces or quotes around the key

**Problem:** `Quota exceeded`

**Solution:**
- Check your API quota
- Wait for quota reset
- Or upgrade your API plan

### Port Already in Use

**Problem:** `Port 8000 is already in use`

**Solution:**
```bash
# Option 1: Change port in .env
APP_PORT=8001

# Option 2: Kill process using port 8000
lsof -ti:8000 | xargs kill -9
```

## 📊 Performance Tips

### Faster Startup

The first request is slow because it loads the embedding model. Subsequent requests are fast.

### Optimize Top-K

- Use `top_k=3` for faster responses
- Use `top_k=5-7` for more comprehensive answers

### Batch Queries

For multiple queries, reuse the same server session instead of restarting.

## 🛑 Stopping the Application

### Stop API Server

Press `Ctrl+C` in the terminal

### Stop Qdrant

```bash
docker-compose down

# Or if using docker run
docker stop qdrant
docker rm qdrant
```

## 📚 Next Steps

1. Explore the [README.md](README.md) for architecture details
2. Check [scripts/example_usage.py](scripts/example_usage.py) for code examples
3. Customize the prompt in `app/rag/pipeline.py`
4. Add more destinations to `data/destinations.json`
5. Integrate into your own applications

## 🆘 Need Help?

- Check logs in `logs/app.log` and `logs/error.log`
- Run test suite: `uv run python scripts/test_setup.py`
- Review API docs: http://localhost:8000/docs

---

✨ **You're all set! Start querying your RAG Travel Assistant!** ✨
