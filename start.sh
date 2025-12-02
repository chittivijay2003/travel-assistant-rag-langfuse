#!/bin/bash

# RAG Travel Assistant - Quick Start Script

echo "================================================"
echo "RAG Travel Assistant - Quick Start"
echo "================================================"
echo ""

# Check if Qdrant is running
echo "Checking Qdrant connection..."
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "✅ Qdrant is running"
else
    echo "❌ Qdrant is not running"
    echo ""
    echo "Starting Qdrant with Docker..."
    docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
    echo "Waiting for Qdrant to start..."
    sleep 5
fi

echo ""
echo "Loading travel destination data into Qdrant..."
uv run python scripts/ingest_data.py

if [ $? -eq 0 ]; then
    echo "✅ Data loaded successfully"
else
    echo "❌ Failed to load data"
    exit 1
fi

echo ""
echo "================================================"
echo "Starting FastAPI server..."
echo "================================================"
echo ""
echo "API will be available at:"
echo "  - Main: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/api/v1/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uv run python main.py
