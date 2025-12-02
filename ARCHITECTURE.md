# Architecture Diagrams

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Client                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTP POST
                            │ /rag-travel-assistant
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Routes                            │  │
│  │  - /rag-travel-assistant  (Main endpoint)                │  │
│  │  - /health                (Health check)                 │  │
│  │  - /collection-info       (Qdrant info)                  │  │
│  └────────────────────────┬─────────────────────────────────┘  │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             │ Invoke Graph
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                            │
│                                                                   │
│  ┌──────────┐     ┌────────────┐     ┌────────────┐            │
│  │  Input   │────▶│ Retrieval  │────▶│ Generation │            │
│  │  Node    │     │   Node     │     │    Node    │            │
│  └──────────┘     └────────────┘     └────────────┘            │
│       │                 │                   │                    │
│       │                 │                   │                    │
│       ▼                 ▼                   ▼                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │              Error Node (if needed)              │            │
│  └─────────────────────────────────────────────────┘            │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────┐            │
│  │                 Output Node                      │            │
│  └─────────────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Qdrant     │    │   Gemini     │    │ Observability│
│  Vector DB   │    │     LLM      │    │  (LangFuse   │
│              │    │              │    │   Datadog)   │
│ - Hybrid     │    │ - Generate   │    │              │
│   Search     │    │   Answers    │    │ - Tracing    │
│ - Dense      │    │ - Context    │    │ - Metrics    │
│   Vectors    │    │   Aware      │    │ - Logging    │
│ - Sparse     │    │              │    │              │
│   Vectors    │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

## LangGraph Workflow Detail

```
                    ┌─────────────────┐
                    │  Start (Entry)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Input Node    │
                    │                 │
                    │ - Validate      │
                    │ - Set defaults  │
                    │ - Prepare state │
                    └────────┬────────┘
                             │
                      route_after_input
                             │
                   ┌─────────┴─────────┐
                   │                   │
                   ▼                   ▼
          ┌─────────────────┐   ┌─────────────┐
          │  Retrieval Node │   │ Error Node  │
          │                 │   │             │
          │ - Hybrid Search │   │ - Handle    │
          │ - Get top_k     │   │   errors    │
          │ - Qdrant query  │   │ - Format    │
          └────────┬────────┘   │   message   │
                   │            └──────┬──────┘
            route_after_retrieval      │
                   │                   │
         ┌─────────┴─────────┐         │
         │                   │         │
         ▼                   ▼         │
┌─────────────────┐   ┌─────────────┐ │
│ Generation Node │   │ Error Node  │ │
│                 │   │             │ │
│ - Format context│   └──────┬──────┘ │
│ - Create prompt │          │        │
│ - Call Gemini   │          │        │
│ - Get answer    │          │        │
└────────┬────────┘          │        │
         │                   │        │
  route_after_generation     │        │
         │                   │        │
         └─────────┬─────────┘        │
                   │                  │
                   ▼                  │
          ┌─────────────────┐         │
          │   Output Node   │◀────────┘
          │                 │
          │ - Format final  │
          │   response      │
          │ - Add metadata  │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   END (Return)  │
          └─────────────────┘
```

## Hybrid Search Flow

```
User Query: "What are visa requirements for Japan?"
    │
    ▼
┌───────────────────────────────────────────┐
│         Hybrid Retriever                  │
│                                           │
│  1. Generate Dense Vector                │
│     ┌─────────────────────────────┐      │
│     │ Sentence Transformer        │      │
│     │ (all-MiniLM-L6-v2)          │      │
│     │                             │      │
│     │ Query → [0.12, -0.34, ...]  │      │
│     │         (384 dimensions)    │      │
│     └─────────────────────────────┘      │
│                                           │
│  2. Generate Sparse Vector               │
│     ┌─────────────────────────────┐      │
│     │ Term Frequency              │      │
│     │                             │      │
│     │ "visa" → hash → index       │      │
│     │ "requirements" → hash → idx │      │
│     │ "japan" → hash → index      │      │
│     └─────────────────────────────┘      │
│                                           │
│  3. Query Qdrant with Prefetch           │
│     ┌─────────────────────────────┐      │
│     │ Semantic Search (dense)     │      │
│     │ Keyword Search (sparse)     │      │
│     │ RRF Fusion                  │      │
│     └─────────────────────────────┘      │
└────────────────┬──────────────────────────┘
                 │
                 ▼
        Retrieved Documents
        [
          {Japan doc, score: 0.95},
          {UAE doc, score: 0.78},
          {UK doc, score: 0.72}
        ]
```

## RAG Pipeline Flow

```
Query Input
    │
    ▼
┌─────────────────────────────────────────────┐
│              RAG Pipeline                    │
│                                              │
│  Step 1: Retrieve                            │
│  ┌────────────────────────────────────┐     │
│  │  Hybrid Search                     │     │
│  │  ↓                                 │     │
│  │  Top K Documents (default: 5)     │     │
│  └────────────────────────────────────┘     │
│                  │                           │
│                  ▼                           │
│  Step 2: Format Context                     │
│  ┌────────────────────────────────────┐     │
│  │ Extract from each doc:             │     │
│  │ - Country, content                 │     │
│  │ - Visa requirements                │     │
│  │ - Attractions                      │     │
│  │ - Best time to visit               │     │
│  │ - Climate, currency, language      │     │
│  │                                    │     │
│  │ Format as structured text          │     │
│  └────────────────────────────────────┘     │
│                  │                           │
│                  ▼                           │
│  Step 3: Create Prompt                      │
│  ┌────────────────────────────────────┐     │
│  │ System: "You are a travel expert"  │     │
│  │ Context: [retrieved docs]          │     │
│  │ Query: "What are visa..."          │     │
│  │ Instructions: "Provide details..." │     │
│  └────────────────────────────────────┘     │
│                  │                           │
│                  ▼                           │
│  Step 4: Generate with Gemini               │
│  ┌────────────────────────────────────┐     │
│  │ Model: gemini-2.0-flash-exp        │     │
│  │ Temperature: 0.7                   │     │
│  │ Max Tokens: 2048                   │     │
│  │                                    │     │
│  │ Generate Response                  │     │
│  └────────────────────────────────────┘     │
│                  │                           │
└──────────────────┼───────────────────────────┘
                   │
                   ▼
           Answer + Sources
```

## Data Flow

```
1. Data Ingestion (One-time)
   
   destinations.json
        │
        ▼
   Load Documents
        │
        ▼
   For each document:
        │
        ├─▶ Generate Dense Vector (Sentence Transformer)
        │
        └─▶ Generate Sparse Vector (Term Frequency)
        │
        ▼
   Insert into Qdrant
        │
        ▼
   Collection Ready


2. Query Processing (Runtime)
   
   User Query
        │
        ▼
   LangGraph Entry
        │
        ▼
   Retrieval Node
        │
        ├─▶ Hybrid Search in Qdrant
        │   │
        │   ├─▶ Semantic Match (Dense)
        │   │
        │   └─▶ Keyword Match (Sparse)
        │   │
        │   └─▶ RRF Fusion
        │
        ▼
   Retrieved Documents
        │
        ▼
   Generation Node
        │
        ├─▶ Format Context
        │
        ├─▶ Create Prompt
        │
        └─▶ Call Gemini API
        │
        ▼
   Generated Answer
        │
        ▼
   Output Node
        │
        ▼
   Return Response
        │
        ▼
   User receives answer + sources
```

## Observability Flow

```
Every Request
     │
     ▼
┌────────────────────────────────────┐
│     LangFuse Tracing               │
│                                    │
│  Trace ID: xyz123                  │
│                                    │
│  ┌──────────────────────────┐     │
│  │ Span: RAG Pipeline       │     │
│  │  ├─ Input: query         │     │
│  │  │                       │     │
│  │  ├─ Span: Retrieval      │     │
│  │  │   ├─ Input: query     │     │
│  │  │   ├─ Output: 5 docs   │     │
│  │  │   └─ Duration: 200ms  │     │
│  │  │                       │     │
│  │  ├─ Span: Generation     │     │
│  │  │   ├─ Input: prompt    │     │
│  │  │   ├─ Output: answer   │     │
│  │  │   └─ Duration: 1500ms │     │
│  │  │                       │     │
│  │  └─ Output: response     │     │
│  └──────────────────────────┘     │
└────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────┐
│     Datadog APM                    │
│                                    │
│  Service: travel-assistant-rag    │
│  Environment: development          │
│                                    │
│  ┌──────────────────────────┐     │
│  │ Trace                    │     │
│  │  ├─ fastapi.request      │     │
│  │  ├─ hybrid_search        │     │
│  │  ├─ llm_generation       │     │
│  │  └─ rag_pipeline         │     │
│  │                          │     │
│  │ Tags:                    │     │
│  │  - query: "visa..."      │     │
│  │  - status: success       │     │
│  │  - sources: 5            │     │
│  └──────────────────────────┘     │
└────────────────────────────────────┘
```

---

**Note:** These are ASCII diagrams. For production documentation, consider using tools like:
- Draw.io / Lucidchart for professional diagrams
- Mermaid for markdown-based diagrams
- PlantUML for code-based diagrams
