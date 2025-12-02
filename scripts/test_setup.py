"""Test script to verify the RAG Travel Assistant setup."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.graph.travel_assistant import get_travel_assistant_graph
from app.config import settings


async def test_rag_pipeline():
    """Test the complete RAG pipeline."""
    print("=" * 80)
    print("Testing RAG Travel Assistant")
    print("=" * 80)
    print()

    # Test queries
    test_queries = [
        "What are visa requirements for Indians traveling to Japan?",
        "Which countries offer visa-free entry for Indian citizens?",
        "Best time to visit Switzerland?",
        "Tell me about attractions in Thailand",
    ]

    try:
        # Initialize graph
        print("Initializing travel assistant graph...")
        graph = get_travel_assistant_graph()
        print("✅ Graph initialized successfully")
        print()

        # Test each query
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'=' * 80}")
            print(f"Test Query {i}/{len(test_queries)}")
            print(f"{'=' * 80}")
            print(f"Query: {query}")
            print()

            # Execute query
            result = await graph.ainvoke(query=query, top_k=3)

            # Display results
            print(f"Answer Preview (first 500 chars):")
            print("-" * 80)
            answer = result.get("answer", "")
            print(answer[:500])
            if len(answer) > 500:
                print("...")
            print()

            print(f"Sources: {result.get('sources_count', 0)}")
            if result.get("sources"):
                for source in result["sources"][:3]:
                    print(
                        f"  - {source.get('country')} (score: {source.get('score', 0):.3f})"
                    )

            print()

            if result.get("error"):
                print(f"⚠️  Error: {result['error']}")
            else:
                print("✅ Query processed successfully")

        print()
        print("=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_configuration():
    """Test configuration."""
    print("Testing Configuration...")
    print("-" * 80)

    configs = {
        "Gemini API Key": bool(
            settings.gemini_api_key
            and settings.gemini_api_key != "your_gemini_api_key_here"
        ),
        "Qdrant URL": settings.qdrant_url,
        "Embedding Model": settings.embedding_model,
        "Gemini Model": settings.gemini_model,
        "Top K Results": settings.top_k_results,
    }

    for key, value in configs.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")

    print()

    if not configs["Gemini API Key"]:
        print("⚠️  Warning: Gemini API key not configured!")
        print("   Set GEMINI_API_KEY in .env file")
        print()


def test_qdrant_connection():
    """Test Qdrant connection."""
    print("Testing Qdrant Connection...")
    print("-" * 80)

    try:
        from app.rag.vector_store import QdrantVectorStore

        vector_store = QdrantVectorStore()
        collections = vector_store.client.get_collections()

        print(f"✅ Connected to Qdrant at {settings.qdrant_url}")
        print(f"   Collections: {len(collections.collections)}")

        # Check if our collection exists
        collection_exists = any(
            col.name == settings.qdrant_collection_name
            for col in collections.collections
        )

        if collection_exists:
            info = vector_store.get_collection_info()
            print(f"✅ Collection '{settings.qdrant_collection_name}' found")
            print(f"   Points: {info.get('points_count', 0)}")
        else:
            print(f"⚠️  Collection '{settings.qdrant_collection_name}' not found")
            print("   Run: uv run python scripts/ingest_data.py")

        print()
        return True

    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("   Make sure Qdrant is running:")
        print("   docker-compose up -d")
        print()
        return False


async def main():
    """Main test function."""
    print()
    print("=" * 80)
    print("RAG Travel Assistant - Test Suite")
    print("=" * 80)
    print()

    # Test configuration
    test_configuration()

    # Test Qdrant connection
    if not test_qdrant_connection():
        print(
            "❌ Qdrant connection failed. Fix the connection before testing RAG pipeline."
        )
        return

    # Test RAG pipeline
    success = await test_rag_pipeline()

    if success:
        print()
        print("🎉 All tests passed!")
        print()
        print("Next steps:")
        print("  1. Start the API: uv run python main.py")
        print("  2. Visit: http://localhost:8000/docs")
        print("  3. Try the /rag-travel-assistant endpoint")
        print()
    else:
        print()
        print("❌ Some tests failed. Please check the errors above.")
        print()


if __name__ == "__main__":
    asyncio.run(main())
