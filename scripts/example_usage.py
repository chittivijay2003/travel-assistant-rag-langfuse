"""Example usage script for the RAG Travel Assistant."""

import asyncio
import json
from app.graph.travel_assistant import get_travel_assistant_graph


async def example_queries():
    """Run example queries through the RAG Travel Assistant."""

    # Initialize the graph
    graph = get_travel_assistant_graph()

    # Example queries
    queries = [
        "What are visa requirements for Indians traveling to Japan?",
        "Which countries offer visa-free entry for Indian citizens?",
        "Best time to visit Switzerland?",
        "Tell me about popular attractions in Bali",
        "What documents do I need for a US tourist visa?",
    ]

    print("RAG Travel Assistant - Example Queries")
    print("=" * 80)
    print()

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)

        # Execute query
        result = await graph.ainvoke(query=query, top_k=3)

        # Display answer
        print("\nAnswer:")
        print(result["answer"][:500])
        if len(result["answer"]) > 500:
            print("...\n(truncated)")

        # Display sources
        print(f"\nSources used: {result.get('sources_count', 0)}")
        if result.get("sources"):
            for source in result["sources"]:
                print(f"  - {source['country']} (relevance: {source['score']:.2f})")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(example_queries())
