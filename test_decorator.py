"""Test to verify LangFuse decorator is working."""

import os
from dotenv import load_dotenv

load_dotenv()

# Check credentials
print(f"Public Key length: {len(os.getenv('LANGFUSE_PUBLIC_KEY', ''))}")
print(f"Secret Key length: {len(os.getenv('LANGFUSE_SECRET_KEY', ''))}")
print(f"Are they placeholders? {os.getenv('LANGFUSE_PUBLIC_KEY') == 'pk-lf-...'}")

from langfuse import Langfuse, observe

# Initialize client
langfuse = Langfuse()


@observe(name="test_retrieval", as_type="retriever")
def test_retrieval(query: str):
    """Test retrieval function."""
    print(f"Processing query: {query}")
    return {"results": [f"doc1 for {query}", f"doc2 for {query}"]}


@observe(name="test_generation", as_type="generation")
def test_generation(context: str):
    """Test generation function."""
    print(f"Generating from context: {context}")
    return f"Generated answer based on {context}"


# Run test
print("\n=== Running Test ===")
docs = test_retrieval("test query")
print(f"Retrieved: {docs}")

answer = test_generation("test context")
print(f"Generated: {answer}")

# Flush
print("\n=== Flushing ===")
langfuse.flush()
print("Done! Check LangFuse dashboard.")
