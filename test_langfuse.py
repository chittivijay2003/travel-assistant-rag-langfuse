"""Test script to verify LangFuse tracing."""

import os
from dotenv import load_dotenv
from langfuse import Langfuse, observe

# Load environment variables
load_dotenv()

# Verify credentials are loaded
print(f"Public Key: {os.getenv('LANGFUSE_PUBLIC_KEY')[:10]}...")
print(f"Secret Key: {os.getenv('LANGFUSE_SECRET_KEY')[:10]}...")
print(f"Host: {os.getenv('LANGFUSE_HOST')}")

# Initialize client
client = Langfuse()
print(f"LangFuse client initialized: {client}")


@observe(name="test_function", as_type="chain")
def test_function(x: str) -> str:
    """Test function with observe decorator."""
    return f"Processed: {x}"


# Test the decorated function
result = test_function("Hello LangFuse!")
print(f"Result: {result}")

# Flush to ensure traces are sent
client.flush()
print("Traces flushed to LangFuse")
