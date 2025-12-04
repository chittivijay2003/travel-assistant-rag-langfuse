"""
Test file for RAG Travel Assistant API
Tests the /rag-travel-assistant endpoint with comprehensive test cases.
Results are saved to output.json in JSON format.

Usage:
1. Start the server: uvicorn main:app --reload --port 8001
2. Run tests: python3 test_main.py
"""

import json
import requests
import time
from datetime import datetime
from typing import List, Dict, Any
import os

# Get port from environment or use default
APP_PORT = int(os.getenv("APP_PORT", "8001"))
API_BASE_URL = f"http://localhost:{APP_PORT}"
API_ENDPOINT = f"{API_BASE_URL}/rag-travel-assistant"
GEMINI_DIRECT_ENDPOINT = f"{API_BASE_URL}/gemini-direct"

# Test scenarios with different queries
TEST_SCENARIOS = [
    {
        "name": "Japan Visa Requirements",
        "query": "What are visa requirements for Indians traveling to Japan?",
        "top_k": 3,
        "return_sources": True,
    },
    {
        "name": "Singapore Travel Info",
        "query": "Tell me about Singapore visa requirements and best time to visit",
        "top_k": 2,
        "return_sources": True,
    },
    {
        "name": "Thailand Tourism",
        "query": "Best time to visit Thailand and attractions",
        "top_k": 3,
        "return_sources": True,
    },
    {
        "name": "General Visa Question",
        "query": "Which countries require visa for Indian citizens?",
        "top_k": 5,
        "return_sources": True,
    },
    {
        "name": "Tourist Attractions",
        "query": "What are the top tourist attractions in Japan?",
        "top_k": 2,
        "return_sources": True,
    },
    {
        "name": "Currency Information",
        "query": "What currency is used in Singapore?",
        "top_k": 1,
        "return_sources": True,
    },
    {
        "name": "Travel Documents",
        "query": "What documents do I need for Thailand travel?",
        "top_k": 2,
        "return_sources": True,
    },
    {
        "name": "Multiple Countries",
        "query": "Compare visa requirements for Japan, Singapore and Thailand",
        "top_k": 5,
        "return_sources": True,
    },
    {
        "name": "Best Time to Visit",
        "query": "When is the best time to visit Japan for cherry blossoms?",
        "top_k": 2,
        "return_sources": True,
    },
    {
        "name": "Processing Time",
        "query": "How long does it take to get a Singapore visa?",
        "top_k": 1,
        "return_sources": True,
    },
    {
        "name": "Gemini Direct (No Retrieval)",
        "query": "What are visa requirements for Indians traveling to Japan?",
        "top_k": 1,
        "return_sources": False,
        "endpoint": "gemini-direct",
    },
]


def test_api_endpoint(scenario: Dict[str, Any], delay: float = 2.0) -> Dict[str, Any]:
    """
    Test the RAG API endpoint with a single scenario.

    Args:
        scenario: Test scenario with query and parameters
        delay: Delay between requests to avoid rate limits

    Returns:
        Test result with request, response, and metadata
    """
    print(f"Testing: {scenario['name']}")
    print(f"  Query: {scenario['query']}")

    # Prepare request payload
    request_payload = {
        "query": scenario["query"],
        "top_k": scenario.get("top_k", 5),
        "return_sources": scenario.get("return_sources", True),
    }

    # Determine endpoint
    endpoint = (
        GEMINI_DIRECT_ENDPOINT
        if scenario.get("endpoint") == "gemini-direct"
        else API_ENDPOINT
    )

    # Make API request
    try:
        start_time = time.time()
        response = requests.post(
            endpoint,
            json=request_payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        end_time = time.time()

        response_time = end_time - start_time

        # Parse response
        if response.status_code == 200:
            response_data = response.json()
            status = "success"
            error_message = None
            print(
                f"  ✅ Success (Sources: {response_data.get('sources_count', 0)}, Time: {response_time:.2f}s)"
            )
        else:
            response_data = (
                response.json()
                if response.headers.get("content-type") == "application/json"
                else {}
            )
            status = "error"
            error_message = response_data.get("detail", response.text)
            print(f"  ❌ Error: {error_message}")

    except requests.exceptions.RequestException as e:
        status = "error"
        error_message = str(e)
        response_data = {}
        response_time = 0
        print(f"  ❌ Request failed: {error_message}")

    # Build result
    result = {
        "scenario_name": scenario["name"],
        "timestamp": datetime.now().isoformat(),
        "request": request_payload,
        "response": response_data,
        "metadata": {
            "status": status,
            "response_time_seconds": round(response_time, 3),
            "error": error_message,
        },
    }

    # Add delay to avoid rate limits
    if delay > 0:
        time.sleep(delay)

    return result


def check_server_health() -> bool:
    """Check if the server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Version: {health_data.get('version')}")
            print(f"   Components: {health_data.get('components')}")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Server is not running or not reachable: {e}")
        print(f"\nPlease start the server with:")
        print(f"  uvicorn main:app --reload --port {APP_PORT}")
        return False


def run_comprehensive_tests() -> Dict[str, Any]:
    """Run all test scenarios and collect results."""
    print("\n" + "=" * 80)
    print("RAG TRAVEL ASSISTANT - API ENDPOINT TESTS")
    print(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Check server health
    print("Checking server health...")
    if not check_server_health():
        return {
            "error": "Server is not running",
            "message": f"Please start the server: uvicorn main:app --reload --port {APP_PORT}",
        }

    print(f"\n{'=' * 80}")
    print(f"Running {len(TEST_SCENARIOS)} test scenarios...")
    print("=" * 80 + "\n")

    # Run all tests
    test_results = []
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"\n[Test {i}/{len(TEST_SCENARIOS)}]")
        result = test_api_endpoint(scenario, delay=2.5)
        test_results.append(result)

    # Calculate statistics
    total_tests = len(test_results)
    successful_tests = sum(
        1 for r in test_results if r["metadata"]["status"] == "success"
    )
    failed_tests = total_tests - successful_tests

    avg_response_time = (
        sum(r["metadata"]["response_time_seconds"] for r in test_results) / total_tests
        if total_tests > 0
        else 0
    )

    # Build comprehensive output
    output = {
        "test_run_metadata": {
            "timestamp": datetime.now().isoformat(),
            "api_endpoint": API_ENDPOINT,
            "total_scenarios": total_tests,
            "successful": successful_tests,
            "failed": failed_tests,
            "average_response_time_seconds": round(avg_response_time, 3),
        },
        "test_results": test_results,
        "summary": {
            "total_queries_tested": total_tests,
            "success_rate_percent": round((successful_tests / total_tests * 100), 2)
            if total_tests > 0
            else 0,
            "total_sources_retrieved": sum(
                r["response"].get("sources_count", 0)
                for r in test_results
                if r["metadata"]["status"] == "success"
            ),
            "scenarios_by_status": {
                "success": [
                    r["scenario_name"]
                    for r in test_results
                    if r["metadata"]["status"] == "success"
                ],
                "failed": [
                    r["scenario_name"]
                    for r in test_results
                    if r["metadata"]["status"] == "error"
                ],
            },
        },
    }

    return output


def main():
    """Main test execution function."""
    # Run tests
    results = run_comprehensive_tests()

    # Save to JSON file
    output_file = "output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        print(f"   {results['message']}")
    else:
        metadata = results["test_run_metadata"]
        summary = results["summary"]

        print(f"Total Tests: {metadata['total_scenarios']}")
        print(f"Successful: {metadata['successful']} ✅")
        print(f"Failed: {metadata['failed']} ❌")
        print(f"Success Rate: {summary['success_rate_percent']}%")
        print(f"Avg Response Time: {metadata['average_response_time_seconds']}s")
        print(f"Total Sources Retrieved: {summary['total_sources_retrieved']}")

        print(f"\n✅ Results saved to: {output_file}")
        print("\nYou can view the detailed JSON output with:")
        print(f"  cat {output_file}")
        print(f"  python3 -m json.tool {output_file}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
