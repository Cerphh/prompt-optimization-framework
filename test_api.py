import requests
import json
import sys

# Test the benchmark endpoint
url = "http://localhost:8000/benchmark"
data = {
    "problem": "What is 2 + 2?",
    "subject": "general"
}

try:
    response = requests.post(url, json=data, timeout=20)
    print(f"Status: {response.status_code}")

    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except ValueError:
        print(f"Response text: {response.text}")

    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    response = getattr(e, "response", None)
    if response is not None:
        print(f"Response text: {response.text}")
    sys.exit(1)
