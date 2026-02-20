import requests
import json

# Test the benchmark endpoint
url = "http://localhost:8000/benchmark"
data = {
    "problem": "What is 2 + 2?",
    "subject": "general"
}

try:
    response = requests.post(url, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'response'):
        print(f"Response text: {e.response.text}")
