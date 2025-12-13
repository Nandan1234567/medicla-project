import requests
import json

# Test the prediction endpoint
url = "http://127.0.0.1:5000/api/prediction/predict"
headers = {"Content-Type": "application/json"}
data = {"symptoms": "fever and headache"}

print("Testing prediction endpoint...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, indent=2)}")
print("-" * 80)

try:
    response = requests.post(url, json=data, headers=headers, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
