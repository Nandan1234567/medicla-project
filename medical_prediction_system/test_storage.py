"""
Test Prediction Storage to MongoDB
This script tests if predictions are being saved correctly
"""

import requests
import json

API_URL = "http://localhost:5000/api"

def test_prediction_storage():
    """Test if predictions are saved to MongoDB"""

    print("\n" + "="*70)
    print("TESTING PREDICTION STORAGE TO MONGODB")
    print("="*70 + "\n")

    # Step 1: Register a test user
    print("1. Registering test user...")
    register_data = {
        "name": "Test User Storage",
        "email": "teststorage@example.com",
        "password": "password123",
        "phone": "9999999999",
        "dateOfBirth": "2000-01-01T00:00:00.000Z",
        "gender": "Other"
    }

    try:
        response = requests.post(f"{API_URL}/auth/register", json=register_data)
        if response.status_code == 201:
            data = response.json()
            token = data['token']
            print(f"   ✓ User registered successfully")
            print(f"   Token: {token[:20]}...")
        elif response.status_code == 400:
            # User might already exist, try login
            print("   User already exists, trying login...")
            login_data = {
                "email": "teststorage@example.com",
                "password": "password123"
            }
            response = requests.post(f"{API_URL}/auth/login", json=login_data)
            if response.status_code == 200:
                data = response.json()
                token = data['token']
                print(f"   ✓ Logged in successfully")
            else:
                print(f"   ✗ Login failed: {response.text}")
                return
        else:
            print(f"   ✗ Registration failed: {response.text}")
            return
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Step 2: Make a prediction
    print("\n2. Making a prediction...")
    prediction_data = {
        "symptoms": "high fever, headache, and body pain",
        "severity": "medium"
    }

    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{API_URL}/prediction/predict",
                                json=prediction_data,
                                headers=headers)

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Prediction successful")
            print(f"   Disease: {result['result']['disease']}")
            print(f"   Confidence: {result['result']['confidence']}%")
            print(f"   Prediction ID: {result.get('predictionId', 'None')}")

            if result.get('predictionId'):
                print(f"   ✅ PREDICTION WAS SAVED TO DATABASE!")
            else:
                print(f"   ⚠️  Warning: No prediction ID returned")
        else:
            print(f"   ✗ Prediction failed: {response.text}")
            return
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return

    # Step 3: Check history
    print("\n3. Checking prediction history...")
    try:
        response = requests.get(f"{API_URL}/prediction/history", headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data['success'] and data['predictions']:
                print(f"   ✓ History retrieved successfully")
                print(f"   Total predictions: {len(data['predictions'])}")
                print(f"   Latest prediction:")
                latest = data['predictions'][0]
                print(f"     - Disease: {latest['result']['disease']}")
                print(f"     - Symptoms: {latest['symptoms']}")
                print(f"     - Date: {latest['createdAt']}")
                print(f"\n   ✅ PREDICTIONS ARE BEING SAVED AND RETRIEVED!")
            else:
                print(f"   ⚠️  No predictions found in history")
        else:
            print(f"   ✗ Failed to get history: {response.text}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_prediction_storage()
