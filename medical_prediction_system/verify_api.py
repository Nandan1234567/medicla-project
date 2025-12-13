import requests
import json

BASE_URL = "http://localhost:5000/api"

def run_test():
    print("1. Registering User...")
    session = requests.Session()
    reg_data = {
        "email": "api_test@test.com",
        "password": "password123",
        "name": "API Tester",
        "dateOfBirth": "1990-01-01",
        "gender": "male",
        "phone": "0000000000"
    }
    try:
        r = session.post(f"{BASE_URL}/auth/register", json=reg_data)
        if r.status_code != 201:
            # If user exists (from previous run), try login
            print(f"   Register failed ({r.status_code}), trying login...")
            r = session.post(f"{BASE_URL}/auth/login", json={"email": "api_test@test.com", "password": "password123"})

        data = r.json()
        if not data.get('success'):
            print("   Auth Failed:", data)
            return

        token = data['token']
        print("   Auth Success! Token received.")

        print("\n2. Testing Prediction (Fever)...")
        headers = {"Authorization": f"Bearer {token}"}
        pred_data = {
            "symptoms": "I have a high fever and chills",
            "age": 30,
            "severity": "medium"
        }

        r = session.post(f"{BASE_URL}/prediction/predict", json=pred_data, headers=headers)
        res = r.json()

        if res.get('success'):
            print("   Prediction Success!")
            print(f"   Disease: {res['result']['disease']}")

            # Verify Wellness Plan
            diet = res.get('dietPlan', [])
            exercise = res.get('exercisePlan', [])
            precautions = res.get('precautions', [])

            print(f"\n   [VERIFICATION] Diet Plan items: {len(diet)}")
            print(f"   [VERIFICATION] Exercise Plan items: {len(exercise)}")
            print(f"   [VERIFICATION] Precautions items: {len(precautions)}")

            if len(diet) > 0 and len(exercise) > 0:
                print("\n   ✅ FEATURE VERIFIED: Wellness Plan data is correctly returned by API.")
            else:
                print("\n   ❌ FEATURE FAILED: Wellness Plan data is missing.")
        else:
            print("   Prediction Failed:", res)

    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    run_test()
