import sys
import os
import json

# Set up paths to import from backend_flask
sys.path.append(os.path.join(os.getcwd(), 'backend_flask'))

from app import predict_disease

def test_prediction(symptom_text, expected_emergency=False, expected_trigger=None):
    print(f"\nTesting: '{symptom_text}'")
    try:
        response = predict_disease(symptom_text)

        if not response['success']:
            print("❌ Prediction failed")
            return

        # Print full JSON for the first test case (swelling throat) or if explicitly requested
        if "swelling throat" in symptom_text:
            print("\n--- FULL JSON OUTPUT ---")
            print(json.dumps(response, indent=2))
            print("------------------------\n")

        print("✅ Prediction success")

        # Check Output Format
        keys = ['predictedDiagnosis', 'confidenceScore', 'emergencyFlag', 'doctorConsultationMessage']
        missing_keys = [k for k in keys if k not in response]
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
        else:
            print("✅ Output format check passed")

        # Check Values
        is_emergency = response['emergencyFlag']
        diagnosis = response.get('predictedDiagnosis')

        print(f"   Diagnosis: {diagnosis}")
        print(f"   Emergency: {is_emergency}")
        print(f"   Message: {response.get('doctorConsultationMessage')}")

        if expected_emergency:
            if is_emergency:
                if expected_trigger and expected_trigger not in diagnosis:
                     print(f"⚠️ warning: Expected diagnosis logic '{expected_trigger}' but got '{diagnosis}'")
                else:
                    print("✅ Correctly identified as Emergency")
            else:
                print("❌ FAILED: Should be Emergency but wasn't")
        else:
            if not is_emergency:
                print("✅ Correctly identified as Non-Emergency")
            else:
                print("❌ FAILED: Should be Non-Emergency but was flagged")

    except Exception as e:
        print(f"❌ detailed error: {e}")

def main():
    print("="*60)
    print("VERIFYING SAFETY RULES AND OUTPUT FORMAT")
    print("="*60)

    # 1. Test Hardcoded Safety Rules
    test_prediction("I have swelling throat and can't breathe", expected_emergency=True, expected_trigger="swelling throat")
    test_prediction("There is a bone sticking out of my leg", expected_emergency=True, expected_trigger="bone sticking out")
    test_prediction("sudden paralysis on left side", expected_emergency=True, expected_trigger="sudden paralysis")

    # 2. Test Emergency Keywords (Secondary Check)
    test_prediction("severe chest pain", expected_emergency=True)

    # 3. Test Normal Prediction
    test_prediction("I have a mild headache and runny nose", expected_emergency=False)

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
