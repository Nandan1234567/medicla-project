"""
Test script to verify medical predictions with different symptom types
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(__file__))

from backend_flask.app import predict_disease
import json

def test_symptom(description, symptoms):
    """Test a symptom and print results"""
    print("\n" + "="*80)
    print(f"TEST: {description}")
    print("="*80)
    print(f"Symptoms: {symptoms}")
    print("-"*80)

    result = predict_disease(symptoms)

    if result.get('success'):
        print(f"✓ Prediction Successful")
        print(f"  Disease: {result['result']['disease']}")
        print(f"  Confidence: {result['result']['confidence']}%")
        print(f"  Urgency: {result['result']['urgencyLevel']}")
        print(f"  Emergency: {'YES' if result['isEmergency'] else 'NO'}")
        print(f"  Doctor Message: {result['doctorConsultationMessage']}")

        if result.get('alternatives'):
            print(f"\n  Alternative Diagnoses:")
            for i, alt in enumerate(result['alternatives'][:3], 1):
                print(f"    {i}. {alt['disease']} ({alt['confidence']:.1f}%)")

        print(f"\n  Precautions ({len(result['precautions'])} items):")
        for i, prec in enumerate(result['precautions'][:3], 1):
            print(f"    {i}. {prec}")
    else:
        print(f"✗ Prediction Failed: {result.get('error')}")

def main():
    """Run all test cases"""
    print("\n" + "="*80)
    print("MEDICAL PREDICTION SYSTEM - VERIFICATION TESTS")
    print("="*80)

    test_cases = [
        # Emergency Cases
        ("Emergency - Swelling Throat", "I have swelling throat and difficulty swallowing"),
        ("Emergency - Bone Injury", "I have a bone sticking out of my arm"),
        ("Emergency - Chest Pain", "severe chest pain and shortness of breath"),

        # Common Conditions
        ("Common - Fever and Headache", "I have high fever, headache, and body aches"),
        ("Common - Cold Symptoms", "runny nose, sneezing, and mild cough"),
        ("Common - Stomach Issues", "stomach pain, nausea, and diarrhea"),

        # Specific Conditions
        ("Specific - Migraine", "severe headache on one side with sensitivity to light"),
        ("Specific - Diabetes Symptoms", "frequent urination, excessive thirst, and fatigue"),
        ("Specific - Skin Issue", "itchy red rash on arms and legs"),

        # Mild Symptoms
        ("Mild - General Fatigue", "feeling tired and weak for a few days"),
        ("Mild - Minor Headache", "mild headache after working on computer"),
    ]

    for description, symptoms in test_cases:
        test_symptom(description, symptoms)

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
