"""
Simple test script to verify medical predictions
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from backend_flask.app import predict_disease

# Test cases
tests = [
    ("Emergency - Swelling Throat", "swelling throat can't breathe"),
    ("Emergency - Bone Injury", "bone sticking out of arm"),
    ("Common - Fever", "high fever headache body pain"),
    ("Common - Cold", "runny nose sneezing cough"),
    ("Mild - Headache", "mild headache tired"),
]

print("\n" + "="*70)
print("MEDICAL PREDICTION VERIFICATION")
print("="*70)

for name, symptoms in tests:
    result = predict_disease(symptoms)

    print(f"\n{name}")
    print(f"Input: {symptoms}")

    if result.get('success'):
        print(f"  → Disease: {result['result']['disease']}")
        print(f"  → Confidence: {result['result']['confidence']}%")
        print(f"  → Emergency: {result['isEmergency']}")
        print(f"  → Urgency: {result['result']['urgencyLevel']}")

        if result.get('alternatives'):
            alts = result['alternatives'][:2]
            alt_text = ', '.join(["{} ({:.0f}%)".format(a['disease'], a['confidence']) for a in alts])
            print(f"  → Alternatives: {alt_text}")
    else:
        print(f"  → ERROR: {result.get('error')}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70 + "\n")
