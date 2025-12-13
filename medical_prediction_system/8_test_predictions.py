from inference_function import predict_disease, format_prediction

print("="*80)
print("TESTING MEDICAL DISEASE PREDICTION SYSTEM")
print("="*80)

# Test cases
test_cases = [
    "I have fever and headache for 2 days",
    "I have chest pain and short breathing",
    "Stomach pain after eating, nausea since morning",
    "severe abdominal pain, vomiting blood, dizziness",
    "running nose, sneezing, sore throat, body ache"
]

print("\n" + "="*80)
print("RUNNING 5 TEST PREDICTIONS")
print("="*80)

for i, symptoms in enumerate(test_cases, 1):
    print(f"\n{'#'*80}")
    print(f"TEST CASE {i}")
    print(f"{'#'*80}")
    print(f"\nSymptoms: \"{symptoms}\"")
    print("")
    
    # Get prediction
    result = predict_disease(symptoms)
    
    # Format and display
    formatted_output = format_prediction(result)
    print(formatted_output)
    
    # Add spacing
    print("\n")

print("="*80)
print("✓ ALL TESTS COMPLETED")
print("="*80)
