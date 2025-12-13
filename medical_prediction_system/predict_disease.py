#!/usr/bin/env python3
"""
Standalone Medical Disease Prediction Script
Usage: python predict_disease.py "your symptoms here"
"""

import pickle
import numpy as np
import sys

# Load model files
try:
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Model files not found. Please ensure all .pkl files are in the same directory.")
    sys.exit(1)

# Medical precautions
PRECAUTIONS = {
    'Viral Fever': ['Rest and stay hydrated', 'Take fever-reducing medication', 'Monitor temperature', 'Consult doctor if persists > 3 days'],
    'Headache': ['Rest in quiet room', 'Stay hydrated', 'Apply compress', 'Take OTC pain relievers', 'See doctor if severe'],
    'Abdominal Pain': ['Avoid solid foods', 'Stay hydrated', 'Avoid spicy foods', 'Apply warm compress', 'See doctor if severe'],
    'Heart Disease': ['⚠️ URGENT - Call emergency services', 'Do not drive', 'Take prescribed meds', 'See cardiologist'],
    'default': ['Monitor symptoms', 'Rest and hydrate', 'Avoid self-medication', 'Consult healthcare provider']
}

URGENT_KEYWORDS = ['chest pain', 'shortness breath', 'difficulty breathing', 'severe pain', 'heart']

def predict(symptoms_text):
    """Predict disease from symptoms"""
    
    symptoms_clean = symptoms_text.strip().lower()
    
    if len(symptoms_clean) < 3:
        return {'disease': 'Error', 'confidence': 0, 'message': 'Please provide more symptoms'}
    
    # Transform and predict
    symptoms_tfidf = vectorizer.transform([symptoms_clean])
    prediction = model.predict(symptoms_tfidf)[0]
    disease = encoder.inverse_transform([prediction])[0]
    
    # Get confidence
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(symptoms_tfidf)[0]
        confidence = float(probs[prediction]) * 100
        
        # Top 3
        top_3 = np.argsort(probs)[-3:][::-1]
        alternatives = [(encoder.inverse_transform([i])[0], probs[i]*100) for i in top_3]
    else:
        confidence = 85.0
        alternatives = [(disease, confidence)]
    
    # Urgency
    is_urgent = any(kw in symptoms_clean for kw in URGENT_KEYWORDS)
    needs_doctor = confidence < 60 or is_urgent or 'heart' in disease.lower()
    
    precautions = PRECAUTIONS.get(disease, PRECAUTIONS['default'])
    
    return {
        'disease': disease,
        'confidence': confidence,
        'alternatives': alternatives,
        'precautions': precautions,
        'urgent': is_urgent,
        'consult_doctor': needs_doctor
    }

def display_result(result):
    """Display formatted prediction"""
    
    print("\n" + "="*70)
    print("MEDICAL DISEASE PREDICTION")
    print("="*70)
    print("\n⚠️  AI PREDICTION - NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL ADVICE")
    print("\n🏥 DIAGNOSIS")
    print(f"   Disease: {result['disease']}")
    print(f"   Confidence: {result['confidence']:.2f}%")
    
    if len(result['alternatives']) > 1:
        print("\n🔍 ALTERNATIVE POSSIBILITIES:")
        for i, (disease, conf) in enumerate(result['alternatives'][:3], 1):
            print(f"   {i}. {disease} ({conf:.2f}%)")
    
    print("\n💊 RECOMMENDED PRECAUTIONS:")
    for i, prec in enumerate(result['precautions'], 1):
        print(f"   {i}. {prec}")
    
    if result['urgent']:
        print("\n⚠️  ⚠️  ⚠️  URGENT - SEEK EMERGENCY MEDICAL CARE IMMEDIATELY")
    elif result['consult_doctor']:
        print("\n⚠️  IMPORTANT: Please consult a doctor")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_disease.py \"your symptoms here\"")
        print("\nExample:")
        print('  python predict_disease.py "I have fever and headache for 2 days"')
        sys.exit(1)
    
    symptoms = " ".join(sys.argv[1:])
    result = predict(symptoms)
    display_result(result)
