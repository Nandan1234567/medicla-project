import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load trained model and preprocessors
with open('/home/user/final_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('/home/user/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
with open('/home/user/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load disease-recommendation mapping from dataset
df = pd.read_csv('/home/user/cleaned_dataset.csv')
disease_recommendations = {}
for disease in df['Final Recommendation'].unique():
    # Get most common symptoms for this disease
    disease_data = df[df['Final Recommendation'] == disease]
    disease_recommendations[disease] = {
        'common_symptoms': disease_data['Symptoms'].value_counts().head(1).index[0] if len(disease_data) > 0 else '',
        'count': len(disease_data)
    }

# Medical recommendations database
MEDICAL_PRECAUTIONS = {
    'Viral Fever': [
        'Rest adequately and stay hydrated',
        'Take fever-reducing medication (acetaminophen or ibuprofen)',
        'Monitor temperature regularly',
        'Avoid contact with others to prevent spread',
        'If fever persists > 3 days or worsens, consult a doctor immediately'
    ],
    'Headache': [
        'Rest in a quiet, dark room',
        'Stay well hydrated',
        'Apply cold or warm compress to head/neck',
        'Take over-the-counter pain relievers as directed',
        'If severe or persistent, seek medical attention'
    ],
    'Abdominal Pain': [
        'Avoid solid foods temporarily',
        'Stay hydrated with clear liquids',
        'Avoid spicy, fatty, or acidic foods',
        'Apply warm compress to abdomen',
        'If pain is severe, persistent, or accompanied by vomiting, see a doctor'
    ],
    'Heart Disease': [
        '⚠️ URGENT: This requires immediate medical attention',
        'Call emergency services if experiencing chest pain',
        'Do not drive yourself to hospital',
        'Take prescribed medications as directed',
        'Follow up with cardiologist regularly'
    ],
    'Respiratory Distress': [
        '⚠️ URGENT: Seek immediate medical care',
        'Sit upright to ease breathing',
        'Use prescribed inhaler if available',
        'Stay calm and breathe slowly',
        'Call emergency services if difficulty breathing worsens'
    ],
    'default': [
        'Monitor your symptoms closely',
        'Rest and maintain good hydration',
        'Avoid self-medication without consultation',
        'Keep a symptom diary',
        'Consult a healthcare provider if symptoms worsen or persist'
    ]
}

# Severity markers that indicate urgent care needed
URGENT_KEYWORDS = [
    'chest pain', 'shortness breath', 'difficulty breathing', 
    'loss of consciousness', 'severe bleeding', 'confusion',
    'severe pain', 'heart', 'stroke', 'seizure', 'unconscious'
]

def predict_disease(symptoms_text):
    """
    Predict disease from symptom text and provide recommendations.
    
    Args:
        symptoms_text (str): Description of symptoms
        
    Returns:
        dict: Contains disease name, confidence score, and precautions
    """
    
    # Preprocess input
    symptoms_clean = symptoms_text.strip().lower()
    
    if len(symptoms_clean) < 3:
        return {
            'disease': 'Insufficient Information',
            'confidence': 0.0,
            'message': 'Please provide more detailed symptoms',
            'precautions': ['Describe your symptoms in more detail'],
            'severity': 'Unknown',
            'consult_doctor': True
        }
    
    # Transform symptoms to TF-IDF features
    symptoms_tfidf = tfidf_vectorizer.transform([symptoms_clean])
    
    # Predict disease
    prediction = model.predict(symptoms_tfidf)[0]
    disease_name = label_encoder.inverse_transform([prediction])[0]
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(symptoms_tfidf)[0]
        confidence = float(probabilities[prediction])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = label_encoder.inverse_transform(top_3_indices)
        top_3_confidences = probabilities[top_3_indices]
        
        alternative_diagnoses = [
            {'disease': top_3_diseases[i], 'confidence': float(top_3_confidences[i])}
            for i in range(min(3, len(top_3_indices)))
        ]
    else:
        confidence = 0.85  # Default for models without probability
        alternative_diagnoses = []
    
    # Determine severity and urgency
    is_urgent = any(keyword in symptoms_clean for keyword in URGENT_KEYWORDS)
    
    # Get precautions
    precautions = MEDICAL_PRECAUTIONS.get(disease_name, MEDICAL_PRECAUTIONS['default'])
    
    # Determine if doctor consultation is needed
    needs_doctor = (
        confidence < 0.6 or  # Low confidence
        is_urgent or  # Urgent symptoms
        'heart' in disease_name.lower() or
        'stroke' in disease_name.lower() or
        'cancer' in disease_name.lower() or
        'severe' in symptoms_clean
    )
    
    # Build response
    result = {
        'disease': disease_name,
        'confidence': round(confidence * 100, 2),
        'alternative_diagnoses': alternative_diagnoses[:3],
        'precautions': precautions,
        'consult_doctor': needs_doctor,
        'urgency': 'URGENT' if is_urgent else ('HIGH' if needs_doctor else 'MODERATE'),
        'ai_disclaimer': '⚠️ This is an AI prediction. Always consult a qualified healthcare professional for accurate diagnosis and treatment.'
    }
    
    return result

def format_prediction(result):
    """Format prediction result for display"""
    
    output = []
    output.append("="*80)
    output.append("MEDICAL DISEASE PREDICTION RESULT")
    output.append("="*80)
    output.append("")
    
    # Disclaimer
    output.append("⚠️  AI SYSTEM DISCLAIMER")
    output.append("-" * 80)
    output.append(result['ai_disclaimer'])
    output.append("")
    
    # Primary diagnosis
    output.append("🏥 PRIMARY DIAGNOSIS")
    output.append("-" * 80)
    output.append(f"Disease: {result['disease']}")
    output.append(f"Confidence: {result['confidence']}%")
    output.append(f"Urgency Level: {result['urgency']}")
    output.append("")
    
    # Alternative diagnoses
    if result.get('alternative_diagnoses'):
        output.append("🔍 ALTERNATIVE POSSIBILITIES")
        output.append("-" * 80)
        for i, alt in enumerate(result['alternative_diagnoses'], 1):
            output.append(f"{i}. {alt['disease']} ({alt['confidence']*100:.2f}%)")
        output.append("")
    
    # Precautions
    output.append("💊 RECOMMENDED PRECAUTIONS")
    output.append("-" * 80)
    for i, precaution in enumerate(result['precautions'], 1):
        output.append(f"{i}. {precaution}")
    output.append("")
    
    # Doctor consultation
    if result['consult_doctor']:
        output.append("⚠️  IMPORTANT")
        output.append("-" * 80)
        output.append("⚠️  PLEASE CONSULT A DOCTOR IMMEDIATELY")
        if result['urgency'] == 'URGENT':
            output.append("⚠️  SEEK EMERGENCY MEDICAL CARE NOW")
        output.append("")
    
    output.append("="*80)
    
    return "\n".join(output)

# Test function
if __name__ == "__main__":
    print("="*80)
    print("MEDICAL DISEASE PREDICTION SYSTEM - INFERENCE FUNCTION")
    print("="*80)
    print("\nSystem loaded successfully!")
    print(f"✓ Model loaded")
    print(f"✓ TF-IDF vectorizer loaded")
    print(f"✓ Label encoder loaded ({len(label_encoder.classes_)} diseases)")
    print("\nReady for predictions!")
