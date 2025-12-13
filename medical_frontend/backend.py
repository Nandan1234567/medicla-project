"""
Medical Disease Prediction Backend API
Flask server that connects frontend to ML model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# ============================================
# Load ML Model and Preprocessors
# ============================================
try:
    # Load the trained model
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    
    MODEL_LOADED = True
    print("✓ ML Model loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠ Warning: Could not load ML models - {e}")
    print("Server will run in demo mode with mock predictions")

# ============================================
# Medical Recommendations Database
# ============================================
RECOMMENDATIONS = {
    'Viral Fever': [
        'Rest and stay hydrated with plenty of fluids',
        'Take fever-reducing medication (paracetamol)',
        'Monitor temperature regularly',
        'Consult doctor if fever persists > 3 days',
        'Avoid going to public places'
    ],
    'Headache': [
        'Rest in a quiet, dark room',
        'Stay well hydrated',
        'Apply cold or warm compress',
        'Take OTC pain relievers if needed',
        'Avoid screens and bright lights'
    ],
    'Abdominal Pain': [
        'Avoid solid foods initially',
        'Stay hydrated with clear liquids',
        'Avoid spicy and fatty foods',
        'Apply warm compress to abdomen',
        'Seek medical help if severe or persistent'
    ],
    'Allergic Reaction': [
        'Identify and avoid allergen',
        'Take antihistamine medication',
        'Apply cold compress to affected area',
        'Monitor for severe reactions',
        'Seek emergency care if breathing difficulty'
    ],
    'Respiratory Distress': [
        'Sit upright to ease breathing',
        'Use prescribed inhaler if available',
        'Avoid triggers and irritants',
        'Seek immediate medical attention',
        'Stay calm and breathe slowly'
    ],
    'default': [
        'Monitor symptoms closely',
        'Rest and maintain good hydration',
        'Maintain a balanced diet',
        'Avoid self-medication',
        'Consult healthcare provider if symptoms worsen'
    ]
}

PRECAUTIONS = {
    'Viral Fever': [
        'Isolate yourself to prevent spread',
        'Wear mask when around others',
        'Wash hands frequently',
        'Disinfect frequently touched surfaces',
        'Get adequate sleep (8+ hours)'
    ],
    'Headache': [
        'Maintain regular sleep schedule',
        'Manage stress through relaxation',
        'Limit caffeine and alcohol',
        'Stay hydrated throughout day',
        'Practice good posture'
    ],
    'Abdominal Pain': [
        'Eat smaller, frequent meals',
        'Avoid trigger foods',
        'Reduce stress levels',
        'Exercise regularly but gently',
        'Stay upright after eating'
    ],
    'default': [
        'Maintain good personal hygiene',
        'Get regular exercise',
        'Eat a balanced, nutritious diet',
        'Get 7-8 hours of sleep',
        'Manage stress effectively',
        'Stay hydrated throughout the day'
    ]
}

# ============================================
# Prediction Endpoint
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        symptoms = data.get('symptoms', '')
        age = data.get('age', '')
        gender = data.get('gender', '')
        duration = data.get('duration', '')
        severity = data.get('severity', '')
        
        # Validate input
        if not symptoms:
            return jsonify({'error': 'Symptoms are required'}), 400
        
        if MODEL_LOADED:
            # Use real ML model
            # Transform symptoms using TF-IDF
            symptoms_vectorized = vectorizer.transform([symptoms])
            
            # Make prediction
            prediction = model.predict(symptoms_vectorized)[0]
            prediction_proba = model.predict_proba(symptoms_vectorized)[0]
            
            # Get disease name
            disease = encoder.inverse_transform([prediction])[0]
            
            # Get confidence (max probability)
            confidence = float(max(prediction_proba) * 100)
        else:
            # Mock prediction for demo
            mock_diseases = ['Viral Fever', 'Headache', 'Abdominal Pain', 'Allergic Reaction']
            disease = mock_diseases[hash(symptoms) % len(mock_diseases)]
            confidence = 75.0 + (hash(symptoms) % 20)
        
        # Get recommendations and precautions
        recommendations = RECOMMENDATIONS.get(disease, RECOMMENDATIONS['default'])
        precautions = PRECAUTIONS.get(disease, PRECAUTIONS['default'])
        
        # Prepare response
        response = {
            'disease': disease,
            'confidence': round(confidence, 2),
            'recommendations': recommendations,
            'precautions': precautions,
            'metadata': {
                'age': age,
                'gender': gender,
                'duration': duration,
                'severity': severity
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# ============================================
# Health Check Endpoint
# ============================================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'message': 'Medical Healthcare API is running'
    })

# ============================================
# Main
# ============================================
if __name__ == '__main__':
    print("="*60)
    print("Medical Healthcare Recommendations API")
    print("="*60)
    print(f"Model Status: {'Loaded' if MODEL_LOADED else 'Demo Mode'}")
    print("Server starting on http://localhost:5000")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
