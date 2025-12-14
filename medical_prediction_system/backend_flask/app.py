from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import os
import sys
import json
from datetime import datetime, timedelta
import re
from bson import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import logging

# Add the parent directory to sys.path to import the ML model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import ML predictor directly
from models.linear_svm_predictor import LinearSVMMedicalPredictor

app = Flask(__name__, static_folder='../frontend', static_url_path='/')

# Configuration
app.config['SECRET_KEY'] = 'medical_healthcare_super_secret_key_2024'
app.config['JWT_SECRET_KEY'] = 'jwt_medical_secret_key_2024'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['MONGO_URI'] = 'mongodb://localhost:27017/medical_healthcare'
app.config['GITHUB_CLIENT_ID'] = os.getenv('GITHUB_CLIENT_ID', 'REPLACE_WITH_YOUR_ID')
app.config['GITHUB_CLIENT_SECRET'] = os.getenv('GITHUB_CLIENT_SECRET', 'REPLACE_WITH_YOUR_SECRET')

# --- GitHub Auth Setup ---
try:
    from authlib.integrations.flask_client import OAuth
    oauth = OAuth(app)
    github = oauth.register(
        name='github',
        client_id=app.config['GITHUB_CLIENT_ID'],
        client_secret=app.config['GITHUB_CLIENT_SECRET'],
        access_token_url='https://github.com/login/oauth/access_token',
        access_token_params=None,
        authorize_url='https://github.com/login/oauth/authorize',
        authorize_params=None,
        api_base_url='https://api.github.com/',
        client_kwargs={'scope': 'user:email'},
    )
except ImportError:
    oauth = None
    github = None
    print("WARNING: Authlib not installed. GitHub login disabled.")

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize extensions
# Simple CORS - allow all origins for development
CORS(app)

# Serve Frontend
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return app.send_static_file(path)
    # Default to index.html for client-side routing if needed, though mostly using simple files now
    if path.endswith('.html'):
         return app.send_static_file(path)
    return app.send_static_file('index.html')

# Try to initialize MongoDB (optional)
try:
    mongo = PyMongo(app)
    # Explicitly test connection
    with app.app_context():
        mongo.cx.admin.command('ping')
    mongo_available = True
    logger.info("MongoDB connected successfully")
except Exception as e:
    mongo_available = False
    logger.warning(f"MongoDB not available: {e}. Running without database (In-Memory Mode).")
    mongo = None

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Helper function to convert ObjectId to string
def serialize_doc(doc):
    if doc is None:
        return None
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, datetime):
                doc[key] = value.isoformat()
            elif isinstance(value, dict) or isinstance(value, list):
                doc[key] = serialize_doc(value)
    return doc

# ML Model Integration
# Initialize predictor once at startup
try:
    # Fix: Point to the correct models directory relative to this file
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    ml_predictor = LinearSVMMedicalPredictor(model_dir=model_dir)
    logger.info(f"ML Predictor loaded successfully from {model_dir}")
except Exception as e:
    logger.error(f"Failed to load ML predictor: {e}")
    ml_predictor = None

def get_disease_description(disease_name):
    """
    Get detailed description of the disease
    Returns: dict with description, causes, symptoms, treatment info
    """
    descriptions = {
        'common cold': {
            'description': 'A viral infection of the upper respiratory tract (nose and throat). Usually harmless but uncomfortable.',
            'causes': 'Caused by viruses (rhinovirus most common). Spreads through airborne droplets and contact.',
            'symptoms': 'Runny/stuffy nose, sneezing, sore throat, cough, mild headache, fatigue',
            'treatment': 'Rest, fluids, over-the-counter cold medicines. Usually resolves in 7-10 days.'
        },
        'viral fever': {
            'description': 'Fever caused by viral infection. Body temperature rises to fight off the virus.',
            'causes': 'Various viruses including flu, dengue, or common cold viruses.',
            'symptoms': 'High temperature (>100.4°F), body aches, weakness, headache, chills',
            'treatment': 'Rest, plenty of fluids, fever reducers (acetaminophen/ibuprofen). See doctor if fever >3 days.'
        },
        'pneumonia': {
            'description': 'Lung infection causing inflammation of air sacs, which may fill with fluid or pus.',
            'causes': 'Bacteria, viruses, or fungi. Can be acquired in community or hospital.',
            'symptoms': 'Chest pain, cough with phlegm, fever, chills, difficulty breathing, fatigue',
            'treatment': 'IMPORTANT: See a doctor. Requires antibiotics for bacterial pneumonia. May need hospitalization.'
        },
        'malaria': {
            'description': 'Parasitic disease transmitted by infected mosquitoes. Common in tropical regions.',
            'causes': 'Plasmodium parasite transmitted through Anopheles mosquito bites.',
            'symptoms': 'High fever, chills, sweating, headache, nausea, vomiting, muscle pain',
            'treatment': 'URGENT: Requires immediate medical treatment with antimalarial drugs. Can be life-threatening.'
        },
        'dengue': {
            'description': 'Mosquito-borne viral disease common in tropical and subtropical areas.',
            'causes': 'Dengue virus transmitted by Aedes mosquitoes.',
            'symptoms': 'High fever, severe headache, pain behind eyes, joint/muscle pain, rash, mild bleeding',
            'treatment': 'No specific cure. Rest, fluids, pain relievers (avoid aspirin). See doctor immediately.'
        },
        'migraine': {
            'description': 'Neurological condition causing intense, throbbing headaches, often on one side of head.',
            'causes': 'Various triggers: stress, certain foods, hormonal changes, lack of sleep, bright lights.',
            'symptoms': 'Severe headache, nausea, vomiting, sensitivity to light/sound, visual disturbances (aura)',
            'treatment': 'Pain relievers, rest in dark quiet room, preventive medications for chronic cases.'
        },
        'diabetes': {
            'description': 'Chronic condition affecting how your body processes blood sugar (glucose).',
            'causes': 'Type 1: Autoimmune. Type 2: Insulin resistance, often related to lifestyle/genetics.',
            'symptoms': 'Increased thirst/hunger, frequent urination, fatigue, blurred vision, slow healing wounds',
            'treatment': 'CHRONIC: Requires ongoing management. Diet, exercise, blood sugar monitoring, medications/insulin.'
        },
        'hypertension': {
            'description': 'High blood pressure - force of blood against artery walls is too high.',
            'causes': 'Often no single cause. Risk factors: age, genetics, obesity, high salt intake, stress.',
            'symptoms': 'Often no symptoms ("silent killer"). May cause headaches, shortness of breath, nosebleeds.',
            'treatment': 'Lifestyle changes (diet, exercise, reduce salt/stress), blood pressure medications if needed.'
        },
        'typhoid': {
            'description': 'Bacterial infection causing high fever and gastrointestinal symptoms.',
            'causes': 'Salmonella typhi bacteria, spread through contaminated food or water.',
            'symptoms': 'Sustained high fever, weakness, stomach pain, headache, loss of appetite, rash',
            'treatment': 'IMPORTANT: Requires antibiotics. See doctor for proper diagnosis and treatment.'
        },
        'jaundice': {
            'description': 'Yellowing of skin and eyes due to high bilirubin levels in blood.',
            'causes': 'Liver problems, bile duct blockage, certain blood disorders, or hepatitis.',
            'symptoms': 'Yellow skin/eyes, dark urine, pale stools, itching, fatigue, abdominal pain',
            'treatment': 'Treat underlying cause. See doctor for diagnosis. May need liver function tests.'
        },
        'chicken pox': {
            'description': 'Highly contagious viral infection causing itchy rash with fluid-filled blisters.',
            'causes': 'Varicella-zoster virus. Spreads easily through air and contact.',
            'symptoms': 'Itchy rash (starts as red bumps, becomes blisters), fever, tiredness, loss of appetite',
            'treatment': 'Usually mild. Anti-itch lotions, fever reducers, rest. Vaccine available for prevention.'
        },
        'fungal infection': {
            'description': 'Skin infection caused by fungi, affecting skin, nails, or scalp.',
            'causes': 'Various fungi. Thrives in warm, moist environments. Can spread through contact.',
            'symptoms': 'Itchy, red, scaly patches, ring-shaped rash, cracked skin, nail discoloration',
            'treatment': 'Antifungal creams/powders for minor cases. Oral medications for severe/persistent infections.'
        },
        'arthritis': {
            'description': 'Joint inflammation causing pain and stiffness. Many types (osteoarthritis, rheumatoid, etc.)',
            'causes': 'Varies by type: wear and tear (osteo), autoimmune (rheumatoid), genetics.',
            'symptoms': 'Joint pain, stiffness (worse in morning), swelling, reduced range of motion',
            'treatment': 'Pain relievers, anti-inflammatory drugs, physical therapy, exercise, heat/cold therapy.'
        }
    }

    # Try to find exact match or partial match
    disease_lower = disease_name.lower()

    # Exact match
    if disease_lower in descriptions:
        return descriptions[disease_lower]

    # Partial match
    for key, value in descriptions.items():
        if key in disease_lower or disease_lower in key:
            return value

    # Default for unknown diseases
    return {
        'description': f'{disease_name} is a medical condition that requires professional evaluation.',
        'causes': 'Specific causes vary. Consult a healthcare provider for detailed information.',
        'symptoms': 'Symptoms depend on the specific condition and individual factors.',
        'treatment': 'Please consult a qualified medical professional for proper diagnosis and treatment plan.'
    }

def predict_disease(symptoms):
    """
    Call the Linear SVM predictor with symptoms using direct import
    """
    try:
        if ml_predictor is None:
            return {
                'success': False,
                'error': 'ML model not available'
            }

        # PRE-CHECK: Hardcoded Safety Rules (Step 10)
        # Rule overrides ML output
        symptoms_lower = symptoms.lower()
        safety_rules = {
            'swelling throat': 'Emergency: Potential Anaphylaxis or Airway Obstruction',
            'bone sticking out': 'Emergency: Open Fracture',
            'sudden paralysis': 'Emergency: Stroke or Spinal Injury',
            'chest pain': 'Emergency: Potential Cardiac Event',
            'difficulty breathing': 'Emergency: Respiratory Distress',
            'unconscious': 'Emergency: Loss of Consciousness',
            'severe bleeding': 'Emergency: Hemorrhage'
        }

        emergency_override = None
        for trigger, message in safety_rules.items():
            if trigger in symptoms_lower:
                emergency_override = message
                break

        if emergency_override:
            # Force emergency result
            result = {
                'disease': emergency_override,
                'confidence': 100.0,
                'urgency': 'CRITICAL'
            }
            is_emergency = True
            alternatives = []
        else:
            # Standard ML prediction
            result = ml_predictor.predict(symptoms, detailed=False)

            # Check for emergency keywords (secondary check)
            emergency_keywords = [
                'chest pain', 'difficulty breathing', 'severe headache', 'stroke',
                'heart attack', 'unconscious', 'bleeding', 'severe pain',
                'can\'t breathe', 'choking', 'seizure', 'overdose'
            ]
            is_emergency = any(keyword in symptoms_lower for keyword in emergency_keywords)
            alternatives = result.get('alternatives', [])[1:4] if len(result.get('alternatives', [])) > 1 else []

        # Generate recommendations
        rec_data = generate_recommendations(
            result.get('disease', 'Unknown'),
            result.get('confidence', 0),
            is_emergency
        )

        # Step 11: Final Prediction Output Format
        doctor_message = "Please consult a doctor immediately." if is_emergency else "If symptoms persist, please see a general practitioner."

        return {
            'success': True,
            'result': {
                'disease': result.get('disease', 'Unknown'),
                'confidence': result.get('confidence', 0),
                'model': 'LinearSVM',
                'urgencyLevel': 'EMERGENCY' if is_emergency else ('MILD' if result.get('confidence', 0) < 60 else 'MODERATE')
            },
            'severity': 'Emergency' if is_emergency else ('Mild' if result.get('confidence', 0) < 60 else 'Moderate'),
            'predictedDiagnosis': result.get('disease', 'Unknown'),
            'confidenceScore': result.get('confidence', 0),
            'emergencyFlag': is_emergency,
            'precautions': rec_data['precautions'],
            'doctorConsultationMessage': doctor_message,
            'alternatives': alternatives,
            'recommendations': rec_data['actions'],
            'dietPlan': rec_data['diet'],
            'exercisePlan': rec_data['exercise'],
            'isEmergency': is_emergency,
            'emergencyKeywords': [kw for kw in safety_rules.keys() if kw in symptoms_lower] if is_emergency else [],
            'diseaseInfo': get_disease_description(result.get('disease', 'Unknown'))
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }

# --- In-Memory Fallback for when MongoDB is missing ---
MOCK_USERS = {}
MOCK_PREDICTIONS = {}

class MockDB:
    def __init__(self):
        self.users = MockCollection(MOCK_USERS)
        self.predictions = MockCollection(MOCK_PREDICTIONS)

class MockCollection:
    def __init__(self, storage):
        self.storage = storage

    def find_one(self, query):
        for item in self.storage.values():
            if all(item.get(k) == v for k, v in query.items() if k != '_id'):
                # Handle _id separately if it's in query
                if '_id' in query and str(item['_id']) != str(query['_id']):
                    continue
                return item
        return None

    def insert_one(self, doc):
        doc['_id'] = ObjectId()
        self.storage[str(doc['_id'])] = doc
        class Result:
            def __init__(self, inserted_id):
                self.inserted_id = inserted_id
        return Result(doc['_id'])

    def update_one(self, query, update):
        item = self.find_one(query)
        if item:
            if '$set' in update:
                item.update(update['$set'])
            if '$inc' in update:
                for k, v in update['$inc'].items():
                    item[k] = item.get(k, 0) + v
            return type('obj', (object,), {'matched_count': 1})
        return type('obj', (object,), {'matched_count': 0})

    def find(self, query):
        results = []
        for item in self.storage.values():
             match = True
             for k, v in query.items():
                 if k == '_id':
                     if str(item['_id']) != str(v):
                         match = False; break
                 elif item.get(k) != v:
                     match = False; break
             if match:
                 results.append(item)

        # Mock sort/skip/limit by returning a list wrapper that supports them vaguely
        # or just returning the list
        class MockCursor(list):
            def sort(self, *args, **kwargs): return self
            def skip(self, *args, **kwargs): return self
            def limit(self, *args, **kwargs): return self

        return MockCursor(results)

    def count_documents(self, query):
        return len(self.find(query))

    def aggregate(self, pipeline):
        # Very basic mock aggregation for dashboard stats
        return [{
            'totalPredictions': len(MOCK_PREDICTIONS),
            'emergencyPredictions': sum(1 for p in MOCK_PREDICTIONS.values() if p.get('isEmergency')),
            'avgConfidence': 80.0,
            'recentPredictions': []
        }]

# Initialize DB Wrapper
db = mongo.db if mongo_available else MockDB()

def generate_recommendations(disease, confidence, is_emergency):

    """
    Generate medical recommendations based on disease and confidence
    """
    recommendations = []

    if is_emergency:
        recommendations.extend([
            {
                'type': 'emergency',
                'description': '🚨 SEEK IMMEDIATE MEDICAL ATTENTION - Call 911 or go to nearest emergency room',
                'priority': 'critical'
            },
            {
                'type': 'emergency',
                'description': 'Do not delay medical care - this could be life-threatening',
                'priority': 'critical'
            }
        ])

    # General recommendations based on confidence
    if confidence >= 80:
        recommendations.append({
            'type': 'consultation',
            'description': 'High confidence prediction - consult healthcare provider for confirmation',
            'priority': 'high'
        })
    elif confidence >= 60:
        recommendations.append({
            'type': 'consultation',
            'description': 'Moderate confidence - consider consulting healthcare provider',
            'priority': 'medium'
        })
    else:
        recommendations.append({
            'type': 'consultation',
            'description': 'Low confidence - multiple conditions possible, see healthcare provider',
            'priority': 'medium'
        })

    # Disease-specific recommendations database
    disease_lower = disease.lower()

    # Initialize plan lists
    diet = []
    workouts = []
    precautions = []

    if 'headache' in disease_lower or 'migraine' in disease_lower:
        recommendations.extend([
            {'type': 'self-care', 'description': 'Rest in a quiet, dark room to reduce sensory overload', 'priority': 'medium'},
            {'type': 'self-care', 'description': 'Apply a cold or warm compress to head or neck', 'priority': 'medium'}
        ])
        diet = ["Ginger tea (anti-inflammatory)", "Magnesium-rich foods (spinach, almonds)", "Stay hydrated with water", "Avoid caffeine and alcohol"]
        workouts = ["Gentle neck stretching", "Yoga (Child's Pose)", "Deep breathing exercises", "Avoid high-intensity cardio"]
        precautions = ["Avoid bright screens and loud noises", "Maintain a regular sleep schedule", "Monitor blood pressure", "Track potential triggers"]

    elif 'fever' in disease_lower or 'flu' in disease_lower:
        recommendations.extend([
            {'type': 'self-care', 'description': 'Get plenty of rest to help your body fight infection', 'priority': 'high'},
            {'type': 'medication', 'description': 'Take over-the-counter fever reducers if uncomfortable', 'priority': 'medium'}
        ])
        diet = ["Chicken soup or bone broth", "Coconut water for electrolytes", "Herbal tea with honey", "Soft fruits like bananas"]
        workouts = ["Complete rest is recommended", "Avoid exercise until fever resolves", "Very gentle walking only if energy permits"]
        precautions = ["Isolate to prevent spreading infection", "Monitor temperature regularly", "Wash hands frequently", "Wear a mask if around others"]

    elif 'cold' in disease_lower:
        recommendations.extend([
            {'type': 'self-care', 'description': 'Use a humidifier to keep air moist', 'priority': 'medium'},
            {'type': 'self-care', 'description': 'Gargle with warm salt water', 'priority': 'low'}
        ])
        diet = ["Warm soup", "Citrus fruits (Vitamin C)", "Garlic and onions", "Hot tea with lemon"]
        workouts = ["Light walking if symptoms are mild", "Gentle stretching", "Avoid strenuous activities"]
        precautions = ["Cover coughs and sneezes", "Disinfect frequently touched surfaces", "Rest your voice"]

    elif 'diabetes' in disease_lower:
        diet = ["Leafy greens (spinach, kale)", "Fatty fish (salmon, mackerel)", "Whole grains (quinoa, brown rice)", "Avoid sugary drinks and processed carbs"]
        workouts = ["Brisk walking (30 mins daily)", "Resistance/Strength training", "Swimming", "Cycling"]
        precautions = ["Monitor blood sugar levels frequently", "Check feet for cuts or sores daily", "Stay hydrated", "Carry an emergency snack"]

    else:
        # Generic / Default Plan
        diet = ["Balanced diet with plenty of vegetables", "Lean proteins", "Whole grains", "Stay well-hydrated"]
        workouts = ["Moderate walking (20-30 mins)", "Gentle stretching", "Yoga for relaxation"]
        precautions = ["Monitor symptoms closely", "Get adequate sleep", "Consult doctor if symptoms persist"]

    # Always add general advice
    recommendations.append({
        'type': 'disclaimer',
        'description': '⚠️ This is an AI prediction. Always consult a qualified healthcare professional for medical advice.',
        'priority': 'high'
    })

    return {
        'actions': recommendations,
        'diet': diet,
        'exercise': workouts,
        'precautions': precautions
    }

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()

        # Validation
        required_fields = ['name', 'email', 'password', 'phone', 'dateOfBirth', 'gender']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field} is required'}), 400

        # Check if user exists
        if db.users.find_one({'email': data['email']}):
            return jsonify({'error': 'User already exists'}), 400

        # Hash password
        hashed_password = generate_password_hash(data['password'])

        # Create user document
        user_doc = {
            'name': data['name'],
            'email': data['email'].lower(),
            'password': hashed_password,
            'phone': data['phone'],
            'dateOfBirth': datetime.fromisoformat(data['dateOfBirth'].replace('Z', '+00:00')),
            'gender': data['gender'],
            'address': data.get('address', ''),
            'emergencyContact': data.get('emergencyContact', ''),
            'medicalHistory': data.get('medicalHistory', ''),
            'isActive': True,
            'isVerified': False,
            'role': 'user',
            'totalPredictions': 0,
            'createdAt': datetime.utcnow(),
            'updatedAt': datetime.utcnow()
        }

        # Insert user
        result = db.users.insert_one(user_doc)
        user_id = str(result.inserted_id)

        # Create access token
        access_token = create_access_token(identity=user_id)

        # Return user data (without password)
        user_doc.pop('password')
        user_doc['_id'] = user_id

        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'user': serialize_doc(user_doc),
            'token': access_token
        }), 201

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()

        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400

        # Find user
        user = db.users.find_one({'email': data['email'].lower()})

        if not user or not check_password_hash(user['password'], data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401

        if not user.get('isActive', True):
            return jsonify({'error': 'Account is deactivated'}), 401

        # Update last login
        db.users.update_one(
            {'_id': user['_id']},
            {'$set': {'lastLogin': datetime.utcnow()}}
        )

        # Create access token
        access_token = create_access_token(identity=str(user['_id']))

        # Return user data (without password)
        user.pop('password')

        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': serialize_doc(user),
            'token': access_token
        }), 200

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

# User Routes
@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_profile():
    try:
        user_id = get_jwt_identity()
        user = db.users.find_one({'_id': ObjectId(user_id)})

        if not user:
            return jsonify({'error': 'User not found'}), 404

        user.pop('password', None)
        return jsonify({
            'success': True,
            'user': serialize_doc(user)
        }), 200

    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        return jsonify({'error': 'Failed to get profile'}), 500

@app.route('/api/user/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()

        # Remove sensitive fields
        data.pop('password', None)
        data.pop('_id', None)
        data['updatedAt'] = datetime.utcnow()

        # Update user
        result = db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': data}
        )

        if result.matched_count == 0:
            return jsonify({'error': 'User not found'}), 404

        # Get updated user
        user = db.users.find_one({'_id': ObjectId(user_id)})
        user.pop('password', None)

        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': serialize_doc(user)
        }), 200

    except Exception as e:
        logger.error(f"Update profile error: {str(e)}")
        return jsonify({'error': 'Failed to update profile'}), 500

# Prediction Routes
@app.route('/api/prediction/predict', methods=['POST'])
@jwt_required(optional=True)
def make_prediction():
    try:
        user_id = get_jwt_identity()  # Will be None if no token provided
        data = request.get_json()

        if not data.get('symptoms'):
            return jsonify({'error': 'Symptoms are required'}), 400

        symptoms = data['symptoms'].strip()
        if len(symptoms) < 3:
            return jsonify({'error': 'Please provide more detailed symptoms'}), 400

        # Record start time
        start_time = datetime.utcnow()

        # Make prediction
        prediction_result = predict_disease(symptoms)

        if not prediction_result['success']:
            return jsonify({'error': prediction_result['error']}), 500

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Save prediction to database only if MongoDB is available and user is logged in
        prediction_id = None
        if user_id: # Always try to save using the db wrapper
            try:
                prediction_doc = {
                    'userId': ObjectId(user_id),
                    'symptoms': symptoms,
                    'result': prediction_result['result'],
                    'alternatives': prediction_result.get('alternatives', []),
                    'recommendations': prediction_result.get('recommendations', []),
                    'precautions': prediction_result.get('precautions', []),
                    'dietPlan': prediction_result.get('dietPlan', []),
                    'exercisePlan': prediction_result.get('exercisePlan', []),
                    'isEmergency': prediction_result.get('isEmergency', False),
                    'emergencyKeywords': prediction_result.get('emergencyKeywords', []),
                    'doctorConsultationMessage': prediction_result.get('doctorConsultationMessage', ''),
                    'processingTime': processing_time,
                    'status': 'completed',
                    'createdAt': datetime.utcnow(),
                    'updatedAt': datetime.utcnow()
                }

                result = db.predictions.insert_one(prediction_doc)
                prediction_id = str(result.inserted_id)

                # Update user's total predictions count
                db.users.update_one(
                    {'_id': ObjectId(user_id)},
                    {'$inc': {'totalPredictions': 1}}
                )
            except Exception as db_error:
                logger.warning(f"Failed to save to database: {db_error}")

        # Add prediction ID to response (will be None for guest users)
        prediction_result['predictionId'] = prediction_id
        prediction_result['processingTime'] = processing_time

        return jsonify({
            'success': True,
            'message': 'Prediction completed successfully',
            **prediction_result
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/prediction/history', methods=['GET'])
@jwt_required()
def get_prediction_history():
    try:
        user_id = get_jwt_identity()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        skip = (page - 1) * limit

        # Get predictions with pagination
        predictions = list(db.predictions.find(
            {'userId': ObjectId(user_id)}
        ).sort('createdAt', -1).skip(skip).limit(limit))

        # Get total count
        total = db.predictions.count_documents({'userId': ObjectId(user_id)})

        return jsonify({
            'success': True,
            'predictions': serialize_doc(predictions),
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'pages': (total + limit - 1) // limit
            }
        }), 200

    except Exception as e:
        logger.error(f"Get history error: {str(e)}")
        return jsonify({'error': 'Failed to get prediction history'}), 500

@app.route('/api/prediction/<prediction_id>', methods=['GET'])
@jwt_required()
def get_prediction(prediction_id):
    try:
        user_id = get_jwt_identity()

        prediction = db.predictions.find_one({
            '_id': ObjectId(prediction_id),
            'userId': ObjectId(user_id)
        })

        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404

        return jsonify({
            'success': True,
            'prediction': serialize_doc(prediction)
        }), 200

    except Exception as e:
        logger.error(f"Get prediction error: {str(e)}")
        return jsonify({'error': 'Failed to get prediction'}), 500

@app.route('/api/prediction/<prediction_id>/feedback', methods=['POST'])
@jwt_required()
def add_feedback(prediction_id):
    try:
        user_id = get_jwt_identity()
        data = request.get_json()

        feedback = {
            'rating': data.get('rating'),
            'helpful': data.get('helpful'),
            'comments': data.get('comments', ''),
            'actualDiagnosis': data.get('actualDiagnosis', ''),
            'submittedAt': datetime.utcnow()
        }

        result = db.predictions.update_one(
            {'_id': ObjectId(prediction_id), 'userId': ObjectId(user_id)},
            {'$set': {'userFeedback': feedback, 'updatedAt': datetime.utcnow()}}
        )

        if result.matched_count == 0:
            return jsonify({'error': 'Prediction not found'}), 404

        return jsonify({
            'success': True,
            'message': 'Feedback added successfully'
        }), 200

    except Exception as e:
        logger.error(f"Add feedback error: {str(e)}")
        return jsonify({'error': 'Failed to add feedback'}), 500

# Dashboard Routes
@app.route('/api/dashboard/stats', methods=['GET'])
@jwt_required()
def get_dashboard_stats():
    try:
        user_id = get_jwt_identity()

        # Get user's prediction statistics
        pipeline = [
            {'$match': {'userId': ObjectId(user_id)}},
            {'$group': {
                '_id': None,
                'totalPredictions': {'$sum': 1},
                'emergencyPredictions': {
                    '$sum': {'$cond': [{'$eq': ['$isEmergency', True]}, 1, 0]}
                },
                'avgConfidence': {'$avg': '$result.confidence'},
                'recentPredictions': {'$push': {
                    'disease': '$result.disease',
                    'confidence': '$result.confidence',
                    'createdAt': '$createdAt',
                    'isEmergency': '$isEmergency'
                }}
            }}
        ]

        stats = list(db.predictions.aggregate(pipeline))

        if not stats:
            stats = [{
                'totalPredictions': 0,
                'emergencyPredictions': 0,
                'avgConfidence': 0,
                'recentPredictions': []
            }]

        # Get recent predictions (last 5)
        recent_predictions = list(db.predictions.find(
            {'userId': ObjectId(user_id)}
        ).sort('createdAt', -1).limit(5))

        return jsonify({
            'success': True,
            'stats': {
                'totalPredictions': stats[0]['totalPredictions'],
                'emergencyPredictions': stats[0]['emergencyPredictions'],
                'avgConfidence': round(stats[0]['avgConfidence'] or 0, 2),
                'recentPredictions': serialize_doc(recent_predictions)
            }
        }), 200

    except Exception as e:
        logger.error(f"Get dashboard stats error: {str(e)}")
        return jsonify({'error': 'Failed to get dashboard stats'}), 500

# Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'message': 'Medical Healthcare Flask API is running',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200

# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Authorization token is required'}), 401

# --- GitHub Auth Routes ---
@app.route('/auth/github')
def github_login():
    if not github:
        return jsonify({'error': 'GitHub Auth not enabled. Please install authlib and configure credentials.'}), 501
    redirect_uri = url_for('github_authorize', _external=True)
    return github.authorize_redirect(redirect_uri)

@app.route('/auth/github/callback')
def github_authorize():
    if not github:
        return jsonify({'error': 'GitHub Auth not enabled.'}), 501

    try:
        # Verify state and token
        token = github.authorize_access_token()
        if not token:
             return jsonify({'error': 'Access denied'}), 401

        resp = github.get('user').json()
        email_resp = github.get('user/emails').json()

        # Get primary email
        email = next((e['email'] for e in email_resp if e['primary']), None)
        if not email:
            email = resp.get('email') # Fallback

        name = resp.get('name') or resp.get('login')

        # Find or Create User in MongoDB
        users = mongo.db.users
        user = users.find_one({'email': email})

        if not user:
            # Create new user
            users.insert_one({
                'name': name,
                'email': email,
                'password': '', # No password for OAuth users
                'created_at': datetime.utcnow(),
                'provider': 'github'
            })

        # Create JWT
        access_token = create_access_token(identity=email)

        # Redirect to Dashboard with token (Insecure for prod, okay for demo)
        # We append token to URL so frontend can grab it
        return redirect(f'/dashboard.html?token={access_token}&user={name}&email={email}')

    except Exception as e:
        logger.error(f"GitHub Auth Error: {str(e)}")
        # For demo purposes, if it fails (e.g. no internet/bad creds), redirect to login with error
        return redirect('/login.html?error=GitHub_Login_Failed')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
