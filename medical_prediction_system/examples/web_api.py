"""
=================================================================================
WEB API EXAMPLE - MEDICAL PREDICTION SYSTEM
=================================================================================

Flask web application and REST API for medical disease prediction
Provides both web interface and API endpoints

Author: ML Engineering Team
Date: December 2024
Version: 3.0 - Production Example
=================================================================================
"""

import sys
import os
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime
import logging

# Add parent directory to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.linear_svm_predictor import LinearSVMMedicalPredictor

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor globally (load once)
try:
    predictor = LinearSVMMedicalPredictor()
    logger.info("✅ Medical predictor loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load predictor: {e}")
    predictor = None

# HTML template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Disease Predictor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #34495e;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
        }
        textarea:focus {
            border-color: #3498db;
            outline: none;
        }
        button {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            transition: background 0.3s;
        }
        button:hover {
            background: linear-gradient(135deg, #2980b9, #21618c);
        }
        .result {
            margin-top: 30px;
            padding: 25px;
            border-radius: 8px;
            border-left: 5px solid;
        }
        .emergency {
            background-color: #ffebee;
            border-left-color: #f44336;
            color: #c62828;
        }
        .urgent {
            background-color: #fff3e0;
            border-left-color: #ff9800;
            color: #ef6c00;
        }
        .high {
            background-color: #fff8e1;
            border-left-color: #ffc107;
            color: #f57f17;
        }
        .normal {
            background-color: #e8f5e8;
            border-left-color: #4caf50;
            color: #2e7d32;
        }
        .result h2 {
            margin-top: 0;
            display: flex;
            align-items: center;
        }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f, #2ecc71);
            transition: width 0.5s ease;
        }
        .alternatives {
            margin-top: 15px;
        }
        .alternatives h3 {
            margin-bottom: 10px;
        }
        .alternative-item {
          background: #f8f9fa;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }
        .recommendations {
            margin-top: 15px;
        }
        .recommendations ul {
            padding-left: 20px;
        }
        .recommendations li {
            margin: 8px 0;
            line-height: 1.4;
        }
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            font-size: 14px;
        }
        .api-info {
            background-color: #e3f2fd;
            border: 1px solid #bbdefb;
            color: #1565c0;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            font-size: 14px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Medical Disease Predictor</h1>
        <p class="subtitle">🎯 97.23% Accuracy | Linear SVM | Production Ready</p>

        <form method="POST" onsubmit="showLoading()">
            <div class="form-group">
                <label for="symptoms">Describe your symptoms in detail:</label>
                <textarea
                    name="symptoms"
                    id="symptoms"
                    placeholder="Example: I have been experiencing fever and headache for the past 2 days, along with body aches and fatigue..."
                    required>{{ request.form.get('symptoms', '') }}</textarea>
            </div>
            <button type="submit">🔍 Analyze Symptoms</button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing symptoms...</p>
        </div>

        {% if result %}
        <div class="result {{ 'emergency' if result.urgency == 'EMERGENCY' else 'urgent' if result.urgency == 'URGENT' else 'high' if result.urgency == 'HIGH' else 'normal' }}">
            <h2>
                {% if result.urgency == 'EMERGENCY' %}🚨{% elif result.urgency in ['URGENT', 'HIGH'] %}⚠️{% else %}✅{% endif %}
                Prediction Result
            </h2>

            <p><strong>Primary Diagnosis:</strong> {{ result.disease }}</p>

            <p><strong>Confidence Level:</strong></p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {{ result.confidence }}%"></div>
            </div>
            <p style="text-align: center; margin: 5px 0;">{{ result.confidence }}%</p>

            <p><strong>Urgency Level:</strong> {{ result.urgency }}</p>
            <p><strong>Model:</strong> {{ result.model }} ({{ result.accuracy }})</p>

            {% if result.urgency == 'EMERGENCY' %}
            <div style="background: #ffcdd2; padding: 15px; border-radius: 5px; margin: 15px 0; text-align: center;">
                <h3 style="color: #d32f2f; margin: 0;">🚨 MEDICAL EMERGENCY 🚨</h3>
                <p style="margin: 10px 0; font-weight: bold;">CALL 911 IMMEDIATELY</p>
                <p style="margin: 0;">Do not drive yourself to the hospital</p>
            </div>
            {% elif result.consult_doctor %}
            <div style="background: #ffe0b2; padding: 15px; border-radius: 5px; margin: 15px 0; text-align: center;">
                <h3 style="color: #ef6c00; margin: 0;">⚠️ Medical Consultation Recommended</h3>
                <p style="margin: 10px 0;">Please consult a healthcare professional</p>
            </div>
            {% endif %}

            {% if result.alternatives and result.alternatives|length > 1 %}
            <div class="alternatives">
                <h3>🔄 Alternative Diagnoses:</h3>
                {% for alt in result.alternatives[:3] %}
                <div class="alternative-item">
                    {{ loop.index }}. {{ alt.disease }} ({{ "%.2f"|format(alt.confidence) }}%)
                </div>
                {% endfor %}
            </div>
            {% endif %}

            <div class="recommendations">
                <h3>💊 Medical Recommendations:</h3>
                <ul>
                {% for rec in result.recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
       </div>
        {% endif %}

        <div class="disclaimer">
            <strong>⚠️ Important Medical Disclaimer:</strong><br>
            This is an AI prediction system for preliminary assessment only. It is NOT a replacement for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice. In emergencies, call 911 or go to the nearest emergency room immediately.
        </div>

        <div class="api-info">
            <strong>🔌 API Access:</strong><br>
            Developers can access the prediction API at <code>POST /api/predict</code><br>
            Send JSON: <code>{"symptoms": "your symptoms here"}</code>
        </div>
    </div>

    <script>
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }

        // Auto-resize textarea
        const textarea = document.getElementById('symptoms');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    """Mainterface"""
    result = None

    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip()

        if symptoms and predictor:
            try:
                result = predictor.predict(symptoms, detailed=True)
                logger.info(f"Prediction made: {symptoms[:50]}... → {result['disease']}")
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                result = {
                    'disease': 'System Error',
                    'confidence': 0.0,
                    'urgency': 'UNKNOWN',
                    'consult_doctor': True,
                    'alternatives': [],
                    'recommendations': ['System error occurred. Please try again or consult a doctor.'],
                    'model': 'ERROR',
                    'accuracy': 'N/A'
                }

    return render_template_string(HTML_TEMPLATE, result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions"""
    try:
        # Check if predictor is available
        if not predictor:
            return jsonify({
                'error': 'Prediction model not available',
                'message': 'Model failed to load during startup'
            }), 503

        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Send JSON with symptoms field'
            }), 400

        # Extract symptoms
        symptoms = data.get('symptoms', '').strip()
        if not symptoms:
            return jsonify({
                'error': 'No symptoms provided',
                'message': 'Include symptoms field in JSON data'
            }), 400

        # Get detailed flag
        detailed = data.get('detailed', False)

        # Make prediction
        result = predictor.predict(symptoms, detailed=detailed)

        # Add metadata
        result['api_version'] = '3.0'
        result['timestamp'] = datetime.now().isoformat()
        result['model_accuracy'] = '97.23%'

        # Log API usage
        logger.info(f"API prediction: {symptoms[:50]}... → {result['disease']} ({result['confidence']:.2f}%)")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if predictor else 'unhealthy',
        'model_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '3.0',
        'accuracy': '97.23%' if predictor else 'N/A'
    }

    return jsonify(status), 200 if predictor else 503

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint"""
    info = {
        'name': 'Medical Disease Prediction API',
        'version': '3.0',
        'model': 'Linear SVM',
        'accuracy': '97.23%',
        'diseases_supported': 242 if predictor else 0,
        'features': 300 if predictor else 0,
        'endpoints': {
            'predict': 'POST /api/predict',
            'health': 'GET /api/health',
            'info': 'GET /api/info'
        },
        'example_request': {
            'url': '/api/predict',
            'method': 'POST',
            'headers': {'Content-Type': 'application/json'},
            'body': {'symptoms': 'fever and headache for 2 days', 'detailed': True}
        }
    }

    return jsonify(info), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'Available endpoints: /, /api/predict, /api/health, /api/info'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please try again later'
    }), 500

def main():
    """Main function to run the web application"""

    print("🏥 MEDICAL PREDICTION WEB API")
    print("=" * 50)

    if not predictor:
        print("❌ Failed to load prediction model")
        print("Please ensure model files are available in models/ directory")
        return

    print("✅ Medical predictor loaded successfully")
    print(f"📊 Model: Linear SVM (97.23% accuracy)")
    print(f"🏥 Diseases: 242 conditions supported")
    print(f"🔢 Features: 300 TF-IDF features")

    print("\n🌐 Starting web server...")
    print("📍 Web Interface: http://localhost:5000")
    print("🔌 API Endpoint: http://localhost:5000/api/predict")
    print("❤️ Health Check: http://localhost:5000/api/health")
    print("ℹ️ API Info: http://localhost:5000/api/info")

    print("\n📖 API Usage Example:")
    print("curl -X POST http://localhost:5000/api/predict \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"symptoms\": \"fever and headache\", \"detailed\": true}'")

    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 50)

    # Run the Flask app
    try:
        app.run(
            host='0.0.0.0',  # Accept connections from any IP
            port=5000,       # Port number
            debug=False,     # Disable debug mode for production
            threaded=True    # Handle multiple requests
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()
