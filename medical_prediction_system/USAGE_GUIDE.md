# 📖 Medical Disease Prediction System - Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Training Your Own Model](#training-your-own-model)
6. [API Integration](#api-integration)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### 1. Ensure You Have the Required Files
```
medical_prediction_system/
├── final_model.pkl          (Required)
├── tfidf_vectorizer.pkl     (Required)
├── label_encoder.pkl        (Required)
├── predict_disease.py       (Required)
├── requirements.txt
└── README.md
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn
```

### 3. Make a Prediction
```bash
python predict_disease.py "I have fever and headache for 2 days"
```

---

## 💾 Installation

### Method 1: Minimal (Inference Only)
```bash
# Install only what's needed for predictions
pip install pandas numpy scikit-learn

# Test it works
python predict_disease.py "test symptoms"
```

### Method 2: Full Installation (Training + Inference)
```bash
# Install all dependencies
pip install -r requirements.txt

# Or manually
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Method 3: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv medical_ai_env

# Activate it
# On Windows:
medical_ai_env\Scripts\activate
# On Mac/Linux:
source medical_ai_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📱 Basic Usage

### Command Line Interface

**Basic Prediction:**
```bash
python predict_disease.py "fever and body ache"
```

**Multiple Word Symptoms:**
```bash
python predict_disease.py "I have stomach pain after eating and nausea"
```

**With Quotes (Recommended):**
```bash
python predict_disease.py "chest pain, difficulty breathing, sweating"
```

### Python Script

**Simple Prediction:**
```python
from predict_disease import predict, display_result

# Get prediction
result = predict("fever, headache, body ache")

# Display formatted result
display_result(result)

# Access specific fields
print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Urgent: {result['urgent']}")
```

**Custom Display:**
```python
from predict_disease import predict

result = predict("stomach pain after eating")

# Access result fields
disease = result['disease']
confidence = result['confidence']
alternatives = result['alternatives']
precautions = result['precautions']

# Your custom logic
if result['urgent']:
    print("⚠️ SEEK EMERGENCY CARE!")
elif confidence > 80:
    print(f"High confidence: {disease}")
else:
    print("Uncertain - consult a doctor")
```

---

## 🎓 Advanced Usage

### Batch Processing

```python
from predict_disease import predict

# List of symptoms
symptoms_list = [
    "fever and headache",
    "chest pain and difficulty breathing",
    "stomach pain and nausea",
    "back pain and stiffness"
]

# Process all
results = []
for symptoms in symptoms_list:
    result = predict(symptoms)
    results.append({
        'symptoms': symptoms,
        'disease': result['disease'],
        'confidence': result['confidence']
    })

# Display summary
for r in results:
    print(f"{r['symptoms'][:30]:30} → {r['disease']:20} ({r['confidence']:.1f}%)")
```

**Output:**
```
fever and headache             → Headache            (99.2%)
chest pain and difficulty b... → Heart Disease       (45.3%)
stomach pain and nausea        → Abdominal Pain      (78.5%)
back pain and stiffness        → Back Pain           (92.1%)
```

### Using the Complete System

```python
from COMPLETE_PRODUCTION_SYSTEM import MedicalPredictor

# Initialize predictor
predictor = MedicalPredictor(
    model_path='final_model.pkl',
    vectorizer_path='tfidf_vectorizer.pkl',
    encoder_path='label_encoder.pkl'
)

# Make prediction
result = predictor.predict("severe headache with vomiting")

# Formatted output
print(predictor.format_prediction(result))

# Access detailed info
print(f"\nDisease: {result['disease']}")
print(f"Confidence: {result['confidence']}%")
print(f"Urgency: {result['urgency']}")
print(f"Alternative diagnoses:")
for alt in result['alternative_diagnoses']:
    print(f"  - {alt['disease']}: {alt['confidence']*100:.2f}%")
```

### CSV File Processing

```python
import pandas as pd
from predict_disease import predict

# Read symptoms from CSV
df = pd.read_csv('patient_symptoms.csv')

# Process each row
predictions = []
for idx, row in df.iterrows():
    result = predict(row['symptoms'])
    predictions.append({
        'patient_id': row['patient_id'],
        'symptoms': row['symptoms'],
        'predicted_disease': result['disease'],
        'confidence': result['confidence'],
        'urgent': result['urgent']
    })

# Save results
results_df = pd.DataFrame(predictions)
results_df.to_csv('prediction_results.csv', index=False)
print(f"Processed {len(predictions)} patients")
```

### Web Application (Flask)

```python
from flask import Flask, request, jsonify, render_template_string
from predict_disease import predict

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Medical Predictor</title></head>
<body>
    <h1>Medical Disease Predictor</h1>
    <form method="POST">
        <textarea name="symptoms" placeholder="Enter symptoms..." rows="5" cols="50"></textarea><br>
        <button type="submit">Predict</button>
    </form>
    {% if result %}
    <h2>Result</h2>
    <p><strong>Disease:</strong> {{ result.disease }}</p>
    <p><strong>Confidence:</strong> {{ result.confidence }}%</p>
    {% if result.urgent %}<p style="color:red;"><strong>⚠️ URGENT - SEEK EMERGENCY CARE</strong></p>{% endif %}
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        result = predict(symptoms)
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Run: `python app.py` and visit `http://localhost:5000`

---

## 🎯 Training Your Own Model

### Using Provided Training Script

```python
from COMPLETE_PRODUCTION_SYSTEM import main_training_pipeline

# Your CSV must have these columns:
# - Symptoms (text description)
# - Final Recommendation (disease name)
# Optional: Gender, Age, Duration, Severity

# Train the model
model, vectorizer, encoder = main_training_pipeline('your_data.csv')

# This will create:
# - final_model.pkl
# - tfidf_vectorizer.pkl
# - label_encoder.pkl
# - confusion_matrix.png
```

### Custom Training Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Load your data
df = pd.read_csv('your_medical_data.csv')

# Prepare features and labels
X = df['Symptoms'].values
y = df['Disease'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train model
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# Save model
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('my_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('my_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
```

---

## 🔌 API Integration

### REST API (Flask)

```python
from flask import Flask, request, jsonify
from predict_disease import predict

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    POST /api/predict
    Body: {"symptoms": "fever and headache"}
    """
    try:
        data = request.json
        symptoms = data.get('symptoms', '')
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        result = predict(symptoms)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

**Test the API:**
```bash
# Start server
python api.py

# Test with curl
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "fever and headache"}'
```

### FastAPI (Modern Alternative)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict_disease import predict

app = FastAPI(title="Medical Predictor API")

class SymptomRequest(BaseModel):
    symptoms: str

@app.post("/predict")
async def predict_disease(request: SymptomRequest):
    try:
        result = predict(request.symptoms)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run: uvicorn api_fast:app --reload --port 8000
```

---

## 🔧 Troubleshooting

### Problem: "FileNotFoundError: final_model.pkl"

**Solution:**
```bash
# Make sure you're in the correct directory
cd /path/to/medical_prediction_system

# Check files exist
ls *.pkl

# If missing, you need to train the model first
python COMPLETE_PRODUCTION_SYSTEM.py
```

### Problem: "ModuleNotFoundError: No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn
# Or for full installation
pip install -r requirements.txt
```

### Problem: Low Confidence Predictions

**Solution:**
```python
from predict_disease import predict

result = predict("your symptoms")

if result['confidence'] < 60:
    print("⚠️ Low confidence - Consider:")
    print("1. Provide more detailed symptoms")
    print("2. Consult a medical professional")
    print("\nAlternative possibilities:")
    for alt in result['alternatives'][:3]:
        print(f"  - {alt[0]}: {alt[1]:.2f}%")
```

### Problem: Urgent Symptoms Not Detected

**Check manually:**
```python
from predict_disease import URGENT_KEYWORDS

symptoms = "your symptoms".lower()
is_urgent = any(keyword in symptoms for keyword in URGENT_KEYWORDS)

if is_urgent:
    print("⚠️ URGENT - SEEK EMERGENCY CARE")
```

### Problem: Model Performance Issues

**Retrain with more data:**
```python
# Collect more samples for underperforming diseases
# Add to your dataset
# Retrain model
from COMPLETE_PRODUCTION_SYSTEM import main_training_pipeline
model, vectorizer, encoder = main_training_pipeline('expanded_data.csv')
```

---

## 📊 Performance Optimization

### For Faster Inference

```python
# Preload model (load once, use many times)
from predict_disease import model, vectorizer, encoder

def fast_predict(symptoms):
    symptoms_tfidf = vectorizer.transform([symptoms.lower()])
    prediction = model.predict(symptoms_tfidf)[0]
    disease = encoder.inverse_transform([prediction])[0]
    confidence = model.predict_proba(symptoms_tfidf)[0][prediction] * 100
    return disease, confidence

# Use in loop
for symptoms in symptom_list:
    disease, conf = fast_predict(symptoms)
    print(f"{disease}: {conf:.2f}%")
```

### For Production Deployment

```python
# Use model caching
import functools

@functools.lru_cache(maxsize=1000)
def cached_predict(symptoms):
    # Predictions for same symptoms are cached
    return predict(symptoms)
```

---

## 📚 Best Practices

1. **Always validate input:**
   ```python
   if len(symptoms.strip()) < 5:
       print("Please provide more detailed symptoms")
   ```

2. **Handle errors gracefully:**
   ```python
   try:
       result = predict(symptoms)
   except Exception as e:
       print(f"Error: {e}")
       print("Please try again or consult a doctor")
   ```

3. **Include disclaimers:**
   ```python
   print("⚠️ This is AI prediction, not medical advice")
   print("Always consult healthcare professionals")
   ```

4. **Log predictions:**
   ```python
   import logging
   logging.info(f"Predicted {result['disease']} with {result['confidence']}%")
   ```

5. **Monitor confidence:**
   ```python
   if result['confidence'] < 60:
       # Flag for review
       # Suggest doctor consultation
   ```

---

## 🎓 Example Use Cases

### Use Case 1: Telemedicine Triage
```python
def triage_patient(symptoms):
    result = predict(symptoms)
    
    if result['urgent']:
        return "EMERGENCY - Direct to ER"
    elif result['confidence'] > 80:
        return f"High confidence: {result['disease']} - Schedule appointment"
    else:
        return "Uncertain - Schedule doctor consultation"
```

### Use Case 2: Symptom Checker App
```python
def symptom_checker(symptoms):
    result = predict(symptoms)
    
    return {
        'primary_diagnosis': result['disease'],
        'confidence': result['confidence'],
        'alternatives': result['alternatives'][:3],
        'recommendations': result['precautions'],
        'urgency_level': result['urgent']
    }
```

### Use Case 3: Research Data Analysis
```python
import pandas as pd

# Analyze symptom patterns
df = pd.read_csv('patient_data.csv')
df['predicted_disease'] = df['symptoms'].apply(
    lambda x: predict(x)['disease']
)
df['prediction_confidence'] = df['symptoms'].apply(
    lambda x: predict(x)['confidence']
)

# Analyze results
print(df.groupby('predicted_disease').size())
print(f"Average confidence: {df['prediction_confidence'].mean():.2f}%")
```

---

## 🆘 Getting Help

**Check logs:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Rerun your code
```

**Verify installation:**
```bash
python -c "import sklearn; print(sklearn.__version__)"
python -c "import pandas; print(pandas.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

**Test with known symptoms:**
```bash
python predict_disease.py "fever"
python predict_disease.py "headache"
python predict_disease.py "chest pain"
```

---

**For more details, see:**
- `README.md` - Complete documentation
- `FINAL_PROJECT_SUMMARY.md` - Technical details
- `COMPLETE_PRODUCTION_SYSTEM.py` - Full source code

*Always consult medical professionals for health concerns.*
