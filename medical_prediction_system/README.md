# 🏥 Medical Disease Prediction System

A production-grade machine learning system that predicts diseases from symptom descriptions and provides medical recommendations.

## 📊 Performance Metrics

- **Accuracy**: 85.86%
- **F1-Score**: 84.67%
- **Recall**: 85.86%
- **Training Time**: 0.9 seconds
- **Diseases Supported**: 244 unique conditions

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Training the Model
```python
python COMPLETE_PRODUCTION_SYSTEM.py
```

### Making Predictions

**Option 1: Command Line**
```bash
python predict_disease.py "I have fever and headache for 2 days"
```

**Option 2: Python Script**
```python
from predict_disease import predict, display_result

result = predict("stomach pain after eating, nausea")
display_result(result)
```

**Option 3: Using Complete System**
```python
from COMPLETE_PRODUCTION_SYSTEM import MedicalPredictor

predictor = MedicalPredictor()
result = predictor.predict("chest pain and difficulty breathing")
print(predictor.format_prediction(result))
```

## 📁 Project Files

### Essential Files (Required for Inference)
- `final_model.pkl` - Trained Naive Bayes model (1.9 MB)
- `tfidf_vectorizer.pkl` - TF-IDF feature transformer (20 KB)
- `label_encoder.pkl` - Disease label encoder (4.9 KB)

### Code Files
- `COMPLETE_PRODUCTION_SYSTEM.py` - Full training and inference pipeline
- `predict_disease.py` - Standalone prediction script
- `inference_function.py` - Production inference module

### Data Files
- `cleaned_dataset.csv` - Cleaned training data (130,377 samples)
- `disease_list.txt` - List of 244 diseases

### Reports & Visualizations
- `confusion_matrix.png` - Model confusion matrix
- `model_performance.png` - Performance comparison charts
- `disease_distribution.png` - Dataset distribution
- `classification_report.txt` - Detailed metrics
- `model_comparison.csv` - Model comparison results

### Documentation
- `README.md` - This file
- `FINAL_PROJECT_SUMMARY.md` - Complete project documentation

## 🔧 Complete Training Pipeline

```python
# Import the training pipeline
from COMPLETE_PRODUCTION_SYSTEM import main_training_pipeline

# Train the model (provide path to your cleaned CSV data)
model, vectorizer, encoder = main_training_pipeline('cleaned_dataset.csv')

# This will:
# 1. Load and validate data
# 2. Clean and preprocess
# 3. Create TF-IDF features
# 4. Train multiple models
# 5. Select best model
# 6. Generate evaluation reports
# 7. Save all artifacts
```

## 📖 API Reference

### MedicalPredictor Class

```python
class MedicalPredictor:
    def __init__(self, model_path, vectorizer_path, encoder_path):
        """Load trained model"""
        
    def predict(self, symptoms_text):
        """
        Predict disease from symptoms
        
        Args:
            symptoms_text (str): Natural language symptom description
            
        Returns:
            dict: {
                'disease': str,
                'confidence': float (0-100),
                'alternative_diagnoses': list,
                'precautions': list,
                'consult_doctor': bool,
                'urgency': str ('URGENT'|'HIGH'|'MODERATE'),
                'ai_disclaimer': str
            }
        """
        
    def format_prediction(self, result):
        """Format prediction for display"""
```

## 🧪 Example Usage

### Example 1: Simple Prediction
```python
from predict_disease import predict, display_result

symptoms = "fever, headache, body ache for 2 days"
result = predict(symptoms)
display_result(result)
```

**Output:**
```
======================================================================
MEDICAL DISEASE PREDICTION
======================================================================

⚠️  AI PREDICTION - NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL ADVICE

🏥 DIAGNOSIS
   Disease: Viral Fever
   Confidence: 92.50%

💊 RECOMMENDED PRECAUTIONS:
   1. Rest and stay hydrated
   2. Take fever-reducing medication
   3. Monitor temperature
   4. Consult doctor if persists > 3 days
======================================================================
```

### Example 2: Urgent Case
```python
result = predict("severe chest pain and difficulty breathing")
display_result(result)
```

**Output includes:**
```
⚠️  ⚠️  ⚠️  URGENT - SEEK EMERGENCY MEDICAL CARE IMMEDIATELY
```

### Example 3: Batch Predictions
```python
from COMPLETE_PRODUCTION_SYSTEM import MedicalPredictor

predictor = MedicalPredictor()

test_cases = [
    "stomach pain after eating",
    "running nose and sneezing",
    "severe headache with vomiting"
]

for symptoms in test_cases:
    result = predictor.predict(symptoms)
    print(f"\nSymptoms: {symptoms}")
    print(f"Prediction: {result['disease']} ({result['confidence']:.2f}%)")
```

## 🎯 Model Architecture

### Pipeline Overview
1. **Input**: Natural language symptom description
2. **Preprocessing**: Text cleaning and normalization
3. **Feature Extraction**: TF-IDF vectorization (500 features, 1-2 grams)
4. **Model**: Multinomial Naive Bayes (optimized for text classification)
5. **Output**: Disease prediction + confidence + recommendations

### Why Naive Bayes?
- ✅ **Best Performance**: 85.86% accuracy, 84.67% F1-score
- ✅ **Fast Training**: 0.9 seconds (vs 13s for Random Forest)
- ✅ **Efficient Inference**: <100ms per prediction
- ✅ **Excellent for Text**: Ideal for TF-IDF features
- ✅ **High Recall**: 85.86% - critical for medical applications

## 📈 Performance Details

### Top 10 Diseases Performance

| Disease | Precision | Recall | F1-Score | Samples |
|---------|-----------|--------|----------|---------|
| Viral Fever | 0.9014 | 0.8876 | 0.8945 | 4,627 |
| Headache | 0.9359 | 0.9465 | 0.9412 | 3,998 |
| Abdominal Pain | 0.9671 | 0.8598 | 0.9103 | 2,290 |
| Edema | 0.9776 | 0.7638 | 0.8576 | 1,770 |
| Allergic Reaction | 0.8731 | 0.8327 | 0.8525 | 843 |
| Respiratory Distress | 0.9617 | 0.7683 | 0.8542 | 751 |
| Fatigue | 0.9755 | 0.6910 | 0.8089 | 576 |
| Syncope | 1.0000 | 0.9146 | 0.9554 | 515 |
| Skin Allergy | 0.7877 | 0.9538 | 0.8629 | 498 |
| General Weakness | 0.9251 | 0.8587 | 0.8906 | 460 |

## ⚠️ Important Disclaimers

### Medical Safety
- ⚠️ **NOT A REPLACEMENT** for professional medical diagnosis
- ⚠️ **AI PREDICTIONS** are probabilistic, not definitive
- ⚠️ **ALWAYS CONSULT** qualified healthcare professionals
- ⚠️ **EMERGENCY SYMPTOMS** require immediate medical attention

### System Limitations
- Trained on specific dataset (may not generalize to all cases)
- Some rare diseases have lower accuracy
- Cannot replace clinical examination and tests
- Should be used as preliminary assessment tool only

## 🔐 Safety Features

### Built-in Safety Mechanisms
1. ✅ **Confidence Thresholds**: Low confidence triggers doctor consultation
2. ✅ **Urgent Keyword Detection**: Flags critical symptoms
3. ✅ **AI Disclaimers**: Every prediction includes warnings
4. ✅ **Alternative Diagnoses**: Provides top 3 possibilities
5. ✅ **Medical Precautions**: Disease-specific recommendations
6. ✅ **Never Returns NULL**: Always provides output

## 📚 Dataset

### Original Dataset
- **Source**: SYNAPSE Medical Dataset
- **Size**: 130,637 rows
- **Features**: Symptoms, Gender, Age, Duration, Severity
- **Diseases**: 244 unique conditions

### Data Processing
- **Cleaned**: 99.80% data retention
- **Train/Test Split**: 80% / 20% stratified
- **Feature Engineering**: TF-IDF with 500 features

## 🛠️ Technical Stack

- **Language**: Python 3.12
- **ML Framework**: scikit-learn 1.3+
- **Feature Engineering**: TfidfVectorizer
- **Model**: MultinomialNB (Naive Bayes)
- **Visualization**: matplotlib, seaborn
- **Data Processing**: pandas, numpy

## 📦 Installation

### Full Installation
```bash
# Clone or download the project
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Minimal Installation (Inference Only)
```bash
# Only need these for predictions
pip install pandas numpy scikit-learn
```

## 🔄 Retraining the Model

### With New Data
```python
from COMPLETE_PRODUCTION_SYSTEM import main_training_pipeline

# Your new dataset should have columns:
# - Symptoms (text)
# - Final Recommendation (disease label)
# - Gender, Age, Duration, Severity (optional)

model, vectorizer, encoder = main_training_pipeline('your_data.csv')
```

### Hyperparameter Tuning
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0]}

# Grid search
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
```

## 🚀 Deployment

### Local API (Flask Example)
```python
from flask import Flask, request, jsonify
from COMPLETE_PRODUCTION_SYSTEM import MedicalPredictor

app = Flask(__name__)
predictor = MedicalPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predictor.predict(data['symptoms'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Cloud Deployment
- Compatible with: AWS Lambda, Google Cloud Functions, Azure Functions
- Model size: ~2 MB (suitable for serverless)
- Cold start: <500ms

## 📞 Support & Contact

For issues, questions, or contributions:
- Create an issue in the repository
- Contact: ML Engineering Team

## 📄 License

This project is for educational and research purposes only.
Always consult medical professionals for actual health concerns.

## 🔮 Future Improvements

1. **Deep Learning**: BERT/BioBERT for better symptom understanding
2. **Multi-language**: Support for multiple languages
3. **More Data**: Collect more samples for rare diseases
4. **Real-time Learning**: Continuous model updates
5. **Integration**: Mobile app, web interface
6. **Doctor Feedback**: Incorporate expert validation

---

**Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Production Ready ✅

*Always consult qualified healthcare professionals for medical advice.*
