"""
=================================================================================
MEDICAL DISEASE PREDICTION SYSTEM - COMPLETE PRODUCTION CODE
=================================================================================

This is a complete, production-grade medical disease prediction system.
It includes all steps from data loading to model training and inference.

Author: ML Engineering Team
Date: December 2024
Version: 1.0

=================================================================================
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("="*80)
print("MEDICAL DISEASE PREDICTION SYSTEM")
print("Complete Production-Grade ML Pipeline")
print("="*80)

# =============================================================================
# STEP 1: DATA LOADING & VALIDATION
# =============================================================================

def load_and_validate_data(file_path):
    """Load and validate the medical dataset"""
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING & VALIDATION")
    print("="*80)
    
    df = pd.read_csv(file_path)
    print(f"\n✓ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Check for issues
    duplicates = df.duplicated().sum()
    missing = df.isnull().sum().sum()
    
    print(f"✓ Duplicate rows: {duplicates}")
    print(f"✓ Missing values: {missing}")
    print(f"✓ Unique diseases: {df['Final Recommendation'].nunique()}")
    
    return df

# =============================================================================
# STEP 2: DATA CLEANING
# =============================================================================

def clean_data(df):
    """Clean and preprocess the dataset"""
    print("\n" + "="*80)
    print("STEP 2: DATA CLEANING")
    print("="*80)
    
    original_count = len(df)
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"✓ Removed {original_count - len(df)} duplicates")
    
    # Remove classes with too few samples (need >= 2 for stratified split)
    disease_counts = df['Final Recommendation'].value_counts()
    valid_diseases = disease_counts[disease_counts >= 2].index
    df = df[df['Final Recommendation'].isin(valid_diseases)]
    print(f"✓ Kept {len(valid_diseases)} diseases with >= 2 samples")
    
    # Standardize text
    df['Symptoms'] = df['Symptoms'].astype(str).str.strip()
    df['Final Recommendation'] = df['Final Recommendation'].astype(str).str.strip()
    
    # Remove invalid symptoms
    df = df[df['Symptoms'].str.len() > 5]
    
    df = df.reset_index(drop=True)
    
    print(f"✓ Final dataset: {len(df):,} rows ({len(df)/original_count*100:.2f}% retained)")
    
    return df

# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================

def create_features(X_train_text, X_test_text):
    """Create TF-IDF features from symptom text"""
    print("\n" + "="*80)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*80)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        lowercase=True
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
    
    print(f"✓ TF-IDF features created: {X_train_tfidf.shape[1]} features")
    print(f"✓ Training samples: {X_train_tfidf.shape[0]:,}")
    print(f"✓ Test samples: {X_test_tfidf.shape[0]:,}")
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

# =============================================================================
# STEP 4: MODEL TRAINING
# =============================================================================

def train_models(X_train, y_train, X_test, y_test):
    """Train and compare multiple models"""
    print("\n" + "="*80)
    print("STEP 4: MODEL TRAINING & COMPARISON")
    print("="*80)
    
    models = {
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=30,
            min_samples_split=5,
            random_state=42,
            n_jobs=2,
            class_weight='balanced'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name}...")
        start = time.time()
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        t = time.time() - start
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'f1': f1,
            'recall': rec,
            'time': t,
            'predictions': y_pred
        }
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  Time: {t:.1f}s")
    
    # Select best model
    best_name = max(results, key=lambda x: results[x]['f1'])
    
    print(f"\n{'='*80}")
    print(f"✓ BEST MODEL: {best_name}")
    print(f"{'='*80}")
    print(f"Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"F1-Score: {results[best_name]['f1']:.4f}")
    print(f"Recall: {results[best_name]['recall']:.4f}")
    
    return results[best_name]['model'], results

# =============================================================================
# STEP 5: MODEL EVALUATION
# =============================================================================

def evaluate_model(y_test, y_pred, label_encoder, save_plots=True):
    """Generate comprehensive evaluation metrics"""
    print("\n" + "="*80)
    print("STEP 5: MODEL EVALUATION")
    print("="*80)
    
    # Classification report for top diseases
    print("\n📊 Classification Report (Top 10 Diseases):")
    print("-" * 80)
    
    # Get top 10 diseases
    disease_counts = pd.Series(label_encoder.inverse_transform(y_test)).value_counts()
    top_diseases = disease_counts.head(10).index.tolist()
    
    # Filter for top diseases
    top_indices = [label_encoder.transform([d])[0] for d in top_diseases]
    mask = np.isin(y_test, top_indices)
    
    report = classification_report(
        y_test[mask],
        y_pred[mask],
        labels=top_indices,
        target_names=top_diseases,
        digits=4
    )
    print(report)
    
    if save_plots:
        # Confusion Matrix
        cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_indices)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[d[:15] for d in top_diseases],
            yticklabels=[d[:15] for d in top_diseases]
        )
        plt.title('Confusion Matrix - Top 10 Diseases', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontweight='bold')
        plt.ylabel('Actual', fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("\n✓ Confusion matrix saved: confusion_matrix.png")
        plt.close()

# =============================================================================
# STEP 6: SAVE MODEL ARTIFACTS
# =============================================================================

def save_model_artifacts(model, tfidf_vectorizer, label_encoder):
    """Save trained model and preprocessing objects"""
    print("\n" + "="*80)
    print("STEP 6: SAVING MODEL ARTIFACTS")
    print("="*80)
    
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ final_model.pkl")
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print("✓ tfidf_vectorizer.pkl")
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✓ label_encoder.pkl")

# =============================================================================
# STEP 7: INFERENCE FUNCTION
# =============================================================================

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
    'default': [
        'Monitor your symptoms closely',
        'Rest and maintain good hydration',
        'Avoid self-medication without consultation',
        'Keep a symptom diary',
        'Consult a healthcare provider if symptoms worsen or persist'
    ]
}

URGENT_KEYWORDS = [
    'chest pain', 'shortness breath', 'difficulty breathing',
    'loss of consciousness', 'severe bleeding', 'confusion',
    'severe pain', 'heart', 'stroke', 'seizure'
]

class MedicalPredictor:
    """Production-ready medical disease predictor"""
    
    def __init__(self, model_path='final_model.pkl', 
                 vectorizer_path='tfidf_vectorizer.pkl',
                 encoder_path='label_encoder.pkl'):
        """Load trained model and preprocessors"""
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(encoder_path, 'rb') as f:
            self.encoder = pickle.load(f)
    
    def predict(self, symptoms_text):
        """
        Predict disease from symptoms
        
        Args:
            symptoms_text (str): Description of symptoms
            
        Returns:
            dict: Prediction results with confidence and recommendations
        """
        
        # Preprocess
        symptoms_clean = symptoms_text.strip().lower()
        
        if len(symptoms_clean) < 3:
            return {
                'disease': 'Insufficient Information',
                'confidence': 0.0,
                'precautions': ['Please provide more detailed symptoms'],
                'consult_doctor': True,
                'urgency': 'Unknown'
            }
        
        # Transform and predict
        symptoms_tfidf = self.vectorizer.transform([symptoms_clean])
        prediction = self.model.predict(symptoms_tfidf)[0]
        disease_name = self.encoder.inverse_transform([prediction])[0]
        
        # Get confidence
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(symptoms_tfidf)[0]
            confidence = float(probabilities[prediction])
            
            # Top 3 predictions
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            alternatives = [
                {
                    'disease': self.encoder.inverse_transform([idx])[0],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_3_idx
            ]
        else:
            confidence = 0.85
            alternatives = []
        
        # Assess urgency
        is_urgent = any(keyword in symptoms_clean for keyword in URGENT_KEYWORDS)
        needs_doctor = (
            confidence < 0.6 or
            is_urgent or
            'heart' in disease_name.lower() or
            'cancer' in disease_name.lower()
        )
        
        # Get precautions
        precautions = MEDICAL_PRECAUTIONS.get(disease_name, MEDICAL_PRECAUTIONS['default'])
        
        return {
            'disease': disease_name,
            'confidence': round(confidence * 100, 2),
            'alternative_diagnoses': alternatives,
            'precautions': precautions,
            'consult_doctor': needs_doctor,
            'urgency': 'URGENT' if is_urgent else ('HIGH' if needs_doctor else 'MODERATE'),
            'ai_disclaimer': '⚠️ This is an AI prediction. Always consult a qualified healthcare professional.'
        }
    
    def format_prediction(self, result):
        """Format prediction for display"""
        
        output = []
        output.append("="*80)
        output.append("MEDICAL DISEASE PREDICTION RESULT")
        output.append("="*80)
        output.append("")
        output.append("⚠️  AI DISCLAIMER")
        output.append("-" * 80)
        output.append(result['ai_disclaimer'])
        output.append("")
        output.append("🏥 DIAGNOSIS")
        output.append("-" * 80)
        output.append(f"Disease: {result['disease']}")
        output.append(f"Confidence: {result['confidence']}%")
        output.append(f"Urgency: {result['urgency']}")
        output.append("")
        
        if result.get('alternative_diagnoses'):
            output.append("🔍 ALTERNATIVES")
            output.append("-" * 80)
            for i, alt in enumerate(result['alternative_diagnoses'][:3], 1):
                output.append(f"{i}. {alt['disease']} ({alt['confidence']*100:.2f}%)")
            output.append("")
        
        output.append("💊 PRECAUTIONS")
        output.append("-" * 80)
        for i, prec in enumerate(result['precautions'], 1):
            output.append(f"{i}. {prec}")
        output.append("")
        
        if result['consult_doctor']:
            output.append("⚠️  IMPORTANT")
            output.append("-" * 80)
            output.append("⚠️  CONSULT A DOCTOR")
            if result['urgency'] == 'URGENT':
                output.append("⚠️  SEEK EMERGENCY CARE NOW")
            output.append("")
        
        output.append("="*80)
        return "\n".join(output)

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main_training_pipeline(data_path):
    """Complete training pipeline"""
    
    print("\n🚀 Starting Complete Training Pipeline\n")
    
    # Step 1: Load data
    df = load_and_validate_data(data_path)
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Prepare features and labels
    X = df['Symptoms'].values
    y = df['Final Recommendation'].values
    
    # Encode labels
    print("\n📝 Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"✓ {len(label_encoder.classes_)} disease classes")
    
    # Train-test split
    print("\n✂️ Splitting data (80-20)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"✓ Train: {len(X_train_text):,}, Test: {len(X_test_text):,}")
    
    # Step 3: Feature engineering
    X_train, X_test, tfidf_vectorizer = create_features(X_train_text, X_test_text)
    
    # Step 4: Train models
    best_model, all_results = train_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Evaluate
    y_pred = best_model.predict(X_test)
    evaluate_model(y_test, y_pred, label_encoder)
    
    # Step 6: Save artifacts
    save_model_artifacts(best_model, tfidf_vectorizer, label_encoder)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nModel files saved:")
    print("  - final_model.pkl")
    print("  - tfidf_vectorizer.pkl")
    print("  - label_encoder.pkl")
    print("  - confusion_matrix.png")
    
    return best_model, tfidf_vectorizer, label_encoder

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    # TRAINING MODE
    print("\n" + "="*80)
    print("TRAINING MODE")
    print("="*80)
    
    # Train the model (uncomment to train)
    # model, vectorizer, encoder = main_training_pipeline('cleaned_dataset.csv')
    
    # INFERENCE MODE
    print("\n" + "="*80)
    print("INFERENCE MODE - TESTING PREDICTIONS")
    print("="*80)
    
    # Load trained model
    try:
        predictor = MedicalPredictor()
        print("\n✓ Model loaded successfully!")
        
        # Test cases
        test_cases = [
            "I have fever and headache for 2 days",
            "chest pain and difficulty breathing",
            "stomach pain after eating, nausea",
            "running nose, sneezing, sore throat",
            "severe back pain, can't move"
        ]
        
        print("\n" + "="*80)
        print("TESTING PREDICTIONS")
        print("="*80)
        
        for i, symptoms in enumerate(test_cases, 1):
            print(f"\n{'#'*80}")
            print(f"TEST CASE {i}: \"{symptoms}\"")
            print(f"{'#'*80}")
            
            result = predictor.predict(symptoms)
            print(predictor.format_prediction(result))
    
    except FileNotFoundError:
        print("\n⚠️ Model files not found. Please train the model first.")
        print("Set the data path and uncomment the training line above.")

print("\n" + "="*80)
print("PROGRAM END")
print("="*80)
