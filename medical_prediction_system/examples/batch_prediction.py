"""
=================================================================================
BATCH PREDICTION EXAMPLE - MEDICAL PREDICTION SYSTEM
=============================================================================

Example script showing how to process multiple symptoms at once
Useful for processing patient lists or research data

Author: ML Engineering Team
Date: December 2024
Version: 3.0 - Production Example
=================================================================================
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.linear_svm_predictor import LinearSVMMedicalPredictor

def batch_predict_from_list(symptoms_list):
    """Process a list of symptoms and return results"""

    print("🏥 BATCH MEDICAL PREDICTION")
    print("🎯 97.23% Accuracy | Linear SVM")
    print("=" * 60)

    # Initialize predictor
    try:
        predictor = LinearSVMMedicalPredictor()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    results = []

    print(f"\n📊 Processing {len(symptoms_list)} cases...")
    print("-" * 60)

    for i, symptoms in enumerate(symptoms_list, 1):
        try:
            # Make prediction
            result = predictor.predict(symptoms)

            # Store result
            results.append({
                'case_id': i,
                'symptoms': symptoms,
                'predicted_disease': result['disease'],
                'confidence': result['confidence'],
                'urgency': result['urgency'],
                'consult_doctor': result['consult_doctor'],
                'model': result['model'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # Display progress
            urgency_icon = "🚨" if result['urgency'] == 'EMERGENCY' else "⚠️" if result['urgency'] in ['URGENT', 'HIGH'] else "✅"
            print(f"{i:2d}. {symptoms[:40]:40} → {result['disease'][:20]:20} ({result['confidence']:5.1f}%) {urgency_icon}")

        except Exception as e:
            print(f"{i:2d}. ❌ Error processing: {symptoms[:40]} - {e}")
            results.append({
                'case_id': i,
                'symptoms': symptoms,
                'predicted_disease': 'ERROR',
                'confidence': 0.0,
                'urgency': 'UNKNOWN',
                'consult_doctor': True,
                'model': 'ERROR',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    return results

def batch_predict_from_csv(csv_file, symptoms_column='symptoms'):
    """Process symptoms from a CSV file"""

    print(f"📊 Loading symptoms from: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} records")

        if symptoms_column not in df.columns:
            print(f"❌ Column '{symptoms_column}' not found in CSV")
            print(f"Available columns: {list(df.columns)}")
            return None

        symptoms_list = df[symptoms_column].tolist()
        return batch_predict_from_list(symptoms_list)

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def save_results(results, output_file='batch_predictions.csv'):
    """Save results to CSV file"""

    if not results:
        print("❌ No results to save")
        return False

    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")

        # Display summary
        print(f"\n📊 BATCH PREDICTION SUMMARY")
        print("=" * 40)
        print(f"Total cases processed: {len(results)}")
        print(f"Average confidence: {df['confidence'].mean():.2f}%")

        # Urgency distribution
        urgency_counts = df['urgency'].value_counts()
        print(f"\nUrgency distribution:")
        for urgency, count in urgency_counts.items():
            print(f"  {urgency}: {count}")

        # Top diseases
        disease_counts = df['predicted_disease'].value_counts()
        print(f"\nTop 5 predicted diseases:")
        for disease, count in disease_counts.head().items():
            print(f"  {disease}: {count}")

        return True

    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

def main():
    """Main function for batch prediction"""

    # Example symptoms list
    example_symptoms = [
        "fever and headache for 2 days",
        "chest pain and difficulty breathing",
        "stomach pain after eating spicy food",
        "persistent cough with phlegm",
        "severe back pain when bending",
        "dizziness and nausea",
        "sore throat and runny nose",
        "joint pain and stiffness",
        "shortness of breath during exercise",
        "frequent urination and thirst"
    ]

    print("🏥 BATCH PREDICTION EXAMPLE")
    print("=" * 60)
    print("\nOptions:")
    print("1. Process example symptoms")
    print("2. Process symptoms from CSV file")
    print("3. Process custom symptoms list")

    choice = input("\nEnter your choice (1-3): ").strip()

    results = None

    if choice == '1':
        # Process example symptoms
        results = batch_predict_from_list(example_symptoms)

    elif choice == '2':
        # Process from CSV
        csv_file = input("Enter CSV file path: ").strip()
        symptoms_column = input("Enter symptoms column name (default: 'symptoms'): ").strip() or 'symptoms'
        results = batch_predict_from_csv(csv_file, symptoms_column)

    elif choice == '3':
        # Process custom list
        print("\nEnter symptoms (one per line, empty line to finish):")
        custom_symptoms = []
        while True:
            symptom = input(f"{len(custom_symptoms)+1}. ").strip()
            if not symptom:
                break
            custom_symptoms.append(symptom)

        if custom_symptoms:
            results = batch_predict_from_list(custom_symptoms)
        else:
            print("❌ No symptoms entered")
            return

    else:
        print("❌ Invalid choice")
        return

    # Save results if successful
    if results:
        output_file = input(f"\nSave results to file (default: batch_predictions.csv): ").strip() or 'batch_predictions.csv'
        save_results(results, output_file)

        # Ask if user wants to see detailed results
        show_details = input("\nShow detailed results? (y/n): ").strip().lower() == 'y'
        if show_details:
            print(f"\n📋 DETAILED RESULTS")
            print("=" * 80)
            for result in results:
                print(f"\nCase {result['case_id']}:")
                print(f"  Symptoms: {result['symptoms']}")
                print(f"  Disease: {result['predicted_disease']}")
                print(f"  Confidence: {result['confidence']:.2f}%")
                print(f"  Urgency: {result['urgency']}")
                print(f"  Consult Doctor: {result['consult_doctor']}")

    print(f"\n✅ Batch prediction complete!")

if __name__ == "__main__":
    main()
