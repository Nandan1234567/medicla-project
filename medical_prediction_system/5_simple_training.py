import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pickle
import time

print("="*80)
print("MEDICAL DISEASE PREDICTION - MODEL TRAINING")
print("="*80)

# Load cleaned dataset
df = pd.read_csv('/home/user/cleaned_dataset.csv')
print(f"\n✓ Loaded: {len(df):,} rows")

# Remove rare classes (keep diseases with at least 2 samples)
disease_counts = df['Final Recommendation'].value_counts()
valid_diseases = disease_counts[disease_counts >= 2].index
df = df[df['Final Recommendation'].isin(valid_diseases)]
print(f"✓ Total unique diseases: {len(valid_diseases)}")

# Prepare data
X = df['Symptoms'].values
y = df['Final Recommendation'].values

# Encode labels
print(f"\n1. Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"   ✓ {len(label_encoder.classes_)} disease classes encoded")

# TF-IDF vectorization
print(f"\n2. Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=500, 
    ngram_range=(1, 2), 
    min_df=2,
    max_df=0.9
)
X_tfidf = tfidf_vectorizer.fit_transform(X)
print(f"   ✓ Feature shape: {X_tfidf.shape}")

# Split data
print(f"\n3. Splitting train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   ✓ Training: {X_train.shape[0]:,} samples")
print(f"   ✓ Testing: {X_test.shape[0]:,} samples")

# Train models
print(f"\n{'='*80}")
print("MODEL TRAINING")
print("="*80)

results = {}

# 1. Random Forest
print(f"\n1. Training Random Forest...")
start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=30,
    min_samples_split=5,
    random_state=42,
    n_jobs=2,  # Limit parallelism
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_time = time.time() - start

rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
rf_rec = recall_score(y_test, rf_pred, average='weighted')

results['Random Forest'] = {
    'model': rf_model,
    'accuracy': rf_acc,
    'f1': rf_f1,
    'recall': rf_rec,
    'time': rf_time,
    'pred': rf_pred
}
print(f"   Accuracy: {rf_acc:.4f}")
print(f"   F1-Score: {rf_f1:.4f}")
print(f"   Recall: {rf_rec:.4f}")
print(f"   Time: {rf_time:.1f}s")

# 2. Naive Bayes
print(f"\n2. Training Naive Bayes...")
start = time.time()
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_time = time.time() - start

nb_acc = accuracy_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred, average='weighted')
nb_rec = recall_score(y_test, nb_pred, average='weighted')

results['Naive Bayes'] = {
    'model': nb_model,
    'accuracy': nb_acc,
    'f1': nb_f1,
    'recall': nb_rec,
    'time': nb_time,
    'pred': nb_pred
}
print(f"   Accuracy: {nb_acc:.4f}")
print(f"   F1-Score: {nb_f1:.4f}")
print(f"   Recall: {nb_rec:.4f}")
print(f"   Time: {nb_time:.1f}s")

# Select best model
print(f"\n{'='*80}")
print("MODEL COMPARISON")
print("="*80)

comparison = []
for name, res in results.items():
    comparison.append({
        'Model': name,
        'Accuracy': res['accuracy'],
        'F1-Score': res['f1'],
        'Recall': res['recall'],
        'Time (s)': res['time']
    })

comparison_df = pd.DataFrame(comparison).sort_values('F1-Score', ascending=False)
print("\n" + comparison_df.to_string(index=False))

best_name = comparison_df.iloc[0]['Model']
best_model = results[best_name]['model']
best_pred = results[best_name]['pred']

print(f"\n{'='*80}")
print(f"✓ BEST MODEL: {best_name}")
print(f"{'='*80}")
print(f"Accuracy: {results[best_name]['accuracy']:.4f}")
print(f"F1-Score: {results[best_name]['f1']:.4f}")
print(f"Recall: {results[best_name]['recall']:.4f}")

# Save all artifacts
print(f"\n{'='*80}")
print("SAVING MODEL ARTIFACTS")
print("="*80)

with open('/home/user/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✓ label_encoder.pkl")

with open('/home/user/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("✓ tfidf_vectorizer.pkl")

with open('/home/user/final_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ final_model.pkl")

np.save('/home/user/y_test.npy', y_test)
np.save('/home/user/y_pred_final.npy', best_pred)
print("✓ Test data saved")

comparison_df.to_csv('/home/user/model_comparison.csv', index=False)
print("✓ model_comparison.csv")

# Save performance summary
with open('/home/user/final_performance.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FINAL MODEL PERFORMANCE SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Best Model: {best_name}\n\n")
    f.write(f"Performance Metrics:\n")
    f.write(f"  - Accuracy: {results[best_name]['accuracy']:.4f}\n")
    f.write(f"  - F1-Score: {results[best_name]['f1']:.4f}\n")
    f.write(f"  - Recall: {results[best_name]['recall']:.4f}\n\n")
    f.write(f"Dataset Information:\n")
    f.write(f"  - Total Diseases: {len(label_encoder.classes_)}\n")
    f.write(f"  - Training Samples: {X_train.shape[0]:,}\n")
    f.write(f"  - Test Samples: {X_test.shape[0]:,}\n")
    f.write(f"  - Feature Dimension: {X_tfidf.shape[1]}\n\n")
    f.write("="*80 + "\n")

print("✓ final_performance.txt")

print(f"\n{'='*80}")
print("✓ TRAINING COMPLETE - ALL FILES SAVED")
print("="*80)
