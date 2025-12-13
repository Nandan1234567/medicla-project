import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score
import pickle
import time

print("="*80)
print("OPTIMIZED MODEL TRAINING")
print("="*80)

# Load cleaned dataset
df = pd.read_csv('/home/user/cleaned_dataset.csv')
print(f"\n✓ Loaded: {len(df):,} rows")

# Remove rare classes
disease_counts = df['Final Recommendation'].value_counts()
valid_diseases = disease_counts[disease_counts >= 2].index
df = df[df['Final Recommendation'].isin(valid_diseases)]
print(f"✓ Diseases: {len(valid_diseases)}")

# Prepare data
X = df['Symptoms'].values
y = df['Final Recommendation'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2)
X_tfidf = tfidf_vectorizer.fit_transform(X)
print(f"✓ Features: {X_tfidf.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"✓ Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

# Train models
print(f"\n{'='*80}")
print("TRAINING MODELS")
print("="*80)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1, class_weight='balanced'),
    'Naive Bayes': MultinomialNB()
}

results = {}
for name, model in models.items():
    print(f"\n{name}...")
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    t = time.time() - start
    
    results[name] = {'model': model, 'accuracy': acc, 'f1': f1, 'recall': rec, 'time': t, 'pred': y_pred}
    print(f"  Acc: {acc:.4f}, F1: {f1:.4f}, Recall: {rec:.4f}, Time: {t:.1f}s")

# Best model
best_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_name]['model']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_name}")
print(f"{'='*80}")
print(f"Accuracy: {results[best_name]['accuracy']:.4f}")
print(f"F1-Score: {results[best_name]['f1']:.4f}")
print(f"Recall: {results[best_name]['recall']:.4f}")

# Save everything
with open('/home/user/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('/home/user/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('/home/user/final_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

np.save('/home/user/y_test.npy', y_test)
np.save('/home/user/y_pred_final.npy', results[best_name]['pred'])

# Save summary
summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'Time (s)': [results[m]['time'] for m in results.keys()]
}).sort_values('F1-Score', ascending=False)

print(f"\n{'='*80}")
print("ALL MODELS COMPARISON")
print("="*80)
print(summary.to_string(index=False))

summary.to_csv('/home/user/model_comparison.csv', index=False)

with open('/home/user/final_performance.txt', 'w') as f:
    f.write(f"BEST MODEL: {best_name}\n")
    f.write(f"Accuracy: {results[best_name]['accuracy']:.4f}\n")
    f.write(f"F1-Score: {results[best_name]['f1']:.4f}\n")
    f.write(f"Recall: {results[best_name]['recall']:.4f}\n")
    f.write(f"\nDiseases: {len(label_encoder.classes_)}\n")
    f.write(f"Train: {X_train.shape[0]:,}\n")
    f.write(f"Test: {X_test.shape[0]:,}\n")

print(f"\n✓ All files saved!")
print("="*80)
