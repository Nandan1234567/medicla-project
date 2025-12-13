import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

print("="*80)
print("STEP 3: PREPROCESSING")
print("="*80)

# Load cleaned dataset
df = pd.read_csv('/home/user/cleaned_dataset.csv')
print(f"\n✓ Loaded cleaned dataset: {len(df):,} rows")

# Prepare features and labels
X = df['Symptoms'].values
y = df['Final Recommendation'].values

print(f"\nDataset shape:")
print(f"  - Features (X): {X.shape}")
print(f"  - Labels (y): {y.shape}")
print(f"  - Unique diseases: {len(np.unique(y))}")

# Encode labels
print("\n1. ENCODING DISEASE LABELS...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"   ✓ Labels encoded: {len(label_encoder.classes_)} classes")

# Save label encoder
with open('/home/user/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"   ✓ Label encoder saved")

# Create TF-IDF features from symptoms
print("\n2. CREATING TF-IDF FEATURES FROM SYMPTOMS...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    lowercase=True,
    strip_accents='unicode'
)
X_tfidf = tfidf_vectorizer.fit_transform(X)
print(f"   ✓ TF-IDF features created: {X_tfidf.shape}")
print(f"   ✓ Feature dimension: {X_tfidf.shape[1]}")

# Save TF-IDF vectorizer
with open('/home/user/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"   ✓ TF-IDF vectorizer saved")

# Train-test split
print("\n3. SPLITTING DATA (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)
print(f"   ✓ Training set: {X_train.shape[0]:,} samples")
print(f"   ✓ Test set: {X_test.shape[0]:,} samples")

# Check class balance
print("\n4. CLASS DISTRIBUTION CHECK...")
unique, counts = np.unique(y_train, return_counts=True)
print(f"   ✓ Training classes: {len(unique)}")
print(f"   ✓ Samples per class (avg): {np.mean(counts):.0f}")
print(f"   ✓ Min samples: {np.min(counts)}, Max samples: {np.max(counts)}")

print("\n" + "="*80)
print("STEP 4: MODEL TRAINING & COMPARISON")
print("="*80)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='linear', random_state=42, probability=True),
    'Naive Bayes': MultinomialNB()
}

results = {}

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training: {name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score (on a subset for speed)
    print(f"   Computing cross-validation score...")
    cv_scores = cross_val_score(model, X_train[:20000], y_train[:20000], cv=3, n_jobs=-1)
    cv_mean = cv_scores.mean()
    
    training_time = time.time() - start_time
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'cv_score': cv_mean,
        'training_time': training_time,
        'predictions': y_pred
    }
    
    print(f"\n   Results:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - CV Score: {cv_mean:.4f}")
    print(f"   - Training Time: {training_time:.2f}s")

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'CV Score': [results[m]['cv_score'] for m in results.keys()],
    'Time (s)': [results[m]['training_time'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
print("\n" + comparison_df.to_string(index=False))

# Select best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*80}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
print(f"Recall: {results[best_model_name]['recall']:.4f}")

# Save results
comparison_df.to_csv('/home/user/model_comparison.csv', index=False)
print(f"\n✓ Model comparison saved to: model_comparison.csv")

# Save preprocessed data and results
np.save('/home/user/X_test.npy', X_test.toarray())
np.save('/home/user/y_test.npy', y_test)
np.save('/home/user/y_pred_best.npy', best_predictions)

# Save training data for later use
print(f"\n✓ Preprocessed data saved")
print(f"✓ Best model: {best_model_name}")

# Store best model info
with open('/home/user/best_model_info.txt', 'w') as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
    f.write(f"F1-Score: {results[best_model_name]['f1_score']:.4f}\n")
    f.write(f"Recall: {results[best_model_name]['recall']:.4f}\n")

# Continue with hyperparameter tuning
print(f"\n{'='*80}")
print("STEP 5: HYPERPARAMETER TUNING")
print(f"{'='*80}")

if best_model_name == 'Random Forest':
    print(f"\nTuning Random Forest...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print("   Running GridSearchCV... (this may take a few minutes)")
    grid_search.fit(X_train[:30000], y_train[:30000])
    
    best_tuned_model = grid_search.best_estimator_
    print(f"\n   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")
    
    # Retrain on full training data
    print(f"\n   Retraining on full training data...")
    best_tuned_model.fit(X_train, y_train)
    
elif best_model_name == 'Gradient Boosting':
    print(f"\nTuning Gradient Boosting...")
    param_grid = {
        'n_estimators': [100, 150],
        'learning_rate': [0.1, 0.05],
        'max_depth': [3, 5]
    }
    
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print("   Running GridSearchCV... (this may take a few minutes)")
    grid_search.fit(X_train[:30000], y_train[:30000])
    
    best_tuned_model = grid_search.best_estimator_
    print(f"\n   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.4f}")
    
    # Retrain on full training data
    print(f"\n   Retraining on full training data...")
    best_tuned_model.fit(X_train, y_train)
    
else:
    print(f"\n   Using best model without further tuning")
    best_tuned_model = best_model
    best_tuned_model.fit(X_train, y_train)

# Final predictions
y_pred_final = best_tuned_model.predict(X_test)

# Final metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final, average='weighted')
final_recall = recall_score(y_test, y_pred_final, average='weighted')

print(f"\n{'='*80}")
print("FINAL MODEL PERFORMANCE")
print(f"{'='*80}")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"F1-Score: {final_f1:.4f}")
print(f"Recall: {final_recall:.4f}")

# Save final model
with open('/home/user/final_model.pkl', 'wb') as f:
    pickle.dump(best_tuned_model, f)
print(f"\n✓ Final model saved to: final_model.pkl")

# Save final predictions
np.save('/home/user/y_pred_final.npy', y_pred_final)

print(f"\n{'='*80}")
print("TRAINING COMPLETE!")
print(f"{'='*80}")
