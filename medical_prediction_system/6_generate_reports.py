import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING EVALUATION REPORTS")
print("="*80)

# Load saved data
y_test = np.load('/home/user/y_test.npy')
y_pred = np.load('/home/user/y_pred_final.npy')

with open('/home/user/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print(f"\n✓ Loaded test data: {len(y_test)} samples")
print(f"✓ Loaded predictions: {len(y_pred)} predictions")
print(f"✓ Diseases: {len(label_encoder.classes_)}")

# Classification report
print(f"\n{'='*80}")
print("CLASSIFICATION REPORT (Top 15 Diseases)")
print("="*80)

# Get top diseases
df = pd.read_csv('/home/user/cleaned_dataset.csv')
top_diseases = df['Final Recommendation'].value_counts().head(15).index.tolist()
top_disease_indices = [i for i, name in enumerate(label_encoder.classes_) if name in top_diseases]

# Filter data for top diseases
mask = np.isin(y_test, top_disease_indices)
y_test_top = y_test[mask]
y_pred_top = y_pred[mask]

# Generate report for top diseases
report = classification_report(
    y_test_top, 
    y_pred_top,
    labels=top_disease_indices,
    target_names=[label_encoder.classes_[i] for i in top_disease_indices],
    digits=4
)
print(report)

# Save full report
with open('/home/user/classification_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLASSIFICATION REPORT - TOP 15 DISEASES\n")
    f.write("="*80 + "\n\n")
    f.write(report)
    f.write("\n" + "="*80 + "\n")
    f.write("NOTE: Report shows top 15 most frequent diseases\n")
    f.write("="*80 + "\n")

print(f"\n✓ Full classification report saved: classification_report.txt")

# Create confusion matrix for top 10 diseases
print(f"\n{'='*80}")
print("CREATING CONFUSION MATRIX")
print("="*80)

top_10_diseases = df['Final Recommendation'].value_counts().head(10).index.tolist()
top_10_indices = [i for i, name in enumerate(label_encoder.classes_) if name in top_10_diseases]

# Filter for top 10
mask_10 = np.isin(y_test, top_10_indices)
y_test_10 = y_test[mask_10]
y_pred_10 = y_pred[mask_10]

# Generate confusion matrix
cm = confusion_matrix(y_test_10, y_pred_10, labels=top_10_indices)

# Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=[label_encoder.classes_[i][:20] for i in top_10_indices],
    yticklabels=[label_encoder.classes_[i][:20] for i in top_10_indices],
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Top 10 Diseases', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Disease', fontsize=12, fontweight='bold')
plt.ylabel('Actual Disease', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/home/user/confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"✓ Confusion matrix saved: confusion_matrix.png")

# Create performance summary visualization
print(f"\n{'='*80}")
print("CREATING PERFORMANCE VISUALIZATION")
print("="*80)

# Load model comparison
comparison_df = pd.read_csv('/home/user/model_comparison.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model Accuracy Comparison
axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'], color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim(0, 1)
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Plot 2: F1-Score Comparison
axes[0, 1].bar(comparison_df['Model'], comparison_df['F1-Score'], color=['#3498db', '#f39c12'])
axes[0, 1].set_title('Model F1-Score Comparison', fontweight='bold')
axes[0, 1].set_ylabel('F1-Score')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(comparison_df['F1-Score']):
    axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Plot 3: Recall Comparison
axes[1, 0].bar(comparison_df['Model'], comparison_df['Recall'], color=['#9b59b6', '#1abc9c'])
axes[1, 0].set_title('Model Recall Comparison', fontweight='bold')
axes[1, 0].set_ylabel('Recall')
axes[1, 0].set_ylim(0, 1)
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(comparison_df['Recall']):
    axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Plot 4: Training Time
axes[1, 1].bar(comparison_df['Model'], comparison_df['Time (s)'], color=['#e67e22', '#95a5a6'])
axes[1, 1].set_title('Training Time Comparison', fontweight='bold')
axes[1, 1].set_ylabel('Time (seconds)')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(comparison_df['Time (s)']):
    axes[1, 1].text(i, v + 0.2, f'{v:.1f}s', ha='center', fontweight='bold')

plt.suptitle('Medical Disease Prediction Model Performance', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/home/user/model_performance.png', dpi=150, bbox_inches='tight')
print(f"✓ Performance visualization saved: model_performance.png")

# Top diseases distribution
print(f"\n{'='*80}")
print("CREATING DISEASE DISTRIBUTION CHART")
print("="*80)

top_15 = df['Final Recommendation'].value_counts().head(15)
plt.figure(figsize=(12, 8))
plt.barh(range(len(top_15)), top_15.values, color=plt.cm.viridis(np.linspace(0, 1, len(top_15))))
plt.yticks(range(len(top_15)), top_15.index)
plt.xlabel('Number of Samples', fontweight='bold')
plt.ylabel('Disease', fontweight='bold')
plt.title('Top 15 Most Frequent Diseases in Dataset', fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
for i, v in enumerate(top_15.values):
    plt.text(v + 100, i, f'{v:,}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/disease_distribution.png', dpi=150, bbox_inches='tight')
print(f"✓ Disease distribution saved: disease_distribution.png")

print(f"\n{'='*80}")
print("✓ ALL REPORTS GENERATED SUCCESSFULLY")
print("="*80)
