import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 1: DATA UNDERSTANDING & VALIDATION")
print("="*80)

# Load the dataset
df = pd.read_csv('/home/user/SYNAPSE_Cleaned.csv')

print(f"\n✓ Dataset loaded successfully!")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "="*80)
print("DATASET STRUCTURE")
print("="*80)
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print("\n" + "="*80)
print("DATA TYPES & MEMORY USAGE")
print("="*80)
print(df.info())

print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(df.describe(include='all'))

print("\n" + "="*80)
print("ISSUE DETECTION REPORT")
print("="*80)

issues = []

# 1. Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"\n1. DUPLICATE ROWS: {duplicate_count:,}")
if duplicate_count > 0:
    issues.append(f"Found {duplicate_count:,} duplicate rows")

# 2. Check for missing values
print(f"\n2. MISSING VALUES:")
missing_values = df.isnull().sum()
for col in df.columns:
    if missing_values[col] > 0:
        print(f"   - {col}: {missing_values[col]:,} ({missing_values[col]/len(df)*100:.2f}%)")
        issues.append(f"{col} has {missing_values[col]:,} missing values")

# 3. Check for empty strings
print(f"\n3. EMPTY STRINGS:")
for col in df.columns:
    empty_count = (df[col].astype(str).str.strip() == '').sum()
    if empty_count > 0:
        print(f"   - {col}: {empty_count:,} empty strings")
        issues.append(f"{col} has {empty_count:,} empty strings")

# 4. Check unique values in categorical columns
print(f"\n4. CATEGORICAL COLUMNS ANALYSIS:")
categorical_cols = ['Gender', 'Age', 'Duration', 'Severity', 'Final Recommendation']
for col in categorical_cols:
    if col in df.columns:
        unique_vals = df[col].nunique()
        print(f"   - {col}: {unique_vals} unique values")
        print(f"     Top 5: {df[col].value_counts().head().to_dict()}")

# 5. Check for inconsistent disease names
print(f"\n5. DISEASE LABELS (Final Recommendation):")
disease_counts = df['Final Recommendation'].value_counts()
print(f"   Total unique diseases: {len(disease_counts)}")
print(f"\n   Top 15 diseases:")
for disease, count in disease_counts.head(15).items():
    print(f"   - {disease}: {count:,}")

# Check for potential misspellings (similar names)
diseases = df['Final Recommendation'].unique()
print(f"\n   Checking for potential inconsistencies...")
disease_lower = {d: d.lower().strip() for d in diseases if pd.notna(d)}

# 6. Check symptom column
print(f"\n6. SYMPTOMS COLUMN:")
print(f"   Sample symptoms (first 3 rows):")
for i in range(min(3, len(df))):
    print(f"   {i+1}. {df['Symptoms'].iloc[i][:100]}...")

# 7. Check for unusual values
print(f"\n7. DATA QUALITY CHECKS:")

# Age check
if 'Age' in df.columns:
    age_values = df['Age'].value_counts()
    print(f"   - Age categories: {list(age_values.index)}")

# Gender check
if 'Gender' in df.columns:
    gender_values = df['Gender'].value_counts()
    print(f"   - Gender distribution: {dict(gender_values)}")
    
# Duration check
if 'Duration' in df.columns:
    duration_values = df['Duration'].value_counts()
    print(f"   - Duration categories: {dict(duration_values)}")

# Severity check
if 'Severity' in df.columns:
    severity_values = df['Severity'].value_counts()
    print(f"   - Severity levels: {dict(severity_values)}")

print("\n" + "="*80)
print("SUMMARY OF ISSUES FOUND")
print("="*80)
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
else:
    print("✓ No major issues detected!")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)

# Save basic statistics for later use
df.to_csv('/home/user/original_data_sample.csv', index=False)
print(f"\n✓ Original data sample saved for reference")
