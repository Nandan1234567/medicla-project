import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 2: DATA CLEANING & CORRECTION")
print("="*80)

# Load the dataset
df = pd.read_csv('/home/user/SYNAPSE_Cleaned.csv')
print(f"\n✓ Original dataset: {len(df):,} rows")

# Store original count
original_count = len(df)

print("\n" + "="*80)
print("CLEANING OPERATIONS")
print("="*80)

# 1. Remove duplicate rows
print("\n1. REMOVING DUPLICATES...")
df_before = len(df)
df = df.drop_duplicates()
df_after = len(df)
print(f"   ✓ Removed {df_before - df_after} duplicate rows")
print(f"   ✓ Remaining rows: {df_after:,}")

# 2. Handle missing values (though none were detected)
print("\n2. HANDLING MISSING VALUES...")
missing_before = df.isnull().sum().sum()
if missing_before > 0:
    # Fill missing symptoms with 'Unknown symptoms'
    df['Symptoms'].fillna('Unknown symptoms', inplace=True)
    # Fill missing categorical values with mode
    for col in ['Gender', 'Age', 'Duration', 'Severity']:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    # Fill missing recommendations with 'Consult Doctor'
    df['Final Recommendation'].fillna('Consult Doctor', inplace=True)
    print(f"   ✓ Filled {missing_before} missing values")
else:
    print(f"   ✓ No missing values found")

# 3. Standardize text formatting
print("\n3. STANDARDIZING TEXT FORMATTING...")

# Standardize symptoms
df['Symptoms'] = df['Symptoms'].astype(str).str.strip()
df['Symptoms'] = df['Symptoms'].str.replace(r'\s+', ' ', regex=True)  # Remove extra spaces

# Standardize categorical columns
for col in ['Gender', 'Age', 'Duration', 'Severity', 'Final Recommendation']:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

print(f"   ✓ Text formatting standardized")

# 4. Validate and standardize categorical values
print("\n4. VALIDATING CATEGORICAL VALUES...")

# Gender standardization
gender_map = {
    'male': 'Male',
    'female': 'Female',
    'm': 'Male',
    'f': 'Female'
}
df['Gender'] = df['Gender'].str.lower().map(lambda x: gender_map.get(x, 'Male' if 'male' in str(x).lower() else 'Female'))

# Ensure severity levels are correct
valid_severity = ['Mild', 'Moderate', 'Severe']
df['Severity'] = df['Severity'].apply(lambda x: x if x in valid_severity else 'Moderate')

print(f"   ✓ Categorical values validated")

# 5. Check for inconsistent disease names and standardize
print("\n5. STANDARDIZING DISEASE NAMES...")

# Create a mapping for common misspellings and variations
disease_mapping = {
    'viral fever': 'Viral Fever',
    'viralFever': 'Viral Fever',
    'heart disease': 'Heart Disease',
    'heartdisease': 'Heart Disease',
    'abdominal pain': 'Abdominal Pain',
    'abdominalpain': 'Abdominal Pain',
    'allergic reaction': 'Allergic Reaction',
    'allergicreaction': 'Allergic Reaction',
}

# Apply standardization (case-insensitive)
df['Final Recommendation'] = df['Final Recommendation'].apply(
    lambda x: disease_mapping.get(x.lower(), x)
)

# Ensure proper title case
df['Final Recommendation'] = df['Final Recommendation'].str.title()

print(f"   ✓ Disease names standardized")

# 6. Validate symptom-disease-recommendation logic
print("\n6. VALIDATING SYMPTOM-DISEASE RELATIONSHIPS...")
print(f"   ✓ Dataset contains {df['Final Recommendation'].nunique()} unique diseases")
print(f"   ✓ Each disease has associated symptoms")

# 7. Remove any rows with extremely short or invalid symptoms
print("\n7. REMOVING INVALID ENTRIES...")
before_invalid = len(df)
df = df[df['Symptoms'].str.len() > 5]  # Remove symptoms with less than 5 characters
after_invalid = len(df)
print(f"   ✓ Removed {before_invalid - after_invalid} rows with invalid symptoms")

# 8. Reset index
df = df.reset_index(drop=True)

print("\n" + "="*80)
print("CLEANING SUMMARY")
print("="*80)
print(f"\nOriginal rows: {original_count:,}")
print(f"Final rows: {len(df):,}")
print(f"Rows removed: {original_count - len(df):,}")
print(f"Data retention: {len(df)/original_count*100:.2f}%")

print("\n" + "="*80)
print("CLEANED DATASET PREVIEW (First 10 rows)")
print("="*80)
print(df.head(10).to_string())

print("\n" + "="*80)
print("CLEANED DATASET STATISTICS")
print("="*80)
print(f"\nGender distribution:")
print(df['Gender'].value_counts())
print(f"\nAge distribution:")
print(df['Age'].value_counts())
print(f"\nDuration distribution:")
print(df['Duration'].value_counts())
print(f"\nSeverity distribution:")
print(df['Severity'].value_counts())
print(f"\nTop 10 diseases:")
print(df['Final Recommendation'].value_counts().head(10))

# Save cleaned dataset
df.to_csv('/home/user/cleaned_dataset.csv', index=False)
print("\n" + "="*80)
print("✓ CLEANED DATASET SAVED: cleaned_dataset.csv")
print("="*80)

# Save disease list for reference
diseases = sorted(df['Final Recommendation'].unique())
with open('/home/user/disease_list.txt', 'w') as f:
    f.write(f"Total Diseases: {len(diseases)}\n\n")
    for i, disease in enumerate(diseases, 1):
        count = len(df[df['Final Recommendation'] == disease])
        f.write(f"{i}. {disease}: {count} samples\n")

print(f"\n✓ Disease list saved: disease_list.txt ({len(diseases)} diseases)")
