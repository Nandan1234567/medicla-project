import pandas as pd
import numpy as np
import re
import string
import os
import sys

# Add parent directory to path to handle imports if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def preprocess_text(text):
    """
    Apply requested text cleaning:
    - Convert to lowercase
    - Remove punctuation
    - Fix common spelling mistakes (basic mapping)
    - Remove light stopwords
    - PRESERVE medical terms
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    # We replace punctuation with space to avoid merging words (e.g. "pain,fever" -> "pain fever")
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # 3. Fix common spelling mistakes (Expanded list based on common medical typos)
    corrections = {
        'headche': 'headache',
        'headach': 'headache',
        'stomach ache': 'stomach pain',
        'stomache': 'stomach pain',
        'temprature': 'temperature',
        'feverr': 'fever',
        'vomitting': 'vomiting',
        'diarea': 'diarrhea',
        'diarhea': 'diarrhea',
        'breathing difficulty': 'difficulty breathing',
        'breathin': 'breathing',
        'shortnes': 'shortness',
        'weekness': 'weakness',
        'dizzines': 'dizziness',
        'nauesa': 'nausea',
        'bloood': 'blood',
        'bleeding': 'bleeding', # ensure correct form
    }

    words = text.split()
    corrected_words = [corrections.get(w, w) for w in words]

    # 4. Remove stopwords (Light list)
    # We barely remove anything to be safe with medical context, just very common noise
    stopwords = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'so', 'not', 'no', 'can', 'could', 'may', 'might', 'will', 'would', 'shall', 'should', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'}

    cleaned_words = [w for w in corrected_words if w not in stopwords]

    # Join back
    text = ' '.join(cleaned_words)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def main():
    print("="*60)
    print("DATA PREPROCESSING SCRIPT")
    print("="*60)

    # Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_dir, 'data', 'cleaned_dataset.csv')
    output_path = os.path.join(base_dir, 'data', 'final_cleaned_dataset.csv')

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at {input_path}")
        return

    print(f"Reading from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Original dataset size: {len(df):,} rows")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 1. Missing Values
    print("\n--- Cleaning Missing Values --")
    initial_count = len(df)
    df.dropna(subset=['Symptoms', 'Final Recommendation'], inplace=True)
    df.fillna('', inplace=True) # Fill other columns with empty string if relevant
    print(f"✓ Removed {initial_count - len(df)} rows with missing critical data.")

    # 2. Empty Symptoms
    print("\n--- Removing Empty/Short Symptoms ---")
    # First pass cleaning to check for truly empty after stripping
    df['Symptoms'] = df['Symptoms'].astype(str).str.strip()
    df['Final Recommendation'] = df['Final Recommendation'].astype(str).str.strip()

    initial_count = len(df)
    df = df[df['Symptoms'].str.len() > 0]
    df = df[df['Final Recommendation'].str.len() > 0]
    print(f"✓ Removed {initial_count - len(df)} rows with empty text.")

    # 3. Duplicates
    print("\n--- Removing Duplicates ---")
    initial_count = len(df)
    df.drop_duplicates(subset=['Symptoms', 'Final Recommendation'], inplace=True)
    print(f"✓ Removed {initial_count - len(df)} duplicate rows.")

    # 4. Text Preprocessing
    print("\n--- Applying Text Cleaning ---")
    print("Converting to lowercase, removing punctuation, fixing spelling, light stopword removal...")

    # Apply to Symptoms
    df['Symptoms'] = df['Symptoms'].apply(preprocess_text)

    # Also clean Target slightly (normalize spaces/case for consistency, but keep original casing often preferred for display?
    # User said "Convert to lowercase" generally. Let's lowercase the target for training consistency,
    # but might want a mapping for display. For now, we will normalize target to Title Case or lower.
    # Usually better to keep target as is or standardized. Let's Title Case the target to look nice.)
    df['Final Recommendation'] = df['Final Recommendation'].str.title().str.strip()

    # Remove rows that might have become empty after preprocessing
    df = df[df['Symptoms'].str.len() > 0]

    print(f"✓ Final dataset size: {len(df):,} rows")

    # Save
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("✅ PREPROCESSING COMPLETE!")

if __name__ == "__main__":
    main()
