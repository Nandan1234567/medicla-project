"""
=================================================================================
DATA VALIDATION UTILITY - MEDICAL PREDICTION SYSTEM
=================================================================================

Comprehensive data validation and quality assessment for medical datasets
Used for analyzing and validating training data before model training

Author: ML Engineering Team
Date: December 2024
Version: 3.0 - Production Utility
=================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

class MedicalDataValidator:
    """Comprehensive medical dataset validator"""

    def __init__(self):
        self.df = None
        self.issues = []
        self.stats = {}

    def load_and_validate(self, file_path):
        """Load dataset and perform comprehensive validation"""

        print("="*80)
        print("MEDICAL DATA VALIDATION SYSTEM")
        print("="*80)

        # Load the dataset
        try:
            self.df = pd.read_csv(file_path)
            print(f"\n✓ Dataset loaded successfully!")
            print(f"📊 Total rows: {len(self.df):,}")
            print(f"📊 Total columns: {len(self.df.columns)}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False

        # Perform validation steps
        self._validate_structure()
        self._check_data_quality()
        self._analyze_medical_content()
        self._generate_report()

        return True

    def _validate_structure(self):
        """Validate dataset structure and basic properties"""
        print(f"\n{'='*60}")
        print("STEP 1: DATASET STRUCTURE VALIDATION")
        print("="*60)

        print(f"\n📋 Columns: {list(self.df.columns)}")
        print(f"\n📊 Data Types:")
        print(self.df.dtypes)

        # Check required columns for medical prediction
        required_columns = ['Symptoms', 'Final Recommendation']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            self.issues.append(f"Missing required columns: {missing_columns}")
            print(f"❌ Missing required columns: {missing_columns}")
        else:
            print(f"✓ All required columns present")

        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"💾 Memory usage: {memory_usage:.2f} MB")
        self.stats['memory_mb'] = memory_usage

    def _check_data_quality(self):
        """Check for data quality issues"""
        print(f"\n{'='*60}")
        print("STEP 2: DATA QUALITY ASSESSMENT")
        print("="*60)

        # 1. Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        print(f"\n🔍 Duplicate rows: {duplicate_count:,}")
        if duplicate_count > 0:
            self.issues.append(f"Found {duplicate_count:,} duplicate rows")
            print(f"   ⚠️ {duplicate_count/len(self.df)*100:.2f}% of data is duplicated")

        # 2. Check for missing values
        print(f"Missing values:")
        missing_values = self.df.isnull().sum()
        total_missing = missing_values.sum()

        if total_missing > 0:
            for col in self.df.columns:
                if missing_values[col] > 0:
                    pct = missing_values[col]/len(self.df)*100
                    print(f"   - {col}: {missing_values[col]:,} ({pct:.2f}%)")
                    self.issues.append(f"{col} has {missing_values[col]:,} missing values")
        else:
            print("   ✓ No missing values found")

        # 3. Check for empty strings
        print(f"\n🔍 Empty strings:")
        empty_found = False
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                empty_count = (self.df[col].astype(str).str.strip() == '').sum()
                if empty_count > 0:
                    print(f"   - {col}: {empty_count:,} empty strings")
                    self.issues.append(f"{col} has {empty_count:,} empty strings")
                    empty_found = True

        if not empty_found:
            print("   ✓ No empty strings found")

        # 4. Check for very short symptoms
        if 'Symptoms' in self.df.columns:
            short_symptoms = (self.df['Symptoms'].astype(str).str.len() < 5).sum()
            if short_symptoms > 0:
                print(f"\n🔍 Very short symptoms (< 5 chars): {short_symptoms:,}")
                self.issues.append(f"Found {short_symptoms:,} very short symptom descriptions")

    def _analyze_medical_content(self):
        """Analyze medical-specific content"""
        print(f"\n{'='*60}")
        print("STEP 3: MEDICAL CONTENT ANALYSIS")
        print("="*60)

        # Analyze symptoms column
        if 'Symptoms' in self.df.columns:
            print(f"\n🏥 Symptoms Analysis:")
            symptoms = self.df['Symptoms'].astype(str)

     avg_length = symptoms.str.len().mean()
            print(f"   - Average symptom length: {avg_length:.1f} characters")

            # Sample symptoms
            print(f"   - Sample symptoms:")
            for i in range(min(3, len(self.df))):
                sample = symptoms.iloc[i][:80]
                print(f"     {i+1}. {sample}...")

            self.stats['avg_symptom_length'] = avg_length

        # Analyze disease labels
        if 'Final Recommendation' in self.df.columns:
            print(f"\n🏥 Disease Labels Analysis:")
            disease_counts = self.df['Final Recommendation'].value_counts()

            print(f"   - Total unique diseases: {len(disease_counts)}")
            print(f"   - Most common disease: {disease_counts.index[0]} ({disease_counts.iloc[0]:,} cases)")
            print(f"   - Least common diseases: {(disease_counts == 1).sum()} diseases with only 1 case")

            # Check for diseases with very few samples
            rare_diseases = disease_counts[disease_counts < 3]
            if len(rare_diseases) > 0:
                print(f"   ⚠️ {len(rare_diseases)} diseases have < 3 samples (may cause training issues)")
                self.issues.append(f"{len(rare_diseases)} diseases have insufficient samples (< 3)")

            # Top 10 diseases
            print(f"\n   📊 Top 10 diseases:")
            for i, (disease, count) in enumerate(disease_counts.head(10).items(), 1):
                pct = count/len(self.df)*100
                print(f"     {i:2d}. {disease[:40]:40} {count:6,} ({pct:5.2f}%)")

            self.stats['total_diseases'] = len(disease_counts)
            self.stats['rare_diseases'] = len(rare_diseases)

        # Analyze other columns if present
        categorical_cols = ['Gender', 'Age', 'Duration', 'Severity']
        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n🏥 {col} Analysis:")
                values = self.df[col].value_counts()
                print(f"   - Unique values: {len(values)}")
                print(f"   - Distribution: {dict(values.head())}")

    def _generate_report(self):
        """Generate comprehensive validation report"""
        print(f"\n{'='*60}")
        print("VALIDATION REPORT SUMMARY")
        print("="*60)

        # Overall statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   - Total records: {len(self.df):,}")
        print(f"   - Total features: {len(self.df.columns)}")
        print(f"   - Memory usage: {self.stats.get('memory_mb', 0):.2f} MB")

        if 'total_diseases' in self.stats:
            print(f"   - Unique diseases: {self.stats['total_diseases']:,}")

        if 'avg_symptom_length' in self.stats:
            print(f"   - Avg symptom length: {self.stats['avg_symptom_length']:.1f} chars")

        # Issues found
        print(f"\n🔍 Issues Identified:")
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("   ✓ No major issues detected!")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if len(self.issues) == 0:
            print("   ✓ Dataset is ready for training!")
        else:
            print("   1. Address identified issues before training")
            if any("duplicate" in issue.lower() for issue in self.issues):
                print("   2. Remove duplicate rows")
            if any("missing" in issue.lower() for issue in self.issues):
                print("   3. Handle missing values (remove or impute)")
            if any("insufficient" in issue.lower() for issue in self.issues):
                print("   4. Consider removing diseases with < 3 samples")

        # Data quality score
        total_issues = len(self.issues)
        if total_issues == 0:
            quality_score = 100
        elif total_issues <= 2:
            quality_score = 85
        elif total_issues <= 5:
            quality_score = 70
        else:
            quality_score = 50

        print(f"\n🎯 Data Quality Score: {quality_score}/100")

        if quality_score >= 85:
            print("   ✅ Excellent - Ready for production training")
        elif quality_score >= 70:
            print("   ⚠️ Good - Minor issues to address")
        else:
            print("   ❌ Poor - Significant cleanup required")

    def save_sample(self, output_path='data_sample.csv', n_samples=1000):
        """Save a sample of the data for inspection"""
        if self.df is not None:
            sample_df = self.df.sample(n=min(n_samples, len(self.df)), random_state=42)
            sample_df.to_csv(output_path, index=False)
            print(f"\n💾 Data sample saved: {output_path}")
            return True
        return False

def validate_medical_dataset(file_path, save_sample=True):
    """Standalone function to validate medical dataset"""
    validator = MedicalDataValidator()

    if validator.load_and_validate(file_path):
        if save_sample:
            validator.save_sample('original_data_sample.csv')
        return validator
    return None

def main():
    """Main validation function"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_validation.py <dataset_path>")
        print("Example: python data_validation.py ../data/cleaned_dataset.csv")
        return

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Run validation
    validator = validate_medical_dataset(file_path)

    if validator:
        print(f"\n✅ Validation complete!")
    else:
        print(f"\n❌ Validation failed!")

if __name__ == "__main__":
    main()
