"""
Setup script for Medical Disease Prediction System
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("🔧 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_model_files():
    """Check if model files exist"""
    print("📦 Checking model files...")

    required_files = [
        "models/linear_svm_model.pkl",
        "models/linear_svm_vectorizer.pkl",
        "models/linear_svm_encoder.pkl"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("⚠️ Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Run training to generate model files:")
        print("   cd training")
        print("   python linear_svm_trainer.py")
        return False
    else:
        print("✅ All model files present!")
        return True

def test_system():
    """Test the prediction system"""
    print("🧪 Testing prediction system...")

    try:
        # Import and test
        sys.path.append('models')
        from models.linear_svm_predictor import LinearSVMMedicalPredictor

        predictor = LinearSVMMedicalPredictor()
        result = predictor.predict("test symptoms")

        if result['disease']:
            print("✅ System test passed!")
            return True
        else:
            print("❌ System test failed!")
            return False

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🏥 MEDICAL DISEASE PREDICTION SYSTEM - SETUP")
    print("=" * 60)

    # Step 1: Install requirements
    if not install_requirements():
        return

    # Step 2: Check model files
    if not check_model_files():
        return

    # Step 3: Test system
    if not test_system():
        return

    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\n💡 Usage:")
    print("   python predict.py \"your symptoms here\"")
    print("\n📖 Documentation:")
    print("   docs/README_FINAL.md")
    print("\n🎯 System ready for production use!")

if __name__ == "__main__":
    main()
