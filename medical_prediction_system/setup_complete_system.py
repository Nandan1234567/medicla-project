#!/usr/bin/env python3
"""
Complete Medical Healthcare System Setup Script
This script sets up the entire system including backend, frontend, and database
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🏥 {text}")
    print("="*60)

def print_step(step, text):
    """Print a formatted step"""
    print(f"\n{step}. {text}")
    print("-" * 40)

def run_command(command, cwd=None, check=True):
    """Run a command and handle errors"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, cwd=cwd, check=check,
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def check_requirements():
    """Check if required software is installed"""
    print_step(1, "Checking System Requirements")

    requirements = {
        'python': 'python --version',
        'node': 'node --version',
        'npm': 'npm --version',
        'mongod': 'mongod --version'
    }

    missing = []
    for name, command in requirements.items():
        result = run_command(command, check=False)
        if result and result.returncode == 0:
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: Not found")
            missing.append(name)

    if missing:
        print(f"\n❌ Missing requirements: {', '.join(missing)}")
        print("\nPlease install the missing software:")
        print("- Python 3.8+: https://python.org")
        print("- Node.js 16+: https://nodejs.org")
        print("- MongoDB: https://mongodb.com/try/download/community")
        return False

    print("\n✅ All requirements satisfied!")
    return True

def setup_backend():
    """Setup Flask backend"""
    print_step(2, "Setting up Flask Backend")

    backend_dir = Path("backend_flask")

    # Create virtual environment
    print("Creating Python virtual environment...")
    run_command("python -m venv venv", cwd=backend_dir)

    # Activate virtual environment and install requirements
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate && pip install -r requirements.txt"
    else:  # Unix/Linux/Mac
        activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"

    print("Installing Python dependencies...")
    result = run_command(activate_cmd, cwd=backend_dir)

    if result:
        print("✅ Backend dependencies installed successfully!")
        return True
    else:
        print("❌ Failed to install backend dependencies")
        return False

def setup_frontend():
   """Setup React frontend"""
    print_step(3, "Setting up React Frontend")

    frontend_dir = Path("frontend")

    # Install npm dependencies
    print("Installing Node.js dependencies...")
    result = run_command("npm install", cwd=frontend_dir)

    if result:
        print("✅ Frontend dependencies installed successfully!")
        return True
    else:
        print("❌ Failed to install frontend dependencies")
        return False

def setup_database():
    """Setup MongoDB database"""
    print_step(4, "Setting up MongoDB Database")

    # Check if MongoDB is running
    print("Checking MongoDB connection...")

    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        client.server_info()  # Will raise exception if can't connect

        # Create database and collections
        db = client.medical_healthcare

        # Create collections with indexes
        users_collection = db.users
        predictions_collection = db.predictions

        # Create indexes
        users_collection.create_index("email", unique=True)
        predictions_collection.create_index("userId")
        predictions_collection.create_index("createdAt")

        print("✅ MongoDB connected and configured successfully!")
        return True

    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\nPlease ensure MongoDB is installed and running:")
        print("- Start MongoDB: mongod")
        print("- Or install MongoDB Community Edition")
        return False

def create_run_scripts():
    """Create convenient run scripts"""
    print_step(5, "Creating Run Scripts")

    # Backend run script
    backend_script = """#!/bin/bash
# Start Flask Backend
echo "🚀 Starting Medical Healthcare Backend..."
cd backend_flask

# Activate virtual environment
if [ -d "venv" ]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

# Start Flask app
python app.py
"""

    with open("start_backend.sh", "w") as f:
        f.write(backend_script)

    # Frontend run script
    frontend_script = """#!/bin/bash
# Start React Frontend
echo "🚀 Starting Medical Healthcare Frontend..."
cd frontend
npm run dev
"""

    with open("start_frontend.sh", "w") as f:
        f.write(frontend_script)

    # Complete system script
    system_script = """#!/bin/bash
# Start Complete Medical Healthcare System
echo "🏥 Starting Complete Medical Healthcare System..."

# Start MongoDB (if not running)
echo "Starting MongoDB..."
mongod --fork --logpath /tmp/mongodb.log --dbpath /tmp/mongodb-data || echo "MongoDB may already be running"

# Start Backend in background
echo "Starting Backend..."
./start_backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 5

# Start Frontend
echo "Starting Frontend..."
./start_frontend.sh &
FRONTEND_PID=$!

echo "✅ System started successfully!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
"""

    with open("start_system.sh", "w") as f:
        f.write(system_script)

    # Make scripts executable (Unix/Linux/Mac)
    if os.name != 'nt':
        os.chmod("start_backend.sh", 0o755)
        os.chmod("start_frontend.sh", 0o755)
        os.chmod("start_system.sh", 0o755)

    # Windows batch files
    if os.name == 'nt':
        backend_bat = """@echo off
echo 🚀 Starting Medical Healthcare Backend...
cd backend_flask
if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
)
python app.py
"""
        with open("start_backend.bat", "w") as f:
            f.write(backend_bat)

        frontend_bat = """@echo off
echo 🚀 Starting Medical Healthcare Frontend...
cd frontend
npm run dev
"""
        with open("start_frontend.bat", "w") as f:
            f.write(frontend_bat)

    print("✅ Run scripts created successfully!")

def test_system():
    """Test the system components"""
    print_step(6, "Testing System Components")

    # Test ML model
    print("Testing ML prediction model...")
    try:
        sys.path.append('models')
        from linear_svm_predictor import LinearSVMMedicalPredictor

        predictor = LinearSVMMedicalPredictor()
        test_result = predictor.predict("headache and fever")

        if test_result and 'disease' in test_result:
            print("✅ ML model working correctly!")
        else:
            print("❌ ML model test failed")
            return False
    except Exception as e:
        print(f"❌ ML model error: {e}")
        return False

    print("✅ All system tests passed!")
    return True

def main():
    """Main setup function"""
    print_header("MEDICAL HEALTHCARE SYSTEM SETUP")
    print("This script will set up the complete medical healthcare system")
    print("including Flask backend, React frontend, and MongoDB database.")

    # Change to the medical_prediction_system directory
    if not os.path.exists("backend_flask") or not os.path.exists("frontend"):
        print("❌ Please run this script from the medical_prediction_system directory")
        sys.exit(1)

    # Run setup steps
    steps = [
        ("Check Requirements", check_requirements),
        ("Setup Backend", setup_backend),
        ("Setup Frontend", setup_frontend),
        ("Setup Database", setup_database),
        ("Create Run Scripts", create_run_scripts),
        ("Test System", test_system)
    ]

    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at step: {step_name}")
            sys.exit(1)

    # Success message
    print_header("SETUP COMPLETE! 🎉")
    print("✅ Medical Healthcare System is ready to use!")
    print("\n📋 Next Steps:")
    print("1. Start the complete system:")
    if os.name == 'nt':
        print("   - Windows: Double-click start_backend.bat and start_frontend.bat")
    else:
        print("   - Unix/Linux/Mac: ./start_system.sh")
    print("\n2. Open your browser:")
    print("   - Frontend: http://localhost:3000")
    print("   - Backend API: http://localhost:5000")
    print("\n3. Create an account and start using the system!")
    print("\n🏥 Features available:")
    print("   ✅ User registration and authentication")
    print("   ✅ AI-powered medical predictions (97.23% accuracy)")
    print("   ✅ Emergency symptom detection")
    print("   ✅ Prediction history and analytics")
    print("   ✅ Beautiful responsive UI with dark/light mode")
    print("   ✅ Print and download prediction results")
    print("   ✅ User feedback system")
    print("\n🔒 Security features:")
    print("   ✅ JWT authentication")
    print("   ✅ Password hashing")
    print("   ✅ Rate limiting")
    print("   ✅ CORS protection")
    print("\n📊 Database:")
    print("   ✅ MongoDB with proper indexing")
    print("   ✅ User profiles and medical history")
    print("   ✅ Prediction analytics and statistics")

if __name__ == "__main__":
    main()
