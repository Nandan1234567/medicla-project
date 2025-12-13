@echo off
echo 🚀 Starting Medical Healthcare Backend...
cd backend_flask

echo Installing Python dependencies...
pip install -r requirements.txt

echo Starting Flask server...
python app.py

pause
