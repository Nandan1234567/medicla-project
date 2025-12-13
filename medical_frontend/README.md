# 🏥 Medical Healthcare Recommendations System

A complete, production-ready web application for AI-powered disease prediction and medical recommendations.

## ✨ Features

- **🎨 Beautiful Leaf Green Theme** - Health-focused color palette
- **🌓 Light/Dark Mode** - Comfortable viewing in any environment
- **📱 Fully Responsive** - Works on desktop, tablet, and mobile
- **🤖 AI-Powered Predictions** - 85.86% accuracy on 244 diseases
- **💊 Medical Recommendations** - Personalized health guidance
- **🛡️ Safety Precautions** - Disease-specific care instructions
- **🖨️ Print Results** - Export predictions for records
- **⚡ Fast & Lightweight** - Optimized performance
- **🔒 Privacy-First** - All processing can be done locally

## 📦 What's Included

```
medical_frontend/
├── index.html          # Main HTML structure
├── styles.css          # Leaf green theme styling
├── script.js           # Frontend logic & API integration
├── backend.py          # Flask backend server
├── README.md           # This file
├── SETUP_GUIDE.md      # Detailed setup instructions
└── requirements.txt    # Python dependencies
```

## 🚀 Quick Start

### Option 1: Full Setup (with ML Model)

1. **Install Python dependencies:**
```bash
pip install flask flask-cors numpy pandas scikit-learn
```

2. **Copy ML model files to this folder:**
   - `final_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `label_encoder.pkl`

3. **Start the backend server:**
```bash
python backend.py
```

4. **Open the frontend:**
   - Simply double-click `index.html`
   - Or open it in your browser

### Option 2: Frontend Only (Demo Mode)

1. **Just open `index.html` in your browser!**
   - The frontend works standalone with mock predictions
   - No backend required for testing the UI

## 🎯 Usage

1. **Describe your symptoms** in the text area
2. **Select your details:**
   - Age group
   - Gender
   - Symptom duration
   - Severity level
3. **Click "Get Health Recommendations"**
4. **View results:**
   - Predicted disease
   - Confidence level
   - Medical recommendations
   - Precautions & care instructions

## 🔧 Configuration

### Backend API URL

Edit `script.js` to change the backend URL:

```javascript
const CONFIG = {
    API_URL: 'http://localhost:5000/predict',  // Change this if needed
    // ...
};
```

### Theme Colors

Customize colors in `styles.css`:

```css
:root {
    --primary-color: #2d7a3e;      /* Main green color */
    --primary-light: #45a857;      /* Light green */
    --primary-dark: #1e5a2c;       /* Dark green */
    /* ... */
}
```

## 🌐 Backend API

### Endpoints

**POST /predict**
```json
Request:
{
    "symptoms": "fever, headache, body ache",
    "age": "31-45 years",
    "gender": "Male",
    "duration": "3-7 days",
    "severity": "Moderate"
}

Response:
{
    "disease": "Viral Fever",
    "confidence": 87.34,
    "recommendations": [...],
    "precautions": [...],
    "metadata": {...}
}
```

**GET /health**
- Check if server is running
- Returns model load status

## 📱 Browser Compatibility

- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Opera
- ⚠️ Internet Explorer (limited support)

## 🎨 Theme Features

### Light Mode (Default)
- Fresh leaf green colors
- High contrast for readability
- Professional medical aesthetic

### Dark Mode
- Comfortable on the eyes
- Deep green accents
- Perfect for night usage

Toggle between themes using the button in the top-right corner!

## 🔒 Privacy & Security

- **No data storage** - predictions are not saved
- **Local processing** - all data stays on your machine
- **CORS enabled** - secure cross-origin requests
- **No tracking** - completely private

## ⚡ Performance

- **Lightweight** - < 100KB total size
- **Fast loading** - Optimized assets
- **Smooth animations** - 60fps transitions
- **Responsive** - Instant UI feedback

## 🐛 Troubleshooting

### Backend Won't Start
- Check if port 5000 is available
- Ensure all dependencies are installed
- Verify model files are present

### Predictions Fail
- Check backend is running: `http://localhost:5000/health`
- Verify CORS is enabled
- Check browser console for errors

### UI Issues
- Clear browser cache
- Try a different browser
- Check console for JavaScript errors

## 📄 License

This project is part of the Medical Healthcare Recommendations System.
For educational and research purposes.

## ⚠️ Medical Disclaimer

This AI prediction system is for informational purposes only and should not replace professional medical advice. Always consult qualified healthcare providers for proper diagnosis and treatment.

## 🆘 Support

For issues or questions:
1. Check `SETUP_GUIDE.md` for detailed instructions
2. Review troubleshooting section above
3. Check backend logs for errors
4. Verify all files are present

## 🎉 Credits

- **ML Model**: 85.86% accuracy on 244 diseases
- **Frontend**: Modern responsive design
- **Backend**: Flask REST API
- **Theme**: Healthcare-focused leaf green aesthetic

---

**Ready to use!** Just open `index.html` and start predicting! 🚀
