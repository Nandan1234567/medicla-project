# 📚 Medical Healthcare System - Complete Setup Guide

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Running the Application](#running-the-application)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### Software Requirements

- **Python**: 3.7 or higher
- **Web Browser**: Chrome, Firefox, Safari, or Edge
- **Text Editor** (optional): VS Code, Sublime, or any editor

### Python Packages

```bash
pip install flask flask-cors numpy pandas scikit-learn
```

---

## Installation Steps

### Step 1: Extract the Files

1. Extract the `medical_frontend.zip` file
2. You should see this structure:

```
medical_frontend/
├── index.html
├── styles.css
├── script.js
├── backend.py
├── README.md
├── SETUP_GUIDE.md
└── requirements.txt
```

### Step 2: Install Dependencies

Open terminal/command prompt in the folder and run:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install flask
pip install flask-cors
pip install numpy
pip install pandas
pip install scikit-learn
```

### Step 3: Add ML Model Files (Optional but Recommended)

Copy these files to the `medical_frontend` folder:
- `final_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`

**Note**: The system works without these (demo mode), but predictions won't be accurate.

---

## Running the Application

### Method 1: Full Setup (Recommended)

**Step 1: Start the Backend**

```bash
python backend.py
```

You should see:
```
Medical Healthcare Recommendations API
Model Status: Loaded
Server starting on http://localhost:5000
```

**Step 2: Open the Frontend**

- Double-click `index.html`, or
- Right-click `index.html` → Open with → Your Browser, or
- Drag `index.html` into your browser

### Method 2: Frontend Only (Demo Mode)

Just open `index.html` in your browser!

- Works without backend
- Uses mock predictions for testing
- Perfect for UI testing

---

## Project Structure

### Frontend Files

**index.html** (Main Structure)
- Patient input form
- Results display sections
- Responsive layout
- Accessibility features

**styles.css** (Styling)
- Leaf green theme colors
- Light/dark mode support
- Responsive breakpoints
- Smooth animations

**script.js** (Logic)
- Form validation
- API communication
- Theme management
- UI state control

### Backend File

**backend.py** (Flask API)
- `/predict` endpoint - Disease prediction
- `/health` endpoint - Server status
- ML model integration
- CORS support

---

## Configuration

### 1. Change Backend URL

Edit `script.js` (line ~20):

```javascript
const CONFIG = {
    API_URL: 'http://your-server:5000/predict',  // Change here
    // ...
};
```

### 2. Change Server Port

Edit `backend.py` (last line):

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

### 3. Customize Theme Colors

Edit `styles.css` (lines 8-15):

```css
:root {
    --primary-color: #2d7a3e;      /* Your green */
    --primary-light: #45a857;      /* Lighter shade */
    --primary-dark: #1e5a2c;       /* Darker shade */
}
```

### 4. Add/Modify Recommendations

Edit `backend.py` (lines ~40-100):

```python
RECOMMENDATIONS = {
    'Disease Name': [
        'Recommendation 1',
        'Recommendation 2',
        // Add more...
    ]
}
```

---

## Troubleshooting

### Problem: Backend Won't Start

**Error**: "Address already in use"

**Solution**:
```bash
# Find process using port 5000
netstat -ano | findstr :5000    # Windows
lsof -i :5000                   # Mac/Linux

# Kill the process or change port in backend.py
```

**Error**: "Module not found"

**Solution**:
```bash
pip install flask flask-cors numpy pandas scikit-learn
```

### Problem: Frontend Can't Connect

**Error**: "Failed to fetch" in console

**Solution**:
1. Check backend is running: Visit `http://localhost:5000/health`
2. Check CORS is enabled in `backend.py`
3. Verify API_URL in `script.js` is correct

### Problem: Predictions Are Wrong

**Cause**: Using demo mode without ML models

**Solution**:
1. Copy the 3 `.pkl` files to the folder
2. Restart the backend
3. Check console for "Model loaded successfully"

### Problem: Theme Won't Switch

**Solution**:
1. Clear browser cache
2. Check browser console for errors
3. Try in incognito/private mode

### Problem: Mobile Layout Broken

**Solution**:
1. Clear cache
2. Check viewport meta tag in `index.html`
3. Test in different browsers

---

## Advanced Usage

### Deploy to Production

#### 1. Using Gunicorn (Linux/Mac)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend:app
```

#### 2. Using Waitress (Windows)

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 backend:app
```

### Host on a Server

1. **Upload files** to your server
2. **Install dependencies** on server
3. **Configure firewall** to allow port 5000
4. **Update API_URL** in `script.js` to your server IP
5. **Use a reverse proxy** (nginx/Apache) for production

### Enable HTTPS

Add to nginx config:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:5000;
    }
}
```

### Add Authentication

In `backend.py`:

```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ...
```

### Database Integration

Add PostgreSQL for logging:

```python
import psycopg2

@app.route('/predict', methods=['POST'])
def predict():
    # ... prediction code ...
    
    # Log to database
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (symptoms, disease, confidence) VALUES (%s, %s, %s)",
        (symptoms, disease, confidence)
    )
    conn.commit()
    
    return jsonify(response)
```

---

## Testing

### Manual Testing Checklist

- [ ] Form validation works
- [ ] All dropdowns selectable
- [ ] Submit button triggers prediction
- [ ] Loading animation shows
- [ ] Results display correctly
- [ ] Confidence bar animates
- [ ] Theme toggle works
- [ ] Print function works
- [ ] Mobile responsive
- [ ] Dark mode works

### Automated Testing

Create `test_backend.py`:

```python
import pytest
from backend import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    rv = client.get('/health')
    assert rv.status_code == 200

def test_predict(client):
    rv = client.post('/predict', json={
        'symptoms': 'fever headache',
        'age': '31-45 years',
        'gender': 'Male',
        'duration': '3-7 days',
        'severity': 'Moderate'
    })
    assert rv.status_code == 200
    assert 'disease' in rv.json
```

Run tests:
```bash
pytest test_backend.py
```

---

## Performance Optimization

### Frontend

1. **Minify CSS/JS** for production
2. **Enable browser caching**
3. **Use CDN** for Font Awesome
4. **Lazy load** images

### Backend

1. **Cache predictions** for common symptoms
2. **Use connection pooling**
3. **Enable gzip compression**
4. **Rate limiting** to prevent abuse

---

## Security Best Practices

1. **Input validation** - Sanitize all user inputs
2. **HTTPS only** - Use SSL in production
3. **API rate limiting** - Prevent abuse
4. **CORS configuration** - Restrict origins
5. **Environment variables** - Don't hardcode secrets
6. **Regular updates** - Keep dependencies current

---

## Getting Help

### Check These First

1. Browser console (F12) for JavaScript errors
2. Backend terminal for Python errors
3. Network tab for API call issues
4. README.md for quick reference

### Common Solutions

- **Restart backend** after code changes
- **Clear browser cache** for CSS/JS changes
- **Check file paths** are correct
- **Verify permissions** on files

---

## Next Steps

After setup:

1. ✅ Test with sample symptoms
2. ✅ Try both light and dark themes
3. ✅ Test on mobile device
4. ✅ Customize colors to your preference
5. ✅ Add your own recommendations
6. ✅ Deploy to production (optional)

---

**Need More Help?** Check the README.md or review the code comments!

Enjoy using the Medical Healthcare Recommendations System! 🏥🌿
