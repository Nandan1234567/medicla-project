# Medical Health Prediction System (AI-Powered)

An advanced web-based healthcare assistant that leverages Machine Learning (SVM) to predict diseases based on user symptoms. The system provides detailed reports, severity analysis, and preventative recommendations.

## 🚀 Key Features
- **Intelligent Disease Prediction**: Uses a Linear Support Vector Machine (SVM) model trained on medical datasets.
- **Severity Classification**:
  - 🟢 **MILD**: Common ailments (Cold, Flu) - Self-care.
  - 🟡 **MODERATE**: Requires medical attention.
  - 🔴 **EMERGENCY**: Critical conditions (Chest pain, Breathing issues) - Immediate care required.
- **Dashboard & Reporting**:
  - Interactive User Dashboard.
  - **PDF Health Report**: Downloadable detailed status report with severity tags.
- **Authentication**:
  - Email/Password Registration.
  - **GitHub OAuth Integration**.
- **Modern UI**: "Leaf" thme with Glassmorphism effects.

---

## 📂 Project Structure

The project is organized into modular components for scalability:

```
medical_prediction_system/
├── backend_flask/          # Flask Backend Application
│   ├── app.py              # Main Entry Point (API Routes, Auth, Config)
├── frontend/               # Static Frontend Assets
│   ├── index.html          # Landing Page (Vision & Team)
│   ├── dashboard.html      # Main User Interface
│   ├── login.html          # Authentication Page
│   ├── js/                 # Application Logic (API calls, UI handling)
│   └── css/                # Styling (Glassmorphism, Animations)
├── data/                   # Dataset Files
│   └── final_cleaned_dataset.csv # Training Data
├── models/                 # Machine Learning Models
│   └── linear_svm_predictor.py # Model Logic & Training Code
├── requirements.txt        # Python Dependencies
└── README.md               # This Documentation
```

---

## 🛠️ Prerequisites

Before running the project, ensure you have the following installed:
1.  **Python 3.8+**: [Download Here](https://www.python.org/downloads/)
2.  **MongoDB**: Must be installed and running locally. [Download Community Server](https://www.mongodb.com/try/download/community)
    -   Default Port: `27017`

---

## 📥 Installation Guide

### 1. Clone the Repository
```bash
git clone <repository_url>
cd medical_prediction_system
```

### 2. Install Dependencies
Navigate to the backend directory and install the required Python packages:

```bash
cd backend_flask
pip install -r ../requirements.txt
# Additional requirements for new features:
pip install Authlib requests
```

### 3. Verify MongoDB
Ensure your MongoDB service is running in the background.
- **Windows**: Open Task Manager > Services > `MongoDB`.
- **Linux/Mac**: `sudo systemctl status mongod`

---

## ⚙️ Configuration (GitHub Auth)

To enable the "Sign in with GitHub" feature:

1.  Go to **[GitHub Developer Settings](https://github.com/settings/applications/new)**.
2.  Register a new OAuth Application:
    -   **Application Name**: Medical AI
    -   **Homepage URL**: `http://localhost:5000`
    -   **Authorization Callback URL**: `http://localhost:5000/auth/github/callback`
3.  **Copy Credentials**: Get your **Client ID** and **Client Secret**.
4.  **Update `app.py`**:
    Open `backend_flask/app.py` and replace the placeholders:
    ```python
    app.config['GITHUB_CLIENT_ID'] = 'YOUR_ACTUAL_CLIENT_ID'
    app.config['GITHUB_CLIENT_SECRET'] = 'YOUR_ACTUAL_CLIENT_SECRET'
    ```

---

## ▶️ Running the Application

### Start the Server
1.  Open your terminal/command prompt.
2.  Navigate to `backend_flask`:
    ```bash
    cd backend_flask
    ```
3.  Run the application:
    ```bash
    python app.py
    ```
4.  You should see output indicating the server is running on `http://0.0.0.0:5000`.

### Access the Web App
Open your browser and navigate to:
👉 **[http://localhost:5000/](http://localhost:5000/)**

---

## 📖 Usage Guide

1.  **Landing Page**: Check out "Our Vision" and "Meet the Developers".
2.  **Sign Up / Login**:
    -   Create an account or use GitHub Login.
3.  **Predict**:
    -   Enter symptoms separated by commas (e.g., `fever, cough, chest pain`).
    -   Click **Analyze Symptoms**.
4.  **View Results**:
    -   See the **Predicted Disease** and **Confidence Score**.
    -   Check the **Severity Badge** (Green/Yellow/Red).
    -   Read **Recommendations** and **Precautions**.
5.  **Download Report**:
    -   Click the **"📄 Download Report"** button at the top of the results card to get your personalized PDF.

---

## 👥 Developers

- **Basvanthraya**
- **Manoj Gowda**
- **Nandan**
- **Praveen**

---

## ❗ Troubleshooting

-   **500 Internal Server Error**: Check the terminal running `app.py` for error logs. Often caused by MongoDB connection issues or missing libraries.
-   **PDF Button Missing**: Use **Ctrl+F5** to hard refresh the browser to clear old caches.
-   **GitHub Login Failed**: Ensure your Client ID/Secret are correct and the Callback URL strictly matches `http://localhost:5000/auth/github/callback`.
