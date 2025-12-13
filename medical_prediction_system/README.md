# Health Care Recommender System 🏥

A modern, AI-powered medical prediction web application that analyzes symptoms to provide preliminary disease diagnosis, severity assessment, and a holistic wellness plan (Diet, Exercise, Precautions).

## 🌟 Features

-   **AI-Powered Diagnosis**: Uses a Linear SVM model to predict diseases from natural language symptoms.
-   **Holistic Wellness Plans**: Generates personalized Diet, Exercise, and Precaution plans based on the diagnosis.
-   **User Authentication**:
    -   Secure Email/Password Login.
    -   Social Login (Google & GitHub - Simulated).
-   **Responsive "Leaf" Theme**: Beautiful glassmorphism UI with Dark Mode support (Deep Forest theme).
-   **Robust Backend**: Flask-based API with MongoDB integration and intelligent In-Memory Fallback (runs even without a database).

## 🚀 Getting Started

### Prerequisites

-   **Python 3.8+**
-   **MongoDB** (Optional - App runs in 'Mock Mode' if missing)

### Installation

1.  **Clone the repository**.
2.  **Navigate to the project directory**:
    ```bash
    cd medical_prediction_system
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Backend Server**:
    The backend serves both the API and the Frontend.
    ```bash
    cd backend_flask
    python app.py
    ```
    *You should see output indicating the server is running on `http://127.0.0.1:5000`.*

2.  **Access the App**:
    Open your web browser and visit:
    **[http://localhost:5000](http://localhost:5000)**

## 💡 How to Use

1.  **Sign Up / Login**: Create an account or use "Google/GitHub" buttons (Simulation Mode).
2.  **Dashboard**: Enter symptoms (e.g., "headache, fever").
3.  **Get Results**: Click "Running Analysis..." to see:
    -   Predicted Disease & Confidence.
    -   **Diet Plan**: Recommended foods.
    -   **Exercise Plan**: Safe activities.
    -   **Precautions**: Steps to take.

## 👨‍💻 Developer

Built with ❤️ for the Kero IDE Project.
