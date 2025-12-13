/**
 * Medical Healthcare Recommendations - Frontend Logic
 * Connects to Python Flask backend for disease prediction
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    API_URL: 'http://localhost:5000/predict', // Backend API endpoint
    THEME_KEY: 'medical-app-theme',
    ANIMATION_DELAY: 300
};

// ============================================
// Theme Management
// ============================================
class ThemeManager {
    constructor() {
        this.themeToggle = document.getElementById('themeToggle');
        this.currentTheme = localStorage.getItem(CONFIG.THEME_KEY) || 'light';
        this.init();
    }

    init() {
        this.setTheme(this.currentTheme);
        this.themeToggle.addEventListener('click', () => this.toggleTheme());
    }

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.currentTheme = theme;
        
        const icon = this.themeToggle.querySelector('i');
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        
        localStorage.setItem(CONFIG.THEME_KEY, theme);
    }

    toggleTheme() {
        const newTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }
}

// ============================================
// Medical Recommendations Database
// ============================================
const MEDICAL_DATABASE = {
    recommendations: {
        'Viral Fever': [
            'Rest and stay hydrated with plenty of fluids',
            'Take fever-reducing medication (paracetamol)',
            'Monitor temperature regularly',
            'Consult doctor if fever persists > 3 days',
            'Avoid going to public places'
        ],
        'Headache': [
            'Rest in a quiet, dark room',
            'Stay well hydrated',
            'Apply cold or warm compress',
            'Take OTC pain relievers if needed',
            'Avoid screens and bright lights'
        ],
        'Abdominal Pain': [
            'Avoid solid foods initially',
            'Stay hydrated with clear liquids',
            'Avoid spicy and fatty foods',
            'Apply warm compress to abdomen',
            'Seek medical help if severe or persistent'
        ],
        'default': [
            'Monitor symptoms closely',
            'Rest and maintain good hydration',
            'Maintain a balanced diet',
            'Avoid self-medication',
            'Consult healthcare provider if symptoms worsen'
        ]
    },

    precautions: {
        'Viral Fever': [
            'Isolate yourself to prevent spread',
            'Wear mask when around others',
            'Wash hands frequently',
            'Disinfect frequently touched surfaces',
            'Get adequate sleep (8+ hours)'
        ],
        'Headache': [
            'Maintain regular sleep schedule',
            'Manage stress through relaxation',
            'Limit caffeine and alcohol',
            'Stay hydrated throughout day',
            'Practice good posture'
        ],
        'Abdominal Pain': [
            'Eat smaller, frequent meals',
            'Avoid trigger foods',
            'Reduce stress levels',
            'Exercise regularly but gently',
            'Stay upright after eating'
        ],
        'default': [
            'Maintain good personal hygiene',
            'Get regular exercise',
            'Eat a balanced, nutritious diet',
            'Get 7-8 hours of sleep',
            'Manage stress effectively',
            'Stay hydrated throughout the day'
        ]
    }
};

// ============================================
// Prediction Service
// ============================================
class PredictionService {
    async predict(formData) {
        try {
            const response = await fetch(CONFIG.API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            // Fallback to mock data if backend is not available
            console.warn('Backend not available, using mock data:', error);
            return this.getMockPrediction(formData);
        }
    }

    getMockPrediction(formData) {
        // Simulated prediction for offline testing
        const mockDiseases = [
            'Viral Fever', 'Headache', 'Abdominal Pain', 'Allergic Reaction',
            'Respiratory Distress', 'Heart Disease', 'Back Pain'
        ];
        
        const randomDisease = mockDiseases[Math.floor(Math.random() * mockDiseases.length)];
        const confidence = (75 + Math.random() * 20).toFixed(2);

        return {
            disease: randomDisease,
            confidence: parseFloat(confidence),
            symptoms: formData.symptoms,
            metadata: {
                age: formData.age,
                gender: formData.gender,
                duration: formData.duration,
                severity: formData.severity
            }
        };
    }

    getRecommendations(disease) {
        return MEDICAL_DATABASE.recommendations[disease] || 
               MEDICAL_DATABASE.recommendations.default;
    }

    getPrecautions(disease) {
        return MEDICAL_DATABASE.precautions[disease] || 
               MEDICAL_DATABASE.precautions.default;
    }
}

// ============================================
// UI Manager
// ============================================
class UIManager {
    constructor() {
        this.form = document.getElementById('healthForm');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        this.inputSection = document.querySelector('.input-section');
    }

    showLoading() {
        this.hideAllSections();
        this.loadingSection.style.display = 'block';
        this.loadingSection.classList.add('fade-in');
    }

    showResults(data) {
        this.hideAllSections();
        this.populateResults(data);
        this.resultsSection.style.display = 'block';
        this.resultsSection.classList.add('fade-in');
        
        // Smooth scroll to results
        setTimeout(() => {
            this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    showError(message) {
        this.hideAllSections();
        document.getElementById('errorMessage').textContent = message;
        this.errorSection.style.display = 'block';
        this.errorSection.classList.add('fade-in');
    }

    hideAllSections() {
        this.loadingSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';
    }

    resetForm() {
        this.form.reset();
        this.hideAllSections();
        this.inputSection.scrollIntoView({ behavior: 'smooth' });
    }

    populateResults(data) {
        // Disease Name
        document.getElementById('diseaseName').textContent = data.disease;

        // Confidence Bar
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceValue = document.getElementById('confidenceValue');
        const confidence = data.confidence;

        confidenceBar.style.width = '0%';
        setTimeout(() => {
            confidenceBar.style.width = `${confidence}%`;
        }, 100);
        
        confidenceValue.textContent = `${confidence}%`;

        // Get recommendations and precautions
        const recommendations = new PredictionService().getRecommendations(data.disease);
        const precautions = new PredictionService().getPrecautions(data.disease);

        // Populate Recommendations
        const recommendationsList = document.getElementById('recommendationsList');
        recommendationsList.innerHTML = '<ul class="recommendations-list">' +
            recommendations.map(rec => 
                `<li><i class="fas fa-check-circle"></i><span>${rec}</span></li>`
            ).join('') +
            '</ul>';

        // Populate Precautions
        const precautionsList = document.getElementById('precautionsList');
        precautionsList.innerHTML = '<ul class="precautions-list">' +
            precautions.map(prec => 
                `<li><i class="fas fa-shield-alt"></i><span>${prec}</span></li>`
            ).join('') +
            '</ul>';
    }
}

// ============================================
// Main Application Controller
// ============================================
class MedicalApp {
    constructor() {
        this.themeManager = new ThemeManager();
        this.predictionService = new PredictionService();
        this.uiManager = new UIManager();
        this.init();
    }

    init() {
        // Form submission
        document.getElementById('healthForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });

        // New prediction button
        document.getElementById('newPredictionBtn').addEventListener('click', () => {
            this.uiManager.resetForm();
        });

        // Retry button
        document.getElementById('retryBtn').addEventListener('click', () => {
            this.uiManager.resetForm();
        });

        // Print button
        document.getElementById('printBtn').addEventListener('click', () => {
            window.print();
        });
    }

    async handlePrediction() {
        // Collect form data
        const formData = {
            symptoms: document.getElementById('symptoms').value,
            age: document.getElementById('age').value,
            gender: document.getElementById('gender').value,
            duration: document.getElementById('duration').value,
            severity: document.getElementById('severity').value
        };

        // Validate
        if (!this.validateForm(formData)) {
            return;
        }

        // Show loading
        this.uiManager.showLoading();

        try {
            // Make prediction
            const result = await this.predictionService.predict(formData);

            // Simulate processing time for better UX
            await this.delay(1500);

            // Show results
            this.uiManager.showResults(result);
        } catch (error) {
            console.error('Prediction error:', error);
            this.uiManager.showError(
                'Unable to process your request. Please check if the backend server is running on http://localhost:5000'
            );
        }
    }

    validateForm(formData) {
        if (!formData.symptoms.trim()) {
            alert('Please describe your symptoms');
            return false;
        }
        if (!formData.age) {
            alert('Please select your age group');
            return false;
        }
        if (!formData.gender) {
            alert('Please select your gender');
            return false;
        }
        if (!formData.duration) {
            alert('Please select symptom duration');
            return false;
        }
        if (!formData.severity) {
            alert('Please rate the severity');
            return false;
        }
        return true;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// ============================================
// Initialize Application
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    new MedicalApp();
    console.log('Medical Healthcare Recommendations System Initialized');
    console.log('Backend URL:', CONFIG.API_URL);
    console.log('Note: If backend is not running, mock data will be used for demo purposes');
});
