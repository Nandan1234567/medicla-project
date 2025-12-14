/**
 * Health Care Recommender - Application Logic
 */

const API_BASE_URL = '/api';

// --- Theme Management ---
function initTheme() {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;

    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);

    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme);
    });
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    if (theme === 'dark') {
        btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
    } else {
        btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>';
    }
}

// --- Authentication ---
let isLoginMode = true;

function toggleAuthMode(e) {
    if (e) e.preventDefault();
    isLoginMode = !isLoginMode;

    const title = document.getElementById('authTitle');
    const subtitle = document.getElementById('authSubtitle');
    const nameGroup = document.getElementById('nameGroup');
    const extraFields = document.getElementById('extraFields');
    const submitBtn = document.getElementById('submitBtnText');
    const toggleAuth = document.getElementById('toggleAuth');

    if (isLoginMode) {
        title.textContent = 'Welcome Back';
        subtitle.textContent = 'Enter your details to access your dashboard.';
        nameGroup.style.display = 'none';
        extraFields.style.display = 'none';
        submitBtn.textContent = 'Sign In';
        toggleAuth.innerHTML = 'Don\'t have an account? <a href="#" onclick="toggleAuthMode(event)" style="color: var(--primary-color);">Sign Up</a>';
    } else {
        title.textContent = 'Create Account';
        subtitle.textContent = 'Join us to get personalized health insights.';
        nameGroup.style.display = 'block';
        extraFields.style.display = 'block';
        submitBtn.textContent = 'Sign Up';
        toggleAuth.innerHTML = 'Already have an account? <a href="#" onclick="toggleAuthMode(event)" style="color: var(--primary-color);">Sign In</a>';
    }
}

async function handleAuth(e) {
    e.preventDefault();
    const errorMsg = document.getElementById('errorMsg');
    const submitBtn = document.querySelector('button[type="submit"]'); // Get button element
    errorMsg.style.display = 'none';

    // Simple UI Loading State
    const originalBtnText = submitBtn.innerHTML;
    submitBtn.innerHTML = 'Loading...';
    submitBtn.disabled = true;

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    const endpoint = isLoginMode ? '/auth/login' : '/auth/register';

    const payload = { email, password };

    if (!isLoginMode) {
        payload.name = document.getElementById('name').value;
        payload.dateOfBirth = document.getElementById('dateOfBirth').value ? new Date(document.getElementById('dateOfBirth').value).toISOString() : new Date().toISOString();
        payload.gender = document.getElementById('gender').value;
        payload.phone = document.getElementById('phone').value;

        // Basic client-side validation for Signup
        if(!payload.name || !payload.phone) {
             errorMsg.textContent = 'Please fill in all required fields.';
             errorMsg.style.display = 'block';
             submitBtn.innerHTML = originalBtnText;
             submitBtn.disabled = false;
             return;
        }
    }

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));
            window.location.href = 'dashboard.html';
        } else {
            errorMsg.textContent = data.error || 'Authentication failed';
            errorMsg.style.display = 'block';
        }
    } catch (err) {
        console.error('Auth Error:', err);
        errorMsg.textContent = 'Something went wrong. Please check your connection.';
        errorMsg.style.display = 'block';
    } finally {
        submitBtn.innerHTML = originalBtnText;
        submitBtn.disabled = false;
    }
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = 'login.html';
}

// --- Dashboard & Prediction ---
async function handlePrediction(e) {
    e.preventDefault();

    const symptoms = document.getElementById('symptoms').value;
    const severity = document.getElementById('severity').value;
    const submitBtn = document.querySelector('#predictionForm button[type="submit"]');

    // UI Loading
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Analyzing Symptoms...';
    submitBtn.disabled = true;

    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/prediction/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ symptoms, severity })
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            alert('Analysis failed: ' + (data.error || 'Unknown error'));
        }
    } catch (err) {
        console.error('Prediction Error:', err);
        alert('Could not connect to analysis engine.');
    } finally {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
}

function displayResults(data) {
    const welcome = document.getElementById('welcomeMessage');
    const results = document.getElementById('resultsArea');

    if(welcome) welcome.style.display = 'none';
    if(results) results.style.display = 'block';

    document.getElementById('diseaseName').textContent = data.result.disease;
    document.getElementById('confidenceValue').textContent = Math.round(data.result.confidence);
    document.getElementById('confidenceBar').style.width = `${data.result.confidence}%`;

    // Urgency & Doctor Message
    const urgencyBadge = document.getElementById('urgencyBadge');
    urgencyBadge.textContent = data.result.urgencyLevel || 'Moderate';

    const doctorMsgEl = document.getElementById('doctorMessage');
    if(doctorMsgEl) doctorMsgEl.textContent = data.doctorConsultationMessage || '';

    // Emergency Handling
    const emergencyWarning = document.getElementById('emergencyWarning');
    const diseaseTitle = document.getElementById('diseaseName');

    if (data.isEmergency) {
        emergencyWarning.style.display = 'block';
        urgencyBadge.style.background = 'var(--error)';
        diseaseTitle.style.color = 'var(--error)'; // Make disease name red
        if(doctorMsgEl) doctorMsgEl.style.color = 'var(--error)';
        if(doctorMsgEl) doctorMsgEl.style.fontWeight = 'bold';
    } else {
        emergencyWarning.style.display = 'none';
        urgencyBadge.style.background = 'var(--primary-light)';
        diseaseTitle.style.color = 'var(--text-primary)';
        if(doctorMsgEl) doctorMsgEl.style.color = 'var(--text-secondary)';
        if(doctorMsgEl) doctorMsgEl.style.fontWeight = 'normal';
    }

    // Recommendations
    const list = document.getElementById('recommendationsList');
    list.innerHTML = '';

    if (data.recommendations && data.recommendations.length > 0) {
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.className = 'recommendation-item';

            let icon = '💡';
            if (rec.type === 'emergency') icon = '🚨';
            if (rec.type === 'medication') icon = '💊';
            if (rec.type === 'self-care') icon = '🍵';

            li.innerHTML = `
                <div class="rec-icon">${icon}</div>
                <div>
                    <strong style="text-transform: capitalize;">${rec.type}</strong>
                    <p style="margin:0; color: var(--text-secondary);">${rec.description}</p>
                </div>
            `;
            list.appendChild(li);
        });
    }

    // Helper to populate simple lists for Wellness Plan
    const populateList = (elementId, items) => {
        const el = document.getElementById(elementId);
        if(!el) return; // Guard clause

        el.innerHTML = '';
        if (items && items.length > 0) {
            items.forEach(item => {
                const li = document.createElement('li');
                li.style.marginBottom = '0.5rem';
                li.textContent = item;
                el.appendChild(li);
            });
        } else {
            el.innerHTML = '<li style="list-style: none; color: var(--text-secondary); font-style: italic;">No specific recommendations available.</li>';
        }
    };

    // Populate Wellness Plan Sections
    populateList('dietList', data.dietPlan);
    populateList('exerciseList', data.exercisePlan);
    populateList('precautionsList', data.precautions);
}

function resetDashboard() {
    document.getElementById('welcomeMessage').style.display = 'block';
    document.getElementById('resultsArea').style.display = 'none';
    document.getElementById('predictionForm').reset();
}

// --- Tab Switching ---
function switchTab(tabName) {
    // Hide all content
    document.getElementById('dashboardContent').style.display = 'none';
    document.getElementById('historyContent').style.display = 'none';
    document.getElementById('profileContent').style.display = 'none';

    // Remove active class from all tabs
    document.getElementById('dashboardTab').classList.remove('active');
    document.getElementById('historyTab').classList.remove('active');
    document.getElementById('profileTab').classList.remove('active');

    // Show selected content and activate tab
    if (tabName === 'dashboard') {
        document.getElementById('dashboardContent').style.display = 'block';
        document.getElementById('dashboardTab').classList.add('active');
    } else if (tabName === 'history') {
        document.getElementById('historyContent').style.display = 'block';
        document.getElementById('historyTab').classList.add('active');
        loadHistory();
    } else if (tabName === 'profile') {
        document.getElementById('profileContent').style.display = 'block';
        document.getElementById('profileTab').classList.add('active');
        loadProfile();
    }
}

// --- Load History ---
async function loadHistory() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">Loading history...</p>';

    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/prediction/history?limit=10`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        const data = await response.json();

        if (data.success && data.predictions && data.predictions.length > 0) {
            historyList.innerHTML = '';
            data.predictions.forEach(pred => {
                const card = document.createElement('div');
                card.className = 'history-card';

                const date = new Date(pred.createdAt).toLocaleString();
                const isEmergency = pred.isEmergency;
                const urgencyColor = isEmergency ? 'var(--error)' : 'var(--primary-color)';

                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                        <div>
                            <h4 style="margin: 0; color: ${urgencyColor};">${pred.result.disease}</h4>
                            <p style="margin: 0.25rem 0 0 0; font-size: 0.875rem; color: var(--text-secondary);">${date}</p>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">${Math.round(pred.result.confidence)}%</div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary);">Confidence</div>
                        </div>
                    </div>
                    <div style="margin-top: 1rem;">
                        <strong>Symptoms:</strong> ${pred.symptoms}
                    </div>
                    ${isEmergency ? '<div style="margin-top: 0.5rem; color: var(--error); font-weight: 600;">⚠️ Emergency</div>' : ''}
                `;

                historyList.appendChild(card);
            });
        } else {
            historyList.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No prediction history yet. Make a prediction to see it here!</p>';
        }
    } catch (err) {
        console.error('History Error:', err);
        historyList.innerHTML = '<p style="text-align: center; color: var(--error);">Failed to load history. Please try again.</p>';
    }
}

// --- Load Profile ---
async function loadProfile() {
    const profileInfo = document.getElementById('profileInfo');
    profileInfo.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">Loading profile...</p>';

    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`${API_BASE_URL}/user/profile`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        const data = await response.json();

        if (data.success && data.user) {
            const user = data.user;
            const registeredDate = new Date(user.createdAt).toLocaleDateString();
            const lastLogin = user.lastLogin ? new Date(user.lastLogin).toLocaleString() : 'Never';

            profileInfo.innerHTML = `
                <div class="glass-panel" style="padding: 1.5rem;">
                    <div class="profile-field">
                        <span class="profile-label">Name</span>
                        <span class="profile-value">${user.name}</span>
                    </div>
                    <div class="profile-field">
                        <span class="profile-label">Email</span>
                        <span class="profile-value">${user.email}</span>
                    </div>
                    <div class="profile-field">
                        <span class="profile-label">Phone</span>
                        <span class="profile-value">${user.phone || 'N/A'}</span>
                    </div>
                    <div class="profile-field">
                        <span class="profile-label">Gender</span>
                        <span class="profile-value">${user.gender || 'N/A'}</span>
                    </div>
                    <div class="profile-field">
                        <span class="profile-label">Member Since</span>
                        <span class="profile-value">${registeredDate}</span>
                    </div>
                    <div class="profile-field">
                        <span class="profile-label">Last Login</span>
                        <span class="profile-value">${lastLogin}</span>
                    </div>
                    <div class="profile-field">
                        <span class="profile-label">Account Status</span>
                        <span class="profile-value" style="color: var(--success);">✓ Active</span>
                    </div>
                </div>
            `;
        } else {
            profileInfo.innerHTML = '<p style="text-align: center; color: var(--error);">Failed to load profile.</p>';
        }
    } catch (err) {
        console.error('Profile Error:', err);
        profileInfo.innerHTML = '<p style="text-align: center; color: var(--error);">Failed to load profile. Please try again.</p>';
    }
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initTheme();

    // Auth Page
    const authForm = document.getElementById('authForm');
    if (authForm) {
        authForm.addEventListener('submit', handleAuth);
    }

    // Dashboard Page
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        // Set User Name
        const userStr = localStorage.getItem('user');
        if (userStr) {
            const user = JSON.parse(userStr);
            const nameEl = document.getElementById('userName');
            if(nameEl) nameEl.textContent = user.name || 'User';
        }

        predictionForm.addEventListener('submit', handlePrediction);
    }
});
