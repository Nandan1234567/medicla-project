# Medical Prediction System

This repository contains a small medical prediction system built with Python. It includes data preparation, a model trainer, a model file, a Flask backend API, and a simple frontend for interacting with the model.

This README explains how to set up the project after cloning, install dependencies, run the backend API, and try the frontend example. Instructions include PowerShell commands for Windows and generic commands for other platforms.

## Repository layout

- `backend_flask/` - Flask-based API to serve predictions (`app.py`).
- `models/` - Model code and predictor (`linear_svm_predictor.py`).
- `training/` - Training code to build models (`linear_svm_trainer.py`).
- `data/` - CSV datasets used by the project.
- `frontend/` - Static HTML/JS/CSS demo interface.
- `utils/` - Utility scripts (data validation, etc.).
- `examples/` - Example scripts for batch and API calls.
- `requirements.txt` - Python dependencies for the whole project.
- `backend_flask/requirements.txt` - Dependencies specific to the Flask backend.

## Prerequisites

- Python 3.8+ (3.10+ recommended). On Windows, use the official installer from python.org and enable "Add Python to PATH".
- pip (bundled with Python) or a virtual environment tool (venv, virtualenv, pipenv).

Optional but recommended:
- Git (to clone the repository)
- A virtual environment for isolating dependencies

## Quick setup (Windows PowerShell)

Open PowerShell and run the following commands from the project root (where the repository was cloned).

```powershell
# create and activate a venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# upgrade pip
python -m pip install --upgrade pip

# install core dependencies (project-wide)
pip install -r requirements.txt

# install backend-specific dependencies (if you only want to run the API)
pip install -r backend_flask\requirements.txt
```

If PowerShell execution policies block script activation, run PowerShell as Administrator and set a permissive policy (you can revert it later):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Quick setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r backend_flask/requirements.txt
```

## Run the Flask API (development)

From the project root (PowerShell):

```powershell
# activate venv if not active
.\.venv\Scripts\Activate.ps1

# run the backend Flask app
python backend_flask\app.py
```

By default the Flask app binds to the address and port configured in `backend_flask/app.py`. Open a browser and visit http://127.0.0.1:5000/ (or the printed URL) to access endpoints.

## Run the simple frontend

The `frontend/` folder contains static HTML and JS to interact with the API. For local testing you can open `frontend/index.html` or `frontend/dashboard.html` in your browser. If your browser blocks requests from file://, serve the folder with a tiny HTTP server:

PowerShell:

```powershell
# From project root
python -m http.server 8000
# Then open http://127.0.0.1:8000/frontend/index.html
```

Or using Node (if installed):

```powershell
npm install -g serve; serve -s frontend
```

## Examples

- `examples/web_api.py` shows how to call the Flask API programmatically.
- `examples/batch_prediction.py` demonstrates using the predictor for batch CSV predictions.

## Training a model

If you want to retrain the model from `data/cleaned_dataset.csv`:

```powershell
# activate venv
.\.venv\Scripts\Activate.ps1

# run trainer
python training\linear_svm_trainer.py
```

Trainer output and where the model is saved depend on code in `training/linear_svm_trainer.py` and `models/linear_svm_predictor.py`.

## Tests

There are no formal unit tests included. To sanity check the API run `test_api.py` from the project root after starting the backend.

```powershell
# start backend in a separate terminal, then
python test_api.py
```

## Troubleshooting

- If imports fail, ensure you have the virtual environment activated and `pip install -r requirements.txt` completed.
- If you get a permission error when activating the venv in PowerShell, run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as Administrator.
- If the Flask server doesn't start, check `backend_flask/app.py` for host/port configuration and any missing environment variables.

## Contributing

Contributions welcome. Open issues or pull requests. Keep changes small and focused. Add tests when adding features.

## License

This project doesn't include a license file. Add a `LICENSE` file if you plan to publish the code or choose an open-source license.

## Contact

If something in this README is unclear, open an issue describing the problem and the environment you're running on (OS, Python version).

---

Generated: README for local developer setup and quickstart.
