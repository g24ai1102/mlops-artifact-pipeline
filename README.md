# 📦 MLOps Assignment 2: Digit Classification Pipeline

This project implements an end-to-end machine learning pipeline for digit classification using GitHub Actions to automate testing, training, and inference workflows.

---

## 🧠 Problem Statement

Classify handwritten digits using Scikit-learn’s built-in `load_digits` dataset and automate the ML lifecycle using multi-stage CI pipelines.

---

## 🗂️ Project Structure

mlops-artifact-pipeline/
├── config/
│ └── config.json # Hyperparameters for training
├── src/
│ ├── train.py # Model training script
│ ├── inference.py # Model inference script
│ └── utils.py # Reusable training function
├── tests/
│ └── test_train.py # Unit tests for training pipeline
├── requirements.txt # Python dependencies
├── .github/
│ └── workflows/
│ ├── test.yml # Unit test workflow
│ ├── train.yml # Model training workflow
│ └── inference.yml # Multi-job pipeline (test → train → infer)



---

## ⚙️ Configuration

File: `config/config.json`

```json
**{
  "C": 1.0,
  "solver": "lbfgs",
  "max_iter": 1000
}

**
✅ Pipeline Overview
🔍 1. Test Workflow (test.yml)
Runs pytest on unit tests.

Validates:

Config loading

Model object returned

Model accuracy above threshold

🏋️ 2. Train Workflow (train.yml)
Loads hyperparameters from config

Trains a Logistic Regression model

Saves model as model_train.pkl

Uploads it as GitHub Actions artifact

🔮 3. Inference Workflow (inference.yml)
Multi-job workflow:

✅ Step 1: Run unit tests

✅ Step 2: Train model (only if tests pass)

✅ Step 3: Download model artifact → run inference

Prints predictions on sample digits

🧪 Libraries Used
Python 3.10

scikit-learn

joblib

pytest

GitHub Actions (CI/CD)

📌 Instructions to Run Locally

# Create virtual environment
conda create -n artifactmlops python=3.10
conda activate artifactmlops

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py

# Run inference
python src/inference.py

# Run tests
pytest tests/test_train.py

🙌 Authors & Acknowledgements
Assignment developed as part of MLOps coursework.
Thanks to the course instructors and GitHub Actions documentation!****
