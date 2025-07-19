# MLOps Assignment 2: Digit Classification with Multi-Stage GitHub Actions

## 📁 Project Structure
- `src/train.py` – Trains Logistic Regression using digits dataset
- `src/inference.py` – Runs inference using saved model
- `src/utils.py` – Reusable training function
- `config/config.json` – Hyperparameters for model
- `tests/test_train.py` – Unit tests for training pipeline
- `.github/workflows/` – CI workflows for test, train, inference

## ⚙️ Workflow Summary

### 1. **Train Workflow**
- Trains model with config values
- Saves `model_train.pkl`
- Uploads model as artifact

### 2. **Test Workflow**
- Tests config loading
- Validates model type & accuracy

### 3. **Inference Workflow**
- Runs only if test + train jobs succeed
- Downloads model and performs prediction

## 🧪 Tools & Tech
- Python 3.10
- Scikit-learn
- Pytest
- GitHub Actions

## ✅ Requirements Covered
- [x] Digits dataset training
- [x] Hyperparam from config
- [x] Unit testing with `pytest`
- [x] Multi-job CI pipeline with `needs`
