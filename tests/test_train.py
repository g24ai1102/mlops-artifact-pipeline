import json
import pytest
import os
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from src.utils import train_model

CONFIG_PATH = "config/config.json"

# ---- (a) CONFIGURATION FILE LOADING ----
def test_config_file_exists():
    assert os.path.exists(CONFIG_PATH), "Config file does not exist."

def test_config_keys_and_types():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    assert "C" in config and isinstance(config["C"], float)
    assert "solver" in config and isinstance(config["solver"], str)
    assert "max_iter" in config and isinstance(config["max_iter"], int)

# ---- (b) MODEL CREATION ----
def test_model_training_returns_logistic_regression():
    digits = load_digits()
    X, y = digits.data, digits.target
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert hasattr(model, "classes_")

# ---- (c) MODEL ACCURACY ----
def test_model_training_accuracy_above_threshold():
    digits = load_digits()
    X, y = digits.data, digits.target
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    model = train_model(X, y, config)
    accuracy = model.score(X, y)
    assert accuracy > 0.8, f"Accuracy too low: {accuracy}"
