import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load config
with open("config/config.json", "r") as f:
    config = json.load(f)

# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(
    C=config["C"],
    solver=config["solver"],
    max_iter=config["max_iter"]
)
model.fit(X_train, y_train)

# Save model
with open("model_train.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model_train.pkl")