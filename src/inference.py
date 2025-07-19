import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

# Load model
model = joblib.load("model_train.pkl")

# Load digits data
digits = load_digits()
X, y = digits.data, digits.target

# Predict
y_pred = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy on full digits dataset: {accuracy:.4f}")