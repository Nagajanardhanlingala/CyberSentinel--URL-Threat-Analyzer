# predict.py
import joblib
import os

# Load saved model and vectorizer
model_path = os.path.join("models", "model.pkl")
vectorizer_path = os.path.join("models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_url(url):
    """Predict whether a given URL is Malicious or Safe"""
    url = url.lower()
    features = vectorizer.transform([url])
    prediction = model.predict(features)[0]
    return "Malicious" if prediction == 1 else "Safe"

# Example usage
# Example usage
if __name__ == "__main__":
    test_url = input("Enter a URL to test: ")
    print("Prediction:", predict_url(test_url))