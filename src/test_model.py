import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# 1️⃣ Load the trained model and vectorizer
model = joblib.load("models/url_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("✅ Model and vectorizer loaded successfully!\n")

# 2️⃣ Define a function to predict if a URL is malicious
def predict_url(url):
    # Convert to list for vectorizer
    url_vec = vectorizer.transform([url])
    prediction = model.predict(url_vec)[0]

    if prediction == 1:
        print(f"🚨 URL: {url}")
        print("Prediction: Malicious ⚠")
    else:
        print(f"✅ URL: {url}")
        print("Prediction: Safe ✔")

# 3️⃣ Example URLs for testing
test_urls = [
    "https://www.google.com",
    "http://malicious-login.ru/phishing",
    "https://secure-update.com/bank-verification",
    "http://facebook.com/login-reset-password.xyz"
]

for url in test_urls:
    predict_url(url)