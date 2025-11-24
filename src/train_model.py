import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1️⃣ Load Dataset
df = pd.read_csv("data/cleaned_malicious_phish.csv")

# 2️⃣ Map labels to numeric values
df['label'] = df['label'].map({
    'benign': 0,
    'phishing': 1,
    'defacement': 1,
    'malware': 1
})

print("Label distribution:")
print(df['label'].value_counts())

# 3️⃣ Split into features (X) and target (y)
X = df['url']
y = df['label']

# 4️⃣ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Convert text URLs into numeric vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 5))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6️⃣ Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 7️⃣ Predict on test data
y_pred = model.predict(X_test_vec)

# 8️⃣ Evaluate the model
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9️⃣ Save the trained model and vectorizer
joblib.dump(model, "models/url_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully in 'models/' directory!")