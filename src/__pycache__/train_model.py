# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
data_path = os.path.join("data", "urls_train.csv")
data = pd.read_csv(data_path)

# Basic preprocessing (convert to lowercase)
data['url'] = data['url'].astype(str).str.lower()

# Features and labels
X = data['url']
y = data['label']

# Vectorization
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
X_vect = vectorizer.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print("✅ Model Training Completed")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Model and Vectorizer saved successfully!")