import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

from preprocess import clean_text

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "hate_speech.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Clean text
df['clean'] = df['text'].apply(clean_text)

# Features and labels
X_text = df['clean'].values
y = df['label'].values

# TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)

X = tfidf.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and TF-IDF vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))
print("Model and TF-IDF vectorizer saved in 'models/' folder.")
