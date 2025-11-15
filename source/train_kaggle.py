import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import clean_text

# --------------------------
# PATHS
# --------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "labeled_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Kaggle dataset column = "tweet"
df['clean'] = df['tweet'].astype(str).apply(clean_text)

X_text = df['clean'].values
y = df['class'].values     # 0 = hate, 1 = offensive, 2 = neither

print("Building TF-IDF features...")

# WORD TF-IDF
word_tfidf = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2
)

# CHARACTER TF-IDF (captures "idiot", "bitch", "f***", etc.)
char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=50000,
    min_df=2
)

# Combine both
features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf)
])

# Build full pipeline
pipeline = Pipeline([
    ("features", features),
    ("model", LogisticRegression(
        max_iter=4000,
        class_weight={0: 4.0, 1: 1.0, 2: 1.5}   # Strong fix for class imbalance
    ))
])

# Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model (please wait)...")
pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Saving model
MODEL_PATH = os.path.join(MODEL_DIR, "model_kaggle.joblib")
joblib.dump(pipeline, MODEL_PATH)

print(f"\nTraining complete! Model saved at: {MODEL_PATH}")
