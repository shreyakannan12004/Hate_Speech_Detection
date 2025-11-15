import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from source.preprocess import clean_text

# Load dataset
df = pd.read_csv("labeled_data.csv")
df["clean"] = df["tweet"].apply(clean_text)

X = df["clean"]
y = df["class"]     # 0 = hate speech, 1 = offensive, 2 = neutral

# Use binary classification (hate vs safe)
y_binary = (df["hate_speech"] > 0).astype(int)

# TF–IDF
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    stop_words="english"
)
X_tfidf = tfidf.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=300, class_weight="balanced")
model.fit(X_tfidf, y_binary)

# Save files
joblib.dump(model, "models/model.joblib")
joblib.dump(tfidf, "models/tfidf.joblib")

print("Training complete! Model updated.")
