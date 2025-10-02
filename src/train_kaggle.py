import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import joblib
import os

df = pd.read_csv("../labeled_data.csv")  # adjust path if needed

df['label'] = df['class'].apply(lambda x: 0 if x in [0,1] else 1)

df['clean'] = df['tweet'].apply(clean_text)

hate_df = df[df['label'] == 0]
safe_df = df[df['label'] == 1]

print("Before balancing:")
print("Hate/Offensive samples:", len(hate_df))
print("Safe samples:", len(safe_df))

min_count = min(len(hate_df), len(safe_df))
hate_df_balanced = resample(hate_df, replace=False, n_samples=min_count, random_state=42)
safe_df_balanced = resample(safe_df, replace=False, n_samples=min_count, random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([hate_df_balanced, safe_df_balanced]).sample(frac=1, random_state=42)  # shuffle

print("After balancing:")
print(df_balanced['label'].value_counts())

X_text = df_balanced['clean'].values
y = df_balanced['label'].values

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = tfidf.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf.joblib"))

print("Model and TF-IDF vectorizer saved in 'models/' folder.")
