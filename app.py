import streamlit as st
import joblib
import numpy as np
from src.preprocess import clean_text

# Load model and TF-IDF vectorizer
model = joblib.load("models/model.joblib")
tfidf = joblib.load("models/tfidf.joblib")

# Streamlit UI
st.set_page_config(page_title="Hate Speech Detection", page_icon="⚠️", layout="centered")
st.title("Hate Speech Detection System ")
st.write("Type a sentence below to check if it contains hate speech or not.")

# Confidence threshold slider
threshold = st.slider("Confidence threshold for Hate Speech ⚠️", 0.5, 1.0, 0.7, 0.01)

# Input text
user_input = st.text_area("Enter text here:", "")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess input
        clean_input = clean_text(user_input)
        vect_input = tfidf.transform([clean_input])
        
        # Predict
        pred_prob = model.predict_proba(vect_input)[0]
        pred = np.argmax(pred_prob)
        conf = pred_prob[pred]

        # Display results with colors, emojis, and threshold
        col1, col2 = st.columns(2)
        col1.write("**Input Text:**")
        col1.write(user_input)

        col2.write("**Prediction:**")
        if pred == 0 and conf >= threshold:
            col2.error(f" Hate Speech! (Confidence: {conf:.2f})")
        else:
            col2.success(f"Safe / Not Hate Speech (Confidence: {conf:.2f})")
    

st.markdown("---")
st.markdown(" Adjust the confidence threshold slider to make the detection stricter or more lenient.")
