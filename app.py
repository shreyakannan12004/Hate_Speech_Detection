import streamlit as st
import joblib
import numpy as np
from source.preprocess import clean_text

# Load Kaggle multi-class model (pipeline)
model = joblib.load("models/model_kaggle.joblib")

# Streamlit UI settings
st.set_page_config(page_title="Hate Speech Detection", page_icon="⚠️", layout="centered")

st.title("Hate Speech Text Classifier")
st.write("This model classifies text into **Hate Speech**, **Offensive**, or **Neither**.")

# Label mapping
label_map = {
    0: "Hate Speech",
    1: "Offensive",
    2: "Neither"
}

# Colors for Streamlit output
color_map = {
    "Hate Speech": "⚠️ **Hate Speech**",
    "Offensive": "🚫 **Offensive**",
    "Neither": "✅ Safe / Neither"
}

# Input textbox
user_input = st.text_area("Enter your text below:", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean input
        clean_in = clean_text(user_input)

        # Predict with pipeline
        pred = model.predict([clean_in])[0]
        pred_prob = model.predict_proba([clean_in])[0]
        confidence = np.max(pred_prob)

        # ------------------------------------------------
        # SAFETY FIX FOR ONE-WORD INPUTS + PRONOUNS
        # ------------------------------------------------

        single_word = user_input.strip().lower()

        harmless_pronouns = {
            "you", "u", "ur", "your", "yours", "he", "she", "it",
            "they", "them", "we", "i", "me", "my", "mine", "ours"
        }

        harmless_words = {
            "kind", "hello", "hi", "nice", "good", "great", "love",
            "happy", "joy", "cool", "thanks", "welcome", "fine","you", "she", "I",
            
        }

        # Words that must ALWAYS stay offensive
        slur_words = {
            "idiot", "bitch", "fuck", "fuk", "moron", "loser", "stupid",
            "dumb", "nigger", "bastard", "shit", "asshole", "retard"
        }

        # One-word input rule
        if len(single_word.split()) == 1:
            # If word is NOT a slur → FORCE to "Neither"
            if single_word not in slur_words:
                pred = 2
                confidence = 0.99

        # Pronouns rule (safe when alone)
        if single_word in harmless_pronouns:
            pred = 2
            confidence = 0.99

        # Harmless words rule
        if single_word in harmless_words:
            pred = 2
            confidence = 0.99

        # ------------------------------------------------
        # FINAL OUTPUT
        # ------------------------------------------------

        result_label = label_map[pred]
        result_display = color_map[result_label]

        st.markdown("### 🔍 Prediction Result")
        st.write(f"{result_display} (Confidence: **{confidence:.2f}**)")

        # Show class probabilities
        st.markdown("---")
        st.subheader("Class Probabilities")
        st.write(f"**Hate Speech (0):** {pred_prob[0]:.2f}")
        st.write(f"**Offensive (1):** {pred_prob[1]:.2f}")
        st.write(f"**Neither (2):** {pred_prob[2]:.2f}")

st.markdown("---")

