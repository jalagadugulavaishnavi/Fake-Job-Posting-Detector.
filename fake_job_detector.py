import pandas as pd
import os
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Streamlit page settings
st.set_page_config(page_title="Fake Job Posting Detector", layout="centered")
st.title("🕵️‍♂️ Fake Job Posting Detector")
st.markdown("Paste a job description, and I’ll tell you if it's real or fake!")

# File paths
model_path = "job_model.pkl"
vectorizer_path = "vectorizer.pkl"
dataset_path = "fake_job_postings.csv"

# Step 1: Load & train model if not already done
@st.cache_resource(show_spinner=True)
def load_or_train_model():
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.info("🔄 Training model for the first time...")

        # Load dataset
        df = pd.read_csv(dataset_path)
        df = df[df['description'].notna()]
        X = df['description']
        y = df['fraudulent']

        # Text vectorization
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        X_vec = vectorizer.fit_transform(X)

        # Train-test split and model training
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Save model and vectorizer
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.success(f"✅ Model trained! Accuracy: {accuracy:.2%}")
    else:
        # Load existing model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

    return model, vectorizer

# Check if dataset is available
if not os.path.exists(dataset_path):
    st.error("""
        ❌ Dataset not found!
        Please download it from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
        and place it in the same folder with the filename: `fake_job_postings.csv`.
    """)
else:
    model, vectorizer = load_or_train_model()

    # Step 2: Input from user
    job_desc = st.text_area("📄 Paste the job description here:")

    if st.button("🧠 Detect"):
        if job_desc.strip() == "":
            st.warning("⚠️ Please enter a job description first.")
        else:
            X_input = vectorizer.transform([job_desc])
            prediction = model.predict(X_input)[0]
            if prediction == 1:
                st.error("🚨 This job posting appears to be **FAKE**.")
            else:
                st.success("🎉 This job posting appears to be **GENUINE**.")
