# 🕵️‍♂️ Fake Job Posting Detector

A machine learning-based Streamlit web app to detect fraudulent job postings from job descriptions. This project uses a dataset of real and fake job posts to train a logistic regression model and help users avoid scams online.

---

## 📌 Features

- 🔍 Predicts whether a job description is **genuine or fake**
- 🧠 Trained using TF-IDF vectorization and Logistic Regression
- 💾 Caches model to avoid retraining every time
- 📊 Built with Streamlit for an interactive UI
- ✅ Easy to use — just paste a job description and click **Detect**

---

## 📂 Dataset

- Dataset: [Fake Job Postings - Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- File used: `fake_job_postings.csv`
- Column used: `description` (text), `fraudulent` (label: 1 = fake, 0 = real)

---
