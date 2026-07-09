import streamlit as st
import os

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    if os.path.exists("images/logo.png"):
        st.image("images/logo.png", width=180)

    st.title("📌 Navigation")

    st.markdown("---")

    st.subheader("👩‍💻 Developer")

    st.write("""
**Name :** Monika

**Course :**
BTech (AI & ML)

**Project :**
Email Spam Detection
""")

    st.markdown("---")

    st.subheader("📖 About Project")

    st.write("""
This application predicts whether an
Email is **Spam** or **Ham** using
Machine Learning and Natural Language
Processing.

The model has been trained using
TF-IDF Vectorization and
Logistic Regression.
""")

    st.markdown("---")

    st.subheader("🤖 Model Used")

    st.success("✔ Logistic Regression")
    st.success("✔ TF-IDF Vectorizer")
    st.success("✔ NLP")
    st.success("✔ Joblib Model")

# ---------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------

st.title("📧 Email Spam Detection Dashboard")

st.markdown("""
### Machine Learning Based Email Classification System
""")

st.write("""
This dashboard allows users to classify
emails into **Spam** or **Ham** using
Natural Language Processing and
Machine Learning techniques.
""")

st.markdown("---")

# ---------------------------------------------------
# PERFORMANCE METRICS
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    "98.4%"
)

col2.metric(
    "Precision",
    "98.2%"
)

col3.metric(
    "Recall",
    "97.9%"
)

col4.metric(
    "F1 Score",
    "98.0%"
)

st.markdown("---")

# ---------------------------------------------------
# INFORMATION CARDS
# ---------------------------------------------------

left, right = st.columns(2)

with left:

    st.info("""

## 🤖 Model Used

✔ Logistic Regression

✔ TF-IDF Vectorizer

✔ Natural Language Processing

✔ Hyperparameter Tuning

✔ Cross Validation

✔ Joblib Model Loading

""")

with right:

    st.success("""

## ✨ Features

✔ Email Prediction

✔ Upload Email File

✔ Confidence Score

✔ Prediction History

✔ Real-Time Classification

✔ Interactive Dashboard

""")

st.markdown("---")

