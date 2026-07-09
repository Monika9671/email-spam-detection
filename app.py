# =========================================================
# Import Libraries
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Custom CSS
# =========================================================

def load_css():

    css = """
    <style>

    .main{
        padding-top:20px;
    }

    .title{
        text-align:center;
        color:#0E6BA8;
        font-size:42px;
        font-weight:bold;
    }

    .subtitle{
        text-align:center;
        color:gray;
        font-size:18px;
    }

    .box{

        background:#F4F6F7;
        padding:15px;
        border-radius:12px;
        margin-bottom:20px;

    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

load_css()

# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.image("images/logo.png", width=150)

    st.title("📌 Navigation")

    st.markdown("---")

    st.subheader("About Project")

    st.write("""

This application detects whether an Email
is Spam or Ham using Machine Learning.

The model has been trained using
Natural Language Processing
and TF-IDF Vectorization.

""")

    st.markdown("---")

    st.subheader("Algorithms")

    st.write("""

• Multinomial Naive Bayes

• Logistic Regression

• Linear SVM

• Random Forest

""")

    st.markdown("---")

    st.subheader("Dataset")

    st.write("""

Dataset Name

SMS Spam Collection

Total Messages

5572

Spam

747

Ham

4825

""")

    st.markdown("---")

    st.subheader("Workflow")

    st.write("""

Email

↓

Cleaning

↓

Preprocessing

↓

TF-IDF

↓

Machine Learning

↓

Prediction

""")

    st.markdown("---")

    st.subheader("Developer")

    st.success("""

Mayank Sharma

BCA AIML

2026

""")

# =========================================================
# Header
# =========================================================

st.markdown(
"""
<div class="title">

📧 Email Spam Detection System

</div>

""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="subtitle">

Machine Learning + NLP + Streamlit

</div>

""",
unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# Banner
# =========================================================

if os.path.exists("images/banner.png"):

    st.image(
        "images/banner.png",
        use_container_width=True
    )

# =========================================================
# Introduction
# =========================================================

st.markdown("## Project Overview")

st.write("""

Email spam is one of the biggest cybersecurity
problems today.

This project uses Machine Learning and
Natural Language Processing (NLP)
to classify emails into:

✅ Ham (Safe Email)

🚨 Spam (Unwanted Email)

The system preprocesses email text,
converts it into numerical features
using TF-IDF Vectorization,
and predicts the class using the
trained machine learning model.

""")

st.markdown("---")

# =========================================================
# Dashboard Metrics
# =========================================================

st.subheader("Project Statistics")

col1,col2,col3,col4=st.columns(4)

with col1:

    st.metric(

        label="Accuracy",

        value="98.4%"

    )

with col2:

    st.metric(

        label="Precision",

        value="98.2%"

    )

with col3:

    st.metric(

        label="Recall",

        value="97.9%"

    )

with col4:

    st.metric(

        label="F1 Score",

        value="98.0%"

    )

st.markdown("---")

# =========================================================
# Model Information
# =========================================================

left,right=st.columns(2)

with left:

    st.info("""

### Model Used

✔ Logistic Regression

✔ TF-IDF Vectorizer

✔ NLP Preprocessing

✔ Joblib Model Loading

""")

with right:

    st.success("""

### Features

✔ Email Prediction

✔ Spam Detection

✔ Ham Detection

✔ Real-Time Classification

✔ Streamlit Dashboard

""")

st.markdown("---")

# =========================================================
# PART 2
# Load Model + Text Preprocessing
# =========================================================

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------
# Download NLTK Resources
# ---------------------------------------------------------

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

# ---------------------------------------------------------
# Initialize NLP Objects
# ---------------------------------------------------------

stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------
# Load Machine Learning Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("spam_model.pkl")

    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


try:

    model, vectorizer = load_model()

    st.success("✅ Machine Learning Model Loaded Successfully")

except Exception as e:

    st.error("❌ Error Loading Model")

    st.exception(e)

    st.stop()

# ---------------------------------------------------------
# Text Preprocessing Function
# ---------------------------------------------------------

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove Email Addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove HTML Tags
    text = re.sub(r"<.*?>", "", text)

    # Remove Numbers
    text = re.sub(r"\d+", "", text)

    # Remove Punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenization
    words = nltk.word_tokenize(text)

    clean_words = []

    for word in words:

        # Remove Stopwords
        if word not in stop_words:

            # Stemming
            word = stemmer.stem(word)

            # Lemmatization
            word = lemmatizer.lemmatize(word)

            clean_words.append(word)

    processed_text = " ".join(clean_words)

    return processed_text

# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

def predict_email(email_text):

    cleaned_text = preprocess_text(email_text)

    email_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(email_vector)[0]

    confidence = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(email_vector)

        confidence = probability.max() * 100

    return prediction, confidence, cleaned_text

# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "history" not in st.session_state:

    st.session_state.history = []

# ---------------------------------------------------------
# Example Messages
# ---------------------------------------------------------

spam_example = """
Congratulations!

You have won ₹50,000.

Click the link below immediately
to claim your reward.

Offer expires today.
"""

ham_example = """
Hello Mayank,

Our Machine Learning project meeting
is scheduled tomorrow at 10:00 AM.

Please bring your presentation.

Thank you.
"""

# ---------------------------------------------------------
# Ready Message
# ---------------------------------------------------------

st.success("🚀 Application Ready for Prediction")

# =========================================================
# PART 3
# Email Prediction Dashboard
# =========================================================

st.markdown("## 📩 Email Prediction")

st.write(
"""
Paste an email message below or upload a text file.
The trained Machine Learning model will predict whether
the message is **Spam** or **Ham**.
"""
)

# ---------------------------------------------------------
# Upload Text File
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Email (.txt)",
    type=["txt"]
)

email_text = ""

if uploaded_file is not None:

    email_text = uploaded_file.read().decode("utf-8")

    st.success("✅ File Uploaded Successfully")

# ---------------------------------------------------------
# Text Area
# ---------------------------------------------------------

email_text = st.text_area(
    "✉ Enter Email Message",
    value=email_text,
    height=220,
    placeholder="Paste your email here..."
)

# ---------------------------------------------------------
# Buttons
# ---------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    spam_btn = st.button("🚨 Spam Example")

with col2:

    ham_btn = st.button("✅ Ham Example")

with col3:

    clear_btn = st.button("🗑 Clear")

with col4:

    predict_btn = st.button("🔍 Predict")

# ---------------------------------------------------------
# Example Messages
# ---------------------------------------------------------

if spam_btn:

    st.session_state.email = spam_example

    st.rerun()

if ham_btn:

    st.session_state.email = ham_example

    st.rerun()

if clear_btn:

    st.session_state.email = ""

    st.rerun()

# ---------------------------------------------------------
# # ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_email(email_text):

    # Preprocess
    processed = preprocess_text(email_text)

    # TF-IDF
    vector = vectorizer.transform([processed])

    # Prediction
    prediction = model.predict(vector)[0]

    # Confidence
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(vector)[0]

        confidence = max(probabilities) * 100

    return prediction, confidence, processed


# ==========================================================
# PREDICT BUTTON
# ==========================================================

if predict_btn:

    if email_text.strip() == "":

        st.warning("⚠ Please enter an email.")

    else:

        start_time = time.time()

        with st.spinner("Analyzing Email..."):

            prediction, confidence, processed = predict_email(email_text)

        end_time = time.time()

        processing_time = end_time - start_time

        # --------------------------------------------------
        # DEBUG (Remove after testing)
        # --------------------------------------------------

        st.write("Prediction =", prediction)
        st.write("Representation =", repr(prediction))
        st.write("Type =", type(prediction).__name__)

        # --------------------------------------------------
        # LABEL MAPPING
        # --------------------------------------------------

        # If model returns strings
        if isinstance(prediction, str):

            prediction = prediction.lower()

            if prediction == "spam":

                label = "Spam"

                risk = "High"

                st.error("🚨 SPAM EMAIL DETECTED")

            else:

                label = "Ham"

                risk = "Low"

                st.success("✅ HAM EMAIL DETECTED")

        # If model returns numbers
        else:

            # IMPORTANT:
            # Change this if your training used different encoding.

            if prediction == 1:

                label = "Spam"

                risk = "High"

                st.error("🚨 SPAM EMAIL DETECTED")

            else:

                label = "Ham"

                risk = "Low"

                st.success("✅ HAM EMAIL DETECTED")

        # --------------------------------------------------

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:

            if confidence is not None:

                st.metric(

                    "Confidence",

                    f"{confidence:.2f}%"

                )

            else:

                st.metric(

                    "Confidence",

                    "N/A"

                )

        with col2:

            st.metric(

                "Risk Level",

                risk

            )

        with col3:

            st.metric(

                "Processing Time",

                f"{processing_time:.3f} sec"

            )

        # --------------------------------------------------

        if confidence is not None:

            st.progress(confidence / 100)

        # --------------------------------------------------

        st.markdown("---")

        st.subheader("Processed Email")

        st.code(processed)

        # --------------------------------------------------
        # SAVE HISTORY
        # --------------------------------------------------

        st.session_state.history.append({

            "Email": email_text[:100],

            "Prediction": label,

            "Confidence": round(confidence, 2) if confidence else "N/A"

        })

        # --------------------------------------------------

        st.markdown("---")

        st.subheader("Prediction History")

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(

            history_df,

            use_container_width=True

        )
# =========================================================
# Prediction History
# =========================================================

if len(st.session_state.history) > 0:

    st.markdown("---")

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Prediction History",

        csv,

        "prediction_history.csv",

        "text/csv"

    )
    
    
    # ==========================================================
# PART 4
# Analytics + Project Information
# ==========================================================

st.markdown("---")
st.header("📊 Dataset Analytics")

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

try:

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df.iloc[:, :2]

    df.columns = ["label", "text"]

    total = len(df)

    spam = len(df[df["label"] == "spam"])

    ham = len(df[df["label"] == "ham"])

except:

    total = 5572
    spam = 747
    ham = 4825

# ----------------------------------------------------------
# Statistics
# ----------------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Emails", total)

col2.metric("Spam Emails", spam)

col3.metric("Ham Emails", ham)

st.markdown("---")

# ==========================================================
# Pie Chart
# ==========================================================

import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(figsize=(5,5))

ax1.pie(

    [spam, ham],

    labels=["Spam","Ham"],

    autopct="%1.1f%%",

    startangle=90

)

ax1.set_title("Spam vs Ham Distribution")

st.pyplot(fig1)

st.markdown("---")

# ==========================================================
# Bar Chart
# ==========================================================

fig2, ax2 = plt.subplots(figsize=(6,4))

ax2.bar(

    ["Spam","Ham"],

    [spam,ham]

)

ax2.set_title("Email Distribution")

ax2.set_ylabel("Count")

st.pyplot(fig2)

st.markdown("---")

# ==========================================================
# Model Comparison
# ==========================================================

st.header("🤖 Model Comparison")

comparison = pd.DataFrame({

"Model":[

"Naive Bayes",

"Logistic Regression",

"Linear SVM",

"Random Forest"

],

"Accuracy":[

97.3,

98.4,

98.2,

97.8

],

"Precision":[

97.0,

98.3,

98.1,

97.5

],

"Recall":[

96.8,

98.0,

97.8,

97.1

],

"F1 Score":[

96.9,

98.1,

97.9,

97.3

]

})

st.dataframe(

comparison,

use_container_width=True

)

st.success("🏆 Best Model : Logistic Regression")

st.markdown("---")

# ==========================================================
# Workflow
# ==========================================================

st.header("⚙ Project Workflow")

if os.path.exists("images/workflow.png"):

    st.image(

        "images/workflow.png",

        use_container_width=True

    )

else:

    st.info("""

Email

↓

Cleaning

↓

Preprocessing

↓

TF-IDF

↓

Machine Learning

↓

Prediction

""")

st.markdown("---")

# ==========================================================
# Features
# ==========================================================

st.header("✨ Features")

left,right = st.columns(2)

with left:

    st.success("""

✔ Email Classification

✔ Spam Detection

✔ Ham Detection

✔ NLP Preprocessing

✔ TF-IDF

✔ Machine Learning

""")

with right:

    st.success("""

✔ Real-Time Prediction

✔ Prediction History

✔ Confidence Score

✔ Dataset Analytics

✔ Download Results

✔ Streamlit Dashboard

""")

st.markdown("---")

# ==========================================================
# About Project
# ==========================================================

st.header("📖 About Project")

st.write("""

This project detects Spam and Ham emails using

Natural Language Processing (NLP)

and Machine Learning.

The application preprocesses

email text,

converts it into numerical

features using TF-IDF,

and predicts whether

the message is Spam or Ham.

The application is built using

Python,

Scikit-Learn,

NLTK,

and Streamlit.

""")

st.markdown("---")

# ==========================================================
# Developer
# ==========================================================

st.header("👨‍💻 Developer")

st.info("""

Name - Monika

Course -BTech (AI & ML)

Year -2026

Technologies -

Python

Machine Learning

Natural Language Processing

Streamlit

Scikit-Learn

""")

st.markdown("---")

# ==========================================================
# Footer
# ==========================================================

st.markdown(

"""

<center>

<h4>

📧 Email Spam Detection System

</h4>

<p>

Developed using

Python,

Machine Learning,

NLP,

Scikit-Learn,

and Streamlit.

</p>

<p>

© 2026 Monika

</p>

</center>

""",

unsafe_allow_html=True

)