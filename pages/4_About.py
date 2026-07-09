
# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="About",

    page_icon="ℹ",

    layout="wide"

)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    # Logo
    if os.path.exists("images/logo.png"):

        st.image(
            "images/logo.png",
            width=170
        )

    st.title("ℹ About Project")

    st.markdown("---")

    st.subheader("Developer")

    st.write("""

👩 **Monika**

🎓 BTech (AI & ML)

""")

    st.markdown("---")

    st.subheader("Project")

    st.success("Email Spam Detection")

    st.markdown("---")

    st.subheader("Version")

    st.info("Version 1.0")

# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("📧 Email Spam Detection System")

st.caption(
    "Machine Learning Minor Project"
)

st.markdown("---")

# ==========================================================
# BANNER IMAGE
# ==========================================================

if os.path.exists("images/banner.png"):

    st.image(

        "images/banner.png",

        use_container_width=True

    )

st.markdown("---")

# ==========================================================
# PROJECT INTRODUCTION
# ==========================================================

st.header("📖 Project Introduction")

st.write("""

Email Spam Detection System is a Machine Learning
application that automatically classifies incoming
emails as **Spam** or **Ham (Not Spam)**.

The system applies **Natural Language Processing (NLP)**
techniques to clean and preprocess textual email data.
The cleaned text is transformed into numerical features
using **TF-IDF Vectorization**, allowing Machine Learning
algorithms to learn meaningful patterns from the data.

Multiple classification algorithms such as
Multinomial Naive Bayes, Logistic Regression,
Linear Support Vector Machine (SVM), and
Random Forest were trained and compared.

The best-performing model was selected using
evaluation metrics including Accuracy,
Precision, Recall, F1-Score,
Cross Validation, and Hyperparameter Tuning.

Finally, the trained model was deployed
using **Streamlit** to provide a user-friendly
web interface where users can classify emails
in real time.

""")

st.markdown("---")

# ==========================================================
# PROBLEM STATEMENT
# ==========================================================

st.header("🎯 Problem Statement")

st.warning("""

Email spam has become one of the major
cybersecurity challenges.

Every day millions of spam emails are sent
containing advertisements, phishing links,
malware, scams, and fraudulent content.

Manually identifying spam emails is
time-consuming and inefficient.

Therefore, an automated Machine Learning
system is required that can accurately
classify emails into Spam and Ham,
reducing human effort while improving
security and communication.

""")

st.markdown("---")

# ==========================================================
# PROJECT OBJECTIVES
# ==========================================================

st.header("🎯 Project Objectives")

col1,col2 = st.columns(2)

with col1:

    st.success("""

### Main Objectives

✔ Detect Spam Emails

✔ Detect Ham Emails

✔ Real-time Prediction

✔ High Accuracy

✔ Interactive Dashboard

""")

with col2:

    st.info("""

### Learning Objectives

✔ Machine Learning

✔ Natural Language Processing

✔ TF-IDF

✔ Streamlit Deployment

✔ Model Comparison

""")

st.markdown("---")

# ==========================================================
# PROJECT FEATURES
# ==========================================================

st.header("⭐ Key Features")

feature1,feature2,feature3 = st.columns(3)

with feature1:

    st.success("""

### Machine Learning

✔ Four ML Models

✔ Hyperparameter Tuning

✔ Cross Validation

✔ Best Model Selection

""")

with feature2:

    st.info("""

### NLP

✔ Tokenization

✔ Stopwords Removal

✔ Stemming

✔ Lemmatization

""")

with feature3:

    st.warning("""

### Web Application

✔ Streamlit

✔ Real-Time Prediction

✔ Dashboard

✔ Analytics

""")

st.markdown("---")

# ==========================================================
# PROJECT WORKFLOW
# ==========================================================

st.header("⚙ Project Workflow")

st.write("""

The Email Spam Detection System follows a complete
Machine Learning workflow starting from data collection
to deployment.

The workflow ensures accurate preprocessing,
feature extraction, model training, evaluation,
and real-time prediction.

""")

if os.path.exists("images/workflow.png"):

    st.image(

        "images/workflow.png",

        caption="Machine Learning Workflow",

        use_container_width=True

    )

else:

    st.warning("workflow.png not found inside images folder.")

st.markdown("---")

# ==========================================================
# MACHINE LEARNING PIPELINE
# ==========================================================

st.header("🧠 Machine Learning Pipeline")

pipeline = """

📂 Dataset Collection

        ↓

🧹 Data Cleaning

        ↓

📊 Exploratory Data Analysis

        ↓

📝 Text Preprocessing

        ↓

🔠 TF-IDF Feature Extraction

        ↓

✂ Train-Test Split

        ↓

🤖 Model Training

        ↓

⚙ Hyperparameter Tuning

        ↓

🔄 Cross Validation

        ↓

📈 Model Evaluation

        ↓

💾 Save Best Model

        ↓

🌐 Streamlit Deployment

"""

st.code(pipeline)

st.markdown("---")

# ==========================================================
# TECHNOLOGIES USED
# ==========================================================

st.header("🛠 Technologies Used")

col1, col2, col3 = st.columns(3)

with col1:

    st.success("""

### Programming

✔ Python

✔ Streamlit

✔ Joblib

✔ Git

✔ GitHub

""")

with col2:

    st.info("""

### Machine Learning

✔ Scikit-Learn

✔ Logistic Regression

✔ Linear SVM

✔ Naive Bayes

✔ Random Forest

""")

with col3:

    st.warning("""

### NLP

✔ TF-IDF

✔ Tokenization

✔ Stopwords

✔ Stemming

✔ Lemmatization

""")

st.markdown("---")

# ==========================================================
# DATASET INFORMATION
# ==========================================================

st.header("📂 Dataset Information")

left, right = st.columns(2)

with left:

    st.info("""

### Dataset Details

Dataset Name

SMS Spam Collection Dataset

Source

UCI Machine Learning Repository

Type

Text Classification Dataset

Language

English

""")

with right:

    st.success("""

### Dataset Statistics

Total Messages : 5572

Spam : 747

Ham : 4825

Features : 2

Target Classes : 2

""")

st.markdown("---")

# ==========================================================
# FOLDER STRUCTURE
# ==========================================================

st.header("📁 Project Folder Structure")

folder = """
Email-Spam-Detection/

│

├── app.py

├── requirements.txt

├── README.md

├── spam_model.pkl

├── tfidf_vectorizer.pkl

│

├── pages/

│     ├── 1_Dashboard.py

│     ├── 2_Analytics.py

│     ├── 3_Model_Comparison.py

│     └── 4_About.py

│

├── images/

│     ├── logo.png

│     ├── banner.png

│     └── workflow.png

│

├── dataset/

│     └── spam.csv

│

└── screenshots/

"""

st.code(folder)

st.markdown("---")

# ==========================================================
# CONCEPTS USED
# ==========================================================

st.header("📚 Concepts Used")

concept1, concept2 = st.columns(2)

with concept1:

    st.success("""

### Machine Learning

✔ Supervised Learning

✔ Classification

✔ Model Evaluation

✔ Cross Validation

✔ Hyperparameter Tuning

""")

with concept2:

    st.info("""

### Natural Language Processing

✔ Text Cleaning

✔ Tokenization

✔ Stopword Removal

✔ Stemming

✔ Lemmatization

✔ TF-IDF Vectorization

""")

st.markdown("---")

# ==========================================================
# PROJECT ARCHITECTURE
# ==========================================================

st.header("🏗 System Architecture")

architecture = """

User

↓

Streamlit Dashboard

↓

Text Preprocessing

↓

TF-IDF Vectorizer

↓

Machine Learning Model

↓

Prediction

↓

Spam / Ham

"""

st.code(architecture)

st.markdown("---")

# ==========================================================
# DEVELOPER INFORMATION
# ==========================================================

st.header("👩‍💻 Developer Information")

left, right = st.columns([1,2])

with left:

    if os.path.exists("images/logo.png"):

        st.image(
            "images/logo.png",
            width=180
        )

with right:

    st.success("""

### Monika

**Course :** BTech (Artificial Intelligence & Machine Learning)

**Project :** Email Spam Detection System

**Project Type :** Machine Learning Minor Project

**Programming Language :** Python

**Framework :** Streamlit

""")

st.markdown("---")

# ==========================================================
# PROJECT ACHIEVEMENTS
# ==========================================================

st.header("🏆 Project Achievements")

col1, col2 = st.columns(2)

with col1:

    st.success("""

### Machine Learning

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ NLP Preprocessing

✔ TF-IDF Feature Extraction

✔ Multiple ML Algorithms

✔ Hyperparameter Tuning

✔ Cross Validation

""")

with col2:

    st.info("""

### Deployment

✔ Streamlit Dashboard

✔ Model Comparison

✔ Interactive Analytics

✔ Real-Time Prediction

✔ GitHub Integration

✔ Professional Documentation

""")

st.markdown("---")

# ==========================================================
# FUTURE SCOPE
# ==========================================================

st.header("🚀 Future Scope")

future1, future2 = st.columns(2)

with future1:

    st.success("""

### Machine Learning

✔ Deep Learning

✔ BERT Model

✔ LSTM

✔ Transformer

✔ Ensemble Models

✔ Explainable AI

""")

with future2:

    st.info("""

### Deployment

✔ Gmail Integration

✔ Outlook Integration

✔ REST API

✔ Android Application

✔ Cloud Deployment

✔ Batch Email Prediction

""")

st.markdown("---")

# ==========================================================
# REFERENCES
# ==========================================================

st.header("📚 References")

st.markdown("""

### Official Documentation

• Streamlit Documentation

• Scikit-Learn Documentation

• Pandas Documentation

• NumPy Documentation

• NLTK Documentation

• Matplotlib Documentation

• Seaborn Documentation

### Dataset

• UCI SMS Spam Collection Dataset

### Learning Resources

• Kaggle

• GitHub

""")

st.markdown("---")

# ==========================================================
# ACKNOWLEDGEMENT
# ==========================================================

st.header("🙏 Acknowledgement")

st.write("""

I sincerely express my gratitude to my faculty members,
department, and institution for providing continuous guidance
and support throughout this Machine Learning project.

I would also like to thank the open-source community,
UCI Machine Learning Repository,
Scikit-Learn,
Streamlit,
and NLTK for providing valuable datasets,
libraries, and documentation that made this project possible.

""")

st.markdown("---")

# ==========================================================
# PROJECT HIGHLIGHTS
# ==========================================================

st.header("⭐ Project Highlights")

highlight1, highlight2, highlight3 = st.columns(3)

with highlight1:

    st.metric(
        "Algorithms",
        "4"
    )

    st.caption("Naive Bayes, Logistic Regression, SVM, Random Forest")

with highlight2:

    st.metric(
        "Accuracy",
        "98.4%"
    )

    st.caption("Best Performing Model")

with highlight3:

    st.metric(
        "Deployment",
        "Streamlit"
    )

    st.caption("Interactive Web Application")

st.markdown("---")

# ==========================================================
# PROJECT LINKS
# ==========================================================

st.header("🔗 Project Links")

github = st.text_input(
    "GitHub Repository",
    "https://github.com/yourusername/Email-Spam-Detection"
)

streamlit = st.text_input(
    "Live Streamlit App",
    "https://your-app.streamlit.app"
)

st.info(
"""
Replace the above placeholder links with your actual
GitHub repository and deployed Streamlit application.
"""
)

st.markdown("---")

# ==========================================================
# THANK YOU
# ==========================================================

st.success("""

🎉 Thank you for visiting the Email Spam Detection System.

This project demonstrates the complete Machine Learning lifecycle
from Data Collection to Deployment using
Natural Language Processing,
TF-IDF Vectorization,
Machine Learning,
and Streamlit.

""")
# ==========================================================
# SKILLS DEVELOPED
# ==========================================================

st.header("💡 Skills Developed During This Project")

skill1, skill2, skill3 = st.columns(3)

with skill1:

    st.success("""

### Programming

✔ Python

✔ Object-Oriented Programming

✔ Exception Handling

✔ File Handling

✔ Modular Coding

""")

with skill2:

    st.info("""

### Data Science

✔ Data Cleaning

✔ Data Visualization

✔ Feature Engineering

✔ Exploratory Data Analysis

✔ Statistical Analysis

""")

with skill3:

    st.warning("""

### Machine Learning

✔ Classification

✔ Model Evaluation

✔ Hyperparameter Tuning

✔ Cross Validation

✔ Model Deployment

""")

st.markdown("---")

# ==========================================================
# PROJECT STATISTICS
# ==========================================================

st.header("📊 Project Statistics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Dataset", "5572 Emails")
c2.metric("Algorithms", "4")
c3.metric("Best Accuracy", "98.4%")
c4.metric("Deployment", "Streamlit")

st.markdown("---")

# ==========================================================
# PROJECT TIMELINE
# ==========================================================

st.header("📅 Project Development Stages")

timeline = pd.DataFrame({

    "Stage":[

        "Requirement Analysis",

        "Dataset Collection",

        "Data Cleaning",

        "EDA",

        "Text Preprocessing",

        "TF-IDF Feature Extraction",

        "Model Training",

        "Hyperparameter Tuning",

        "Model Evaluation",

        "Model Saving",

        "Streamlit Development",

        "GitHub Deployment"

    ],

    "Status":[

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed"

    ]

})

st.dataframe(

    timeline,

    use_container_width=True

)

st.markdown("---")

# ==========================================================
# CERTIFICATIONS
# ==========================================================

st.header("🏅 Knowledge Applied")

left, right = st.columns(2)

with left:

    st.success("""

### Machine Learning

✔ Supervised Learning

✔ Classification

✔ Evaluation Metrics

✔ Feature Extraction

✔ Model Selection

""")

with right:

    st.info("""

### NLP

✔ Tokenization

✔ Stopwords Removal

✔ Stemming

✔ Lemmatization

✔ TF-IDF

""")

st.markdown("---")

# ==========================================================
# LEARNING OUTCOMES
# ==========================================================

st.header("📚 Learning Outcomes")

st.write("""

During the development of this project,
the following concepts were learned
and successfully implemented:

• Data Collection

• Data Cleaning

• Exploratory Data Analysis

• Text Preprocessing

• Feature Engineering

• Machine Learning Algorithms

• Hyperparameter Tuning

• Cross Validation

• Model Evaluation

• Streamlit Deployment

• GitHub Version Control

• Project Documentation

""")

st.markdown("---")

# ==========================================================
# CONTACT INFORMATION
# ==========================================================

st.header("📩 Contact")

contact1, contact2 = st.columns(2)

with contact1:

    st.write("""

### Developer

**Name**

Monika

**Course**

BTech (AI & ML)

""")

with contact2:

    st.write("""

### Project

Email Spam Detection System

Machine Learning Minor Project

Python + Streamlit

""")

st.markdown("---")

# ==========================================================
# FINAL CONCLUSION
# ==========================================================

st.header("🎯 Conclusion")

st.success("""

The Email Spam Detection System
successfully classifies email messages
into Spam and Ham using
Machine Learning techniques.

The project demonstrates the complete
Machine Learning lifecycle including

✔ Data Collection

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Natural Language Processing

✔ Feature Engineering

✔ Model Training

✔ Hyperparameter Tuning

✔ Cross Validation

✔ Model Evaluation

✔ Model Deployment

This project provides an effective
solution for detecting spam emails
with high accuracy while demonstrating
practical implementation of
Machine Learning concepts.

""")

st.markdown("---")

# ==========================================================
# THANK YOU
# ==========================================================

st.balloons()

st.success("""

🎉 Thank You!

Thank you for visiting the
Email Spam Detection System.

We hope this project demonstrates
the practical implementation of
Machine Learning and Natural Language
Processing for solving real-world
classification problems.

""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""

---

<div style="text-align:center">

<h2>📧 Email Spam Detection System</h2>

<h4>Machine Learning Minor Project</h4>

<p>

Developed using

<strong>

Python • Streamlit • Scikit-Learn • Pandas • NumPy • NLTK

</strong>

</p>

<p>

👩‍💻 Developer : <b>Monika</b>

</p>

<p>

🎓 Course : <b>BTech (Artificial Intelligence & Machine Learning)</b>

</p>

<p>

© 2026 All Rights Reserved

</p>

</div>

""", unsafe_allow_html=True)