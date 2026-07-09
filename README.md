# Email Spam Detection System

A Machine Learning and Natural Language Processing (NLP) based application that classifies email messages as **Spam** or **Ham (Not Spam)**. The project uses text preprocessing, TF-IDF feature extraction, and multiple machine learning algorithms to achieve high prediction accuracy. The final model is deployed using **Streamlit**.

---

## Project Introduction

Email spam is one of the most common cybersecurity problems. Spam emails often contain advertisements, phishing links, fake offers, malware, or fraudulent messages.

This project develops an intelligent Email Spam Detection System using Machine Learning. The application preprocesses email text using NLP techniques, converts it into numerical vectors using TF-IDF, and predicts whether an email is Spam or Ham.

The system is deployed using Streamlit to provide an interactive web interface where users can test emails in real time.

---

# Objectives

- Detect Spam emails accurately.
- Compare multiple Machine Learning models.
- Perform advanced NLP preprocessing.
- Deploy the model using Streamlit.
- Provide a simple user-friendly interface.

---

# Dataset

**Dataset Name**

SMS Spam Collection Dataset

**Source**

https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

**Total Messages**

5572

**Spam**

747

**Ham**

4825

---

# Technologies Used

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| IDE | VS Code |
| Notebook | Google Colab |
| Framework | Streamlit |
| Libraries | Pandas, NumPy, NLTK, Scikit-Learn |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| Deployment | GitHub, Streamlit Community Cloud |

---

# Concepts Used

- Natural Language Processing (NLP)
- Text Cleaning
- Tokenization
- Stopword Removal
- Stemming
- Lemmatization
- TF-IDF Vectorization
- Feature Extraction
- Machine Learning Classification
- Hyperparameter Tuning
- Cross Validation
- Model Evaluation
- Streamlit Deployment

---

# Machine Learning Models

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)
- Random Forest

---

# Project Workflow

```text
Dataset

↓

Data Cleaning

↓

Exploratory Data Analysis

↓

Text Preprocessing

↓

TF-IDF Feature Extraction

↓

Train-Test Split

↓

Model Building

↓

Hyperparameter Tuning

↓

Cross Validation

↓

Model Evaluation

↓

Save Model

↓

Streamlit Web Application
```

---

# Folder Structure

```text
Email-Spam-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── spam.csv
│
├── pages/
│
├── utils/
│
├── images/
│
├── css/
│
└── screenshots/
```

---

# Exploratory Data Analysis

- Dataset Information
- Missing Values Analysis
- Duplicate Removal
- Class Distribution
- Message Length Analysis
- Word Count Analysis
- WordCloud
- Correlation Heatmap

---

# Text Preprocessing

- Convert text to lowercase
- Remove URLs
- Remove HTML Tags
- Remove Numbers
- Remove Punctuation
- Tokenization
- Stopword Removal
- Stemming
- Lemmatization

---

# Feature Engineering

TF-IDF Vectorization converts textual data into numerical vectors suitable for Machine Learning algorithms.

---

# Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

# Model Comparison

| Model | Accuracy |
|---------|----------|
| Multinomial Naive Bayes | 97.3% |
| Logistic Regression | 98.4% |
| Linear SVM | 98.2% |
| Random Forest | 97.8% |

 **Best Model:** Logistic Regression

---

# Features

- Real-Time Spam Detection
- NLP Text Preprocessing
- TF-IDF Feature Extraction
- Confidence Score
- Spam/Ham Prediction
- Upload Text File
- Prediction History
- Download Prediction History
- Dataset Analytics Dashboard
- Interactive Charts

---

# Screenshots

## Home Page

Add image here

## Spam Prediction

Add image here

## Ham Prediction

Add image here

## Analytics Dashboard

Add image here

---

# Installation

Clone Repository

```bash
git clone https://github.com/yourusername/Email-Spam-Detection.git
```

Open Project

```bash
cd Email-Spam-Detection
```

Install Libraries

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

# Deployment

The project can be deployed using:

- GitHub
- Streamlit Community Cloud

---

# Future Scope

- Deep Learning (LSTM/BERT)
- Multi-language Spam Detection
- Batch Email Prediction
- PDF Email Classification
- REST API Integration
- Cloud Deployment
- Mobile Application

---

# Developer

**Monika**

BTech(AI & ML)

2026

---

# License

This project is developed for educational purposes.