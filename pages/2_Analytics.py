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

    page_title="Analytics",

    page_icon="📊",

    layout="wide"

)

# ==========================================================
# PAGE TITLE
# ==========================================================

st.title("📊 Email Spam Dataset Analytics")

st.caption(
    "Exploratory Data Analysis (EDA) of the Spam Dataset"
)

st.markdown("---")

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(
        "spam.csv",
        encoding="latin-1"
    )

    df = df.iloc[:, :2]

    df.columns = [
        "label",
        "text"
    ]

    return df

try:

    data = load_dataset()

except Exception as e:

    st.error("❌ Unable to load dataset.")

    st.exception(e)

    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    if os.path.exists("images/logo.png"):

        st.image(
            "images/logo.png",
            width=170
        )

    st.title("📊 Analytics")

    st.markdown("---")

    st.subheader("Dataset Information")

    st.write("""

Dataset Name

SMS Spam Collection

Total Records

5572

Spam

747

Ham

4825

""")

    st.markdown("---")

    st.subheader("Developer")

    st.write("""

Monika

BTech (AI & ML)

""")

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

st.header("📂 Dataset Overview")

left, right = st.columns([3,1])

with left:

    st.write("""

The dataset contains email/SMS messages
classified into two categories:

• **Spam**

• **Ham (Not Spam)**

This dataset is used to train and evaluate
Machine Learning models for spam detection.

""")

with right:

    if os.path.exists("images/banner.png"):

        st.image(
            "images/banner.png",
            use_container_width=True
        )

# ==========================================================
# DATASET METRICS
# ==========================================================

total_messages = len(data)

spam_messages = len(
    data[data["label"]=="spam"]
)

ham_messages = len(
    data[data["label"]=="ham"]
)

spam_percentage = round(
    spam_messages / total_messages * 100,
    2
)

ham_percentage = round(
    ham_messages / total_messages * 100,
    2
)

m1,m2,m3,m4 = st.columns(4)

m1.metric(
    "Total Emails",
    total_messages
)

m2.metric(
    "Spam Emails",
    spam_messages
)

m3.metric(
    "Ham Emails",
    ham_messages
)

m4.metric(
    "Spam %",
    f"{spam_percentage}%"
)

st.markdown("---")

# ==========================================================
# DATA QUALITY
# ==========================================================

st.header("🧹 Dataset Quality")

c1,c2,c3 = st.columns(3)

c1.metric(
    "Missing Values",
    data.isnull().sum().sum()
)

c2.metric(
    "Duplicate Rows",
    data.duplicated().sum()
)

memory = round(
    data.memory_usage().sum()/1024,
    2
)

c3.metric(
    "Memory Usage",
    f"{memory} KB"
)

st.markdown("---")

# ==========================================================
# DATA TYPES
# ==========================================================

st.header("📋 Dataset Columns")

datatype = pd.DataFrame({

    "Column":data.columns,

    "Data Type":data.dtypes.astype(str)

})

st.dataframe(

    datatype,

    use_container_width=True

)

st.markdown("---")

# ==========================================================
# DATASET PREVIEW
# ==========================================================

st.header("👀 Dataset Preview")

option = st.selectbox(

    "Select Preview",

    [

        "First 5 Rows",

        "Last 5 Rows",

        "Random 5 Rows"

    ]

)

if option == "First 5 Rows":

    st.dataframe(

        data.head(),

        use_container_width=True

    )

elif option == "Last 5 Rows":

    st.dataframe(

        data.tail(),

        use_container_width=True

    )

else:

    st.dataframe(

        data.sample(5),

        use_container_width=True

    )

st.markdown("---")

# ==========================================================
# MESSAGE LENGTH
# ==========================================================

data["Message Length"] = data["text"].apply(len)

st.header("📏 Message Length Statistics")

stats = pd.DataFrame({

    "Statistic":[

        "Minimum",

        "Maximum",

        "Average",

        "Median"

    ],

    "Value":[

        data["Message Length"].min(),

        data["Message Length"].max(),

        round(data["Message Length"].mean(),2),

        data["Message Length"].median()

    ]

})

st.dataframe(

    stats,

    use_container_width=True

)

st.markdown("---")

# ==========================================================
# CLASS DISTRIBUTION TABLE
# ==========================================================

st.header("📊 Class Distribution")

distribution = pd.DataFrame({

    "Class":[

        "Spam",

        "Ham"

    ],

    "Count":[

        spam_messages,

        ham_messages

    ],

    "Percentage":[

        spam_percentage,

        ham_percentage

    ]

})

st.dataframe(

    distribution,

    use_container_width=True

)

st.markdown("---")

# ==========================================================
# VISUALIZATION SECTION
# ==========================================================

st.header("📈 Dataset Visualizations")

# ==========================================================
# CLASS DISTRIBUTION BAR CHART
# ==========================================================

st.subheader("📊 Spam vs Ham Distribution")

fig, ax = plt.subplots(figsize=(6,4))

sns.countplot(
    data=data,
    x="label",
    palette="viridis",
    ax=ax
)

ax.set_xlabel("Class")
ax.set_ylabel("Count")
ax.set_title("Spam vs Ham Emails")

st.pyplot(fig)

# ==========================================================
# PIE CHART
# ==========================================================

st.subheader("🥧 Class Percentage")

fig, ax = plt.subplots(figsize=(6,6))

sizes = [spam_messages, ham_messages]

labels = ["Spam", "Ham"]

colors = ["#ff6b6b", "#4CAF50"]

explode = (0.08,0)

ax.pie(

    sizes,

    labels=labels,

    autopct="%1.1f%%",

    explode=explode,

    colors=colors,

    shadow=True,

    startangle=90

)

ax.axis("equal")

st.pyplot(fig)

# ==========================================================
# HISTOGRAM
# ==========================================================

st.subheader("📈 Message Length Histogram")

fig, ax = plt.subplots(figsize=(8,4))

sns.histplot(

    data["Message Length"],

    bins=40,

    kde=True,

    color="royalblue",

    ax=ax

)

ax.set_xlabel("Message Length")

ax.set_ylabel("Frequency")

ax.set_title("Distribution of Message Length")

st.pyplot(fig)

# ==========================================================
# BOXPLOT
# ==========================================================

st.subheader("📦 Message Length Box Plot")

fig, ax = plt.subplots(figsize=(8,4))

sns.boxplot(

    x=data["Message Length"],

    color="orange",

    ax=ax

)

ax.set_title("Outlier Detection")

st.pyplot(fig)

# ==========================================================
# MESSAGE LENGTH BY CLASS
# ==========================================================

st.subheader("📊 Average Message Length")

avg_length = data.groupby("label")["Message Length"].mean()

fig, ax = plt.subplots(figsize=(6,4))

avg_length.plot(

    kind="bar",

    color=["red","green"],

    ax=ax

)

ax.set_ylabel("Average Characters")

ax.set_xlabel("Email Type")

ax.set_title("Average Length of Spam and Ham")

st.pyplot(fig)

# ==========================================================
# SUMMARY TABLE
# ==========================================================

st.subheader("📋 Summary")

summary = pd.DataFrame({

    "Metric":[

        "Total Messages",

        "Spam Messages",

        "Ham Messages",

        "Average Length",

        "Maximum Length",

        "Minimum Length"

    ],

    "Value":[

        total_messages,

        spam_messages,

        ham_messages,

        round(data["Message Length"].mean(),2),

        data["Message Length"].max(),

        data["Message Length"].min()

    ]

})

st.dataframe(

    summary,

    use_container_width=True

)

st.markdown("---")

# ==========================================================
# IMPORT WORDCLOUD
# ==========================================================

from wordcloud import WordCloud

# ==========================================================
# WORD CLOUDS
# ==========================================================

st.header("☁️ WordCloud Analysis")

spam_text = " ".join(
    data[data["label"] == "spam"]["text"]
)

ham_text = " ".join(
    data[data["label"] == "ham"]["text"]
)

left, right = st.columns(2)

# -------------------------
# Spam WordCloud
# -------------------------

with left:

    st.subheader("🚨 Spam WordCloud")

    spam_wc = WordCloud(

        width=900,
        height=450,
        background_color="white",
        colormap="Reds"

    ).generate(spam_text)

    fig, ax = plt.subplots(figsize=(8,4))

    ax.imshow(spam_wc)

    ax.axis("off")

    st.pyplot(fig)

# -------------------------
# Ham WordCloud
# -------------------------

with right:

    st.subheader("✅ Ham WordCloud")

    ham_wc = WordCloud(

        width=900,
        height=450,
        background_color="white",
        colormap="Greens"

    ).generate(ham_text)

    fig, ax = plt.subplots(figsize=(8,4))

    ax.imshow(ham_wc)

    ax.axis("off")

    st.pyplot(fig)

st.markdown("---")

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

st.header("🔥 Correlation Heatmap")

heatmap_df = data.copy()

heatmap_df["Spam"] = heatmap_df["label"].map({

    "ham":0,

    "spam":1

})

heatmap_df["Length"] = heatmap_df["text"].apply(len)

corr = heatmap_df[["Spam","Length"]].corr()

fig, ax = plt.subplots(figsize=(5,4))

sns.heatmap(

    corr,

    annot=True,

    cmap="coolwarm",

    linewidths=1,

    ax=ax

)

st.pyplot(fig)

st.markdown("---")

# ==========================================================
# RANDOM DATASET SAMPLE
# ==========================================================

st.header("🎲 Random Dataset Sample")

if st.button("Generate Random Emails"):

    st.dataframe(

        data.sample(10),

        use_container_width=True

    )

st.markdown("---")

# ==========================================================
# DOWNLOAD DATASET
# ==========================================================

st.header("📥 Download Dataset")

csv = data.to_csv(

    index=False

).encode("utf-8")

st.download_button(

    label="⬇ Download Dataset",

    data=csv,

    file_name="spam_dataset.csv",

    mime="text/csv"

)

st.markdown("---")

# ==========================================================
# KEY INSIGHTS
# ==========================================================

st.header("📌 Key Insights")

st.success("""

✔ Majority of emails belong to the Ham category.

✔ Spam emails are fewer than Ham emails.

✔ Spam messages are generally longer.

✔ Dataset contains no significant missing values.

✔ Machine Learning models perform well
after TF-IDF feature extraction.

✔ Logistic Regression achieved the best
classification performance.

""")

st.markdown("---")

# ==========================================================
# TECHNOLOGIES USED
# ==========================================================

st.header("🛠 Technologies Used")

tech1, tech2 = st.columns(2)

with tech1:

    st.info("""

### Programming

• Python

• Streamlit

• Pandas

• NumPy

""")

with tech2:

    st.info("""

### Machine Learning

• Scikit-learn

• TF-IDF

• NLP

• Logistic Regression

""")

st.markdown("---")

# ==========================================================
# PROJECT WORKFLOW
# ==========================================================

st.header("⚙ Project Workflow")

if os.path.exists("images/workflow.png"):

    st.image(

        "images/workflow.png",

        use_container_width=True

    )

else:

    st.warning("workflow.png not found inside images folder.")

st.markdown("---")

# ==========================================================
# DEVELOPER
# ==========================================================

st.header("👩‍💻 Developer")

st.write("""

**Name : Monika**

**Course : BTech (AI & ML)**

This Email Spam Detection System was
developed as a Machine Learning Minor
Project using Natural Language Processing,
TF-IDF Vectorization, Multiple Machine
Learning Algorithms, Hyperparameter
Tuning, Cross Validation and Streamlit.

""")

st.markdown("---")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
---
<center>

### 📧 Email Spam Detection System

Developed using ❤️ Python • Streamlit • Scikit-Learn

**Developer : Monika**

</center>
""",
unsafe_allow_html=True
)