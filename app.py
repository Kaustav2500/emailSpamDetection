import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Configuration
DATA_URL = "https://raw.githubusercontent.com/Kaustav2500/emailSpamDetection/main/email_spam.csv"

# Data Loading and Caching
@st.cache_data
def load_data():
    """Loads, cleans, and preprocesses the dataset."""
    try:
        df = pd.read_csv(DATA_URL, encoding='ISO-8859-1')
        df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
        df.drop(columns={'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'}, inplace=True)
        df.drop_duplicates(inplace=True)
        df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_model(df):
    """Trains the Spam detection pipeline (Vectoriser + Naive Bayes)."""
    if df.empty:
        return None

    X = df.Message
    y = df.Spam

    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])
    clf.fit(X, y)
    return clf

# Streamlit App UI
data_df = load_data()
model_pipeline = train_model(data_df)

st.set_page_config(
    page_title="Email Spam Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("✉️ Email Spam Detector")
st.markdown(
    "A simple web tool built using Streamlit and a Multinomial Naive Bayes classifier."
)

if model_pipeline is None:
    st.stop()

# Prediction Interface
st.header("Try it out!")
email_text = st.text_area(
    "Enter an email message below:", 
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. "
    "Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075",
    height=200
)

if st.button("Analyze Message", type="primary"):
    if email_text:
        prediction = model_pipeline.predict([email_text])[0]
        probability = model_pipeline.predict_proba([email_text])[0]
        confidence = max(probability) * 100

        st.markdown("---")
        
        if prediction == 1:
            st.error("🚨 This is a **SPAM** Email!")
            st.subheader(f"Confidence: {confidence:.2f}%")
        else:
            st.success("✅ This is a **HAM** (Legitimate) Email!")
            st.subheader(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter a message to analyze.")

