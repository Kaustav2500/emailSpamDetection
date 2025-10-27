# ✉️ Email Spam Detector

This project is an **email spam detection system** built with **Streamlit** and a **Multinomial Naive Bayes classifier**.  
It analyzes the textual content of emails to determine whether a message is **SPAM** or **HAM (legitimate)** using probabilistic natural language processing (NLP) techniques.  
The model is trained on a labeled dataset of real email messages to classify incoming text accurately and efficiently.

---

## 🧠 Features
- Real-time spam detection directly from the browser  
- Machine learning model (Multinomial Naive Bayes) trained on SMS/Email text data  
- Automatic text preprocessing and vectorization using **CountVectorizer**  
- Confidence-based probability output for each classification  
- Simple, clean Streamlit web interface for user interaction  

---

## 🧩 Technologies Used
- **Python 3**  
- **Streamlit** for the web interface  
- **Pandas** and **NumPy** for data processing  
- **Scikit-learn** for model building and text vectorization  

---

## ⚙️ How It Works
1. The system loads and preprocesses the email dataset (spam vs ham).  
2. Each message is converted into numerical features using **CountVectorizer**.  
3. A **Multinomial Naive Bayes classifier** learns the statistical patterns of spam and ham messages.  
4. When a user inputs a new email, the model predicts whether it is spam or legitimate.  
5. The output includes a clear classification and a **confidence score** representing prediction certainty.

---

### 🏁 **Conclusion**
This project demonstrates how **machine learning and NLP** can effectively detect unwanted or malicious messages.  
Using a lightweight Naive Bayes model, the app provides **fast**, **accurate**, and **interpretable** results suitable for practical spam filtering applications.

---

### 💻 **Try the App**
Explore the Email Spam Detection App yourself! The app is deployed on Streamlit. Click the link below to get started:
[**Email Spam Detection App**](https://emailspamdetection-lj8zxj2j9bxsfgy7nycv3m.streamlit.app/)


