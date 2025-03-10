# 📊 Leicester City Council Data Visualization🚀

### **🔍 Project Overview**
This project focuses on **analyzing and visualizing Twitter data** along with **Leicester City Council data** to gain insights into **public engagement, sentiment analysis, and civic participation**. The goal is to help the Leicester City Council understand how citizens interact with the council on social media and use these insights to **improve engagement strategies and decision-making**.

The project integrates **data visualization techniques, machine learning models, and a web-based dashboard** to present meaningful insights extracted from Twitter and city council datasets.

**Tools & Technologies Used**
✅ **Python** – Data processing, NLP, and machine learning  
✅ **Flask** – Backend framework for UI integration  
✅ **SNScrape** – Twitter data collection  
✅ **Pandas & NumPy** – Data cleaning and transformation  
✅ **Matplotlib & Seaborn** – Data visualization  
✅ **Scikit-learn** – Machine learning models for sentiment analysis  
✅ **NLTK & TextBlob** – Natural language processing (NLP)  
✅ **SMOTE** – Handling imbalanced datasets  
✅ **HTML, CSS** – Frontend development  


**📊 Data Processing & Cleaning**
Before analysis, the dataset underwent **data preprocessing** to improve data quality:
- **Twitter data collection** using SNScrape and filtering for Leicester City Council-related tweets.
- **Removing missing values** and handling incorrect data (e.g., invalid age, incorrect credit scores).
- **Lemmatization, stopword removal, and tokenization** for text preprocessing.
- **Categorizing tweets into sentiment classes** (positive, negative, neutral).
- **Balancing imbalanced data** using SMOTE for better model performance.

---

## **📊 Key Insights from Data Analysis**
✔️ **Sentiment Analysis:** Majority of tweets were **neutral**, followed by **negative and positive** interactions.  
✔️ **Engagement Patterns:** Most engagement occurred during **public events and policy announcements**.  
✔️ **Twitter Trends:** Keywords like **"Leicester", "council", "policy", "budget"** were the most discussed topics.  
✔️ **Machine Learning Predictions:** Random Forest & Logistic Regression models provided the **best accuracy** for classifying sentiment.  

---

## **📌 Machine Learning Models Used**
Several machine learning models were implemented and compared:
- **Logistic Regression** – Baseline sentiment classification model.
- **Decision Tree** – Rule-based sentiment classification.
- **Random Forest** – Ensemble learning model for better accuracy.
- **Support Vector Machine (SVM)** – Used for text classification.
- **Multinomial Naive Bayes (NB)** – Best suited for NLP-based sentiment analysis.

---

## **📊 Web Dashboard & Visualization**
A web-based dashboard was created using **Flask** to allow users to interact with the data visually. The UI includes:
- **Login & Signup pages** for user authentication.
- **Sentiment analysis charts** showing tweet sentiment distribution.
- **Word clouds** displaying frequently used words in tweets.
- **Time-series analysis** to observe trends over time.
- **Filter options** to explore data based on tweet count, sentiment, and date.

🔗 **Live Dashboard Link (if hosted):** _Add link here_

---

## **🚀 Future Enhancements**
🔹 **Expand Data Sources** – Integrate additional data sources like Facebook and public surveys.  
🔹 **Improve Sentiment Analysis** – Use advanced NLP techniques like BERT for better accuracy.  
🔹 **Real-time Streaming** – Enable real-time tweet monitoring for Leicester City Council.  
🔹 **Interactive UI** – Add features like user comments and recommendations for council engagement.  

---

## **📌 How to Run the Project Locally**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/leicester-city-council-data-visualization.git
cd leicester-city-council-data-visualization
